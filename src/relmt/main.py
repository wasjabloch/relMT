#!/usr/bin/env python

# relMT - Program to compute relative earthquake moment tensors
# Copyright (C) 2024 Wasja Bloch, Doriane Drolet, Michael Bostock
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Main relMT executables"""

import logging
from relmt import io, utils, align, core, signal, extra, amp, ls, mt, qc
from scipy.sparse import coo_matrix, save_npz, load_npz
from pathlib import Path
import yaml
import numpy as np
import multiprocessing as mp
from argparse import ArgumentParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def get_arguments(args=None):
    """Get command line options for :func:`main_align()`"""

    parser = ArgumentParser(
        usage="%(prog)s [arguments] <station database>",
        description=(
            "Align waveforms using multi channel cross correlation"
            "and pricinipal component analyses."
        ),
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Use this configuration file",
        default=core.file("config"),
    )

    parser.add_argument(
        "-n", "--n_align", type=int, help="Alignment iteration", default=0
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing files",
        action="store_true",
    )

    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Exlude data with no data or data containing NaNs",
    )

    parser.add_argument(
        "--snr",
        action="store_true",
        help=(
            "Exlude data with signal to noise ratio higher than "
            "'min_signal_noise_ratio' in the configuration file"
        ),
    )

    parser.add_argument(
        "--ecn",
        action="store_true",
        help=(
            "Exlude data with expansion coefficient norm higher than "
            "'min_expansion_coefficient_norm' in the configuration file"
        ),
    )

    return parser.parse_args(args)


def main_align(args=None):
    """
    Align waveform files and write results into next alignment directory
    """

    args = get_arguments(args)

    iteration = args.n_align
    overwrite = args.overwrite
    conff = args.config

    conf = io.read_config(conff)

    evf = conf["event_file"]
    stf = conf["station_file"]

    evl = io.read_event_table(evf)
    stas = io.read_station_table(stf)
    excl = io.read_exclude_file(core.file("exclude"))

    # Exclude some observations
    stas = set(stas) - set(excl["station"])
    wvids = set(core.iterate_waveid(stas)) - set(excl["waveform"])

    ncpu = conf["ncpu"]

    args = []
    for wvid in wvids:

        if wvid in excl["waveform"]:
            continue

        logger.debug(f"Getting arguments for: {wvid}")

        station, phase = core.split_waveid(wvid)
        source = (station, phase, iteration)
        dest = (station, phase, iteration + 1)

        # Check the destination
        if not overwrite:  # Try to read the destination files
            try:
                _ = io.read_waveform_array_header(*dest)
            except FileNotFoundError:
                pass  # Continue when they are absent
            else:
                continue  # Do not process when they exist

        # Read the soure
        try:
            arr, hdr = io.read_waveform_array_header(*source)
        except FileNotFoundError:
            logger.debug("Trying Matlab file.")
            try:
                arr, hdr = io.read_waveform_array_header(*source, matlab=True)
            except OSError:
                logger.debug("Expected an absent file. Continuing.")
                continue

        # QC: Events included for this wavetrain
        iin, _ = qc.included_events(
            excl, return_bool=True, **hdr.kwargs(qc.included_events)
        )

        # Events that are non-zero and finite
        inz = qc.index_nonzero_events(arr, return_bool=True)
        iin &= inz

        # Any events left?
        if not any(iin):
            continue

        # Only parse valid events
        hdr["events"] = list(np.array(hdr["events"])[iin])
        arr = arr[iin, :, :]

        # Let us set some values in case they are not there yet
        if hdr["lowpass"] is None:
            mmax = max([ev.mag for ev in evl])
            hdr["lowpass"] = utils.corner_frequency(mmax, hdr["phase"], 5e7, 3500)

        args.append((arr, hdr, dest))

    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            pool.starmap(align.run, args)
    else:
        for arg in args:
            align.run(*arg)


def main_exclude(args=None):
    args = get_arguments(args)

    iteration = args.n_align
    overwrite = args.overwrite
    donodata = args.no_data
    dosnr = args.snr
    doecn = args.ecn

    conf = io.read_config(args.config)

    exf = core.file("exclude")

    staf = conf["station_file"]
    stas = io.read_station_table(staf)

    try:
        logger.info(f"Reading excludes from: {exf}")
        excl = io.read_exclude_file(exf)
    except OSError:
        logger.info(f"No exlusion file found. Creating: {exf}")
        excl = core.exclude
        io.save_yaml(exf, excl)

    stas = set(stas) - set(excl["station"])

    excludes = {"no_data": [], "snr": [], "ecn": []}

    for wvid in core.iterate_waveid(stas):

        print(f"Working on: {wvid}")

        sta, pha = core.split_waveid(wvid)

        try:
            arr, hdr = io.read_waveform_array_header(
                sta,
                pha,
                iteration,
            )
        except FileNotFoundError:
            continue

        events = np.array(hdr["events"])

        # Boolean indices of bad events
        ind = qc.index_nonzero_events(arr, return_bool=True, return_not=True)

        if all(ind):
            excl["waveform"] += [wvid]
            continue

        arr = signal.demean_filter_window(
            arr, **hdr.kwargs(signal.demean_filter_window)
        )

        # Compute QC metrics
        ec_score = qc.expansion_coefficient_norm(arr, pha)
        snr = signal.signal_noise_ratio(arr, **hdr.kwargs(signal.signal_noise_ratio))

        isnr = np.full_like(ind, False)
        if hdr["min_signal_noise_ratio"] is not None:
            isnr = snr < hdr["min_signal_noise_ratio"]

        iecn = np.full_like(ind, False)
        if hdr["min_expansion_coefficient_norm"] is not None:
            iecn = ec_score < hdr["min_expansion_coefficient_norm"]

        # Write the full phase ID to the exclude lists
        excludes["no_data"] += [core.join_phaseid(iev, sta, pha) for iev in events[ind]]
        excludes["snr"] += [core.join_phaseid(iev, sta, pha) for iev in events[isnr]]
        excludes["ecn"] += [core.join_phaseid(iev, sta, pha) for iev in events[iecn]]

    if not overwrite:
        excludes["no_data"] += excl["phase_auto_nodata"]
        excludes["snr"] += excl["phase_auto_snr"]
        excludes["ecn"] += excl["phase_auto_ecn"]

    if donodata:
        excl["phase_auto_nodata"] = excludes["no_data"]

    if dosnr:
        excl["phase_auto_snr"] = excludes["snr"]

    if doecn:
        excl["phase_auto_ecn"] = excludes["ecn"]

    # Save it to file
    io.save_yaml(exf, excl)


@core._doc_config_args
def phase_passbands(
    arr: np.ndarray,
    hdr: core.Header,
    evl: list[core.Event],
    exclude: core.Exclude | None = None,
    auto_lowpass_method: str | None = None,
    auto_lowpass_stressdrop_range: tuple[float, float] = [1.0e6, 1.0e6],
    auto_bandpass_snr_target: float | None = None,
) -> dict[str, list[float]]:
    """Compute the bandpass filter corners for a waveform array.

    This function computes the bandpass filter corners for each event in the
    waveform array. The following logic is applied:

    The lowpass corner should be the corner frequency of the phase spectrum. We
    estimate it as:

    - 1/source duration of the event magnitude when `lowpass_method` is 'duration'.
    - Based on the stressdrop within  `lowpass_stressdrop_range` when
    `lowpass_method` is 'stressdrop':
      - If the upper bound is smaller or equal the lower bound (i.e. no range is
        given), estimate the corner frequency using
        :func:`utils.corner_frequency` with an S-wave velocity of 4 km/s.
      - If a range is given, we convert it to a corner frequency range as above
        and search for the maximum of the phase velocity spectrum within this range
        using :func:`extra.apparent_corner_frequency`

    The default highpass corner is chosen as 1/phase length.

    When `bandpass_snr_target` is given, we determine the frequency band in the
    signal and noise spectra for which the signal-to-noise ratio is larger than
    `bandpass_snr_target` (dB), compare with the above estimates, and return the
    highest highpass and the lowest lowpass corner.

    Parameters
    ----------
    arr:
        The waveform array with shape ``(events, channels, samples)``.
    hdr:
        The header containing metadata about the waveform array, including
        phase phase start and end times, sampling rate, included events.
    evl:
        The seismic event catalog.
    exclude:
        An optional exclude object with observations to be excluded from the
        computation. If None, all observatinos are included.

    Returns
    -------
    Dictionary mapping phaseIDs (event ID, station, phase) to filter corners
    [highpass, lowpass].

    Raises
    ------
    ValueError:
        If an unknown lowpass method is specified.
    """

    ievs, evns = qc.included_events(exclude, **hdr.kwargs(qc.included_events))

    pha = hdr["phase"]

    # At least one period within window
    fmin = 1 / (hdr["phase_end"] - hdr["phase_start"])

    # One sample more than the Nyquist frequency
    fnyq = (arr.shape[-1] - 1) / hdr["data_window"] / 2

    bpd = {}
    for iev, evn in zip(ievs, evns):
        print("{:02d} events to go   ".format(len(evns) - iev), end="\r")

        ev = evl[evn]

        phase_arr = arr[iev, :, :]

        # First get the corner frequency
        if auto_lowpass_method == "duration":
            # Corner frequency from duration
            fc = 1 / utils.source_duration(ev.mag)

        elif auto_lowpass_method == "stressdrop":
            # Corner frequency from stress drop

            if auto_lowpass_stressdrop_range[0] >= auto_lowpass_stressdrop_range[1]:
                fc = utils.corner_frequency(
                    ev.mag, pha, auto_lowpass_stressdrop_range[0], 4000
                )

            else:
                # Convert stressdrop range to possile corner frequencies
                fcmin = utils.corner_frequency(
                    ev.mag, pha, auto_lowpass_stressdrop_range[0], 4000
                )
                fcmax = utils.corner_frequency(
                    ev.mag, pha, auto_lowpass_stressdrop_range[1], 4000
                )

                # Isolate the signal
                isig, _ = signal.indices_signal(**hdr.kwargs(signal.indices_signal))
                sig = signal.demean(phase_arr[:, isig:])

                # Try to compute the corner frequency
                try:
                    fc = extra.apparent_corner_frequency(
                        sig, hdr["sampling_rate"], fmin=fcmin, fmax=fcmax
                    )
                except ValueError:
                    # It might be outside the range of the signal
                    fc = fcmax
        else:
            raise ValueError(f"Unknown lowpass method: {auto_lowpass_method}")

        # Now try to optimize the SNR
        if auto_bandpass_snr_target is not None:
            hpas, lpas = extra.optimal_bandpass(
                phase_arr,
                fmin=min(fmin, fnyq),
                fmax=min(fc, fnyq),
                min_snr=auto_bandpass_snr_target,
                **hdr.kwargs(extra.optimal_bandpass),
            )
        else:
            # No SNR optimization, use the corner frequency
            hpas = min(fmin, fnyq)
            lpas = min(fc, fnyq)

        # Return the filter corners
        bpd[evn] = [float(hpas), float(lpas)]

    return bpd


def main_amplitude(args=None):
    # Subdirectory, e.g. A_Muji
    args = get_arguments(args)

    iteration = args.n_align
    conff = args.config
    overwrite = args.overwrite

    directory = conff.parent

    config = io.read_config(conff)

    ampdir = directory / "amplitude"
    if not ampdir.exists():
        logger.info(f"Target directory does not exist: {ampdir}. Creating.")
        ampdir.mkdir()

    stf = directory / config["station_file"]
    evf = directory / config["event_file"]
    ncpu = config["ncpu"]
    compare_method = config["amplitude_measure"]  # combination or principal
    filter_method = config["amplitude_filter"]  # auto or manual

    ls.logger.setLevel("ERROR")
    signal.logger.setLevel("ERROR")
    align.logger.setLevel("WARNING")

    exclude = io.read_exclude_file(core.file("exclude", directory=directory))

    stas = io.read_station_table(stf)

    # Exclude some observations
    stas = set(stas) - set(exclude["station"])
    wvids = set(core.iterate_waveid(stas)) - set(exclude["waveform"])

    # Read the events
    event_list = io.read_event_table(evf)

    pasbnds = {}
    if filter_method == "manual":
        logger.info("Applying filters set in header files.")

        # Write the manually set filter corners into the dictionary
        for wvid in wvids:
            logger.debug(f"Working on: {wvid}")
            sta, pha = core.split_waveid(wvid)

            try:
                hdr = io.read_header(
                    core.file("waveform_header", sta, pha, iteration, directory),
                    default_name=core.file(
                        "waveform_header", n_align=iteration, directory=directory
                    ),
                )
            except FileNotFoundError as e:
                logger.warning(e)
                continue

            pasbnds[wvid] = {
                evn: [hdr["highpass"], hdr["lowpass"]] for evn in hdr["events"]
            }

    elif filter_method == "auto":
        logger.info("Finding filter automatically")

        # Read or compute bassbands
        bpf = core.file(
            "bandpass", directory=directory, suffix=config["amplitude_suffix"]
        )

        # Read the bandpass if it exists
        if bpf.exists() and not overwrite:
            logger.info(f"Reading bandpass from file: {bpf}")
            with open(bpf, "r") as fid:
                pasbnds = yaml.safe_load(fid)

        # Compute it if not
        else:
            logger.info("No bandpass file found or overwrite option set. Computing.")

            for wvid in wvids:
                logger.debug(f"Working on: {wvid}")
                sta, pha = core.split_waveid(wvid)

                try:
                    arr, hdr = io.read_waveform_array_header(
                        sta, pha, iteration, directory
                    )
                except FileNotFoundError as e:
                    logger.warning(e)
                    continue

                pasbnds[wvid] = phase_passbands(
                    arr,
                    hdr,
                    event_list,
                    **config.kwargs(phase_passbands),
                    exclude=exclude,
                )

            io.save_yaml(bpf, pasbnds)
            logger.info(f"Saved bandpass to file: {bpf}")

    else:
        raise ValueError(f"Unknown 'amplitude_filter': {filter_method}")

    # Collect the arguments to the amplitude function
    pargs = []
    sargs = []

    if compare_method == "combination":
        for wvid in wvids:
            sta, pha = core.split_waveid(wvid)

            try:
                arr, hdr = io.read_waveform_array_header(sta, pha, iteration, directory)
            except FileNotFoundError as e:
                logger.warning(e)
                continue

            # Make sure what's excluded is excluded
            ievs, evns = qc.included_events(exclude, **hdr.kwargs(qc.included_events))
            xarr = arr[ievs, :, :]
            hdr["events"] = evns

            # Look up correct passband and collect the arguments
            if pha == "P":
                for a, b, _, _ in core.iterate_event_pair(len(evns)):

                    hpas, lpas = signal.choose_passband(
                        [pasbnds[wvid][evns[i]][0] for i in [a, b]],
                        [pasbnds[wvid][evns[i]][1] for i in [a, b]],
                        config["min_dynamic_range"],
                    )

                    # Passband is within dynamic range
                    if hpas is not None:
                        pargs.append(
                            (xarr[[a, b], :], hdr, hpas, lpas, evns[a], evns[b])
                        )

            if pha == "S":
                for a, b, c, _, _, _ in core.iterate_event_triplet(len(evns), evns):
                    hpas, lpas = signal.choose_passband(
                        [pasbnds[wvid][evns[i]][0] for i in [a, b, c]],
                        [pasbnds[wvid][evns[i]][1] for i in [a, b, c]],
                        config["min_dynamic_range"],
                    )

                    # Passband is within dynamic range
                    if hpas is not None:
                        sargs.append(
                            (
                                xarr[[a, b, c], :],
                                hdr,
                                hpas,
                                lpas,
                                evns[a],
                                evns[b],
                                evns[c],
                            )
                        )

        # Argumets above pertain to these functions
        p_amp_fun = amp.paired_p_amplitudes
        s_amp_fun = amp.triplet_s_amplitudes

    elif compare_method == "principal":

        for wvid in wvids:
            sta, pha = core.split_waveid(wvid)

            try:
                arr, hdr = io.read_waveform_array_header(sta, pha, iteration, directory)
            except FileNotFoundError as e:
                logger.warning(e)
                continue

            # Exclude the excluded events
            ievs, evns = qc.included_events(exclude, **hdr.kwargs(qc.included_events))
            xarr = arr[ievs, :, :]
            hdr["events"] = evns

            # Choose highest highpass and lowest lowpass
            # Note all event passbands are equl if filter method is "manual"
            hpas, lpas = signal.choose_passband(
                [pasbnds[wvid][evn][0] for evn in pasbnds[wvid] if evn in evns],
                [pasbnds[wvid][evn][1] for evn in pasbnds[wvid] if evn in evns],
                config["min_dynamic_range"],
            )

            # If we are strict (positive min_dynamic_range), don't process.
            if hpas is None:
                continue

            if pha == "P":
                pargs += [(xarr, hdr, hpas, lpas)]

            if pha == "S":
                sargs += [(xarr, hdr, hpas, lpas)]

        # Arguments above pertain to these functions
        p_amp_fun = amp.principal_p_amplitudes
        s_amp_fun = amp.principal_s_amplitudes

    else:
        raise ValueError(f"Unknown 'amplitude_measure': {compare_method}")

    # First process and save P ...
    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            result = pool.starmap(p_amp_fun, pargs)
    else:
        result = [p_amp_fun(*arg) for arg in pargs]

    if compare_method == "principal":
        # Result is a list of list. Let's make it just a list
        abA = []
        for pamplist in result:
            abA.extend(pamplist)
    else:
        abA = result

    io.save_amplitudes(
        core.file(
            "amplitude_observation",
            sta,
            "P",
            directory=directory,
            suffix=config["amplitude_suffix"],
        ),
        abA,
    )

    # ... later S
    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            result = pool.starmap(s_amp_fun, sargs)
    else:
        result = [s_amp_fun(*arg) for arg in sargs]

    if compare_method == "principal":
        # Result is a list of list. Let's make it just a list
        abcB = []
        for samplist in result:
            abcB.extend(samplist)
    else:
        abcB = result

    io.save_amplitudes(
        core.file(
            "amplitude_observation",
            sta,
            "S",
            directory=directory,
            suffix=config["amplitude_suffix"],
        ),
        abcB,
    )


def main_linear_system(args=None):

    ls.logger.setLevel("WARNING")

    args = get_arguments(args)
    suffix = "-" + args.suffix
    conff = args.config

    conf = io.read_config(conff)

    evf = conf["event_file"]
    stf = conf["station_file"]
    phf = conf["phase_file"]
    refmtf = conf["reference_mt_file"]

    max_misfit = conf["max_amplitude_misfit"]
    all_ref_mts = conf["reference_mts"]
    constraint = conf["mt_constraint"]
    refmt_weight = conf["reference_weight"]

    mt_elements = ls.mt_elements(constraint)

    ref_id = "-".join([f"{iref}" for iref in all_ref_mts])

    all_evl = io.read_event_table(evf)
    all_phd = io.read_phase_table(phf)
    stad = io.read_station_table(stf)
    all_mtd = io.read_mt_table(refmtf)

    outsuffix = suffix + "_iref_" + ref_id + f"_{constraint}"

    # Read amplitudes from file
    all_p_amplitudes = io.read_amplitudes(
        core.file("amplitude_observation", phase="P", suffix=suffix),
        "P",
    )
    all_s_amplitudes = io.read_amplitudes(
        core.file("amplitude_observation", phase="S", suffix=suffix),
        "S",
    )

    # Include only events in event list that are acutally good.
    ab = np.array([(amp.event_a, amp.event_b) for amp in all_p_amplitudes])
    abc = np.array(
        [(amp.event_a, amp.event_b, amp.event_c) for amp in all_s_amplitudes]
    )
    inev = sorted(set(ab.flat).union(set(abc.flat)))

    np.savetxt(core.file("inev.txt"), inev, fmt="%.0f")

    # Change all the indexing
    evl = [all_evl[n] for n in inev]
    ref_mts = [inev.index(n) for n in all_ref_mts]

    phd = {
        core.join_phaseid(
            inev.index(core.split_phaseid(phid)[0]), *core.split_phaseid(phid)[1:]
        ): pha
        for phid, pha in all_phd.items()
        if core.split_phaseid(phid)[0] in inev
    }
    p_amplitudes = [
        core.P_Amplitude_Ratio(
            am.station,
            inev.index(am.event_a),
            inev.index(am.event_b),
            am.amp_ab,
            am.misfit,
        )
        for am in all_p_amplitudes
        if am.event_a in inev and am.event_b in inev
    ]
    s_amplitudes = [
        core.S_Amplitude_Ratios(
            am.station,
            inev.index(am.event_a),
            inev.index(am.event_b),
            inev.index(am.event_c),
            am.amp_abc,
            am.amp_acb,
            am.misfit,
        )
        for am in all_s_amplitudes
        if am.event_a in inev and am.event_b in inev and am.event_c in inev
    ]
    mtd = {inev.index(iev): mt for iev, mt in all_mtd.items() if iev in inev}

    # Build homogenos part of linear system
    Ah, bh = ls.homogenous_amplitude_equations(
        p_amplitudes, s_amplitudes, stad, evl, phd, constraint
    )

    # Normalization applied to columns
    ev_norm = ls.norm_event_median_amplitude(Ah, mt_elements)
    Ah *= ev_norm

    mis_weights = np.vstack(
        [ls.weight_misfit(amp, max_misfit, "P") for amp in p_amplitudes]
        + [ls.weight_misfit(amp, max_misfit, "S") for amp in s_amplitudes]
    )

    amp_weights = np.vstack(
        [1.0 for _ in p_amplitudes]
        + [ls.weight_s_amplitude(amp) for amp in s_amplitudes]
    )

    Ah *= mis_weights * amp_weights

    # Weight applied by row
    eq_norm = ls.condition_homogenous_matrix_by_norm(Ah)
    Ah *= eq_norm

    # Build inhomogenous equations
    Ai, bi = ls.reference_mt_equations(ref_mts, mtd, len(evl), constraint)

    # Collect and apply weights
    mean_moment = mt.mean_moment([mtd[iev] for iev in ref_mts])
    refev_norm = ls.reference_mt_event_norm(ev_norm, ref_mts, mt_elements)
    Ai *= refmt_weight
    bi *= refmt_weight / mean_moment / refev_norm

    # Scale of resulting relative moment tensors
    ev_scale = mean_moment * ev_norm

    A = coo_matrix(np.vstack((Ah, Ai))).tocsc()
    b = np.vstack((bh, bi))

    # np.save(core.file("amplitude_matrix", directory=directory, suffix=outsuffix), A)
    save_npz(core.file("amplitude_matrix", suffix=outsuffix), A)
    np.save(core.file("amplitude_data_vector", suffix=outsuffix), b)
    np.save(core.file("amplitude_scale", suffix=outsuffix), ev_scale)


def main_solve(args=None):

    args = get_arguments(args)
    suffix = "-" + args.suffix
    conf = args.config

    ref_mts = conf["reference_mts"]
    constraint = conf["mt_constraint"]
    nboot = conf["bootstrap_samples"]

    mt_elements = ls.mt_elements(constraint)

    ref_id = "-".join([f"{iref}" for iref in ref_mts])

    outsuf = suffix + "_iref_" + ref_id + f"_{constraint}"

    pamps = io.read_amplitudes(
        core.file("amplitude_observation", phase="P", suffix=suffix), "P"
    )
    samps = io.read_amplitudes(
        core.file("amplitude_observation", phase="S", suffix=suffix), "S"
    )

    n_p = len(pamps)
    n_s = len(samps) * 2
    n_ref = len(ref_mts)

    # Load data
    # A = np.load(
    #    core.file("amplitude_matrix", directory=directory, suffix=outsuf),
    #    allow_pickle=True,
    # ).item()
    A = load_npz(core.file("amplitude_matrix", suffix=outsuf))
    b = np.load(core.file("amplitude_data_vector", suffix=outsuf))
    ev_scale = np.load(core.file("amplitude_scale", suffix=outsuf))

    # Invert and save results
    m, residuals = ls.solve_lsmr(A, b, ev_scale)
    p_residuals, s_residuals, _ = ls.unpack_resiudals(
        residuals, n_p, n_ref, mt_elements
    )

    np.savetxt(core.file("moment_residual", phase="P", suffix=outsuf), p_residuals)
    np.savetxt(core.file("moment_residual", phase="S", suffix=outsuf), s_residuals)

    try:
        inev = np.loadtxt(core.file("inev.txt")).astype(int)
    except FileNotFoundError:
        inev = list(range(int(A.shape[1] // mt_elements)))

    relmts = {
        inev[i]: momt for i, momt in enumerate(mt.mt_tuples(m, constraint)) if any(momt)
    }
    io.make_mt_table(relmts, core.file("relative_mt", suffix=outsuf))

    # Bootstrap
    if nboot:
        m_boots = ls.bootstrap_lsmr(A, b, ev_scale, n_p, n_s, nboot, 0, 1)

        # Convert to MT dict
        bootmts = {i: [] for i in range(m_boots.shape[1] // mt_elements)}
        for m_boot in m_boots:
            for i, momt in enumerate(mt.mt_tuples(m_boot, constraint)):
                bootmts[i].append(momt)

        # Make and save a
        io.make_mt_table(bootmts, core.file("relative_mt", suffix=outsuf + "_boot"))


if __name__ == "__main__":

    # We will eventually be able to invoke relMT from the command line
    pass
