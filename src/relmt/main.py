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

from relmt import io, utils, align, core, signal, extra, amp, ls, mt, qc, angle, plot
from scipy import sparse
from pathlib import Path
import yaml
import numpy as np
import sys
import multiprocessing as mp
from multiprocessing import shared_memory as sm
from argparse import ArgumentParser

logger = core.register_logger(__name__)


def main_align(
    config: core.Config,
    directory: Path,
    iteration: int,
    do_mccc: bool = True,
    do_pca: bool = True,
    overwrite: bool = False,
):
    """
    Align waveform files and write results into next alignment directory
    """

    stf = directory / config["station_file"]

    stas = io.read_station_table(stf)
    excl = io.read_exclude_file(core.file("exclude", directory=directory))

    # Exclude some observations
    stas = set(stas) - set(excl["station"])
    wvids = set(core.iterate_waveid(stas)) - set(excl["waveform"])

    ncpu = config["ncpu"]

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

        # Read optional pair list and  check for singolton events
        if hdr["combinations_from_file"]:
            pairs = io.read_combinations(core.file("combination", *source))
            iin &= np.isin(hdr["events_"], np.unique(list(pairs)))

        # Enough events left?
        if (phase == "P" and sum(iin) < 2) or (phase == "S" and sum(iin) < 3):
            continue

        # Only parse valid events
        hdr["events_"] = list(np.array(hdr["events_"])[iin])

        # Convert pair to triplet combinations and return indices into arr
        combinations = np.array([])
        if hdr["combinations_from_file"]:
            logger.debug("Finding valid combinations")
            combinations = utils.valid_combinations(hdr["events_"], pairs, hdr["phase"])
            logger.info(f"Found {combinations.shape[0]} combinations for {wvid}")

        arr = arr[iin, :, :]

        args.append((arr, hdr, dest, do_mccc, do_pca, combinations))

    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            pool.starmap(align.run, args)
    else:
        for arg in args:
            align.run(*arg)


def main_exclude(
    config: core.Config,
    iteration: int,
    overwrite: bool,
    directory: Path = Path("."),
    do_nodata: bool = False,
    do_snr: bool = False,
    do_cc: bool = False,
    do_ecn: bool = False,
):
    """Exclude observations from alignment procedure"""

    exf = core.file("exclude", directory=directory)

    staf = directory / config["station_file"]
    stas = io.read_station_table(staf)

    try:
        logger.info(f"Reading excludes from: {exf}")
        excl = io.read_exclude_file(exf)
    except OSError:
        logger.info(f"No exlusion file found. Creating: {exf}")
        excl = core.exclude
        io.save_yaml(exf, excl)

    stas = set(stas) - set(excl["station"])

    # Collect new excludes in this dictionary
    excludes = {"no_data": [], "snr": [], "cc": [], "ecn": []}

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

        events = np.array(hdr["events_"])

        # Boolean indices of events with no data
        ind = qc.index_nonzero_events(
            arr, null_threshold=hdr["null_threshold"], return_bool=True, return_not=True
        )
        logger.debug(f"{wvid}: {sum(ind)} traces with no data")

        if all(ind):
            excl["waveform"] += [wvid]
            continue

        isnr = np.full_like(ind, False)
        if (minsnr := hdr["min_signal_noise_ratio"]) is not None and do_snr:
            snr = signal.signal_noise_ratio(
                arr, **hdr.kwargs(signal.signal_noise_ratio)
            )
            isnr = snr < minsnr
            logger.debug(f"{wvid}: {sum(isnr)} traces with SNR < {minsnr}")

        icc = np.full_like(ind, False)
        if (mincc := hdr["min_correlation"]) is not None and do_cc:
            mat = utils.concat_components(
                signal.demean_filter_window(
                    arr, **hdr.kwargs(signal.demean_filter_window)
                )
            )
            cc = signal.reconstruction_correlation_averages(mat, hdr["phase"])[2]
            icc = cc < mincc
            logger.debug(f"{wvid}: {sum(icc)} traces with CC < {mincc}")

        iecn = np.full_like(ind, False)
        if (minecn := hdr["min_expansion_coefficient_norm"]) is not None and do_ecn:
            arr = signal.demean_filter_window(
                arr, **hdr.kwargs(signal.demean_filter_window)
            )
            ec_score = qc.expansion_coefficient_norm(arr, pha)
            iecn = ec_score < minecn
            logger.debug(f"{wvid}: {sum(iecn)} traces with ECN < {minecn}")

        # Write the full phase ID to the exclude lists
        excludes["no_data"] += [core.join_phaseid(iev, sta, pha) for iev in events[ind]]
        excludes["snr"] += [core.join_phaseid(iev, sta, pha) for iev in events[isnr]]
        excludes["cc"] += [core.join_phaseid(iev, sta, pha) for iev in events[icc]]
        excludes["ecn"] += [core.join_phaseid(iev, sta, pha) for iev in events[iecn]]

    # Add the exludes already present, in case we don't want to overwrite
    if not overwrite:
        excludes["no_data"] += excl["phase_auto_nodata"]
        excludes["snr"] += excl["phase_auto_snr"]
        excludes["cc"] += excl["phase_auto_cc"]
        excludes["ecn"] += excl["phase_auto_ecn"]

    # Write everything into the exclude dict
    if do_nodata:
        logger.info(f"Excluding {len(excludes['no_data'])} invalid traces")
        excl["phase_auto_nodata"] = excludes["no_data"]

    if do_snr:
        logger.info(f"Excluding {len(excludes['snr'])} traces with SNR < {minsnr}")
        excl["phase_auto_snr"] = excludes["snr"]

    if do_cc:
        logger.info(f"Excluding {len(excludes['cc'])} traces with CC < {mincc}")
        excl["phase_auto_cc"] = excludes["cc"]

    if do_ecn:
        logger.info(f"Excluding {len(excludes['ecn'])} traces with ECN < {minecn}")
        excl["phase_auto_ecn"] = excludes["ecn"]

    # Save it to file
    io.save_yaml(exf, excl)


@core._doc_config_args
def phase_passbands(
    arr: np.ndarray,
    hdr: core.Header,
    evd: dict[int, core.Event],
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
    - Based on the corner frequency pertaining to a stressdrop (Pa) within
    `lowpass_stressdrop_range` when `lowpass_method` is 'corner':
      - If the upper bound is smaller or equal the lower bound (i.e. no range is
        given), estimate the corner frequency using
        :func:`utils.corner_frequency` with an S-wave velocity of 4 km/s.
      - If a range is given, we convert it to a corner frequency range as above
      and search the apparent corner frequency as the maximum of the phase
      velocity spectrum within this range using
      :func:`extra.apparent_corner_frequency`

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
    evd:
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
    fwin = 1.0 / (hdr["phase_end"] - hdr["phase_start"])

    # One sample more than the Nyquist frequency
    fnyq = (arr.shape[-1] - 1.0) / hdr["data_window"] / 2

    bpd = {}
    for iev, evn in zip(ievs, evns):
        print("{:02d} events to go   ".format(len(evns) - iev), end="\r")

        ev = evd[evn]

        phase_arr = arr[iev, :, :]

        # First get the corner frequency
        if auto_lowpass_method == "duration":
            # Corner frequency from duration
            fc = 1 / utils.source_duration(ev.mag)

        elif auto_lowpass_method == "corner":
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

        # No SNR optimization, use the corner frequency
        hpas = min(fwin, fnyq)
        lpas = min(fc, fnyq)

        if auto_bandpass_snr_target is not None and hpas < lpas:
            # Try to optimize bandpass within range
            hpas, lpas = extra.optimal_bandpass(
                phase_arr,
                fmin=hpas,
                fmax=lpas,
                min_snr=auto_bandpass_snr_target,
                **hdr.kwargs(extra.optimal_bandpass),
            )

        # Return the filter corners
        bpd[evn] = [float(hpas), float(lpas)]

    return bpd


def main_amplitude(
    config: core.Config,
    directory: Path = Path(),
    iteration: int = 0,
    overwrite: bool = False,
):
    """
    Compute relative amplitudes using options in config. Assemble path directory
    and iteration. Choose 'overwrite' to overwrite existing files.
    """

    ampdir = directory / "amplitude"
    if not ampdir.exists():
        logger.info(f"Target directory does not exist: {ampdir}. Creating.")
        ampdir.mkdir()

    stf = directory / config["station_file"]
    evf = directory / config["event_file"]
    ncpu = config["ncpu"]
    compare_method = config["amplitude_measure"]  # direct or indirect
    filter_method = config["amplitude_filter"]  # auto or manual

    exclude = io.read_exclude_file(core.file("exclude", directory=directory))

    stas = io.read_station_table(stf)

    # Exclude some observations
    stas = set(stas) - set(exclude["station"])
    wvids = set(core.iterate_waveid(stas)) - set(exclude["waveform"])

    # Read the events
    event_dict = io.read_event_table(evf)

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
                logger.debug(e)
                continue

            pasbnds[wvid] = {
                evn: [hdr["highpass"], hdr["lowpass"]] for evn in hdr["events_"]
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
                    logger.debug(e)
                    continue

                pasbnds[wvid] = phase_passbands(
                    arr,
                    hdr,
                    event_dict,
                    **config.kwargs(phase_passbands),
                    exclude=exclude,
                )

            io.save_yaml(bpf, pasbnds, format_bandpass=True)
            logger.info(f"Saved bandpass to file: {bpf}")

    else:
        raise ValueError(f"Unknown 'amplitude_filter': {filter_method}")

    # Collect the arguments to the amplitude function
    pargs = []
    sargs = []
    shmd = {}

    logger.info("Collecting observations. This may take a while...")
    if compare_method == "direct":
        for wvid in wvids:
            sta, pha = core.split_waveid(wvid)

            try:
                arr, hdr = io.read_waveform_array_header(sta, pha, iteration, directory)
            except FileNotFoundError as e:
                logger.debug(e)
                continue

            # Make sure what's excluded is excluded
            ievs, evns = qc.included_events(exclude, **hdr.kwargs(qc.included_events))

            if len(ievs) < 1:
                logger.warning(f"No events included for {wvid}. Continuing.")
                continue

            xarr = arr[ievs, :, :]
            hdr["events_"] = evns

            if hdr["combinations_from_file"]:
                pairs = io.read_combinations(
                    core.file("combination", sta, pha, iteration, directory)
                )
                combs = utils.valid_combinations(evns, pairs, pha)
            elif pha == "P":
                combs = core.iterate_event_pair(len(evns))
            else:
                combs = core.iterate_event_triplet(len(evns))

            # Make a shared array (Inspired by CatGPT 4o)
            dtype = xarr.dtype
            shape = xarr.shape
            shmd[wvid] = sm.SharedMemory(create=True, size=xarr.nbytes)
            shm = shmd[wvid]
            sharr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            sharr[:] = xarr[:]  # Copy data into shared memory
            mname = shm.name

            # Look up correct passband and collect the arguments
            if pha == "P":
                for a, b in combs:
                    hpas, lpas = signal.choose_passband(
                        [pasbnds[wvid][evns[i]][0] for i in [a, b]],
                        [pasbnds[wvid][evns[i]][1] for i in [a, b]],
                        config["min_dynamic_range"],
                    )

                    # Passband is within dynamic range
                    if hpas is not None:
                        # Old: Copy the arrays
                        # pargs.append(
                        #    (xarr[[a, b], :], hdr, hpas, lpas, evns[a], evns[b])
                        # )
                        # New: Using shared memory
                        pargs.append(
                            (
                                (a, b),
                                mname,
                                shape,
                                dtype,
                                hdr,
                                hpas,
                                lpas,
                                evns[a],
                                evns[b],
                            )
                        )

            if pha == "S":
                for a, b, c in combs:
                    hpas, lpas = signal.choose_passband(
                        [pasbnds[wvid][evns[i]][0] for i in [a, b, c]],
                        [pasbnds[wvid][evns[i]][1] for i in [a, b, c]],
                        config["min_dynamic_range"],
                    )

                    # Passband is within dynamic range
                    if hpas is not None:
                        # Old: Copy the arrays
                        # sargs.append(
                        #    (
                        #        xarr[[a, b, c], :],
                        #        hdr,
                        #        hpas,
                        #        lpas,
                        #        evns[a],
                        #        evns[b],
                        #        evns[c],
                        #    )
                        # )
                        # New: Using shared memory
                        sargs.append(
                            (
                                (a, b, c),
                                mname,
                                shape,
                                dtype,
                                hdr,
                                hpas,
                                lpas,
                                evns[a],
                                evns[b],
                                evns[c],
                            )
                        )

        # Argumets above pertain to these functions
        # p_amp_fun = amp.paired_p_amplitude_copies
        # s_amp_fun = amp.triplet_s_amplitude_copies
        p_amp_fun = amp.paired_p_amplitudes
        s_amp_fun = amp.triplet_s_amplitudes

    elif compare_method == "indirect":
        # TODO: implement shared memory

        for wvid in wvids:
            sta, pha = core.split_waveid(wvid)

            try:
                arr, hdr = io.read_waveform_array_header(sta, pha, iteration, directory)
            except FileNotFoundError as e:
                logger.debug(e)
                continue

            # Exclude the excluded events
            ievs, evns = qc.included_events(exclude, **hdr.kwargs(qc.included_events))
            xarr = arr[ievs, :, :]
            hdr["events_"] = evns

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

    logger.info(f"Collected {len(pargs)} P- and {len(sargs)} S-combinations")
    logger.info("Computing relative P-amplitudes...")
    # First process and save P ...
    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            result = pool.starmap(p_amp_fun, pargs)
    else:
        result = [p_amp_fun(*arg) for arg in pargs]

    if compare_method == "indirect":
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

    logger.info("Computing relative S-amplitudes...")
    # ... later S
    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            result = pool.starmap(s_amp_fun, sargs)
    else:
        result = [s_amp_fun(*arg) for arg in sargs]

    if compare_method == "indirect":
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

    # Let's release the shared memory
    logger.debug("Releasing shared memory")
    for shm in shmd.values():
        shm.close()
        shm.unlink()


def main_qc(config: core.Config, directory: Path):
    """Read amplitudes and discard according to criteria in config

    Before constructing the system, apply amplitude exlusion criteria and
    minumum number of equation constraints.
    """

    # A-priori meassures
    max_mis = config["max_amplitude_misfit"]
    max_mag_diff = config["max_magnitude_difference"]
    max_s1 = config["max_s_sigma1"]
    max_ev_dist = config["max_event_distance"]
    min_eq = config["min_equations"]
    max_gap = config["max_gap"]
    keep_other_s_equation = config["keep_other_s_equation"]
    max_s_equations = config["max_s_equations"]

    exclude = io.read_yaml(core.file("exclude", directory=directory))

    exclude_wvid = exclude["waveform"]

    ampsuf = config["amplitude_suffix"]

    exclude_phase = set(exclude["phase_manual"]).union(
        exclude["phase_auto_nodata"]
        + exclude["phase_auto_snr"]
        + exclude["phase_auto_ecn"]
    )

    exclude_events = set(exclude["event"])

    evd = io.read_event_table(config["event_file"])
    std = io.read_station_table(config["station_file"])
    phd = io.read_phase_table(config["phase_file"])

    for ph in "PS":
        infile = core.file(
            "amplitude_observation", directory=directory, phase=ph, suffix=ampsuf
        )

        amps = io.read_amplitudes(infile, ph)

        exclude_stations = set(
            core.split_waveid(wvid)[0]
            for wvid in exclude_wvid
            if core.split_waveid(wvid)[1] == ph
        ).union(exclude["station"])

        # Read the files again, this time unpack QCed values in arrays
        if ph == "P":
            sta, eva, evb, amp1, mis, cc, *_ = io.read_amplitudes(
                infile, ph, unpack=True
            )
            s1 = np.full_like(mis, -np.inf)  # Never exclud
            amp2 = np.zeros_like(amp1)  # No second amplitude
            evc = eva
        else:
            sta, eva, evb, evc, amp1, amp2, mis, cc, s1, *_ = io.read_amplitudes(
                infile, ph, unpack=True
            )

        # Direct exlusions using numpy

        # Stations
        iout = np.isin(sta, exclude_stations)

        # Phases
        iout |= [
            np.any(
                (
                    core.join_phaseid(a, s, ph) in exclude_phase,
                    core.join_phaseid(b, s, ph) in exclude_phase,
                    core.join_phaseid(c, s, ph) in exclude_phase,
                    a in exclude_events,
                    b in exclude_events,
                    c in exclude_events,
                )
            )
            for a, b, c, s in zip(eva, evb, evc, sta)
        ]

        logger.info(
            f"Excluded {(nout := sum(iout))} {ph}-observations from exclude file"
        )

        # Valid amplitude
        iout |= ~np.isfinite(amp1) | ~np.isfinite(amp2)
        logger.info(
            f"Excluded {sum(iout) - nout} more {ph}-observations due bad amplitude"
        )

        # Misfit
        nout = sum(iout)
        if max_mis is not None:
            iout |= ~(mis < max_mis)  # Exclude also nans
            logger.info(
                f"Excluded {sum(iout) - nout} more {ph}-observations due to high misfit"
            )

        # Sigma1
        nout = sum(iout)
        if max_s1 is not None:
            iout |= ~(s1 < max_s1)  # Exclude also nans
            logger.info(
                f"Excluded {sum(iout) - nout} more {ph}-observations due high sigma1"
            )

        iin = (~iout).nonzero()[0]  # Included amplitude indices

        # Reduce the list using indices
        amps = [amps[n] for n in iin]

        # Only include observations for which we have takeoff angles, ...
        amps = qc.clean_by_valid_takeoff_angle(amps, phd)

        # ... within magnitude difference ...
        amps = qc.clean_by_magnitude_difference(amps, evd, max_mag_diff)

        # ... within inter-event distance ...
        amps = qc.clean_by_event_distance(amps, evd, max_ev_dist)

        # Let's not overwrite, but save for later use
        if ph == "P":
            pamps = amps.copy()
        else:
            samps = amps.copy()

    # Make sure we have enough equations
    pamps, samps = qc.clean_by_equation_count_gap(
        pamps, samps, phd, min_eq, max_gap, keep_other_s_equation
    )

    # Make sure we don't have too many equations
    s_equations = len(samps) * (1 + int(keep_other_s_equation))
    if max_s_equations is not None:
        excess_eq = s_equations - max_s_equations
        logger.info(f"Have {s_equations} S-equations, need to reduce by {excess_eq}")

        batches = 10
        nex = int(excess_eq // batches + 1)

        while s_equations > max_s_equations:
            triplets, stas, miss, s1s = map(
                np.array,
                zip(
                    *[
                        (
                            (samp.event_a, samp.event_b, samp.event_c),
                            samp.station,
                            samp.misfit,
                            samp.sigma1,
                        )
                        for samp in samps
                    ]
                ),
            )
            std_sub = {stn: std[stn] for stn in set(stas)}

            sta_gap = utils.station_gap(std_sub, evd)

            # Redundancy score per observation (highest score least important)
            red_score = utils.pair_redundancy(triplets)

            # Gap score per observation (lowest score least important)
            gap_score = np.array([sta_gap[sta] for sta in stas])

            # Combine misfit and sigma meassure
            # Prefer disinct S-wave combinations with low misfit
            # Lower is better
            mis_sigma = (1 - (max_s1 - s1s)) * (1 - (max_mis - miss))

            # score sort: from most important to least important
            ssort = np.lexsort((mis_sigma, -gap_score, red_score))
            samps = [samps[i] for i in ssort[:-nex]]

            # Check once more we have enough equations
            pamps, samps = qc.clean_by_equation_count_gap(
                pamps, samps, phd, min_eq, max_gap, keep_other_s_equation
            )

            s_equations = len(samps) * (1 + int(keep_other_s_equation))
            logger.debug(f"Reduced to {s_equations} S-equations")

            # TODO: implement variable nex

    if len(pamps) + len(samps) == 0:
        raise RuntimeError("No observations left. Relax your QC criteria.")

    # Write to file
    for ph, amps in zip("PS", [pamps, samps]):
        outfile = core.file(
            "amplitude_observation",
            directory=directory,
            phase=ph,
            suffix=ampsuf + core.clean_amplitude_suffix,
        )
        io.save_amplitudes(outfile, amps)


def main_solve(
    config: core.Config, directory: Path = Path(), iteration: int = 0, do_predict=False
):
    """Construct and validate linear system from amplitude measurement. Solve
    for moment tensors."""

    evf = directory / config["event_file"]
    stf = directory / config["station_file"]
    phf = directory / config["phase_file"]
    refmtf = directory / config["reference_mt_file"]

    max_misfit = config["max_amplitude_misfit"]
    min_misfit = config["min_amplitude_misfit"]
    min_weight = config["min_amplitude_weight"]
    irefs = config["reference_mts"]
    constraint = config["mt_constraint"]
    refmt_weight = config["reference_weight"]
    ncpu = config["ncpu"]
    keep_other_s_equation = config["keep_other_s_equation"]
    sfac = 1 + int(keep_other_s_equation)

    nboot = config["bootstrap_samples"]

    insuf = config["amplitude_suffix"] + core.clean_amplitude_suffix
    outsuf = config["amplitude_suffix"] + config["result_suffix"]
    synsuf = outsuf + core.synthetic_amplitude_suffix

    mt_elements = ls.mt_elements(constraint)

    evd = io.read_event_table(evf)
    phd = io.read_phase_table(phf)
    stad = io.read_station_table(stf)
    mtd = io.read_mt_table(refmtf)

    # Read amplitudes from file
    p_amplitudes = io.read_amplitudes(
        core.file(
            "amplitude_observation", phase="P", suffix=insuf, directory=directory
        ),
        "P",
    )

    s_amplitudes = io.read_amplitudes(
        core.file(
            "amplitude_observation", phase="S", suffix=insuf, directory=directory
        ),
        "S",
    )

    # Find the connected events, given moment tensors
    links = qc.connected_events(irefs, p_amplitudes, s_amplitudes)
    incl_ev = list(links)  # Only event list, no connection count

    # Reduced set of amplitudes, containing only valid observations
    pamp_subset = [
        pamp
        for pamp in p_amplitudes
        if all(np.isin([pamp.event_a, pamp.event_b], incl_ev))
    ]

    samp_subset = [
        samp
        for samp in s_amplitudes
        if all(np.isin([samp.event_a, samp.event_b, samp.event_c], incl_ev))
    ]

    n_p = len(pamp_subset)
    n_s = len(samp_subset) * sfac
    n_ref = len(irefs)

    # Build homogenos part of linear system
    isparse = True
    if isparse:
        # as sparse array
        Ah, bh = ls.homogenous_amplitude_equations_sparse(
            pamp_subset,
            samp_subset,
            incl_ev,
            stad,
            evd,
            phd,
            constraint,
            None,
            keep_other_s_equation,
            ncpu,
        )
    else:
        # as dense array
        Ah, bh = ls.homogenous_amplitude_equations(
            pamp_subset,
            samp_subset,
            incl_ev,
            stad,
            evd,
            phd,
            constraint,
            None,
            keep_other_s_equation,
        )

    # Normalization applied to columns
    ev_norm = ls.norm_event_median_amplitude(Ah, mt_elements)

    if isparse:
        Ah = (Ah * ev_norm).tocsc(copy=False)
    else:
        Ah *= ev_norm

    # Weight applied by row
    mis_weights = np.vstack(
        [
            ls.weight_misfit(amp, min_misfit, max_misfit, min_weight, "P")
            for amp in pamp_subset
        ]
        + [
            ls.weight_misfit(
                amp, min_misfit, max_misfit, min_weight, "S", keep_other_s_equation
            )
            for amp in samp_subset
        ]
    )

    amp_weights = np.vstack(
        [1.0 for _ in pamp_subset]
        + [ls.weight_s_amplitude(amp, keep_other_s_equation) for amp in samp_subset]
    )

    # Equation norm
    eq_norm = ls.condition_homogenous_matrix_by_norm(Ah)
    p_norm, s_norm, _ = ls.unpack_equation_vector(
        eq_norm, n_p, 0, mt_elements, keep_other_s_equation
    )

    # Apply the weights only after measuring the norm
    if isparse:
        Ah = (Ah * (mis_weights * amp_weights)).tocsc(copy=False)
        Ah = (Ah * eq_norm).tocsc(copy=False)
    else:
        Ah *= mis_weights * amp_weights
        Ah *= eq_norm

    # Build inhomogenous equations
    Ai, bi = ls.reference_mt_equations(irefs, mtd, incl_ev, constraint)

    # Collect and apply weights
    mean_moment = mt.mean_moment([mtd[iev] for iev in irefs])

    # Indices of reference events
    incl_ref = [incl_ev.index(iref) for iref in irefs]

    # Normalization of the reference event
    refev_norm = ls.reference_mt_event_norm(ev_norm, incl_ref, mt_elements)
    Ai *= refmt_weight
    bi *= refmt_weight / mean_moment / refev_norm

    # Scale of resulting relative moment tensors
    ev_scale = mean_moment * ev_norm

    # Dense matrix
    if isparse:
        # Sparse matrix
        A = sparse.vstack((Ah, Ai), format="csc")
    else:
        # Dense matrix
        A = sparse.coo_matrix(np.vstack((Ah, Ai))).tocsc()

    b = np.vstack((bh, bi))

    # save_npz(core.file("amplitude_matrix", directory=directory, suffix=outsuf), A)

    logger.info(
        f"Solving linear system of {A.shape[1]} variables and {A.shape[0]} equations..."
    )

    # Invert and save results
    m, residuals = ls.solve_lsmr(A, b, ev_scale)

    # Moment residuals
    p_residuals, s_residuals, _ = ls.unpack_equation_vector(
        residuals, n_p, n_ref, mt_elements, keep_other_s_equation
    )

    relmts = {
        incl_ev[i]: momt
        for i, momt in enumerate(mt.mt_tuples(m, constraint))
        if any(momt)
    }

    # Save the results right away
    io.make_mt_table(
        relmts, core.file("relative_mt", suffix=outsuf, directory=directory)
    )

    # Indices in subsets of events
    ievp = utils.event_indices(pamp_subset)
    ievs = utils.event_indices(samp_subset)

    if do_predict:
        # Compute synthetic amplitudes and posteori-residuals
        logger.info("Computing synthetic amplitudes and residuals")
        p_pairs = [(pamp.station, pamp.event_a, pamp.event_b) for pamp in pamp_subset]
        p_sta = set(p_pair[0] for p_pair in p_pairs)

        Asyn, p_sigmas = amp.synthetic_p(relmts, evd, stad, phd, p_pairs)

        s_triplets = [
            (samp.station, samp.event_a, samp.event_b, samp.event_c)
            for samp in samp_subset
        ]
        s_sta = set(s_trip[0] for s_trip in s_triplets)

        Bsyn, _, s_sigmas = amp.synthetic_s(
            relmts,
            evd,
            stad,
            phd,
            s_triplets,
            False,
        )

        # Load the waveforms to compute a posteori misfits and ccs
        arrd = {}
        hdrd = {}
        for stas, pha in zip([p_sta, s_sta], "PS"):
            for sta in stas:
                wvid = core.join_waveid(sta, pha)
                arrd[wvid], hdrd[wvid] = io.read_waveform_array_header(
                    sta, pha, iteration, directory
                )

        logger.info("Computing amplitude misfits and correlations...")

        # ppms, ppcs, spms, spcs = amp.predicted_misfit_correlation(
        #    pamp_subset, Asyn, samp_subset, Bsyn, arrd, hdrd, max_workers=ncpu,
        # )
        ppms, ppcs = zip(
            *[
                (
                    amp.p_misfit(
                        mat := utils.concat_components(
                            signal.demean_filter_window(
                                arrd[core.join_waveid(pamp.station, "P")][
                                    [
                                        (
                                            hdr := hdrd[
                                                core.join_waveid(pamp.station, "P")
                                            ]
                                        )["events_"].index(pamp.event_a),
                                        hdr["events_"].index(pamp.event_b),
                                    ]
                                ],
                                hdr["sampling_rate"],
                                hdr["phase_start"],
                                hdr["phase_end"],
                                hdr["taper_length"],
                                pamp.highpass,
                                pamp.lowpass,
                            )
                        ),
                        Aab,
                    ),
                    amp.p_reconstruction_correlation(mat),
                )
                for pamp, Aab in zip(pamp_subset, Asyn)
            ]
        )

        # S-wave posterior misfit and correlation
        spms, spcs = zip(
            *[
                (
                    amp.s_misfit(
                        mat := utils.concat_components(
                            signal.demean_filter_window(
                                arrd[core.join_waveid(samp.station, "S")][
                                    [
                                        (
                                            hdr := hdrd[
                                                core.join_waveid(samp.station, "S")
                                            ]
                                        )["events_"].index(samp.event_a),
                                        hdr["events_"].index(samp.event_b),
                                        hdr["events_"].index(samp.event_c),
                                    ]
                                ],
                                hdr["sampling_rate"],
                                hdr["phase_start"],
                                hdr["phase_end"],
                                hdr["taper_length"],
                                samp.highpass,
                                samp.lowpass,
                            )
                        ),
                        B[0],
                        B[1],
                    ),
                    amp.s_reconstruction_correlation(mat, B[0], B[1]),
                )
                for samp, B in zip(samp_subset, Bsyn)
            ]
        )

        logger.info("Collecting results...")
        pamp_synthetic = [
            core.P_Amplitude_Ratio(
                pamp.station,
                pamp.event_a,
                pamp.event_b,
                Aab,
                ppm,
                ppc,
                p_sigma[0],
                p_sigma[1],
                pamp.highpass,
                pamp.lowpass,
            )
            for pamp, Aab, p_sigma, ppm, ppc in zip(
                pamp_subset, Asyn, p_sigmas, ppms, ppcs
            )
        ]

        samp_synthetic = [
            core.S_Amplitude_Ratios(
                samp.station,
                samp.event_a,
                samp.event_b,
                samp.event_c,
                B[0],
                B[1],
                spm,
                spc,
                ss[0],
                ss[1],
                ss[2],
                samp.highpass,
                samp.lowpass,
            )
            for samp, B, ss, spm, spc in zip(samp_subset, Bsyn, s_sigmas, spms, spcs)
        ]

        # Amplitude oberservations
        Aobs = np.array([pamp.amp_ab for pamp in pamp_subset])
        B1obs = np.array([samp.amp_abc for samp in samp_subset])
        B2obs = np.array([samp.amp_acb for samp in samp_subset])

        # Amplitude residuals
        Ares = Aobs / Asyn
        B1res = B1obs / Bsyn[:, 0]
        B2res = B2obs / Bsyn[:, 1]

        logger.info("Collecting event statistics...")

        # amplitude RMS per event
        amp_rmss = {
            evn: np.sqrt(
                np.sum(utils.signed_log(Ares[ievp[evn]]) ** 2)
                + np.sum(utils.signed_log(B1res[ievs[evn]]) ** 2)
                + np.sum(utils.signed_log(B2res[ievs[evn]]) ** 2)
            )
            / (len(ievp[evn]) + sfac * len(ievs[evn]))
            for evn in relmts
        }

        # Synthetic amplitudes
        for ph, amps in zip("PS", [pamp_synthetic, samp_synthetic]):
            outfile = core.file(
                "amplitude_observation",
                directory=directory,
                phase=ph,
                suffix=synsuf,
            )
            io.save_amplitudes(outfile, amps)

        # Save reduced amplitude set to file. Use the output suffix.
        io.save_amplitudes(
            core.file(
                "amplitude_summary", phase="P", suffix=outsuf, directory=directory
            ),
            pamp_subset,
            [
                p_residuals,
                Ares,
                mis_weights[:n_p].flat[:],
                amp_weights[:n_p].flat[:],
                p_norm,
                Asyn,
                ppms,
            ],
            [
                " MomResidual",
                "AmpResidual",
                "MisfitWght",
                "AmplWght",
                "EquaNorm",
                "PredAab ",
                "PredMis",
            ],
            [
                "{:11.3e}",
                "{:11.3e}",
                "{:10.5f}",
                "{:8.4f}",
                "{:7.2e}",
                "{:8.2e}",
                "{:7.5f}",
            ],
        )

        # Recall: s_residual and eq_norm do not have nan for the 2nd S equation, if we excluded them
        io.save_amplitudes(
            core.file(
                "amplitude_summary", phase="S", suffix=outsuf, directory=directory
            ),
            samp_subset,
            [
                s_residuals[:, 0],
                s_residuals[:, 1],
                B1res,
                B2res,
                mis_weights[n_p : n_p + n_s : sfac].flat[:],
                amp_weights[n_p : n_p + n_s : sfac].flat[:],
                s_norm[:, 0],
                s_norm[:, 1],
                Bsyn[:, 0],
                Bsyn[:, 1],
                spms,
            ],
            [
                " MomResidual1",
                "MomResidual2",
                "AmpResidual1",
                "AmpResidual2",
                "MisfitWght",
                "AmplWght",
                "EquaNorm1",
                "EquaNorm2",
                "PredBabc",
                "PredBacb",
                "PredMis",
            ],
            [
                "{:12.3e}",
                "{:12.3e}",
                "{:12.3e}",
                "{:12.3e}",
                "{:10.5f}",
                "{:8.4f}",
                "{:9.3e}",
                "{:9.3e}",
                "{:8.2e}",
                "{:8.2e}",
                "{:7.5f}",
            ],
        )

    # Moment RMS per event
    mom_rmss = {
        evn: np.sqrt(
            np.sum(p_residuals[ievp[evn]] ** 2) + np.sum(s_residuals[ievs[evn], :] ** 2)
        )
        / (len(ievp[evn]) + 2 * len(ievs[evn]))
        for evn in relmts
    }

    # Average misfits and correlation coefficients
    misps, ccps = np.array([(amp.misfit, amp.correlation) for amp in pamp_subset]).T
    misss, ccss = np.array([(amp.misfit, amp.correlation) for amp in samp_subset]).T

    avmiss = {
        evn: np.mean(np.concatenate((misps[ievp[evn]], misss[ievs[evn]])))
        for evn in relmts
    }

    avccs = {
        evn: utils.fisher_average(np.concatenate((ccps[ievp[evn]], ccss[ievs[evn]])))
        for evn in relmts
    }

    gaps = angle.azimuth_gap(phd, pamp_subset, samp_subset)

    if not do_predict:
        # We need predictions to compute the amplitude RMS
        amp_rmss = {evn: np.nan for evn in relmts}

    logger.info("Saving files.")
    io.save_mt_result_summary(
        core.file("mt_summary", suffix=outsuf, directory=directory),
        evd,
        relmts,
        gaps,
        links,
        avmiss,
        avccs,
        mom_rmss,
        amp_rmss,
    )

    # Bootstrap
    if nboot is not None and nboot > 0:
        m_boots = ls.bootstrap_lsmr(A, b, ev_scale, n_p, n_s, nboot, 0, 1)

        # Convert to MT dict
        # bootmts = {i: [] for i in range(m_boots.shape[1] // mt_elements)}
        bootmts = {i: [] for i in incl_ev}
        for m_boot in m_boots:
            for i, momt in enumerate(mt.mt_tuples(m_boot, constraint)):
                bootmts[incl_ev[i]].append(momt)

        # Make and save a
        io.make_mt_table(
            bootmts,
            core.file("relative_mt", suffix=outsuf + "_boot", directory=directory),
        )


def main_plot_alignment(
    arrf: Path,
    config: core.Config = None,
    do_exclude=False,
    sort: str = "pci",
    highligh_events: list[int] = [],
):
    """Plot the waveform array with parameters relevant to judging the alignment"""

    # Find where we are
    subdir = arrf.parts[-2]
    directory = Path(*arrf.parts[:-2])

    iteration = 0
    if subdir.startswith("align"):
        iteration = int(subdir.replace("align", ""))

    # Load the header first
    hdrf = arrf.with_name(arrf.stem.replace("-wvarr", "-hdr.yaml"))
    hdr = io.read_header(
        hdrf,
        default_name=core.file(
            "waveform_header", n_align=iteration, directory=directory
        ),
    )
    phase = hdr["phase"]
    station = hdr["station"]
    event_list = np.array(hdr["events_"])

    event_dict = {}
    station_dict = {}
    if config is not None:
        if config["event_file"] is not None:
            evf = directory / config["event_file"]
            event_dict = io.read_event_table(evf)
        if config["station_file"] is not None:
            stf = directory / config["station_file"]
            station_dict = io.read_station_table(stf)

    dest = (station, phase, iteration, directory)
    arr = np.load(arrf)

    try:
        dt_mccc, dt_rms = np.loadtxt(core.file("mccc_time_shift", *dest), unpack=True)
    except (FileNotFoundError, ValueError):
        dt_mccc = dt_rms = None

    try:
        dt_pca = np.loadtxt(core.file("pca_time_shift", *dest))
    except FileNotFoundError:
        dt_pca = None

    try:
        ccij = np.loadtxt(core.file("cc_matrix", *dest))
    except FileNotFoundError:
        ccij = None

    if do_exclude:
        excl = io.read_exclude_file(core.file("exclude", directory=directory))
        _, event_list = qc.included_events(excl, **hdr.kwargs(qc.included_events))

    plot.alignment(
        arr,
        hdr,
        dt_mccc,
        dt_rms,
        dt_pca,
        ccij,
        event_list,
        event_dict,
        station_dict,
        sort,
        highligh_events,
    )
    input("Press any key to continue...")


def get_arguments(args=None):
    """Get command line options for :func:`main_align()`"""

    parser = ArgumentParser(
        description="""
Software for computing relative seismic moment tensors"""
    )

    subpars = parser.add_subparsers(dest="mode")

    # Subparsers
    init_p = subpars.add_parser("init", help="Initialize default directories and files")

    align_p = subpars.add_parser(
        "align",
        help="Align waveforms",
        epilog=("When neither '--pca' nor '--mccc' are given, " "we assume both."),
    )

    exclude_p = subpars.add_parser(
        "exclude", help="Exclude phase observations from alignment"
    )

    amp_p = subpars.add_parser(
        "amplitude", help="Measure relative amplitudes on aligned waveforms"
    )

    qc_p = subpars.add_parser(
        "qc", help="Apply quality control parameters to amplitude measurements"
    )

    solve_p = subpars.add_parser(
        "solve", help="Compute moment tensors from amplitude measurements"
    )

    plot_p = subpars.add_parser("plot", help="Plot results to screen")

    # Now set the functions to be called
    init_p.set_defaults(command=core.init)
    align_p.set_defaults(command=main_align)
    exclude_p.set_defaults(command=main_exclude)
    amp_p.set_defaults(command=main_amplitude)
    qc_p.set_defaults(command=main_qc)
    solve_p.set_defaults(command=main_solve)

    # Global arguments
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Use this configuration file",
        default=core.file("config"),
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing files",
        action="store_true",
    )

    parser.add_argument(
        "-n", "--n_align", nargs="?", type=int, help="Alignment iteration", default=0
    )

    # Subparser arguments
    init_p.add_argument(
        "directory",
        type=Path,
        default=".",
        nargs="?",
        help="Name of the directory to initiate",
    )

    # Sub arguments of the alignment routine
    align_p.add_argument(
        "--mccc",
        action="store_true",
        help="Align with Multi-Channel Cross Correlation (MCCC)",
    )

    # Sub arguments of the alignment routine
    align_p.add_argument(
        "--pca",
        action="store_true",
        help="Align with principal component analysis (PCA).",
    )

    # Sub arguments of the exlusion routine
    exclude_p.add_argument(
        "--no-data",
        action="store_true",
        help="Exlude data with no data or data containing NaNs",
    )

    exclude_p.add_argument(
        "--snr",
        action="store_true",
        help=(
            "Exlude data with signal to noise ratio lower than "
            "'min_signal_noise_ratio' in the station header file"
        ),
    )

    exclude_p.add_argument(
        "--cc",
        action="store_true",
        help=(
            "Exlude data with correlation coefficient lower than "
            "'min_correlation' in the station header file"
        ),
    )

    exclude_p.add_argument(
        "--ecn",
        action="store_true",
        help=(
            "Exlude data with expansion coefficient norm lower than "
            "'min_expansion_coefficient_norm' in the station header file"
        ),
    )

    solve_p.add_argument(
        "--predict",
        action="store_true",
        help=(
            "Predict relative amplitudes of the solution and compute "
            "prediction misfits"
        ),
    )

    plot_p.add_argument(
        "what",
        choices=["alignment"],
        help=(
            "What kind of plot to make: \n"
            "* alignment: Plot alignment diagnostics. Give path to a -wvarr.npy file."
        ),
    )

    plot_p.add_argument(
        "file",
        type=Path,
        help="The file to be plotted",
    )

    plot_p.add_argument(
        "--sort",
        type=str,
        help="The sorting to apply: 'pci' (default), 'magnitude', 'none'",
        choices=["pci", "magnitude", "none"],
        default="pci",
    )

    plot_p.add_argument(
        "--exclude",
        action="store_true",
        help="Exclude events listed in the exclude file",
    )

    plot_p.add_argument(
        "--highlight",
        type=int,
        nargs="+",
        help="Event IDs to highligh in the plot",
        default=[],
    )

    parsed = parser.parse_args(args)

    if parsed.mode is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parsed


def main(args=None):
    # Subdirectory, e.g. A_Muji
    parsed = get_arguments(args)

    if parsed.mode == "init":
        parent = parsed.directory
        parsed.command(parent)
        return

    conff = parsed.config
    config = io.read_config(conff)

    # Let's parse the keyword arguments explicitly
    n_align = parsed.n_align
    overwrite = parsed.overwrite
    parent = conff.parent

    # Convert parsers to function kwargs
    kwargs = dict(directory=parent)
    if parsed.mode == "solve":
        kwargs.update(dict(iteration=n_align, do_predict=parsed.predict))

    if parsed.mode == "amplitude" or parsed.mode == "align" or parsed.mode == "exclude":
        kwargs.update(dict(iteration=n_align, overwrite=overwrite))

    if parsed.mode == "align":
        if parsed.mccc:
            logger.info("Aligning with MCCC")
        if parsed.pca:
            logger.info("Aligning with PCA")
        if not (parsed.pca or parsed.mccc):
            logger.info("Aligning with MCCC and PCA")
            parsed.pca = True
            parsed.mccc = True
        kwargs.update(dict(do_pca=parsed.pca, do_mccc=parsed.mccc))

    if parsed.mode == "exclude":
        kwargs.update(
            dict(
                do_nodata=parsed.no_data,
                do_snr=parsed.snr,
                do_cc=parsed.cc,
                do_ecn=parsed.ecn,
            )
        )

    if parsed.mode == "plot":
        if parsed.what == "alignment":
            main_plot_alignment(
                parsed.file,
                config,
                parsed.exclude,
                parsed.sort,
                parsed.highlight,
            )
            return

    # The command to be executed is defined above for each of the subparsers
    parsed.command(config, **kwargs)


if __name__ == "__main__":
    main()
