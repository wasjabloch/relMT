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
import sys
import multiprocessing as mp
from argparse import ArgumentParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


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
    """Exclude observations from alignment procedure"""
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


def main_amplitude(
    config: core.Config, directory: Path, iteration: int, overwrite: bool = False
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

            io.save_yaml(bpf, pasbnds, format_bandpass=True)
            logger.info(f"Saved bandpass to file: {bpf}")

    else:
        raise ValueError(f"Unknown 'amplitude_filter': {filter_method}")

    # Collect the arguments to the amplitude function
    pargs = []
    sargs = []

    if compare_method == "direct":
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

    elif compare_method == "indirect":

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


def main_qc(config: core.Config, directory: Path):
    """Read amplitudes and discard according to criteria in config

    Before constructing the system, apply amplitude exlusion criteria and
    minumum number of equation constraints.
    """

    max_mis = config["max_amplitude_misfit"]
    max_mag_diff = config["max_magnitude_difference"]
    max_s1 = config["max_s_sigma1"]
    max_ev_dist = config["max_event_distance"]
    min_eq = config["min_equations"]

    exclude = io.read_yaml(core.file("exclude", directory=directory))

    exclude_wvid = exclude["waveform"]

    ampsuf = config["amplitude_suffix"]

    evl = io.read_event_table(config["event_file"])
    phd = io.read_phase_table(config["phase_file"])

    samps = io.read_amplitudes(
        core.file(
            "amplitude_observation", directory=directory, phase="S", suffix=ampsuf
        ),
        "S",
    )

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
            sta, _, _, _, mis, *_ = io.read_amplitudes(infile, ph, unpack=True)
            s1 = np.full_like(mis, -np.inf)  # Never exclud
        else:
            sta, _, _, _, _, _, mis, s1, *_ = io.read_amplitudes(
                infile, ph, unpack=True
            )

        # Direct exlusions using numpy

        # Stations
        iout = np.isin(sta, exclude_stations)
        logger.info(
            f"Excluded {(nout := sum(iout))} {ph}-observations from exclude file"
        )

        # Misfit
        if max_mis is not None:
            iout |= mis > max_mis
            logger.info(
                f"Excluded {sum(iout) - nout} more {ph}-observations due to high misfit"
            )

        # Sigma1
        nout = sum(iout)
        if max_s1 is not None:
            iout |= s1 > max_s1
            logger.info(
                f"Excluded {sum(iout) - nout} more {ph}-observations due high sigma1"
            )

        iin = (~iout).nonzero()[0]  # Included amplitude indices

        # Reduce the list using indices
        amps = [amps[n] for n in iin]

        # Only include observations for which we have takeoff angles, ...
        amps = qc.clean_by_valid_takeoff_angle(amps, phd)

        # ... within magnitude difference ...
        amps = qc.clean_by_magnitude_difference(amps, evl, max_mag_diff)

        # ... within inter-event distance ...
        amps = qc.clean_by_event_distance(amps, evl, max_ev_dist)

        # Let's not overwrite, but save for later use
        if ph == "P":
            pamps = amps.copy()
        else:
            samps = amps.copy()

    # Make sure we have enough equations
    pamps, samps = qc.clean_by_equation_count(pamps, samps, min_eq)

    # Write to file
    for ph, amps in zip("PS", [pamps, samps]):
        outfile = core.file(
            "amplitude_observation",
            directory=directory,
            phase=ph,
            suffix=ampsuf + core.clean_amplitude_suffix,
        )
        io.save_amplitudes(outfile, amps)


def main_solve(config: core.Config, directory: Path = Path()):
    """Construct and validate linear system from amplitude measurement. Solve
    for moment tensors."""

    ls.logger.setLevel("WARNING")

    evf = directory / config["event_file"]
    stf = directory / config["station_file"]
    phf = directory / config["phase_file"]
    refmtf = directory / config["reference_mt_file"]

    max_misfit = config["max_amplitude_misfit"]
    min_misfit = config["min_amplitude_misfit"]
    irefs = config["reference_mts"]
    constraint = config["mt_constraint"]
    refmt_weight = config["reference_weight"]

    nboot = config["bootstrap_samples"]

    # Avoid ovewriting reduced amplitude files.
    if config["amplitude_suffix"] == config["result_suffix"]:
        msg = "Naming conflict in configuration. Please supply a "
        msg += "'result_suffix' that is different from 'amplitude_suffix'."
        raise ValueError(msg)

    insuf = config["amplitude_suffix"] + core.clean_amplitude_suffix
    outsuf = config["result_suffix"]

    mt_elements = ls.mt_elements(constraint)

    evl = io.read_event_table(evf)
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
    inev = qc.connected_events(irefs, p_amplitudes, s_amplitudes)

    # Reduced set of amplitudes, containing only valid observations
    pamp_subset = [
        pamp
        for pamp in p_amplitudes
        if all(np.isin([pamp.event_a, pamp.event_b], inev))
    ]

    samp_subset = [
        samp
        for samp in s_amplitudes
        if all(np.isin([samp.event_a, samp.event_b, samp.event_c], inev))
    ]

    # Build homogenos part of linear system
    Ah, bh = ls.homogenous_amplitude_equations(
        pamp_subset, samp_subset, inev, stad, evl, phd, constraint
    )

    # Normalization applied to columns
    ev_norm = ls.norm_event_median_amplitude(Ah, mt_elements)
    Ah *= ev_norm

    # Weight applied by row
    min_weight = 0.05
    mis_weights = np.vstack(
        [
            ls.weight_misfit(amp, min_misfit, max_misfit, min_weight, "P")
            for amp in pamp_subset
        ]
        + [
            ls.weight_misfit(amp, min_misfit, max_misfit, min_weight, "S")
            for amp in samp_subset
        ]
    )

    amp_weights = np.vstack(
        [1.0 for _ in pamp_subset] + [ls.weight_s_amplitude(amp) for amp in samp_subset]
    )

    eq_norm = ls.condition_homogenous_matrix_by_norm(Ah)

    # Apply the weights only after measuring the norm
    Ah *= mis_weights * amp_weights

    Ah *= eq_norm

    # Build inhomogenous equations
    Ai, bi = ls.reference_mt_equations(irefs, mtd, len(evl), constraint)

    # Collect and apply weights
    mean_moment = mt.mean_moment([mtd[iev] for iev in irefs])
    refev_norm = ls.reference_mt_event_norm(ev_norm, irefs, mt_elements)
    Ai *= refmt_weight
    bi *= refmt_weight / mean_moment / refev_norm

    # Scale of resulting relative moment tensors
    ev_scale = mean_moment * ev_norm

    A = coo_matrix(np.vstack((Ah, Ai))).tocsc()
    b = np.vstack((bh, bi))

    # save_npz(core.file("amplitude_matrix", directory=directory, suffix=outsuf), A)

    n_p = len(pamp_subset)
    n_s = len(samp_subset) * 2
    n_ref = len(irefs)

    # Invert and save results
    m, residuals = ls.solve_lsmr(A, b, ev_scale)
    p_residuals, s_residuals, _ = ls.unpack_resiudals(
        residuals, n_p, n_ref, mt_elements
    )

    relmts = {
        inev[i]: momt for i, momt in enumerate(mt.mt_tuples(m, constraint)) if any(momt)
    }

    # Compute synthetic amplitudes and posteori-residuals
    Asyn, Bsyn, *_ = amp.synthetic(
        relmts,
        evl,
        stad,
        phd,
        [(pamp.station, pamp.event_a, pamp.event_b) for pamp in pamp_subset],
        [
            (samp.station, samp.event_a, samp.event_b, samp.event_c)
            for samp in samp_subset
        ],
        False,
    )

    Aobs = np.array([pamp.amp_ab for pamp in pamp_subset])
    B1obs = np.array([samp.amp_abc for samp in samp_subset])
    B2obs = np.array([samp.amp_acb for samp in samp_subset])

    Ares = Aobs / Asyn
    B1res = B1obs / Bsyn[:, 0]
    B2res = B2obs / Bsyn[:, 1]

    io.make_mt_table(
        relmts, core.file("relative_mt", suffix=outsuf, directory=directory)
    )

    # Save reduced amplitude set to file. Use the output suffix.
    io.save_amplitudes(
        core.file("amplitude_summary", phase="P", suffix=outsuf, directory=directory),
        pamp_subset,
        [
            p_residuals,
            Ares,
            mis_weights[:n_p].flat[:],
            amp_weights[:n_p].flat[:],
            eq_norm[:n_p].flat[:],
        ],
        ["Residual_AB", "Aab_Misprediction", "mis_weight", "amp_weight", "eq_norm"],
    )

    io.save_amplitudes(
        core.file("amplitude_summary", phase="S", suffix=outsuf, directory=directory),
        samp_subset,
        [
            s_residuals[:, 0],
            s_residuals[:, 1],
            B1res,
            B2res,
            mis_weights[n_p : n_p + n_s : 2].flat[:],
            amp_weights[n_p : n_p + n_s : 2].flat[:],
            eq_norm[n_p : n_p + n_s : 2].flat[:],
            eq_norm[n_p + 1 : n_p + n_s : 2].flat[:],
        ],
        [
            "Residual_ABC",
            "Residual_ACB",
            "ABC_Misprediction",
            "ACB_Misprediction",
            "mis_weight",
            "amp_weight",
            "eq_norm",
            "eq_norm2",
        ],
    )

    # Bootstrap
    if nboot > 0:
        m_boots = ls.bootstrap_lsmr(A, b, ev_scale, n_p, n_s, nboot, 0, 1)

        # Convert to MT dict
        bootmts = {i: [] for i in range(m_boots.shape[1] // mt_elements)}
        for m_boot in m_boots:
            for i, momt in enumerate(mt.mt_tuples(m_boot, constraint)):
                bootmts[i].append(momt)

        # Make and save a
        io.make_mt_table(
            bootmts,
            core.file("relative_mt", suffix=outsuf + "_boot", directory=directory),
        )


def get_arguments(args=None):
    """Get command line options for :func:`main_align()`"""

    parser = ArgumentParser(
        description="""
Software for computing relative seismic moment tensors"""
    )

    subpars = parser.add_subparsers(dest="mode")

    init_p = subpars.add_parser("init", help="Initialize default directories and files")
    align_p = subpars.add_parser("align", help="Align waveforms")
    amp_p = subpars.add_parser(
        "amplitude", help="Measure relative amplitudes on aligned waveforms"
    )
    qc_p = subpars.add_parser(
        "qc", help="Apply quality control parameters to amplitude measurements"
    )
    solve_p = subpars.add_parser(
        "solve", help="Compute moment tensors from amplitude measurements"
    )

    # Now set the functions to be called
    init_p.set_defaults(command=core.init)
    align_p.set_defaults(command=main_align)
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
        "-n", "--n_align", type=int, help="Alignment iteration", default=0
    )

    # Subparser arguments
    init_p.add_argument(
        "directory",
        type=Path,
        default=".",
        nargs="?",
        help="Name of the directory to initiate",
    )

    # Align sub arguments
    align_p.add_argument(
        "--no-data",
        action="store_true",
        help="Exlude data with no data or data containing NaNs",
    )

    align_p.add_argument(
        "--snr",
        action="store_true",
        help=(
            "Exlude data with signal to noise ratio higher than "
            "'min_signal_noise_ratio' in the configuration file"
        ),
    )

    align_p.add_argument(
        "--ecn",
        action="store_true",
        help=(
            "Exlude data with expansion coefficient norm higher than "
            "'min_expansion_coefficient_norm' in the configuration file"
        ),
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

    kwargs = dict(directory=parent)
    if parsed.mode == "amplitude" or parsed.mode == "align":
        kwargs.update(dict(iteration=n_align, overwrite=overwrite))

    # The command to be executed is defined above for each of the subparsers
    parsed.command(config, **kwargs)


if __name__ == "__main__":
    main()
