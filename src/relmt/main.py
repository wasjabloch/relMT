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

"""Entry points for relMT command line interface."""

from relmt import io, utils, align, core, signal, extra, amp, ls, mt, qc, angle, plot
from scipy import sparse
from pathlib import Path
import yaml
import numpy as np
import sys
import multiprocessing as mp
from multiprocessing import shared_memory as sm
from argparse import ArgumentParser, Namespace

logger = core.register_logger(__name__)


def align_entry(
    config: core.Config,
    directory: Path = Path(),
    iteration: int = 0,
    do_mccc: bool = True,
    do_pca: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Align waveform files and write results into next alignment directory.

    This function is called when executing 'relmt align' from the command line.

    Parameters
    ----------
    config:
        Configuration object with alignment parameters. Content of the file read
        by the '--config' option.
    directory:
        Root directory of the project, containing the 'data/' and 'align?/'
        subfolders. Path of the file referenced by the '--config' option.
    iteration:
        Current alignment iteration number. `0`: read from 'data/', `>0` read from
        alignment iteration folder. Number parsed to the '--alignment' option.
    do_mccc:
        Align by multi-channel cross-correlation. Activated by '--mccc'
    do_pca:
        Align by principal component analysis. Activated by '--pca'
    overwrite:
        Overwrite existing aligned waveform files. Activated by '--overwrite'
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


def exclude_entry(
    config: core.Config,
    iteration: int = 0,
    overwrite: bool = False,
    directory: Path = Path("."),
    do_nodata: bool = False,
    do_snr: bool = False,
    do_cc: bool = False,
    do_ecn: bool = False,
) -> None:
    """Exclude phase observations based on waveform quality criteria.

    Thresholds are read from the resective waveform header files and can be
    different for each phase. Default values from a `default-hdr.yaml` are
    respected. `None` thresholds are ignored.

    This function is called when executing 'relmt exclude' from the command line.

    Parameters
    ----------
    config:
        Configuration object with station file. Content of the file read
        by the '--config' option.
    iteration:
        Current alignment iteration number. `0`: read from 'data/', `>0` read from
        alignment iteration folder. Number parsed to the '--alignment' option.
    overwrite:
        Overwrite existing entry exclude file. Activated by '--overwrite'. If
        False, append to existing lists instead.
    directory:
        Root directory of the project, containing the 'data/' and 'align?/'
        subfolders. Path of the file referenced by the '--config' option.
    do_nodata:
        Exclude traces with `NaN` data, null data or data with all values below
        absolute 'null_threshold'. Activated by '--nodata'
    do_snr:
        Exclude traces with signal-to-noise ratio below 'min_signal_noise_ratio'.
        Activated by '--snr'
    do_cc:
        Exclude traces with average cross-correlation below 'min_correlation'.
        Activated by '--cc'
    do_ecn:
        Exclude traces with expansion coefficient norm below
        'min_expansion_coefficient_norm'. Activated by '--ecn'
    """

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
            # TODO: allow to read cc from file instead
            cc = signal.correlation_averages(mat, hdr["phase"])[2]
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
        logger.info(f"Excluding {len(excludes['snr'])} traces due to high SNR")
        excl["phase_auto_snr"] = excludes["snr"]

    if do_cc:
        logger.info(f"Excluding {len(excludes['cc'])} traces due to low CC")
        excl["phase_auto_cc"] = excludes["cc"]

    if do_ecn:
        logger.info(f"Excluding {len(excludes['ecn'])} traces due to low ECN")
        excl["phase_auto_ecn"] = excludes["ecn"]

    # Save it to file
    io.save_yaml(exf, excl)


def amplitude_entry(
    config: core.Config,
    directory: Path = Path(),
    iteration: int = 0,
    overwrite: bool = False,
) -> None:
    """Compute relative amplitude measurements and save to file.

    This function is called when executing 'relmt amplitude' from the command line.

    Parameters
    ----------
    config:
        Configuration object with amplitude measurement parameters. Content of
        the file read by the '--config' option.
    directory:
        Root directory of the project, containing the 'align?/' and 'amplitude/'
        subfolders. Path of the file referenced by the '--config' option.
    iteration:
        Read waveforms from this alignment iteration. `0`: read from 'data/',
        `>0` read from 'align?'. Number parsed to the '--alignment' option.
    overwrite:
        Overwrite existing amplitude measurement and passband files. Activated
        by '--overwrite'
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

                pasbnds[wvid] = signal.phase_passbands(
                    arr,
                    hdr,
                    event_dict,
                    **config.kwargs(signal.phase_passbands),
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


def qc_entry(config: core.Config, directory: Path = Path()) -> None:
    """Quality control amplitude measurements

    This function is called when executing 'relmt qc' from the command line.

    Parameters
    ----------
    config:
        Configuration object with QC parameters. Content of the file read
        by the '--config' option.
    directory:
        Root directory of the project, containing the 'amplitude/' subfolder.
        Path of the file referenced by the '--config' option.
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
    eq_batches = config["equation_batches"]
    keep_ev = config["keep_events"]

    exclude = io.read_yaml(core.file("exclude", directory=directory))

    exclude_wvid = exclude["waveform"]

    ampsuf = config["amplitude_suffix"]
    qcsuf = config["qc_suffix"]
    outsuf = f"{ampsuf}-{qcsuf}"

    exclude_phase = set(exclude["phase_manual"]).union(
        exclude["phase_auto_nodata"]
        + exclude["phase_auto_snr"]
        + exclude["phase_auto_ecn"]
    )

    exclude_events = set(exclude["event"])

    evd = io.read_event_table(directory / config["event_file"])
    std = io.read_station_table(directory / config["station_file"])
    phd = io.read_phase_table(directory / config["phase_file"])

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
            samps = amps  # No need to copy as we leave the "PS"-loop either way

    # Make sure we have enough equations
    pamps, samps = qc.clean_by_equation_count_gap(
        pamps, samps, phd, min_eq, max_gap, keep_other_s_equation
    )

    # Make sure we don't have too many equations
    s_equations = len(samps) * (1 + int(keep_other_s_equation))
    if max_s_equations is not None:
        excess_eq = s_equations - max_s_equations
        logger.info(f"Have {s_equations} S-equations, need to reduce by {excess_eq}")

        if eq_batches is None:
            eq_batches = 0
            logger.debug(
                "'equation_batches' not set. Reducing S equations in one batch."
            )

        if max_s1 is None:
            max_s1 = 1.0

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

            # Redundancy score per observation (highest score least important)
            red_score = utils.pair_redundancy(triplets, ignore=keep_ev)

            # Gap score per observation (lowest score least important)
            # Alternative score based on station azimutal gap
            # std_sub = {stn: std[stn] for stn in set(stas)}
            # sta_gap = utils.station_gap(std_sub, evd)
            # gap_score = np.array([sta_gap[sta] for sta in stas])

            sta_count = utils.item_count(stas)

            # Combine misfit and sigma meassure
            # Prefer disinct S-wave combinations with low misfit
            # Lower is better
            mis_sigma = (1 - (max_s1 - s1s)) * (1 - (max_mis - miss))

            # Adapt number of exclusion in case some events dropped out
            nex = int(excess_eq // eq_batches + 1)

            # score sort: from most important to least important
            ssort = np.lexsort((mis_sigma, sta_count, red_score))
            samps = [samps[i] for i in ssort[: -nex + 1]]

            # Check once more we have enough equations
            pamps, samps = qc.clean_by_equation_count_gap(
                pamps, samps, phd, min_eq, max_gap, keep_other_s_equation
            )

            s_equations = len(samps) * (1 + int(keep_other_s_equation))

            # In case some events dropped out, how many equations are left?
            excess_eq = s_equations - max_s_equations
            logger.debug(f"Reduced to {s_equations} S-equations. {excess_eq} left.")
            eq_batches -= 1

    if len(pamps) + len(samps) == 0:
        raise RuntimeError("No observations left. Relax your QC criteria.")

    # Write to file
    for ph, amps in zip("PS", [pamps, samps]):
        outfile = core.file(
            "amplitude_observation",
            directory=directory,
            phase=ph,
            suffix=outsuf,
        )
        io.save_amplitudes(outfile, amps)


def solve_entry(
    config: core.Config, directory: Path = Path(), do_predict=False, iteration: int = 0
) -> None:
    """Construct linear system ans solve for moment tensors.

    This function is called when executing 'relmt solve' from the command line.

    Parameters
    ----------
    config:
        Configuration object with solver parameters. Content of the file read
        by the '--config' option.
    directory:
        Root directory of the project, containing the 'amplitude/' and 'result/'
        subfolders. Path of the file referenced by the '--config' option.
    do_predict:
        If `True`, additionaly predict amplitudes of the found solution. This
        option is slow.
    iteration:
        Current alignment iteration number. `0`: read from 'data/', `>0` read from
        alignment iteration folder. Number parsed to the '--alignment' option.
        Only required when do_predict is `True`
    """

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

    ampsuf = config["amplitude_suffix"]
    qcsuf = config["qc_suffix"]
    resuf = config["result_suffix"]
    insuf = f"{ampsuf}-{qcsuf}"
    outsuf = f"{insuf}-{resuf}"
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
    logger.info(
        f"Building linear system of {n_p} P, {n_s} S, and {n_ref} reference equations."
    )
    isparse = True
    if isparse:
        # as sparse array
        Ah, bh = ls.homogenous_equations_sparse(
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
        Ah, bh = ls.homogenous_equations(
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

    logger.info("Computing norms and weights...")
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
    eq_norm = ls.condition_by_norm(Ah)
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
    io.write_mt_table(
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
                p_norm.flat[:],
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

    # Bootstrap
    if nboot is not None and nboot > 0:
        logger.info(f"Computing {nboot} bootstrap samples...")
        m_boots = ls.bootstrap_lsmr(A, b, ev_scale, n_p, n_s, nboot, 0, 1)

        # Convert to MT dict
        # bootmts = {i: [] for i in range(m_boots.shape[1] // mt_elements)}
        bootmts = {i: [] for i in incl_ev}
        for m_boot in m_boots:
            for i, momt in enumerate(mt.mt_tuples(m_boot, constraint)):
                bootmts[incl_ev[i]].append(momt)

        # Make and save a
        io.write_mt_table(
            bootmts,
            core.file("relative_mt", suffix=outsuf + "-boot", directory=directory),
        )

        boot_rms = mt.norm_rms(bootmts, relmts)
        boot_kag = mt.kagan_rms(bootmts, relmts)
    else:
        boot_rms = {}
        boot_kag = {}

    sumf = core.file("mt_summary", suffix=outsuf, directory=directory)
    logger.info(f"Saving full summary to: {sumf}")
    io.save_mt_result_summary(
        sumf,
        evd,
        relmts,
        gaps,
        links,
        avmiss,
        avccs,
        mom_rmss,
        amp_rmss,
        boot_rms,
        boot_kag,
    )


def plot_alignment_entry(
    arrf: Path,
    config: core.Config | None = None,
    do_exclude: bool = False,
    sort: str = "pci",
    highligh_events: list[int] = [],
) -> None:
    """Plot the waveform array and parameters relevant to judging the alignment

    Parameters
    ----------
    arrf:
        Path to the waveform array file to plot
    config:
        Configuration object. If given, event and station tables are read from
        files specified in the configuration.
    do_exclude:
        Read the exclude file and only plot events that are not excluded.
    sort:
        Sorting method for events. See `plot.alignment` for options.
    highligh_events:
        List of event IDs to highlight in the plot.
    """

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

    # TODO: allow ccij to be computed instead
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


def make_parser() -> ArgumentParser:
    """Create the ArgumentParser for the relMT command line interface."""

    # Common options for various modes
    option = {
        "overwrite": (
            ("-o", "--overwrite"),
            dict(help="Overwrite existing files", action="store_true"),
        ),
        "config": (
            ("-c", "--config"),
            dict(
                type=Path,
                help="Use this configuration file",
                default=core.file("config"),
            ),
        ),
        "alignment": (
            ("-a", "--alignment"),
            dict(type=int, nargs="?", help="Alignment iteration", default=0),
        ),
        "highlight": (
            ("--highlight",),
            dict(
                type=int,
                nargs="+",
                help="Event IDs to highligh in the plot",
                default=[],
            ),
        ),
        "exclude": (
            ("--exclude",),
            dict(
                action="store_true",
                help="Exclude events listed in the exclude file",
            ),
        ),
    }

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
        "exclude",
        help=(
            "Exclude phase observations from alignment based on criteria in "
            "header files"
        ),
    )

    amp_p = subpars.add_parser(
        "amplitude", help="Measure relative amplitudes on aligned waveforms"
    )

    qc_p = subpars.add_parser(
        "qc",
        help=(
            "Apply quality control parameters from configuration file to "
            "amplitude measurements"
        ),
    )

    solve_p = subpars.add_parser(
        "solve", help="Compute moment tensors from amplitude measurements"
    )

    # Now set the functions to be called
    init_p.set_defaults(command=core.init)
    align_p.set_defaults(command=align_entry)
    exclude_p.set_defaults(command=exclude_entry)
    amp_p.set_defaults(command=amplitude_entry)
    qc_p.set_defaults(command=qc_entry)
    solve_p.set_defaults(command=solve_entry)

    # Subparser arguments
    init_p.add_argument(
        "directory",
        type=Path,
        default=".",
        nargs="?",
        help="Name of the directory to initiate",
    )

    # Sub arguments of the alignment routine
    align_p.add_argument(*option["config"][0], **option["config"][1])
    align_p.add_argument(*option["alignment"][0], **option["alignment"][1])
    align_p.add_argument(*option["overwrite"][0], **option["overwrite"][1])

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
        "--nodata",
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

    exclude_p.add_argument(
        "--overwrite",
        "-o",
        help=(
            "Overwrite existing entries per category. Never overwrites manually"
            "exluded phases"
        ),
        action="store_true",
    )

    exclude_p.add_argument(*option["alignment"][0], **option["alignment"][1])
    exclude_p.add_argument(*option["config"][0], **option["config"][1])

    # Amplitude sub-arguments
    amp_p.add_argument(*option["config"][0], **option["config"][1])
    amp_p.add_argument(*option["alignment"][0], **option["alignment"][1])
    amp_p.add_argument(*option["overwrite"][0], **option["overwrite"][1])

    # QC sub-arguments
    qc_p.add_argument(*option["config"][0], **option["config"][1])

    # Sub arguments of the solve routine
    solve_p.add_argument(*option["config"][0], **option["config"][1])
    solve_p.add_argument(*option["alignment"][0], **option["alignment"][1])
    solve_p.add_argument(*option["overwrite"][0], **option["overwrite"][1])
    solve_p.add_argument(
        "--predict",
        action="store_true",
        help=(
            "Predict relative amplitudes of the solution and compute "
            "prediction misfits"
        ),
    )

    # Plot alignment
    plot_align_p = subpars.add_parser(
        "plot-alignment", help="Plot waveform alignment resuts to screen"
    )
    plot_align_p.set_defaults(command=plot_alignment_entry)

    plot_align_p.add_argument(*option["config"][0], **option["config"][1])

    plot_align_p.add_argument(
        "file",
        type=Path,
        help="Path to -wvarr.npy file",
    )

    plot_align_p.add_argument(
        "--sort",
        type=str,
        help="The sorting to apply: 'pci' (default), 'magnitude', 'none'",
        choices=["pci", "magnitude", "none"],
        default="pci",
    )

    plot_align_p.add_argument(*option["highlight"][0], **option["highlight"][1])
    plot_align_p.add_argument(*option["exclude"][0], **option["exclude"][1])


    return parser


def get_arguments(args: list[str] | None = None) -> Namespace:
    """Collect the command line arguments.

    Parameters
    ----------
    args:
        List of command line arguments. If ``None``, collect via ArgumentParser.

    Returns
    -------
    Parsed arguments
    """

    parser = make_parser()

    parsed = parser.parse_args(args)

    if parsed.mode is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parsed


def main(args=None):
    """Entry point for the relMT command line interface

    Parameters
    ----------
    args:
        Optional list of command line arguments. If ``None``, use
        :data:`sys.argv`.

    Returns
    -------
    None
        Executes the selected subcommand and exits
    """
    # Subdirectory, e.g. A_Muji
    parsed = get_arguments(args)

    if parsed.mode == "init":
        parent = parsed.directory
        parsed.command(parent)
        return

    conff = parsed.config
    config = io.read_config(conff)

    # Let's parse the keyword arguments explicitly
    parent = conff.parent

    if not parsed.mode.startswith("plot-"):
        n_align = parsed.alignment
        overwrite = parsed.overwrite

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
                do_nodata=parsed.nodata,
                do_snr=parsed.snr,
                do_cc=parsed.cc,
                do_ecn=parsed.ecn,
            )
        )

    if parsed.mode == "plot-alignment":
        plot_alignment_entry(
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
