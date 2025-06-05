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
import yaml
import numpy as np
import multiprocessing as mp
from argparse import ArgumentParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def get_arguments():
    """Get command line options for :func:`main_align()`"""

    parser = ArgumentParser(
        usage="%(prog)s [arguments] <station database>",
        description=(
            "Align waveforms using multi channel cross correlation"
            "and pricinipal component analyses."
        ),
    )

    parser.add_argument(
        "-c", "--config", type=str, help="Use this configuration file", default=""
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

    return parser.parse_args()


def main_align():
    """
    Align waveform files and write results into next alignment directory
    """

    args = get_arguments()

    iteration = args.n_align
    overwrite = args.overwrite
    conf = args.config

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
            except FileNotFoundError:
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

        if hdr["maxshift"] is None:
            hdr["maxshift"] = hdr["phase_end"] - hdr["phase_start"]

        args.append((arr, hdr, dest))

    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            pool.starmap(align.run, args)
    else:
        for arg in args:
            align.run(*arg)


def main_exclude():
    args = get_arguments()

    iteration = args.n_align
    overwrite = args.overwrite
    donodata = args.no_data
    dosnr = args.snr
    doecn = args.ecn

    conf = args.config

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
        if conf["min_signal_noise_ratio"] is not None:
            isnr = snr < conf["min_signal_noise_ratio"]

        iecn = np.full_like(ind, False)
        if conf["min_expansion_coefficient_norm"] is not None:
            iecn = ec_score < conf["min_expansion_coefficient_norm"]

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


def main_bandpass():

    args = get_arguments()

    iteration = args.n_align
    overwrite = args.overwrite
    conf = args.config

    evf = conf["event_file"]
    stf = conf["station_file"]

    evl = io.read_event_table(evf)
    stad = io.read_station_table(stf)

    excl = io.read_exclude_file(core.file("exclude"))
    bpf = core.file("bandpass")

    if bpf.exists() and not overwrite:
        msg = f"{bpf} exists. Use --overwrite option to overwrite. Exiting."
        logger.critical(msg)
        return

    stas = set(stad) - set(excl["station"])

    stressdrop_range = conf["stressdrop_range"]

    # Exclude excluded waveforms
    wvids = set(core.iterate_waveid(stas)) - set(excl["waveform"])

    bpd = {}
    for wvid in wvids:

        print(f"Working on: {wvid}")

        sta, pha = core.split_waveid(wvid)

        try:
            arr, hdr = io.read_waveform_array_header(sta, pha, iteration)
        except FileNotFoundError:
            continue

        bpd[wvid] = {}

        ievs, evns = qc.included_events(excl, **hdr.kwargs(qc.included_events))

        sr = hdr["sampling_rate"]

        # At least one period within window
        fmin = 1 / (hdr["phase_end"] - hdr["phase_start"])

        # Signal and noise indices
        isig, _ = signal.indices_signal(**hdr.kwargs(signal.indices_signal))

        for iev, evn in zip(ievs, evns):
            print("{:02d} events to go   ".format(len(evns) - iev), end="\r")

            ev = evl[evn]

            # Corner frequency from stress drop
            fcmin = utils.corner_frequency(ev.mag, pha, stressdrop_range[0], 4000)
            fcmax = utils.corner_frequency(ev.mag, pha, stressdrop_range[1], 4000)

            sig = signal.demean(arr[iev, :, isig:])

            ohpas, olpas = extra.optimal_bandpass(
                arr[iev, :, :],
                fmin=fmin,
                fmax=fcmax,
                min_snr=1.5,
                **hdr.kwargs(extra.optimal_bandpass),
            )

            # Don't look below minimum frequency
            try:
                fc = extra.apparent_corner_frequency(sig, sr, fmin=fcmin, fmax=fcmax)
            except ValueError:
                fc = fcmax

            # Filter below corner frequency or optimal lowpass, whichever is
            # lower
            lpas = min((olpas, fc))
            hpas = ohpas

            # Save to dictionary
            bpd[wvid][evn] = [float(hpas), float(lpas)]

    io.save_yaml(bpf, bpd)


def main_amplitude():
    # Subdirectory, e.g. A_Muji
    args = get_arguments()

    iteration = args.n_align
    conf = args.config

    stf = conf["station_file"]
    evf = conf["event_file"]

    ls.logger.setLevel("ERROR")
    signal.logger.setLevel("ERROR")
    align.logger.setLevel("WARNING")

    excl = io.read_exclude_file(core.file("exclude"))

    ncpu = conf["ncpu"]
    min_dynamic_range = conf["min_dynamic_range"]

    stas = io.read_station_table(stf)

    # Exclude some observations
    stas = set(stas) - set(excl["station"])
    wvids = set(core.iterate_waveid(stas)) - set(excl["waveform"])

    evl = io.read_event_table(evf)
    nev = len(evl)

    bpf = core.file("bandpass")
    with open(bpf, "r") as fid:
        pasbnds = yaml.safe_load(fid)  # Pass bands

    pargs = []
    sargs = []
    for wvid in wvids:
        sta, pha = core.split_waveid(wvid)

        try:
            arr, hdr = io.read_waveform_array_header(sta, pha, iteration)
        except FileNotFoundError:
            continue

        _, evns = qc.included_events(excl, **hdr.kwargs(qc.included_events))

        if pha == "P":
            pargs += [
                (arr, hdr, pasbnds[wvid], min_dynamic_range, *iabs)
                for iabs in core.iterate_event_pair(nev, evns)
            ]

        if pha == "S":
            sargs += [
                (arr, hdr, pasbnds[wvid], min_dynamic_range, *iabcs)
                for iabcs in core.iterate_event_triplet(nev, evns)
            ]

    # First process and save P ...
    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            abA = pool.starmap(amp.process_p, pargs)
    else:
        abA = [amp.process_p(*arg) for arg in pargs]

    abA = [tup for tup in abA if tup is not None]
    io.save_amplitudes(core.file("amplitude_observation", sta, "P"), abA)

    # ... later S
    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            abcB = pool.starmap(amp.process_s, sargs)
    else:
        abcB = [amp.process_s(*arg) for arg in sargs]

    abcB = [tup for tup in abcB if tup is not None]
    io.save_amplitudes(core.file("amplitude_observation", sta, "S"), abcB)


def main_linear_system():

    ls.logger.setLevel("WARNING")

    args = get_arguments()
    suffix = "-" + args.suffix
    conf = args.config

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


def main_solve():

    args = get_arguments()
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
