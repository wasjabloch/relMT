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

import numpy as np
from scipy.linalg import svd
from scipy.interpolate import interpn
from scipy.special import comb
from datetime import datetime
from typing import Iterable
from relmt import core, mt, signal, qc, angle
from collections import defaultdict
from itertools import combinations

logger = core.register_logger(__name__)


def xyzarray(
    table: dict[str, core.Station] | dict[int, core.Event] | core.Event | core.Station,
) -> np.ndarray:
    """Spatial coordinates as an array

    Parameters
    ----------
    table:
        Event, event dictionary, station, or station dictionary

    Returns
    -------
    ``(rows, 3)`` or ``(3,)`` north-east-down coordinates"""
    try:
        # Is it a dictionary?
        return np.array([table[key][:3] for key in table])
    except TypeError:
        try:
            # ... or a list?
            return np.array([item[:3] for item in table])
        except (IndexError, TypeError):
            # ... or just an item?
            return np.array(table[:3])


def cartesian_distance(
    x1: float | np.ndarray,
    y1: float | np.ndarray,
    z1: float | np.ndarray,
    x2: float | np.ndarray,
    y2: float | np.ndarray,
    z2: float | np.ndarray,
) -> float | np.ndarray:
    """Cartesian distance between points (x1, y1, z1) and (x2, y2, z2)"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def approx_time_lookup(
    time1: list[str] | list[float],
    time2: list[str] | list[float],
    include_at: float = np.inf,
) -> dict:
    """
    Create lookup table for `time1` approximately matching `time2`

    All events in `time1s` must be present in `time2`,
    i.e. ``len(time1) <= len(time2)``.

    Parameters
    ----------
    time1, time2:
        Times either as `date_string` understood by
        :meth:`datetime.datetime.fromisoformat`, or as `float` in arbitrary
        units
    include_at:
        Maximum time difference (seconds) to consider a match.


    Returns
    -------
    Lookup table time1 -> time2

    Raises
    ------
    LookupError:
        When `time1` is longer than `time2`
    """

    logger.debug("Creating approximate time lookup table")

    if len(time1) > len(time2):
        raise LookupError("times1 must be shorter or equal than times2.")

    time1 = sorted(time1)
    time2 = sorted(time2)

    try:
        t1s = list(map(datetime.fromisoformat, time1))
        t2s = list(map(datetime.fromisoformat, time2))
        istime = True
    except (ValueError, TypeError):
        t1s = list(map(float, time1))
        t2s = list(map(float, time2))
        istime = False

    logger.debug(f"Interpreting input as: {type(t1s[0])}")

    lut = {}
    n0 = 0
    for ot1, t1 in zip(time1, t1s):

        dtmin = np.inf
        for n, (ot2, t2) in enumerate(zip(time2[n0:], t2s[n0:])):
            dt = abs(t1 - t2)
            if istime:
                dt = abs(t1 - t2).total_seconds()

            if dt < dtmin:
                dtmin = dt
                thisot2 = ot2
            else:
                break

        if dtmin <= include_at:
            lut[ot1] = thisot2
        else:
            msg = (
                "Time difference between events is: {:.0f} seconds\n"
                "Event 1: {:}\n"
                "Event 2: {:}\n"
                "Did not include event to lookup table!"
            ).format(dtmin, ot1, ot2)
            logger.warning(msg)
        n0 = n0 + n

    return lut


def reshape_ccvec(ccvec: np.ndarray, ns: int, combinations=None) -> np.ndarray:
    """Reshape 1D vector of cc coefficeints c_ijk to ``(n, n, n)`` 3D matrix"""
    iterator = core.ijk_ccvec(ns)

    if combinations is not None:

        def iterate_combinations(combinations):
            for i, j, k in combinations:
                yield (i, j, k)
                yield (k, i, j)
                yield (j, k, i)

        iterator = iterate_combinations(combinations)

    cc = np.full((ns, ns, ns), 0.0)
    for n, (i, j, k) in enumerate(iterator):
        val = max(min(ccvec[n], 1.0), -1.0)
        cc[i, j, k] = val
        cc[i, k, j] = val
        cc[k, i, j] = val
        cc[k, j, i] = val
        cc[j, k, i] = val
        cc[j, i, k] = val

    return cc


def event_indices(
    amps: list[core.P_Amplitude_Ratio] | list[core.S_Amplitude_Ratios],
) -> dict[int, np.ndarray]:
    """Index arrays of unique event numbers

    For each event, find the constributing pair or triplet wise amplitude observations

    Parameters
    ----------
    amps:
        List of amplitude observations, either P or S

    Returns
    -------
    Mapping from event number to observation indices
    """

    ip, _ = qc._ps_amplitudes(amps)

    if ip:
        eva, evb = np.array([(amp.event_a, amp.event_b) for amp in amps]).T
        evc = eva
    else:
        eva, evb, evc = np.array(
            [(amp.event_a, amp.event_b, amp.event_c) for amp in amps]
        ).T

    evs = np.union1d(np.union1d(eva, evb), evc)
    indices = defaultdict(
        list,
        {ev: ((eva == ev) | (evb == ev) | (evc == ev)).nonzero()[0] for ev in evs},
    )
    return indices


def signed_log(x):
    """Signed log transform: sign(x) log(1 + | x |)"""
    return np.sign(x) * np.log1p(np.abs(x))


def signed_log_inverse(y):
    """Inverse of the signed log transform: sign(x) log(1 + | x |)"""
    return np.sign(y) * (np.expm1(np.abs(y)))


def fisher_average(ccarr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average cross correlation coefficients applying Fisher z-transform

    .. note:
        We assume one `0` value as a place-holder for the auto-correlation. The
        `0` does not contribute to the sum. We divide by `n-1` so the
        auto-correaltion neither has a weight.

    Parameters
    ----------
    ccarr:
        1, 2, or 3 dimensional array holding values bound in interval `[-1, 1]`
    axis:
        Array axis along which to operate

    Returns
    -------
    Averaged array, reduced by one dimension
    """

    for shape in ccarr.shape:
        assert shape == ccarr.shape[0]

    dims = len(ccarr.shape)
    nel = ccarr.shape[0] - (dims - 1)  # Don't count diagonal elements

    if nel < 1:
        raise ValueError("Input array is too small")

    with np.errstate(divide="ignore"):  # It is OK if ccmat == 1 or -1

        # Apply Fisher z-transform
        ccarr = np.arctanh(ccarr)

        # Average
        ccarr = np.sum(ccarr, axis=axis)

        ccarr /= nel

        # Untransform
        ccarr = np.tanh(ccarr)

    return ccarr


def phase_dict_azimuth(
    phase_dict: dict[str, core.Phase],
    event_dict: dict[int, core.Event],
    station_dict: dict[str, core.Station],
    overwrite: bool = False,
) -> dict[str, core.Phase]:
    """
    Fill phase dictionary with azimuth from trigonometry

    Parameters
    ----------
    phase_dict:
        All seismic phases
    event_dict:
        The seismic event catalog
    station_dict:
        Station table
    overwrite:
        Overwrite existing azimuth values (`False`: only replace `NaN` values)

    Returns
    -------
    New phase dictionary containing computed plunges
    """

    azis = np.array(
        [
            (
                angle.azimuth(
                    *xyzarray(event_dict[core.split_phaseid(phid)[0]])[:2],
                    *xyzarray(station_dict[core.split_phaseid(phid)[1]])[:2],
                ),
            )
            for phid in phase_dict
        ]
    )

    new_phase_dict = {
        phid: (
            # Assign original plunge when not nan and we are not overwriting
            core.Phase(ph.time, ph.azimuth, ph.plunge)
            if np.isfinite(ph.azimuth) and not overwrite
            else core.Phase(ph.time, azis[nph][0], ph.plunge)
        )
        for nph, (phid, ph) in enumerate(phase_dict.items())
    }

    return new_phase_dict


def phase_dict_hash_plunge(
    phase_dict: dict[str, core.Phase],
    event_dict: dict[int, core.Event],
    station_dict: dict[str, core.Station],
    vmodel: np.ndarray,
    overwrite: bool = False,
    nquerry: int = 100,
    nray=1000,
) -> dict[str, core.Phase]:
    """
    Fill phase dictionary with plunge values from HASH.

    Uses the python implementation of the HASH ray-tracer (Skoumal, Hardebeck &
    Shearer, 2024; Hardebeck & Shearer, 2002) by courtesey of USGS.

    Parameters
    ----------
    phase_dict:
        All seismic phases
    event_dict:
        The seismic event catalog
    station_dict:
        Station table
    vmodel:
        ``(layers, 3)`` array of depth (m), P- and S-wave velocities (m/s)
    overwrite:
        Overwrite existing plunge values (`False`: only replace `NaN` values)
    nquerry:
        Number of distance and depth querry points for plunge lookup table
    nray:
        Number of trail rays (should be larger than `nquerry`)

    Returns
    -------
    New phase dictionary containing computed plunges
    """

    dist_dep = np.array(
        [
            (
                cartesian_distance(
                    *xyzarray(station_dict[core.split_phaseid(phid)[1]])[:2],
                    0,
                    *xyzarray(event_dict[core.split_phaseid(phid)[0]])[:2],
                    0,
                ),
                event_dict[core.split_phaseid(phid)[0]].depth,
            )
            for phid in phase_dict
        ]
    )

    # Distance and depth coordinate vectors
    # Distance is expected to begin at 0, else strange IndexErrors occurr
    maxdist = max(dist_dep[:, 0])
    maxdep = max(dist_dep[:, 1])
    logger.debug(f"Found maximum event-station distance: {maxdist} m")
    logger.debug(f"Found maximum event depth: {maxdep} m")

    distv = np.linspace(0.0, maxdist, nquerry)
    depv = np.linspace(0.0, maxdep, nquerry)

    # P wave ...
    pluta_p = angle.hash_plunge_table(vmodel[:, :2], depv, distv, nray)

    # S wave plunge lookup table
    pluta_s = angle.hash_plunge_table(vmodel[:, [0, 2]], depv, distv, nray)

    # Interpolate P and S plunges, regardless of phase
    plup = interpn((distv, depv), pluta_p, dist_dep)
    plus = interpn((distv, depv), pluta_s, dist_dep)

    new_phase_dict = {
        phid: (
            # Assign original plunge when not nan and we are not overwriting
            core.Phase(ph.time, ph.azimuth, ph.plunge)
            if np.isfinite(ph.plunge) and not overwrite
            # Else, Assign P or S plunge, depending on phase id
            else (
                core.Phase(ph.time, ph.azimuth, plup[nph])
                if core.split_phaseid(phid)[2] == "P"
                else core.Phase(ph.time, ph.azimuth, plus[nph])
            )
        )
        for nph, (phid, ph) in enumerate(phase_dict.items())
    }

    return new_phase_dict


def interpolate_phase_dict(
    phase_dict: dict[str, core.Phase],
    event_dict: dict[int, core.Event],
    station_dict: dict[str, core.Station],
    obs_min: int = 1,
) -> dict[str, core.Phase]:
    """
    Interpolate phase arrivals, ray azimuth and plunge

    Arrival time is interpolated via an average velocity along the ray path.

    Ray plunge is interpolated linearly between neighboring points in depth

    Azimuth is calculated assuming a horizontally straight ray.

    Parameters
    ----------
    phase_dict:
        All seismic phases
    event_dict:
        The seismic event catalog
    station_dict:
        Station table
    obs_min:
        Minimum number of observations to make interpolation

    Returns
    -------
    New phase dictionary containing the interpolated phases
    """

    def _fix_non_monotonic(inc, dep):
        # Fix non monotonically increasing depth to interpolate ray plunge
        # If depth are not monotonically increasing
        isort = np.lexsort((inc, dep))  # sort by depth
        izero = np.nonzero(np.diff(dep[isort]) > 0)[0]
        while any(izero):
            # Average Ray plunge
            inc = np.concatenate(
                (
                    inc[: izero[0]],
                    [np.sum(inc[izero[0] : izero[0] + 2]) / 2],
                    inc[izero[0] + 2 :],
                )
            )
            # Drop depth
            dep = np.concatenate(
                (
                    dep[: izero[0]],
                    dep[izero[0] + 1 :],
                )
            )
            isort = np.lexsort((inc, dep))  # sort by depth
            izero = np.nonzero(np.diff(dep[isort]) > 0)[0]

        return inc, dep, isort

    def interpolate_plunge(depq, deps, incs):
        # Interpolate incidence at querry depth depq, given measurements
        # of inclindation angles inc and depths deps
        # Take-off plunge vs. depth of non-nan plunge
        isort = np.lexsort((incs, deps))  # sort by depth
        incs, deps, isort = _fix_non_monotonic(incs, deps)
        if any(incs):
            return np.interp(depq, deps[isort], incs[isort])
        return np.nan

    _, stas, phs = zip(*map(core.split_phaseid, phase_dict.keys()))

    # new phase dictionary
    nphd = {}
    for ph in set(phs):
        for st in set(stas):
            logger.debug(f"Working on phase {ph}, station {st}")
            try:
                sxyz = station_dict[st][:3]
            except KeyError:
                logger.warning(f"Station {st} not in station_dict. Skipping.")
                continue

            # For all events with phase readings on station, get:
            # Travel times, distances, azimuths, plunge and depth
            tdaiz = np.array(
                [
                    (
                        phase_dict[pid][0]
                        - event_dict[event_index][3],  # arrival - origin time
                        cartesian_distance(
                            *sxyz, *event_dict[event_index][:3]
                        ),  # distance
                        phase_dict[pid].azimuth,
                        phase_dict[pid].plunge,
                        event_dict[event_index][2],  # event z-coordinate
                    )
                    for pid in phase_dict
                    for event_index in event_dict
                    if event_index == core.split_phaseid(pid)[0]  # event match
                    and st == core.split_phaseid(pid)[1]  # station match
                    and ph == core.split_phaseid(pid)[2]  # phase match
                ],
                ndmin=1,
            )

            n = tdaiz.shape[0]
            if n < obs_min:
                # Too few readings
                logger.warning(
                    f"Station {st} has only {n} {ph}-wave readings. obs_min is {obs_min}. Skipping"
                )
                continue

            tt = tdaiz[:, 0]
            d = tdaiz[:, 1]

            # Average path velocity
            v = np.average(d / tt)
            logger.info(
                f"{ph}-wave velocity along raypath to station {st} is "
                + "{:.0f} +/- {:.0f} m/s".format(v, np.std(d / tt))
            )

            pluns, deps = tdaiz[np.isfinite(tdaiz[:, 3]), 3:5].T

            # Now look for missing events
            for event_index in event_dict:
                thisphid = core.join_phaseid(event_index, st, ph)
                ex, ey, ez, t0 = event_dict[event_index][:4]

                # Interpolate time, azimuth, incidence
                td = cartesian_distance(ex, ey, ez, *sxyz)  # distance
                et = t0 + td / v  # arrival time
                ea = angle.azimuth(
                    ex, ey, sxyz[0], sxyz[1]
                )  # event to station azimuth should be correct

                ep = interpolate_plunge(ez, deps, pluns)  # ray incidence
                try:
                    # Try to get meassured values
                    mt, ma, mi = phase_dict[thisphid]
                    if any(~np.isfinite([mt, ma, mi])):
                        # Substitute any NaNs
                        et = np.where(np.isfinite(mt), mt, et)
                        ea = np.where(np.isfinite(ma), ma, ea)
                        ep = np.where(np.isfinite(mi), mi, ep)
                        nphd[thisphid] = core.Phase(et, ea, ep)
                except KeyError:
                    nphd[thisphid] = core.Phase(et, ea, ep)

    return nphd


def next_fftw_size(samples: int, devisor: int = 3, odd: bool = True) -> int:
    """
    The next integer for which FFTW is optimized

    Parameters
    ----------
    samlpes:
        Number of target samples
    devisor:
        The result must be devisable by this number
    odd:
        The result divided by devisor must be odd
    """
    nfft = np.asarray(_fftw_sizes(5, 5, 5, 5))
    inext = nfft > samples
    idev = nfft % devisor == 0
    iodd = np.full_like(nfft, True)
    if odd:
        iodd = nfft / devisor % 2 == 1
    return nfft[inext & idev & iodd][0]


# @core._doc_config_args
def fftw_data_window(
    minimum_time: float, sampling_rate: float, components: str
) -> float:
    """
    Set 'data_window' based on sampling rate and number of components

    Parameters
    ----------
    minimum_time:
        Minimum length of single-channel time window
    sampling_rate:
        Sampling rate of the seismic waveform (Hertz)
    components:
        One-character component names ordered as in the waveform array, as one
        string (e.g. 'ZNE')

    Returns
    -------
    Length of single-channel time window. The concated components will be fftw optimized
    """

    ncha = len(components)
    nfftw = next_fftw_size(
        int(minimum_time * sampling_rate * ncha), devisor=ncha, odd=True
    )
    time_window = nfftw / ncha / sampling_rate
    return time_window


def _fftw_sizes(na: int, nb: int, nc: int, nd: int) -> list[int]:
    """
    Return a sorted list of sample sized for which FFTW is optimized

    This is an integer of the form:

    2**a * 3**b * 5**c * 7**d * 11**e * 13**f

    a, b, c, and d are arbitrary  and e+f is either 0 or 1.

    Parameters
    ----------
    na, nb, nc, nc, nd: (int)
        Insert values in range(n) in the equation

    Returns
    -------
    ns: list[int]
        Sorted list of integers that obeys the equation
    """
    # TODO: implement this as parameter-less generator:
    # a, b, c, d, e, f = (0, 0, 0, 0, 0, 0)
    # a, b, c, d, e, f = (1, 0, 0, 0, 0, 0)
    # a, b, c, d, e, f = (0, 1, 0, 0, 0, 0)
    # a, b, c, d, e, f = (2, 0, 0, 0, 0, 0)
    # a, b, c, d, e, f = (0, 0, 1, 0, 0, 0)
    # a, b, c, d, e, f = (1, 1, 0, 0, 0, 0)
    # a, b, c, d, e, f = (0, 0, 0, 1, 0, 0)
    # a, b, c, d, e, f = (3, 0, 0, 0, 0, 0)
    # a, b, c, d, e, f = (0, 2, 0, 0, 0, 0)
    # a, b, c, d, e, f = (1, 0, 1, 0, 0, 0)
    # a, b, c, d, e, f = (0, 0, 0, 0, 1, 0)
    # yield 2**a * 3**b * 5**c * 7**d * 11**e * 13**f

    ns = [
        2**a * 3**b * 5**c * 7**d * 11**e * 13**f
        for a in range(na)
        for b in range(nb)
        for c in range(nc)
        for d in range(nd)
        for e in range(2)
        for f in range(2)
        if e + f <= 1
    ]
    return sorted(ns)


def concat_components(arr: np.ndarray) -> np.ndarray:
    """
    Rearange waveform array so that components are concatenated

    Parameters
    ----------
    arr:
        Waveform array of shape ``(events, components, samples)``

    Returns
    -------
    ``(events, components * samples)`` Waveform matrix
    """
    ne, nc, ns = np.shape(arr)  # events, components, samples
    return arr.reshape(ne, ns * nc)  # events, components * samples


def select_events(arr: np.ndarray, select: list[int], events: list[int]) -> np.ndarray:
    """Return waveforms of specific evetn numbers

    arr:
       ``(events, ...)`` or waveform array or matrix
    select:
        List of event numbers to select
    events:
        List of event numbers in the array

    Returns
    -------
    ``(selected_events, ...)`` Waveform array or matrix
    """

    iin = [events.index(i) for i in select]
    return arr[iin, ...]


def valid_combinations(
    events: list[int], pairs: set[tuple[int, int]], phase: str
) -> np.ndarray:
    """Indices to the event combinations that are in pairs

    Parameters
    ----------
    pairs:
        Event numbers (a, b) that may be combined pairwise. We assume sorted
        pairs: a < b
    events:
        List of event numbers
    phase:
        Phase type, either 'P' or 'S'

    Returns
    -------
    Array of shape ``(combinations, 2)`` (if phase='P') or ``(combinations, 3)``
    (if phase='S') with the indices to the valid event combinations
    """

    def _triplets(pairs):
        """Yield the triplets that can be built from pairs"""

        # Inspired by ChatGPT-5

        # Find neigbours
        nbrs = defaultdict(list)
        for u, v in ipairs:
            nbrs[u].append(v)
            nbrs[v].append(u)
        for x in nbrs:
            nbrs[x].sort()

        # For each edge (u, v) with u < v, intersect N(u) and N(v)
        # but only keep neighbors > v to avoid duplicates.
        for u, v in sorted(ipairs):
            nu = nbrs[u]
            nv = nbrs[v]
            i = j = 0

            # Advance i to first entry > v (since we only want w > v)
            while i < len(nu) and nu[i] <= v:
                i += 1

            # Similarly skip <= v on nv:
            while j < len(nv) and nv[j] <= v:
                j += 1

            # Two-pointer intersection (both lists are sorted)
            while i < len(nu) and j < len(nv):
                if nu[i] == nv[j]:
                    w = nu[i]  # v guaranteed by the skips above
                    yield (u, v, w)

                    i += 1
                    j += 1
                elif nu[i] < nv[j]:
                    i += 1
                else:
                    j += 1

    ipairs = set(
        (
            (events.index(a), events.index(b))
            if a < b
            else (events.index(b), events.index(a))
        )
        for a, b in pairs
        if a in events and b in events
    )

    if phase == "P":
        # All event name combinations
        return np.array(sorted(ipairs))

    elif phase == "S":
        logger.debug("Finding valid event triplets")
        return np.array(list(_triplets(ipairs)))

    else:
        raise ValueError("Phase must be 'P' or 'S'")


def pair_redundancy(
    triplets: np.ndarray, ignore: list[int] | None = None
) -> np.ndarray:
    """Count of the number of contributing pairs per triplet

    Parameters
    ----------
    triplets:
        ``(n, 3)`` array of event index triplets
    ignore:
        Do not count pairs that contain these event indices

    Returns
    -------
    Number of pairs contributing to each triplet
    """

    # Normalize triangles (sort rows so a<b<c)
    T = np.sort(triplets, axis=1)  # (n,3)

    # Build all pairs for each triangle: (a,b), (a,c), (b,c) and normalize (u<=v)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    pairs = np.sort(
        np.stack(
            [
                np.stack([a, b], axis=1),
                np.stack([a, c], axis=1),
                np.stack([b, c], axis=1),
            ],
            axis=1,
        ),
        axis=2,
    )  # (n,3,2)

    # Flatten to (3n,2)
    P2 = pairs.reshape(-1, 2)

    # (3n,) True where at least one element of the pair is ignored
    iignore = np.bitwise_or.reduce(np.isin(P2, ignore), axis=1)

    # Get counts per unique pair
    _, inverse, counts = np.unique(P2, axis=0, return_inverse=True, return_counts=True)

    # Map each pair occurrence to its frequency
    pair_freqs = counts[inverse]  # (3n,)

    # But don't count pairs which have at least one ignored element
    pair_freqs[iignore] = 0

    # Reshape to pair occurences in triplets
    freqs_per_triplet = pair_freqs.reshape(-1, 3)  # (n,3)

    # And sum all occurences
    scores = freqs_per_triplet.sum(axis=1)

    return scores


def item_count(array: Iterable) -> np.ndarray:
    """Count of the items in the list"""
    _, inv, cnt = np.unique(array, return_counts=True, return_inverse=True)
    return cnt[inv]


def station_gap(
    station_dict: dict[str, core.Station], event_dict: dict[int, core.Event]
) -> dict[str, float]:
    """Azimuthal gap from center of event clud that would remain if station was removed

    Parameters
    ----------
    station_dict:
        Station table
    event_dict:
        The seismic event catalog

    Returns
    -------
    Mapping from station name to azimuthal gap (degree)
    """

    # Center of event cloud
    evxyz = xyzarray(event_dict)
    orig = evxyz.mean(axis=0)

    # Stations and corresponding aziumuths
    stas = np.array(list(station_dict.keys()))
    azis = np.array(
        [
            angle.azimuth(orig[0], orig[1], sta.north, sta.east)
            for sta in station_dict.values()
        ]
    )

    # Sort to get gap
    azsort = np.argsort(azis)
    sazis = azis[azsort]
    azgap = np.diff(list(sazis) + [sazis[0] + 360])

    # Unsort to get names sorted by azimuth
    sstas = stas[azsort]

    # Add neigboring gaps to get gap if station is removed
    gap = {sta: azgap[n] + azgap[n - 1] for n, sta in enumerate(sstas)}

    return gap


def collect_takeoff(
    phase_dict: dict[str : core.Phase],
    event_index: int,
    stations: set[str] | list[str] | None = None,
):
    """Take-off azimuth and plunge angles for an event

    Parameters
    ----------
    phase_dict:
        Lookup table for phase arrivals
    event_index:
        Index of the event to consider
    stations:
        Limit output to certain stations (e.g. those with actual amplitude
        observations)

    Returns
    -------
    azimuths: np.ndarray
        Take-off azimuths (degree)
    plunges: np.ndarray
        Take-off plunges (degree)
    stas: np.ndarray
        Corrsponding station names
    phases: np.ndarray
        Corresponding phase types
    """

    apsp = np.array(
        [
            (
                phase_dict[phid].azimuth,
                phase_dict[phid].plunge,
                core.split_phaseid(phid)[1],  # Station
                core.split_phaseid(phid)[2],  # Phase
            )
            for phid in phase_dict
            if core.split_phaseid(phid)[0] == event_index
            and ((stations is None) or (core.split_phaseid(phid)[1] in stations))
            and np.isfinite(phase_dict[phid].plunge)
        ],
        ndmin=2,
        dtype=object,
    )

    azis = apsp[:, 0].astype(float)
    plus = apsp[:, 1].astype(float)
    stas = apsp[:, 2].astype(str)
    phas = apsp[:, 3].astype(str)

    return azis, plus, stas, phas


def source_duration(magnitude: float) -> float:
    """
    Approximate duration of an earthquake with given magnitude

    .. math::
        t = 3^{M-5}

    Parameters
    ----------
    magnitude:
        Magnitude :math:`M` of the event

    Returns
    -------
    Approximate source duration :math:`t` (second)
    """

    return 3 ** (magnitude - 5)


def corner_frequency(
    magnitude: float, phase: str, stress_drop: float, vs: float
) -> float:
    """
    Return a rought estimate of the corner frequency based on

    .. math::
        f_c = (16/7 \\sigma k_s^3 v_S^3 / M_0 )^{1/3}

    Parameters
    ----------
    magnitude: float
        Magnitude :math:`M_0`
    phase : str
        P (:math:`k_s` = 0.375) or S (:math:`k_s` = 0.21)
    stress_drop : float
        Stress drop :math:`\\sigma` in Pa (use ~5e6)
    vs : float
        S wave velocity :math:`v_S` in m/s

    Returns
    -------
    Corner frequency in Hz
    """

    ks = 0.375
    if phase == "S":
        ks = 0.21

    M0 = mt.moment_of_magnitude(magnitude)

    return (stress_drop * ks**3 * vs**3 * 16 / 7 / M0) ** (1 / 3)


def pc_index(mtx: np.ndarray, phase: str) -> np.ndarray:
    """
    Return principal component sorting of seismogram matrix

    For P-phases, sort by zero-th left hand singular vector
    For S-phases, sort angle in the plane spanned by the zero-th and first left
    hand singular vectors

    Parameters
    ----------
    mtx:
        Waveform matrix of shape ``(events, components * samples)``
    phase:
        P or S phase

    Returns
    -------
    Index array

    Raises
    ------
    ValueError:
        If phase is not `P` or `S`
    """

    # Avoid NaN or all-zero values
    inz = qc.index_nonzero_events(mtx)
    normed = np.zeros_like(mtx)
    normed[inz, :] = signal.norm_power(mtx[inz, :])

    # Do the SVD on the verified data
    U, s, _ = svd(normed, False)
    if phase == "P":
        return np.argsort(s[0] * U[:, 0])
    elif phase == "S":
        return np.argsort(np.arctan2(s[1] * U[:, 1], s[0] * U[:, 0]))
    else:
        raise ValueError(f"Unknown phase: {phase}")


def subset_list(lst: list, indices: Iterable[int]) -> list:
    """
    Return subset of event dictionary

    Parameters
    ----------
    event_list:
        Event list
    indices:
        Indices of event_list to include in dictionary

    Returns
    -------
    Subset list
    """
    logger.warning("Do not use this function, but a list comprehension instead.")
    return [lst[i] for i in indices]


def _ccorf3_from_d_e_f_vectorized(d, e, f, M, trips):
    """
    Vectorized core of the ccorf3 algorithm to compute triplet-wise S-wave
    correlations. Use combinations of  using only the three off-diagonal
    elements of the wavefom Gram Matrix M.T.M .

    Parameters
    ----------
    d, e, f:
        ``(N, )`` combinations of off-diagonal elements of the Gram Marix
        (d=⟨i,j⟩, e=⟨j,k⟩, f=⟨k,i⟩).
    M:
        The normalized wavform matrix
    trips:
        ``(N, 3)`` event combination triplets to compute

    ..note: Translated to NumPy from M. G. Bostocks code with the help of
    ChatGPT-o5
    """
    d = np.asarray(d, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)

    d2, e2, f2 = d * d, e * e, f * f
    de, ef = d * e, e * f

    # Major (vectorized)
    x1 = 3.0 * (d2 + e2 + f2)
    x2 = -54.0 * (de * f)
    sx2 = 2.0 * np.sqrt(x1)

    rad = np.maximum(0.0, 4.0 * x1**3 - x2**2)
    root = np.sqrt(rad)
    pi = np.pi

    # piecewise phi
    phi = np.empty_like(x2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ar = np.arctan(root / x2)

    # x2 > 0
    m = x2 > 0
    phi[m] = ar[m]
    # x2 < 0
    m = x2 < 0
    phi[m] = ar[m] + pi
    # x2 == 0
    m = ~((x2 > 0) | (x2 < 0))
    phi[m] = pi / 2.0

    ev0 = (3.0 - sx2 * np.cos(phi / 3.0)) / 3.0
    ev1 = (3.0 + sx2 * np.cos((phi - pi) / 3.0)) / 3.0
    ev2 = (3.0 + sx2 * np.cos((phi + pi) / 3.0)) / 3.0

    # We need the largest (λ1) and middle (λ2) eigenvalues for each row
    EV = np.stack([ev0, ev1, ev2], axis=1)  # (N, 3)
    EV_sorted = np.sort(EV, axis=1)[:, ::-1]  # desc
    lam1 = EV_sorted[:, 0]  # largest
    lam2 = EV_sorted[:, 1]  # middle

    # Slopes m1, m2
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = (d * (1.0 - lam1) - ef) / (f * (1.0 - lam1) - d * e)
        m2 = (d * (1.0 - lam2) - ef) / (f * (1.0 - lam2) - d * e)

    # Eigenvectors (unnormalized), components correspond to (i, j, k) = (a, b, c)
    # v1 = [(lam1-1 - e*m1)/f,   m1, 1]
    # v2 = [(lam2-1 - e*m2)/f,   m2, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        v1a = (lam1 - 1.0 - e * m1) / f
        v2a = (lam2 - 1.0 - e * m2) / f

    v1b, v1c = m1, np.ones_like(m1)
    v2b, v2c = m2, np.ones_like(m2)

    # Normalize v1, v2
    n1 = np.sqrt(v1a * v1a + v1b * v1b + v1c * v1c)
    n2 = np.sqrt(v2a * v2a + v2b * v2b + v2c * v2c)
    with np.errstate(invalid="ignore"):
        v1a, v1b, v1c = v1a / n1, v1b / n1, v1c / n1
        v2a, v2b, v2c = v2a / n2, v2b / n2, v2c / n2

    v1 = np.stack([v1a, v1b, v1c], axis=1)
    v2 = np.stack([v2a, v2b, v2c], axis=1)

    def dot(a, b):
        return np.sum(a * b, axis=0)

    cc3 = np.zeros((trips.shape[0], 3))

    for ip, (i, j, k) in enumerate([[0, 1, 2], [1, 2, 0], [2, 0, 1]]):

        ti = trips[:, i]
        tj = trips[:, j]
        tk = trips[:, k]

        c1 = (v2[:, k] * v1[:, i] - v1[:, k] * v2[:, i]) / (
            v1[:, j] * v2[:, k] - v1[:, k] * v2[:, j]
        )
        c2 = (-v2[:, j] * v1[:, i] + v1[:, j] * v2[:, i]) / (
            v1[:, j] * v2[:, k] - v1[:, k] * v2[:, j]
        )

        cc3[:, ip] = (
            c1 * dot(M[:, ti], M[:, tj]) + c2 * dot(M[:, ti], M[:, tk])
        ) / np.sqrt(c1**2 + c2**2 + 2 * c1 * c2 * dot(M[:, tj], M[:, tk]))

    return cc3


def ccorf3_all(gmat: np.ndarray, batch_size: int = 200000):
    """
    Triplet-wise S-wave correlations of seismogram matrix Generalized and optimized ccorf3 for all column triples.

    Parameters
    ----------
    gmat:
        ``(samples, events)``
        Input matrix. Each event is assumed to be normalized
    batch_size:
        Process up to this many triples per batch to control memory.

    Returns
    -------
    cc_all : (C(event,3), 3) float64
        ``(events, events, events)`` S-wave cross correlation coefficient per
        event triplet
    """
    A = np.asarray(gmat, dtype=np.float64)
    M = A.shape[1]
    if M < 3:
        raise ValueError("Need at least 3 columns to form triples.")

    # Gram matrix (M x M)
    G = A.T @ A

    # Preallocate outputs
    # float32 minimum precission required for Fisher averaging
    ccijk = np.zeros((M, M, M), dtype=np.float64)

    # Enumerate all triples of columns (i<j<k)
    comb_iter = combinations(range(M), 3)

    ncomb = int(comb(M, 3))

    logger.info(f"Computing correlations of {ncomb} combinations...")

    # Process in batches to keep memory bounded
    nbatch = int(ncomb // batch_size)
    while True:
        nbatch -= 1
        logger.debug(f"{nbatch} left...")

        # Pull up to batch_size triples
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(comb_iter))
        except StopIteration:
            pass

        if not batch:
            break

        idx = np.array(batch)  # (B, 3) with columns (i,j,k)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]

        # Pull the three off-diagonal correlations from the Gram
        d = G[i, j]  # ⟨i,j⟩
        e = G[j, k]  # ⟨j,k⟩
        f = G[k, i]  # ⟨k,i⟩

        cc_batch = _ccorf3_from_d_e_f_vectorized(d, e, f, gmat, idx)  # (B, 3)

        ccijk[i, j, k] = cc_batch[:, 0]
        ccijk[i, k, j] = cc_batch[:, 1]
        ccijk[j, k, i] = cc_batch[:, 2]
        ccijk[j, i, k] = cc_batch[:, 0]
        ccijk[k, i, j] = cc_batch[:, 1]
        ccijk[k, j, i] = cc_batch[:, 2]

    return ccijk


def mt_clusters(
    mt_list: list[core.MT],
    method: str = "kagan",
    distance_matrix: np.ndarray | None = None,
    max_distance: float = 30,
    min_ev: int = 1,
    link_method: str = "average",
):
    """Cluster moment tensors

    Parameters
    ----------
    mt_list:
        list of moment tensors to cluster
    method:
        Method to compare two events. Choices are:
        - 'ccp' Correlation coefficient of the P radiation
        - 'ccs' Correlation coefficient of the S radiation
        - 'scalar' Normaized scalar product of the tenors
        - 'kagan' Kagan angle of the DC components
    distance_matrix:
        Ignore 'method', but give a pre-computed distance matrix. Can Accalerate
        the process in finding `max_dist` and `ev_min`. Second return argument
        from a previous run is suited. See :func:`scipy.spacial.distance.pdist`
        for format.
    max_distance:
        maximum distance for pairs to include in a cluster
    min_ev:
        Minimum number of events per cluster. Unclustered events will receive
        `0` label.
    link_method:
        Method of linking events passed on to :func:`scipy.cluster.hirearchy.linkage`

    Returns
    -------
    labels:
        Labels of the event clusters, where 0 is the label for unclustered events
    distance_matrix:
        Compressed pairwise distance matrix
    representative:
        The representative element for each label
    """

    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster

    if method == "kagan":

        def distance_function(mt1, mt2):
            return mt.kagan_angle(mt1, mt2)

    elif method == "ccp":

        def distance_function(mt1, mt2):
            return 1 - mt.correlation(mt1, mt2)[0]

    elif method == "ccs":

        def distance_function(mt1, mt2):
            return 1 - mt.correlation(mt1, mt2)[1]

    elif method == "scalar":

        def distance_function(mt1, mt2):
            return 1 - mt.norm_scalar_product(mt1, mt2)

    else:
        raise ValueError(f"Unknown method: '{method}'")

    if distance_matrix is None:
        distance_matrix = pdist(mt_list, distance_function)

    links = linkage(distance_matrix, method=link_method)
    labels = fcluster(links, max_distance, criterion="distance")
    newlabels = labels.copy()

    uniq, cnt = np.unique(labels, return_counts=True)

    isort = np.argsort(cnt)[::-1]

    # Label by cluster size. If smaller min_ev, set label to 0
    for n, ul in enumerate(uniq[isort]):
        il = labels == ul
        newlabel = n + 1
        if sum(il) < min_ev:
            newlabel = 0

        newlabels[il] = newlabel

    # Now find the representative events
    uls = np.unique(newlabels)
    members = {ul: (newlabels == ul).nonzero()[0] for ul in uls}
    repr = {}
    square = squareform(distance_matrix)
    for cid, idxs in members.items():
        sub = square[np.ix_(idxs, idxs)]
        iclose = idxs[np.argmin(sub.mean(axis=1))]
        repr[cid] = iclose

    return newlabels, distance_matrix, repr
