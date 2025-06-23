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
from datetime import datetime
import logging
from typing import Iterable
from relmt import core, mt, signal, qc, angle
import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def xyzarray(
    table: dict[str, core.Station] | list[core.Event] | core.Event | core.Station,
) -> np.ndarray:
    """Spatial coordinates as an array

    Parameters
    ----------
    table:
        Event, event list, station, or station dictionary

    Returns
    -------
    ``(rows, 3)`` or ``(3,) north-east-down coordinates"""
    try:
        # Is it a dictionary?
        return np.array([table[key][:3] for key in sorted(table)])
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


def reshape_ccvec(ccvec: np.ndarray, ns: int) -> np.ndarray:
    """Reshape 1D vector of cc coefficeints c_ijk to ``(n, n, n)`` 3D matrix"""

    cc = np.full((ns, ns, ns), 0.0)
    for n, (i, j, k) in enumerate(core.ijk_ccvec(ns)):
        val = max(min(ccvec[n], 1.0), -1.0)
        cc[i, j, k] = val
        cc[i, k, j] = val
        cc[k, i, j] = val
        cc[k, j, i] = val
        cc[j, k, i] = val
        cc[j, i, k] = val

    return cc


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
    event_list: list[core.Event],
    station_dict: dict[str, core.Station],
    overwrite: bool = False,
) -> dict[str, core.Phase]:
    """
    Fill phase dictionary with azimuth from trigonometry

    Parameters
    ----------
    phase_dict:
        All seismic phases
    event_list:
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
                    *xyzarray(event_list[core.split_phaseid(phid)[0]])[:2],
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
            else core.Phase(ph.time, azis[nph], ph.plunge)
        )
        for nph, (phid, ph) in enumerate(phase_dict.items())
    }

    return new_phase_dict


def phase_dict_hash_plunge(
    phase_dict: dict[str, core.Phase],
    event_list: list[core.Event],
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
    event_list:
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
                    *xyzarray(event_list[core.split_phaseid(phid)[0]])[:2],
                    0,
                ),
                event_list[core.split_phaseid(phid)[0]].depth,
            )
            for phid in phase_dict
        ]
    )

    # Distance and depth coordinate vectors
    # (TODO: if interpolation errors occurr, add a small margin here)
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
    event_list: list[core.Event],
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
    event_list:
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
                        - event_list[event_index][3],  # arrival - origin time
                        cartesian_distance(
                            *sxyz, *event_list[event_index][:3]
                        ),  # distance
                        phase_dict[pid].azimuth,
                        phase_dict[pid].plunge,
                        event_list[event_index][2],  # event z-coordinate
                    )
                    for pid in phase_dict
                    for event_index in range(len(event_list))
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
            logger.debug(
                "Travel times are: "
                + ", ".join(["{:.1e}".format(t) for t in tt])
                + " sec"
            )
            logger.debug(
                "Distances are: "
                + ", ".join(["{:.1e}".format(dd) for dd in d])
                + " meters"
            )

            # Average path velocity
            v = np.average(d / tt)
            logger.info(
                f"{ph}-wave velocity along raypath to station {st} is "
                + "{:.0f} +/- {:.0f} m/s".format(v, np.std(d / tt))
            )

            pluns, deps = tdaiz[np.isfinite(tdaiz[:, 3]), 3:5].T

            # Now look for missing events
            for event_index in range(len(event_list)):
                thisphid = core.join_phaseid(event_index, st, ph)
                ex, ey, ez, t0 = event_list[event_index][:4]

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


@core._doc_config_args
def fftw_data_window(
    minimum_time: float, sampling_rate: float, components: str
) -> float:
    """
    Set 'data_window' based on sampling rate and number of components

    Parameters
    ----------
    minimum_time:
        Minimum length of single-channel time window

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
        f_c = (16/7 \sigma k_s^3 v_S^3 / M_0 )^{1/3}

    Parameters
    ----------
    magnitude: float
        Magnitude :math:`M_0`
    phase : str
        P (:math:`k_s` = 0.375) or S (:math:`k_s` = 0.21)
    stress_drop : float
        Stress drop :math:`\sigma` in Pa (use ~5e6)
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


def subset_list(event_list: list, indices: Iterable[int]) -> list:
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
    return [event_list[i] for i in indices]
