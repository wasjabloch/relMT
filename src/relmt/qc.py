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

"""Functions for Quality Control"""

import numpy as np
import logging
from relmt import core, signal, mt, utils, qc, angle
from scipy.stats import skew, kurtosis
from scipy.linalg import svd
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from typing import Iterable
from collections import Counter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def _switch_return_bool_not(
    iin: Iterable[bool], return_not: bool, return_bool: bool
) -> np.ndarray:
    """Shorthand to convert boolean index array to numeric index array and/or
    apply not"""
    iin = np.asarray(iin)
    if return_not:
        iin = ~iin
    if return_bool:
        return iin
    return iin.nonzero()[0]


def _ps_amplitudes(
    amplitudes: list[core.P_Amplitude_Ratio] | list[core.S_Amplitude_Ratios],
) -> tuple[bool, slice]:
    """True, slice(1,3) for P amplitudes. False, slice(1,4) for S amplitdues"""

    ip = isinstance(amplitudes[0], core.P_Amplitude_Ratio)

    iev = slice(1, 3)
    if not ip:
        iev = slice(1, 4)

    return ip, iev


def clean_by_misfit(
    amplitudes: list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios],
    max_misfit: float,
) -> list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios]:
    """
    Remove amplitude readings with a misfit larger `max_misfit`

    Parameters
    ----------
    amplitudes:
        List of amplitude observations
    max_misfit:
        maximum misfit to keep

    Returns
    -------
    Cleaned list of amplitude observations
    """
    return [amp for amp in amplitudes if amp.misfit < max_misfit]


def clean_by_station(
    amplitudes: list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios],
    exclude_stations: list[str],
) -> list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios]:
    """
    Remove amplitude readings made on certain stations

    Parameters
    ----------
    amplitudes:
        List of amplitude observations
    exclude_stations:
        Station observations to exclude

    Returns
    -------
    Cleaned list of amplitude observations
    """
    return [amp for amp in amplitudes if amp.station not in exclude_stations]


def clean_by_event(
    amplitudes: list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios],
    exclude_events: list[int],
) -> list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios]:
    """
    Remove amplitude readings made for certain events

    Parameters
    ----------
    amplitdues:
        List of amplitude observations
    exclude_events:
        Event observations to exclude

    Returns
    -------
    Cleaned list of amplitude observations
    """
    ip, _ = _ps_amplitudes(amplitudes)

    if ip:
        # P phases
        return [
            amp
            for amp in amplitudes
            if amp.event_a not in exclude_events and amp.event_b not in exclude_events
        ]
    else:
        return [
            amp
            for amp in amplitudes
            if amp.event_a not in exclude_events
            and amp.event_b not in exclude_events
            and amp.event_c not in exclude_events
        ]


def clean_by_magnitude_difference(
    amplitudes: list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios],
    event_dict: dict[int, core.Event],
    magnitude_difference: list[str] | None,
) -> list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios]:
    """
    Remove amplitude readings made for certain events

    Parameters
    ----------
    amplitudes:
        List of amplitude observations
    event_dict:
        The seismic event catalog
    magnitude_difference:
        Maximum allowed difference in magnitude of participating events

    Returns
    -------
    Cleaned list of amplitude observations
    """

    if magnitude_difference is None:
        return amplitudes

    ip, _ = _ps_amplitudes(amplitudes)
    if ip:
        return [
            amp
            for amp in amplitudes
            if abs(event_dict[amp[1]].mag - event_dict[amp[2]].mag)
            < magnitude_difference
        ]
    return [
        amp
        for amp in amplitudes
        if all(
            [
                abs(event_dict[amp[1]].mag - event_dict[amp[2]].mag)
                < magnitude_difference,
                abs(event_dict[amp[1]].mag - event_dict[amp[3]].mag)
                < magnitude_difference,
                abs(event_dict[amp[2]].mag - event_dict[amp[3]].mag)
                < magnitude_difference,
            ]
        )
    ]


def clean_by_event_distance(
    amplitudes: list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios],
    event_dict: dict[int, core.Event],
    event_distance: float,
) -> list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios]:
    """Remove amplitude readings of distant event combinations

    Event pairs or triplets whose maximum distance is larger than
    `event_distance` will be removed.

    Parameters
    ----------
    amplitudes:
        List of amplitude observations
    event_dict:
        The seismic event catalog
    event_distance:
        Maximum distance between events

    Returns
    -------
    Cleaned list of amplitude observations
    """

    if event_distance is None:
        return amplitudes

    ip, _ = _ps_amplitudes(amplitudes)

    xyz = utils.xyzarray(event_dict)

    # Event indices
    ievd = {evn: iev for iev, evn in enumerate(event_dict)}

    if ip:
        ievs = np.array([(ievd[amp.event_a], ievd[amp.event_b]) for amp in amplitudes])
        dist = utils.cartesian_distance(*xyz[ievs[:, 0], :].T, *xyz[ievs[:, 1], :].T)
    else:
        ievs = np.array(
            [
                (ievd[amp.event_a], ievd[amp.event_b], ievd[amp.event_c])
                for amp in amplitudes
            ]
        )
        ab = utils.cartesian_distance(*xyz[ievs[:, 0], :].T, *xyz[ievs[:, 1], :].T)
        ac = utils.cartesian_distance(*xyz[ievs[:, 0], :].T, *xyz[ievs[:, 2], :].T)
        bc = utils.cartesian_distance(*xyz[ievs[:, 1], :].T, *xyz[ievs[:, 2], :].T)
        # Maximum distance between any two events
        dist = np.max(np.array([ab, ac, bc]), axis=0)

    iin = (dist <= event_distance).nonzero()[0]

    if not any(iin):
        logger.warning("Excluded all observations.")

    return [amplitudes[n] for n in iin]


def clean_by_equation_count_gap(
    p_amplitudes: list[core.P_Amplitude_Ratio],
    s_amplitudes: list[core.S_Amplitude_Ratios],
    phase_dict: dict[str, core.Phase],
    min_equations: int | None,
    max_gap: float | None = None,
) -> tuple[list[core.P_Amplitude_Ratio], list[core.S_Amplitude_Ratios]]:
    """Remove observations that occurr in less than `min_equations` iteratively

    Counts the number of ocurrences of each event. Events that occurr fewer than
    `min_equations` times are discarded, togther with the events they are
    connected to. This procedure is repeated until all remaining events are
    constrained by at least 'min_equations` equations.

    Parameters
    ----------
    p_amplitudes:
        List of P amplitude observations
    S_amplitudes:
        List of S amplitude observations
    min_equations:
        Minimum number of occurrences of each event

    Returns
    -------
    Cleaned list of P- and S-amplitude observations
    """

    if min_equations is None:
        min_equations = 0

    if max_gap is None:
        max_gap = 360.0

    if min_equations is None and max_gap is None:
        return p_amplitudes, s_amplitudes

    p_pairs = np.array([(amp.event_a, amp.event_b) for amp in p_amplitudes])
    s_triplets = np.array(
        [(amp.event_a, amp.event_b, amp.event_c) for amp in s_amplitudes]
    )

    # Below created with ChatGPT o4-mini-high
    keep_p = np.full(len(p_pairs), True)
    keep_s = np.full(len(s_triplets), True)

    while True:
        # Count occurrences of each integer over the kept events
        cnt = Counter()

        # P and S indeces
        pin = np.nonzero(keep_p)[0]
        sin = np.nonzero(keep_s)[0]

        for i in pin:
            cnt.update(p_pairs[i])

        for i in sin:
            cnt.update(s_triplets[i])

        # Amplitdue subsets
        psub = [p_amplitudes[i] for i in pin]
        ssub = [s_amplitudes[i] for i in sin]

        gap = angle.azimuth_gap(phase_dict, psub, ssub)

        # Events with too few observations
        bad = {evn for evn in cnt if cnt[evn] < min_equations or gap[evn][0] > max_gap}

        if any(bad):
            logger.debug("Excluding events: " + ", ".join(f"{f}" for f in bad))

        # Drop all occurrences of the events with too few observations
        new_keep_p = keep_p.copy()
        for i in np.nonzero(keep_p)[0]:
            if any(x in bad for x in p_pairs[i]):
                new_keep_p[i] = False

        new_keep_s = keep_s.copy()
        for i in np.nonzero(keep_s)[0]:
            if any(x in bad for x in s_triplets[i]):
                new_keep_s[i] = False

        # If nothing was dropped, exit the loop
        if np.array_equal(new_keep_p, keep_p) and np.array_equal(new_keep_s, keep_s):
            break

        # Else, start counting again
        keep_p, keep_s = new_keep_p, new_keep_s

    # Included P and S observations
    pin = np.nonzero(keep_p)[0]
    sin = np.nonzero(keep_s)[0]

    if not any(pin):
        logger.warning("Excluded all P-observations.")

    if not any(sin):
        logger.warning("Excluded all S-observations.")

    return [p_amplitudes[n] for n in pin], [s_amplitudes[n] for n in sin]


def clean_by_valid_takeoff_angle(
    amplitudes: list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios],
    phase_dictionary: dict[str, core.Phase],
) -> list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios]:
    """
    Remove amplitude readings with invalid corresponding take-off angle

    Parameters
    ----------
    amplitudes:
        List of amplitude observations
    phase_dictionary:
        All seismic phases

    Returns
    -------
    Cleaned list of amplitude observations
    """

    # Check if we are dealing with P or S
    ip, iev = _ps_amplitudes(amplitudes)
    pha = "S"
    if ip:
        pha = "P"

    # Return amplitude only if a ll take-off angles are finite
    return [
        amp
        for amp in amplitudes
        # First test if all phases are present
        if np.all(
            [
                core.join_phaseid(ev, amp.station, pha) in phase_dictionary
                for ev in amp[iev]
            ]
        )
        # Then if they are valid
        and np.all(
            [
                np.isfinite(
                    [
                        phase_dictionary[
                            core.join_phaseid(ev, amp.station, pha)
                        ].azimuth,
                        phase_dictionary[
                            core.join_phaseid(ev, amp.station, pha)
                        ].plunge,
                    ]
                )
                for ev in amp[iev]
            ]
        )
    ]


def clean_by_kurtosis(
    amplitudes: list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios],
    event_list: list[core.Event],
    max_kurtosis: float,
) -> list:
    """
    Remove amplitude readings until amplitude distribution has max kurtosis

    Expected amplitude ratios are calculated from seismic moments, wich are
    derived from the magnitudes.

    For P-waves, we investigate the distribution:

    .. math::
        \log_{10}( A_{ab} M_0^b / M_0^a )

    For S-waves:

    .. math::
        \log_{10}( (B_{abc} M_0^b + B_{acb} M_0^c) / M_0^a )

    and all equivalent event combinations.

    Parameters
    ----------
    amplitudes:
        List of amplitude observations
    event_list:
        List of events with magnitude information
    max_kurtosis:
        Maximum allowed curtosis for distribution (0 is normal distribution)

    Returns
    -------
    Cleaned list of amplitude observations

    Raises
    ------
    ValueError:
        If amplitudes is not all P or S Amplitude Ratios
    """

    ip, iev = _ps_amplitudes(amplitudes)

    # Moment of event 1, 2 (, 3) for each amplitude ratio reading
    moms = np.array(
        [
            [mt.moment_of_magnitude(event_list[ev].mag) for ev in amp[iev]]
            for amp in amplitudes
        ]
    )

    # Expected moment ratio
    # Spell out all 2 (3) combinations of events
    if ip:
        Aab = np.array([amp.amp_ab for amp in amplitudes])
        momr = np.array(
            [Aab * moms[:, 1] / moms[:, 0], 1 / Aab * moms[:, 0] / moms[:, 1]]
        ).T
    else:
        Babc = np.array([amp.amp_abc for amp in amplitudes])
        Bacb = np.array([amp.amp_acb for amp in amplitudes])
        momr = np.array(
            [
                (Babc * moms[:, 1] + Bacb * moms[:, 2]) / moms[:, 0],
                (1 / Babc * moms[:, 0] - Bacb / Babc * moms[:, 2]) / moms[:, 1],
                (1 / Bacb * moms[:, 0] - Babc / Bacb * moms[:, 1]) / moms[:, 2],
            ]
        ).T

    # Corresponding event indices
    events = np.array([amp[iev] for amp in amplitudes])
    stations = np.array([amp.station for amp in amplitudes])

    for sta in sorted(set(stations)):
        for eva in sorted(set(events.flatten())):
            # Boolean indices to master event
            iin = (events == eva) & (sta == stations)[:, np.newaxis]
            inrow, _ = iin.nonzero()

            smrd = np.log10(np.abs(momr[iin]))

            while kurtosis(smrd) > max_kurtosis:
                # Scaled moment ratio distribution
                ske = skew(smrd)
                if ske > 0:
                    # cut positive end
                    ibad = np.argmax(smrd)
                else:
                    # cut negative end
                    ibad = np.argmin(smrd)

                badrow = inrow[ibad]
                logger.info(f"Removing observation {badrow}: {amplitudes[badrow]}")

                # Remove bad observation
                momr = np.delete(momr, badrow, axis=0)
                smrd = np.delete(smrd, ibad)
                logger.info(f"Kurtosis is: {kurtosis(smrd)}")

                # Remove bad station event combination
                stations = np.delete(stations, badrow, axis=0)
                events = np.delete(events, badrow, axis=0)
                amplitudes.pop(badrow)

                iin = (events == eva) & (sta == stations)[:, np.newaxis]
                inrow, _ = iin.nonzero()

    return amplitudes


def expansion_coefficient_norm(arr: np.ndarray, phase: str) -> np.ndarray:
    """Normalized expansion coefficient sum for each seismogram in matrix

    Decompose seismogram matrix into principal seismograms. Return squared
    contribution of first (and second) principal seismogram to each original
    seismogram for P- (S-) waves.

    1 (best) indicates principal seismogram(s) fully reconstruct the respective
    waveform

    0 (worst) indicates the respective waveform cannot be represented by the
    principal seismogram(s)

    Parameters
    ----------
    arr:
        Waveform ``(events, channels, samples)`` array or ``(events, channels *
        samples)`` matrix
    phase:
        `P` or `S` wave type

    Returns:
    --------
    ``(events,)`` score per event

    Raises:
    -------
    IndexError:
        If 'arr' has not 2 or 3 dimensions.
    """

    if (ndim := len(arr.shape)) == 3:
        mat = utils.concat_components(arr)
    elif ndim == 2:
        mat = arr
    else:
        msg = f"'arr' must have 2 or 3 dimension, not: {ndim}"
        raise IndexError(msg)

    mat = signal.norm_power(mat)

    # Zero invalid events
    ibad = qc.index_nonzero_events(mat, return_not=True)
    mat[ibad, :] = 0.0

    U, s, _ = svd(mat, False)

    # We sum the squared values
    ec = (s[0] * U[:, 0]) ** 2
    if phase == "S":
        try:
            ec += (s[1] * U[:, 1]) ** 2
        except IndexError:
            ec += np.nan

    # And return the root
    return np.sqrt(ec)


def included_events(
    exclude: core.Exclude | None,
    station: str,
    phase: str,
    events: list[int],
    return_not: bool = False,
    return_bool: bool = False,
) -> tuple[list[int | bool], list[int]]:
    """
    Return events that are not excluded according to `exclude` dictionary

    Parameters
    ----------
    exclude:
        Dictionary holding exclusion criteria. If `None`, consider all events
        included.
    station:
        Station code to consider
    phase:
        Phase type to consider (`P` or `S`)
    events:
        Event IDs on that station
    return_not:
        Instead, return events that are excluded
    return_bool:
        Return boolean array (not an index array)

    Returns
    -------
    ievs:
        Indices into `events` of included (or excluded if `return_not=True`)
        events
    evns:
        Global event IDs of included (or excluded if `return_not=True`)
        events
    """

    # Get phases already excluded
    exphids = []

    # Start excluding if exlude is not None
    if exclude is not None:
        for key in core.exclude:
            if key.startswith("phase_"):
                exphids += exclude.get(key, [])

    exevs = []
    if exclude is not None:
        exevs += exclude.get("event", [])

    # Excluded event IDs
    exevs += [
        core.split_phaseid(phid)[0]
        for phid in exphids
        if core.split_phaseid(phid)[1:] == (station, phase)
    ]

    # Boolean index of not-excluded (=included) events
    ievs = np.array([False if evn in exevs else True for evn in events])

    # Included event names
    inevns = np.array(events)[ievs].astype(int)

    # excluded event names
    exevns = np.array(events)[~ievs].astype(int)

    if return_not:
        if return_bool:
            return ~ievs, exevns
        return (~ievs.nonzero()[0]).tolist(), exevns.tolist()

    if return_bool:
        return ievs, inevns

    return ievs.nonzero()[0].tolist(), inevns.tolist()


def index_nonzero_events(
    array: np.ndarray,
    null_threshold: float | None = None,
    return_not: bool = False,
    return_bool: bool = False,
) -> np.ndarray:
    """Return indices of events with non-zero and finite data

    Parameters
    ----------
    array:
        Waveform array to investigate
    null_threshold:
        Consider absolute value at or below as zero.
    return_not:
        Instead, return events with all zero-data
    return_bool:
        Return boolean array (not an index array)

    Returns
    -------
    Indices where array is non-zero (or all-zero if `return_not=True`)
    """

    if null_threshold is None:
        null_threshold = 0.0

    # Any sample needs to be non-zero
    inz = np.any(array, axis=-1)
    inz = np.max(abs(array), axis=-1) > null_threshold

    # All samples needs to be finite
    ifi = np.all(np.isfinite(array), axis=-1)
    iin = inz & ifi

    # Irrespective of the shape of the array, return a 1D index
    while len(iin.shape) > 1:
        # We want all components to have data
        inz = np.all(inz, axis=-1)
        ifi = np.all(ifi, axis=-1)
        iin = inz & ifi

    return _switch_return_bool_not(iin, return_not, return_bool)


def index_high_value(
    values: np.ndarray,
    threshold: float,
    return_not: bool = False,
    return_bool: bool = False,
) -> np.ndarray:
    """Return indices of events with value above threshold

    Parameters
    ----------
    values:
        Vector to investigate
    threshold:
        Value above which to return the corresponding index
    return_not:
        Instead, return low values
    return_bool:
        Return boolean index array (not an index array)

    Returns
    -------
    Indices where value is above threshold (or below or equal, if `reutrn_not=True`)

    """
    iin = values > threshold
    return _switch_return_bool_not(iin, return_not, return_bool)


def connected_events(
    reference_mts: list[int],
    p_amplitudes: list[core.P_Amplitude_Ratio] | None = None,
    s_amplitudes: list[core.S_Amplitude_Ratios] | None = None,
) -> dict[int, tuple[int, int]]:
    """Event connected to the reference MTs

    Investigate the paired P- and triplet S-observations for connectivity. Only
    return those events that are connected to at least one reference MT.

    Parameters
    ----------
    reference_mts:
        List of indices to the reference events
    p_amplitudes:
        List of P-amplitude observations
    s_amplitudes:
        List of S-amplitude observations

    Returns
    -------
    Mapping of event indices of the connected events to number of P- and
    S-connections

    Raises
    ------
    ValueError:
        If any reference MT has no amplitude observation
    RuntimeError:
        If reference MTs are not connected with each other
    """

    # Collect unique connections
    pcons = set()
    scons = set()

    # events in order of appearance
    evns = []

    if p_amplitudes is not None:
        for amp in p_amplitudes:
            eva = amp.event_a
            evb = amp.event_b

            if eva not in evns:
                evns.append(eva)
            if evb not in evns:
                evns.append(evb)

            # Keep the integers small and positive in order not to create an
            # enormous matrix down the road
            pcons.add(tuple(sorted((evns.index(eva), evns.index(evb)))))

    if s_amplitudes is not None:
        for amp in s_amplitudes:
            eva = amp.event_a
            evb = amp.event_b
            evc = amp.event_c

            if eva not in evns:
                evns.append(eva)
            if evb not in evns:
                evns.append(evb)
            if evc not in evns:
                evns.append(evc)

            scons.add(tuple(sorted((evns.index(eva), evns.index(evb)))))
            scons.add(tuple(sorted((evns.index(eva), evns.index(evc)))))
            scons.add(tuple(sorted((evns.index(evb), evns.index(evc)))))

    cons = pcons.union(scons)

    if len(cons) < 1:
        raise ValueError("No connections")

    # Build a graph from the connections
    rows, cols = zip(*sorted(cons))  # Arbitrary order
    nev = len(evns)
    data = [1] * len(rows)  # All edges count the same

    # Check if all MTs are represented
    iin = np.isin(reference_mts, evns)
    if not all(iin):
        msg = "These reference MTs have no amplitude measurement: "
        msg += ", ".join(f"{reference_mts[i]}" for i in (~iin).nonzero()[0])
        raise ValueError(msg)

    # Connected components assumes a square matrix
    graph = csr_array((data, (rows, cols)), (nev, nev))

    # Event index -> cluster within graph
    _, groups = connected_components(graph, False)

    # Group indices of moment tensors
    igs = [groups[evns.index(imt)] for imt in reference_mts]

    # Check if all MTs are connected to each other
    if len(set(igs)) > 1:
        msg = (
            "Not all reference MTs are connected with each other. Consider to "
            "solve connected groups individually or to improve connectivity by "
            "excluding fewer amplitude observations.\n"
        )
        msg += "MT groupings are: " + ", ".join(
            f"MT {imt}: Group {ig}" for imt, ig in zip(reference_mts, igs)
        )
        raise RuntimeError(msg)

    # If everything has worked, let's count the connections
    pcount = Counter(np.array(list(pcons)).flat)
    scount = Counter(np.array(list(scons)).flat)

    connections = {
        evns[nev]: (pcount[nev], scount[nev]) for nev in (groups == igs[0]).nonzero()[0]
    }

    return connections


def events_phases_above_misfit(
    amplitudes: list[core.P_Amplitude_Ratio] | list[core.S_Amplitude_Ratios],
    misfit: float,
    subtract_below: float = 0.0,
) -> Counter[str, int]:
    """Count the phases that have a misfit larger than the given one

    Parameters
    ----------
    amplitudes:
        List of amplitude observations
    misfit:
        Maximum misfit to consider
    subtract_below:
        Subtract the number of observations with a misfit below this value
    Returns
    -------
    events:
        Counter with events as keys and number of bad observations as values
    phases:
        Counter with phase names as keys and number of bad observations as values
    """

    phases = Counter()
    events = Counter()

    isubtract = subtract_below > 0

    for amp in amplitudes:
        if amp.misfit >= misfit:
            if isinstance(amp, core.P_Amplitude_Ratio):
                for iev in [amp.event_a, amp.event_b]:
                    phases[core.join_phaseid(iev, amp.station, "P")] += 1
                    events[iev] += 1
            elif isinstance(amp, core.S_Amplitude_Ratios):
                for iev in [amp.event_a, amp.event_b, amp.event_c]:
                    phases[core.join_phaseid(iev, amp.station, "S")] += 1
                    events[iev] += 1
        elif isubtract and amp.misfit < subtract_below:
            if isinstance(amp, core.P_Amplitude_Ratio):
                for iev in [amp.event_a, amp.event_b]:
                    phases[core.join_phaseid(iev, amp.station, "P")] -= 1
                    events[iev] -= 1
            elif isinstance(amp, core.S_Amplitude_Ratios):
                for iev in [amp.event_a, amp.event_b, amp.event_c]:
                    phases[core.join_phaseid(iev, amp.station, "S")] -= 1
                    events[iev] -= 1

    return events, phases
