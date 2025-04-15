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
from relmt import core, signal, mt
from scipy.stats import skew, kurtosis
from typing import Iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def _switch_return_bool_not(
    iin: Iterable[int], return_not: bool, return_bool: bool
) -> np.ndarray:
    """Shorthand to convert boolean index array to numeric index array and/or apply not"""
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
    return [amp for amp in amplitudes if amp.misfit <= max_misfit]


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
    exclude_events: list[str],
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
    event_list: list[core.Event],
    magnitude_difference: list[str],
) -> list[core.P_Amplitude_Ratio | core.S_Amplitude_Ratios]:
    """
    Remove amplitude readings made for certain events

    Parameters
    ----------
    amplitudes:
        List of amplitude observations
    event_list:
        The seismic event catalog
    magnitude_difference:
        Maximum allowed difference in magnitude of participating events

    Returns
    -------
    Cleaned list of amplitude observations
    """
    ip, _ = _ps_amplitudes(amplitudes)
    if ip:
        return [
            amp
            for amp in amplitudes
            if abs(event_list[amp[1]].mag - event_list[amp[2]].mag)
            < magnitude_difference
        ]
    return [
        amp
        for amp in amplitudes
        if all(
            [
                abs(event_list[amp[1]].mag - event_list[amp[2]].mag)
                < magnitude_difference,
                abs(event_list[amp[1]].mag - event_list[amp[3]].mag)
                < magnitude_difference,
                abs(event_list[amp[2]].mag - event_list[amp[3]].mag)
                < magnitude_difference,
            ]
        )
    ]


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
        if np.all(
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


def index_nonzero_events(
    array: np.ndarray, return_not: bool = False, return_bool: bool = False
) -> np.ndarray:
    """Return indices of events with non-zero data

    Parameters
    ----------
    array:
        Waveform array to investigate
    return_not:
        Instead, return events with all zero-data
    return_bool:
        Return boolean array (not an index array)

    Returns
    -------
    Indices where array is non-zero (or all-zero if `return_not=True`)
    """

    iin = np.any(array, axis=-1)

    while len(iin.shape) > 1:
        iin = np.any(iin, axis=-1)

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
