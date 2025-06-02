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

"""Functions to set up and solve the linear systems"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import lsmr
from scipy.sparse import spmatrix
from relmt import utils, core
from relmt import mt as relmtmt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def gamma(azimuth: float, plunge: float) -> np.ndarray:
    """
    Produce 3-element direction cosine vector

    Parameters
    ----------
    azimuth:
        Degree east of north
    plunge:
        Degree down from horizontal

    Returns
    -------
    ``(3,)`` direction cosines along north (x), east (y), down (z) axis
    """

    azi = azimuth * np.pi / 180
    plu = plunge * np.pi / 180
    return np.array([np.cos(azi) * np.cos(plu), np.sin(azi) * np.cos(plu), np.sin(plu)])


def directional_coefficient_general_p(gamma: np.ndarray) -> np.ndarray:
    """
    6-element direction coefficeint vector for unconstrained moment tensor

    Implementation of Ploudre & Bostock (2019, GJI), Eq. A2

    Parameters
    ----------
    gamma:
        Direction cosines produced by :func:`gamma`

    Returns
    -------
    ``(6,)`` directional coefficients
    """
    return np.array(
        [
            gamma[0] ** 2,
            gamma[1] ** 2,
            gamma[2] ** 2,
            2 * gamma[0] * gamma[1],
            2 * gamma[0] * gamma[2],
            2 * gamma[1] * gamma[2],
        ]
    )


def directional_coefficient_deviatoric_p(gamma: np.ndarray) -> np.ndarray:
    """
    5-element direction coefficeint vector for deviatoric moment tensor

    Implementation of Ploudre & Bostock (2019, GJI), Eq. A8

    Parameters
    ----------
    gamma:
        Direction cosines produced by :func:`gamma`

    Returns
    -------
    ``(5,)`` directional coefficients
    """
    return np.array(
        [
            gamma[0] ** 2 - gamma[2] ** 2,
            gamma[1] ** 2 - gamma[2] ** 2,
            2 * gamma[0] * gamma[1],
            2 * gamma[0] * gamma[2],
            2 * gamma[1] * gamma[2],
        ]
    )


def directional_coefficient_general_s(
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3 6-element direction coefficeint vectors for unconstrained moment tensor

    Implementation of Ploudre & Bostock (2019, GJI), Eq. A3

    Parameters
    ----------
    gamma:
        Direction cosines produced by :func:`gamma`

    Returns
    -------
    ``(6,)`` directional coefficients
    """
    gs1 = np.array(
        [
            gamma[0] * (1 - gamma[0] ** 2),
            -gamma[0] * gamma[1] ** 2,
            -gamma[0] * gamma[2] ** 2,
            gamma[1] * (1 - 2 * gamma[0] ** 2),
            gamma[2] * (1 - 2 * gamma[0] ** 2),
            -2 * gamma[0] * gamma[1] * gamma[2],
        ]
    )
    gs2 = np.array(
        [
            -gamma[0] ** 2 * gamma[1],
            gamma[1] * (1 - gamma[1] ** 2),
            -gamma[1] * gamma[2] ** 2,
            gamma[0] * (1 - 2 * gamma[1] ** 2),
            -2 * gamma[0] * gamma[1] * gamma[2],
            gamma[2] * (1 - 2 * gamma[1] ** 2),
        ]
    )
    gs3 = np.array(
        [
            -gamma[0] ** 2 * gamma[2],
            -gamma[1] ** 2 * gamma[2],
            gamma[2] * (1 - gamma[2] ** 2),
            -2 * gamma[0] * gamma[1] * gamma[2],
            gamma[0] * (1 - 2 * gamma[2] ** 2),
            gamma[1] * (1 - 2 * gamma[2] ** 2),
        ]
    )
    return gs1, gs2, gs3


def directional_coefficient_deviatoric_s(
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3 5-element direction coefficeint vectors for deviatoric moment tensor

    Implementation of Ploudre & Bostock (2019, GJI), Eq. A9

    Parameters
    ----------
    gamma: :class:`numpy.ndarray`
        Direction cosines produced by :func:`gamma`

    Returns
    -------
    ``(5,)`` directional coefficients
    """
    gs1 = np.array(
        [
            gamma[0] * (1 - gamma[0] ** 2 + gamma[2] ** 2),
            gamma[0] * (gamma[2] ** 2 - gamma[1] ** 2),
            gamma[1] * (1 - 2 * gamma[0] ** 2),
            gamma[2] * (1 - 2 * gamma[0] ** 2),
            -2 * gamma[0] * gamma[1] * gamma[2],
        ]
    )
    gs2 = np.array(
        [
            gamma[1] * (gamma[2] ** 2 - gamma[0] ** 2),
            gamma[1] * (1 - gamma[1] ** 2 + gamma[2] ** 2),
            gamma[0] * (1 - 2 * gamma[1] ** 2),
            -2 * gamma[0] * gamma[1] * gamma[2],
            gamma[2] * (1 - 2 * gamma[1] ** 2),
        ]
    )
    gs3 = np.array(
        [
            gamma[2] * (gamma[2] ** 2 - gamma[0] ** 2 - 1),
            gamma[2] * (gamma[2] ** 2 - gamma[1] ** 2 - 1),
            -2 * gamma[0] * gamma[1] * gamma[2],
            gamma[0] * (1 - 2 * gamma[2] ** 2),
            gamma[1] * (1 - 2 * gamma[2] ** 2),
        ]
    )
    return gs1, gs2, gs3


def distance_ratio(
    event_a: core.Event, event_b: core.Event, station: core.Station
) -> float | np.ndarray:
    """Ratio of the distance of `event_a` and `event_b` to `station`

    R = distance(event_a, station) / distance(event_b, station)
    """
    return utils.cartesian_distance(
        *event_a[:3], *station[:3]
    ) / utils.cartesian_distance(*event_b[:3], *station[:3])


def p_equation(
    p_amplitude: core.P_Amplitude_Ratio,
    station: core.Station,
    events: list[core.Event],
    phase_dictionary: dict[str, core.Phase],
    nmt: int,
) -> np.ndarray:
    """
    Equation with P wave constraint on the liniear system

    Relative amplitudes are corrected by direction coefficient g and the
    distance ratio from the event pair to the station.

    Parameters
    ----------
    p_amplitude:
        One relative P amplitude observation
    station:
        Station on which the observation has been made
    events:
        The full seismic event catalog
    phase_dictionary:
        All phase observations
    nmt:
        Number of moment tensor elements

    Returns
    -------
    ``(1, events * nmt)`` line of the left hand side of the linear system
    """

    def _gamma(ev, sta, pha):
        phid = core.join_phaseid(ev, sta, pha)
        azi, inc = phase_dictionary[phid].azimuth, phase_dictionary[phid].plunge
        return gamma(azi, inc)

    # Set up one line of the A matrix
    ampl = p_amplitude.amp_ab
    ieva = p_amplitude.event_a
    ievb = p_amplitude.event_b

    # Make sure amplitude < 1
    if abs(ampl) > 1:
        # Invert amplitude and switch indexing
        ampl = 1 / ampl
        ievb = p_amplitude.event_a
        ieva = p_amplitude.event_b

    ga = _gamma(ieva, station.name, "P")
    gb = _gamma(ievb, station.name, "P")

    line = np.zeros(nmt * len(events))

    for iev, g in zip([ieva, ievb], [ga, gb]):
        if not np.all(np.isfinite(g)):
            logging.debug(f"Missing take-off angle in event {iev}. Retruning all zeros")
            return np.array(line)

    rab = distance_ratio(events[ieva], events[ievb], station)
    if nmt == 6:
        gap = directional_coefficient_general_p(ga)
        gbp = directional_coefficient_general_p(gb)
    elif nmt == 5:
        gap = directional_coefficient_deviatoric_p(ga)
        gbp = directional_coefficient_deviatoric_p(gb)
    else:
        raise ValueError(f"'nmt must be '5' or '6', not: '{nmt}'")

    line[nmt * ieva : nmt * ieva + nmt] = -gap
    line[nmt * ievb : nmt * ievb + nmt] = gbp * ampl * rab

    return np.array(line)


def s_equations(
    s_amplitude: core.S_Amplitude_Ratios,
    station: core.Station,
    events: list[core.Event],
    phase_dictionary: dict[str, core.Phase],
    nmt: int,
    choose_coefficients: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Two equations with S-wave constraints on the linear system

    Relative amplitudes are corrected by direction coefficient g and the
    distance ratio from the event pair to the station.

    Parameters
    ----------
    s_amplitude:
        One pair of relative S amplitude observations
    station:
        Station on which the observation has been made
    events:
        The full seismic event catalog
    phase_dictionary:
        All phase observations
    nmt:
        Number of moment tensor elements
    choose_coefficients:
        Select two indices (0, 1, 2) of directional coefficients to use. If
        `None`, chose those with the largest norm

    Returns
    -------
    ``(2, events * nmt)`` lines of the left hand side of the linear system
    """

    def _gamma(ev, sta, pha):
        phid = core.join_phaseid(ev, sta, pha)
        azi, inc = phase_dictionary[phid].azimuth, phase_dictionary[phid].plunge
        return gamma(azi, inc)

    nev = len(events)

    line1 = np.zeros((nmt * nev))
    line2 = np.zeros((nmt * nev))

    ieva = s_amplitude.event_a
    ievb = s_amplitude.event_b
    ievc = s_amplitude.event_c

    ga = _gamma(ieva, station.name, "P")
    gb = _gamma(ievb, station.name, "P")
    gc = _gamma(ievc, station.name, "P")

    for iev, g in zip([ieva, ievb, ievc], [ga, gb, gc]):
        if not np.all(np.isfinite(g)):
            logging.debug(f"Missing take-off angle in event {iev}. Retruning all zeros")
            return np.array([line1, line2])

    amp_abc = s_amplitude.amp_abc
    amp_acb = s_amplitude.amp_acb
    logger.debug(f"Events: {ieva}, {ievb}, {ievc}. Babc: {amp_abc}, Bacb: {amp_acb}")

    rab = distance_ratio(events[ieva], events[ievb], station)
    rac = distance_ratio(events[ieva], events[ievc], station)

    # g[abc]s are 3-tuple. Only two are independent
    if nmt == 6:
        gas = directional_coefficient_general_s(ga)
        gbs = directional_coefficient_general_s(gb)
        gcs = directional_coefficient_general_s(gc)
    elif nmt == 5:
        gas = directional_coefficient_deviatoric_s(ga)
        gbs = directional_coefficient_deviatoric_s(gb)
        gcs = directional_coefficient_deviatoric_s(gc)
    else:
        raise ValueError(f"'nmt must be '5' or '6', not: '{nmt}'")

    # Now find set of directional coefficients that records most of the amplitude:
    # Choose the coefficients with the two highest norms
    if choose_coefficients is None:
        i1, i2 = np.argsort(
            np.linalg.norm(
                (
                    np.concatenate((gas[0], gbs[0], gcs[0])),
                    np.concatenate((gas[1], gbs[1], gcs[1])),
                    np.concatenate((gas[2], gbs[2], gcs[2])),
                ),
                axis=1,
            ),
            kind="stable",  # Keep order of equal elements, so tests succeed.
        )[[1, 2]]
    else:
        i1 = choose_coefficients[0]
        i2 = choose_coefficients[1]

    logger.debug(f"Selected directional coefficients: {i1}, {i2}")
    for i in [i1, i2]:
        for gs, abc in zip([gas, gbs, gcs], "abc"):
            logger.debug(
                f"g{abc}{i}: [" + ", ".join(["{:3.1e}".format(v) for v in gs[i]]) + "]"
            )

    # For S waves, each event twriplet has two lines in the matrix
    line1[nmt * ieva : nmt * ieva + nmt] = -gas[i1]
    line1[nmt * ievb : nmt * ievb + nmt] = amp_abc * rab * gbs[i1]
    line1[nmt * ievc : nmt * ievc + nmt] = amp_acb * rac * gcs[i1]

    line2[nmt * ieva : nmt * ieva + nmt] = -gas[i2]
    line2[nmt * ievb : nmt * ievb + nmt] = amp_abc * rab * gbs[i2]
    line2[nmt * ievc : nmt * ievc + nmt] = amp_acb * rac * gcs[i2]

    return np.array([line1, line2])


def weight_misfit(
    amplitude: core.S_Amplitude_Ratios | core.P_Amplitude_Ratio,
    max_misfit: float,
    phase: str,
) -> np.ndarray:
    """
    Weights [0 ... 1] for each row of the amplitude matrix by misfit

    weight = max(0, maximum_misfit - amp.[ps]_misfit)

    Note
    ----
    A misfit of 0 indicates that the reconstruction and the target waveform are
    identical. A misfit of 1 that the residual of the waveform reconstruction
    has the same norm that the waveform itself.

    Parameters
    ----------
    amplitude:
        One amplitude ratio
    max_misfit:
        Maximum allowed misfit that receives non-zero weight
    phase:
        If 'S', return each weight twice to account for two equations per S wave
        observation

    Returns
    -------
    Weight of shape ``(1,)`` if `phase=P`, or ``(2,)`` if `phase=S`

    Raises
    ------
    ValueError:
        If 'phase' is not 'P' or 'S'
    """

    if phase.upper() == "P":
        return np.array(max(0.0, max_misfit - amplitude.misfit))
    elif phase.upper() == "S":
        return np.array(2 * [max(0.0, max_misfit - amplitude.misfit)])[:, np.newaxis]
    else:
        raise ValueError("'phase' must be 'P' or 'S'")


def weight_s_amplitude(s_amplitudes: core.S_Amplitude_Ratios) -> np.ndarray:
    """Weights for each row of the amplitude matrix by S-wave amplitdue

    Weight is the inverse of the larger amplitudte, but not more than 1.

    weight = max(1, (1 / max(abs(ampl.amp_abc, ampl.amp_acb))))

    Parameters
    ----------
    s_amplitudes:
        One pair of S-amplitude ratios

    Returns
    -------
    ``(2, 1)`` column vector of weights
    """

    # There are 2 equations per S amplitude reading.
    return np.array(
        2 * [min(1.0, (1 / max(np.abs([s_amplitudes.amp_abc, s_amplitudes.amp_acb]))))]
    )[:, np.newaxis]


def homogenous_amplitude_equations(
    p_amplitudes: list[core.P_Amplitude_Ratio],
    s_amplitudes: list[core.S_Amplitude_Ratios],
    station_dictionary: dict[str, core.Station],
    event_list: list[core.Event],
    phase_dictionary: dict[str, core.Phase],
    constraint: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Homogenous part of the linear system Am = b

    Some more description of the function

    Parameters
    ----------
    p_amplitudes:
        P observations to include in the system
    s_amplitudes:
        S observations to include in the system
    station_dictionary:
        Lookup table for station coordinates
    event_list:
        The seismic event catalog
    phase_dictionary:
        Lookup table for ray take-off angles
    constraint:
        Constraint on the moment tensor solution

    Returns
    -------
    Ah: :class:`numpy.ndarray`
        ``(p_amplitudes + 2 * s_amplitudes, event_list * mt_elements)`` left-hand side of the homogenous part of linear system
    bh: :class:`numpy.ndarray`
        ``(event_list * mt_elements, 1)`` zero column vector, right-hand side of the homogenous linear system
    """

    nmt = mt_elements(constraint)

    # Number of equations
    peq = len(p_amplitudes)  # P observations
    seq = 2 * len(s_amplitudes)  # S observations
    neq = peq + seq

    nmod = nmt * len(event_list)  # Length of model vector

    # Set up homogenous matrix and data vector
    Ah = np.zeros((neq, nmod))
    bh = np.zeros((neq, 1))

    # Populate first with P amplitdues
    for n, pamp in enumerate(p_amplitudes):
        station = station_dictionary[pamp.station]

        # Create one P observation
        Ah[n, :] = p_equation(pamp, station, event_list, phase_dictionary, nmt)

    # Populate then with S amplitdues
    for n, samp in enumerate(s_amplitudes):
        station = station_dictionary[samp.station]

        # Create two S observations
        lines = s_equations(samp, station, event_list, phase_dictionary, nmt, (0, 1))

        row = peq + 2 * n
        Ah[[row, row + 1], :] = lines

    return Ah, bh


_mt_elements = {"none": 6, "deviatoric": 5}


def mt_elements(constraint: str) -> int:
    """
    Number of elements of the moment tensor

    Parameters
    ----------
    constraint:
        * 'none' - invert for full moment tensor
        * 'deviatoric' - no isotropic component

    Returns
    -------
    `6` for 'none', `5` for 'deviatoric'

    Raises
    ------
    ValueError:
        If unknown constaint
    """
    try:
        return _mt_elements[constraint]
    except KeyError:
        msg = (
            "Constraint must be one of: "
            + ", ".join(_mt_elements.keys())
            + f". Not: '{constraint}'"
        )
        raise ValueError(msg)


def reference_mt_equations(
    reference_events: list[int],
    refmt_dict: dict[int, core.MT],
    number_events: int,
    constraint: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inhomogenous part of the linear system Am = b

    Absolute moment tensor constraint on the linear system

    Parameters
    ----------
    reference_events:
        Indices to the reference events
    refmt_dict:
        Lookup table from event index to moment tensor
    number_events:
        Length of the seismic event catalog
    constraint:
        Constraint to impose on the moment tensor

    Returns
    -------
    Ai: :class:`numpy.ndarray`
        ``(reference_events * mt_elements, number_events * mt_elements)``
        Left-hand side of the inhomogenous part of the linear system.
    bi: :class:`numpy.ndarray`
        ``(reference_events * mt_elements, 1)``
        Elements of the reference moment tensor, right-hand side of the
        inhomogenous part of the linear system.
    """

    nmt = mt_elements(constraint)

    nref = len(reference_events)
    bi = np.zeros((nmt * nref, 1))
    Ai = np.zeros((nmt * nref, nmt * number_events))
    for n, iev in enumerate(reference_events):
        # Identity matrix for reference event
        Ai[n * nmt : n * nmt + nmt, :] = _reference_mt_matrix(iev, number_events, nmt)

        # Reference MT refomated into column vector
        bi[n * nmt : n * nmt + nmt] = _reference_mt_data_vector(
            refmt_dict[iev],
            constraint=constraint,
        )

    return Ai, bi


def reference_mt_event_norm(
    ev_norm: np.ndarray, ref_mts: list[int], nmt: int
) -> np.ndarray:
    """Pull event normalization factor for reference moment tensor out of matrix event normalization"""
    indices = [iev * nmt + mt_element for iev in ref_mts for mt_element in range(nmt)]
    return ev_norm[indices][:, np.newaxis]


def _reference_mt_matrix(
    reference_index: int, number_of_events: int, nmt: int
) -> np.ndarray:
    """
    Left-hand side of the inhomogenous part of the linear system.

    Identity matrix that imposes one reference moment tensor constraint.

    Parameters
    ----------
    reference_index:
        Event index of the reference event
    number_of_events:
        Total number of events considered
    nmt:
        Number of elements of the moment tensor

    Returns
    -------
    :class:``numpy.ndarray``
        ``(nmt, nmt*number_of_events)`` identity matrix with ones at inidices
        corresponding to event columns and mt_element rows
    """
    lines = np.zeros((nmt, nmt * number_of_events))
    lines[:, nmt * reference_index : nmt * reference_index + nmt] = np.eye(nmt)
    return np.array(lines)


def _reference_mt_data_vector(mt: core.MT, constraint: str) -> np.ndarray:
    """
    Column vector holding the reference moment tensor

    Parameters
    ----------
    mt:
        Tuple containing reference moment tensor
    constraint:
        Which constraint to apply to the moment tensor

    Returns
    -------
    :class:``numpy.ndarray``
        Column vector of the reference moment tensor

    Warns
    -----
    When a deviatoric constraint is supplied, but `mt` is > 1% isotropic
    """

    moment = relmtmt.moment_of_tensor(relmtmt.mt_array(mt))
    if constraint == "none":
        return np.array(mt)[:, np.newaxis]
    elif constraint == "deviatoric":
        if (iso := np.abs(np.sum([mt.nn, mt.ee, mt.dd])) / moment) > 0.01:
            logger.warning(
                "Reference moment tensor is not deviatoric ({:.0f}% isotorpic)".format(
                    iso * 100
                )
            )
        return np.array([mt.nn, mt.ee, mt.ne, mt.nd, mt.ed])[:, np.newaxis]
    else:
        raise ValueError(
            f"'contraint' must be 'none' or 'deviatoric', not: '{constraint}'"
        )


def condition_homogenous_matrix_by_norm(
    mat: np.ndarray, n_homogenous: int | None = None
) -> np.ndarray:
    """
    Weights to condition the homogenous part of the amplitude matrix

    Each weight is the inverse L2-norm of the non-zero values of each equation

    Parameters
    ----------
    mat:
        ``(N, ev*nmt)`` amplitude matrix
    n_homogenous:
        Number of rows that constitute the homogenous part of the matrix. If
        given, only produce weights for the first 'n_homogenous' lines of the
        matrix.

    Returns
    -------
    norms: :class:`numpy.ndarray`
        ``(1, ev*nmt)`` column vector of equation normalization factors
    """

    if n_homogenous is None:
        n_homogenous = mat.shape[0]

    factors = np.ones(mat.shape[0])
    for n, line in enumerate(mat[:n_homogenous, :]):
        norm = 1 / np.linalg.norm(line[line.nonzero()])
        if ~np.isfinite(norm):
            logger.warning(f"Non finite norm for Eq. {n}. Setting to 0.")
            norm = 0.0

        factors[n] = norm

    # Column vector multiplies as expected with matrix rows
    return factors[:, np.newaxis]


def norm_event_median_amplitude(
    mat: NDArray, nmt: int, n_homogenous: int | None = None
) -> NDArray:
    """
    Factors to normalize amplitude matrix by median value per event

    Weight is the inverse median of all non-zero absolute vlues

    Parameters
    ----------
    mat:
        ``(N, ev*nmt)`` amplitude matrix
    nmt:
        Number of free moment tensor paramters
    n_homogenous:
        Homogenous equations (=number of amplitude observations) in the matrix.
        If given, only consider the first 'n_homogenous' lines in the matrix to
        estimate median amplitude

    Returns
    -------
    norms: :class:`numpy.ndarray`
        ``(1, ev*nmt)`` column vector of event norms

    Raises
    ------
    IndexError:
        If number of columns in matrix does not match 'nmt'
    """

    # Check integrity
    ncol = mat.shape[1]
    if ncol % nmt != 0:
        msg = f"Number of columns ({ncol}) is must be devisable by "
        msg += f"'nmt' ({nmt})"
        raise IndexError(msg)

    if n_homogenous is None:
        n_homogenous = mat.shape[0]

    # Get number of events
    nev = int(mat.shape[1] / nmt)
    norms = np.ones(ncol)

    # Iterate over events
    for iev in range(nev):
        # Amplitude values per event
        amps = mat[:n_homogenous, nmt * iev : nmt * (iev + 1)]

        # Weight is inverse median of non-zero absolute amplitudes
        norm = 1 / np.median(np.abs(amps[amps != 0]))

        if ~np.isfinite(norm):
            logger.warning(
                f"Encountered non-finite norm {norm} for event {iev}. Setting to zero."
            )
            norm = 0

        norms[nmt * iev : nmt * iev + nmt] = norm

    return norms


def solve_lsmr(
    A: np.ndarray, b: np.ndarray, ev_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve linear system using :func:`scipy.sparse.linalg.lsmr`

    Parameters
    ----------
    A:
        Matrix A in the linear system Ax = b of shape ``(M, N)``
    b:
        Data vector  b ``(1,M)`` in the linear system Ax = b
    ev_scale:
        Event-wise condition factors ``(1,M)`` A was devided by

    Returns
    -------
    x: :class:`numpy.ndarray`
        The resulting model vector
    eps: :class:`numpy.ndarray`
        Residuals Am - b -> 0. Residuals are computed before multiplying `m`
        with `ev_scale`
    """

    logger.info("Running lsmr...")
    m, istop, itn, *_ = lsmr(A, b, btol=1e-8, atol=1e-8, maxiter=10000)
    logger.info(f"lsmr ended after {itn} iterations with exit code {istop}")

    # Residuals
    # eps = A[:, :] * m
    eps = A @ m - b.flatten()  # make b 1-dimensional

    return m * ev_scale, eps


def unpack_resiudals(
    residuals: np.ndarray, p_lines: int, ref_lines: int, nmt: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split residual vector into P-, S-, and reference MT residuals

    Parameters
    ----------
    residuals:
        Reisdual vector
    p_lines:
        Number of equations with P-information
    ref_lines:
        Number of equations with reference MT information
    nmt:
        Number of elements that parameterize an MT

    Returns
    -------
    p_residuals, s_residuals, ref_residuals: :class:`numpy.ndarray`
        Vector contianing the residuals of P-, S-, and Reference residuals
    """

    residuals = np.array(residuals)

    peps = residuals[:p_lines]  #  P-amplitude ...
    seps = residuals[p_lines : -nmt * ref_lines].reshape(-1, 2)  # S-amplitude ...
    reps = residuals[-nmt * ref_lines :]  # Reference residuals

    return peps, seps, reps


def bootstrap_lsmr(
    A: np.ndarray,
    b: np.ndarray,
    ev_scale: float,
    peq: int,
    seq: int,
    bootstrap_samples: int,
    seed: int = 0,
    ncpu: int = 1,
) -> np.ndarray:
    """
    Draw bootstrap samples from A and solve the modified linear system Am = b

    Rows of A a drawn with repetion, where:
    - Rows representing P observations are drawn individually.
    - Rows representing S observations are drawn as pairs that contain the same
    Bacb and Bacb relative amplitudes.
    - Rows representing the reference MT are always included

    Parameters
    ----------
    A:
        ``(N, M)`` martix of the linear system Am = b
    b:
        ``(1, N)`` solution vector of the linear system Am = b
    ev_scale:
        ``(1, N)`` event-wise condition factors A was devided by
    peq, seq:
        Number of equations in A representing P and S wave constraints, respectivley
    bootstrap_samples:
        Number of random realizations of the linear system
    seed:
        Random number seed supplied to np.random.default_rng
    ncpu:
        Number of parallel process to launch

    Returns
    -------
    ``(bootstrap_samples * M)`` bootstrap model vectors
    """

    def run_lsmr(A, b):
        m, istop, *_ = lsmr(A, b, btol=1e-8, atol=1e-8, maxiter=10000)
        if istop <= 2:
            return m * ev_scale
        else:
            return np.full_like(m, np.nan)

    rng = np.random.default_rng(seed)

    n_homogenous = peq + seq

    # Always draw the inhomogenous equations
    samples_i = np.array((np.arange(n_homogenous, A.shape[0]),) * bootstrap_samples)

    # Bootstrap the homogenous part
    # Draw single P
    samples_p = rng.choice(peq, size=(bootstrap_samples, peq))

    # Only draw every 2nd S equation...
    samples_s1 = rng.choice(
        range(peq, peq + seq, 2), size=(bootstrap_samples, int(seq / 2))
    )

    # ... and it's pair
    samples_s2 = samples_s1 + 1

    # Stack P, S, and Reference equations.
    # Remeber we iterate over over first dimension, so rows in A are here
    # stored in 2nd dimension
    samples = np.hstack((samples_p, samples_s1, samples_s2, samples_i))

    m_boot = np.full((bootstrap_samples, A.shape[1]), np.nan)

    if ncpu <= 1:
        for n, isamp in enumerate(samples):
            logger.debug(f"Bootstrap: {n}")
            m_boot[n, :] = run_lsmr(A[isamp, :], b[isamp, :])
    else:
        raise NotImplementedError("Parallel processesing needs to be implemented")

    return m_boot


def solve_irls_sparse(
    G: spmatrix,
    d: np.ndarray,
    tolerance: float = 1e-5,
    eps: float = 1e-6,
    efac: float = 1.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively reweighted least squares for sparse matrices

    Solves for m in d = Gm using :func:`scipy.sparse.linalg.lsmr` and iteratiley re-scaling d

    Parameters
    ----------
    G:
        Matrix G in the linear system
    d:
        Vector d in the linear system (pair-wise time shifts)
    tolerance:
        Average absolute change in model update mean(abs(m - m0)) to stop
        iteration.  Should fall below sampling intervall (Units of m)
    eps:
        Truncate smallest residuals to this value to stabilize 1/residual
        weighting (Units of d)
    efac:
        Multiply epsilon by this value when iterative solution stagnates

    Returns
    -------
    m : :class:`numpy.ndarray`
        Model vector (absolute time shifts per trace)
    r : :class:`numpy.ndarray`
        Data residuals (unaccounted pair wise time shifts)
    """

    # First iteration solve using least squares.
    m = lsmr(G, d)[0]

    tol = np.inf
    it = 0
    while tol > tolerance:
        tol1 = tol
        m0 = m
        it = it + 1

        # Compute residual vector, and take those near zero values
        # to epsilon.
        res = np.abs(G @ m0 - d)
        res[res < eps] = eps

        # Normalize by max value and reciprocate. Take sqrt so as to use the
        # LSQR function that requires G not (G.T G)
        whgt = np.sqrt(1 / res)

        # Now convert to diagonal weighting elements according to 1-norm.
        # Extra square root is just to allow the problem to be solved using
        # the LSQR (least squares) division, i.e.:
        #
        # G'*R*(G*m-d)=0 or
        # G'*sqrt(R)'*(sqrt(R)*G*m-sqrt(R)*d) = B'*(B*m-d0) = 0.
        #
        # Thus we solve a modified system where
        # A-->B=sqrt(R)*A and
        # d-->d0=sqrt(R)*d.
        #
        # Create the equivalent least-squares problem.
        d0 = whgt * d

        # Broadcasting whgt across columns.
        G0 = G.copy()
        G0.data = G0.data * np.take(whgt, G0.indices)
        m, istop = lsmr(G0, d0)[:2]
        logger.debug("LSMR finished with exit code {:d}".format(istop))

        # Evaluate tolerance as mean np.abs(m-m0). This should go to zero as
        # solution to NXN subsystem is approached and should fall below sample
        # interval dt.
        tol = np.mean(np.abs(m - m0))
        logger.debug("Iteration {:d} misfit is {:3.1e}".format(it, tol))

        # Increase eps if solution starts to stagnate.
        if tol > tol1:
            eps = eps * efac
            logger.debug("eps increased to: {:3.1e}".format(eps))

    r = G @ m - d

    return m, r
