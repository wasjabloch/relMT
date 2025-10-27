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
import math
from relmt import core, ls


logger = core.register_logger(__name__)


def mt_array(mt: core.MT | np.ndarray) -> np.ndarray:
    """Return 3x3 array representation of moment tensor"""
    return np.array(
        [[mt[0], mt[3], mt[4]], [mt[3], mt[1], mt[5]], [mt[4], mt[5], mt[2]]]
    )


def mt_tuple(mt_arr: np.ndarray) -> core.MT:
    """Return tuple representation of 3x3 moment tensor array"""
    return core.MT(
        mt_arr[0, 0],
        mt_arr[1, 1],
        mt_arr[2, 2],
        mt_arr[0, 1],
        mt_arr[0, 2],
        mt_arr[1, 2],
    )


def mt_tuples(mt_vec: np.ndarray, mt_constraint: str) -> list[core.MT]:
    """Return tuple representations of moment tensor vector."""

    mt_elements = ls.mt_elements(mt_constraint)

    if len(mt_vec) % mt_elements != 0:
        msg = f"length of mt_vec ({len(mt_vec)} not devisiable by "
        msg += f"'mt_elements' ({mt_elements})"
        raise IndexError(msg)

    mts = []
    for iev in range(int(len(mt_vec) / mt_elements)):

        # First index of MT corresoponding to iev
        i0 = iev * mt_elements

        if mt_elements == 6:
            mt = core.MT(*mt_vec[i0 : i0 + mt_elements])

        elif mt_elements == 5:
            mt = core.MT(
                mt_vec[i0],
                mt_vec[i0 + 1],
                -mt_vec[i0] - mt_vec[i0 + 1],
                mt_vec[i0 + 2],
                mt_vec[i0 + 3],
                mt_vec[i0 + 4],
            )

        mts.append(mt)
    return mts


def moment_of_tensor(mt_arr: np.ndarray) -> float:
    """Seismic moment of a moment tensor"""
    return np.linalg.norm(mt_arr) / np.sqrt(2)


def moment_of_vector(m: np.ndarray | core.MT) -> float | np.ndarray:
    """Seismic moment of tensor in 6-element vector notation

    For arrays, sum moment along last dimension and return array of moments
    """

    try:
        return np.sqrt(
            m[..., 0] ** 2
            + m[..., 1] ** 2
            + m[..., 2] ** 2
            + 2 * m[..., 3] ** 2
            + 2 * m[..., 4] ** 2
            + 2 * m[..., 5] ** 2
        ) / np.sqrt(2)
    except TypeError:
        return np.sqrt(
            m[0] ** 2
            + m[1] ** 2
            + m[2] ** 2
            + 2 * m[3] ** 2
            + 2 * m[4] ** 2
            + 2 * m[5] ** 2
        ) / np.sqrt(2)


def moment_of_magnitude(magnitude: float) -> float:
    """Seismic moment of a magnitude"""
    return 10 ** (1.5 * (magnitude + 10.7)) * 1e-7  # in Nm


def magnitude_of_moment(moment: float) -> float:
    """Magnitude of a seismic moment"""
    return np.log10(moment * 1e7) / 1.5 - 10.7


def mean_moment(mts: list[core.MT]) -> float:
    """Mean seismic moment of list of moment tensors"""
    return sum((moment_of_tensor(mt_array(mt)) for mt in mts)) / len(mts)


def p_radiation(
    M: np.ndarray,
    azi: float,
    plu: float,
    dist: float,
    rho: float,
    alpha: float,
    only_first: bool = False,
) -> np.ndarray:
    """P radiation pattern of a moment tensor M

    Parameters
    ----------
    M:
        ``(3,3)`` moment tensor (Nm)
    azi:
        Source to receiver azimuth in degree E of N
    inc:
        Source to receiver ray plunge in degreen down from horizontal
    dist:
        Source to Receiver distance (m)
    rho:
        Density of the medium (kg m^-3)
    alpha:
        P-wave velocity of the medim (m/s)
    only_first:
        If True, only return the first component of the displacement vector

    Returns
    -------
    ``(3,)`` (or ``(1,)`` when `only_first=True`) displacement vector at receiver
    """

    g = ls.gamma(azi, plu)
    gMg = g @ M @ g
    u = gMg * g / (4.0 * np.pi * rho * alpha**3.0 * dist)
    return u


def s_radiation(
    M: np.ndarray, azi: float, plu: float, dist: float, rho: float, beta: float
) -> np.ndarray:
    """
    S radiation pattern of a moment tensor M

    Parameters
    ----------
    M:
        ``(3, 3)`` moment tensor (Nm)
    azi:
        Source to receiver azimuth in degree E of N
    inc:
        Source to receiver ray plunge in degreen down from horizontal
    dist:
        Source to Receiver distance (m)
    rho:
        Density of the medium (kg m^-3)
    beta:
        S-wave velocity of the medim (m/s)

    Returns
    -------
    ``(3,)`` displacement vector at receiver
    """

    g = ls.gamma(azi, plu)
    gMg = g @ M @ g
    u = ((M @ g) - gMg * g) / (4.0 * np.pi * rho * beta**3.0 * dist)

    return u


def norm_rms(
    mt_list: list[core.MT] | dict[int, list[core.MT]], mt_true: core.MT | None = None
) -> float | dict[int, float]:
    """Normalized RMS misfit of MTs

    Parameters
    ----------
    mt_list:
        List of MT observations to compare, or dict of lists
    mt_true:
        A true MT from which to compute deviation. It None, use mean of `mt_list`

    Returns
    -------
    normalized root-mean-square misfit
    """

    if type(mt_list) == dict:
        return {evn: norm_rms(mtl) for evn, mtl in mt_list.items()}

    mt_array = np.array(mt_list)

    mean = mt_array.mean(axis=0)
    m0 = mean_moment(mt_list)
    if mt_true is not None:
        mean = np.array(mt_true)
        m0 = moment_of_vector(mt_true)

    return np.sqrt(np.sum((mt_array - mean) ** 2) / mt_array.size) / m0


def kagan_rms(
    mt_list: list[core.MT] | dict[int, list[core.MT]], mt_true: core.MT | None = None
) -> float | dict[int, float]:
    """RMS of the kagan angles between MTs

    Parameters
    ----------
    mt_list:
        List of MT observations to compare, or dict of lists
    mt_true:
        A true MT from which to compute deviation. It None, use mean of `mt_list`

    Returns
    -------
    RMS of the kagan angles
    """

    if type(mt_list) == dict:
        return {evn: norm_rms(mtl) for evn, mtl in mt_list.items()}

    mt_array = np.array(mt_list)

    mean = mt_array.mean(axis=0)
    if mt_true is not None:
        mean = np.array(mt_true)

    kagans = np.array([kagan_angle(mean, this) for this in mt_array])

    return np.sqrt(np.sum(kagans**2) / mt_array.size)


_pbt2tpb = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))


def _tpb2q(t, p, b):
    eps = 0.001
    tqw = 1.0 + t[0] + p[1] + b[2]
    tqx = 1.0 + t[0] - p[1] - b[2]
    tqy = 1.0 - t[0] + p[1] - b[2]
    tqz = 1.0 - t[0] - p[1] + b[2]

    q = np.zeros(4)
    if tqw > eps:
        q[0] = 0.5 * math.sqrt(tqw)
        q[1:] = p[2] - b[1], b[0] - t[2], t[1] - p[0]
    elif tqx > eps:
        q[0] = 0.5 * math.sqrt(tqx)
        q[1:] = p[2] - b[1], p[0] + t[1], b[0] + t[2]
    elif tqy > eps:
        q[0] = 0.5 * math.sqrt(tqy)
        q[1:] = b[0] - t[2], p[0] + t[1], b[1] + p[2]
    elif tqz > eps:
        q[0] = 0.5 * math.sqrt(tqz)
        q[1:] = t[1] - p[0], b[0] + t[2], b[1] + p[2]
    else:
        raise Exception("should not reach this line, check theory!")
        # q[0] = max(0.5 * math.sqrt(tqx), eps)
        # q[1:] = p[2] - b[1], p[0] + t[1], b[0] + t[2]

    q[1:] /= 4.0 * q[0]

    q /= math.sqrt(np.sum(q**2))

    return q


def kagan_angle(mt1: core.MT, mt2: core.MT) -> float:
    """
    Given two moment tensors, return the Kagan angle in degrees.

    ..note: Implementation curtesy of :func:`pyrocko.moment_tensor.kagan_angle`

    After Kagan (1991) and Tape & Tape (2012).
    """

    mta1 = mt_array(mt1)
    mta2 = mt_array(mt2)

    eve1 = np.linalg.eigh(mta1)[1]
    eve2 = np.linalg.eigh(mta2)[1]

    if np.linalg.det(eve1) < 0.0:
        eve1 *= -1.0
    if np.linalg.det(eve2) < 0.0:
        eve2 *= -1.0

    ai = np.dot(_pbt2tpb, eve1.T)
    aj = np.dot(_pbt2tpb, eve2.T)

    u = np.dot(ai, aj.T)

    tk, pk, bk = u

    qk = _tpb2q(tk, pk, bk)

    return 2.0 * math.acos(np.max(np.abs(qk))) * 180 / np.pi


def rtf2ned(
    mrr: float, mtt: float, mff: float, mrt: float, mrf: float, mtf: float
) -> tuple[float, float, float, float, float, float]:
    """
    Convert moment tensor in `r` (up), `t` (south), `p` (east) coordinates to
    north-east-down
    """
    #      mnn, mee, mdd,  mne, mnd,  med
    return core.MT(mtt, mff, mrr, -mtf, mrt, -mrf)


def ned2rtf(
    mnn: float, mee: float, mdd: float, mne: float, mnd: float, med: float
) -> tuple[float, float, float, float, float, float]:
    """
    Convert moment tensor in north-east-down coordinates to
    `r` (up), `t` (south), `f` (east)
    """
    #      mrr, mtt, mff, mrt,  mrf,  mtf
    return mdd, mnn, mee, mnd, -med, -mne


def correlation(M1: core.MT, M2: core.MT) -> tuple[float, float]:
    """
    P and S correlation coefficients between two moment tensors

    Implements Equation 5 of Kuge and Kawakatsu (1993, PEPI)

    Parameters
    ----------
    M1, M2:
        The two moment tensors to compare

    Returns
    -------
    Correlation coefficients for the P-SV and SH system
    """

    m1 = ned2rtf(*M1)
    m2 = ned2rtf(*M2)

    # M = Mrr Mtt Mff Mrt Mrf Mtf
    #     0   1   2   3   4   5

    def I(M):
        return (M[0] + M[1] + M[2]) / 3

    def C(M):
        return (M[1] + M[2] - 2 * M[0]) / 3

    def D(M):
        return (M[1] - M[2]) / 2

    def A(M, l, m):
        if l == 0:
            return 2 * np.pi * I(M)
        if l == 2:
            if m == 0:
                return -2 * np.sqrt(np.pi / 5) * C(M)
            if m == 1:
                return -2 * np.sqrt(2 * np.pi / 15) + M[3] + 1j * M[4]
            if m == -1:
                return -2 * np.sqrt(2 * np.pi / 15) - M[3] + 1j * M[4]
            if m == 2:
                return 2 * np.sqrt(2 * np.pi / 15) * (D(M) + 1j * M[5])
            if m == -2:
                return 2 * np.sqrt(2 * np.pi / 15) * (D(M) - 1j * M[5])

    def B(M, l, m):
        if l == 0 or m == 0:
            return 0.0
        if l == 2:
            return A(M, l, m)

    def doublesum(first, second, coefficient):
        return np.sum(
            [
                coefficient(first, l, m) * np.conjugate(coefficient(second, l, m))
                for l in [0, 2]  # Not evaluated at l=1
                for m in range(-l, l + 1)
            ]
        )

    def plusminus_one(number):
        return min(1, max(-1, number))

    eta_p = np.real(
        doublesum(m1, m2, A)
        / np.sqrt(doublesum(m1, m1, A))
        / np.sqrt(doublesum(m2, m2, A))
    )

    eta_h = np.real(
        doublesum(m1, m2, B)
        / np.sqrt(doublesum(m1, m1, B))
        / np.sqrt(doublesum(m2, m2, B))
    )

    return plusminus_one(eta_p), plusminus_one(eta_h)


def norm_scalar_product(M1: core.MT, M2: core.MT) -> float:
    """
    Return the normalized scalar product of two moment tensors

    See Michael (1987, JGR) Eq. 1 for motivation
    """

    m1 = mt_array(M1)
    m2 = mt_array(M2)

    return np.sum(m1 * m2) / (np.sqrt(np.sum(m1 * m1)) * np.sqrt(np.sum(m2 * m2)))
