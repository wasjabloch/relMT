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

"""Functions to estimate relative waveform amplitudes"""

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import svd, norm, solve
from relmt import core, signal, qc
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def pca_amplitude_2p(mtx_ab: np.ndarray) -> float:
    """
    Calculate the relative amplitude of a pair of P-waves

    The relative amplitudes :math:`A^{ab}` that predicts waveform :math:`u_a` from
    waveform :math:`u_b`, such that:

    .. math::
        A^{ab} u_b = u_a

    Parameters
    ----------
    mtx_ab:
        Waveform matrix of shape ``(2, samples)`` holding events `a` and `b`

    Returns
    -------
    Relative ampltiude between events a and b
    """

    # The line below here was
    _, _, Vh = svd(mtx_ab.T / np.max(abs(mtx_ab)), full_matrices=False)

    Aab = Vh[0, 0] / Vh[0, 1]

    return Aab


def p_misfit(mtx_ab: np.ndarray, Aab: float) -> float:
    """
    Misfit of the P waveform reconstruction

    The residual norm of the reconstruction devided by the norm of the predicted
    waveform :math:`u_a`:

    .. math::
        \Psi_P = || A^{ab} u_b - u_a || / || u_a ||

    Parameters
    ----------
    mtx_ab:
        Waveform matrix of shape ``(2, samples)`` holding events `a` and `b`
    Aab:
        Relative ampltiude between events `a` and `b`

    Returns
    -------
    Normalized reconstruction misfit
    """

    return norm(mtx_ab[0, :] - Aab * mtx_ab[1, :]) / norm(mtx_ab[0, :])


def pca_amplitude_3s(mtx_abc: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Relative amplitudes between triplet of S-waves.

    Given waveforms of three events, determine which two waveforms :math:`b` and
    :math:`c` are most different from each other and compute the relative
    amplitudes :math:`B^{abc}` and :math:`B^{acb}` that predict the third
    waveform :math:'u_a', such that:

    .. math::
        B^{abc} u_b + B^{acb} * u_c = u_a

    Parameters
    ----------
    mtx_abc:
        Waveform matrix of shape ``(3, samples)`` holding events `a`, `b` and
        `c`

    Returns
    -------
    Babc: float
        Relative ampltiude between event `a` and `b`, given the third event is `c`
    Bacb: float
        Relative ampltiude between event `a` and `c`, given the third event is `b`
    iord:  ndarray
        Resulting ``(3,)`` row indices into `mtx_abc` of events `a`, `b` and `c`
    """
    iord = order_by_ccsum(mtx_abc)

    _, ss, Vh = svd(mtx_abc[iord].T / np.max(abs(mtx_abc)), full_matrices=False)

    PCs = np.diag(ss) @ Vh
    A = PCs[0:2, 1:3]
    bb = PCs[0:2, 0]
    if bb[1] == 0:
        logger.warning("All wavefroms are similar, so treated with Bacb = 0")
        return bb[0], 0.0, iord
    else:
        try:
            Aabc = solve(A, bb)
        except LinAlgError as e:
            msg = f"Met error in SVD: {e.__repr__()}. Assuming: Babc = Bacb = 0.5"
            logger.warning(msg)
            return 0.5, 0.5, iord
        return Aabc[0], Aabc[1], iord


def s_misfit(mtx_abc: np.ndarray, Babc: float, Bacb: float) -> float:
    """
    Misfit of the S waveform reconstruction

    The residual norm of the reconstruction devided by the norm of the predicted
    waveform :math:`u_a`:

    .. math::
        \Psi_S = || B^{abc} u_b + B^{acb} u_c - u_a || / || u_a ||

    Parameters
    ----------
    mtx_abc:
        Waveform matrix of shape ``(3, samples)`` holding events `a`, `b` and
        `c`
    Babc:
        Relative ampltiude between event `a` and `b`, given the third event is `c`
    Bacb:
        Relative ampltiude between event `a` and `c`, given the third event is `b`

    Returns
    -------
    Normalized reconstruction misfit
    """

    return norm(Babc * mtx_abc[1, :] + Bacb * mtx_abc[2, :] - mtx_abc[0, :]) / norm(
        mtx_abc[0, :]
    )


def order_by_ccsum(mtx_abc: np.ndarray) -> np.ndarray:
    """
    Order waveforms by pairwise summed cross-correlation coefficients

    Parameters
    ----------
    mtx_abc:
        Waveform of shape ``(3, samples)`` of events `a`, `b`, and `c`

    Returns
    -------
    iord: :class:`~numpy.ndarray`
        Indices ``(3,) that order :math:`u_a, u_b, u_c` by cc
    """
    # Calculate cross-correlation coefficients between all pairs of waveforms
    ab = abs(signal.cc_coef(mtx_abc[0, :], mtx_abc[1, :]))
    ac = abs(signal.cc_coef(mtx_abc[0, :], mtx_abc[2, :]))
    bc = abs(signal.cc_coef(mtx_abc[1, :], mtx_abc[2, :]))

    # Re-order events based on waveform similarity such that evs B and C are most different
    iord = np.argsort([ab + ac, ab + bc, ac + bc])[::-1]
    return iord


def info(
    amplitudes: list[core.P_Amplitude_Ratio] | list[core.S_Amplitude_Ratios],
    width: int = 80,
):
    """Print statistics of amplitude list to screen"""

    ip, _ = qc._ps_amplitudes(amplitudes)

    if ip:
        stas, evs, b = map(
            list, zip(*[(amp.station, amp.event_a, amp.event_b) for amp in amplitudes])
        )
        ip = True
        print("P wave amplitdue observations")
    else:
        stas, evs, b, c = map(
            list,
            zip(
                *[
                    (amp.station, amp.event_a, amp.event_b, amp.event_c)
                    for amp in amplitudes
                ]
            ),
        )
        ip = False
        print("S wave amplitdue observations")

    # Join event oversvations
    evs.extend(b)

    if not ip:
        evs.extend(c)

    nsta, usta = zip(
        *sorted([(stas.count(sta), sta) for sta in set(stas)], reverse=True)
    )

    nev, uev = zip(*sorted([(evs.count(ev), ev) for ev in set(evs)], reverse=True))

    necha = len(str(nev[0]))
    nscha = len(str(nev[0]))
    evcha = max(len(str(ev)) for ev in uev)
    stcha = max([len(st) for st in usta])

    out = "Stations:\n"
    previous = 0
    blockwidth = stcha + nscha + 3
    for nb, (us, ns) in enumerate(zip(usta, nsta)):
        if (llength := (nb * blockwidth) % width) < previous:
            out += "\n"

        out += "{sname:{swidth}}: {snum:{nwdth}} ".format(
            sname=us, swidth=stcha, snum=ns, nwdth=nscha
        )

        previous = llength

    out += "\n\nEvents:\n"
    previous = 0
    blockwidth = evcha + necha + 3
    for nb, (us, ns) in enumerate(zip(uev, nev)):
        if (llength := (nb * blockwidth) % width) < previous:
            out += "\n"

        out += "{sname:{swidth}}: {snum:{nwdth}} ".format(
            sname=us, swidth=evcha, snum=ns, nwdth=necha
        )

        previous = llength

    print(out)
