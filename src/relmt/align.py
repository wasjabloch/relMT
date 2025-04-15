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
from scipy.sparse import coo_matrix
import logging
from relmt import utils, signal, core, ls
import mccore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def _test_phase_shape(phase, mtx):
    if phase not in ["P", "S"]:
        raise ValueError("Phase must be either 'P' or 'S'")

    if (phase == "P" and mtx.shape[1] < 2) or (phase == "S" and mtx.shape[1] < 3):
        msg = f"mtx contains only {mtx.shape[1]} events. "
        msg += "At least 2 are required for P-waves, 3 for S waves."
        raise IndexError(msg)


def mccc_align(
    mtx: np.ndarray,
    phase: str,
    sampling_rate: float,
    maxshift: float,
    ndec: int = 1,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute time shifts that align the seismogram matrix

    Applies the multi-channel cross-correlation method (Bostock et al. 2021, BSSA)

    Parameters
    ----------
    mtx:
        Waveform matrix of shape ``(events, samples)``, with time aligned along
        last axis
    phase:
        Seismic phase ('P' or 'S')
    sampling_rate:
        Sampling rate of the wavefrom matrix (Hertz)
    maxshift:
        Maximum shift to test (seconds)
    ndec:
        Test correlation function at every ndec'th point
    verbose:
        Print diagnostic output

    Returns
    -------
    time_shifts: :class:`numpy.ndarray`
        Zero-mean time shifts of shape ``(events,)`` that maximize cc
    cc: :class:`numpy.ndarray`
        Nominal correlation coefficients between events at optimal time lag
    dd: :class:`numpy.ndarray`
        Pairwise differential arrival times
    dd_res: :class:`numpy.ndarray`
        Resiudal of pairwise differential arrival times

    Raises
    ------
    IndexError:
        When mtx contains less than 2 events for P waves, or less than 3
        for S waves
    ValueError:
        When phase is not 'P' or 'S'
    """
    _test_phase_shape(phase, mtx)
    sampling_interval = 1 / sampling_rate

    if phase == "P":
        fun = mccore.mccc_ppf
    elif phase == "S":
        fun = mccore.mccc_ssf0

    # Data, sampling_interval, maxlag, ndec
    rowi, coli, valu, dd, cc = fun(mtx.T, sampling_interval, maxshift, ndec, verbose)

    # Make cc cubic (S) or symmetric square (P) matrix
    if phase == "S":
        cc = utils.reshape_ccvec(cc, mtx.shape[0])
    elif phase == "P":
        cc = cc + cc.T

    A = coo_matrix((valu, (rowi, coli)), dtype=np.float64).tocsc()

    dt, dd_res = ls.solve_irls_sparse(A, dd)
    return dt, cc, dd, dd_res


def pca_align(
    mtx: np.ndarray,
    sampling_rate: float,
    phase: str,
    iterations: int = 200,
    dphi: float = 0,
    dtime: float = 0,
) -> tuple[np.ndarray, float]:
    """
    Align waveforms by principal component analysis (Bostock et al. 2021, BSSA).

    For P-waves, we maximize the first of the singular values :math:`s` of
    :math:`n` waveforms:

    .. math::
        \phi = s_0^2 / n

    For S-waves, we maximize the sum of the first two singular values:

    .. math::
        \phi = 1 - s_2 / (s_0 + s_1),

    where :math:`\phi` ranges from `0` (worst) to `1` (best)

    Parameters
    ----------
    mtx:
        Waveform matrix of shape ``(events, samples)``, with time aligned along
        last axis
    sampling_rate:
        Sampling rate of the wavefrom matrix (Hertz)
    phase:
        Which seismic phase to consider ('P' or 'S')
    iterations:
        Maximum number of iterations
    dphi:
        Minimum change in objective function to stop iteration
    dtime:
        Smallest maximal time shift to stop iteration

    Returns
    -------
    time_shifts : :class:`numpy.ndarray`
        Zero-mean time shifts of shape ``(events,)`` that maximize phi
    phi : float
        Value of objective function

    Raises
    ------
    IndexError:
        When mtx contains less than 2 events for P waves, or less than 3
        for S waves
    ValueError:
        When phase is not 'P' or 'S'
    """

    _test_phase_shape(phase, mtx)
    ns = mtx.shape[0]
    tshift_tot = np.zeros(ns)
    tshift = np.zeros(ns)

    if phase == "P":
        ip = True

        def phi(s, ns):
            # worst 0 -> 1 best
            return s[0] ** 2 / ns

    elif phase == "S":
        ip = False

        def phi(s, _):
            # worst 0 -> 1 best
            return 1 - (s[2] / (s[0] + s[1]))

    # Remove mean and normalize.
    scomp = signal.norm_power(signal.demean(mtx))
    scomp1 = np.zeros(scomp.shape)

    # Decompose seismograms into principal components
    # s * U are the contributions of U to each seismogram
    # s are the singular values ("relative importance")
    # Vh are the "principal seismograms"
    U, s, Vh = svd(scomp)

    # Delta epsilon (objective function)
    this_dphi = np.inf

    # Concentrate energy on first principal component.
    phi_old = phi(s, ns)

    logger.info("Singular values are:")
    logger.info(" ".join(["{:1.1e}".format(v) for v in s]))
    logger.info(f"Starting pca_align for {phase} Phase")
    logger.info("Objective Function: {:1.1e}".format(phi_old))

    iter = 0
    logger.info(f"{iter == 0}, {dtime < max(abs(tshift))}")
    while (
        iter < iterations
        and dphi < this_dphi
        and (iter == 0 or dtime < max(abs(tshift)))
    ):

        # Keep running sum for total shift
        tshift_tot = tshift_tot - tshift
        iter += 1

        if ip:
            # Create rank 1 ...
            scomp1 = s[0] * np.outer(U[:, 0], Vh[0, :])
        else:
            # ... rank 2 approximation from singular values
            scomp1 = np.zeros(scomp.shape)
            for i in np.arange(2):
                scomp1 += s[i] * np.outer(U[:, i], Vh[i, :])

        tshift = np.zeros((ns))

        # Differentiate waveforms: 2 options - differentiate full waveform or its
        # rank 2 approximation. The first can converge more slowly and sometimes
        # achieves better objective but not always. Split difference by averaging the two.
        scomp1d = (signal.differentiate(scomp + scomp1, sampling_rate)) / 2

        # Create perturbation from higher PC's (>2), resembling derivative.
        scomp2N = scomp - scomp1

        # Compute estimate of shift and correlation coefficients.
        # (Eq. 15) Bostock et al. (2021, BSSA)
        for i in range(ns):
            tshift[i] = np.dot(scomp1d[i,], scomp2N[i,]) / np.dot(
                scomp1d[
                    i,
                ],
                scomp1d[
                    i,
                ],
            )

        # Apply zero sum constraint and shift
        tshift -= tshift.mean()
        scomp = signal.shift(scomp, tshift, sampling_rate)

        # Re-computed SVD and compute new objective function
        U, s, Vh = svd(scomp)
        phi_new = phi(s, ns)

        this_dphi = phi_new - phi_old

        phi_old = phi_new

        logger.info(
            "Iteration #{:d} phi: {:1.3e} (dphi: {:1.3e})".format(
                iter, phi_new, this_dphi
            )
        )
        logger.debug(
            f"Time shifts (max={max(abs(tshift))}): "
            + " ".join(["{:1.3e}".format(v) for v in tshift_tot])
        )

    logger.info("Finished after {:} iterations with eobj {:1.3e}".format(iter, phi_old))

    return tshift_tot, phi_old
