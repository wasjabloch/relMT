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
import logging
from relmt.utils import logsh, differentiate, shift

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logsh)

def irls_sparse(A, d, tolerance=1e-5, eps=1e-6, efac=1.3):

    import scipy.sparse as ss

    # Initialization.
    m = np.zeros(A.shape[1])

    # First iteration solve using least squares.
    m, _, _, _ = ss.linalg.lsmr(A, d)[:4]

    # While loop with convergence criterion.
    tol = 1e3
    # print('EPS now at', eps)
    k = 0
    #    while tol > tolerance and k < 20:
    #    while tol > tolerance and k < 50:
    while tol > tolerance:
        tol1 = tol
        m0 = m
        k = k + 1

        # Compute residual vector, and take those near zero values
        # to epsilon. Then normalize by max value and reciprocate. Take
        # sqrt so as to use the LSQR function that requires A not (A.T A)
        res = np.abs(A @ m0 - d)
        ix = np.where(res < eps)
        res[ix] = eps
        resr = np.sqrt(1 / res)

        # Now convert to diagonal weighting elements according to 1-norm.
        # Extra square root is just to allow the problem to be solved using
        # the LSQR (least squares) division, ie A'*R*(A*m-d)=0 or
        # A'*sqrt(R)'*(sqrt(R)*A*m-sqrt(R)*d)=B'*(B*m-d0)=0. Thus just
        # solve a modified system where A-->B=sqrt(R)*A and d-->d0=sqrt(R)*d.
        # Create equivalent least-squares problem.
        d0 = resr * d

        # Broadcasting resr across columns (use tocsr to broadcast along rows).
        A0 = A.copy()
        A0.data = A0.data * np.take(resr, A0.indices)
        m, istop, _, _ = ss.linalg.lsmr(A0, d0)[:4]
        logger.info("LSMR finished with exit code {:d}".format(istop))

        # Evaluate tolerance as mean np.abs(m-m0). This should go to zero as
        # solution to NXN subsystem is approached and should fall below sample
        # interval dt.
        tol = np.mean(np.abs(m - m0))
        logger.info("Iteration {:d} misfit is {:3.1e}".format(k, tol))

        # Increase eps if solution starts to stagnate.
        if tol > tol1:
            eps = eps * efac
            logger.info("EPS increased to: {:3.1e}".format(eps))

    return m


def pca_align(scomp0, dt, phase="P", nit=200, etol=0):

    ns = scomp0.shape[0]
    scomp = np.zeros(scomp0.shape)
    tshiftl = np.zeros(ns)
    tshift = np.zeros(ns)

    if phase == "P":
        ip = True

        def phi(s, ns):
            return s[0] ** 2 / ns

    elif phase == "S":
        ip = False

        def phi(s, _):
            return s[2] / (s[0] + s[1])

    else:
        msg = f"unrecognized phase: {phase}. Must be 'P' or 'S'"
        raise ValueError(msg)

    # Remove mean and normalize.
    for ix in np.arange(ns):
        scomp[ix, :] = scomp0[ix,] - np.mean(scomp0[ix,])
        scomp[ix, :] = scomp[ix,] / np.linalg.norm(scomp[ix,])
    scomp1 = np.zeros(scomp.shape)

    # Decompose seismograms into principal components
    # U are the "principal seismograms"
    # s are the singular values ("relative importance")
    # s * V are the contributions of U to each seismogram
    U, s, Vh = svd(scomp)

    # Delta epsilon (objective function)
    de = 1.0

    # Concentrate energy on first pricipal component.
    eob_old = phi(s, ns)

    logger.info("Singular values are:")
    logger.info(" ".join(["{:1.1e}".format(v) for v in s]))
    logger.info(f"Starting pca_align for {phase} Phase")
    logger.info("Objective Function: {:1.1e}".format(eob_old))

    iter = 0
    while de > etol and iter < nit:

        # Keep running sum for total shift
        tshiftl = tshiftl - tshift
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
        scomp1d = (differentiate(scomp + scomp1, dt)) / 2

        # Create perturbation from higher PC's (>2), resembling derivative.
        scomp2N = scomp - scomp1

        # Compute estimate of shift and correlation coefficients.
        # (Eq. 15) Bostock et al. (2021, BSSA)
        for i in range(ns):
            tshift[i] = np.dot(scomp1d[i,], scomp2N[i,]) / np.dot(
                scomp1d[i,], scomp1d[i,]
            )

        # Apply zero sum constraint and shift
        tshift -= tshift.mean()
        scomp = shift(scomp, dt, tshift)

        # Re-computed SVD and compute new objective function
        U, s, Vh = svd(scomp)
        eob_new = phi(s, ns)

        de = eob_new - eob_old
        if not ip:
            de *= -1

        eob_old = eob_new

        logger.info(
            "Iteration #{:d} eobj: {:1.5f} (delta: {:1.3e})".format(iter, eob_new, de)
        )
        logger.info("Time shifts: " + " ".join(["{:1.2f}".format(v) for v in tshiftl]))

    logger.info("Finished after {:} iterations with eobj {:1.1e}".format(iter, eob_new))

    return tshiftl