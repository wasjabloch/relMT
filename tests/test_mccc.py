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
"""
Test the MCCC functions
"""

import pytest
import numpy as np
from scipy.sparse import coo_matrix
from relmt.plot import section
import relmt.utils as utils
import matplotlib.pyplot as plt

import mccore as mcf
import relmt.mccc as mu

# TODO: How to do this proberly?
# Global input time shifts
dtins = np.array([1., 5., -4., 12., -12.])
dtins -= np.mean(dtins)  # zero mean

def p_wavelet():
    ns = 1024  # number of samples
    p = 60  # period length

    return np.vstack([utils.make_wavelet(ns, p, "sin", 30, dt, dt) for dt in dtins])


def s_wavelet():
    ns = 1024  # number of samples
    p = 60  # period length
    # fraction of the cosine signal on the E component
    cosef = [0.0, 0.2, 0.5, 0.8, 1.0]

    # Input 3-component waveforms (Z=zero, E=cos, N=rec)
    return np.vstack(
        [
            np.hstack(
                (
                    np.zeros(ns),
                    cf * utils.make_wavelet(ns, 2*p, "cos", 30, dt, dt)
                    + (1 - cf) * utils.make_wavelet(ns, p, "sin", 60, dt, dt),
                    cf * utils.make_wavelet(ns, 2*p, "sin", 60, dt, dt)
                    + (1 - cf) * utils.make_wavelet(ns, p, "cos", 30, dt, dt),
                )
            )
            for dt, cf in zip(dtins, cosef)
        ]
    )


def test_mccc_ppf(iplot=False):

    ds = 1  # sampling intervall

    wvin = p_wavelet()

    if iplot:
        plt.ion()
        _, ax = section(wvin)
        ax.axvline(wvin.shape[1] / 2)

    # rowi, coli, valu are indices and values of non-zero elements in time difference matrix A
    # cc are maximal cross correlation coefficient pairs.
    # dd is matrix of optimal lag time pairs.
    rowi, coli, valu, dd, cc = mcf.mccc_ppf(wvin.T, ds, 20, 1)

    A = coo_matrix((valu, (rowi, coli)), dtype=np.float64).tocsc()
    dtouts = mu.irls_sparse(A, dd, 1e-4)

    if iplot:
        wvout = mu.shift(wvin, 1, -dtouts)
        _, ax = section(wvout)
        ax.axvline(wvin.shape[1] / 2)

    # Are the time shifts recovered?
    assert dtins == pytest.approx(dtouts, 1e-4)


    # TODO: understand why these test using the mccc cc fail
    # Are waveforms identical, except for time shift?
    # assert cc[np.triu_indices(5, 1)] == pytest.approx(1.0, 5e-2)
    # Did no cycle skipping occurr?
    # assert all(cc[np.triu_indices(5, 1)] > 0)


def test_mccc_ssf(iplot=False):

    wvin = s_wavelet()

    ds = 1  # sampling intervall

    if iplot:
        plt.ion()
        _, ax = section(wvin)
        ax.axvline(wvin.shape[1] / 2)

    rowi, coli, valu, dd, cc = mcf.mccc_ssf(wvin.T, ds, 20, 1)

    A = coo_matrix((valu, (rowi, coli)), dtype=np.float64).tocsc()
    dtouts = mu.irls_sparse(A, dd, 1e-4)

    if iplot:
        wvout = mu.shift(wvin, 1, -dtouts)
        _, ax = section(wvout)
        ax.axvline(wvin.shape[1] / 2)

    # Are the time shifts recovered?
    assert dtins == pytest.approx(dtouts, 1e-4)


def test_linshift_p(iplot=False):
    wvin = p_wavelet()
    dt = 1

    dtouts = mu.pca_align(wvin, dt, "P", 50, 0)

    if iplot:
        wvout = mu.shift(wvin, 1, -dtouts)
        _, ax = section(wvout)
        ax.axvline(wvin.shape[1] / 2)

    assert pytest.approx(dtouts) == dtins


def test_linshift_s(iplot=False):
    wvin = s_wavelet()
    dt = 1

    dtouts = mu.pca_align(wvin, dt, "S", 200, 1e-9)

    if iplot:
        wvout = mu.shift(wvin, 1, -dtouts)
        _, ax = section(wvout)
        ax.axvline(wvin.shape[1] / 2)

    assert pytest.approx(dtouts) == dtins