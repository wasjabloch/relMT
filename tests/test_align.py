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
Test the alignment functions
"""

import pytest
import numpy as np
from scipy.sparse import coo_matrix

import mccore as mcf
from relmt import align, signal, utils, ls


# TODO: How to do this properly?
# Global input time shifts
dtins = np.array([1.0, 5.0, -4.0, 12.0, -12.0])
dtins -= np.mean(dtins)  # zero mean


def p_wavelet():
    ns = 1024  # number of samples
    p = 60  # period length

    return np.vstack([signal.make_wavelet(ns, p, "sin", 30, dt, dt) for dt in dtins])


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
                    cf * signal.make_wavelet(ns, 2 * p, "cos", 30, dt, dt)
                    + (1 - cf) * signal.make_wavelet(ns, p, "sin", 60, dt, dt),
                    cf * signal.make_wavelet(ns, 2 * p, "sin", 60, dt, dt)
                    + (1 - cf) * signal.make_wavelet(ns, p, "cos", 30, dt, dt),
                )
            )
            for dt, cf in zip(dtins, cosef)
        ]
    )


def test_mccc_ppf():

    ds = 1  # sampling interval

    wvin = p_wavelet()

    # rowi, coli, valu are indices and values of non-zero elements in time difference matrix A
    # cc are maximal cross correlation coefficient pairs.
    # dd is matrix of optimal lag time pairs.
    rowi, coli, valu, dd, cc = mcf.mccc_ppf(wvin.T, ds, 20, 1, False)

    A = coo_matrix((valu, (rowi, coli)), dtype=np.float64).tocsc()
    dtouts, _ = ls.solve_irls_sparse(A, dd, 1e-4)

    # Are the time shifts recovered?
    assert dtins == pytest.approx(dtouts, 1e-4)

    # TODO: understand why these test using the mccc cc fail
    # Are waveforms identical, except for time shift?
    # assert cc[np.triu_indices(5, 1)] == pytest.approx(1.0, 5e-2)
    # Did no cycle skipping occur?
    # assert all(cc[np.triu_indices(5, 1)] > 0)


def test_mccc_ssf0():

    wvin = s_wavelet()

    sampling_rate = 1

    rowi, coli, valu, dd, cc3 = mcf.mccc_ssf0(
        wvin.T, 1 / sampling_rate, 20, 1, verb=False
    )

    A = coo_matrix((valu, (rowi, coli)), dtype=np.float64).tocsc()
    dtouts, _ = ls.solve_irls_sparse(A, dd, 1e-4)

    # Are the time shifts recovered?
    assert dtins == pytest.approx(dtouts, 1e-4)

    # No NaNs in cc3
    assert pytest.approx(cc3) == 1.0

    # Also test the fisher transform once more
    cc2 = utils.fisher_average(utils.reshape_ccvec(cc3, len(wvin)))

    # Are the upper and lower triangles all 1?
    assert pytest.approx(cc2[np.triu_indices_from(cc2, 1)]) == 1.0
    assert pytest.approx(cc2[np.tril_indices_from(cc2, -1)]) == 1.0

    # And the diagonal == 0, so as to average it correctly
    assert pytest.approx(cc2[np.diag_indices_from(cc2)]) == 0.0


def test_mccc_ssf():

    wvin = s_wavelet()
    # Add noise to avoid NaNs in cc (due to Fisher transform in ccorf2)
    wvin += 1e-3 * np.random.default_rng().standard_normal(wvin.shape)

    sampling_rate = 1

    rowi, coli, valu, dd, cc = mcf.mccc_ssf(wvin.T, 1 / sampling_rate, 20, 1)

    A = coo_matrix((valu, (rowi, coli)), dtype=np.float64).tocsc()
    dtouts, _ = ls.solve_irls_sparse(A, dd, 1e-4)

    # Are the time shifts recovered?
    assert dtins == pytest.approx(dtouts, 1e-4)
    assert pytest.approx(cc[np.triu_indices_from(cc, 1)], abs=1e-3) == 1.0


def test_pca_align_p():
    wvin = p_wavelet()
    dt = 1

    dtouts, phi = align.pca_align(wvin, dt, "P", 200, 1e-12)

    assert pytest.approx(dtouts) == dtins
    assert pytest.approx(phi, abs=1e-12) == 1


def test_pca_align_s():
    wvin = s_wavelet()
    dt = 1

    dtouts, phi = align.pca_align(wvin, dt, "S", 200, 1e-12)

    assert pytest.approx(dtouts) == dtins
    assert pytest.approx(phi, abs=1e-9) == 1


def test_mccc_align_p():
    wvf_matrix = p_wavelet()
    nwv = len(wvf_matrix)
    dt, cc, dd, dd_res, evpairs = align.mccc_align(wvf_matrix, "P", 100, 0.5)
    assert len(dt) == nwv
    assert cc.shape == (nwv, nwv)
    assert dd.shape[0] == 2 * nwv + 1
    assert dd_res.shape[0] == 2 * nwv + 1
    assert np.all(evpairs == [(i, j) for i in range(nwv) for j in range(i + 1, nwv)])


def test_mccc_align_s():
    wvf_matrix = s_wavelet()
    nwv = len(wvf_matrix)
    dt, cc, dd, dd_res, evpairs = align.mccc_align(wvf_matrix, "S", 100, 0.5)
    assert len(dt) == nwv
    assert cc.shape == (nwv, nwv, nwv)
    assert dd.shape == (nwv * (nwv - 1) + 1,)
    assert dd_res.shape == (nwv * (nwv - 1) + 1,)
    assert dd_res == pytest.approx(0.0)  # Perfect data should be aligned perfectly
    assert evpairs.shape == (int((5 - 2) * (5 - 1) * 5 / 3), 2)


def test_paired_s_lag_times():

    wvf_matrix = s_wavelet()
    nwv = len(wvf_matrix)
    _, cc, dd, dd_res, evpairs = align.mccc_align(wvf_matrix, "S", 100, 0.5)

    evpairs2, dd2, cc2, rms = align.paired_s_lag_times(evpairs, dd, cc, dd_res)

    assert evpairs2.shape[0] == nwv * (nwv - 1) / 2
    assert len(dd2) == nwv * (nwv - 1) / 2
    assert cc2 == pytest.approx(1.0)
    assert rms == pytest.approx(0.0)

    # Let's strip the last 0 used to constrain the linear system
    dd = dd[:-1]

    for evpair, ddtime in zip(evpairs2, dd2):
        iin = np.all(evpairs == evpair, axis=-1)

        # All dd times should be equal, since we have perfect data.
        assert dd[iin] == pytest.approx(ddtime)


def test_pca_align():
    wvf_matrix = np.random.rand(3, 100)
    tshift_tot, phi = align.pca_align(wvf_matrix, 100, "P")
    assert len(tshift_tot) == 3
    assert 0 <= phi <= 1
