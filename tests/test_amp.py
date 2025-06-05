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
Test the amplitude functions
"""

from relmt import mt, signal, amp, core
import numpy as np

import pytest
from numpy.random import Generator, PCG64

try:
    # This is what IPython likes
    from . import conftest
except ImportError:
    # This is what Pytest likes
    import conftest

RNG = Generator(PCG64())


def test_relP_PCA():
    # Test if a P waveform as a multiple from another one.
    n = 512
    Aab = 1 / 50
    wvA = signal.make_wavelet(n, 30, "sin", 60)
    wvB = wvA / Aab

    Aout = amp.pca_amplitude_2p(np.array([wvA, wvB]))
    assert Aout == pytest.approx(Aab)


def test_p_misfit():
    # Test if the misfit is 0 for perfect data
    n = 512
    Aab = 1 / 50
    wvA = signal.make_wavelet(n, 30, "sin", 60)
    wvB = wvA / Aab

    Aout = amp.pca_amplitude_2p(np.array([wvA, wvB]))
    mis = amp.p_misfit(np.array([wvA, wvB]), Aout)

    assert mis == pytest.approx(0)


def test_pca_amplitude_3s():
    # Test if an S waveform is reconstructed from a linear combination of two
    # others, and if the order is preserved
    n = 512
    Babc = 0.7
    Bacb = 0.3
    wvA = signal.make_wavelet(n, 60, "sin", 80)  # 0: Singal
    wvB = 2 * RNG.random(n) - 1  # 1: Noise
    wvC = Babc * wvA + Bacb * wvB  # 2: More signal than noise

    Babc2, Bacb2, iord = amp.pca_amplitude_3s(np.array([wvA, wvB, wvC]))
    assert iord == pytest.approx([2, 0, 1])
    assert Babc == pytest.approx(Babc2)
    assert Bacb == pytest.approx(Bacb2)


def test_pca_amplitude_3s_2():
    # Test if an S waveform is reconstructed from a linear combination of two
    # others
    n = 512
    Babc = 2
    Bacb = 5
    wvC = signal.make_wavelet(n, 30, "sin", 60)
    wvB = signal.make_wavelet(n, 60, "sin", 80)
    wvA = wvB * Babc + wvC * Bacb

    Babc2, Bacb2, iord = amp.pca_amplitude_3s(np.array([wvA, wvB, wvC]))
    assert iord == pytest.approx(
        [0, 2, 1]
    )  # Note: the algorithm exchanged b and c, so ...
    assert Babc == pytest.approx(Bacb2)  # ... we expect b and c to be exchanged
    assert Bacb == pytest.approx(Babc2)


def test_pca_amplitude_3s_3():
    # Test the case where two waveforms are identical
    n = 512
    Babc = 4
    wvC = signal.make_wavelet(n, 30, "sin", 60)
    wvB = signal.make_wavelet(n, 60, "sin", 80)
    wvA = wvB * Babc

    Babc2, Bacb2, iord = amp.pca_amplitude_3s(np.array([wvA, wvB, wvC]))
    assert iord == pytest.approx(
        [1, 0, 2]
    )  # Note: the alogorithm exchanged a and b, so ...
    assert Babc2 == pytest.approx(1 / Babc)  # ... we expect a and b to be exchanged
    assert Bacb2 == pytest.approx(0.0)


def test_synthetic():
    # Test if synthetic amplitudes are computed correctly

    nev = 4
    nsta = 6
    epi_dist = 100e3  # 100 km epicentral distance
    elev = -10e3  # 10 km elevation
    dist = np.sqrt(epi_dist**2 + elev**2)

    evl, stad, phd = conftest.make_events_stations_phases(nev, nsta, epi_dist, elev)

    M0 = [mt.moment_of_magnitude(ev.mag) for ev in evl]

    # Moment tensors in tuple notation
    mtd = {
        0: core.MT(0, 0, 0, M0[0], 0, 0),  # DC: P -> NE
        1: core.MT(0, 0, 0, 0, M0[1], 0),  # DC: P -> NZ
        2: core.MT(0, 0, 0, 0, 0, M0[2]),  # DC: P -> EZ
        3: core.MT(
            M0[3] / 3, M0[3] / 3, -2 / 3 * M0[3], M0[3] / 3, M0[3] / 3, M0[3] / 3
        ),  # A DC with something on each component
    }

    # Displacement observations
    up, us = conftest.make_radiation(phd, mtd, dist)

    # Create station event pairs and triplets, only for those for which we have
    # a moment tensor and phase information
    p_pairs = [
        (s, i, j) for s in stad for i in range(len(evl)) for j in range(i + 1, len(evl))
    ]

    s_triplets = [
        (s, i, j, k)
        for s in stad
        for i in range(len(evl))
        for j in range(i + 1, len(evl))
        for k in range(j + 1, len(evl))
    ]

    Aab, Babc, orders = amp.synthetic(mtd, evl, stad, phd, p_pairs, s_triplets)

    # Check if the P amplitudes are computed correctly
    for n, (s, a, b) in enumerate(p_pairs):
        assert pytest.approx(Aab[n] * up[b, int(s), :]) == up[a, int(s), :]

    # Now check the S amplitudes
    for n, (s, a, b, c) in enumerate(s_triplets):

        # Order the syntetic waveform as in Babc, Bacb comparison
        aa, bb, cc = np.array([a, b, c])[orders[n, :]]
        ua = us[aa, int(s), :]
        ub = us[bb, int(s), :]
        uc = us[cc, int(s), :]

        assert pytest.approx(Babc[n, 0] * ub + Babc[n, 1] * uc) == ua

    # Let's try not to reorder the S wavefroms for PCA
    _, Babc, _ = amp.synthetic(mtd, evl, stad, phd, p_pairs, s_triplets, order=False)

    # Now check the S amplitudes
    for n, (s, a, b, c) in enumerate(s_triplets):

        ua = us[a, int(s), :]
        ub = us[b, int(s), :]
        uc = us[c, int(s), :]

        assert pytest.approx(Babc[n, 0] * ub + Babc[n, 1] * uc) == ua


def test_s_misfit():
    # Test if identical waveforms yield 0 misfit
    n = 512
    wvA = signal.make_wavelet(n, 60, "sin", 80)
    mis = amp.s_misfit(np.array([wvA, wvA, wvA]), 0.5, 0.5)
    assert mis == pytest.approx(0)


def test_s_misfit2():
    # Test if noise-free fit yield 0 misfit
    n = 512
    Babc = 2
    Bacb = 5
    wvC = signal.make_wavelet(n, 30, "sin", 60)
    wvB = signal.make_wavelet(n, 60, "sin", 80)
    wvA = wvB * Babc + wvC * Bacb
    mis = amp.s_misfit(np.array([wvA, wvB, wvC]), Babc, Bacb)
    assert mis == pytest.approx(0)


def test_event_order():
    n = 512
    wvA = signal.make_wavelet(n, 60, "sin", 80)  # 0: Singal
    wvB = 2 * RNG.random(n) - 1  # 1: Noise
    wvC = 0.7 * wvA + 0.3 * wvB  # 2: More signal than noise
    iord = amp.order_by_ccsum(np.array([wvA, wvB, wvC]))
    assert iord == pytest.approx([2, 0, 1])


# Created by Cvscode Co-pilot
def test_pca_amplitude_2p():
    wvfAB = np.random.rand(2, 100)
    result = amp.pca_amplitude_2p(wvfAB)
    assert isinstance(result, float)


def test_p_misfit():
    AB = np.random.rand(2, 100)
    Aab = 0.5
    result = amp.p_misfit(AB, Aab)
    assert result >= 0


def test_info_p():

    amplitudes = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1.0, 0.1),
        core.P_Amplitude_Ratio("STA1", 0, 2, 1.1, 1.5),
        core.P_Amplitude_Ratio("STA1", 0, 3, 1.2, 1.5),
    ]

    amp.info(amplitudes, width=5)


def test_info_s():

    amplitudes = [
        core.S_Amplitude_Ratios("STA1", 0, 1, 2, 1.0, 1.0, 0.1),
        core.S_Amplitude_Ratios("STA1", 0, 1, 3, 1.1, 1.1, 1.5),
        core.S_Amplitude_Ratios("STA1", 0, 3, 4, 0.9, 0.9, 1.5),
    ]

    amp.info(amplitudes, width=5)
