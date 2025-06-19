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
Test the plotting functions
"""

import numpy as np
from relmt import plot, signal, core
import matplotlib.pyplot as plt


def _2d_signal_to_plot():
    return np.vstack([signal.make_wavelet(64, 10, "sin", 30, 0, 0) for _ in range(5)])


def _3d_signal_to_plot():
    return np.array(
        [_2d_signal_to_plot(), _2d_signal_to_plot(), _2d_signal_to_plot()]
    ).transpose(
        (1, 0, 2)
    )  # Events, Channels, Samples


def test_section_3d(iplot=False):
    if iplot:
        plt.ion()

    wvarr = _3d_signal_to_plot()
    hdr = core.Header(
        station="STAT",
        phase="P",
        events=[0, 1, 2, 6, 7],
        components="ZNE",
        sampling_rate=100,
    )

    plot.section_3d(wvarr)
    plot.section_3d(wvarr, **hdr.kwargs(plot.section_3d))
    plot.section_3d(wvarr, 0, **hdr.kwargs(plot.section_3d))
    plot.section_3d(wvarr, 1, **hdr.kwargs(plot.section_3d))

    _, ax = plt.subplots()
    plot.section_3d(wvarr, ax=ax)


def test_section_2d(iplot=False):
    if iplot:
        plt.ion()

    plot.section_2d(_2d_signal_to_plot())
    _, ax = plt.subplots()
    plot.section_2d(_2d_signal_to_plot(), ax=ax)


def test_section_2d_image(iplot=False):
    if iplot:
        plt.ion()

    plot.section_2d(_2d_signal_to_plot(), image=True, wiggle=False)


def test_p_reconstruction(iplot=False):
    n = 512
    Aab = 10
    wvA = signal.make_wavelet(n, 30, "sin", 60)
    wvB = (wvA + np.random.standard_normal(wvA.shape) / 10) / Aab

    if iplot:
        plt.ion()

    plot.p_reconstruction(wvA, wvB, Aab)
    plot.p_reconstruction(wvA, wvB, Aab, 100.0)
    plot.p_reconstruction(wvA, wvB, Aab, 100.0, (1, 2))

    _, axs = plt.subplots(1, 3)
    plot.p_reconstruction(wvA, wvB, Aab, 100.0, (1, 2), axs)


def test_s_reconstruction(iplot=False):
    n = 512
    Babc = 2
    Bacb = 5
    wvC = signal.make_wavelet(n, 30, "sin", 60)
    wvB = signal.make_wavelet(n, 60, "sin", 80)
    wvA = wvB * Babc + wvC * Bacb + np.random.standard_normal(wvC.shape)

    if iplot:
        plt.ion()

    plot.s_reconstruction(wvA, wvB, wvC, Babc, Bacb)
    plot.s_reconstruction(wvA, wvB, wvC, Babc, Bacb, 100.0)
    plot.s_reconstruction(wvA, wvB, wvC, Babc, Bacb, 100.0, (1, 2, 3))

    _, axs = plt.subplots(1, 4)
    plot.s_reconstruction(wvA, wvB, wvC, Babc, Bacb, 100.0, (1, 2, 3), axs)


def test_bootstrap_matrix(iplot=False):
    mts = [core.MT(0, 0, 0, 0, 0, 0), core.MT(1, 1, 1, 1, 1, 1)]
    best_mt = core.MT(2, 2, 2, 2, 2, 2)
    takeoff = np.array([[0.0, 45.0], [90.0, 0.0]])

    fig, ax = plot.bootstrap_matrix(
        mts,
        plot_beachball=True,
        takeoff=takeoff,
        best_mt=best_mt,
        subplot_kwargs={"figsize": (3, 3)},
    )

    if iplot:
        fig.show()
