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
        events_=[0, 1, 2, 6, 7],
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


def test_alignment_pv_yticklabels_visible():
    arr = _3d_signal_to_plot()
    hdr = core.Header(
        station="STAT",
        phase="P",
        events_=[0, 1, 2, 6, 7],
        components="ZNE",
        sampling_rate=10,
        data_window=6.4,
        phase_start=-1.0,
        phase_end=2.0,
        taper_length=1.0,
        highpass=1.5,
        lowpass=2.0,
        null_threshold=0.0,
        min_signal_noise_ratio=0.0,
        min_correlation=0.5,
        min_expansion_coefficient_norm=0.5,
        combinations_from_file=False,
    )

    fig, axs = plot.alignment(arr, hdr)
    fig.canvas.draw()

    labels = axs["pv"].get_yticklabels()
    assert [label.get_text() for label in labels] == ["0", "1"]
    assert all(label.get_visible() for label in labels)

    plt.close(fig)


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


def test_amplitudes_p_plot():
    amplitudes = [
        core.P_Amplitude_Ratio("AAA", "P", 1, 2, 2.0, 0.10, 0.9, 1.0, 0.5, 1.0, 5.0),
        core.P_Amplitude_Ratio("AAA", "P", 2, 3, 3.0, 0.30, 0.8, 1.0, 0.5, 1.0, 5.0),
        core.P_Amplitude_Ratio("BBB", "P", 3, 4, 4.0, 0.20, 0.7, 1.0, 0.5, 1.0, 5.0),
    ]

    fig, axs = plot.amplitudes(amplitudes)

    assert axs.shape == (2, 2)
    assert fig._suptitle.get_text() == "P-amplitudes"
    assert axs[0, 0].get_ylabel() == "Relative Amplitude"
    assert axs[1, 0].get_ylabel() == "Norm. ampl. reconstr. misfit"
    assert axs[1, 0].get_xlabel() == "Observation"
    assert all(ax.get_yscale() == "log" for ax in axs[:2, :].flat)
    assert axs[0, 0].get_legend() is not None

    plt.close(fig)


def test_amplitudes_with_weights_and_norms():
    amplitudes = [
        core.S_Amplitude_Ratios(
            "AAA", "S", 1, 2, 3, 2.0, 1.5, 0.10, 0.9, 1.0, 0.5, 0.2, 1.0, 5.0
        ),
        core.S_Amplitude_Ratios(
            "AAA", "S", 2, 3, 4, 3.0, 2.0, 0.50, 0.8, 1.0, 0.5, 0.2, 1.0, 5.0
        ),
        core.S_Amplitude_Ratios(
            "BBB", "S", 4, 5, 6, 4.0, 2.5, 0.20, 0.7, 1.0, 0.5, 0.2, 1.0, 5.0
        ),
    ]
    weights = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    norms = np.array([[1.0], [0.5], [0.8]])

    fig, axs = plot.amplitudes(
        amplitudes,
        reference_events=[2, 4, 6],
        title="Custom",
        weights=weights,
        norms=norms,
    )

    assert axs.shape == (4, 2)
    assert fig._suptitle.get_text() == "Custom"
    assert axs[2, 0].get_ylabel() == "Weight"
    assert axs[3, 0].get_ylabel() == "Norm"
    assert axs[3, 0].get_xlabel() == "Observation"
    weight_lines = [
        line for line in axs[2, 0].lines if line.get_label().startswith("Weight")
    ]
    norm_lines = [
        line for line in axs[3, 0].lines if line.get_label().startswith("Norm")
    ]
    assert len(weight_lines) == 2
    assert len(norm_lines) == 1
    assert not axs[2, 1].axison
    assert not axs[3, 1].axison
    legend = axs[0, 0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["2", "4", "6"]

    plt.close(fig)


def test_amplitudes_empty():
    try:
        plot.amplitudes([])
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty amplitudes")


def test_amplitudes_invalid_panel_shape():
    amplitudes = [
        core.P_Amplitude_Ratio("AAA", "P", 1, 2, 2.0, 0.10, 0.9, 1.0, 0.5, 1.0, 5.0),
        core.P_Amplitude_Ratio("AAA", "P", 2, 3, 3.0, 0.30, 0.8, 1.0, 0.5, 1.0, 5.0),
        core.P_Amplitude_Ratio("BBB", "P", 3, 4, 4.0, 0.20, 0.7, 1.0, 0.5, 1.0, 5.0),
    ]

    try:
        plot.amplitudes(amplitudes, weights=np.ones((3, 4)))
    except ValueError as exc:
        assert "between 1 and 3 columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid weights shape")
