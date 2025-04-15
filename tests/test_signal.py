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
Test the utility functions
"""

import pytest
import numpy as np
from relmt import signal, qc


def test_shift():
    # Test if a trace is shifted in time as in Fourier domain
    n = 512
    shift = 10
    inp = signal.make_wavelet(n, 5, "sin", 30, 0, 0)
    expc = signal.make_wavelet(n, 5, "sin", 30, shift, shift)[np.newaxis, :]
    out = signal.shift(inp, 1, shift)
    assert expc == pytest.approx(out)


def test_shift_3d():
    # Test if a trace is shifted in time as in Fourier domain
    n = 512
    c = 3
    ev = 10
    shift = 10
    inp = np.ones((ev, c, n)) * signal.make_wavelet(n, 5, "sin", 30, 0, 0)
    expc = np.ones((ev, c, n)) * signal.make_wavelet(n, 5, "sin", 30, shift, shift)
    out = signal.shift_3d(inp, 1, shift)
    assert expc == pytest.approx(out)


def test_shift_shift_3d():
    # Test if a shift and shift_3d do the same thing
    n = 512
    c = 3
    ev = 10
    shift = 10
    inp = np.ones((ev, c, n)) * signal.make_wavelet(n, 5, "sin", 30, 0, 0)
    expc = signal.shift(inp, 1, shift)
    out = signal.shift_3d(inp, 1, shift)  # Is shift_3d not necessary!?
    assert expc == pytest.approx(out)


# def test_xcorrc1():
#    # Test if a wavelet correlates with itself
#    n = 512
#    x = signal.make_wavelet(n, 5, "sin", 30, 0, 0)
#    corr = signal._xcorrc(x, x)
#    assert corr[n - 1] == pytest.approx(1)
#
#
# def test_xcorrc2():
#    # Test if 1/2 periot shifted wavelet anti-correlates
#    n = 512
#    x = signal.make_wavelet(n, 10, "sin", 30, 0, 0)
#    y = signal.make_wavelet(n, 10, "sin", 30, 5, 0)
#    corr = signal._xcorrc(x, y)
#    assert corr[n - 1] == pytest.approx(-1)


def test_make_wavelet_sin():
    # Test if make_wavelet makes a sine
    n = 512
    x = signal.make_wavelet(n, n, "sin", np.inf, 0, 0)
    y = np.sin(np.arange(n) * 2 * np.pi / n)
    assert x == pytest.approx(y)


def test_make_wavelet_cos():
    # Test if make_wavelet makes a cosine
    n = 512
    x = signal.make_wavelet(n, n, "cos", np.inf, 0, 0)
    y = np.cos(np.arange(n) * 2 * np.pi / n)
    assert x == pytest.approx(y)


def test_make_wavelet_rec():
    # Test if make_wavelet makes a rectancle
    n = 512
    x = signal.make_wavelet(n, n, "rec", np.inf, 0, 0)
    assert len(x) == n


def test_make_wavelet_error():
    # Just call make wavelet
    n = 64
    with pytest.raises(ValueError):
        _ = signal.make_wavelet(n, 10, "invalid", 30, 0, 0)


def test_indices_noise():
    i0, i1 = signal.indices_noise(100, -1.0, 1.0, 10.0)
    assert i1 == 400  # start of phase
    assert i0 == 200  # 2s (=phase window) before start of phase


def test_norm_power():
    shape = (3, 512)
    fac = np.array([1, 2, 3])
    arr = np.ones(shape)
    arr *= fac[:, np.newaxis]
    arrout = signal.norm_power(arr)
    test = fac / np.sqrt(fac**2 * shape[1])
    for arrow, tes in zip(arrout, test):
        assert all(arrow == tes)


def test_zero_events():
    # Test if events are zeroed
    shape = (3, 512)
    arr = np.ones(shape)
    out = signal.zero_events(arr, [0, 1])
    assert qc.index_nonzero_events(out) == [2]


def test_taper_both():
    # Test if taper is applied correctly at both ends
    n = 500
    sampling_rate = 100  # Trace is 5 seconds long
    t1 = 2 - n / sampling_rate / 2
    t2 = 3 - n / sampling_rate / 2
    length = 2
    x = np.ones(n)
    y = signal.cosine_taper(x, length, sampling_rate, t1, t2)

    # Before 1s is zeroed
    assert y[:100] == pytest.approx(0)

    # After 4s is zeroed
    assert y[400:] == pytest.approx(0)

    # 2-3s is untouched
    assert y[200:300] == pytest.approx(1)

    # 1-2s is tapered
    assert all(y[101:200] > 0)
    assert all(y[101:200] < 1)

    # 3-4s is tapered
    assert all(y[301:400] > 0)
    assert all(y[301:400] < 1)


def test_taper_only_start():
    # Test if taper is applied correctly at the start
    n = 500
    sampling_rate = 100  # Trace is 5 seconds long
    t1 = 2 - n / 2 / sampling_rate
    t2 = None
    length = 2
    x = np.ones(n)
    y = signal.cosine_taper(x, length, sampling_rate, t1, t2)

    # Before 1s is zeroed
    assert y[:100] == pytest.approx(0)

    # 2-end is untouched
    assert y[200:] == pytest.approx(1)

    # 1-2s is tapered
    assert all(y[101:200] > 0)


def test_taper_only_end():
    # Test if taper is applied correctly at the end
    n = 500
    sampling_rate = 100  # Trace is 5 seconds long
    t1 = None
    t2 = 3 - n / 2 / sampling_rate
    length = 2
    x = np.ones(n)
    y = signal.cosine_taper(x, length, sampling_rate, t1, t2)

    # After 4s is zeroed
    assert y[400:] == pytest.approx(0)

    # start - 3s is untouched
    assert y[:300] == pytest.approx(1)

    # 3-4s is tapered
    assert all(y[301:400] > 0)
    assert all(y[301:400] < 1)


def test_taper_error():
    # Test if taper is ValueError is raised outside range
    n = 500
    sampling_rate = 100  # Trace is 5 seconds long
    t1 = 1 - n / 2 / sampling_rate
    t2 = 4.5 - n / 2 / sampling_rate
    nt = 1  # This taper is longer than trace
    x = np.ones(n)
    with pytest.raises(ValueError):
        signal.cosine_taper(x, nt, sampling_rate, t1, t2)


def test_demean():
    # Test if the mean is removed
    zeromean = [0, 1, -1, 2, -2, 5, -5]
    mean = 2
    x = np.array(zeromean) + mean
    assert np.mean(x) == pytest.approx(mean)

    y = signal.demean(x)
    assert zeromean == pytest.approx(y)


def test_destep():
    # Test if a mean is
    step = [0, 1, -1, 2, -2, 5, -5]

    y = signal.destep(step)
    assert y[0] == y[-1]


def test_rotate_nez_rtz():
    n = 512
    nez = np.array(
        [
            np.zeros(n),
            signal.make_wavelet(n, 5, "sin", 30, 0, 0),
            np.zeros(n),
        ]
    )
    rtz = np.array(
        [
            signal.make_wavelet(n, 5, "sin", 30, 0, 0),
            np.zeros(n),
            np.zeros(n),
        ]
    )

    rtz2 = signal.rotate_nez_rtz(nez, -90)
    assert pytest.approx(rtz) == rtz2


def test_differentiate():
    # Test if d/dx of sin(x) is indeed cos(x)
    n = 512
    x = np.sin(np.arange(n) * 2 * np.pi / n)

    # d/dx sin(x) = cos(x)
    dx = np.cos(np.arange(n) * 2 * np.pi / n)
    out = signal.differentiate(x, n / 2 / np.pi)
    assert out == pytest.approx(dx)


def test_integrate():
    n = 512
    dx = np.cos(np.arange(n) * 2 * np.pi / n)
    x = np.sin(np.arange(n) * 2 * np.pi / n)
    out = signal.integrate(dx, n / 2 / np.pi)
    assert out == pytest.approx(x)


def _10s_signal():
    """Return sine wavelet with 10 samples period length"""
    n = 1024
    fin = 0.1  # Input frequency
    return signal.make_wavelet(n, 1 / fin, "sin", 60, 0, 0)


def test_filter():
    # Just test if all the filter options can be called
    sig = _10s_signal()
    for typ in ["bessel", "butter"]:
        _ = signal.filter(sig, 1, lowpass=0.2, typ=typ)
        _ = signal.filter(sig, 1, lowpass=0.2, typ=typ)  # Try to get a cached filter
        _ = signal.filter(sig, 1, highpass=0.2, typ=typ)
        _ = signal.filter(sig, 1, highpass=0.2, lowpass=0.4, typ=typ)
        _ = signal.filter(sig, 1, highpass=0.2, lowpass=0.4, typ=typ, zerophase=True)


def test_filter_error():
    sig = _10s_signal()
    with pytest.raises(ValueError):
        # No filter corner given
        sig2 = signal.filter(sig, 1, typ="butter")
    with pytest.raises(ValueError):
        # Unknown type
        _ = signal.filter(sig, 1, highpass=0.2, lowpass=0.4, typ="foo")
    with pytest.raises(ValueError):
        # hpas > lowpass :(
        _ = signal.filter(sig, 1, highpass=0.4, lowpass=0.2)


def test_filter_invalid_type():
    sig = _10s_signal()
    with pytest.raises(ValueError):
        signal.filter(sig, 1, typ="invalid")


def test_indices_inside_taper():
    i0, i1 = signal.indices_inside_taper(
        sampling_rate=100,
        taper_length=1,
        phase_start=-0.5,
        phase_end=1.5,
        data_window=12,
    )
    assert (i0, i1) == (600 - 50 - 50, 600 + 150 + 50)


def test_make_wavelet():
    result = signal.make_wavelet(100, 10, typ="sin")
    assert result.shape == (100,)


def test_indices_signal():
    result = signal.indices_signal(100, -0.5, 0.5, 2)
    assert result == (50, 150)


def test_choose_passband():

    # Choose highest highpass and lowest lowpass
    hpas, lpas = signal.choose_passband([0.5, 0.3, 0.6], [12.0, 18.0, 14.0])
    assert hpas == 0.6
    assert lpas == 12.0

    # Best filter is invalid at default dynamic range
    hpas, lpas = signal.choose_passband([0.5, 0.3, 1.0], [1.0, 18.0, 14.0])
    assert hpas is None
    assert lpas is None

    # Best filter is invalid in given dynamic range
    hpas, lpas = signal.choose_passband(
        [0.5, 0.3, 1.0], [1.5, 18.0, 14.0], min_dynamic_range=5.0
    )
    assert hpas is None
    assert lpas is None


def test_dB():
    assert pytest.approx(0.0) == signal.dB(1.0)


def test_signal_noise_ratio():
    n = 1000
    snr1 = 10

    sig = signal.make_wavelet(n, 10, "sin", 100, 0, 400) * snr1
    noi = signal.make_wavelet(n, 5, "sin", 100, 0, -400)

    noisy_signal = sig + noi

    snr2 = signal.signal_noise_ratio(noisy_signal, 1, 0, 450, 50)
    assert pytest.approx(snr2, rel=0.1) == signal.dB(snr1)

    # Try to filter out noise
    snr3 = signal.signal_noise_ratio(noisy_signal, 1, 0, 450, 50, lowpass=1 / 7.5)
    assert snr3 > signal.dB(snr2)

    # Multi-dimensional matrix
    three_d = np.ones((5, 3, n))
    sig2 = three_d * signal.make_wavelet(n, 10, "sin", 100, 0, 400) * snr1
    noi2 = three_d * signal.make_wavelet(n, 5, "sin", 100, 0, -400)

    noisy_signal2 = sig2 + noi2

    snr4 = signal.signal_noise_ratio(noisy_signal2, 1, 0, 450, 50)
    assert pytest.approx(snr4, rel=0.1) == signal.dB(snr1)


def test_cc_coef():
    n = 1000
    sig1 = signal.make_wavelet(n, 5, "sin", 100, 0, 0)
    sig2 = signal.make_wavelet(n, 5, "sin", 100, 2.5, 0)  # 1/2 phase shift

    assert pytest.approx(signal.cc_coef(sig1, sig1)) == 1.0
    assert pytest.approx(signal.cc_coef(sig1, sig2)) == -1.0
