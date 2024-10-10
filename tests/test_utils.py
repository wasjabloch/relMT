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
import relmt.utils as utils


def test_shift():
    # Test if a trace is shifted in time as in Fourier domain
    n = 512
    shift = 10
    inp = utils.make_wavelet(n, 5, "sin", 30, 0, 0)
    expc = utils.make_wavelet(n, 5, "sin", 30, shift, shift)
    out = utils.shift(inp, 1, shift)[0]
    assert expc == pytest.approx(out)


def test_xcorrc1():
    # Test if a wavelet correlates with itself
    n = 512
    x = utils.make_wavelet(n, 5, "sin", 30, 0, 0)
    corr = utils.xcorrc(x, x)
    assert corr[n - 1] == pytest.approx(1)


def test_xcorrc2():
    # Test if 1/2 periot shifted wavelet anti-correlates
    n = 512
    x = utils.make_wavelet(n, 10, "sin", 30, 0, 0)
    y = utils.make_wavelet(n, 10, "sin", 30, 5, 0)
    corr = utils.xcorrc(x, y)
    assert corr[n - 1] == pytest.approx(-1)


def test_make_wavelet_sin():
    # Test if make_wavelet make a sine
    n = 512
    x = utils.make_wavelet(n, n, "sin", np.inf, 0, 0)
    y = np.sin(np.arange(n) * 2 * np.pi / n)
    assert x == pytest.approx(y)


def test_make_wavelet_cos():
    # Test if make_wavelet make a sine
    n = 512
    x = utils.make_wavelet(n, n, "cos", np.inf, 0, 0)
    y = np.cos(np.arange(n) * 2 * np.pi / n)
    assert x == pytest.approx(y)


def test_taper_both():
    # Test if taper is applied correctly at both ends
    n = 500
    dt = 1 / 100  # Trace is 5 seconds long
    t1 = 2
    t2 = 3
    nt = 1
    x = np.ones(n)
    y = utils.taper(x, nt, dt, t1, t2)

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
    dt = 1 / 100  # Trace is 5 seconds long
    t1 = 2
    t2 = -1
    nt = 1
    x = np.ones(n)
    y = utils.taper(x, nt, dt, t1, t2)

    # Before 1s is zeroed
    assert y[:100] == pytest.approx(0)

    # 2-end is untouched
    assert y[200:] == pytest.approx(1)

    # 1-2s is tapered
    assert all(y[101:200] > 0)


def test_taper_only_end():
    # Test if taper is applied correctly at the end
    n = 500
    dt = 1 / 100  # Trace is 5 seconds long
    t1 = -1
    t2 = 3
    nt = 1
    x = np.ones(n)
    y = utils.taper(x, nt, dt, t1, t2)

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
    dt = 1 / 100  # Trace is 5 seconds long
    t1 = 1
    t2 = 4.5
    nt = 1  # This taper is longer than trace
    x = np.ones(n)
    with pytest.raises(ValueError):
        utils.taper(x, nt, dt, t1, t2)


def test_demean():
    # Test if a mean is
    zeromean = [0, 1, -1, 2, -2, 5, -5]
    mean = 2
    x = np.array(zeromean) + mean
    assert np.mean(x) == pytest.approx(mean)

    y = utils.demean(x)
    assert zeromean == pytest.approx(y)


def test_differentiate():
    # Test if d/dx of sin(x) is indeed cos(x)
    n = 512
    x = np.sin(np.arange(n) * 2 * np.pi / n)

    # d/dx sin(x) = cos(x)
    dx = np.cos(np.arange(n) * 2 * np.pi / n)
    out = utils.differentiate(x, 2 * np.pi / n)
    assert out == pytest.approx(dx)


def test_corner_frequency_P():
    cf = utils.corner_frequency(5, 5e6, 3500, "P")
    assert cf == pytest.approx(0.9, 1e-3)


def test_corner_frequency_S():
    cf = utils.corner_frequency(5, 5e6, 3500, "S")
    assert cf == pytest.approx(0.5, 1e-2)


def _10s_signal():
    """Return sine wavelet with 10 samples period length"""
    n = 1024
    fin = 0.1  # Input frequency
    return utils.make_wavelet(n, 1 / fin, "sin", 60, 0, 0)


def test_filter():
    # Just test if all the filter options can be called
    sig = _10s_signal()
    for typ in ["bessel", "butter"]:
        _ = utils.filter_wvm(sig, 1, lpas=0.2, typ=typ)
        _ = utils.filter_wvm(sig, 1, hpas=0.2, typ=typ)
        _ = utils.filter_wvm(sig, 1, hpas=0.4, lpas=0.2, typ=typ)


def test_filter_error():
    sig = _10s_signal()
    with pytest.raises(ValueError):
        _ = utils.filter_wvm(sig, 1)
    with pytest.raises(ValueError):
        _ = utils.filter_wvm(sig, 1, hpas=0.4, lpas=0.2, typ="foo")


def test_xyzarray_sta():
    # Test if xyzarray works with station dictionary format
    xyz = utils.xyzarray({"A": (1, 2, 3), "B": (3, 4, 5)})
    assert xyz == pytest.approx(np.array([[1, 2, 3], [3, 4, 5]]))


def test_xyzarray_ev():
    # Test if xyzarray works with event dictionary format
    xyz = utils.xyzarray({1: (1, 2, 3, "a"), 2: (3, 4, 5, "b")})
    assert xyz == pytest.approx(np.array([[1, 2, 3], [3, 4, 5]]))


def test_approx_time_lookup_float():
    t1s = [0, 2, 5]
    t2s = [0, 4, 1.5, 99]
    lut = utils.approx_time_lookup(t1s, t2s)
    assert lut == {0: 0, 2: 1.5, 5: 4}


def test_approx_time_lookup_include():
    t1s = [0, 2, 5]
    t2s = [0, 4, 1.5, 99]
    lut = utils.approx_time_lookup(t1s, t2s, include_at=0.1)
    assert lut == {0: 0}


def test_approx_time_lookup_time():
    t1 = ["2015-03-02T15:30:12", "2016-03-02T15:30:12"]
    t2 = ["2015-03-02T15:30:13", "2016-03-02T15:30:13"]
    lut = utils.approx_time_lookup(t1, t2)
    assert lut == {t1[0]: t2[0], t1[1]: t2[1]}


def test_approx_time_lookup_error():
    t1s = [0, 2]
    t2s = [0]
    with pytest.raises(LookupError):
        _ = utils.approx_time_lookup(t1s, t2s, include_at=0.1)


def test_interpolate_phd_time():
    # Test if phases arrival times are interpolated correctly
    phd = {"0_A_P": (10, 0, 0)}
    std = {"A": (1, 0, 0)}
    evd = {0: (0, 0, 0, 5, -1, "ev0"), 1: (0, 0, 0, 15, -1, "ev1")}
    phdn = utils.interpolate_phd(phd, evd, std)
    assert phdn == pytest.approx({"1_A_P": (20, 0, 0)})


def test_interpolate_phd_aziinc():
    # Test if phases take-off angles are interpolated correctly
    phd = {"0_A_P": (10, 90.1, -20), "1_A_P": (10, 89.9, -10)}
    std = {"A": (0, 1, 0)}
    evd = {
        0: (0, 0, 3, 5, -1, "ev0"),
        1: (0, 0, 1, 5, -1, "ev1"),
        2: (0, 0, 2, 5, -1, "ev2"),
    }
    phdn = utils.interpolate_phd(phd, evd, std)
    # Just created this one new key
    assert phdn.keys() == pytest.approx(["2_A_P"])
    # azimuth is along east-axis
    assert phdn["2_A_P"][1] == pytest.approx(90)
    # inclination is in the middle between -20 an -10
    assert phdn["2_A_P"][2] == pytest.approx(-15)


def test_interpolate_phd_nmin():
    # Test if phases are interpolated correctly
    phd = {"0_A_P": (10, 0, 0)}
    std = {"A": (1, 0, 0)}
    evd = {0: (0, 0, 0, 5, -1, "ev0"), 1: (0, 0, 0, 15, -1, "ev1")}
    phdn = utils.interpolate_phd(phd, evd, std, nmin=2)
    assert phdn == {}


def test_interpolate_phd_missing_stations():
    # Test if missing stations are handled correctly
    phd = {"0_B_P": (10, 0, 0)}
    std = {"A": (1, 0, 0)}
    evd = {0: (0, 0, 0, 5, -1, "ev0")}
    phdn = utils.interpolate_phd(phd, evd, std)
    assert phdn == {}