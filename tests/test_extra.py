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
Test the extra functions
"""

import pytest
from relmt import extra, core, io, signal
import tempfile
import os
import numpy as np
from pathlib import Path

try:
    from obspy import Stream, Trace, UTCDateTime
except ModuleNotFoundError:
    msg = "Could not import utm.\n"
    msg += "Please install relMT with optional 'geo' dependencies:\n"
    msg += "pip install relmt[geo]"
    raise ModuleNotFoundError(msg)


def test_obspy_inventory_files():
    # Test if inventory file is read in and content is consistent
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = extra.read_obspy_inventory_files(fns)
    assert len(inv) == 1
    assert len(inv.get_contents()["stations"]) == 2


def test_utmzone():
    # Check if UTM zone is returned coorrectly
    num, let = extra.get_utm_zone(latitudes=[38.6, 37.5], longitudes=[69.1, 71.6])
    assert num == 42
    assert let == "S"


def test_utmzone_err1():
    # Check if LookupError is raised
    with pytest.raises(LookupError):
        # One station in Tajikistan, the other on the equator!
        _, _ = extra.get_utm_zone(latitudes=[38.6, 0], longitudes=[69.1, 71.6])


def test_utmzone_err2():
    # Check if LookupError is raised
    with pytest.raises(LookupError):
        # 2nd is on null meridian
        _, _ = extra.get_utm_zone(latitudes=[38.6, 37.5], longitudes=[69.1, 0])


def my_geoconverter(lat, lon, dep):
    return extra.geoconverter_latlon2utm(lat, lon, dep, 42, "S")


def test_read_station_inventory():
    # See if a station table is returned coorectly
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = extra.read_obspy_inventory_files(fns)
    stad = extra.read_station_inventory(inv, my_geoconverter)

    # Has 2 header lines + 2 stations
    assert "CHGR" in stad
    assert "MANEM" in stad
    assert stad["CHGR"][:3] == pytest.approx((4278715, 513765, -1049), abs=1)
    assert stad["CHGR"][3] == "CHGR"


def test_read_station_inventory2():
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = extra.read_obspy_inventory_files(fns)
    # Test if alternate and historic codes are handled correctly
    inv[0][0].alternate_code = "ALT"
    inv[0][1].historical_code = "HIST"
    stad = extra.read_station_inventory(inv, my_geoconverter)
    assert len(stad) == 4
    assert "ALT" in stad
    assert "HIST" in stad


def test_read_station_inventory_error():
    # Test if a key Error is raised when doubled stations occurr
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = extra.read_obspy_inventory_files(fns)

    # Test if double codes write an error
    inv[0][0].alternate_code = "MANEM"
    with pytest.raises(KeyError):
        _ = extra.read_station_inventory(inv, my_geoconverter)

    # But don't, when they should not
    _ = extra.read_station_inventory(inv, my_geoconverter, strict=False)


def test_read_station_inventory_error2():
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = extra.read_obspy_inventory_files(fns)

    # Test if underscore produces an error
    inv[0][0].alternate_code = "MA_EM"
    with pytest.raises(ValueError):
        _ = extra.read_station_inventory(inv, my_geoconverter)


def test_geoconverter_latlon2utm():
    # Test if UTM conversion works
    n, e, d = extra.geoconverter_latlon2utm(39.198, 74.295, 4.69, 43, "S")
    assert int(n) == 4338985
    assert int(e) == 439123
    assert int(d) == 4690


def test_geoconverter_utm2latlon():
    # Test if reverse UTM conversion works
    lat, lon, dep = extra.geoconverter_utm2latlon(4338985.0, 439123.0, 4690.0, 43, "S")
    assert lat == pytest.approx(39.198, abs=1e-3)
    assert lon == pytest.approx(74.295, abs=1e-3)
    assert dep == pytest.approx(4.69, abs=1e-3)


def test_make_waveform_array():
    sampling_rate = 100
    data_window = 2
    components = "ZNE"
    stream = Stream(
        [
            Trace(
                signal.make_wavelet(5000, 50, we=100, de=-1500),
                header={
                    "station": "A",
                    "channel": f"HH{cha}",
                    "delta": 1 / sampling_rate,
                    "starttime": UTCDateTime(t0),
                },
            )
            for cha in components
            for t0 in [0.0, 100.0]
        ]
    )

    phase_dict = {
        "0_A_P": core.Phase(10.0, 0, 0),
        "0_A_S": core.Phase(15.0, 0, 0),
        "1_A_P": core.Phase(110.0, 0, 0),
        "1_A_S": core.Phase(115.0, 0, 0),
    }

    header = core.Header(
        station="A",
        phase="P",
        data_window=data_window,
        sampling_rate=sampling_rate,
        components=components,
    )
    arr, hdr = extra.make_waveform_array(header, phase_dict, stream)
    assert hdr["events"] == [0, 1]
    assert arr.shape == (2, 3, sampling_rate * data_window)


def test_spectrum():
    # Test if the spectrum captures a dominant frequency
    period = 10
    sig = signal.make_wavelet(512, period, "sin", 10)
    freq, spec = extra.spectrum(sig, 1)
    f_dom = freq[np.argmax(spec)]  # Dominant frequency of the signal

    # account for discrete sampling of frequencies
    assert f_dom == pytest.approx(1 / period, rel=0.01)


def test_apparent_corner_frequency():
    # The corner frequency is the dominant frequency of the velocity spectrum
    period = 10
    sig = signal.make_wavelet(512, period, "sin", 10)
    fc = extra.apparent_corner_frequency(sig, 1)

    # account for discrete sampling of frequencies
    assert fc == pytest.approx(1 / period, rel=0.01)


def test_apparent_corner_frequency_multi_channel():
    # The corner frequency is the dominant frequency of the volocity spectrum
    period = 10

    # three-component signal
    sig = np.array([signal.make_wavelet(512, period, "sin", 10)] * 3)
    fc = extra.apparent_corner_frequency(sig, 1)

    # account for discrete sampling of frequencies
    assert fc == pytest.approx(1 / period, rel=0.01)


def test_apparent_corner_frequency_bracket():
    # The corner frequency is the dominant frequency of the volocity spectrum
    period1 = 10
    period2 = 20

    # Superimpose two signals
    sig = signal.make_wavelet(512, period1, "sin", 10)
    sig += signal.make_wavelet(512, period2, "sin", 10)

    # Try and find the higher frequency
    fc = extra.apparent_corner_frequency(sig, 1, fmin=1 / 15)
    assert fc == pytest.approx(1 / period1, rel=0.01)

    # Try and find the lower frequency
    fc = extra.apparent_corner_frequency(sig, 1, fmax=1 / 15)
    assert fc == pytest.approx(1 / period2, rel=0.01)


def test_optimal_bandpass():
    # The corner frequency is the dominant frequency of the volocity spectrum
    period_sig = 20
    period_hfn = 10  # high frequency noise
    period_lfn = 50  # low frequency noise

    # Superimpose two signals
    # Signal pulse near the end of the trace
    sig = signal.make_wavelet(512, period_sig, "sin", 10, de=100)

    # Continous high amplitude noise at frequencies aboove and below signal
    sig += signal.make_wavelet(512, period_hfn, "cos") * 10
    sig += signal.make_wavelet(512, period_lfn, "cos") * 10

    # Find best bandpass
    hpas, lpas = extra.optimal_bandpass(
        sig, sampling_rate=1, data_window=512, phase_start=0, phase_end=128
    )

    # Assert highpass is above low frequency noise
    assert hpas > 1 / period_lfn

    # Assert lowpass is below high frequency noise
    assert lpas > 1 / period_hfn


def test_read_waveform_array():
    sr = 100
    tw = 2
    sta = "A"
    pha = "P"
    it = 0
    components = "ZNE"
    header = core.Header(
        station=sta,
        phase=pha,
        components=components,
        sampling_rate=sr,
        data_window=tw,
    )

    st = Stream(
        [
            Trace(
                signal.make_wavelet(5000, 50, we=100, de=-1500),
                header={
                    "station": sta,
                    "channel": f"HH{com}",
                    "delta": 1 / sr,
                    "starttime": UTCDateTime(0),
                },
            )
            for com in components
        ]
    )

    phd = {f"0_{sta}_{pha}": core.Phase(10.0, 0, 0)}

    arr, hdr2 = extra.make_waveform_array(header, phd, st)

    with tempfile.TemporaryDirectory() as temp_dir:
        # iteration = 0 implies data is saved in "data" subdir
        os.makedirs(Path(temp_dir) / "data")
        wvf = core.file("waveform_array", sta, pha, it, temp_dir)
        hdf = core.file("waveform_header", sta, pha, it, temp_dir)
        # Save waveform and header to designated files
        np.save(str(wvf), arr)
        hdr2.to_file(hdf)
        wave_arr, hdr3 = io.read_waveform_array_header(sta, pha, it, temp_dir)

    assert wave_arr.shape == (1, 3, sr * tw)
    assert not all(np.flatnonzero(wave_arr[0, 0, :]))
    assert hdr3 == hdr2
