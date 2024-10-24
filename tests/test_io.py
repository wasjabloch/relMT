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
Test the in- and output functions
"""

from relmt import io, core, utils
from pathlib import Path
import tempfile
import numpy as np
import os
from obspy import UTCDateTime, Trace, Stream

import pytest


def my_geoconverter(lat, lon, dep):
    return io.geoconverter_latlon2utm(lat, lon, dep, 42, "S")


def test_obspy_inventory_files():
    # Test if inventory file is read in and content is consitent
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = io.read_obspy_inventory_files(fns)
    assert len(inv) == 1
    assert len(inv.get_contents()["stations"]) == 2


def test_utmzone():
    # Check if UTM zone is returned coorrectly
    num, let = io.get_utm_zone(latitudes=[38.6, 37.5], longitudes=[69.1, 71.6])
    assert num == 42
    assert let == "S"


def test_utmzone_err1():
    # Check if LookupError is raised
    with pytest.raises(LookupError):
        # One station in Tajikistan, the other on the equator!
        _, _ = io.get_utm_zone(latitudes=[38.6, 0], longitudes=[69.1, 71.6])


def test_utmzone_err2():
    # Check if LookupError is raised
    with pytest.raises(LookupError):
        # 2nd is on null meridian
        _, _ = io.get_utm_zone(latitudes=[38.6, 37.5], longitudes=[69.1, 0])


def test_read_station_inventory():
    # See if a station table is retuned coorectly
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = io.read_obspy_inventory_files(fns)
    stad = io.read_station_inventory(inv, my_geoconverter)

    # Has 2 header lines + 2 stations
    assert "CHGR" in stad
    assert "MANEM" in stad
    assert stad["CHGR"] == pytest.approx((4278715, 513765, -1049), abs=1)


def test_read_station_inventory2():
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = io.read_obspy_inventory_files(fns)
    # Test if alternate and historic codes are handled correctly
    inv[0][0].alternate_code = "ALT"
    inv[0][1].historical_code = "HIST"
    stad = io.read_station_inventory(inv, my_geoconverter)
    assert len(stad) == 4
    assert "ALT" in stad
    assert "HIST" in stad


def test_read_station_inventory_error():
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = io.read_obspy_inventory_files(fns)
    # Test if double codes write an error
    inv[0][0].alternate_code = "MANEM"
    with pytest.raises(KeyError):
        _ = io.read_station_inventory(inv, my_geoconverter)

    # But don't, when they should not
    _ = io.read_station_inventory(inv, my_geoconverter, strict=False)


def test_read_station_inventory_error2():
    pwd = Path(__file__).parent
    fns = [pwd / "data" / "test-net.txt"]
    inv = io.read_obspy_inventory_files(fns)
    # Test if underscore produces an error
    inv[0][0].alternate_code = "MA_EM"
    with pytest.raises(ValueError):
        _ = io.read_station_inventory(inv, my_geoconverter)


def test_read_nll_hypfile_keys():
    # Test if Hypfile is translated correctly into dictionary
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    phd = io.read_phase_nll_hypfile(fn, evid, False, False)
    assert evid + "_YCH" + "_S" in phd
    assert evid + "_EP27" + "_P" in phd
    assert evid + "_EP27" + "_S" in phd
    assert evid + "_EP07" + "_P" in phd


def test_read_nll_hypfile_raw_times():
    # Test if Hypfile is translated correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    t0 = 1446833532.44
    phd = io.read_phase_nll_hypfile(fn, evid, False, False)
    assert phd[evid + "_YCH" + "_S"][0] == t0


def test_read_nll_hypfile_stacorr_times():
    # Test if Hypfile is translated correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    t0 = 1446833532.44
    phd = io.read_phase_nll_hypfile(
        fn, evid, substract_residual=False, substract_stationterm=True
    )
    assert phd[evid + "_YCH" + "_S"][0] == pytest.approx(t0 - 0.33)


def test_read_nll_hypfile_residual_times():
    # Test if Hypfile is translated correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    t0 = 1446833532.44
    phd = io.read_phase_nll_hypfile(
        fn, evid, substract_residual=True, substract_stationterm=False
    )
    assert phd[evid + "_YCH" + "_S"][0] == pytest.approx(t0 - 1.1678)


def test_read_nll_hypfile_azimuth():
    # Test if Hypfile is translated correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    phd = io.read_phase_nll_hypfile(fn, evid, False, False)
    assert phd[evid + "_YCH" + "_S"][1] == 93.0


def test_read_nll_hypfile_inclination():
    # Test if Hypfile is translated correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    phd = io.read_phase_nll_hypfile(fn, evid, False, False)
    assert phd[evid + "_YCH" + "_S"][2] == 90 - 100.7


def test_make_phase_table(iprint=False):
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    phd = io.read_phase_nll_hypfile(fn, evid, False, False)
    tab = io.make_phase_table(phd)

    if iprint:
        print(tab)

    assert len(tab.splitlines()) == 6
    assert len(tab.splitlines()[-1].split()) == 6


def test_read_phase_table():
    # Test if a phase table is read correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    phd = io.read_phase_nll_hypfile(fn, evid, False, False)
    tab = io.make_phase_table(phd)
    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        fid.write(tab)
        fid.close()
        phd2 = io.read_phase_table(fid.name)
    os.remove(fid.name)

    for phid, phid2 in zip(phd, phd2):
        assert phid == phid2

    for (t1, azi1, inc1), (t2, azi2, inc2) in zip(phd.values(), phd2.values()):
        assert t1 == pytest.approx(t2)
        assert azi1 == pytest.approx(azi2)
        assert inc1 == pytest.approx(inc2)


def test_geoconverter_latlon2utm():
    # Test if UTM conversion works
    n, e, d = io.geoconverter_latlon2utm(39.198, 74.295, 4.69, 43, "S")
    assert int(n) == 4338985
    assert int(e) == 439123
    assert int(d) == 4690


def test_make_read_station_table():
    # Test if a station table is read correctly
    evdin = {"A": (0.0, 0.0, 0.0), "B": (1.0, 1.0, 1.0)}
    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        tab = io.make_station_table(evdin)
        fid.write(tab)
        fid.close()
        evdout = io.read_station_table(fid.name)
    os.remove(fid.name)

    for ind, outd in zip(evdin, evdout):
        assert ind == outd


def test_make_read_event_table():
    # Test if an event table is read correctly
    evlist1 = [
        core.Event(0.0, 0.0, 0.0, 0.0, 0.0, "0"),
        core.Event(1.0, 1.0, 1.0, 1.0, 1.0, "1"),
    ]
    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        tab = io.make_event_table(evlist1)
        fid.write(tab)
        fid.close()
        evlist2 = io.read_event_table(fid.name)
    os.remove(fid.name)

    for ind, outd in zip(evlist1, evlist2):
        assert ind == outd


def test_read_ext_event_table():
    # Test if an event table is read correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test_ext_event_table.txt"
    evd = io.read_ext_event_table(fn, 8, 7, 9, 6, 15, 6)
    assert len(evd) == 2
    assert evd[0] == (39.19792, 74.29494, 4.69, 1480103177.08, 4.17, "1480103177.08")
    assert evd[1] == (39.19689, 74.29641, 4.37, 1480112087.32, 2.53, "1480112087.32")


def test_read_ext_event_table_key_error():
    # Test if an key error is raised with forbidden key
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test_ext_event_table.txt"
    with pytest.raises(KeyError):
        _ = io.read_ext_event_table(
            fn, 8, 7, 9, 6, 15, 6, loadtxt_kwargs={"unpack": False}
        )


def test_read_ext_event_table_geoconvert():
    # Test if coordinates are converted correctly
    def my_geoconverter(lat, lon, dep):
        return io.geoconverter_latlon2utm(lat, lon, dep, 43, "S")

    pwd = Path(__file__).parent
    fn = pwd / "data" / "test_ext_event_table.txt"
    evd = io.read_ext_event_table(fn, 8, 7, 9, 6, 15, 6, my_geoconverter, None, None)
    assert len(evd) == 2
    assert evd[0][:-3] == pytest.approx((4338977, 439117, 4690), abs=1)
    assert evd[1][:-3] == pytest.approx((4338861, 439243, 4370), abs=1)


def test_read_ext_event_table_idconvert():
    # Test if event ID is converted correctly

    def my_idconverter(evid):
        return UTCDateTime(float(evid)).strftime("%y%m%d%H%M%S")

    pwd = Path(__file__).parent
    fn = pwd / "data" / "test_ext_event_table.txt"

    evd = io.read_ext_event_table(fn, 8, 7, 9, 6, 15, 6, None, None, my_idconverter)
    assert len(evd) == 2
    assert evd[0] == (39.19792, 74.29494, 4.69, 1480103177.08, 4.17, "161125194617")
    assert evd[1] == (39.19689, 74.29641, 4.37, 1480112087.32, 2.53, "161125221447")


def test_make_event_table(iprint=False):
    # Test if an event table is retuned coorectly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test_ext_event_table.txt"
    evd = io.read_ext_event_table(fn, 8, 7, 9, 6, 15, 6)
    table = io.make_event_table(evd)

    if iprint:
        print(table)

    # Has 2 header lines + 2 events
    assert len(table.splitlines()) == 4


def test_make_waveform_array():
    sampling_rate = 100
    time_window = 2
    stream = Stream(
        [
            Trace(
                utils.make_wavelet(5000, 50, we=100, de=-1500),
                header={
                    "station": "A",
                    "channel": f"HH{cha}",
                    "delta": 1 / sampling_rate,
                    "starttime": UTCDateTime(t0),
                },
            )
            for cha in "ZNE"
            for t0 in [0.0, 100.0]
        ]
    )

    phase_dict = {
        "0_A_P": (10.0, 0, 0),
        "0_A_S": (15.0, 0, 0),
        "1_A_P": (110.0, 0, 0),
        "1_A_S": (115.0, 0, 0),
    }

    wave_dict = io.make_waveform_arrays(stream, phase_dict, time_window, sampling_rate)
    assert len(wave_dict) == 2
    assert wave_dict["A_P"].shape == (2, 3, sampling_rate * time_window + 1)
    assert not all(np.flatnonzero(wave_dict["A_S"][0, 0, :]))


def test_read_waveform_array():
    sr = 100
    tw = 2
    st = Stream(
        [
            Trace(
                utils.make_wavelet(5000, 50, we=100, de=-1500),
                header={
                    "station": "A",
                    "channel": f"HH{cha}",
                    "delta": 1 / sr,
                    "starttime": UTCDateTime(0),
                },
            )
            for cha in "ZNE"
        ]
    )

    phd = {"0_A_P": (10.0, 0, 0)}

    wvdict = io.make_waveform_arrays(st, phd, tw, sr)

    with tempfile.TemporaryDirectory() as temp_dir:
        fil = Path(temp_dir) / "A_P.npy"
        np.save(str(fil), wvdict["A_P"])
        wave_arr = io.read_waveform_file(str(fil))

    assert wave_arr.shape == (1, 3, sr * tw + 1)
    assert not all(np.flatnonzero(wave_arr[0, 0, :]))


def test_read_config():
    # Test the io.read_config() function
    filename = "myconfig.yaml"
    config1 = core.Config(sampling_rate=1, data_window=1, exclude_events=[1, 2, 3])
    with tempfile.TemporaryDirectory() as tempdir:
        config1.to_file(filename=str(tempdir + filename))
        config2 = io.read_config(str(tempdir + filename))
    assert config1 == config2
