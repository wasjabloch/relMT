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

import pytest
from relmt import io, core, mt
from datetime import datetime
import zoneinfo
import tempfile
import numpy as np
import os
import yaml
from pathlib import Path


def test_make_read_station_table():
    # Test if a station table is read correctly
    evdin = {"A": (0.0, 0.0, 0.0, "A"), "B": (1.0, 1.0, 1.0, "B")}
    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        tab = io.make_station_table(evdin, fid.name)
        fid.close()
        evdout = io.read_station_table(fid.name)
        codes, ns, es, ds = io.read_station_table(fid.name, unpack=True)
    os.remove(fid.name)

    # Assert that station names are part of the tab string
    assert "A" in tab
    assert "B" in tab

    assert pytest.approx(codes) == ["A", "B"]
    assert pytest.approx(ns) == [0.0, 1.0]
    assert pytest.approx(es) == [0.0, 1.0]
    assert pytest.approx(ds) == [0.0, 1.0]

    # Assert that the dictionary is created correctly
    for (ink, inv), (outk, outv) in zip(evdin.items(), evdout.items()):
        assert ink == outk
        assert inv == outv


def test_save_read_exclude_file():
    excl_in = {
        "phase_manual": ["1_STA1_P"],
        "station": ["STA2"],
        "event": 1,
        "waveform": ["STA1_P"],
    }
    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        io.save_yaml(fid.name, excl_in)
        fid.close()
        excl_out = io.read_exclude_file(fid.name)
    os.remove(fid.name)

    # Assert that the dictionary is created correctly
    for key, val in excl_in.items():
        assert key in excl_out
        assert val == excl_out[key]


def test_make_read_event_table():
    # Test if an event table is read correctly
    evlist1 = [
        core.Event(0.0, 0.0, 0.0, 0.0, 0.0, "Event0"),
        core.Event(1.0, 1.0, 1.0, 1.0, 1.0, "Event1"),
    ]
    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        tab = io.make_event_table(evlist1, fid.name)
        fid.close()
        evlist2 = io.read_event_table(fid.name)
    os.remove(fid.name)

    assert "Event0" in tab
    assert "Event1" in tab

    for ind, outd in zip(evlist1, evlist2):
        assert ind == outd


def test_read_config():
    # Test the io.read_config() function
    filename = "myconfig.yaml"
    config1 = core.Config(ncpu=1, auto_lowpass_method="duration")
    with tempfile.TemporaryDirectory() as tempdir:
        config1.to_file(filename=str(tempdir + filename))
        config2 = io.read_config(str(tempdir + filename))
    assert config1 == config2


def test_read_config_invalid_file():
    with pytest.raises(FileNotFoundError):
        io.read_config("non_existent_file.yaml")


@pytest.fixture
def mock_mt_table_file(tmp_path):
    file = tmp_path / "mt_table.txt"
    data = "1 0.1 0.2 0.3 0.4 0.5 0.6\n"
    data += "2 0.7 0.8 0.9 1.0 1.1 1.2"
    file.write_text(data)
    return file


def test_read_mt_table(mock_mt_table_file):
    result = io.read_mt_table(str(mock_mt_table_file))
    assert isinstance(result, dict)
    assert len(result) == 2


def test_read_mt_table_force_list(mock_mt_table_file):
    result = io.read_mt_table(str(mock_mt_table_file), force_list=True)
    assert isinstance(result, dict)
    assert isinstance(result[1], list)
    assert len(result) == 2


def test_read_mt_table_unpack(mock_mt_table_file):
    result = io.read_mt_table(str(mock_mt_table_file), unpack=True)
    assert isinstance(result, tuple)
    assert len(result) == 7  # eventIDs + 6 MT elements
    assert len(result[0]) == 2  # Event IDs


def test_make_mt_table():
    # Test if a station table is read correctly
    mtin = {0: core.MT(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)}
    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        tab = io.make_mt_table(mtin, fid.name)
        fid.close()
        mtout = io.read_mt_table(fid.name)
    os.remove(fid.name)

    # Assert that something meaningful it written in the table
    assert "5.0" in tab

    # Assert that the dictionary is created correctly
    for (ink, inv), (outk, outv) in zip(mtin.items(), mtout.items()):
        assert ink == outk
        assert inv == outv


def test_make_mt_table_list():
    # Test if a station table is read correctly
    mtin = {
        0: [
            core.MT(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            core.MT(0.1, 1.1, 2.1, 3.1, 4.1, 5.1),
        ]
    }

    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        tab = io.make_mt_table(mtin, fid.name)
        fid.close()
        mtout = io.read_mt_table(fid.name)
    os.remove(fid.name)

    # Assert that something meaningful it written in the table
    assert "5.0" in tab
    assert "5.1" in tab

    # Assert that the dictionary is created correctly
    for (ink, inv), (outk, outv) in zip(mtin.items(), mtout.items()):
        assert ink == outk
        for iv, ov in zip(inv, outv):
            assert iv == ov


@pytest.fixture
def mock_event_table_file(tmp_path):
    file = tmp_path / "event_table.txt"
    data = "0 0.1 0.2 0.3 3000.4 4.5 event1\n"
    data += "1 0.4 0.5 0.6 4000.5 5.0 event2"
    file.write_text(data)
    return file


@pytest.fixture
def mock_broken_event_table_file(tmp_path):
    file = tmp_path / "event_table.txt"
    data = "0 0.1 0.2 0.3 3000.4 4.5 event1\n"
    # The next line has a wrong index
    data += "2 0.4 0.5 0.6 4000.5 5.0 event2"
    file.write_text(data)
    return file


def test_read_event_table(mock_event_table_file):
    result = io.read_event_table(str(mock_event_table_file))
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], core.Event)


def test_read_event_table_error(mock_broken_event_table_file):
    with pytest.raises(IndexError):
        _ = io.read_event_table(str(mock_broken_event_table_file))


def test_read_event_table_unpack(mock_event_table_file):
    result = io.read_event_table(str(mock_event_table_file), unpack=True)
    assert isinstance(result, tuple)
    assert len(result) == 5  # Assuming 5 columns in the event table
    assert len(result[0]) == 2  # Number of events


@pytest.fixture
def mock_bandpass_file(tmp_path):
    file = tmp_path / "bandpass.yaml"
    # Some variations of the ".5g" general format
    data = """
STA1_P:
  0: [5.4152, 9.5668e+05]
  1: [5.4152, 9.5668e+05]
STA1_S:
  0: [5.4152e-05, 95668]
  1: [5.4152e-05, 95668]
    """
    file.write_text(data)
    return file


def test_read_bandpass(mock_bandpass_file):
    bpd = io.read_yaml(str(mock_bandpass_file))
    assert isinstance(bpd, dict)
    assert "STA1_P" in bpd
    assert "STA1_S" in bpd
    assert isinstance(bpd["STA1_P"], dict)
    assert 0 in bpd["STA1_P"]
    assert 1 in bpd["STA1_S"]
    assert pytest.approx(bpd["STA1_P"][0]) == [5.4152, 9.5668e05]


def test_write_bandpass(mock_bandpass_file):
    bpd = io.read_yaml(str(mock_bandpass_file))
    io.save_yaml(str(mock_bandpass_file), bpd, format_bandpass=True)


def test_make_read_phase_table():
    # Test if a phase table is read correctly
    phd1 = {
        core.join_phaseid(0, "STA1", "P"): core.Phase(0.0, 0.0, 0.0),
        core.join_phaseid(0, "STA1", "S"): core.Phase(0.0, 0.0, 0.0),
        core.join_phaseid(1, "STA2", "P"): core.Phase(0.0, 0.0, 0.0),
        core.join_phaseid(1, "STA2", "S"): core.Phase(0.0, 0.0, 0.0),
    }
    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        tab = io.make_phase_table(phd1, fid.name)
        fid.close()
        phd2 = io.read_phase_table(fid.name)
        phids, times, azs, incs = io.read_phase_table(fid.name, unpack=True)
    os.remove(fid.name)

    # Assert that string table is written correctly
    assert "STA1" in tab
    assert "STA2" in tab

    # Assert everyting is unpacked correctly
    assert pytest.approx(phids) == ["0_STA1_P", "0_STA1_S", "1_STA2_P", "1_STA2_S"]
    assert not any(times)
    assert not any(azs)
    assert not any(incs)  # all zeros

    # Assert the dictionary is written correctly
    for (ph1key, ph1val), (ph2key, ph2val) in zip(phd1.items(), phd2.items()):
        assert ph1key == ph2key
        assert ph1val == ph2val


def test_read_waveform_array_header():
    # Test the io.read_config() function
    station = "STA1"
    phase = "P"
    comps = "NEZ"
    data_window = 10.0
    sampling_rate = 20.0
    events = [0, 2, 3]
    def_hdr = core.Header(sampling_rate=sampling_rate, data_window=data_window)
    hdr = core.Header(station=station, phase=phase, components=comps)
    wvarr = np.ones((len(events), len(comps), int(data_window * sampling_rate)))

    with tempfile.TemporaryDirectory() as tempdir:
        os.mkdir(Path(tempdir) / "data")

        # Save the wavform array
        np.save(tempdir / core.file("waveform_array", station, phase), wvarr)

        # Save the header
        hdr.to_file(
            tempdir / core.file("waveform_header", station=station, phase=phase)
        )

        # Load once without a default header
        _ = io.read_waveform_array_header(station, phase, directory=tempdir)

        # Save the default header
        def_hdr.to_file(tempdir / core.file("waveform_header"))

        # Load everything
        wvarr2, hdr2 = io.read_waveform_array_header(station, phase, directory=tempdir)

    assert pytest.approx(wvarr2) == wvarr

    hdr2.update(def_hdr)
    for key in hdr:
        assert hdr[key] == hdr2[key]


def test_save_read_p_amplitudes():
    pamps1 = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1e0, 0.5, 1.0, 0.0, 0.5, 20.0),
        core.P_Amplitude_Ratio("STA1", 0, 2, -1e0, 1.5, 1.0, 0.0, 0.5, 20.0),
    ]

    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        io.save_amplitudes(fid.name, pamps1)
        pamps2 = io.read_amplitudes(fid.name, "P")
        sta, ev1, ev2, amp, mis, sig1, sig2, hpas, lpas = io.read_amplitudes(
            fid.name, "P", unpack=True
        )
    os.remove(fid.name)

    assert pytest.approx(pamps1) == pamps2
    assert pytest.approx(sta) == ["STA1", "STA1"]
    assert pytest.approx(ev1) == [0, 0]
    assert pytest.approx(ev2) == [1, 2]
    assert pytest.approx(amp) == [1e0, -1e0]
    assert pytest.approx(mis) == [0.5, 1.5]
    assert pytest.approx(sig1) == [1.0, 1.0]
    assert pytest.approx(sig2) == [0.0, 0.0]
    assert pytest.approx(hpas) == [0.5, 0.5]
    assert pytest.approx(lpas) == [20.0, 20.0]


def test_save_read_s_amplitudes():
    samps1 = [
        core.S_Amplitude_Ratios(
            "STA1", 0, 1, 3, 1e0, 1e1, 0.5, 0.7, 0.3, 0.0, 0.5, 20.0
        ),
        core.S_Amplitude_Ratios(
            "STA1", 0, 2, 3, -1e0, -1e1, 1.5, 0.7, 0.3, 0.0, 0.5, 20.0
        ),
    ]

    with tempfile.NamedTemporaryFile("w", delete=False) as fid:
        io.save_amplitudes(fid.name, samps1)
        samps2 = io.read_amplitudes(fid.name, "S")
        (
            sta,
            ev1,
            ev2,
            ev3,
            amp12,
            amp13,
            mis,
            sig1,
            sig2,
            sig3,
            hpas,
            lpas,
        ) = io.read_amplitudes(fid.name, "S", unpack=True)
    os.remove(fid.name)

    assert pytest.approx(samps1) == samps2
    assert pytest.approx(sta) == ["STA1", "STA1"]
    assert pytest.approx(ev1) == [0, 0]
    assert pytest.approx(ev2) == [1, 2]
    assert pytest.approx(ev3) == [3, 3]
    assert pytest.approx(amp12) == [1e0, -1e0]
    assert pytest.approx(amp13) == [1e1, -1e1]
    assert pytest.approx(mis) == [0.5, 1.5]
    assert pytest.approx(sig1) == [0.7, 0.7]
    assert pytest.approx(sig2) == [0.3, 0.3]
    assert pytest.approx(sig3) == [0.0, 0.0]
    assert pytest.approx(hpas) == [0.5, 0.5]
    assert pytest.approx(lpas) == [20.0, 20.0]


def test_save_error():

    # This is not allowed
    broken = [("STA1"), ("STA2")]

    with pytest.raises(IndexError):
        io.save_amplitudes("irrelevant", broken)


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
        return (10, 20, 30)

    pwd = Path(__file__).parent
    fn = pwd / "data" / "test_ext_event_table.txt"
    evd = io.read_ext_event_table(fn, 8, 7, 9, 6, 15, 6, my_geoconverter, None, None)
    assert len(evd) == 2
    assert evd[0][:-3] == pytest.approx((10, 20, 30))
    assert evd[1][:-3] == pytest.approx((10, 20, 30))


def test_read_ext_event_table_idconvert():
    # Test if event ID is converted correctly

    def my_idconverter(evid):
        return datetime.fromtimestamp(
            float(evid), tz=zoneinfo.ZoneInfo("UTC")
        ).strftime("%y%m%d%H%M%S")

    pwd = Path(__file__).parent
    fn = pwd / "data" / "test_ext_event_table.txt"

    evd = io.read_ext_event_table(fn, 8, 7, 9, 6, 15, 6, None, None, my_idconverter)
    assert len(evd) == 2
    assert evd[0] == (39.19792, 74.29494, 4.69, 1480103177.08, 4.17, "161125194617")
    assert evd[1] == (39.19689, 74.29641, 4.37, 1480112087.32, 2.53, "161125221447")


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
    assert phd[evid + "_YCH" + "_S"][0] == pytest.approx(t0)


def test_read_nll_hypfile_stacorr_times():
    # Test if Hypfile is translated correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    t0 = 1446833532.44
    phd = io.read_phase_nll_hypfile(
        fn, evid, subtract_residual=False, subtract_stationterm=True
    )
    assert phd[evid + "_YCH" + "_S"][0] == pytest.approx(t0 - 0.33)


def test_read_nll_hypfile_residual_times():
    # Test if Hypfile is translated correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    t0 = 1446833532.44
    phd = io.read_phase_nll_hypfile(
        fn, evid, subtract_residual=True, subtract_stationterm=False
    )
    assert phd[evid + "_YCH" + "_S"][0] == pytest.approx(t0 - 1.1678)


def test_read_nll_hypfile_azimuth():
    # Test if Hypfile is translated correctly
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test.hyp"
    evid = "1511061812"
    phd = io.read_phase_nll_hypfile(fn, evid, False, False)
    assert phd[evid + "_YCH" + "_S"][1] == 273.0


def test_read_nll_hypfile_plunge():
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


def test_make_gmt_meca_input():
    # Check if input for gmt is produced correctly

    # Test list input
    mtl = [core.MT(1, 2, 3, 4, 5, 6), core.MT(1e3, 2e3, 3e3, 4e3, 5e3, 6e3)]

    # Test dict input
    mtd = {i: momt for i, momt in enumerate(mtl)}

    evl = [
        core.Event(10, 20, 30, np.nan, 1, "Ev0"),
        core.Event(100, 200, 300, np.nan, 2, "Ev1"),
    ]

    def geoconverter(n, e, d):
        return n * 1e-3, e * 1e-4, d * 1e-5

    with tempfile.NamedTemporaryFile("w") as fid:
        tab1 = io.make_gmt_meca_table(mtl, evl, geoconverter, fid.name)
        tab2 = io.make_gmt_meca_table(mtd, evl, geoconverter, fid.name)

    mtcomps0 = np.array(mt.ned2rtf(*mtl[0]))
    mtcomps1 = np.array(mt.ned2rtf(*mtl[1])) / 1e3

    # Both tables are the same
    assert tab1 == pytest.approx(tab2)

    # All input has been converted correctly
    assert tab1[0, :] == pytest.approx([20e-4, 10e-3, 30e-5, *mtcomps0, -7])
    assert tab1[1, :] == pytest.approx([2e-2, 1e-1, 3e-3, *mtcomps1, -4])


def test_read_ext_mt_table():
    pwd = Path(__file__).parent
    fn = pwd / "data" / "test_ext_mt_table.txt"
    mt0 = core.MT(-24.054e15, 18.344e15, 5.716e15, 19.423e15, 2.462e15, -11.999e15)
    mt1 = core.MT(-26.009e15, -3.779e15, 29.788e15, 17.093e15, 38.050e15, -21.978e15)

    def name_converter(date: str) -> str:
        # Convert '/' seperated date into '-' seperated date
        return date.replace("/", "-")

    def mt_converter(mrr, mtt, mff, mrt, mrf, mtf, exp):
        # Convert GMT convention into NED convention
        fac = 10**exp * 1e-7  # dyne cm -> Nm
        dd = mrr * fac
        nd = mrt * fac
        nn = mtt * fac
        ed = -mrf * fac
        ee = mff * fac
        ne = -mtf * fac
        return nn, ee, dd, ne, nd, ed

    evl = [
        core.Event(10, 20, 30, 40, 50, "2015-12-07"),  # In MT table
        core.Event(11, 21, 31, 41, 51, "2016-01-18"),  # Not in MT table
        core.Event(12, 22, 32, 42, 52, "2017-05-22"),  # In MT table
    ]

    mt_dict = io.read_ext_mt_table(
        fn,
        (6, 7, 8, 9, 10, 11),
        exponent_index=12,
        mtconverter=mt_converter,
    )

    # All event are found when no event name is matched.
    # But we did not attempt any event association
    assert tuple(mt_dict.keys()) == (0, 1)  # Events present in table
    assert mt_dict[0] == pytest.approx(mt0)
    assert mt_dict[1] == pytest.approx(mt1, rel=5e-2)  # Rounding error

    mt_dict = io.read_ext_mt_table(
        fn,
        nn_ee_dd_ne_nd_ed_indices=(6, 7, 8, 9, 10, 11),
        name_index=0,  # Date
        event_list=evl,
        exponent_index=12,
        mtconverter=mt_converter,
        nameconverter=name_converter,
    )

    assert tuple(mt_dict.keys()) == (0, 2)  # Events present in table
    assert mt_dict[0] == pytest.approx(mt0)
    assert mt_dict[2] == pytest.approx(mt1, rel=5e-2)
