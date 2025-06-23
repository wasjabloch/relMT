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
from pathlib import Path
from relmt import signal, utils, core, io, extra


def test_concat_wvf_array():
    n = 512
    c = 3
    ev = 10
    inp = np.ones((ev, c, n))
    out = utils.concat_components(inp)
    assert out.shape == (ev, n * c)


def test_pc_index():
    n = 512
    ev = 3
    inp = np.zeros((ev, n))
    inp[1, :] = signal.make_wavelet(n, 10, "cos", 10, 0, 0)
    inp[2, :] = signal.make_wavelet(n, 10, "cos", 15, 0, 0)

    out = utils.pc_index(inp, "P")
    assert out == pytest.approx([1, 2, 0])

    out = utils.pc_index(inp, "S")
    assert out == pytest.approx([2, 0, 1])

    with pytest.raises(ValueError):
        _ = utils.pc_index(inp, "invalid")


def test_corner_frequency_P():
    cf = utils.corner_frequency(5, "P", 5e6, 3500)
    assert cf == pytest.approx(0.9, 1e-3)


def test_corner_frequency_S():
    cf = utils.corner_frequency(5, "S", 5e6, 3500)
    assert cf == pytest.approx(0.5, 1e-2)


def test_xyzarray_sta():
    # Test if xyzarray works with station dictionary format
    xyz = utils.xyzarray(
        {"A": core.Station(1, 2, 3, "A"), "B": core.Station(3, 4, 5, "B")}
    )
    assert xyz == pytest.approx(np.array([[1, 2, 3], [3, 4, 5]]))

    # Single station
    xyz = utils.xyzarray(core.Station(1, 2, 3, "A"))
    assert xyz == pytest.approx(np.array([1, 2, 3]))


def test_xyzarray_ev():
    # Test if xyzarray works with event dictionary format
    xyz = utils.xyzarray(
        [core.Event(1, 2, 3, 0, 5, "a"), core.Event(3, 4, 5, 0, 6, "b")]
    )
    assert xyz == pytest.approx(np.array([[1, 2, 3], [3, 4, 5]]))

    # Single Event
    xyz = utils.xyzarray(core.Event(1, 2, 3, 0, 5, "a"))
    assert xyz == pytest.approx(np.array([1, 2, 3]))


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


def test_phase_dict_azimuth():
    # Test if phases arrival times are estimated correctly
    # Ground truth comes from SKHASH example 'maacama'

    pwd = Path(__file__).parent
    datf = pwd / "data" / "skhash_maacama_example.txt"
    stas = np.loadtxt(datf, usecols=0, dtype=str, skiprows=1)
    slat, slon, sdep, elat, elon, edep = np.loadtxt(
        datf, usecols=(1, 2, 3, 4, 5, 6), unpack=True, skiprows=1
    )

    # Convert to Cartesian
    num, let = extra.get_utm_zone(slat, slon)
    sn, se, sd = extra.geoconverter_latlon2utm(slat, slon, sdep, num, let)
    en, ee, ed = extra.geoconverter_latlon2utm(elat, elon, edep, num, let)

    # Make a station dictionary
    std = {name: core.Station(n, e, d, name) for name, n, e, d in zip(stas, sn, se, sd)}

    # Only one event in list
    evl = [core.Event(en[0], ee[0], ed[0], -1, -1, "0")]

    # Make a phase dictionary
    phd = {
        core.join_phaseid(0, sta, pha): core.Phase(-1, np.nan, -1)
        for sta in std
        for pha in "PS"
    }

    # Actual values
    azimuths = np.loadtxt(datf, usecols=9, unpack=True, skiprows=1)

    newphd = utils.phase_dict_azimuth(phd, evl, std)

    for sta, azimuth in zip(stas, azimuths):
        azp = newphd[core.join_phaseid(0, sta, "P")].azimuth
        azs = newphd[core.join_phaseid(0, sta, "S")].azimuth

        # P and S plunge are the same, because we have a constant Vp/Vs
        assert pytest.approx(azp) == azs

        assert pytest.approx(azimuth, 1e-2) == azp

    # A trial set value
    phd["0_BARR_S"] = core.Phase(-1, 101.0, -1)

    # Let's try not to overwrite the set value
    newphd = utils.phase_dict_azimuth(phd, evl, std, overwrite=False)
    assert newphd["0_BARR_S"].azimuth == 101.0


def test_phase_dict_hash_plunge():
    # Test if phases arrival times are estimated correctly
    # Ground truth comes from SKHASH example 'maacama'

    pwd = Path(__file__).parent
    datf = pwd / "data" / "skhash_maacama_example.txt"
    vmodf = pwd / "data" / "mvel.txt"
    stas = np.loadtxt(datf, usecols=0, dtype=str, skiprows=1)
    slat, slon, sdep, elat, elon, edep = np.loadtxt(
        datf, usecols=(1, 2, 3, 4, 5, 6), unpack=True, skiprows=1
    )

    # Convert to Cartesian
    num, let = extra.get_utm_zone(slat, slon)
    sn, se, sd = extra.geoconverter_latlon2utm(slat, slon, sdep, num, let)
    en, ee, ed = extra.geoconverter_latlon2utm(elat, elon, edep, num, let)

    # Make a station dictionary
    std = {name: core.Station(n, e, d, name) for name, n, e, d in zip(stas, sn, se, sd)}

    # Only one event in list
    evl = [core.Event(en[0], ee[0], ed[0], -1, -1, "0")]

    # Make a phase dictionary
    phd = {
        core.join_phaseid(0, sta, pha): core.Phase(-1, -1, np.nan)
        for sta in std
        for pha in "PS"
    }

    # Actual values
    takeoff = np.loadtxt(datf, usecols=8, unpack=True, skiprows=1)

    # Convert to relMT convention
    plunges = takeoff - 90.0

    # Table has Depth, Vp, Vs
    vmodel = io.read_velocity_model(vmodf, has_kilometer=True)

    newphd = utils.phase_dict_hash_plunge(phd, evl, std, vmodel)

    for sta, plunge in zip(stas, plunges):
        pp = newphd[core.join_phaseid(0, sta, "P")].plunge
        ps = newphd[core.join_phaseid(0, sta, "S")].plunge

        # P and S plunge are the same, because we have a constant Vp/Vs
        assert pytest.approx(pp) == ps

        assert pytest.approx(plunge, abs=2) == pp

    # A trial set value
    phd["0_BARR_S"] = core.Phase(-1, -1, 101.0)

    # Let's try not to overwrite the set value
    newphd = utils.phase_dict_hash_plunge(phd, evl, std, vmodel, overwrite=False)
    assert newphd["0_BARR_S"].plunge == 101.0


def test_interpolate_phd_time():
    # Test if phases arrival times are interpolated correctly
    phd = {"0_A_P": core.Phase(10, 0, 0)}
    std = {"A": (1, 0, 0, "A")}
    evd = {0: (0, 0, 0, 5, -1, "ev0"), 1: (0, 0, 0, 15, -1, "ev1")}
    phdn = utils.interpolate_phase_dict(phd, evd, std)
    assert phdn["1_A_P"][0] == pytest.approx(20)


def test_interpolate_phd_aziinc():
    # Test if phases take-off angles are interpolated correctly
    phd = {"0_A_P": core.Phase(10, 90.1, -20), "1_A_P": core.Phase(10, 89.9, -10)}
    std = {"A": (0, 1, 0)}
    evd = {
        0: (0, 0, 3, 5, -1, "ev0"),
        1: (0, 0, 1, 5, -1, "ev1"),
        2: (0, 0, 2, 5, -1, "ev2"),
    }
    phdn = utils.interpolate_phase_dict(phd, evd, std)
    # Just created this one new key
    assert phdn.keys() == pytest.approx(["2_A_P"])
    # azimuth is along east-axis
    assert phdn["2_A_P"][1] == pytest.approx(90.0)
    # inclination is in the middle between -20 an -10
    assert phdn["2_A_P"][2] == pytest.approx(-15.0)


def test_interpolate_phd_obs_min():
    # Test if phases are interpolated correctly
    phd = {"0_A_P": core.Phase(10, 0, 0)}
    std = {"A": (1, 0, 0)}
    evd = {0: (0, 0, 0, 5, -1, "ev0"), 1: (0, 0, 0, 15, -1, "ev1")}
    phdn = utils.interpolate_phase_dict(phd, evd, std, obs_min=2)
    assert phdn == {}


def test_interpolate_phd_missing_stations():
    # Test if missing stations are handled correctly
    phd = {"0_B_P": (10, 0, 0)}
    std = {"A": (1, 0, 0)}
    evd = {0: (0, 0, 0, 5, -1, "ev0")}
    phdn = utils.interpolate_phase_dict(phd, evd, std)
    assert phdn == {}


def test_concat_wvarray():
    n, c, s = 5, 3, 101
    arr = np.zeros((n, c, s))
    assert utils.concat_components(arr).shape == (n, c * s)


def test_collect_takeoff():
    phd = {
        "0_STA1_P": core.Phase(0, 11.0, 21.0),
        "1_STA1_P": core.Phase(0, 12.0, 22.0),
        "1_STA2_P": core.Phase(0, 13.0, 23.0),
        "1_STA1_S": core.Phase(0, 14.0, 24.0),
    }

    az, pl, st, ph = utils.collect_takeoff(phd, 0)

    assert pytest.approx(az) == [11.0]
    assert pytest.approx(pl) == [21.0]
    assert pytest.approx(st) == ["STA1"]
    assert pytest.approx(ph) == ["P"]

    az, pl, st, ph = utils.collect_takeoff(phd, 1)

    assert pytest.approx(az) == [12.0, 13.0, 14.0]
    assert pytest.approx(pl) == [22.0, 23.0, 24.0]
    assert pytest.approx(st) == ["STA1", "STA2", "STA1"]
    assert pytest.approx(ph) == ["P", "P", "S"]

    az, pl, st, ph = utils.collect_takeoff(phd, 1, stations=["STA1"])

    assert pytest.approx(az) == [12.0, 14.0]
    assert pytest.approx(pl) == [22.0, 24.0]
    assert pytest.approx(st) == ["STA1"] * 2
    assert pytest.approx(ph) == ["P", "S"]


def test_reshape_ccvec():
    ns = 5
    ccin = 0.8
    ijk = list(core.ijk_ccvec(5))
    ccvec = [ccin] * len(ijk)
    ccmat = utils.reshape_ccvec(ccvec, ns)
    for i, j, k in ijk:
        assert ccmat[i, j, k] == ccin


def test_fisher_average():
    ns = 5
    ccin = 0.8
    ijk = list(core.ijk_ccvec(5))
    ccvec = [ccin] * len(ijk)
    ccmat3 = utils.reshape_ccvec(ccvec, ns)
    ccmat2 = utils.fisher_average(ccmat3)

    for i in range(ns):
        for j in range(ns):
            if i == j:
                assert ccmat2[i, j] == 0.0
            else:
                assert pytest.approx(ccmat2[i, j]) == ccin

    ccmat1 = utils.fisher_average(ccmat2)
    assert pytest.approx(ccmat1) == ccin

    ccmat0 = utils.fisher_average(ccmat1)
    assert pytest.approx(ccmat0) == ccin


def test_set_data_window_fftw():
    # Assert that only arguments needed for function are returned
    utils.fftw_data_window(12.0, 100.0, "NEZ") == 12.25


# Created by Cvscode Co-pilot
def test_xyzarray():
    input_data = [
        core.Station(1, 2, 3, "Station 1"),
        core.Station(4, 5, 6, "Station 2"),
    ]
    result = utils.xyzarray(input_data)
    assert result.shape == (2, 3)


def test_cartesian_distance():
    result = utils.cartesian_distance(0, 0, 0, 1, 1, 1)
    assert np.isclose(result, np.sqrt(3))


def test_cartesian_distance_zero():
    result = utils.cartesian_distance(0, 0, 0, 0, 0, 0)
    assert result == 0


def test_next_fftw_size():
    result = utils.next_fftw_size(100)
    assert result > 100
    assert result % 3 == 0


def test_source_duration():
    assert pytest.approx(utils.source_duration(5)) == 1.0
