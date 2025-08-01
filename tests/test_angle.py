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
Test the the angle functions
"""

import pytest
import numpy as np
from relmt import angle, io, core
from scipy.interpolate import interpn
from pathlib import Path


def test_hash_plunge_table():
    # Test if we can reproduce some of the SKHASH maacama example
    pwd = Path(__file__).parent
    datf = pwd / "data" / "skhash_maacama_example.txt"
    vmodf = pwd / "data" / "mvel.txt"
    depth, dists, takeoff = np.loadtxt(datf, usecols=(6, 7, 8), unpack=True, skiprows=1)

    # Convert to relMT convention
    depth *= 1e3
    dists *= 1e3
    plunge = takeoff - 90.0

    # Table has Depth, Vp, Vs
    vmodel = io.read_velocity_model(vmodf, has_kilometer=True)

    # Make sure vecorts start at 0
    disv = np.linspace(0, max(dists), 100)
    depv = np.linspace(0, 2 * depth[0], 100)

    # Supply only Vp
    # Output is distance, depth
    tab = angle.hash_plunge_table(vmodel[:, :2], depv, disv, nray=5000)

    plunge_out = interpn((disv, depv), tab, np.array([dists, depth]).T)

    # Near-horizontal angle is least stable. Increasing distance and depth
    # sampling helps at the cost of longer runtimes
    assert pytest.approx(plunge_out, abs=2) == plunge


def test_azimuth():
    result = angle.azimuth(0, 0, 1, 1)
    assert np.isclose(result, 45.0)


def test_plunge():
    result = angle.plunge(0, 0, 0, 0, 0, -1)
    assert np.isclose(result, -90.0)

    result = angle.plunge(0, 0, 0, 10, 12, 0)
    assert np.isclose(result, 0.0)

    result = angle.plunge(100, 0.0, 100, 0, 0, 0)
    assert np.isclose(result, -45.0)


def test_azimuth_gap():
    pamps = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1.0, 0.1, 0.33, 0.9, 0.1, 0.5, 20.0),
        core.P_Amplitude_Ratio("STA2", 0, 2, 1.1, 1.5, 0.34, 0.9, 0.1, 0.5, 20.0),
    ]

    samps = [
        core.S_Amplitude_Ratios(
            "STA3", 0, 1, 2, 1.0, 11.2, 0.1, 0.33, 0.9, 0.1, 0.0, 0.5, 20.0
        ),
        # Below an amplitude without observation
        core.S_Amplitude_Ratios(
            "STA4", 0, 1, 2, 1.0, 11.2, 0.1, 0.33, 0.9, 0.1, 0.0, 0.5, 20.0
        ),
    ]

    phd = {
        "0_STA1_P": core.Phase(0, 10, 0),
        "0_STA2_P": core.Phase(0, 180, 0),
        "0_STA3_S": core.Phase(0, 355, 0),
    }

    gaps = [175, 170, 15]

    gapd = angle.azimuth_gap(phd, pamps, samps)
    assert pytest.approx(gapd[0]) == gaps
    assert 1 not in gapd
    assert 2 not in gapd  # No azimuth info available
