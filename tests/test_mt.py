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
Test the moment tensor functions
"""

from relmt import mt
import numpy as np
import pytest
from relmt import core


def test_mt_tuple():
    mtarr = np.array([[11, 12, 13], [12, 22, 23], [13, 12, 33]])
    res = mt.mt_tuple(mtarr)
    assert res.nn == 11
    assert res.ee == 22
    assert res.dd == 33
    assert res.ne == 12
    assert res.nd == 13
    assert res.ed == 23


# Created by Cvscode Co-pilot
def test_mt_array():
    mt_obj = core.MT(1, 2, 3, 4, 5, 6)
    result = mt.mt_array(mt_obj)
    assert result.shape == (3, 3)


def test_magnitude_of_moment():
    moment = 1122018454.301956
    magnitude = mt.magnitude_of_moment(moment)
    assert pytest.approx(magnitude) == 0.0


def test_moment_of_magnitude():
    magnitude = 0.0
    moment = mt.moment_of_magnitude(magnitude)
    assert pytest.approx(moment) == 1122018454.301956


def test_moment_of_vector_tensor():

    momt1 = core.MT(1e3, 1e3, 1e3, 1e3, 1e3, 1e3)
    momt2 = core.MT(5e3, 5e3, 5e3, 5e3, 5e3, 5e3)

    M01 = mt.moment_of_tensor(mt.mt_array(momt1))
    M02 = mt.moment_of_tensor(mt.mt_array(momt2))

    moment = mt.moment_of_vector(momt1)
    assert pytest.approx(moment) == M01

    moments = mt.moment_of_vector(np.array([momt1, momt2]))
    assert pytest.approx(moments) == [M01, M02]


def test_mean_moment():
    mts = [core.MT(1, 2, 3, 4, 5, 6), core.MT(2, 3, 4, 5, 6, 7)]
    result = mt.mean_moment(mts)
    assert result > 0


def test_mt_tuples_none_constraint():
    mt_vec = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
    result = list(mt.mt_tuples(mt_vec, "none"))
    assert len(result) == 2
    assert result[0] == (1, 2, 3, 4, 5, 6)
    assert result[1] == (11, 12, 13, 14, 15, 16)


def test_mt_tuples_deviatoric_constraint():
    mt_vec = [1, 2, 4, 5, 6, 11, 12, 14, 15, 16]
    result = list(mt.mt_tuples(mt_vec, "deviatoric"))
    assert len(result) == 2
    assert result[0] == (1, 2, -3, 4, 5, 6)
    assert result[1] == (11, 12, -23, 14, 15, 16)


def test_mt_tuples_error():
    mt_vec = [1, 2, 3, 4]
    with pytest.raises(IndexError):
        _ = list(mt.mt_tuples(mt_vec, "none"))


def test_rtf2ned2rtf():
    inmt = core.MT(1, 2, 3, 4, 5, 6)
    mt_rtf = mt.ned2rtf(*inmt)
    mt_ned = mt.rtf2ned(*mt_rtf)
    assert inmt == mt_ned


def test_correlation():
    # Test if an MT correlates perfectly with itself
    mt1 = core.MT(1, 2, 3, 4, 5, 6)
    eta_p, eta_s = mt.correlation(mt1, mt1)
    assert pytest.approx(1) == eta_p
    assert pytest.approx(1) == eta_s


def test_norm_scalar_product():
    # Test if an MT has unity scalar product
    mt1 = core.MT(1, 2, 3, 4, 5, 6)
    res = mt.norm_scalar_product(mt1, mt1)
    assert pytest.approx(1) == res


def test_p_radiation():
    # Unit Double Couple with NE-trending P axis
    M = mt.mt_array(core.MT(0, 0, 0, 1, 0, 0))
    dist = 1.0
    rho = 1.0
    alpha = 1.0

    # Test the nodal planes
    assert mt.p_radiation(M, 0, 0, dist, rho, alpha) == pytest.approx([0, 0, 0])
    assert mt.p_radiation(M, 90, 0, dist, rho, alpha) == pytest.approx([0, 0, 0])
    assert mt.p_radiation(M, 180, 0, dist, rho, alpha) == pytest.approx([0, 0, 0])
    assert mt.p_radiation(M, 270, 0, dist, rho, alpha) == pytest.approx([0, 0, 0])
    assert mt.p_radiation(M, 10, 90, dist, rho, alpha) == pytest.approx([0, 0, 0])
    assert mt.p_radiation(M, 10, -90, dist, rho, alpha) == pytest.approx([0, 0, 0])


def test_s_radiation():
    # Unit Double Couple with NE-trending P axis
    M = mt.mt_array(core.MT(0, 0, 0, 1, 0, 0))
    dist = 1.0
    rho = 1.0
    alpha = 1.0

    # Test the nodal points
    assert mt.s_radiation(M, 45, 0, dist, rho, alpha) == pytest.approx([0, 0, 0])
    assert mt.s_radiation(M, 135, 0, dist, rho, alpha) == pytest.approx([0, 0, 0])
    assert mt.s_radiation(M, 225, 0, dist, rho, alpha) == pytest.approx([0, 0, 0])
    assert mt.s_radiation(M, 315, 0, dist, rho, alpha) == pytest.approx([0, 0, 0])
