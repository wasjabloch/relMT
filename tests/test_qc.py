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
Test the quality control functions
"""

import pytest
import numpy as np
from relmt import qc, core


def test_return_bool_not():
    # Test if boolean arrays are correctly converted
    iin = [True, False, False, True, False]

    assert pytest.approx([0, 3]) == qc._switch_return_bool_not(
        iin, return_not=False, return_bool=False
    )
    assert pytest.approx([1, 2, 4]) == qc._switch_return_bool_not(
        iin, return_not=True, return_bool=False
    )
    assert all(
        iin == qc._switch_return_bool_not(iin, return_not=False, return_bool=True)
    )
    assert all(
        ~np.array(iin)
        == qc._switch_return_bool_not(iin, return_not=True, return_bool=True)
    )


def test_nonzero_events():
    # Test if Non-zero events are detected
    shape = (3, 512)
    arr = np.zeros(shape)
    arr[1, 500] = 1
    assert [1] == qc.index_nonzero_events(arr)
    assert all([False, True, False] == qc.index_nonzero_events(arr, return_bool=True))
    assert pytest.approx([0, 2]) == qc.index_nonzero_events(arr, return_not=True)

    arr[2, 0] = 1
    inz = qc.index_nonzero_events(arr)
    assert inz == pytest.approx([1, 2], abs=0)


def test_nonzero_events_non_finite():
    # Test if Non-zero events are detected
    shape = (3, 512)
    arr = np.ones(shape)
    arr[0, 500] = np.nan
    arr[2, 500] = np.nan
    assert [1] == qc.index_nonzero_events(arr)
    assert all([False, True, False] == qc.index_nonzero_events(arr, return_bool=True))
    assert pytest.approx([0, 2]) == qc.index_nonzero_events(arr, return_not=True)


def test_nonzero_events_3d():
    # Test if Non-zero events are detected
    shape = (3, 3, 512)
    arr = np.zeros(shape)
    arr[1, 0, 500] = 1
    assert [1] == qc.index_nonzero_events(arr)
    assert all([False, True, False] == qc.index_nonzero_events(arr, return_bool=True))
    assert pytest.approx([0, 2]) == qc.index_nonzero_events(arr, return_not=True)


def test_nonzero_events_non_finite_3d():
    # Test if Non-zero events are detected
    shape = (3, 3, 512)
    arr = np.ones(shape)
    arr[0, 1, 500] = np.nan
    arr[2, 1, 500] = np.nan
    assert [1] == qc.index_nonzero_events(arr)
    assert all([False, True, False] == qc.index_nonzero_events(arr, return_bool=True))
    assert pytest.approx([0, 2]) == qc.index_nonzero_events(arr, return_not=True)


# Created by VS-code Co-pilot
def test_index_nonzero_events():
    array = np.array([[0, 0], [1, 0], [0, 1]])
    result = qc.index_nonzero_events(array)
    assert len(result) == 2


def test_index_high_value():
    array = np.array([0, 1, 0, 0, 0])
    result = qc.index_high_value(array, 0.5)
    assert result == pytest.approx([1])

    result = qc.index_high_value(array, 0.5, return_not=True)
    assert result == pytest.approx([0, 2, 3, 4])

    result = qc.index_high_value(array, 0.5, return_bool=True)
    assert all(result == [False, True, False, False, False])


def test_clean_by_station():
    amplitudes = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1.0, 0.1),
        core.P_Amplitude_Ratio("STA2", 0, 1, 1.0, 0.1),
    ]
    result = qc.clean_by_station(amplitudes, ["STA1"])
    assert len(result) == 1
    assert result[0].station == "STA2"


def test_clean_by_misfit():
    amplitudes = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1.0, 0.1),
        core.P_Amplitude_Ratio("STA2", 0, 1, 1.0, 1.5),
    ]
    result = qc.clean_by_misfit(amplitudes, 1.0)
    assert len(result) == 1
    assert result[0].station == "STA1"


def test_clean_by_event_p():
    amplitudes = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1.0, 0.1),
        core.P_Amplitude_Ratio("STA2", 0, 2, 1.0, 1.5),
    ]
    result = qc.clean_by_event(amplitudes, [2])
    assert len(result) == 1
    assert result[0].station == "STA1"


def test_clean_by_event_s():
    amplitudes = [
        core.S_Amplitude_Ratios("STA1", 0, 1, 2, 1.0, 1.0, 0.1),
        core.S_Amplitude_Ratios("STA2", 0, 2, 3, 1.0, 1.0, 1.5),
    ]
    result = qc.clean_by_event(amplitudes, [3])
    assert len(result) == 1
    assert result[0].station == "STA1"


def test_clean_by_magnitude_difference_p():
    events = [
        core.Event(0, 0, 0, 0, 5, "Ev1"),
        core.Event(0, 0, 0, 0, 6, "Ev1"),
        core.Event(0, 0, 0, 0, 7, "Ev1"),
    ]
    amplitudes = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1.0, 0.1),
        core.P_Amplitude_Ratio("STA2", 0, 2, 1.0, 1.5),
    ]
    result = qc.clean_by_magnitude_difference(amplitudes, events, 1.5)
    assert len(result) == 1
    assert result[0].station == "STA1"


def test_clean_by_magnitude_difference_s():
    events = [
        core.Event(0, 0, 0, 0, 5, "Ev1"),
        core.Event(0, 0, 0, 0, 5, "Ev1"),
        core.Event(0, 0, 0, 0, 6, "Ev1"),
        core.Event(0, 0, 0, 0, 7, "Ev1"),
    ]

    amplitudes = [
        core.S_Amplitude_Ratios("STA1", 0, 1, 2, 1.0, 1.0, 0.1),
        core.S_Amplitude_Ratios("STA2", 0, 1, 3, 1.0, 1.0, 0.1),
    ]

    result = qc.clean_by_magnitude_difference(amplitudes, events, 1.5)

    assert len(result) == 1
    assert result[0].station == "STA1"


def test_clean_by_valid_takeoff_angle():
    phd = {
        "0_STA1_P": core.Phase(0, 0, 0),
        "1_STA1_P": core.Phase(0, 0, 0),
        "2_STA1_P": core.Phase(0, np.nan, np.nan),  # Exclude me!
    }

    amplitudes = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1.0, 0.1),
        core.P_Amplitude_Ratio("STA1", 0, 2, 1.0, 0.1),
    ]

    result = qc.clean_by_valid_takeoff_angle(amplitudes, phd)

    assert len(result) == 1
    assert result[0].event_b == 1


def test_clean_by_kurtosis_p():

    amplitudes = [
        core.P_Amplitude_Ratio("STA1", 0, 1, 1.0, 0.1),
        core.P_Amplitude_Ratio("STA1", 0, 2, 1.1, 1.5),
        core.P_Amplitude_Ratio("STA1", 0, 3, 1.2, 1.5),
        core.P_Amplitude_Ratio("STA1", 1, 3, 1e13, 1.5),
    ]

    events = [
        core.Event(0, 0, 0, 0, 1, "Ev0"),
        core.Event(0, 0, 0, 0, 1, "Ev1"),
        core.Event(0, 0, 0, 0, 1, "Ev2"),
        core.Event(0, 0, 0, 0, 9, "Ev3"),
    ]

    result = qc.clean_by_kurtosis(amplitudes, events, -2)

    assert len(result) == 3


def test_clean_by_kurtosis_s():

    amplitudes = [
        core.S_Amplitude_Ratios("STA1", 0, 1, 2, 1.0, 1.0, 0.1),
        core.S_Amplitude_Ratios("STA1", 0, 1, 3, 1.1, 1.1, 1.5),
        core.S_Amplitude_Ratios("STA1", 0, 1, 4, 1.2, 1e-13, 1.5),
        core.S_Amplitude_Ratios("STA1", 0, 2, 4, 1.1, 1.1e-13, 1.5),
        core.S_Amplitude_Ratios("STA1", 0, 3, 4, 0.9, 0.9, 1.5),  # exclude me
    ]

    events = [
        core.Event(0, 0, 0, 0, 1, "Ev0"),
        core.Event(0, 0, 0, 0, 1, "Ev1"),
        core.Event(0, 0, 0, 0, 1, "Ev2"),
        core.Event(0, 0, 0, 0, 1, "Ev3"),
        core.Event(0, 0, 0, 0, 9, "Ev4"),
    ]

    result = qc.clean_by_kurtosis(amplitudes, events, -1)

    assert len(result) == 4


def test_ps_amplitudes_p():
    pamps = [core.P_Amplitude_Ratio("STA1", 0, 1, 0.1, 0.1)]

    ip, _ = qc._ps_amplitudes(pamps)
    assert ip


def test_ps_amplitudes_s():
    samps = [core.S_Amplitude_Ratios("STA1", 0, 1, 3, 0.1, 0.2, 0.1)]
    ip, _ = qc._ps_amplitudes(samps)
    assert not ip
