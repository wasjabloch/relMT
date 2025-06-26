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
Test the main functions
"""

from relmt import main, core, default, io, amp, qc
import pytest

try:
    # This is what IPython likes
    from . import conftest
except ImportError:
    # This is what Pytest likes
    import conftest


def test_main_amplitdue(synthetic_aligned_waveforms):
    """
    Test the main amplitude function
    """

    # For S-wave, an accuracy of 1% can only be archived when excluding at a
    # sigma_1 threshold of 0.8. This is because for many waveforms, Babc and
    # Bacb are not lineraily independent and so can take any value, depending on
    # the exact initialization of the problem. As waveform data is filtered
    # before measuring the amplitdues, the observed and synthetic data are
    # initialized slighly different, leading to drastically different outcomes
    # for both realizations of the (almost) same data. Increasing s1_thrs to
    # 0.85 requires increasing srel to ~3.

    srel = 0.01
    s1_thres = 0.8

    # Get the synthetic aligned waveforms
    mtd = {
        n: momt
        for n, momt in enumerate(
            [
                conftest.mt_rl_strike_slip_north_south_m0(1),
                conftest.mt_rl_strike_slip_north_south_m0(2),
                conftest.mt_ll_strike_slip_north_south_m0(3),
                conftest.mt_ll_strike_slip_north_south_m0(1),
                conftest.mt_normal_dip30_north_m0(2),
                conftest.mt_normal_dip30_east_m0(3),
                conftest.mt_reverse_dip30_east_m0(1),
                conftest.mt_reverse_dip30_north_m0(2),
            ]
        )
    }

    # Events in a 10m circle at 1000m depth
    evl = conftest.event_circle(len(mtd), 10, 1000)

    # 10 Events at the surface, at 1000m distance
    std = conftest.station_circle(50)

    phd = conftest.phases(std, evl)

    # Temporary path, array dictionary, header dictionary
    path, arrd, hdrd = synthetic_aligned_waveforms(mtd, evl, std, phd, 0.0)

    config = default.config
    config["amplitude_filter"] = "manual"

    main.main_amplitude(config, path, 0)

    # Read the created amplitude observation
    pamps = io.read_amplitudes(
        core.file("amplitude_observation", phase="P", directory=path), "P"
    )
    samps = io.read_amplitudes(
        core.file("amplitude_observation", phase="S", directory=path), "S"
    )

    samps = [amp for amp in samps if amp.sigma1 < s1_thres]

    # Read out the observation indices in order
    p_pairs = [(amp.station, amp.event_a, amp.event_b) for amp in pamps]
    s_triplets = [(amp.station, amp.event_a, amp.event_b, amp.event_c) for amp in samps]

    # Create synthetic data in the same order
    p_syn, s_syn, order, p_sig, s_sig = amp.synthetic(
        mtd, evl, std, phd, p_pairs, s_triplets, order=False
    )

    # S-wave degenerecy may as well occurr in the synthetic data.
    s_syn = s_syn[s_sig[:, 0] < s1_thres]

    samps = [amp for n, amp in enumerate(samps) if s_sig[n, 0] < s1_thres]
    s_sig = s_sig[s_sig[:, 0] < s1_thres]

    # Read out the amplitude observation
    p_obs = [amp.amp_ab for amp in pamps]
    s_obs0 = [amp.amp_abc for amp in samps]
    s_obs1 = [amp.amp_acb for amp in samps]

    # Check if they agree
    assert (order == [0, 1, 2]).all()

    # Uncertainty due to filter applied in the oversvation
    assert pytest.approx(p_syn, rel=0.001) == p_obs

    # Large uncertainties due to unhandeld degenerecies
    assert pytest.approx(s_syn[:, 0], rel=srel) == s_obs0
    assert pytest.approx(s_syn[:, 1], rel=srel) == s_obs1
