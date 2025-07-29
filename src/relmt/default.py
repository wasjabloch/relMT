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

"""Default configuration and header values for relMT."""

from relmt import core


config = core.Config(
    event_file=core.file("event"),
    station_file=core.file("station"),
    phase_file=core.file("phase"),
    reference_mt_file=core.file("reference_mt"),
    amplitude_suffix="",
    result_suffix="",
    compute_synthetics=False,
    solve_synthetics=False,
    reference_mts=[0],
    mt_constraint="none",
    reference_weight=1000.0,
    min_amplitude_misfit=0.1,
    max_amplitude_misfit=0.8,
    max_s_sigma1=0.95,
    amplitude_measure="indirect",
    amplitude_filter="auto",
    auto_lowpass_method="duration",
    auto_lowpass_stressdrop_range=[1.0e6, 1.0e8],
    auto_bandpass_snr_target=0.0,
    min_dynamic_range=2.0,
    min_equations=8,
    max_magnitude_difference=float("inf"),
    max_event_distance=float("inf"),
    bootstrap_samples=100,
    ncpu=1,
)

header = core.Header(
    station="STATION",
    phase="P",
    components="ZNE",
    variable_name="",
    sampling_rate=100,
    events=[0, 1, 2],
    data_window=10.0,
    phase_start=-1.0,
    phase_end=2.0,
    taper_length=1.0,
    highpass=0.1,
    lowpass=10.0,
    null_threshold=0.0,
    min_signal_noise_ratio=0.0,
    min_expansion_coefficient_norm=0.5,
)
