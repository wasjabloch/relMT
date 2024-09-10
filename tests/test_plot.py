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
Test the plotting functions
"""

import numpy as np
import yaml
from relmt.plot import section, wvmatrix
from relmt import utils
import matplotlib.pyplot as plt
from pathlib import Path


def _signal_to_plot():
    return np.vstack([utils.make_wavelet(64, 10, "sin", 30, 0, 0) for _ in range(5)])


def test_wvmatrix(iplot=False):
    if iplot:
        plt.ion()
    pwd = Path(__file__).parent
    wvf = pwd / "data" / "subcatalog_A.mats.npy"
    hdf = pwd / "data" / "header_A.yaml"
    mats = np.load(wvf)
    with open(hdf, "r") as fid:
        hdr = yaml.safe_load(fid)

    wvmatrix(mats)
    wvmatrix(mats, hdr)
    wvmatrix(mats, hdr, 0)
    wvmatrix(mats, hdr, 1)

    _, ax = plt.subplots()
    wvmatrix(mats, ax=ax)


def test_section(iplot=False):
    if iplot:
        plt.ion()

    section(_signal_to_plot())
    _, ax = plt.subplots()
    section(_signal_to_plot(), ax=ax)
