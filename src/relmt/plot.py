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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def wvmatrix(
    wvm: np.array, hdr: dict = {}, scale: float = -1.0, ax: Axes | None = None
) -> Axes:
    """
    Plot seismograms in waveform matrix containing N events, C channels and S samples

    Parameters
    ----------
    wvm : (N,C,S) :class:`~numpy.array`:
        Waveform matrix
    hdr : `dict`
        Header dictionary
    scale : `float`
        < 0 each trace is scaled to its maximum amplitude (default)
        = 0 each trace is scaled to the maximum amplitude of the section
        > 0 then each trace is scaled to that amplitude
    ax : :class:`matplotlib.axes.Axes`
        When supplied, plot into this axis instead of creating a new figure

    Returns:
    --------
    ax : :class:`matplotlib.axes.Axes`
        Axis containing the plot
    """

    ne, nc, ns = np.shape(wvm)  # events, channels, samples
    ny = ne * nc  # seismograms in y direction
    xymat = np.zeros((ny, ns))

    # Maximum per component
    if scale < 0:
        iy = 0
        for ie in range(ne):
            for ic in range(nc):
                xymat[iy, :] = (
                    iy
                    - 0.2 * (ic - 1)
                    - 0.5 * wvm[ie, ic, :] / np.max(np.absolute(wvm[ie, ic, :]))
                )
                iy += 1

    # Maximum per section
    elif scale == 0:
        iy = 0
        for ie in range(ne):
            for ic in range(nc):
                xymat[iy, :] = iy - 8.0 * wvm[ie, ic, :] / np.max(np.absolute(wvm))

                iy += 1

    # Unscaled
    else:
        iy = 0
        for ie in range(ne):
            for ic in range(nc):
                xymat[iy, :] = iy - 1.0 * wvm[ie, ic, :] / scale
                iy += 1

    # Default axes
    xlabel = "Samples"
    time = np.arange(0, ns)
    t0 = ns / 2

    ylabel = "Trace #"
    yt = np.arange(ne) * nc + 1
    ytl = ["{:d}".format(n) for n in np.arange(ne)]
    left = 0.1

    # Overwrite defaults with axes from header
    if hdr:
        xlabel = "Time (s)"
        time = np.arange(-ns / 2, ns / 2) / hdr["sr"]
        t0 = 0

        ylabel = "ID"
        yt = np.arange(len(hdr["ids"])) * nc + 1
        ytl = hdr["ids"]
        left = 0.23  # Make some space to accomodate labels

    if ax is None:
        _, ax = plt.subplots(
            gridspec_kw={"top": 0.99, "bottom": 0.1, "right": 0.95, "left": left}
        )

    # Seismograms
    ax.plot(time, xymat.T, "k", linewidth=1)

    # Pick location
    ax.axvline(t0)

    # Time left to right
    ax.set_xlim((time[0], time[-1]))
    ax.set_xlabel(xlabel)

    # Seismograms top to bottom
    ax.set_ylim([ny, -1])
    ax.set_ylabel(ylabel)
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl)

    # Channel names to the right
    if "chs" in hdr:
        axx = ax.twinx()
        axx.yaxis.tick_right()
        axx.set_yticks(np.arange(ny))
        axx.set_yticklabels(hdr["chs"] * ne)
        axx.set_ylim([ny, -1])

    return ax


def section(
    seis: np.array,
    beg: float = 0.0,
    dt: float = 1.0,
    aflag: int = -1,
    ax: Axes | None = None,
) -> Axes:
    """
    SECTION(SEIS,BEG,DT,AFLAG) plots a seismogram section of
    seismograms in 2D array SEIS. BEG is begin time of section, DT
    is sample interval and AFLAG determines amplitude scaling. If
    AFLAG < 0 each trace is scaled to its maximum amplitude (default),
    if AFLAG = 0 each trace is scaled to the maximum amplitude of the
    entire section, and if AFLAG > 0 then each trace is scaled to
    that amplitude (note this option allows direct comparison of
    plots produced by different calls to SECTION).
    """

    ny, nt = np.shape(seis)
    xymat = np.zeros((ny, nt))

    eps = np.spacing(1)

    if aflag < 0:
        for iy in range(0, ny):
            xymat[iy, :] = (iy + 1) - 0.7 * seis[iy, :] / max(
                np.absolute(seis[iy, :]) + eps
            )

    elif aflag == 0:
        for iy in range(0, ny):
            xymat[iy, :] = (iy + 1) - 8.0 * seis[iy, :] / max(
                max(np.absolute(seis)) + eps
            )

    else:
        for iy in range(0, ny):
            xymat[iy, :] = (iy + 1) - 1.0 * seis[iy, :] / aflag

    time = np.arange(0, nt) * dt + beg
    ttime = np.tile(time, [ny, 1])

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(ttime.transpose(), xymat.transpose(), "k", linewidth=1)
    ax.set_ylim([ny + 1, 0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trace #")

    return ax
