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

"""Plotting utilities for relMT"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
from relmt import core, mt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)

norsar_lightblue = (0.0, 145 / 255, 214 / 255)
norsar_gray = (156 / 255, 188 / 255, 205 / 255)


@core._doc_config_args
def section_3d(
    arr: np.ndarray,
    scale: float = -1.0,
    ax: Axes | None = None,
    sampling_rate: float | None = None,
    components: str | None = None,
    station: str | None = None,
    events: list | None = None,
    phase: str | None = None,
    plot_kwargs: dict = {},
) -> Axes:
    """
    Plot seismograms in waveform matrix containing N events, C components and S samples

    Parameters
    ----------
    arr:
        ``(events, components, samples)`` Waveform matrix
    scale:
        * ``< 0`` each trace is scaled to its maximum amplitude (default)
        * ``= 0`` each trace is scaled to the maximum amplitude of the section
        * ``> 0`` then each trace is scaled to that amplitude
    ax:
        When supplied, plot into this axis instead of creating a new figure
    plot_kwargs:
        Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`

    Returns
    -------
    Axis containing the plot
    """

    plot_defaults = {"color": "black", "linewidth": 1}
    plot_defaults.update(plot_kwargs)

    ne, nc, ns = np.shape(arr)  # events, components, samples
    ny = ne * nc  # seismograms in y direction
    xymat = np.zeros((ny, ns))

    # Maximum per component
    if scale < 0:
        it = 0
        for ie in range(ne):
            for ic in range(nc):
                xymat[it, :] = (
                    it
                    - 0.2 * (ic - 1)
                    - 0.5
                    * -scale
                    * arr[ie, ic, :]
                    / np.max(np.absolute(arr[ie, ic, :]))
                )
                it += 1

    # Maximum per section
    elif scale == 0:
        it = 0
        for ie in range(ne):
            for ic in range(nc):
                xymat[it, :] = it - 8.0 * arr[ie, ic, :] / np.max(np.absolute(arr))

                it += 1

    # Unscaled
    else:
        it = 0
        for ie in range(ne):
            for ic in range(nc):
                xymat[it, :] = it - 1.0 * arr[ie, ic, :] / scale
                it += 1

    # Default axes
    xlabel = "Samples"
    time = np.arange(0, ns)
    t0 = ns / 2

    ylabel = "Event #"
    yt = np.arange(ne) * nc + 1
    ytl = ["{:d}".format(n) for n in np.arange(ne)]
    left = 0.1

    if events is not None:
        if len(events) != ne:
            msg = f"Number of events in array ({ne}) unequal length of 'events'"
            raise IndexError(msg)
        yt = np.arange(ne) * nc + 1
        ytl = ["{:d}".format(ev) for ev in events]

    # Overwrite defaults with axes from header
    if sampling_rate is not None:
        xlabel = "Time (s)"
        time = np.arange(-ns / 2, ns / 2) / sampling_rate
        t0 = 0

    if ax is None:
        _, ax = plt.subplots(
            gridspec_kw={"top": 0.99, "bottom": 0.1, "right": 0.95, "left": left}
        )

    if station is not None:
        title = f"Station: {station}"
        if phase is not None:
            title += f" ({phase})"
        ax.set_title(title)

    # Seismograms
    ax.plot(time, xymat.T, **plot_defaults)

    # Pick location
    ax.axvline(t0, color=norsar_lightblue, zorder=-5)

    # Time left to right
    ax.set_xlim((time[0], time[-1]))
    ax.set_xlabel(xlabel)

    # Seismograms top to bottom
    ax.set_ylim([ny, -1])
    ax.set_ylabel(ylabel)
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl)

    # Channel names to the right
    if components is not None:
        axx = ax.twinx()
        axx.yaxis.tick_right()
        axx.set_yticks(np.arange(ny))
        axx.set_yticklabels(components * ne)
        axx.set_ylim([ny, -1])

    return ax


def section_2d(
    mat: np.ndarray,
    scaling: float | None = None,
    time_shift: float = 0.0,
    sampling_rate: float = 1.0,
    components: str | None = None,
    ax: Axes | None = None,
    wiggle: bool = True,
    image: bool = False,
    plot_kwargs: dict = {},
    image_kwargs: dict = {},
) -> Axes:
    """
    Plot a section of seismograms in 2D array.

    Parameters
    ----------
    mat:
        ``(events, samples)`` Waveform matrix
    scaling:
        If `None`, scale each trace to its absolute maximum value. If 0, scale
        by absolute value of the section. If non-zero float, scale each trace by
        this factor.
    time_shift:
        Add time shift to tick labels (seconds)
    sampling_rate:
        Sampling rate of the seismograms (Hertz)
    components:
        String of channel names in order
    ax:
        Place plot in existing axis. If None, create an axis
    wiggle:
        Produce a wiggle plot
    image:
        Produce a colored image plot
    plot_kwargs:
        Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`
        (wiggle plot)
    image_kwargs:
        Additional keyword arguments passed to :func:`matplotlib.pyplot.imshow`
        (image plot)

    Returns
    -------
    Axis containing the plot
    """

    plot_defaults = {"color": "black", "linewidth": 1, "zorder": 5}
    plot_defaults.update(plot_kwargs)

    ny, nt = np.shape(mat)
    xymat = np.zeros((ny, nt))

    if scaling is None:
        for it in range(0, ny):
            xymat[it, :] = it + 0.5 * mat[it, :] / np.nanmax(np.absolute(mat[it, :]))

    elif scaling == 0:
        for it in range(0, ny):
            xymat[it, :] = it + mat[it, :] / np.nanmax(np.absolute(mat))

    else:
        for it in range(0, ny):
            xymat[it, :] = it + mat[it, :] / scaling

    time = np.arange(0, nt) / sampling_rate + time_shift
    ttime = np.tile(time, [ny, 1])

    if ax is None:
        _, ax = plt.subplots()

    if wiggle:
        ax.plot(ttime.transpose(), xymat.transpose(), **plot_defaults)

    if image:
        # Subtract the traces offset
        immat = (
            xymat - np.ones_like(xymat) * np.array(range(xymat.shape[0]))[:, np.newaxis]
        )
        image_defaults = dict(
            cmap="RdBu_r",
            aspect="auto",
            vmin=np.min(immat),
            vmax=np.max(immat),
            extent=(time[0], time[-1], ny - 0.5, -0.5),
            interpolation="none",
            zorder=-6,
        )
        image_defaults.update(image_kwargs)
        ax.imshow(immat, **image_defaults)

    if components:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ncha = len(components)
        xlines = [(n + 1) * (time[-1] - time[0]) / ncha for n in range(ncha)]
        xtext = [(n + 0.5) * (time[-1] - time[0]) / ncha for n in range(ncha)]
        for n, (cha, xt, xl) in enumerate(zip(components, xtext, xlines)):
            ax.text(xt, 1, cha, ha="center", va="bottom", transform=trans)
            if n < (len(components) - 1):
                ax.axvline(xl, color=norsar_gray, zorder=-5)

    ax.set_ylim([ny, -1])
    ax.set_ylabel("Trace #")

    ax.set_xlim((time[0], time[-1]))
    ax.set_xlabel("Time (s)")

    return ax


def p_reconstruction(
    wvfA: np.ndarray, wvfB: np.ndarray, Aab: float, sampling_rate: float | None = None
) -> tuple[Axes, Axes, Axes]:
    """
    Plot reconstruction of P wave train of event B from event A

    Top panel shows reconstucted data over waveform A. Bottom two panels show
    waveform A and B, respectively.

    Parameters
    ----------
    wvfA, wvfB:
        ``(samles,)`` waveforms of event A and B
    Aab:
        Relative amplitude between A and B
    sampling_rate:
        Sampling rate (seconds)

    Returns
    -------
    Axes of the resulting subplots
    """

    _, axs = plt.subplots(nrows=3, ncols=1, sharex=True, layout="constrained")

    xx = np.arange(wvfA.shape[0], dtype=float)

    xlabel = "Samples"
    if sampling_rate is not None:
        xx /= sampling_rate
        xlabel = "Time (s)"
    rcA = Aab * wvfB

    titles = ["Reconstruction Aab = {:.1e}".format(Aab), "Event B", "Event A"]

    for n, (tit, wv, ax) in enumerate(zip(titles, [wvfA, wvfB, wvfA], axs)):

        if n == 0:
            ax.plot(xx, wv, color="gray", zorder=-5)
            ax.plot(xx, rcA, "--", color="red", zorder=5)
        else:
            ax.plot(xx, wv, color="black")

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(tit, loc="left")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")

    return axs


def s_reconstruction(
    wvfA: np.ndarray,
    wvfB: np.ndarray,
    wvfC: np.ndarray,
    Babc: float,
    Bacb: float,
    sampling_rate: float | None = None,
) -> tuple[Axes, Axes, Axes, Axes]:
    """
    Plot reconstruction of S wave train of event A from events B and C

    Top panel shows reconstucted data over waveform A. Bottom two panels show
    waveform A, B, and C, respectively.

    Parameters
    ----------
    wvfA, wvfB, wvfC : (S,) :class:`~numpy.array`:
        Waveform of events A, B, and C
    Babc:
        Relative contribution of B to A
    Bacb:
        Relative contribution of C to A

    Returns
    -------
    Axes of the resulting subplots
    """

    rcA = Babc * wvfB + Bacb * wvfC

    xx = np.arange(wvfA.shape[0], dtype=float)

    xlabel = "Samples"
    if sampling_rate is not None:
        xx /= sampling_rate
        xlabel = "Time (s)"

    _, axs = plt.subplots(nrows=4, ncols=1, sharex=True, layout="constrained")

    titles = [
        "Reconstruction Babc = {:.1e}, Bacb = {:.1e}".format(Babc, Bacb),
        "Event C",
        "Event B",
        "Event A",
    ]
    for n, (tit, wv, ax) in enumerate(zip(titles, [wvfA, wvfC, wvfB, wvfA], axs)):

        if n == 0:
            ax.plot(xx, wv, color="gray", zorder=-5)
            ax.plot(xx, rcA, "--", color="red", zorder=5)
        else:
            ax.plot(xx, wv, color="black")

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(tit, loc="left")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")

    return ax


def bootstrap_matrix(
    moment_tensors: list[core.MT],
    plot_beachball: bool = False,
    best_mt: core.MT | None = None,
    takeoff: np.ndarray | None = None,
    subplot_kwargs: dict = {"figsize": (8, 9)},
    # axes: np.ndarray | None = None,
):
    """Plot bootstrapped moment tensor components

    Parameters
    ----------
    moment_tensors:
        List of bootstrap results
    plot_beachball:
        Plot bootstrap results as beacball plot. Requires :module:`pyrocko`
    best_mt:
        Also show the best moment tensor, when `plot_beachball=True` Requires
        :module:`pyrocko`.
    takeoff:
        `(2, N)` array of takeoff azimuth and plunge angles (degree).
    subplot_kwargs:
        Keyword arguments passed on to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
    fig:
        The :class:`class matplotlib..figure.Figure` that holds the plot
    axs:
        ``(6, 6)`` array of :class:`class matplotlib.axes.Axes`
    """

    # axes:
    #    ``(6, 6)`` array of :class:`class matplotlib.axes.Axes` to place the plots

    # Scale the numbers
    m0 = mt.mean_moment(moment_tensors)
    exp = int(np.floor(np.log10(m0)))
    fac = 10**exp
    arr = np.array(moment_tensors) / fac

    # Make a best moment tensor, if there is one
    best = np.array([np.nan] * 6)
    if best_mt is not None:
        best = np.array(best_mt) / fac

    # Common ticks and labels
    mi = 1.05 * np.min(arr)
    ma = 1.05 * np.max(arr)
    ticks = [np.min(arr), 0, np.max(arr)]
    ticklabels = ["{:.1f}".format(t) if t != 0 else "" for t in ticks]
    comp = core.MT._fields

    # Set up the plot
    fig, axs = plt.subplots(6, 6, layout="constrained", **subplot_kwargs)
    fig.suptitle(f"Bootstraped moment tensor elements ($10^{{{exp}}}$ Nm)")

    # TODO:
    # Passing external axes causes problems down the road when placing beachball
    # if axes is None:
    #    fig, axs = plt.subplots(6, 6, layout="constrained", **subplot_kwargs)
    #    fig.suptitle(f"Bootstraped moment tensor elements ($10^{{{exp}}}$ Nm)")
    # else:
    #    axs = axes
    #    fig = axes[0, 0].get_figure()

    # Iterate the lower off-dagonal triangle
    ij = ((i, j) for i in range(6) for j in range(i + 1, 6))

    # Scatter plots of pairwise different components
    for i, j in ij:
        ax = axs[j, i]

        x = arr[:, i]
        y = arr[:, j]

        ax.scatter(x, y, fc="none", ec="black")
        ax.scatter(best[i], best[j], fc="none", ec="red")

        # Set ticks and labels only at the bottom and right most axes
        if i == 0:
            ax.set_ylabel(comp[j])
            ax.set_yticks(ticks, ticklabels)
        else:
            ax.set_yticks([])

        if j == 5:
            ax.set_xlabel(comp[i])
            ax.set_xticks(ticks, ticklabels, rotation="vertical")
        else:
            ax.set_xticks([])

        ax.set_xlim((mi, ma))
        ax.set_ylim((mi, ma))
        ax.axhline(0, color="silver", zorder=-5)
        ax.axvline(0, color="silver", zorder=-5)

        # Make the other symmetric axis invisible
        axs[i, j].set_visible(False)

    # Histograms along the diagonal
    for i in range(6):
        ax = axs[i, i]
        x = arr[:, i]
        ax.hist(x, density=True, stacked=True, fc="black")
        ax.axvline(best[i], color="red")

        ax.set_xlim((mi, ma))
        ax.set_ylim((0, 1))
        ax.spines[["left", "top", "right"]].set_visible(False)

        ax.set_xticks(ticks, [])
        ax.set_yticks([])

        if i == 5:
            ax.set_xlabel(core.MT._fields[i])
            ax.set_xticks(ticks, ticklabels, rotation="vertical")

        # Report mean an standard deviation
        tit = "{:.2f} $\pm$ {:.2f}".format(np.mean(x), np.std(x))
        ax.set_title(tit, size="small")

    if plot_beachball:

        try:
            from pyrocko import moment_tensor as pmt
            from pyrocko.plot import beachball
        except ModuleNotFoundError:
            raise ModuleNotFoundError(core._module_hint("pyrocko"))

        # Make an axis for the mt in the top right corner
        gs = axs[0, 5].get_gridspec()
        for iremove in [(0, 4), (0, 5), (1, 4), (1, 5)]:
            axs[iremove].remove()
        axmt = fig.add_subplot(gs[:2, 4:])
        axpo = fig.add_subplot(gs[:2, 4:], projection="polar")
        axmt.set_axis_off()
        axpo.set_axis_off()
        axpo.set_theta_direction(-1)  # Clockwise...
        axpo.set_theta_zero_location("N")  # from North
        axpo.set_rlim((np.pi / 2, 0))

        # rotate N -> y and E -> x
        rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Convert MTs to pyrocko format
        pmts = [
            pmt.MomentTensor(mt.mt_array(momt)).rotated(rot) for momt in moment_tensors
        ]

        # Make a best MT (or not)
        pbest = None
        if best_mt is not None:
            pbest = pmt.MomentTensor(mt.mt_array(best_mt)).rotated(rot)

        if plot_beachball:
            # Plot it fuzzy.
            beachball.plot_fuzzy_beachball_mpl_pixmap(
                pmts,
                axmt,
                beachball_type="full",
                best_mt=pbest,
                # size=200,
                position=(0, 0),
                color_t="black",
                linewidth=1.0,
                size=2,
                size_units="data",
            )

        if takeoff is not None:
            takeoff *= np.pi / 180

            az = takeoff[0, :]
            pl = takeoff[1, :]

            # Project upgoing rays to the opposite side of the lower hemisphere
            iup = pl < 0
            az[iup] = (az[iup] + np.pi) % (2 * np.pi)
            pl[iup] = -pl[iup]

            # TODO: Actually project to Lambert

            axpo.scatter(az[iup], pl[iup], marker="o", fc="none", ec="lightblue")
            axpo.scatter(az[~iup], pl[~iup], marker="x", color="lightblue")

    return fig, axs
