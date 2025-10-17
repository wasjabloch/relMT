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
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.transforms as transforms
from scipy.linalg import svd
from relmt import core, mt, amp, qc, signal, utils, align, angle

plt.ion()

logger = core.register_logger(__name__)

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
            interpolation="nearest",
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
    wvfA: np.ndarray,
    wvfB: np.ndarray,
    Aab: float,
    sampling_rate: float | None = None,
    events_ab: tuple[int, int] | None = None,
    axs: tuple[Axes, Axes, Axes] | None = None,
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
    events_ab:
        Show event numbers in axis titles
    axs:
        A tuple of axes to plot into. If `None`, create new axes.

    Returns
    -------
    Axes of the resulting subplots
    """

    xx = np.arange(wvfA.shape[0], dtype=float)

    xlabel = "Samples"
    if sampling_rate is not None:
        xx /= sampling_rate
        xlabel = "Time (s)"

    if events_ab is None:
        events_ab = ("", "")

    if axs is None:
        _, axs = plt.subplots(nrows=3, ncols=1, sharex=True, layout="constrained")

    rcA = Aab * wvfB
    mis = amp.p_misfit(np.array([wvfA, wvfB]), Aab)

    titles = [
        "Reconstruction $A_{{ab}}$ = {:.3g} misfit = {:.5f}".format(Aab, mis),
        f"Event {events_ab[0]} (A)",
        f"Event {events_ab[1]} (B)",
    ]

    for n, (tit, wv, ax) in enumerate(zip(titles, [wvfA, wvfA, wvfB], axs)):

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
    events_abc: tuple[int, int, int] | None = None,
    axs: tuple[Axes, Axes, Axes, Axes] | None = None,
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
    samling_rate:
        Supply to show a time axis
    events_abc:
        Show event numbers in axis titles.
    axs:
        Tuple of axes to plot into. If `None`, create new axes.

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

    if events_abc is None:
        events_abc = ("", "", "")

    if axs is None:
        _, axs = plt.subplots(nrows=4, ncols=1, sharex=True, layout="constrained")

    mis = amp.s_misfit(np.array([wvfA, wvfB, wvfC]), Babc, Bacb)

    titles = [
        "Reconstruction $B_{{abc}}$ = {:.3g}, $B_{{acb}}$ = {:.3g} misfit = {:.5f}".format(
            Babc, Bacb, mis
        ),
        f"Event {events_abc[0]} (A)",
        f"Event {events_abc[1]} (B)",
        f"Event {events_abc[2]} (C)",
    ]

    for n, (tit, wv, ax) in enumerate(zip(titles, [wvfA, wvfA, wvfB, wvfC], axs)):

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


def amplitude_connections(
    amplitudes: list[core.P_Amplitude_Ratio] | list[core.S_Amplitude_Ratios],
    s_amplitudes: list[core.S_Amplitude_Ratios] | None = None,
    reference_mts: list[int] | None = None,
    ax: Axes | None = None,
    node_size: float = 250.0,
    node_linewidth: float = 1.0,
) -> Axes:
    """Plot a graph representation of event connections.

    Events are connected as pairs or triplets through relative amplitude
    measurements. Represent these connections as a graph and colour by number of
    connections

    ..note:
        Requires :module:`networkx` to be installed.

    Parameters
    ----------
    amplitudes:
        Event pair- or tripletwise P- or S-amplitude mesuremetns.
    s_samplitudes:
        Second set of S-amplitude measurements. Observations will be combined
    referenc_mts:
        Highlight these reference moment tensors with a larger node and thicker
        outline
    ax:
        Plot into this axis. If `None`, create one
    node_size:
        Size of a node. Increase if you have big numbers
    node_linewidth:
        Width of the node edge

    Returns
    -------
    The axis of the plot
    """

    # Prototype by ChatGPT o4-mini-high
    import networkx as nx

    ip, _ = qc._ps_amplitudes(amplitudes)

    if ip:
        evs = [(amp.event_a, amp.event_b) for amp in amplitudes]
        this_cmap = plt.cm.Reds
    else:
        evs = [(amp.event_a, amp.event_b, amp.event_c) for amp in amplitudes]
        this_cmap = plt.cm.Blues

    if s_amplitudes is not None:
        this_cmap = plt.cm.Purples

    # Crop brightest end from colormap
    cmap = LinearSegmentedColormap.from_list(
        "cmap", this_cmap(np.linspace(0.3, 1.0, 256))
    )

    MG = nx.MultiGraph()

    if ip:
        MG.add_edges_from(evs, conn_type="P")
    else:
        for a, b, c in evs:
            MG.add_edges_from([(a, b), (b, c), (a, c)], conn_type="S")

    if s_amplitudes is not None:
        evs_extra = [(amp.event_a, amp.event_b, amp.event_c) for amp in s_amplitudes]
        for a, b, c in evs_extra:
            MG.add_edges_from([(a, b), (b, c), (a, c)], conn_type="S")

    deg = MG.degree()
    nodes = MG.nodes()

    # A connection between two nodes counts only once, even when it occurs in
    # many triplets.
    connections = np.array([deg[n] for n in nodes])

    # Constant node sizes and line widhts
    node_sizes = [node_size] * len(nodes)
    linewidths = [node_linewidth] * len(nodes)

    # When given, make reference MTs larger and line thicker
    if reference_mts is not None:
        node_sizes = [
            node_size if n not in reference_mts else node_size * 2 for n in nodes
        ]
        linewidths = [
            node_linewidth if n not in reference_mts else node_linewidth * 2
            for n in nodes
        ]

    # Collapse to graph after counting connections for efficient plotting
    G = nx.Graph()
    G.add_nodes_from(MG.nodes(data=True))
    for u, v, _, data in MG.edges(keys=True, data=True):
        ct = data.get("conn_type", None)
        if G.has_edge(u, v):
            # there is already an edge in G: merge the conn_type
            prev = G[u][v].get("conn_type")
            if prev != ct:
                # if they disagree, mark as mixed
                G[u][v]["conn_type"] = "PS"
        else:
            # first time we see (u,v) — just copy it
            G.add_edge(u, v, conn_type=ct)

    p_edges = [(u, v) for u, v, d in G.edges(data=True) if d["conn_type"] == "P"]
    s_edges = [(u, v) for u, v, d in G.edges(data=True) if d["conn_type"] == "S"]
    ps_edges = [(u, v) for u, v, d in G.edges(data=True) if d["conn_type"] == "PS"]

    pos = nx.spring_layout(G)

    # Set up the figure
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6), layout="tight")

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=node_sizes,
        linewidths=linewidths,
        node_color=connections,
        edgecolors="white",
        cmap=cmap,
        ax=ax,
    )

    # Node labels
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n in nodes},
        font_color="white",
        font_size=8,
        ax=ax,
    )

    for edges, style, label in zip(
        [p_edges, s_edges, ps_edges], ["dotted", "dashed", "solid"], ["P", "S", "P+S"]
    ):
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, style=style, alpha=0.5, ax=ax, label=label
        )

    ax.legend(title="Type")

    vmin = connections.min()
    vmax = connections.max()
    ticks = np.linspace(vmin, vmax, 5)
    labels = ["{:.0f}".format(t) for t in ticks]

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Connections")

    cbar.set_ticks(ticks, labels=labels)

    ax.axis("off")
    return ax


def mt_matrix(
    mtd: dict[int, core.MT],
    highlight: list[int] = [],
    names: dict[int, str] = {},
    values: dict[int, float] = {},
    valuename: str = "Value",
    cmap=plt.cm.cividis,
    overlay_dc_at: float = 1.0,
    ax: Axes | None = None,
) -> Axes:
    """Plot moment tensors into a square matrix

    Parameters
    ----------
    mtd:
        Dictionary holding the moment tensors
    ax:
        Plot into existing axes
    highlight:
        Highlight these moment tensors
    overlay_dc_at:
        Overlay a double couple beachball if DC fraction is at least this large

    Return
    ------
    Axes of the plot

    .. note:
        Requires pyrocko

    """
    from pyrocko import moment_tensor as pmt
    from pyrocko.plot import beachball

    highlightc = "xkcd:lipstick red"

    nmts = len(mtd)
    nrow = int(np.sqrt(nmts))
    ncol = nmts // nrow

    if values:
        vmin, vmax = np.min(list(values.values())), np.max(list(values.values()))
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = {evn: cmap(norm(val)) for evn, val in values.items()}

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(nrow, ncol), layout="tight")

    for nev, (iev, momt) in enumerate(mtd.items()):
        thismt = pmt.MomentTensor(mt.mt_array(momt))

        x = nev % nrow
        y = nev // nrow

        fc = "xkcd:twilight blue"
        ec = "black"

        if iev in highlight:
            fc = highlightc

        if values:
            if iev in highlight:
                ec = highlightc
            fc = colors.get(iev, fc)

        idc = thismt.standard_decomposition()[1][1] >= overlay_dc_at

        mtlw = 1.0
        if idc:
            mtlw = 0.0

        beachball.plot_beachball_mpl(
            thismt,
            ax,
            beachball_type="full",
            size=40,
            position=(x, y),
            color_t=fc,
            edgecolor=ec,
            linewidth=mtlw,
        )

        if idc:
            beachball.plot_beachball_mpl(
                thismt,
                ax,
                beachball_type="dc",
                size=40,
                position=(x, y),
                color_t="none",
                color_p="none",
                edgecolor=ec,
                linewidth=1.0,
            )

        ax.annotate(names.get(iev, iev), (x, y), (-20, 22), textcoords="offset points")

    ax.set_ylim([ncol + 0.5, -0.5])
    ax.set_xlim([-0.5, nrow - 0.5])
    ax.axis("off")

    if values:
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.3, fraction=0.05, pad=0.01)
        cbar.set_label(valuename)

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
        tit = "{:.2f} $\\pm$ {:.2f}".format(np.mean(x), np.std(x))
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


def alignment(
    arr: np.ndarray,
    hdr: core.Header,
    dt_mccc: np.ndarray | None = None,
    dt_rms: np.ndarray | None = None,
    dt_pca: np.ndarray | None = None,
    ccij: np.ndarray | None = None,
    event_list: list[int] = None,
    event_dict: dict[int, core.Event] = {},
    station_dict: dict[int, core.Event] = {},
    sort: str = "pci",
    highlight_events: list[int] = [],
) -> tuple[Figure, dict[str, Axes]]:
    """Plot waveform array with alignment diagnostics

    Parameters
    ----------
    arr:
        ``(events, components, samples)`` Waveform array
    hdr:
        Header of the waveform array
    event_list:
        List of event IDs to show
    event_dict:
        The seismic event catalog
    dt_mccc:
        Time shifts from MCCC alignment
    dt_rms:
        RMS of time shifts from MCCC alignment
    dt_pca:
        Time shifts from PCA alignment
    ccij:
        2D Cross-correlation matrix between all event pairs
    sort:
        Sort by "magnitude", "pci" (principal component index), or
        "time" (input order)
    refevs:
        Highlight these reference events in red

    Returns
    -------
    fig:
        The :class:`matplotlib.figure.Figure` containing the plot
    axs:
        Dictionary of :class:`matplotlib.axes.Axes` in the plot
        with keys:
        - "dt": Time shifts
        - "snr": Signal to noise ratios
        - "pv": Principal seismograms
        - "wv": Waveform section
        - "cci": Average correlation of each event
        - "ccij": Cross-correlation matrix
        - "ec": Expansion coefficients
    """

    # Get station and phase
    phase = hdr["phase"]

    # Principal components to show (two significant and the first insignificant)
    icomps = [0, 1]
    compc = ["black", "gray"]
    comps = "ox"
    if phase == "S":
        icomps += [2]
        compc = ["black", "green", "gray"]
        comps = "o+x"

    if event_list is None:
        event_list = hdr["events_"]

    nin = len(event_list)
    ievs = np.array(range(nin))

    if not any(event_list):
        raise ValueError("No events in list")

    if (phase == "P" and nin < 2) or (phase == "S" and nin < 3):
        raise ValueError("No enough events for phase")

    # Taper and filter the actual phase data
    snr = signal.signal_noise_ratio(arr, **hdr.kwargs(signal.signal_noise_ratio))
    arr = signal.demean_filter_window(arr, **hdr.kwargs(signal.demean_filter_window))
    i0, i1 = signal.indices_inside_taper(**hdr.kwargs(signal.indices_inside_taper))

    mat = utils.concat_components(arr[ievs, :, i0:i1])

    if sort == "magnitude":
        if event_dict is None or not event_dict:
            raise ValueError("Need event_dict to sort by magnitude")
        mags = [ev.mag for iev, ev in event_dict.items() if iev in event_list]
        isort = np.argsort(mags)[::-1]
    elif sort == "pci":
        isort = utils.pc_index(mat, phase=phase)
    elif sort == "time":
        isort = ievs
    else:
        raise ValueError(f"unknown sort: {sort}")

    snr = snr[isort]

    if ccij is None:
        ccij = np.full((nin, nin), np.nan)

    cci = utils.fisher_average(np.abs(ccij))
    cc = utils.fisher_average(cci)

    # Fill in auto-correlation values and sort
    ccij = ccij[isort, :][:, isort]
    cci = cci[isort]

    # Event labels
    if event_dict:
        evlabels = [event_dict[ie].name for ie in event_list[isort]]
    else:
        evlabels = [str(ie) for ie in event_list[isort]]

    # Set up subplot arragement
    mosaic = [
        ["map", "map", "pv", ".", "cbar", "."],
        ["map", "map", "pv", ".", ".", "."],
        ["dt", "snr", "wv", "cci", "ccij", "ec"],
    ]

    fig, axs = plt.subplot_mosaic(
        mosaic,
        width_ratios=[0.1, 0.1, 0.5, 0.1, 0.3, 0.2],
        height_ratios=[0.03, 0.07, 0.9],
        gridspec_kw=dict(
            left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.05
        ),
        figsize=(18, 10),
    )

    fig.suptitle(f"Alignment {hdr['station']} {phase}", ha="left", va="top")

    axs["dt"].sharey(axs["snr"])
    axs["wv"].sharey(axs["dt"])
    axs["wv"].sharex(axs["pv"])
    axs["cci"].sharey(axs["wv"])
    axs["ccij"].sharey(axs["cci"])
    axs["ec"].sharey(axs["ccij"])

    if len(highlight_events) > 0:
        # Sorted event list for indexing
        sevl = list(event_list[isort])
        for ax in axs.values():
            for refev in highlight_events:
                ax.axhline(sevl.index(refev), color="red", zorder=5)

    # Station and event map

    # If we have everything to plot a map
    if event_dict and station_dict:
        # Place a polar plot in the existing axis
        subplotspec = axs["map"].get_subplotspec()
        axs["map"].remove()
        axs["map"] = fig.add_subplot(subplotspec, projection="polar")
        ax = axs["map"]
        ax.set_theta_zero_location("N")  # theta=0 at the top
        ax.set_theta_direction(-1)
        ista = (np.array(list(station_dict.keys())) == hdr["station"]).nonzero()[0][0]

        # Coordinates relative to center of event cloud
        evxyz = utils.xyzarray(event_dict) * 1e-3
        orig = np.mean(evxyz, axis=0)
        evxyz = evxyz - orig
        staxyz = utils.xyzarray(station_dict) * 1e-3 - orig

        # Station azimuth and distance
        st0 = np.zeros(staxyz.shape[0]).T
        staz = angle.azimuth(st0, st0, *staxyz[:, :2].T) * np.pi / 180
        stad = utils.cartesian_distance(st0, st0, st0, *staxyz[:, :2].T, st0)

        # Event azimuth and distance
        ev0 = np.zeros(evxyz.shape[0]).T
        evaz = angle.azimuth(ev0, ev0, *evxyz[:, :2].T) * np.pi / 180
        evad = utils.cartesian_distance(ev0, ev0, ev0, *evxyz[:, :2].T, ev0)

        ax.scatter(evaz, evad, s=5, color="black")
        ax.scatter(staz, stad, marker="v", color="gray")
        ax.scatter(staz[ista], stad[ista], marker="v", color="orange")
        ax.set_ylim((0, 1.1 * stad[ista]))
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2], "NESW")
        ax.set_yticks([stad[ista]], ["{:.0f}km".format(stad[ista])])

    else:
        ax.axis("off")

    # Time shift plot
    ax = axs["dt"]

    if dt_pca is not None:
        ax.plot(dt_pca[isort], range(nin), color="blue", label="pca")

    if dt_mccc is not None:
        dt_mccc = dt_mccc[isort]
        ax.plot(dt_mccc, range(nin), color="green", label="mccc")
        if dt_rms is not None:
            dt_rms = dt_rms[isort]
            ax.errorbar(dt_mccc, range(nin), xerr=dt_rms, color="green")

    if dt_pca is not None or not dt_mccc is None:
        ax.set_xlabel("Shift (s)")
        ax.set_ylabel("Event #")
        ax.set_yticks(range(nin), evlabels)
        ax.grid(axis="y")
        ax.legend()

    else:
        ax.axis("off")
        axs["snr"].set_yticks(range(nin), evlabels)

    # Signal noise ratio plot
    ax = axs["snr"]
    ax.plot(snr, range(nin), color="black")
    ax.set_xlabel("SNR (dB)")
    ax.axvline(0, color="silver")
    if (minsnr := hdr["min_signal_noise_ratio"]) is not None:
        ax.axvline(minsnr, color="silver")
    ax.grid(axis="y")
    ax.tick_params(labelleft=False)

    # Principal seismograms
    ax = axs["pv"]
    try:
        U, s, Vh = svd(signal.norm_power(mat[isort, :]), False)
    except ValueError:
        logger.warning("Could not compute SVD")
        U = np.full((mat.shape[0], mat.shape[0]), np.nan)
        s = np.full(mat.shape[0], np.nan)
        Vh = np.full((mat.shape[0], mat.shape[1]), np.nan)

    phi = align.pca_objective(s, phase, mat.shape[0])
    section_2d(Vh[icomps, :], **hdr.kwargs(section_2d), ax=ax)
    ax.set_title("$\Phi={:.4f}$".format(phi), pad=12)
    ax.set_ylabel("Principal\nSeismogram")
    ax.set_yticks(icomps)
    ax.set_xlabel("")

    # Waveform plot
    ax = axs["wv"]
    section_2d(
        mat[isort, :],
        **hdr.kwargs(section_2d),
        wiggle=False,
        image=True,
        ax=ax,
    )
    ax.set_ylabel("")
    ax.set_xlabel("Time (s)")
    ax.tick_params(labelleft=False, labelbottom=True)
    ax.grid(axis="x")

    # Cross correlation plot
    ax = axs["cci"]
    ax.plot(cci, range(nin), color="red")
    ax.axvline(0, color="silver")
    ax.set_xlabel("$\hat{{{C}}}_i$")
    ax.grid(axis="y")

    if (mincc := hdr["min_correlation"]) is not None:
        ax.axvline(mincc, color="silver")
    ax.tick_params(labelleft=False)

    for ax in [axs["dt"], axs["snr"], axs["cci"], axs["wv"], axs["ccij"], axs["ec"]]:
        ax.set_ylim([nin - 0.5, -0.5])
        ax.spines[["right", "left"]].set_visible(False)

    # Cross correlation coefficient matrix
    ax = axs["ccij"]
    ax.set_title("$\hat{{{|C|}}}$ = " + "{:.3f}".format(cc))
    cmap = ax.imshow(
        ccij, vmin=-1, vmax=1, cmap="RdGy", interpolation="nearest", aspect="auto"
    )
    ax.set_xticks(range(nin), evlabels, rotation=90)
    ax.tick_params(labelleft=False)

    plt.colorbar(
        cmap,
        cax=axs["cbar"],
        location="top",
        ticks=[-1, 0, 1],
        label="$cc_{ij}$",
    )

    # Expansion coefficeints
    ax = axs["ec"]
    ec_score = qc.expansion_coefficient_norm(mat, phase)[isort]

    for i, col, sym in zip(icomps, compc, comps):
        e0 = abs(s[i] * U[:, i])
        ax.plot(e0, range(nin), sym, mec=col, mfc="none", label=f"EC{i}")
    ax.plot(ec_score, range(nin), "|", mec="red", mfc="none", label=f"Score")

    if (ecn := hdr["min_expansion_coefficient_norm"]) is not None:
        ax.axvline(ecn, color="silver")

    ax.legend()
    ax.set_yticks(range(nin), event_list[isort])
    ax.set_xlim((0, 1))
    ax.grid(axis="y")
    ax.tick_params(labelleft=False, labelright=True)

    return fig, axs
