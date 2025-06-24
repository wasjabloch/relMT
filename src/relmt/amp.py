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

"""Functions to estimate relative waveform amplitudes"""

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import svd, norm, solve
from relmt import core, signal, qc, mt, utils
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def pca_amplitude_2p(mtx_ab: np.ndarray) -> float:
    """
    Calculate the relative amplitude of a pair of P-waves

    The relative amplitudes :math:`A^{ab}` that predicts waveform :math:`u_a` from
    waveform :math:`u_b`, such that:

    .. math::
        A^{ab} u_b = u_a

    Parameters
    ----------
    mtx_ab:
        Waveform matrix of shape ``(2, samples)`` holding events `a` and `b`

    Returns
    -------
    Relative ampltiude between events a and b
    """

    # The line below here was
    _, ss, Vh = svd(mtx_ab.T / np.max(abs(mtx_ab)), full_matrices=False)

    Aab = Vh[0, 0] / Vh[0, 1]

    return Aab, ss[:2] / np.sum(ss)


def pca_amplitudes_p(mtx: np.ndarray) -> np.ndarray:
    """
    Calculate the relative amplitude of all P-wave pairs in mtx as the relative
    contribution of the principal seismogram

    Parameters
    ----------
    mtx:
        ``(events, samples)`` waveform matrix

    Returns
    -------
    ``(events * (events - 1) / 2, )`` relative ampltiude between all pairwise
    event combinations.
    """

    _, ss, Vh = svd(mtx.T / np.max(abs(mtx)), full_matrices=False)

    aa, bb, _, _ = np.array(list(core.iterate_event_pair(mtx.shape[0]))).T

    Aabs = Vh[0, aa] / Vh[0, bb]

    return Aabs, ss[:2] / np.sum(ss)


def p_misfit(mtx_ab: np.ndarray, Aab: float) -> float:
    """
    Misfit of the P waveform reconstruction

    The residual norm of the reconstruction devided by the norm of the predicted
    waveform :math:`u_a`:

    .. math::
        \Psi_P = || A^{ab} u_b - u_a || / || u_a ||

    Parameters
    ----------
    mtx_ab:
        Waveform matrix of shape ``(2, samples)`` holding events `a` and `b`
    Aab:
        Relative ampltiude between events `a` and `b`

    Returns
    -------
    Normalized reconstruction misfit
    """

    return norm(mtx_ab[0, :] - Aab * mtx_ab[1, :]) / norm(mtx_ab[0, :])


def pca_amplitude_3s(
    mtx_abc: np.ndarray, order: bool = True
) -> tuple[float, float, np.ndarray]:
    """
    Relative amplitudes between triplet of S-waves.

    Given waveforms of three events, determine which two waveforms :math:`b` and
    :math:`c` are most different from each other and compute the relative
    amplitudes :math:`B^{abc}` and :math:`B^{acb}` that predict the third
    waveform :math:'u_a', such that:

    .. math::
        B^{abc} u_b + B^{acb} * u_c = u_a

    Parameters
    ----------
    mtx_abc:
        Waveform matrix of shape ``(3, samples)`` holding events `a`, `b` and
        `c`
    order:
        If True, order the waveforms by pairwise summed cross-correlation
        coefficients before computing the relative amplitudes. If False, use
        the order of the input matrix.

    Returns
    -------
    Babc:
        Relative ampltiude between event `a` and `b`, given the third event is `c`
    Bacb:
        Relative ampltiude between event `a` and `c`, given the third event is `b`
    iord:
        Resulting ``(3,)`` row indices into `mtx_abc` of events `a`, `b` and `c`
    sigmas:
        ``(3,)`` first three singular values of the seismogram decomposition
    """

    iord = np.array([0, 1, 2])
    if order:
        iord = order_by_ccsum(mtx_abc)

    _, ss, Vh = svd(mtx_abc.T / np.max(abs(mtx_abc)), full_matrices=False)
    sigmas = ss[:3] / np.sum(ss)

    # Bostock et al. (2021) Eq. 10
    # Expansion coefficients
    ecs = np.diag(ss) @ Vh

    # Pull out expansion coefficients b, c...
    # (LHS Eq. 10)
    A = ecs[0:2, iord[1:]]

    # ... and a
    # (RHS Eq. 10)
    bb = ecs[0:2, iord[0]]

    if bb[1] == 0:
        logger.warning("All wavefroms are similar, so treated with Bacb = 0")
        return bb[0], 0.0, iord, sigmas

    else:
        try:
            Babc, Bacb = solve(A, bb)

        except LinAlgError as e:
            msg = f"Met error in SVD: {e.__repr__()}. Returning NaN values."
            logger.warning(msg)
            return np.nan, np.nan, iord, sigmas

        return Babc, Bacb, iord, sigmas


def pca_amplitudes_s(
    mtx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the relative amplitude of all P-wave pairs in mtx as the relative
    contribution of the principal seismogram

    Parameters
    ----------
    mtx:
        ``(events, samples)`` waveform matrix

    Returns
    -------
    Babc, Bacb:
        ``(events * (events - 1) / 2, )`` relative ampltiude between all
        tripletwise event combinations.
    iord:
        ``(events * (events - 1) / 2, 3)``
    sigmas:
        ``(3,)`` first three singular values of seismogram decomposition
    """

    _, ss, Vh = svd(mtx.T / np.max(abs(mtx)), full_matrices=False)

    # Expansion coefficients
    ecs = np.diag(ss) @ Vh

    aa, bb, cc, *_ = np.array(list(core.iterate_event_triplet(mtx.shape[0]))).T

    nn = len(aa)

    Babcs = np.full(nn, np.nan)
    Bacbs = np.full(nn, np.nan)
    iords = np.full((nn, 3), [0, 1, 2])
    sigmas = ss[:3] / np.sum(ss)

    for n, (a, b, c) in enumerate(zip(aa, bb, cc)):

        # oa is the one most similar to the other two
        # ob and oc are most different from each other
        iords[n] = order_by_ccsum(mtx[[a, b, c], :])
        oa, ob, oc = np.array([a, b, c])[iords[n]]

        # Bostock et al. (2021) Eq. 10
        # Pull out expansion coefficients b, c...
        # (LHS Eq. 10)
        lhs = ecs[0:2, [ob, oc]]

        # ... and a
        # (RHS Eq. 10)
        rhs = ecs[0:2, oa]

        if rhs[1] == 0:
            logger.warning("All wavefroms are similar, so treated with Bacb = 0")
            Babcs[n], Bacbs[n] = rhs[0], 0.0
        else:
            try:
                Babcs[n], Bacbs[n] = solve(lhs, rhs)[:2]
            except LinAlgError as e:
                msg = f"Met error in SVD: {e.__repr__()}. Returning NaN values."
                logger.warning(msg)
                Babcs[n], Bacbs[n] = np.nan, np.nan

    return Babcs, Bacbs, iords, sigmas


def s_misfit(mtx_abc: np.ndarray, Babc: float, Bacb: float) -> float:
    """
    Misfit of the S waveform reconstruction

    The residual norm of the reconstruction devided by the norm of the predicted
    waveform :math:`u_a`:

    .. math::
        \Psi_S = || B^{abc} u_b + B^{acb} u_c - u_a || / || u_a ||

    Parameters
    ----------
    mtx_abc:
        Waveform matrix of shape ``(3, samples)`` holding events `a`, `b` and
        `c`
    Babc:
        Relative ampltiude between event `a` and `b`, given the third event is `c`
    Bacb:
        Relative ampltiude between event `a` and `c`, given the third event is `b`

    Returns
    -------
    Normalized reconstruction misfit
    """

    return norm(Babc * mtx_abc[1, :] + Bacb * mtx_abc[2, :] - mtx_abc[0, :]) / norm(
        mtx_abc[0, :]
    )


def order_by_ccsum(mtx_abc: np.ndarray) -> np.ndarray:
    """
    Order waveforms by pairwise summed cross-correlation coefficients

    Parameters
    ----------
    mtx_abc:
        Waveform of shape ``(3, samples)`` of events `a`, `b`, and `c`

    Returns
    -------
    iord: :class:`~numpy.ndarray`
        Indices ``(3,)`` that order :math:`u_a, u_b, u_c` by cc
    """
    # Calculate cross-correlation coefficients between all pairs of waveforms
    ab = abs(signal.cc_coef(mtx_abc[0, :], mtx_abc[1, :]))
    ac = abs(signal.cc_coef(mtx_abc[0, :], mtx_abc[2, :]))
    bc = abs(signal.cc_coef(mtx_abc[1, :], mtx_abc[2, :]))

    # Re-order events based on waveform similarity such that evs B and C are most different
    iord = np.argsort([ab + ac, ab + bc, ac + bc])[::-1]
    return iord


def synthetic(
    moment_tensors: dict[int, core.MT],
    event_list: list[core.Event],
    station_dictionary: dict[str, core.Station],
    phase_dictionary: dict[str, core.Phase],
    p_pairs: list[tuple[str, int, int]],
    s_triplets: list[tuple[str, int, int, int]],
    order: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic relative amplitude meassurements.

    Either compute all possible event combinations from the supplied phase
    dictionary and mment tensors, or supply explicit `p_pairs` and `s_tripltes`
    (e.g. from existing :class:`core.P_Amplitude_Ratio` and
    :class:`core.S_Amplitude_Ratios` objects).

    Parameters
    ----------
    moment_tensors:
        Dictionary of moment tensors indexed by event ID.
    event_list:
        List of events with locations
    station_dictionary:
        Dictionary of stations with locations indexed by station name.
    phase_dictionary:
        Dictionary of phases with take-off angles indexed by phase name.
    p_pairs:
        List of tuples of the form `(station, event_a, event_b)` for P-wave
        relative amplitude pairs.
    s_triplets:
        List of tuples of the form `(station, event_a, event_b, event_c)` for
        S-wave relative amplitude triplets.
    order:
        If True, order the waveforms by pairwise summed cross-correlation
        coefficients before computing the relative S amplitudes. If False, use
        the order of the input matrix. In any case, the applied order will be
        returned in the `orders` variable.

    Returns
    -------
    p_ratios:
        ``(len(p_pairs),)`` arrays of synthetic P- ...
    s_ratios:
        ``(len(s_triplets, 2)`` ... and S-relative amplitude measurements.
    orders:
        ``(len(s_triplets, 3)`` array of indices that order the S waveforms
        according to greatest differences in waveforms.
    p_sigmas:
        ``(len(p_pairs), 2)`` First two singular values of the P-amplitude
        decomposition
    s_sigmas
        ``(len(s_triplets), 3)`` first three singular values of the S-amplitude
        decomposition
    """

    rho = 3600
    alpha = 6000
    beta = 4500  # Dummy density and velocity cancel out upon devision

    # Create empty array to hold synthetic amplitudes
    p_ratios = np.full(len(p_pairs), np.nan)
    s_ratios = np.full((len(s_triplets), 2), np.nan)
    orders = np.full((len(s_triplets), 3), [0, 1, 2], dtype=int)
    p_sigmas = np.full((len(p_pairs), 2), np.nan)
    s_sigmas = np.full((len(s_triplets), 3), np.nan)

    # First compute the P amplitude ratios
    for i, (s, a, b) in enumerate(p_pairs):
        phid_a = core.join_phaseid(a, s, "P")
        phid_b = core.join_phaseid(b, s, "P")

        # Get the moment tensors for events a and b
        try:
            mt_a = mt.mt_array(moment_tensors[a])
            mt_b = mt.mt_array(moment_tensors[b])

            azi_a, plu_a = (
                phase_dictionary[phid_a].azimuth,
                phase_dictionary[phid_a].plunge,
            )
            azi_b, plu_b = (
                phase_dictionary[phid_b].azimuth,
                phase_dictionary[phid_b].plunge,
            )
        except KeyError:
            logger.warning(
                "Missing moment tensor or phase information for pair "
                f"{s}, {a}, {b}. Skipping."
            )
            continue

        dist_a = utils.cartesian_distance(
            *station_dictionary[s][:3], *event_list[a][:3]
        )
        dist_b = utils.cartesian_distance(
            *station_dictionary[s][:3], *event_list[b][:3]
        )

        # Calculate the relative P amplitude
        p_ratios[i], p_sigmas[i, :] = pca_amplitude_2p(
            np.array(
                [
                    mt.p_radiation(mt_a, azi_a, plu_a, dist_a, rho, alpha),
                    mt.p_radiation(mt_b, azi_b, plu_b, dist_b, rho, alpha),
                ]
            )
        )

    # ... then the S amplitude ratios
    for i, (s, a, b, c) in enumerate(s_triplets):
        phid_a = core.join_phaseid(a, s, "S")
        phid_b = core.join_phaseid(b, s, "S")
        phid_c = core.join_phaseid(c, s, "S")

        # Get the moment tensors for events a, b, and c
        try:
            mt_a = mt.mt_array(moment_tensors[a])
            mt_b = mt.mt_array(moment_tensors[b])
            mt_c = mt.mt_array(moment_tensors[c])

            azi_a, plu_a = (
                phase_dictionary[phid_a].azimuth,
                phase_dictionary[phid_a].plunge,
            )
            azi_b, plu_b = (
                phase_dictionary[phid_b].azimuth,
                phase_dictionary[phid_b].plunge,
            )
            azi_c, plu_c = (
                phase_dictionary[phid_c].azimuth,
                phase_dictionary[phid_c].plunge,
            )
        except KeyError:
            logger.warning(
                "Missing moment tensor or phase information for triplet "
                f"{s}, {a}, {b}, {c}. Skipping."
            )
            continue

        dist_a = utils.cartesian_distance(
            *station_dictionary[s][:3], *event_list[a][:3]
        )
        dist_b = utils.cartesian_distance(
            *station_dictionary[s][:3], *event_list[b][:3]
        )
        dist_c = utils.cartesian_distance(
            *station_dictionary[s][:3], *event_list[c][:3]
        )

        us_a = mt.s_radiation(mt_a, azi_a, plu_a, dist_a, rho, beta)
        us_b = mt.s_radiation(mt_b, azi_b, plu_b, dist_b, rho, beta)
        us_c = mt.s_radiation(mt_c, azi_c, plu_c, dist_c, rho, beta)

        Babc, Bacb, iord, s_sigmas[i, :] = pca_amplitude_3s(
            np.array([us_a, us_b, us_c]), order=order
        )
        print(f"Triplet {s}, {a}, {b}, {c}: Babc = {Babc}, Bacb = {Bacb}")
        print(f"ua: ", us_a)
        print(f"ub: ", us_b)
        print(f"uc: ", us_c)

        # Calculate the relative P amplitude
        s_ratios[i, 0] = Babc
        s_ratios[i, 1] = Bacb
        orders[i, :] = iord

    return p_ratios, s_ratios, orders, p_sigmas, s_sigmas


def principal_p_amplitudes(
    arr: np.ndarray, hdr: core.Header, highpass: float, lowpass: float
) -> list[core.P_Amplitude_Ratio]:
    """Compute relative P amplitude ratios for all event combinations in arr

    Apply a common filter to all events and meassure amplitude ratio as the
    ratio of principal seismogram contribtions.

    ..note:
        The here implemented approach may be more stable against noise than
        :func:`paired_p_amplitudes`, but requires a common filter for all events

    Parameters
    ----------
    arr:
        Waveform matrix of shape ``(events, components, samples)`` holding all events
    hdr:
        Header with metadata, including sampling rate, phase start and end,
        taper length, and event information
    highpass:
        Highpass filter frequency in Hz
    lowpass:
        Lowpass filter frequency in Hz

    Returns
    -------
    List of relative P amplitude ratios for all event pairs
    """

    evns = hdr["events"]

    mat = utils.concat_components(
        signal.demean_filter_window(
            arr,
            hdr["sampling_rate"],
            hdr["phase_start"],
            hdr["phase_end"],
            hdr["taper_length"],
            highpass,
            lowpass,
        )
    )

    As, sigmas = pca_amplitudes_p(mat)

    # Iterate through solutions and assign event numbers
    p_amplitudes = [
        core.P_Amplitude_Ratio(
            hdr["station"],
            evns[a],
            evns[b],
            As[n],
            p_misfit(mat[[a, b], :], As[n]),
            *sigmas,
            highpass,
            lowpass,
        )
        for n, (a, b, _, _) in enumerate(core.iterate_event_pair(len(evns)))
    ]

    return p_amplitudes


def paired_p_amplitudes(
    arr: np.ndarray,
    hdr: core.Header,
    highpass: float,
    lowpass: float,
    a: int,
    b: int,
    realign: bool = False,
):
    """Compute relative P amplitude ratios for one event pair in arr

    ..note:
        The here implemented approach allows to filter each event pair
        individually allowing for more flexibility than
        :func:`principal_p_amplitudes` when comparing large differences in
        magnitude

    Parameters
    ----------
    arr:
        Waveform array of shape ``(2, components, samples)`` holding the event pair
    hdr:
        Header with metadata, including sampling rate, phase start and end,
        taper length, and event information
    highpass:
        Highpass filter frequency in Hz
    lowpass:
        Lowpass filter frequency in Hz
    a:
        Number of event a
    b:
        Number of event b
    realign:
        Re-align seismograms after applying filter

    Returns
    -------
    P amplitude ratio
    """

    if realign:
        mat = signal.subset_filter_align(
            arr, [0, 1], highpass, lowpass, **hdr.kwargs(signal.subset_filter_align)
        )

    else:
        arr_sub = signal.demean_filter_window(
            arr,
            hdr["sampling_rate"],
            hdr["phase_start"],
            hdr["phase_end"],
            hdr["taper_length"],
            highpass,
            lowpass,
        )

        mat = utils.concat_components(arr_sub)

    A, sigmas = pca_amplitude_2p(mat)
    mis = p_misfit(mat, A)

    return core.P_Amplitude_Ratio(
        hdr["station"], a, b, A, mis, *sigmas, highpass, lowpass
    )


def principal_s_amplitudes(
    arr: np.ndarray, hdr: core.Header, highpass: float, lowpass: float
) -> list[core.S_Amplitude_Ratios]:
    """Compute relative S amplitude ratios for all event combinations in arr

    Apply a common filter to all events and meassure amplitude ratio as the
    ratio of principal seismogram contribtions.

    ..note:
        The here implemented approach may be more stable against noise than
        :func:`triplet_s_amplitudes`, but requires a common filter for all events

    Parameters
    ----------
    arr:
        Waveform matrix of shape ``(events, components, samples)`` holding all events
    hdr:
        Header with metadata, including sampling rate, phase start and end,
        taper length, and event information
    highpass:
        Highpass filter frequency in Hz
    lowpass:
        Lowpass filter frequency in Hz

    Returns
    -------
    List of relative S amplitude ratios for all event triplet combinations
    """

    evns = np.array(hdr["events"])

    mat = utils.concat_components(
        signal.demean_filter_window(
            arr,
            hdr["sampling_rate"],
            hdr["phase_start"],
            hdr["phase_end"],
            hdr["taper_length"],
            highpass,
            lowpass,
        )
    )

    Babcs, Bacbs, isorts, sigmas = pca_amplitudes_s(mat)

    # Iterate through solutions and assign event number in order
    s_amplitudes = [
        core.S_Amplitude_Ratios(
            hdr["station"],
            # Order events as in amplitude comparison
            # and reference to event list
            *evns[np.array([a, b, c])[isorts[n]]],
            Babcs[n],
            Bacbs[n],
            # Remember to re-order events also here
            s_misfit(mat[np.array([a, b, c])[isorts[n]], :], Babcs[n], Bacbs[n]),
            *sigmas,  # sigma1, sigma2, sigma3
            highpass,
            lowpass,
        )
        for n, (a, b, c, _, _, _) in enumerate(core.iterate_event_triplet(len(evns)))
    ]

    return s_amplitudes


def triplet_s_amplitudes(
    arr: np.ndarray,
    hdr: core.Header,
    highpass: float,
    lowpass: float,
    a: int,
    b: int,
    c: int,
    realign: bool = False,
) -> list[core.S_Amplitude_Ratios]:
    """Compute relative S amplitude ratios for one event triplet in arr

    ..note:
        The here implemented approach allows to filter each event pair
        individually allowing for more flexibility than
        :func:`principal_s_amplitudes` when comparing large differences in
        magnitude

    Parameters
    ----------
    arr:
        Waveform array of shape ``(3, components, samples)`` holding the event triplet
    hdr:
        Header with metadata, including sampling rate, phase start and end,
        taper length, and event information
    highpass:
        Highpass filter frequency in Hz
    lowpass:
        Lowpass filter frequency in Hz
    a:
        Number of event a
    b:
        Number of event b
    c:
        Number of event c
    realign:
        Re-align seismograms after applying filter

    Returns
    -------
    S amplitude ratios
    """

    if realign:
        mat = signal.subset_filter_align(
            arr, [0, 1, 2], highpass, lowpass, **hdr.kwargs(signal.subset_filter_align)
        )
    else:
        arr_sub = signal.demean_filter_window(
            arr,
            hdr["sampling_rate"],
            hdr["phase_start"],
            hdr["phase_end"],
            hdr["taper_length"],
            highpass,
            lowpass,
        )

        mat = utils.concat_components(arr_sub)

    Babc, Bacb, iord, sigmas = pca_amplitude_3s(mat)

    mis = s_misfit(mat[iord, :], Babc, Bacb)

    return core.S_Amplitude_Ratios(
        hdr["station"],
        *np.array((a, b, c))[iord],
        Babc,
        Bacb,
        mis,
        sigmas,
        highpass,
        lowpass,
    )


def info(
    amplitudes: list[core.P_Amplitude_Ratio] | list[core.S_Amplitude_Ratios],
    width: int = 80,
):
    """Print statistics of amplitude list to screen"""

    ip, _ = qc._ps_amplitudes(amplitudes)

    if ip:
        stas, evs, b = map(
            list, zip(*[(amp.station, amp.event_a, amp.event_b) for amp in amplitudes])
        )
        ip = True
        print("P wave amplitdue observations")
    else:
        stas, evs, b, c = map(
            list,
            zip(
                *[
                    (amp.station, amp.event_a, amp.event_b, amp.event_c)
                    for amp in amplitudes
                ]
            ),
        )
        ip = False
        print("S wave amplitdue observations")

    # Join event oversvations
    evs.extend(b)

    if not ip:
        evs.extend(c)

    nsta, usta = zip(
        *sorted([(stas.count(sta), sta) for sta in set(stas)], reverse=True)
    )

    nev, uev = zip(*sorted([(evs.count(ev), ev) for ev in set(evs)], reverse=True))

    necha = len(str(nev[0]))
    nscha = len(str(nev[0]))
    evcha = max(len(str(ev)) for ev in uev)
    stcha = max([len(st) for st in usta])

    out = "Stations:\n"
    previous = 0
    blockwidth = stcha + nscha + 3
    for nb, (us, ns) in enumerate(zip(usta, nsta)):
        if (llength := (nb * blockwidth) % width) < previous:
            out += "\n"

        out += "{sname:{swidth}}: {snum:{nwdth}} ".format(
            sname=us, swidth=stcha, snum=ns, nwdth=nscha
        )

        previous = llength

    out += "\n\nEvents:\n"
    previous = 0
    blockwidth = evcha + necha + 3
    for nb, (us, ns) in enumerate(zip(uev, nev)):
        if (llength := (nb * blockwidth) % width) < previous:
            out += "\n"

        out += "{sname:{swidth}}: {snum:{nwdth}} ".format(
            sname=us, swidth=evcha, snum=ns, nwdth=necha
        )

        previous = llength

    print(out)
