#!/usr/bin/env python

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
from scipy.linalg import svd
from scipy.sparse import coo_matrix
from itertools import combinations as combs
from relmt import utils, signal, core, ls, io
import mccore

logger = core.register_logger(__name__)


def _test_phase_shape(phase, mtx):
    """Validate phase label and number of events in waveform matrix

    Parameters
    ----------
    phase:
        Phase identifier, must be ``'P'`` or ``'S'``
    mtx:
        Waveform matrix ``(events, samples)`` or ``(events, components, samples)``

    Raises
    ------
    ValueError:
        If `phase` is not ``'P'`` or ``'S'``
    IndexError:
        When not enough events are provided for the chosen phase
    """
    if phase not in ["P", "S"]:
        raise ValueError("Phase must be either 'P' or 'S'")

    if (phase == "P" and mtx.shape[1] < 2) or (phase == "S" and mtx.shape[1] < 3):
        msg = f"mtx contains only {mtx.shape[1]} events. "
        msg += "At least 2 are required for P-waves, 3 for S waves."
        raise IndexError(msg)


def mccc_align(
    mtx: np.ndarray,
    phase: str,
    sampling_rate: float,
    maxshift: float,
    ndec: int = 1,
    combinations: np.ndarray = np.array([]),
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute time shifts that align the seismogram matrix

    Applies the multi-channel cross-correlation method (Bostock et al. 2021, BSSA)

    Parameters
    ----------
    mtx:
        Waveform matrix of shape ``(events, samples)``, with time aligned along
        last axis
    phase:
        Seismic phase ('P' or 'S')
    sampling_rate:
        Sampling rate of the wavefrom matrix (Hertz)
    maxshift:
        Maximum shift to test (seconds)
    ndec:
        Test correlation function at every ndec'th point
    combinations:
        Only combine these indices of mtx. ``(combinations, 2)`` for P-waves,
        ``(combinations, 3)`` for S-waves. When empty, combine all events.
    verbose:
        Print diagnostic output

    Returns
    -------
    time_shifts: :class:`numpy.ndarray`
        Zero-mean time shifts of shape ``(events,)`` that maximize cc
    cc: :class:`numpy.ndarray`
        Nominal correlation coefficients between events at optimal time lag
    dd: :class:`numpy.ndarray`
        Pairwise differential arrival times
    dd_res: :class:`numpy.ndarray`
        Resiudal when computing time lags from pairwise differential arrival
        times
    pairs: :class:`numpy.ndarray`
        Pairwise event combination indices pointing into ``mtx`` of shape
        ``(C,2)``, where `C` is `events * (events-1) / 2` for P waves and
        `events * (events-1) * (events-2)/ 3` for S waves

    Raises
    ------
    IndexError:
        When mtx contains less than 2 events for P waves, or less than 3
        for S waves
    ValueError:
        When phase is not 'P' or 'S'
    """
    _test_phase_shape(phase, mtx)
    sampling_interval = 1 / sampling_rate
    nev = mtx.shape[0]

    if phase == "P":
        fun = mccore.mccc_ppf
    elif phase == "S":
        fun = mccore.mccc_ssf0

    icombine = True
    if len(combinations) == 0:
        icombine = False
        ndim = 2
        ncombi = int(nev * (nev - 1) / 2)
        if phase == "S":
            ndim = 3
            ncombi = int(nev * (nev - 1) * (nev - 2) / 6)
        # combinations = np.empty((0, ndim), int)
        # Pre-combute combinations. This is also handled in the fortran codes,
        # but combination needs to be consistent with actual array sizes, so
        # parsing an empty array violates this
        combinations = np.array(list(combs(range(nev), ndim)))
    else:
        ncombi = combinations.shape[0]
        uc = np.unique(combinations)
        if np.any(uc < 0) or np.any(uc > mtx.shape[0]):
            outs = [f"{i}" for i in np.concatenate((uc[uc < 0], uc[uc > mtx.shape[0]]))]
            msg = "'combinations' indices out of bounds: "
            msg += ", ".join(outs)
            raise IndexError(msg)

    combinations += 1  # Fortran indexing

    # Data, sampling_interval, maxlag, ndec
    rowi, coli, valu, dd, cc = fun(
        mtx.T, sampling_interval, maxshift, ndec, combinations, verbose, nci=ncombi
    )

    combinations -= 1  # Return to python indexing

    # When limiting combinations, the last elements of the matrix of the matrix
    # remain unallocated. We detect this as -1 column indices
    if icombine:
        ivalid = coli > -1
        rowi = rowi[ivalid]
        coli = coli[ivalid]
        valu = valu[ivalid]
        dd = dd[: rowi[-1] + 1]

    # Make cc cubic (S) or symmetric square (P) matrix
    if phase == "S" and not icombine:
        cc = utils.reshape_ccvec(cc, nev)
    elif phase == "S":
        cc = utils.reshape_ccvec(cc, nev, combinations)

    A = coo_matrix((valu, (rowi, coli)), dtype=np.float64).tocsc()

    # TODO: The line below is quite a performance bottleneck
    dt, dd_res = ls.solve_irls_sparse(A, dd)

    # Event pairs in dd[:-1]
    evpair = np.zeros((len(dd) - 1, 2), dtype=int)
    evpair[:, 0] = coli[0 : 2 * (len(dd) - 1) : 2]
    evpair[:, 1] = coli[1 : 2 * (len(dd) - 1) : 2]

    return dt, cc, dd, dd_res, evpair


def pca_objective(sigma: np.ndarray, phase: str, ns: int) -> float:
    """Principal component alignment objective function

    for `P` phases:

    .. math::
        \\phi = s_0^2 / n

    For `S` phases:

    .. math::
        \\phi = 1 - s_2 / (s_0 + s_1)

    Parameters
    ----------
    sigma:
        Singular value vector of the singular value decompostion
    phase:
        Seismic phase to consider. 'P' or 'S'.
    ns:
        Number of seismograms

    Returns
    -------
    The objective value
    """
    # worst 0 -> 1 best
    if phase == "P":
        return sigma[0] ** 2 / ns
    elif phase == "S":
        return 1 - (sigma[2] / (sigma[0] + sigma[1]))


def pca_align(
    mtx: np.ndarray,
    sampling_rate: float,
    phase: str,
    iterations: int = 50,
    dphi: float = 0,
    dtime: float = 0,
) -> tuple[np.ndarray, float]:
    """Align waveforms by principal component analysis

    Implements method of Bostock et al. (2021, BSSA).

    For P-waves, we maximize the first of the singular values :math:`s` of
    :math:`n` waveforms:

    .. math::
        \\phi = s_0^2 / n

    For S-waves, we maximize the sum of the first two singular values:

    .. math::
        \\phi = 1 - s_2 / (s_0 + s_1),

    where :math:`\\phi` ranges from `0` (worst) to `1` (best)

    Parameters
    ----------
    mtx:
        Waveform matrix of shape ``(events, samples)``, with time aligned along
        last axis
    sampling_rate:
        Sampling rate of the wavefrom matrix (Hertz)
    phase:
        Which seismic phase to consider ('P' or 'S')
    iterations:
        Maximum number of iterations
    dphi:
        Minimum change in objective function to stop iteration
    dtime:
        Smallest maximal time shift to stop iteration

    Returns
    -------
    time_shifts : :class:`numpy.ndarray`
        Zero-mean time shifts of shape ``(events,)`` that maximize phi
    phi : float
        Value of objective function

    Raises
    ------
    IndexError:
        When mtx contains less than 2 events for P waves, or less than 3
        for S waves
    ValueError:
        When phase is not 'P' or 'S'
    """

    _test_phase_shape(phase, mtx)
    ns = mtx.shape[0]
    tshift_tot = np.zeros(ns)
    tshift = np.zeros(ns)

    ip = True
    if phase == "S":
        ip = False

    # Remove mean and normalize.
    scomp = signal.norm_power(signal.demean(mtx))
    scomp1 = np.zeros(scomp.shape)

    # Decompose seismograms into principal components
    # s * U are the contributions of U to each seismogram
    # s are the singular values ("relative importance")
    # Vh are the "principal seismograms"
    U, s, Vh = svd(scomp)

    # Delta epsilon (objective function)
    this_dphi = np.inf

    # Concentrate energy on first principal component.
    phi_old = pca_objective(s, phase, ns)

    logger.info(f"Starting pca_align for {phase} Phase")
    logger.debug(f"Phi {phase} = {phi_old :1.1e}")

    iter = 0
    while (
        iter < iterations
        and dphi < this_dphi
        and (iter == 0 or dtime < max(abs(tshift)))
    ):

        # Keep running sum for total shift
        tshift_tot = tshift_tot - tshift
        iter += 1

        if ip:
            # Create rank 1 ...
            scomp1 = s[0] * np.outer(U[:, 0], Vh[0, :])
        else:
            # ... rank 2 approximation from singular values
            scomp1 = np.zeros(scomp.shape)
            for i in np.arange(2):
                scomp1 += s[i] * np.outer(U[:, i], Vh[i, :])

        tshift = np.zeros((ns))

        # Differentiate waveforms: 2 options - differentiate full waveform or its
        # rank 2 approximation. The first can converge more slowly and sometimes
        # achieves better objective but not always. Split difference by averaging the two.
        scomp1d = (signal.differentiate(scomp + scomp1, sampling_rate)) / 2

        # Create perturbation from higher PC's (>2), resembling derivative.
        scomp2N = scomp - scomp1

        # Compute estimate of shift and correlation coefficients.
        # (Eq. 15) Bostock et al. (2021, BSSA)
        for i in range(ns):
            tshift[i] = np.dot(scomp1d[i, :], scomp2N[i, :]) / np.dot(
                scomp1d[i, :], scomp1d[i, :]
            )

        # Apply zero sum constraint and shift
        tshift -= tshift.mean()
        scomp = signal.shift(scomp, tshift, sampling_rate)

        # Re-computed SVD and compute new objective function
        U, s, Vh = svd(scomp)
        phi_new = pca_objective(s, phase, ns)

        this_dphi = phi_new - phi_old

        phi_old = phi_new

        logger.debug(
            "Iteration #{:d} phi: {:1.3e} (dphi: {:1.3e})".format(
                iter, phi_new, this_dphi
            )
        )

    logger.info("Finished after {:} iterations with eobj {:1.3e}".format(iter, phi_old))

    return tshift_tot, phi_old


def complete_paired_s_lag_times(
    evpairs: np.ndarray,
    dd: np.ndarray,
    cc: np.ndarray,
    dd_res: np.ndarray,
    method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair-wise from triplet-wise lag times from complete set.

    Compute pair-wise from the triplet-wise lag times ``dd`` returned by
    :func:`mccc_align`. Assume all combinations are present, i.e.
    `npair = events * (events-1) * (events-2) / 3`

    Parameters
    ----------
    ev_pair:
        ``(npair, 2)`` event indices
    dd:
        Corresponding ``(npair,)`` pairwise differential arrival times
    cc:
        ``(events, events, events)`` Cross correlation matrix
    dd_res:
        ``(npair,)`` residuals of the alignment
    method:
       - 'median': take the median of the tripletwise lag times
       - 'residual' choose the one with the lowest residual
       - 'cc' choose the one with the highest cross-correlation coefficient


    Returns
    -------
    ev_pair: :class:`numpy.ndarray`
        ``(events * (events-1) / 2, 2)`` event indices
    dd_pair: :class:`numpy.ndarray`
        Corresponding median differential arrival times
    cc_pair: :class:`numpy.ndarray`
        Corresponding average cross correlation coefficient
    res_pair: :class:`numpy.ndarray`
        Corresponding residuals of the differential arrival times. If 'method'
        is 'median', the RMS of all lag times
    """
    logger.debug("Gathering complete pair of S lag times")

    # All event indices
    nev = evpairs[-1, -1] + 1  # We except the highest event index at the end

    # Strip last zero that constrained the linear system to solve for time shifts
    dd1 = dd[:-1]
    dd_res1 = dd_res[:-1]

    if (
        (nev * (nev - 1) * (nev - 2) / 3 != evpairs.shape[0])
        or (evpairs.shape[0] != dd1.shape[0])
        or (nev - 1) != np.max(evpairs)
    ):
        msg = f"Arrays must be of same and correct length. Found {nev} events."
        raise IndexError(msg)

    # Event pair order for P
    ev_pair = np.array(
        [(a, b) for a in range(nev - 1) for b in range(a + 1, nev)], dtype=int
    )
    ddm = np.zeros(ev_pair.shape[0], dtype=float)
    ccm = np.zeros_like(ddm)
    res = np.zeros_like(ddm)

    # Third implicit differential time of the triplets
    implicit_dd = dd1[1::2] - dd1[0::2]
    implicit_dd_res = dd_res1[1::2] - dd_res1[0::2]
    implicit_evpairs = np.array([evpairs[0::2, 1], evpairs[1::2, 1]]).T

    for n, ab in enumerate(ev_pair):
        iin = np.all(evpairs == ab, axis=-1)
        iin2 = np.all(implicit_evpairs == ab, axis=-1)
        all_dds = np.concatenate((dd1[iin], implicit_dd[iin2]))
        all_dd_res = np.abs(np.concatenate((dd_res1[iin], implicit_dd_res[iin2])))

        # Choose a best time and weight from the set
        if method == "median":
            ddm[n] = np.median(all_dds)
            ccm[n] = utils.fisher_average(cc[ab[0], ab[1], :])
            res[n] = np.sqrt(np.sum(all_dd_res**2) / all_dds.shape[0])
        else:
            iother = np.full(cc.shape[2], True)
            if method == "cc":
                # Ignore zero cc values that arise from triplets containing the
                # considered event pair
                iother[ab] = False
                ibest = np.argmax(cc[ab[0], ab[1], iother])
            elif method == "residual":
                ibest = np.argmin(all_dd_res)
            else:
                raise ValueError(f"Unknown 'measure': {method}")
            ddm[n] = all_dds[ibest]
            ccm[n] = cc[ab[0], ab[1], iother][ibest]
            res[n] = all_dd_res[ibest]

    return ev_pair, ddm, ccm, res


def incomplete_paired_s_lag_times(
    evpairs: np.ndarray,
    dd: np.ndarray,
    cc: np.ndarray,
    dd_res: np.ndarray,
    combinations: np.ndarray,
    method: str = "cc",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair-wise from triplet-wise lag times from incomplete set.

    Compute pair-wise from the triplet-wise lag times ``dd`` returned by
    :func:`mccc_align` with a `combinations` list parsed. Do not attempt to
    compute implicit combinations

    Parameters
    ----------
    evpairs:
        ``(npair, 2)`` event indices
    dd:
        Corresponding pairwise differential arrival times
    cc:
        ``(events, events, events)`` Cross correlation matrix
    dd_res:
        ``(npair,)`` residuals of the alignment
    combinations:
        ``(npair/2, 3)`` triplet combinations that form the incomplete
        observations
    method:
       - 'residual' choose the one with the lowest residual
       - 'cc' choose the one with the highest cross-correlation coefficient


    Returns
    -------
    ev_pair: :class:`numpy.ndarray`
        ``(events * (events-1) / 2, 2)`` event indices
    dd_pair: :class:`numpy.ndarray`
        Corresponding median differential arrival times
    cc_pair: :class:`numpy.ndarray`
        Corresponding average cross correlation coefficient
    res_pair: :class:`numpy.ndarray`
        Corresponding residuals of the differential arrival times.
    """
    logger.debug("Gathering incomplete pair of S lag times")

    # Strip last zero that constrained the linear system to solve for time shifts
    dd1 = dd[:-1]
    dd_res1 = dd_res[:-1]

    if evpairs.shape[0] != dd1.shape[0]:
        msg = "Arrays must be of same length."
        raise IndexError(msg)

    # Present event pairs
    uniq_pairs = np.array(sorted(set([(a, b) for a, b in evpairs])))
    ddm = np.zeros(uniq_pairs.shape[0], dtype=float)
    ccm = np.zeros_like(ddm)
    res = np.zeros_like(ddm)

    for n, ab in enumerate(uniq_pairs):
        iin = np.all(evpairs == ab, axis=-1)

        # All 3rd events of the combination
        iab = np.sum((ab[0] == combinations) + (ab[1] == combinations), axis=-1) == 2
        ic = (combinations[iab, :] != ab[0]) & (combinations[iab, :] != ab[1])
        # But the 3rd event is not supposed to be on the first index, because
        # this would be an "implicit" event pair, which we are not considering
        # here.
        ic[:, 0] = False
        c = combinations[ic.nonzero()]  # 3rd events of the 'ab' indexed by iin

        all_dds = dd1[iin]
        all_dd_res = dd_res1[iin]

        if method == "cc":
            ibest = np.argmax(cc[ab[0], ab[1], c])

        elif method == "residual":
            ibest = np.argmin(all_dd_res)
        else:
            raise ValueError(f"Unknown 'measure': {method}")

        ddm[n] = all_dds[ibest]
        ccm[n] = cc[ab[0], ab[1], c][ibest]
        res[n] = all_dd_res[ibest]

    return uniq_pairs, ddm, ccm, res


def run(
    wvarr: np.ndarray,
    header: core.Header,
    destination: tuple[str, str, int, str],
    do_mccc: bool = True,
    do_pca: bool = True,
    mccc_combinations: np.ndarray = np.array([]),
):
    """Align waveforms and save results to disk

    Parameters
    ----------
    wvarr:
        ``(events, channels, samples)`` waveform array
    header:
        Dictionary holding the processing parameters
    destination:
        Tuple holding station name, phase type, alignment iteration number and
        destination root directory
    do_mccc:
        Align using multi-channel cross correlation.
    do_pca:
        Align using principal component analyses. This allows for sub-sample
        alignment, but impedes to combine cross correlation lag times of two
        successive runs.
    mccc_combinations:
        When aligning using mccc, only combine these pairs (P-waves) or triplets
        (S-waves).
    """

    if not (do_pca or do_mccc):
        raise ValueError(
            "Nothing to do. Must set at least one of: 'do_pca' or 'do_mccc'"
        )

    pwv = utils.concat_components(
        signal.demean_filter_window(wvarr, **header.kwargs(signal.demean_filter_window))
    )

    # Number of events
    nev = pwv.shape[0]

    if do_mccc:
        maxshift = header["phase_end"] - header["phase_start"]

        # Align using MCCC
        dt_cc, ccmc, ddmc, ddresmc, evpairsmc = mccc_align(
            pwv,
            verbose=True,
            maxshift=maxshift,
            combinations=mccc_combinations,
            **header.kwargs(mccc_align),
        )

        if len(evpairsmc) < 1:
            logger.warning(
                f"{header['station']}_{header['phase']}: Nothing to process."
            )
            return

        # Shift the traces
        arr_shift = signal.shift_3d(wvarr, -dt_cc, **header.kwargs(signal.shift_3d))
        wvmat_cc = utils.concat_components(
            signal.demean_filter_window(
                arr_shift, **header.kwargs(signal.demean_filter_window)
            )
        )

        # Re-compute the cc's on the shifted traces
        logger.info("Computing aligned correlation coefficients")

        # Recompute correlations with shifted traces
        ccijk, ccij, *_ = signal.correlation_averages(
            wvmat_cc,
            header["phase"],
            set_autocorrelation=False,
        )

        methods = [""]
        if header["phase"] == "S":
            methods = ["median", "residual", "cc"]

        # Use differen methods to extract S lag times
        for method in methods:
            if header["phase"] == "S":
                if evpairsmc.shape[0] == nev * (nev - 1) * (nev - 2) / 3:
                    # Complete set of lag time combinations
                    evpairs, dd, ccp, dd_res = complete_paired_s_lag_times(
                        evpairsmc, ddmc, ccijk, ddresmc, method
                    )
                else:
                    # The set is incomplete, so we can't compute the implicit
                    # combinations
                    if method == "median":
                        continue
                    evpairs, dd, ccp, dd_res = incomplete_paired_s_lag_times(
                        evpairsmc, ddmc, ccijk, ddresmc, mccc_combinations, method
                    )
            else:
                dd = ddmc[:-1]
                dd_res = ddresmc[:-1]
                ccp = ccij[evpairsmc[:, 0], evpairsmc[:, 1]]
                evpairs = evpairsmc
            # Silly format before saving lag times

            # Look up actual event names
            evns = np.vectorize(header["events_"].__getitem__)(evpairs)

            evdd = np.char.mod(
                ["% 9.0f", "% 9.0f", "%13.6e", "%6.3f", "%13.2e"],
                np.hstack(
                    (evns, dd[:, np.newaxis], ccp[:, np.newaxis], dd_res[:, np.newaxis])
                ),
            )

            # Save the lag times
            np.savetxt(
                core.file("mccc_lag_times", *destination, suffix=method),
                evdd,
                fmt="%s",
                header=" EventA    EventB    LagTime(s)     CC   Residual(s)",
            )

        # Sum alignment residuals for each event
        nev = len(header["events_"])
        rms = np.zeros(nev)
        for iev in range(nev):
            try:
                iin = np.any(evpairs == iev, axis=-1)
                rms[iev] = np.sqrt(np.sum(dd_res[iin] ** 2) / np.sum(iin))
            except IndexError:
                rms[iev] = np.nan

        # Save the time shifts
        io.save_results(
            core.file("mccc_time_shift", *destination), np.vstack((dt_cc, rms)).T
        )

        # Save the cc matrix, values are on the shifted traces
        io.save_results(core.file("cc_matrix", *destination), ccij)

    else:
        # If we don't align with mccc, just, preprocess the input array
        wvmat_cc = utils.concat_components(
            signal.demean_filter_window(
                wvarr, **header.kwargs(signal.demean_filter_window)
            )
        )
        dt_cc = np.zeros(wvarr.shape[0])

    if do_pca:
        # Align using PCA
        dt_pca, phi = pca_align(wvmat_cc, dphi=1e-9, **header.kwargs(pca_align))

        io.save_results(core.file("pca_time_shift", *destination), dt_pca)
        io.save_results(core.file("pca_objective", *destination), phi)

        # Apply shift to input array and save
        arr_shift = signal.shift_3d(
            wvarr,
            -dt_pca - dt_cc,
            **header.kwargs(signal.shift_3d),
        )

    # We are saving a numpy array, not matlab
    header["variable_name"] = None

    # Fresh start with QC parameters
    header["min_expansion_coefficient_norm"] = None
    header["min_signal_noise_ratio"] = None
    header["min_correlation"] = None

    # Write out header and array files
    arrf = core.file("waveform_array", *destination)
    hdrf = core.file("waveform_header", *destination)

    # Save everything
    np.save(arrf, arr_shift)
    header.to_file(hdrf, True)
