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
from scipy.signal import bessel, butter, lfilter, filtfilt
import scipy.fft as fft
from typing import Iterable
from relmt import utils, core, align, extra, qc

logger = core.register_logger(__name__)


def _gauss(n: int, sig: float, de: float):
    """Gaussian envelope used by :func:`make_wavelet`

    Parameters
    ----------
    n:
        Number of samples
    sig:
        Standard deviation of the Gaussian (samples)
    de:
        Phase shift of the envelope center (samples)

    Returns
    -------
    The specified Gauss envelope
    """
    return np.exp(-1 / 2 * (np.arange(-n / 2 - de, n / 2 - de) / sig) ** 2)


def dB(ratio: float | np.ndarray) -> float | np.ndarray:
    """Ratio expressed in decibel"""
    return 10 * np.log10(ratio)


def fraction(db: float | np.ndarray) -> float | np.ndarray:
    """Ratio in decibel expressed as fraction"""
    return 10 ** (db / 10)


def make_wavelet(
    n: int,
    p: float,
    typ: str = "sin",
    we: float = np.inf,
    ds: float = 0.0,
    de: float = 0.0,
) -> np.ndarray:
    """
    Make a wavelet

    Parameters
    ----------
    n:
        length of wavelet (samples)
    p:
        period length of the signal (samples)
    typ:
        Return a sine ('sin'), cosine ('cos'), or rectangle ('rec') function
    we:
        1-sigma width of the gaussian envelope (samples)
    ds:
        shift of signal to the right (samples)
    de:
        shift of envelope to the right (samples)
    Returns
    -------
    The wavelet
    """

    typs = ["sin", "cos", "rec"]

    if typ == "sin":
        osc = np.sin(np.arange(-ds, n - ds) * 2 * np.pi / p)
    elif typ == "cos":
        osc = np.cos(np.arange(-ds, n - ds) * 2 * np.pi / p)
    elif typ == "rec":
        osc = (np.arange(-ds, n - ds) // (p // 2) % 2) * 2 - 1
    else:
        msg = f"Unrecognized typ {typ}. Must be one of: " + ", ".join(typs)
        raise ValueError(msg)

    return _gauss(n, we, de) * osc


def norm_power(array: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalize input array by power

    .. math::
        A / \\sqrt \\sum A^2

    Parameters
    ----------
    array:
        to normalize
    axis:
        of the array along which to operate

    Returns
    -------
    Normalized array
    """
    return array / np.sqrt(np.sum(array**2, axis=axis, keepdims=True))


# @core._doc_config_args
def indices_inside_taper(
    sampling_rate: float,
    taper_length: float,
    phase_start: float,
    phase_end: float,
    data_window: float,
) -> tuple[int, int]:
    """First and last indices of trace that are not zeroed through tapering

    Parameters
    ----------
    sampling_rate:
        Sampling rate of the seismic waveform (Hertz)
    taper_length:
        Combined length of taper that is applied at both ends beyond the phase
        window. (seconds)
    phase_start:
        Start of the phase window before the arrival time pick (negative seconds
        before pick).
    phase_end:
        End of the phase window after the arrival time pick (seconds after
        pick).
    data_window:
        Time window symmetric about the phase pick (i.e. pick is near the
        central sample) (seconds)

    Returns
    -------
    First and last index inside the taper
    """

    t0 = data_window / 2 + phase_start
    t1 = data_window / 2 + phase_end
    tl = taper_length / 2
    i0 = max(0, int((t0 - tl) * sampling_rate))
    i1 = min(int(data_window * sampling_rate), int((t1 + tl) * sampling_rate))
    return i0, i1


# @core._doc_config_args
def indices_signal(
    sampling_rate: float,
    phase_start: float,
    phase_end: float,
    data_window: float,
) -> tuple[int, int]:
    """Return first and last indices of the phase signal

    Parameters
    ----------
    sampling_rate:
        Sampling rate of the seismic waveform (Hertz)
    phase_start:
        Start of the phase window before the arrival time pick (negative seconds
        before pick).
    phase_end:
        End of the phase window after the arrival time pick (seconds after
        pick).
    data_window:
        Time window symmetric about the phase pick (i.e. pick is near the
        central sample) (seconds)

    Returns
    -------
    First and last index inside the phase window
    """

    t0 = data_window / 2 + phase_start
    t1 = data_window / 2 + phase_end
    i0 = max(0, int(t0 * sampling_rate))
    i1 = min(int(data_window * sampling_rate), int(t1 * sampling_rate))
    return i0, i1


def indices_noise(
    sampling_rate: float,
    phase_start: float,
    phase_end: float,
    data_window: float,
) -> tuple[int, int]:
    """Return first and last indices of the pre-singal noise window. Attempt equal length.

    Parameters
    ----------

    Returns
    -------
    First and last index inside the noise window
    """

    t1 = data_window / 2 + phase_start
    dt = phase_end - phase_start

    i0 = max(0, int((t1 - dt) * sampling_rate))
    i1 = int(t1 * sampling_rate)

    return i0, i1


def shift_3d(wvf_array: np.ndarray, dt: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Shift a waveform matrix in Fourier space

    TODO: Tests indicate this is superseed by shift

    Parameters
    ----------
    wvf_array : (E, C, S) :class:`~numpy.array`:
        Waveform array, with time aligned along last axis
    dt : (S,) :class:`~numpy.array`:
        Time shifts positive to the right per seismogram (seconds)
    sampling_rate: float
        Sampling rate (Hertz)

    Returns
    -------
    out : (E, C, S) :class:`~numpy.array`:
        Shifted seismogram section
    """

    # TODO: how generalize  relmt.signal.shift

    out = np.zeros(wvf_array.shape)
    for ic in range(wvf_array.shape[1]):
        out[:, ic, :] = shift(wvf_array[:, ic, :], dt, sampling_rate)
    return out


def shift(mtx: np.ndarray, dt: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Shift a seismogram section in Fourier space

    Parameters
    ----------
    mtx:
        Seismogram section of shape ``(events, samples)``
    dt:
        Time shifts positive to the right per seismogram (seconds)
    sampling_rate
        Sampling rate (Hertz)

    Returns
    -------
    ``(events, samples)`` shifted seismogram section
    """

    n = mtx.shape[-1]
    n2 = n // 2 + 1
    sfn = fft.irfft(
        fft.rfft(mtx)
        * np.exp(
            np.outer(
                dt, (-1j * np.array(np.arange(n2) * 2 * np.pi / (n / sampling_rate)))
            )
        ),
        n,
    )
    return sfn


# @core._doc_config_args
def cosine_taper(
    arr: np.ndarray,
    taper_length: float,
    sampling_rate: float,
    phase_start: float | None = None,
    phase_end: float | None = None,
) -> np.ndarray:
    """
    Taper time series along last axis.

    Parameters
    ----------
    arr:
        ``(..., N)`` seismogram section, with time aligned along last axis
    taper_length:
        Combined length of taper that is applied at both ends beyond the phase
        window. (seconds)
    sampling_rate:
        Sampling rate of the seismic waveform (Hertz)
    phase_start:
        Start of the phase window before the arrival time pick (negative seconds
        before pick).
    phase_end:
        End of the phase window after the arrival time pick (seconds after
        pick).

    Returns
    -------
    Tapered seismogram section

    Raises
    ------
    ValueError:
        If taper falls outside array dimensions
    """

    nx = arr.shape[-1]

    taper = np.ones(nx)
    tt = taper_length / 2  # Taper time
    nt = int(tt * sampling_rate)  # Numper of taper indices
    it = np.arange(nt + 1) / sampling_rate / tt
    ct = 0.5 * (1 - np.cos(np.pi * it))

    if phase_start is not None:
        n1 = nx / 2 + phase_start * sampling_rate
        it1 = int(n1 + 1)
        if n1 < tt:
            msg = f"phase_start ({phase_start}) is outside domain of funct: "
            msg += f"talper_length/2 ({tt})"
            raise ValueError(msg)
        taper[it1 - nt - 1 : it1] = ct
        taper[0 : it1 - nt] = 0
    if phase_end is not None:
        n2 = nx / 2 + phase_end * sampling_rate
        it2 = int(n2 + 1)
        if n2 > nx - tt:
            msg = f"phase_end ({phase_end}) is outside domain of funct: "
            msg += (
                f"nx / sampling_rate - taper_length/2 ({nx} / {sampling_rate} - {tt})"
            )
            raise ValueError(msg)
        taper[it2 - 1 : it2 + nt] = np.flip(ct)
        taper[it2 + nt - 1 : nx] = 0

    def _apply(arr):
        return arr * taper

    return np.apply_along_axis(_apply, -1, arr)


def demean(arr: np.ndarray) -> np.ndarray:
    """Remove mean value along last dimension from array"""

    def _demean(arr):
        return arr - np.mean(arr)

    return np.apply_along_axis(_demean, -1, arr)


def destep(wvf_array: np.ndarray) -> np.ndarray:
    """Remove step between first and last sample value along last dimension
    from array."""

    def _destep(arr):
        ndat = len(arr)
        return arr - arr[0] + np.arange(ndat) * (arr[0] - arr[-1]) / float(ndat - 1)

    return np.apply_along_axis(_destep, -1, wvf_array)


def rotate_nez_rtz(nez: np.ndarray, backazimuth: float) -> np.ndarray:
    """Rotate seismogram array from N-E-Z into R-T-Z system"""
    baz = np.radians(backazimuth)
    rtz = nez.copy()
    rtz[..., 0, :] = -np.cos(baz) * nez[..., 0, :] - np.sin(baz) * nez[..., 1, :]
    rtz[..., 1, :] = np.sin(baz) * nez[..., 0, :] - np.cos(baz) * nez[..., 1, :]
    return rtz


def zero_events(wvf_array: np.ndarray, event_indices: Iterable[int]) -> np.ndarray:
    """Set seismograms of event_indices to zero"""
    zeros = np.zeros(wvf_array.shape[-1])
    wvf_array[event_indices, ..., :] = zeros
    return wvf_array


def differentiate(arr: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Differentiate a seismogram section in Fourier space

    Parameters
    ----------
    arr:
        ``(..., samples)`` Seismogram section, with time aligned along last axis
    sampling_rate : float
        Sampling rate (Hertz)

    Returns
    -------
    Differentiated seismogram section
    """

    n = np.shape(arr)[-1]
    n2 = n // 2 + 1
    w = 1j * np.arange(n2) * 2 * np.pi / (n / sampling_rate)
    sfn = fft.irfft(fft.rfft(arr) * w, n)
    return sfn


def integrate(arr: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Integrate a seismogram section in Fourier space

    Parameters
    ----------
    arr:
        ``(..., samples)`` Seismogram section, with time aligned along last axis
    sampling_rate : float
        Sampling rate (Hertz)

    Returns
    -------
    Integrated seismogram section
    """

    n = np.shape(arr)[-1]
    n2 = n // 2 + 1

    # Avoid zero frequency
    # w = 1j * np.arange(1, n2 + 1) * 2 * np.pi / (n / sampling_rate)
    w = 1j * np.arange(1, n2 + 1) * np.pi / (n / sampling_rate)
    sfn = fft.irfft(fft.rfft(arr) / w, n)
    return sfn


def choose_passband(
    highpasses: list[float], lowpasses: list[float], min_dynamic_range: float = 1
) -> tuple[float, float] | tuple[None, None]:
    """Return the highest highpass and the lowest lowpass filter band

    Parameters
    ----------
    highpasses, lowpasses:
        List of highpass and lowpass filter corner candidates
    min_dynamic_range:
        Minimum ratio (dB) between lowpass and highpass. If resulting filter
        bandwidth is lower, and a positive value is given, return `None`. If a
        negative value is given, interpret the absolute value and relax the
        highpass filter corner.

    Returns
    -------
    highpass, lowpass:
        Filter corners (Hz), `None` if filter band is below the positive
        `min_dynamic_range` or any of `highpasses` or `lowpasses` is not
        finite.
    """

    hpas = np.max(highpasses)
    lpas = np.min(lowpasses)

    if any(~np.isfinite([hpas, lpas])):
        return None, None

    # Positive strict, negative not strict
    strict = bool(np.sign(min_dynamic_range) + 1)

    if (dr := dB(lpas / hpas)) < abs(min_dynamic_range) and strict:
        msg = "Dynamic range of {:.3g} below positive threshold.".format(dr)
        msg += "Returning `None`"
        logger.debug(msg)
        return None, None

    elif dr < abs(min_dynamic_range) and not strict:
        msg = "Dynamic range of {:.3g} below absolute negative threshold.".format(dr)
        msg += "Relaxing highpass."
        logger.debug(msg)
        hpas = lpas / 10 ** (abs(min_dynamic_range) / 10)

    return hpas, lpas


# @core._doc_config_args
def signal_noise_ratio(
    arr: np.ndarray,
    sampling_rate: float,
    phase_start: float,
    phase_end: float,
    taper_length: float,
    highpass: float | None = None,
    lowpass: float | None = None,
) -> float | np.ndarray:
    """Calculate combined signal to noise ratio of all components in arr

    Parameters
    ----------
    arr:
        ``(..., samples)`` Seismogram section, with time aligned along last axis
    sampling_rate:
        Sampling rate of the seismic waveform (Hertz)
    phase_start:
        Start of the phase window before the arrival time pick (negative seconds
        before pick).
    phase_end:
        End of the phase window after the arrival time pick (seconds after
        pick).
    taper_length:
        Combined length of taper that is applied at both ends beyond the phase
        window. (seconds)
    highpass:
        Common high-pass filter corner of the waveform (Hertz)
    lowpass:
        Common low-pass filter corner of the waveform (Hertz)

    Returns
    -------
    Signal noise ratio in dB
    """

    def rms(arr):
        # Root mean square
        return np.sqrt(np.mean(np.square(arr), axis=-1))

    data_window = arr.shape[-1] / sampling_rate
    taper_offset = data_window / 2 - taper_length / 2

    # Pre-process
    arr = demean(arr)
    arr = cosine_taper(
        arr,
        sampling_rate=sampling_rate,
        phase_start=-taper_offset,
        phase_end=taper_offset - 1 / sampling_rate,
        taper_length=taper_length,
    )

    if highpass is not None or lowpass is not None:
        arr = filter(
            arr, sampling_rate=sampling_rate, highpass=highpass, lowpass=lowpass
        )

    # Indices inside phase window
    is0, is1 = indices_inside_taper(
        sampling_rate, taper_length, phase_start, phase_end, data_window
    )

    # Reduce array dimensions
    if arr.ndim > 2:
        sig = utils.concat_components(arr[:, :, is0:is1])
        noi = utils.concat_components(arr[:, :, :is0])
    else:
        sig = arr[..., is0:is1]
        noi = arr[..., :is0]

    nms = rms(noi)
    sms = rms(sig)

    # Calculate signal/noise ratio in dB
    return dB(sms / nms)


_filter_coefficients: dict[
    tuple[tuple[float | None, float | None], int, str], tuple[float, float]
] = {}


def _compute_filter_coefficients(rel_highpass, rel_lowpass, order, typ):
    """
    Coeffiecients of scipy filt and filtfilt

    Parameters
    ----------
    rel_highpass, rel_lowpass
        highpass and lowpass frequency devided by Nyquist frequency
    order : int
        Order of the filter
    typ : str
        "bessel" or "butter" for Bessel or Butterworth filter

    Returns
    -------
    b, a
        filter coefficients
    """

    if typ == "bessel":
        ffun = bessel
    elif typ == "butter":
        ffun = butter
    else:
        msg = f"Unknown filter type: {typ}"
        raise ValueError(msg)

    if rel_highpass is None:
        if rel_lowpass is None:
            raise ValueError("Neither lpas nor hpas are specified")
        else:
            wn = rel_lowpass
            return ffun(order, wn, "lowpass")
    else:
        if rel_lowpass is None:
            wn = rel_highpass
            return ffun(order, wn, "highpass")
        else:
            wn = (rel_highpass, rel_lowpass)
            return ffun(order, wn, "bandpass")


def filter(
    wvf: np.ndarray,
    sampling_rate: float,
    highpass: float | None = None,
    lowpass: float | None = None,
    order: int = 2,
    zerophase: bool = False,
    typ: str = "bessel",
) -> np.ndarray:
    """
    Filter the waveform array or matrix along last dimension. Highpass, lowpass,
    or bandpass is applied when high, low, or both frequency corners are given.

    Parameters
    ----------
    wvf:
        Waveform array or matrix
    sampling_rate:
        Sampling rate (Hertz)
    highpass, lowpass:
        Highpass and lowpass filter corners (Hz)
    order:
        Order of the filter
    zerophase: bool
        Use scipy.filtfilt instead of lfilt to get a zerophase filter
    typ:
        "bessel" or "butter" for Bessel or Butterworth filter

    Returns
    -------
    Filtered waveforms

    Raises
    ------
    ValueError:
        If neither highpass, nor lowpass are given
    ValueError:
        If given unknown filter type
    """

    nyq = 0.5 * sampling_rate

    if highpass is not None:
        highpass /= nyq

    if lowpass is not None:
        lowpass /= nyq

    fkey = ((highpass, lowpass), order, typ)

    if fkey in _filter_coefficients:
        b, a = _filter_coefficients[fkey]
    else:
        b, a = _compute_filter_coefficients(highpass, lowpass, order, typ)
        _filter_coefficients[fkey] = (b, a)

    if zerophase:
        return filtfilt(b, a, wvf)

    return lfilter(b, a, wvf)


# @core._doc_config_args
def demean_filter_window(
    arr: np.ndarray,
    sampling_rate: float,
    phase_start: float,
    phase_end: float,
    taper_length: float,
    highpass: float | None = None,
    lowpass: float | None = None,
    zerophase: bool = True,
) -> np.ndarray:
    """
    Demean, filter and taper waveform array

    Parameters
    ----------
    arr:
        Waveform array to process
    sampling_rate:
        Sampling rate of the seismic waveform (Hertz)
    phase_start:
        Start of the phase window before the arrival time pick (negative seconds
        before pick).
    phase_end:
        End of the phase window after the arrival time pick (seconds after
        pick).
    taper_length:
        Combined length of taper that is applied at both ends beyond the phase
        window. (seconds)
    highpass:
        Common high-pass filter corner of the waveform (Hertz)
    lowpass:
        Common low-pass filter corner of the waveform (Hertz)
    zerophase:
        Apply filter a second time reversed, so that resulting signal has no
        phase shift

    Returns
    -------
    Filtered, windowed and tapered waveform array

    """
    arr = demean(arr)
    arr = filter(
        arr,
        sampling_rate=sampling_rate,
        highpass=highpass,
        lowpass=lowpass,
        zerophase=zerophase,
    )
    arr = cosine_taper(
        arr,
        sampling_rate=sampling_rate,
        phase_start=phase_start,
        phase_end=phase_end,
        taper_length=taper_length,
    )
    return arr


# @core._doc_config_args
def subset_filter_align(
    arr: np.ndarray,
    indices: list[int],
    hpas: float,
    lpas: float,
    phase: str,
    sampling_rate: float,
    phase_start: float,
    phase_end: float,
    taper_length: float,
) -> np.ndarray:
    """Choose seismogram subset and align within filterband

    Parameters
    ----------
    arr:
        ``(events, channels, samples)`` 3-D seismogram array
    indices:
        Event indices to select from `arr`
    hpas:
        Highpass, ...
    lpas:
        Lowpass filter corners
    phase:
        Seismic phase type to consider ('P' or 'S')
    sampling_rate:
        Sampling rate of the seismic waveform (Hertz)
    phase_start:
        Start of the phase window before the arrival time pick (negative seconds
        before pick).
    phase_end:
        End of the phase window after the arrival time pick (seconds after
        pick).
    taper_length:
        Combined length of taper that is applied at both ends beyond the phase
        window. (seconds)


    Returns
    -------
    ``(events, samples * channels)`` 2-D seismogram matrix
    """
    # Pre-process
    arr_sub = demean_filter_window(
        arr[indices, :, :],
        sampling_rate,
        phase_start,
        phase_end,
        taper_length,
        hpas,
        lpas,
    )

    mat = utils.concat_components(arr_sub)

    # Align
    dt, *_ = align.mccc_align(mat, phase, sampling_rate, 1 / lpas)
    arr_sub = shift_3d(arr[indices, :, :], -dt, sampling_rate)

    # Re-process to avoid shifting artifacts
    arr_sub = demean_filter_window(
        arr_sub,
        sampling_rate,
        phase_start,
        phase_end,
        taper_length,
        hpas,
        lpas,
    )

    return utils.concat_components(arr_sub)


def cc_coef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the correlation coefficient between two equal-length vectors:

    .. math::
        \\sum_i(x_i y_i) (\\sum_i x_i^2 \\sum_i y_i^2)^{-1/2}

    Parameters
    ----------
    x, y :
        Input vectors of same length ``(samples,)``

    Returns
    -------
    Correlation coefficient between x and y.

    Note
    ----
    The correlation coefficient is a measure of linear association between two
    variables.
    """
    return sum(x * y) / np.sqrt(sum(x**2) * sum(y**2))


def correlation_averages(
    mat: np.ndarray,
    phase: str,
    set_autocorrelation=True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Correlation coefficients between event pairs or triplets

    For each pair (if `phase` is 'P') of triplet (if `phase` is 'S') of
    waveforms, compute the reconstruction correlation coefficient. Consolidate
    by also computing the average values.

    Note
    ----
    Consolidated (averaged) correlation coefficients are of absolute values and
    hence positive

    Parameters
    ----------
    mat:
        Waveform matrix of shape ``(events, samples)``
    phase:
        'P' for pairwise correlation, 'S' for triplet-wise correlation
    set_autocorrelation:
        Set correlations with two matching indices to unity in ccijk and ccij

    Returns
    -------
    ccijk:
        Correlation coefficients between event triplets of shape
        ``(events, events, events)``. NaN if `phase` is 'P'
    ccij:
        Correlation coefficients between event pairs if `phase` is 'P'.
        Averaged correlation over `k` if `phase` is 'S'. Shape ``(events, events)``.
    cci:
        Average absolute correlation coefficient of all pairs or triplets of shape
        ``(events,)``
    cc:
        Average absolute correlation coefficient of the matrix
    """
    logger.info(
        "Computing correlation coefficients of all event combinations. This may take a while..."
    )
    n = mat.shape[0]

    if phase == "P":
        ccij = np.corrcoef(mat)

        # Fisher average expects no contribution of auto-correlation
        ccij[np.diag_indices_from(ccij)] = 0.0

        # Triplet wise correlation are not defined
        ccijk = np.full((n, n, n), np.nan)

    elif phase == "S":
        # ccorf3 expects normalized matrix
        nmat = mat / np.linalg.norm(mat, axis=-1)[:, np.newaxis]

        ccijk = utils.ccorf3_all(nmat.T)

        # Correct nummerical corner cases in ccorf3
        ccijk[np.isnan(ccijk)] = 1.0
        ccijk[ccijk < -1.0] = -1.0
        ccijk[ccijk > 1.0] = 1.0

        ccij = utils.fisher_average(ccijk)

        # Two indices that are the same (due to ChatGPT o4-mini-high)
        if set_autocorrelation:
            iij = np.argwhere(
                (I := np.eye(n, dtype=bool))[:, :, None] | I[:, None, :] | I[None, :, :]
            )
            ccijk[iij] = 1.0

    else:
        raise ValueError(f"'phase' must be 'P' or 'S', not: '{phase}'")

    cci = utils.fisher_average(abs(ccij))
    cc = utils.fisher_average(cci)

    if set_autocorrelation:
        ccij[np.diag_indices_from(ccij)] = 1.0

    return ccijk, ccij, cci, cc


def phase_passbands(
    arr: np.ndarray,
    hdr: core.Header,
    evd: dict[int, core.Event],
    exclude: core.Exclude | None = None,
    auto_lowpass_method: str | None = None,
    auto_lowpass_stressdrop_range: tuple[float, float] = [1.0e6, 1.0e6],
    auto_bandpass_snr_target: float | None = None,
) -> dict[str, list[float]]:
    """Compute the bandpass filter corners for a waveform array.

    This function computes the bandpass filter corners for each event in the
    waveform array. The following logic is applied:

    The lowpass corner should be the corner frequency of the phase spectrum. We
    estimate it as:

    - 1/source duration of the event magnitude when `lowpass_method` is
      'duration'.

    - Based on the corner frequency pertaining to a stressdrop (Pa) within
      `lowpass_stressdrop_range` when `lowpass_method` is 'corner':

    - If the upper bound is smaller or equal the lower bound (i.e. no range is
      given), estimate the corner frequency using :func:`utils.corner_frequency`
      with an S-wave velocity of 4 km/s.

    - If a range is given, we convert it to a corner frequency range as above
      and search the apparent corner frequency as the maximum of the phase
      velocity spectrum within this range using
      :func:`extra.apparent_corner_frequency`

    The default highpass corner is chosen as 1/phase length.

    When `bandpass_snr_target` is given, we determine the frequency band in the
    signal and noise spectra for which the signal-to-noise ratio is larger than
    `bandpass_snr_target` (dB), compare with the above estimates, and return the
    highest highpass and the lowest lowpass corner.

    Parameters
    ----------
    arr:
        The waveform array with shape ``(events, channels, samples)``.
    hdr:
        The header containing metadata about the waveform array, including
        phase phase start and end times, sampling rate, included events.
    evd:
        The seismic event catalog.
    exclude:
        An optional exclude object with observations to be excluded from the
        computation. If None, all observatinos are included.

    Returns
    -------
    Dictionary mapping phaseIDs (event ID, station, phase) to filter corners
    [highpass, lowpass].

    Raises
    ------
    ValueError:
        If an unknown lowpass method is specified.
    """

    ievs, evns = qc.included_events(exclude, **hdr.kwargs(qc.included_events))

    pha = hdr["phase"]
    sta = hdr["sta"]

    # At least one period within window
    fwin = 1.0 / (hdr["phase_end"] - hdr["phase_start"])

    # One sample more than the Nyquist frequency
    fnyq = (arr.shape[-1] - 1.0) / hdr["data_window"] / 2

    bpd = {}
    for nev, (iev, evn) in enumerate(zip(ievs, evns)):
        print(
            "{:6s} {:1s}: {: 4d} events to go   ".format(sta, pha, len(evns) - nev),
            end="\r",
        )

        ev = evd[evn]

        phase_arr = arr[iev, :, :]

        # First get the corner frequency
        if auto_lowpass_method == "duration":
            # Corner frequency from duration
            fc = 1 / utils.source_duration(ev.mag)

        elif auto_lowpass_method == "corner":
            # Corner frequency from stress drop

            if auto_lowpass_stressdrop_range[0] >= auto_lowpass_stressdrop_range[1]:
                # If no range is given, don't look into the spectrum
                fc = utils.corner_frequency(
                    ev.mag, pha, auto_lowpass_stressdrop_range[0], 4000
                )

            else:
                # Convert stressdrop range to possile corner frequencies
                fcmin = utils.corner_frequency(
                    ev.mag, pha, auto_lowpass_stressdrop_range[0], 4000
                )
                fcmax = utils.corner_frequency(
                    ev.mag, pha, auto_lowpass_stressdrop_range[1], 4000
                )

                # Isolate the signal
                isig, _ = indices_signal(**hdr.kwargs(indices_signal))
                sig = demean(phase_arr[:, isig:])

                # Try to compute the corner frequency
                try:
                    fc = extra.apparent_corner_frequency(
                        sig, hdr["sampling_rate"], fmin=fcmin, fmax=fcmax
                    )
                except ValueError:
                    # It might be outside the range of the signal
                    fc = fcmax
        else:
            raise ValueError(f"Unknown lowpass method: {auto_lowpass_method}")

        # No SNR optimization, use the corner frequency
        hpas = min(fwin, fnyq)
        lpas = min(fc, fnyq)

        if auto_bandpass_snr_target is not None and hpas < lpas:
            # Try to optimize bandpass within range
            hpas, lpas = extra.optimal_bandpass(
                phase_arr,
                fmin=hpas,
                fmax=lpas,
                min_snr=auto_bandpass_snr_target,
                **hdr.kwargs(extra.optimal_bandpass),
            )

        # Return the filter corners
        bpd[evn] = [float(hpas), float(lpas)]

    return bpd
