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
import logging
from typing import Iterable
from relmt import utils, core, align

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def _gauss(n: int, sig: float, de: float):
    return np.exp(-1 / 2 * (np.arange(-n / 2 - de, n / 2 - de) / sig) ** 2)


def dB(ratio: float | np.ndarray) -> float | np.ndarray:
    """Return ratio in decibel"""
    return 10 * np.log10(ratio)


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
        A / \sqrt \sum A^2

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


@core._doc_config_args
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


@core._doc_config_args
def indices_signal(
    sampling_rate: float,
    phase_start: float,
    phase_end: float,
    data_window: float,
) -> tuple[int, int]:
    """Return first and last indices of the phase signal

    Parameters
    ----------

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

    logger.error("Use relmt.signal.shift instead of shift_3d")

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


@core._doc_config_args
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
):
    """
    Return the highest highpass and the lowest lowpass filter band

    Parameters
    ----------
    highpasses, lowpasses:
        List of highpass and lowpass filter corner candidates
    min_dynamic_range:
        Minimum ratio (dB) between lowpass and highpass

    Returns
    -------
    highpass, lowpass: float
        Filter corners (Hz), `None` if filter band is below dynamic range.
    """

    hpas = np.max(highpasses)
    lpas = np.min(lowpasses)

    if (dr := dB(lpas / hpas)) < min_dynamic_range:
        logger.debug(
            "Dynamic range of {:.1e} below threshold. Returning None".format(dr)
        )
        return None, None

    return hpas, lpas


@core._doc_config_args
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

    Returns
    -------
    Signal noise ratio
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


@core._doc_config_args
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


@core._doc_config_args
def subset_filter_align(
    arr: np.ndarray,
    indices: tuple[int],
    lpas: float,
    hpas: float,
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
    lpas, hpas:
        Highpass, lowpass filter corners


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
        \sum_i(x_i y_i) (\sum_i x_i^2 \sum_i y_i^2)^{-1/2}

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
