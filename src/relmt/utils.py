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
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import svd
from scipy.signal import bessel, butter, lfilter
import scipy.fft as fft
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logsh = logging.StreamHandler()
logsh.setLevel(logging.DEBUG)
logsh.setFormatter(
    logging.Formatter(
        "{levelname: <8s} {asctime} {name}.{funcName}: {message}", style="{"
    )
)

logger.addHandler(logsh)


def _gauss(n: int, sig: float, de: float):
    return np.exp(-1 / 2 * (np.arange(-n / 2 - de, n / 2 - de) / sig) ** 2)


def make_wavelet(
    n: int,
    p: float,
    typ: str = "sin",
    we: float = np.inf,
    ds: float = 0.0,
    de: float = 0.0,
) -> NDArray:
    """
    Make a wavelet
    n: length of wavelet (samples)
    p: period length of the signal (samples)
    we: 1-sigma width of the gaussian envelope (samples)
    ds: shift of signal to the right (samples)
    de: shift of envelope to the right (samples)
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


def shift(fn: ArrayLike, dt: float, t0: float) -> NDArray:
    """
    Shift a seismogram section in Fourier space

    Parameters
    ----------
    fn : (S, N) :class:`~numpy.array`:
        Seismogram section, with time aligned along axis 1
    dt : float
        Sampling intervall (seconds)
    t0 : (S,) :class:`~numpy.array`:
        Time shifts positive to the right per seismogram (seconds)

    Returns:
    --------
    out : (S, N) :class:`~numpy.array`:
        Shifted seismogram section
    """

    n = fn.shape[-1]
    n2 = n // 2 + 1
    sfn = fft.irfft(
        fft.rfft(fn)
        * np.exp(np.outer(t0, (-1j * np.array(np.arange(n2) * 2 * np.pi / (n * dt))))),
        n,
    )
    return sfn


def xcorrc(x: ArrayLike, y: ArrayLike) -> NDArray:
    """
    Function to compute crosscorrelation coefficient (rather than
    raw correlation from np.correlate.
    """

    z = np.correlate(x.T, y.T, "full")
    z = z / np.sqrt(np.correlate(x, x) * np.correlate(y, y))

    return z


def taper(fn: ArrayLike, nt: float, dt: float, t1: float, t2: float) -> NDArray:
    """
    Taper time series.

    Taper time series `fn` sampled at `dt` with a cosine taper `nt` seconds long
    from begin point `t1 - nt` and with reverse cosine taper from point `t2` to
    point `t2 + nt`. Points inside the range `(t1, t2)` remain unchanged. Points
    outside the range `(t1 - nt, t2 + nt)` are zeroed. If `t1` or `t2` is
    negative then taper is not implemented at beginning or end. If `fn` is an
    array of seismograms, then the taper is applied to the last dimension of
    `fn`.

    Parameters
    ----------
    fn : (..., N) :class:`~numpy.array`:
        Seismogram section, with time aligned along last axis
    nt : float:
        Length of taper (seconds)
    dt : float:
        Sampling intervall (seconds)

    Returns:
    --------
    out : (..., N) :class:`~numpy.array`:
        Tapered seismogram section
    """

    nx = fn.shape[-1]
    taper = np.ones(nx)
    it = np.arange(int(nt / dt + 1)) * dt / nt
    ct = 0.5 * (1 - np.cos(np.pi * it))
    it1 = int(t1 / dt + 1)
    it2 = int(t2 / dt + 1)

    if t1 > 0:
        taper[it1 - int(nt / dt) - 1 : it1] = ct
        taper[0 : it1 - int(nt / dt)] = np.zeros(np.size(np.arange(it1 - int(nt / dt))))
    if t2 > 0:
        if t2 > nx * dt - nt:
            msg = "T2 is outside domain of fn: T2 > NX * DT - NT"
            raise ValueError(msg)
        taper[it2 - 1 : it2 + int(nt / dt)] = np.flip(ct)
        taper[it2 + int(nt / dt) - 1 : nx] = np.zeros(
            np.shape((taper[it2 + int(nt / dt) - 1 : nx]))
        )

    def _apply(arr):
        return arr * taper

    return np.apply_along_axis(_apply, -1, fn)


def demean(fn: ArrayLike) -> NDArray:
    """
    Remove mean value along last dimension from array.

    Parameters
    ----------
    fn : (..., N) :class:`~numpy.array`:

    Returns:
    --------
    out : (..., N) :class:`~numpy.array`:
        De-meaned seismogram section
    """

    def _demean(arr):
        return arr - np.mean(arr)

    return np.apply_along_axis(_demean, -1, fn)


def _pow2(x: int) -> int:
    """Return the next integer that is a power of 2"""
    power = 0
    while power < x:
        power *= 2
    return power


def differentiate(fn: ArrayLike, dt: float) -> NDArray:
    """
    Differentiate a seismogram section in Fourier space

    Parameters
    ----------
    fn : (..., N) :class:`~numpy.array`:
        Seismogram section, with time aligned along last axis
    dt : float
        Sampling intervall (seconds)

    Returns:
    --------
    out : (..., N) :class:`~numpy.array`:
        Differentiated seismogram section
    """

    n = np.shape(fn)[-1]
    n2 = n // 2 + 1
    sfn = fft.irfft(
        fft.rfft(fn) * 1j * np.arange(n2) * 2 * np.pi / (n * dt),
        n,
    )
    return sfn


def concat_component(wvm: ArrayLike) -> NDArray:
    """
    Rearange waveform matrix so that components are concatenated

    Parameters
    ----------
    wvm : (N,C,S) :class:`~numpy.array`:
        Waveform matrix

    Returns:
    --------
    ax : :class:`matplotlib.axes.Axes`
        Axis containing the plot
    """
    ne, nc, ns = np.shape(wvm)  # events, channels, samples

    return wvm.reshape(ne, ns * nc)


def corner_frequency(M: float, sigma: float, vs: float, phase : str="P") -> float: 
    """
    Return a rought estimate of the corner frequency based on

    fc = (7/16 * M0 * sigma * ks**3 * vs**3)**(1/3)

    Parameters
    ----------
    M : float
        Magnitude
    sigma : float
        Stress drop in Pa (use ~5e6)
    vs : float
        S wave velocity in m/s
    phase : str
        P (ks = 0.375) or S (ks = 0.21)

    Returns:
    --------
    fc : float
        Corner frequency in Hz
    """

    ks = 0.375
    if phase == "S":
        ks = 0.21

    M0 = 10 ** (3 / 2 * (M + 10.7)) * 1e-7  # in Nm

    return (sigma * ks**3 * vs**3 * 16 / 7 / M0) ** (1 / 3)


def filter_wvm(
    wvm: ArrayLike,
    dt: float,
    hpas: float | None = None,
    lpas: float | None = None,
    corners: int = 2,
    typ: str = "bessel",
) -> NDArray:
    """
    Filter the waveform matrix. Highpass, lowpass, or bandpass is applied when
    high, low, or both frequency corners are given.

    Parameters
    ----------
    wvm : ArrayLike
        Waveform matrix
    dt : float
        Sampling interval (s)
    hpas, lpas : float
        Highpass and lowpass frequency (Hz)
    corners : int
        Number of filter corners
    typ : str
        "bessel" or "butter" for Bessel or Butterworth filter

    Returns:
    --------
    out : ArrayLike
        Filtered waveform matrix
    """

    nyq = 0.5 / dt
    if typ == "bessel":
        ffun = bessel
    elif typ == "butter":
        ffun = butter
    else:
        msg = f"Unknown filter type: {typ}"
        raise ValueError(msg)

    # Compute filter coefficients
    if hpas is None:
        if lpas is None:
            raise ValueError("Neither lpas nor hpas are specified. Give at least one.")
        else:
            wn = lpas / nyq
            [b, a] = ffun(corners, wn, "lowpass")
    else:
        if lpas is None:
            wn = hpas / nyq
            [b, a] = ffun(corners, wn, "highpass")
        else:
            wn = (lpas / nyq, hpas / nyq)
            [b, a] = ffun(corners, wn, "bandpass")

    return lfilter(b, a, wvm)


def pc_index(scomp):
    """
    Return principal component sorting of seismogram matrix
    """
    U, _, _ = svd(scomp)
    return np.argsort(np.arctan2(U[:, 1], U[:, 0]))
