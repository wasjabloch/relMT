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
from datetime import datetime
import scipy.fft as fft
import logging
from typing import Iterable

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


def join_phid(event_index: int, station: str, phase: str) -> str:
    """Join event, station and phase names to phase identifier"""
    return f"{event_index}_{station}_{phase}"


def split_phid(phid: str) -> tuple[int, str, str]:
    """Split phase identifier into eventID, station and phase"""
    event_index, station, phase = phid.split("_")
    return int(event_index), station, phase


def join_waveid(station: str, phase: str) -> str:
    """Join station and phase names to observation identifier"""
    return f"{station}_{phase}"


def split_waveid(waveid: str) -> tuple[str, str]:
    """Split observation identifier into station and phase"""
    station, phase = waveid.split("_")
    return station, phase


def xyzarray(iterable: dict | list) -> NDArray:
    """Return spatial coordinates from event list or station dictionary"""
    try:
        return np.array([iterable[key][:3] for key in sorted(iterable)])
    except TypeError:
        return np.array([item[:3] for item in iterable])


def cartesian_distance(
    x1: ArrayLike,
    y1: ArrayLike,
    z1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    z2: ArrayLike,
) -> NDArray:
    """Cartesian distance between points (x1, y1, z1) and (x2, y2, z2)"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def azimuth(x1, y1, x2, y2):
    """Azimuth from point (x1, y1) to (x2, y2)"""
    return np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi


def approx_time_lookup(
    time1: str | float, time2: str | float, include_at=np.inf
) -> dict:
    """
    Create lookup table for time1s approximatley matching time2s

    All events in time1s must be present in time2s, i.e. len(time1s) <= len(time2s).

    Parameters
    ----------
    times1, times2: (float or str)
        Times either as understood by datetime.isoformat, or as float in seconds
    include_at: (float)
        Maximum time difference (seconds) to consider a match.:w


    Returns
    -------
    lut: (dict)
        Lookup table time1s -> time2s

    Raises
    ------
    LookupError: when time1s is longer than time2s
    """

    logger.info("Creating approximate time lookup table")

    if len(time1) > len(time2):
        raise LookupError("times1 must be shorter or equal than times2.")

    time1 = sorted(time1)
    time2 = sorted(time2)

    try:
        t1s = list(map(datetime.fromisoformat, time1))
        t2s = list(map(datetime.fromisoformat, time2))
        istime = True
    except (ValueError, TypeError):
        t1s = list(map(float, time1))
        t2s = list(map(float, time2))
        istime = False

    logger.info(f"Interpreting input as: {type(t1s[0])}")

    lut = {}
    n0 = 0
    for ot1, t1 in zip(time1, t1s):

        dtmin = np.inf
        for n, (ot2, t2) in enumerate(zip(time2[n0:], t2s[n0:])):
            dt = abs(t1 - t2)
            if istime:
                dt = abs(t1 - t2).total_seconds()

            if dt < dtmin:
                dtmin = dt
                thisot2 = ot2
            else:
                break

        if dtmin <= include_at:
            lut[ot1] = thisot2
        else:
            msg = (
                "Time difference between events is: {:.0f} seconds\n"
                "Event 1: {:}\n"
                "Event 2: {:}\n"
                "Did not include event to lookup table!"
            ).format(dtmin, ot1, ot2)
            logger.warning(msg)
        n0 = n0 + n

    return lut


def interpolate_phase_dict(
    phase_dict: dict, event_dict: dict, station_dict: dict, obs_min: int = 1
) -> dict:
    """
    Interpolate phase arrivals, ray azimuth and inclination

    Arrival time is interpolated via an average velocity along the ray path.

    Ray inclination is interpolated linearly between neigboring points in depth

    Azimuth is calculated assuming a horizontally straight ray.

    Parameters
    ----------
    phase_dict, event_dict, station_dict: (dict)
        Phase, event and station dictionaries
    obs_min: (int)
        Minimum number of observations to make interpolation

    Returns
    -------
    nphd: (dict)
        New phase dictionary containing the interpolated phases
    """

    _, stas, phs = zip(*map(split_phid, phase_dict.keys()))

    # new phase dictionary
    nphd = {}
    for ph in set(phs):
        for st in set(stas):
            logger.debug(f"Working on phase {ph}, station {st}")
            try:
                sxyz = station_dict[st][:3]
            except KeyError:
                logger.warning(f"Station {st} not in station_dict. Skipping.")
                continue

            # For all events with phase readings on station, get:
            # Travel times, distances, azimuths, inclination and depth
            tdaiz = np.array(
                [
                    (
                        phase_dict[pid][0]
                        - event_dict[event_index][3],  # arrival - origin time
                        cartesian_distance(
                            *sxyz, *event_dict[event_index][:3]
                        ),  # distance
                        phase_dict[pid][1],  # azimuth
                        phase_dict[pid][2],  # inclination
                        event_dict[event_index][2],  # event z-coordinate
                    )
                    for pid in phase_dict
                    for event_index in event_dict
                    if event_index == split_phid(pid)[0]  # event match
                    and st == split_phid(pid)[1]  # station match
                    and ph == split_phid(pid)[2]  # phase match
                ]
            )

            n = tdaiz.shape[0]
            if n < obs_min:
                # Too few readings
                logger.warning(
                    f"Station {st} has only {n} {ph}-wave readings. obs_min is {obs_min}. Skipping"
                )
                continue

            tt = tdaiz[:, 0]
            d = tdaiz[:, 1]
            logger.debug(
                "Travel times are: "
                + ", ".join(["{:.1e}".format(t) for t in tt])
                + " sec"
            )
            logger.debug(
                "Distances are: "
                + ", ".join(["{:.1e}".format(dd) for dd in d])
                + " meters"
            )

            # Average path velocity
            v = np.average(d / tt)
            logger.info(
                f"{ph}-wave velocity along raypath to station {st} is "
                + "{:.0f} +/- {:.0f} m/s".format(v, np.std(d / tt))
            )

            i, z = tdaiz[:, 3:5].T
            isort = np.lexsort((i, z))  # sort by depth

            # Now look for missing events
            for event_index in event_dict:
                thisphid = join_phid(event_index, st, ph)
                if thisphid not in phase_dict:
                    ex, ey, ez, t0 = event_dict[event_index][:4]
                    td = cartesian_distance(ex, ey, ez, *sxyz)  # distance
                    t = t0 + td / v  # arrival time
                    ei = np.interp(ez, z[isort], i[isort])  # ray inclination
                    ea = azimuth(ex, ey, sxyz[0], sxyz[1])  # ray azimuth
                    nphd[thisphid] = (t, ea, ei)
    return nphd


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


def norm_power(array: ArrayLike, axis: int = -1) -> NDArray:
    """
    Normalize input array by power

    out = array / square_root(sum(array**2)))

    Parameters
    ----------
    funct : :class:`~numpy.array`:
        Seismogram section

    Returns
    -------
    out : :class:`~numpy.array`:
        Normalized seismogram section
    """
    return array / np.sqrt(np.sum(array**2, axis=axis, keepdims=True))


def nonzero_events(array: ArrayLike) -> NDArray:
    """Return indices of events with non-zero data"""
    return np.asarray(sorted(set(np.transpose(array.nonzero())[:, 0])))


def shift_3d(wvf_array: ArrayLike, dt: float, t0: float) -> NDArray:
    """
    Shift a waveform matrix in Fourier space

    Parameters
    ----------
    wvf_array : (E, C, S) :class:`~numpy.array`:
        Waveform array, with time aligned along last axis
    dt : float
        Sampling intervall (seconds)
    t0 : (S,) :class:`~numpy.array`:
        Time shifts positive to the right per seismogram (seconds)

    Returns
    -------
    out : (E, C, S) :class:`~numpy.array`:
        Shifted seismogram section
    """

    out = np.zeros(wvf_array.shape)
    for ic in range(wvf_array.shape[1]):
        out[:, ic, :] = shift(wvf_array[:, ic, :], dt, t0)
    return out


def shift(wvf_matrix: ArrayLike, dt: float, t0: float) -> NDArray:
    """
    Shift a seismogram section in Fourier space

    Parameters
    ----------
    wvf_matrix : (S, N) :class:`~numpy.array`:
        Seismogram section, with time aligned along axis 1
    dt : float
        Sampling intervall (seconds)
    t0 : (S,) :class:`~numpy.array`:
        Time shifts positive to the right per seismogram (seconds)

    Returns
    -------
    out : (S, N) :class:`~numpy.array`:
        Shifted seismogram section
    """

    n = wvf_matrix.shape[-1]
    n2 = n // 2 + 1
    sfn = fft.irfft(
        fft.rfft(wvf_matrix)
        * np.exp(np.outer(t0, (-1j * np.array(np.arange(n2) * 2 * np.pi / (n * dt))))),
        n,
    )
    return sfn


def _xcorrc(x: ArrayLike, y: ArrayLike) -> NDArray:
    """
    Function to compute crosscorrelation coefficient (rather than
    raw correlation from np.correlate.

    ..note:
    Currently unused
    """

    z = np.correlate(x.T, y.T, "full")
    z = z / np.sqrt(np.correlate(x, x) * np.correlate(y, y))

    return z


def cc_coef(x, y):
    """
    Calculate the correlation coefficient between two equal-length vectors using the formula:

    cc = sum(x * y) / sqrt(sum(x**2) * sum(y**2)).

    Parameters
    ----------
    x, y : (S,) :class:`~numpy.array`:
        input vectors of same length

    Returns:
    --------
    out : (float)
        Correlation coefficient between x and y.

    .. notes:
        The correlation coefficient is a measure of linear association between two variables.
    """
    return sum(x * y) / np.sqrt(sum(x**2) * sum(y**2))


def cosine_taper(
    wvf_array: ArrayLike, length: float, dt: float, t1: float, t2: float
) -> NDArray:
    """
    Taper time series along last axis.

    Taper time series in last dimension of `wvf_array` sampled at `dt` with a
    cosine taper `length` seconds long from begin point `t1 - length/2` and with
    reverse cosine taper from point `t2` to point `t2 + length/2`. Points inside
    the range `(t1, t2)` remain unchanged. Points outside the range `(t1 -
    length/2, t2 + length/2)` are zeroed. If `t1` or `t2` is negative then taper
    is not implemented at beginning or end.

    Parameters
    ----------
    wvf_array: (..., N) :class:`~numpy.array`:
        Seismogram section, with time aligned along last axis
    length : float
        Combined length of taper at both ends(seconds)
    dt : float
        Sampling intervall (seconds)
    t1, t2 : float
        Start, end time of untapered segment from beginning of trace (seconds)

    Returns
    -------
    out : (..., N) :class:`~numpy.array`:
        Tapered seismogram section

    Raises
    ------
    ValueError: If t2 + length/2 is longer than the time series
    """

    nx = wvf_array.shape[-1]
    taper = np.ones(nx)
    tt = length / 2  # Taper time
    nt = int(tt / dt)  # Numper of taper indices
    it = np.arange(nt + 1) * dt / tt
    ct = 0.5 * (1 - np.cos(np.pi * it))
    it1 = int(t1 / dt + 1)
    it2 = int(t2 / dt + 1)

    if t1 > 0:
        taper[it1 - nt - 1 : it1] = ct
        taper[0 : it1 - nt] = 0
    if t2 > 0:
        if t2 > nx * dt - tt:
            msg = f"t2 ({t2}) is outside domain of funct: NX * DT - LENGTH/2 ({nx} * {dt} - {tt})"
            raise ValueError(msg)
        taper[it2 - 1 : it2 + nt] = np.flip(ct)
        taper[it2 + nt - 1 : nx] = 0
    def _apply(arr):
        return arr * taper

    return np.apply_along_axis(_apply, -1, wvf_array)


def demean(wvf_array: ArrayLike) -> NDArray:
    """
    Remove mean value along last dimension from array.

    Parameters
    ----------
    wvf_array : (..., N) :class:`~numpy.array`:

    Returns:
    --------
    out : (..., N) :class:`~numpy.array`:
        De-meaned seismogram section
    """

    def _demean(arr):
        return arr - np.mean(arr)

    return np.apply_along_axis(_demean, -1, wvf_array)


def zero_events(wvf_array: ArrayLike, event_indices: Iterable[int]) -> NDArray:
    """Set seismograms of event_indices to zero"""
    zeros = np.zeros(wvf_array.shape[-1])
    wvf_array[event_indices, ..., :] = zeros
    return wvf_array


def _pow2(x: int) -> int:
    """Return the next integer that is a power of 2"""
    power = 0
    while power < x:
        power *= 2
    return power


def differentiate(funct: ArrayLike, dt: float) -> NDArray:
    """
    Differentiate a seismogram section in Fourier space

    Parameters
    ----------
    funct : (..., N) :class:`~numpy.array`:
        Seismogram section, with time aligned along last axis
    dt : float
        Sampling intervall (seconds)

    Returns:
    --------
    out : (..., N) :class:`~numpy.array`:
        Differentiated seismogram section
    """

    n = np.shape(funct)[-1]
    n2 = n // 2 + 1
    sfn = fft.irfft(
        fft.rfft(funct) * 1j * np.arange(n2) * 2 * np.pi / (n * dt),
        n,
    )
    return sfn


def concat_wvf_array(wvf_array: ArrayLike) -> NDArray:
    """
    Rearange waveform array so that components are concatenated

    Parameters
    ----------
    wvf_array : (N, C, S) :class:`~numpy.array`:
        Waveform array

    Returns:
    --------
    wvf_matrix : (N, C * S) :class:`~numpy.array`:
        Waveform matrix
    """
    ne, nc, ns = np.shape(wvf_array)  # events, channels, samples
    return wvf_array.reshape(ne, ns * nc)  # events, channels * samples


def corner_frequency(M: float, sigma: float, vs: float, phase: str = "P") -> float:
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


def filter_wvf(
    wvf: ArrayLike,
    dt: float,
    hpas: float | None = None,
    lpas: float | None = None,
    corners: int = 2,
    typ: str = "bessel",
) -> NDArray:
    """
    Filter the waveform array or matrix along last dimension. Highpass, lowpass,
    or bandpass is applied when high, low, or both frequency corners are given.

    Parameters
    ----------
    wvf : ArrayLike
        Waveform array or matrix
    dt : float
        Sampling interval (s)
    hpas, lpas : float
        Highpass and lowpass filter corners (Hz)
    corners : int
        Number of filter corners
    typ : str
        "bessel" or "butter" for Bessel or Butterworth filter

    Returns:
    --------
    out : ArrayLike
        Filtered waveform matrix

    Raises
    ------
    ValueError: If neither hpas, nor lpas are given
    ValueError: If given unknown filter type
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
            wn = (hpas / nyq, lpas / nyq)
            [b, a] = ffun(corners, wn, "bandpass")

    return lfilter(b, a, wvf)


def pc_index(wvf_matrix: ArrayLike, phase: str) -> Iterable[int]:
    """
    Return principal component sorting of seismogram matrix

    For P-phases, sort by zero-th left hand singular vector
    For S-phases, sort angle in the plane spanned by the zero-th and fisrt left
    hand singular vectors

    Parameters
    ----------
    wvf_matrix: np.array(E, S*C)
        Waveform matrix
    phase : str
        P or S phase

    Returns:
    --------
    out : ArrayLike
        Filtered waveform matrix

    Raises
    ------
    ValueError: If phase is not P or S
    """
    U, _, _ = svd(wvf_matrix)
    if phase == "P":
        return np.argsort(U[:, 0])
    elif phase == "S":
        return np.argsort(np.arctan2(U[:, 1], U[:, 0]))
    else:
        raise ValueError(f"Unknown phase: {phase}")
