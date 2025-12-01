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
from typing import Callable, Iterable
from relmt import core, signal

logger = core.register_logger(__name__)


def make_waveform_array(
    header: core.Header,
    phase_dict: dict[str, core.Phase],
    stream,
) -> tuple[np.ndarray, core.Header]:
    """
    Isolate time windows around picks and write to waveform array

    Parameters
    ----------
    header:
        Station and phase meta data
    phase_dict:
        Phase dictionary containing phase arrival times
    stream: :class:`obspy.core.stream.Stream`
        Seismic event waveforms

    Returns
    -------
    arr: :class:`numpy.ndarray`
        Waveform array
    hdr: :class:`relmt.core.Header`
        Updated waveform header

    Attention
    ---------
    Depends on ``obspy``
    """

    try:
        from obspy import Stream
    except ModuleNotFoundError:
        msg = "Could not import obspy.\n"
        msg += "Please install relMT with optional obspy dependencies:\n"
        msg += "pip install relmt[obspy]"
        raise ModuleNotFoundError(msg)

    station = header["station"]
    phase = header["phase"]
    data_window = header["data_window"]
    sampling_rate = header["sampling_rate"]
    ncha = len(header["components"])
    chaset = set(header["components"])

    sta_tr = stream.select(station=station)
    sta_tr.resample(sampling_rate)

    evid_tps = np.array(
        [
            (core.split_phaseid(phid)[0], phase_dict[phid].time)
            for phid in phase_dict
            if core.split_phaseid(phid)[1:] == (station, phase)
        ]
    )

    try:
        evids = evid_tps[:, 0].astype(int)
        tps = evid_tps[:, 1]
    except IndexError as err:
        logger.error("No station / phase match. Is the header set correctly?")
        logger.debug(f"Header['station']: {station}'")
        logger.debug(f"Header['phase']: {phase}'")
        raise err

    # Beginning and end times
    tbs = tps - data_window / 2
    tes = tps + data_window / 2

    # Lookup table: Event ID -> Trace indices (1 per channel, account for
    # missing / duplicate data)
    tr_lut = {evid: [] for evid in evids}
    for evid, tb, te in zip(evids, tbs, tes):
        for ntr, tr in enumerate(sta_tr):
            # Start and endtime are within trace
            if tb >= tr.stats.starttime.timestamp and te <= tr.stats.endtime.timestamp:
                tr_lut[evid].append(ntr)

    logger.debug(
        f"Created trace lookup table with {sum(len(v) for v in tr_lut.values())} entries."
    )

    # Only consider events with complete channel information
    evgood = sorted(
        [
            evid
            for evid, itrs in tr_lut.items()
            if chaset.issubset(set(sta_tr[itr].stats.component for itr in itrs))
        ]
    )

    logger.debug(f"Found {len(evgood)} matching events.")

    # Initialize output arrays
    nsamp = int(data_window * sampling_rate)
    wvarr = np.zeros((len(evgood), ncha, nsamp))

    # Iterate over event IDs
    for iev, evid in enumerate(evgood):
        phid = core.join_phaseid(evid, station, phase)
        logger.debug(f"Working on phase: {phid}")

        # Pick time
        tp = phase_dict[phid].time

        # Start time and index
        tb = tp - data_window / 2

        # Select traces for event
        wvtrs = Stream([sta_tr[itr] for itr in tr_lut[evid]])

        # Iterate over seismogram components
        for ic, cha in enumerate(header["components"]):
            logger.debug(f"Working on component {ic} ({tr.stats.channel})")
            try:
                tr = wvtrs.select(component=cha)[0]
            except IndexError as err:
                logger.error(f"Missing component {cha} for phase: {phid}")
                raise err

            ib = np.argmin(abs(tr.times("timestamp") - tb))
            ie = ib + nsamp

            data = tr.data[ib:ie]
            wvarr[iev, ic, :] = data - data.mean()

    header["events_"] = evgood

    return wvarr, header


def read_obspy_inventory_files(filenames: Iterable[str]):
    """
    Read station files using :func:`obspy.core.inventory.inventory.read_inventory`

    Parameters
    ----------
    filenames:
        Names of the compliant inventroy files

    Returns
    -------
    :class:`obspy.core.inventory.inventory.Inventory`

    Attention
    ---------
    Depends on ``obspy``
    """

    try:
        from obspy import Inventory, read_inventory
    except ModuleNotFoundError:
        msg = "Could not import obspy.\n"
        msg += "Please install relMT with optional obspy dependencies:\n"
        msg += "pip install relmt[obspy]"
        raise ModuleNotFoundError(msg)

    inv = Inventory()
    for fn in filenames:
        inv += read_inventory(fn)

    return inv


def get_utm_zone(
    latitudes: Iterable[float], longitudes: Iterable[float]
) -> tuple[int, str]:
    """
    Get UTM zone number and letter from coordinates.

    Parameters
    ----------
    latitudes, longitudes:
        Geographical coordinates to include in estimate

    Returns
    -------
    num: int
        Consensus UTM zone number
    let: str
        Consensus UTM zone letter

    Raises
    ------
    LookupError:
        If coordinates exceed one zone.

    Attention
    ---------
    Depends on ``utm``
    """

    try:
        import utm
    except ModuleNotFoundError:
        msg = "Could not import utm.\n"
        msg += "Please install relMT with optional obspy dependencies:\n"
        msg += "pip install relmt[geo]"
        raise ModuleNotFoundError(msg)

    # Get possible zone letters
    lets = [utm.latitude_to_zone_letter(lat) for lat in latitudes]
    nums = [
        utm.latlon_to_zone_number(lat, lon) for lat, lon in zip(latitudes, longitudes)
    ]
    zones = set([f"{num}{let}" for num, let in zip(nums, lets)])

    if len(zones) > 1:
        msg = (
            "No UTM zone consensus. Choices are: "
            + ", ".join([i for i in zones])
            + ". Choose one or narrow down coordinates."
        )
        raise LookupError(msg)

    return nums[0], lets[0]


def geoconverter_utm2latlon(
    northing: float, easting: float, depth: float, utm_num: int, utm_let: str
) -> tuple[float, float, float]:
    """
    Convert geographic to local UTM coordinates

    Parameters
    ----------
    northing, easting:
        Cartesian coordinates (meters)
    depth: float
        Positive down in meters
    utm_num:
        UTM zone number
    utm_let:
        UTM zone letter

    Return
    ------
    latitude, longitude, depth : float
        Geographical coordinates in degree and kilometers

    Attention
    ---------
    Depends on ``utm``
    """
    try:
        import utm
    except ModuleNotFoundError:
        msg = "Could not import utm.\n"
        msg += "Please install relMT with optional geo dependencies:\n"
        msg += "pip install relmt[geo]"
        raise ModuleNotFoundError(msg)

    latitude, longitude = utm.to_latlon(
        easting, northing, utm_num, utm_let, strict=False
    )

    depth /= 1000

    return latitude, longitude, depth


def geoconverter_latlon2utm(
    latitude: float, longitude: float, depth: float, utm_num: int, utm_let: str
) -> tuple[float, float, float]:
    """
    Convert geographic to local UTM coordinates

    Parameters
    ----------
    latitude, longitude:
        Geographic coordinates (degree)
    depth:
        Positive down in kilometers
    utm_num:
        UTM zone number
    utm_let:
        UTM zone letter

    Return
    ------
    north, east, depth : float
        Local UTM coordinates in meters

    Hint
    ----
    Wrap into a custom function to use as a geoconverter when reading station
    or event tables

    .. code-block::

        def geconverter_oslo(lat, lon, dep):
            return geoconverter_latlon2utm(lat, lon, dep, 32, "V")

    Attention
    ---------
    Depends on ``utm``
    """
    try:
        import utm
    except ModuleNotFoundError:
        msg = "Could not import utm.\n"
        msg += "Please install relMT with optional 'geo' dependencies:\n"
        msg += "pip install relmt[geo]"
        raise ModuleNotFoundError(msg)

    east, north, _, _ = utm.from_latlon(
        latitude, longitude, force_zone_number=utm_num, force_zone_letter=utm_let
    )

    depth *= 1000

    return north, east, depth


def read_station_inventory(
    inventory, geoconverter: Callable, strict: bool = True
) -> dict[str, core.Station]:
    """
    Parameters
    ----------
    inventory: :class:`obspy.core.inventory.inventory.Inventory`
        Must contain network and station codes, and coordinates.
    geoconverter:
        Function that takes longitude, latitude and depth as arguments and returns local
        northing and easting and depth coordinates in meters. Depth conversion unused,
        solely required for compatibility with geoconverter argument in
        :func:`relmt.io.read_event_table`.
    strict:
        Raise a KeyError for repeated station codes with unequal coordinates. If
        False, last occurrence overwrites previeous occurrences

    Returns
    -------
    Station dictionary

    Raises
    ------
    ValueError:
        When station code contains the reserved character '_'
    KeyError:
        When repeated station codes are met and `stict=True`

    Attention
    ---------
    Depends on ``obspy``
    """

    station_dict = dict()
    for net in inventory:
        for sta in net:
            north, east, _ = geoconverter(sta.latitude, sta.longitude, 0)
            depth = -sta.elevation

            # One entry for standard, altanate and historical station code each
            for scode in [sta.code, sta.alternate_code, sta.historical_code]:
                if scode:
                    if "_" in scode:
                        msg = f"'_' character in station code {scode} is reserved."
                        raise ValueError(msg)
                    if scode in station_dict and strict:
                        n, e, d, _ = station_dict[scode]
                        if n != north or e != east or d != depth:
                            msg = f"Station code {scode} already contained."
                            raise KeyError(msg)
                    station_dict.update(
                        {scode: core.Station(north, east, depth, scode)}
                    )

    return station_dict


def spectrum(
    sig: np.ndarray, sampling_rate: float, nfft: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrun the spectrum of a 1D signal

    Parameters
    ----------
    sig:
        ``(samples,)`` array holding the waveform
    sampling_rate:
        in Hertz
    nfft:
        Number of points in the fft

    Returns
    -------
    Frequency vector (Hz) and corresponding amplitudes (input units)

    Attention
    ---------
    Depends on ``multitaper``
    """
    try:
        from multitaper import MTSpec
    except ModuleNotFoundError:
        msg = "Could not import multitaper.\n"
        msg += "Please install relMT with optional multitaper dependencies:\n"
        msg += "pip install relmt[multitaper]"
        raise ModuleNotFoundError(msg)

    tw = sig.shape[-1] / sampling_rate
    freq, spec = MTSpec(sig, dt=1 / sampling_rate, nfft=nfft).rspec()
    spec = np.sqrt(spec * tw)
    return freq, spec


def apparent_corner_frequency(
    sig: np.ndarray,
    sampling_rate: float,
    fmin: float | None = None,
    fmax: float | None = None,
) -> float:
    """
    Find apparant corner frequency in velocity spectrum of 1D seismogram

    Parameters
    ----------
    sig:
        ``(samples,)`` seismogram
    sampling_rate:
        in Hertz
    fmin, fmax:
        Minimum, maximum frequency to test (Hz)


    Returns
    -------
    Maximum of the spectrum (Hz)

    Attention
    ---------
    Depends on ``multitaper``
    """

    if sig.ndim == 1:
        freqs, specs = spectrum(sig, sampling_rate)
    else:
        # Compute average spectrum of components
        freqs, _ = spectrum(sig[0, :], sampling_rate)
        specs = np.mean(
            [spectrum(sig[ncha, :], sampling_rate)[1] for ncha in range(sig.shape[0])],
            axis=0,
        )

    if0 = 0
    if1 = len(freqs)

    if fmin is not None:
        if0 = np.argmin(abs(freqs - fmin))
    if fmax is not None:
        if1 = np.argmin(abs(freqs - fmax))

    return freqs[if0:if1][np.argmax(specs[if0:if1])][0]


#@core._doc_config_args
def optimal_bandpass(
    wvarr: np.ndarray,
    sampling_rate: float,
    data_window: float,
    phase_start: float,
    phase_end: float,
    fmin: float | None = None,
    fmax: float | None = None,
    min_snr: float = 0,
) -> tuple[float, float] | tuple[None, None]:
    """Find bandpass with signal-to-noise ratio above threshold

    Parameters
    ----------
    wvarr:
        1D or 2D array holding the velocity event waveform(s)
    fmin, fmax:
        Minimum, maximum frequency to consider
    min_snr:
        Minimum signal to noise ratio (dB) to include

    Returns
    -------
    highpass, lowpass: float
        Optimal filter corners (Hz)

    Attention
    ---------
    Depends on ``multitaper``
    """

    # Get signal and nois windows
    isig, _ = signal.indices_signal(sampling_rate, phase_start, phase_end, data_window)

    # Convert dB to fraction
    min_snr = signal.fraction(min_snr)

    # Force noise spectrum same size as frequence spectrum
    if wvarr.ndim == 1:
        # Preprocess
        sig = signal.demean(wvarr[isig:])
        noi = signal.demean(wvarr[:isig])

        # Find frequency range of signal
        freqs, specs = spectrum(sig, sampling_rate)
        _, specn = spectrum(noi, sampling_rate, nfft=2 * len(freqs) - 1)
    else:
        sig = signal.demean(wvarr[:, isig:])
        noi = signal.demean(wvarr[:, :isig])

        # Compute average spectrum of components
        freqs, _ = spectrum(sig[0, :], sampling_rate)
        specs = np.mean(
            [
                spectrum(sig[ncha, :], sampling_rate, nfft=2 * len(freqs) - 1)[1]
                for ncha in range(sig.shape[0])
            ],
            axis=0,
        )
        specn = np.mean(
            [
                spectrum(noi[ncha, :], sampling_rate, nfft=2 * len(freqs) - 1)[1]
                for ncha in range(noi.shape[0])
            ],
            axis=0,
        )

    # If no frequency bracket is given, use spectrums range
    if fmin is None:
        fmin = np.min(freqs)
    if fmax is None:
        fmax = np.max(freqs)

    # Find sampled frequencies
    if0 = np.argmin(abs(freqs - fmin))
    if1 = np.argmin(abs(freqs - fmax))

    nfreq = if1 - if0

    if nfreq < 1:
        logger.warning("No band below threshold. Returning NaN.")
        return np.nan, np.nan

    # Find longest band above SNR threshold
    trialbands = np.zeros((nfreq, nfreq))
    dpower = np.log10(specs[if0:if1]) - np.log10(specn[if0:if1]) - np.log10(min_snr)
    sumdp = np.cumsum(dpower)
    for ifreq in range(nfreq):
        trialbands[ifreq, ifreq:] = sumdp[ifreq:] - sumdp[ifreq]

    # Passband that maximizes SNR
    jf0, jf1 = np.unravel_index(np.argmax(trialbands), (nfreq, nfreq))

    hpas = freqs[if0 + jf0][0]
    lpas = freqs[if0 + jf1][0]

    return hpas, lpas


def focal_mechanism_to_mt(
    strike: float | list[float],
    dip: float | list[float],
    rake: float | list[float],
    magnitude: float | list[float] | None = None,
) -> core.MT | list[core.MT]:
    """Convert focal mechanism to moment tensor"""
    try:
        from pyrocko.moment_tensor import MomentTensor
    except ModuleNotFoundError:
        raise ModuleNotFoundError(core._module_hint("pyrocko"))

    try:
        momt = core.MT(
            *MomentTensor(strike=strike, dip=dip, rake=rake, magnitude=magnitude).m6()
        )
    except TypeError:
        # Make we iterate over all elements
        if magnitude is None:
            magnitude = [None] * len(strike)
        assert len(strike) == len(rake) == len(dip) == len(magnitude)
        momt = [
            core.MT(*MomentTensor(strike=s, dip=d, rake=r, magnitude=m).m6())
            for s, d, r, m in zip(strike, dip, rake, magnitude)
        ]

    return momt
