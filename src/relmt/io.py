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

import obspy
from obspy import Inventory, UTCDateTime, Stream
import utm
from numpy.typing import NDArray
import numpy as np
import logging
from relmt import utils
from relmt import core
import yaml

from collections.abc import Iterable, Callable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(utils.logsh)


def read_obspy_inventory_files(filenames: Iterable[str]) -> Inventory:
    """Read station files using obspy's read_inventory"""
    inv = Inventory()
    for fn in filenames:
        inv += obspy.read_inventory(fn)
    return inv


def read_phase_nll_hypfile(
    filename: str,
    event_index: str,
    substract_residual: bool,
    substract_stationterm: bool,
):
    """
    Read arrival times and take-off angles from NonLinLoc .hyp file.

    Arrival times can be corrected for location residual and station term.

    Ray take-off azimuth is output in degree east of North (X-direction)

    Ray take-off inclination is output degree down from horizontal (in positive
    Z-direction)

    Parameters
    ----------
    filename : str
        Name of the NonLinLoc hypocenter (.hyp) file
    event_index : str
        Event index of this set of phase observations. Must correspond to event
        index in event dictionary.
    substract_residual, substract_stationterm: (boolean)
        Whether to substract localization residual or station term from the
        arrival time

    Returns
    -------
    phase_dict: dict
        Lookup table phase ID -> (Arrivaltime, azimuth, inclination)
    """
    with open(filename, "r") as fid:
        lines = fid.readlines()

    i1, i2 = (None, None)
    for i, line in enumerate(lines):
        if line.startswith("PHASE "):
            i1 = i
        elif line.startswith("END_PHASE"):
            i2 = i

    if i1 is None:
        msg = "'PHASE' line prefix is missing."
        raise RuntimeError(msg)

    if i2 is None:
        msg = "'END_PHASE' line prefix is missing."
        raise RuntimeError(msg)

    phlines = lines[i1 + 1 : i2]

    # go through all phase info lines
    outd = dict()
    for line in phlines:
        line = line.split()
        date, hourmin, sec = map(str, line[6:9])

        try:
            t = UTCDateTime.strptime(date + hourmin, "%Y%m%d%H%M") + float(sec)
        except ValueError:
            msg = f"Could not convert pick time to UTCDateTime: {date + hourmin}. Skipping."
            logger.warning(msg)
            continue

        if substract_residual:
            t -= float(line[16])
        if substract_stationterm:
            t -= float(line[26])

        azi = float(line[23])
        inc = 90 - float(line[24])  # NLL has 0=down

        sta = str(line[0])
        pha = str(line[4])

        phid = utils.join_phid(event_index, sta, pha)
        outd[phid] = (t.timestamp, azi, inc)

    return outd


def get_utm_zone(
    latitudes: Iterable[float], longitudes: Iterable[float]
) -> tuple[int, str]:
    """
    Get UTM zone number and letter from coordinates.

    Parameters
    ----------
    latitudes, longitudes : Iterable[float]
        geographical coordinates to include in estimate

    Returns
    -------
    num: int
        Consensus UTM zone number
    let: str
        Consensus UTM zone letter

    Raises
    ------
    LookupError if coordinates exceed one zone.
    """

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


def geoconverter_latlon2utm(latitude, longitude, depth, utm_num, utm_let):
    """
    Convert geographic to local UTM coordinates

    Parameters
    ----------
    latitude, longitude : float
        Geographic coordiantes (degree)
    depth : float
        Positive down in kilometers
    utm_num, utm_let: int, str
        UTM zone number and letter

    Return
    ------
    north, east, depth : float
        Local UTM coordinates in meters

    ..hint:
        Wrap into a custom function to use as a geoconverter when reading station
        or event tables
        ```
        def geconverter_oslo(lat, lon, dep):
            return geoconverter_latlon2utm(lat, lon, dep, 32, "V")
        ```
    """

    east, north, _, _ = utm.from_latlon(
        latitude, longitude, force_zone_number=utm_num, force_zone_letter=utm_let
    )

    depth *= 1000

    return north, east, depth


def read_station_inventory(
    inventory: Inventory, geoconverter: Callable, strict=True
) -> dict[str : core.Coordinate]:
    """
    Parameters
    ----------
    inventory : obspy.Inventory
        Must contain network and station codes, and coordinates.
    geoconverter : callable
        Function that takes longitude, latitude and depth as arguments and returns local
        northing and easting and depth coordinates in meters. Depth conversion unused,
        solely required for compatibility with geoconverter argument in
        read_event_table().
    strict : (bool)
        Raise a KeyError for repeated station codes with unequal coordinates. If
        False, last occurence overwrites previeous ocurrences

    Returns
    -------
    station_dict: dict
        Station dictionary Code -> Northing, Easting, Depth

    Raises
    ------
        ValueError when station code contains the reserved character '_'
        KeyError when repeated station codes are met and `stict` is True
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
                        n, e, d = station_dict[scode]
                        if n != north or e != east or d != depth:
                            msg = f"Station code {scode} already contained."
                            raise KeyError(msg)
                    station_dict.update({scode: core.Coordinate(north, east, depth)})

    return station_dict


def make_station_table(station_dict: dict) -> str:
    """
    Convert station dictionary to relMT compliant station table

    The resulting table has one line per station. Columns are:

    Station Northing Easting Depth
    (code)   (meter) (meter) (meter)

    Parameters
    ----------
    station_dict: dict
        Station dictionary Code -> Northing, Easting, Depth

    Returns
    -------
    out : str
        Tabled station data
    """

    # Header
    out = "#Station   Northing    Easting     Depth\n"
    out += "# (code)    (meter)    (meter)   (meter) \n"
    form = "{:>8s} {:>10.1f} {:>10.1f} {:>9.1f}\n"

    for code, (north, east, depth) in station_dict.items():
        out += form.format(code, north, east, depth)

    return out


def read_station_table(filename: str) -> dict:
    """
    Read a relMT station table into dictionary structrue

    Parameters
    ----------
    filename : str
        Name of the station table file

    Returns
    -------
    station_dict: dict
        Station code -> norting, easting, depth

    """
    code = np.loadtxt(filename, usecols=(0), unpack=True, dtype=str)
    (
        north,
        east,
        depth,
    ) = np.loadtxt(filename, usecols=(1, 2, 3), unpack=True, dtype=float)
    return {c: core.Coordinate(n, e, d) for c, n, e, d in zip(code, north, east, depth)}


def make_event_table(event_list: list[core.Event]) -> str:
    """
    Convert event dictionary to relMT compliant event table.

    Number Northing Easting   Depth      Time Magnitude       Name
     (int)  (meter) (meter) (meter) (seconds)       (-)      (str)

    Parameters
    ----------
    event_dict: dict
        Event dictionary
        EventIndex -> Northing, Easting, Depth, Originime, Magnitude, Identifier

    Returns
    -------
    out : str
        Tabled event data
    """

    # Header
    out = "#Number   Northing    Easting     Depth         Origintime Magnitude"
    out += "             Name\n"
    out += "# (int)    (meter)    (meter)   (meter)          (seconds)       (-)"
    out += "            (str)\n"

    form = "{:>7d} {:>10.1f} {:>10.1f} {:>9.1f} {: 12.6f} {:>9.4f} {:>16s}\n"

    for iev, (north, east, depth, time, mag, name) in enumerate(event_list):
        out += form.format(iev, north, east, depth, time, mag, name)

    return out


def read_ext_event_table(
    filename: str,
    north_index: int,
    east_index: int,
    depth_index: int,
    time_index: int,
    magnitude_index: int,
    evid_index: int,
    geoconverter: Callable | None = None,
    timeconverter: Callable | None = None,
    nameconverter: Callable | None = None,
    loadtxt_kwargs: dict = {},
) -> dict[int : core.Event]:
    """
    Read an external event table into an event dictionary.

    Parameters
    ----------
    filename : str
        Name of the event table file
    north_index, east_index, depth_index, time_index, magnitude_index, evid_index : int
        Column indices of the northing, easting, depth, time, magnitude, eventID
    geoconverter : callable or None
        Function that takes north, east and depth as arguments and returns local
        northing and easting and depth coordinates in meters (e.g. interpret
        north and east as latitude and longitude, and return UTM northing and
        easting; or convert kilometer depth to meter)
    timeconverter : callable
        Function that takes time string as agrument and returns time in seconds
        as a float (e.g. epoch timestamp. Must be consistent with the reference
        frame of the waveforms)
    nameconverter : callable or None
        Function that takes a string as argument and returns a user defined
        event name string.
    loadtxt_kwargs: dict
        Additional keyword arguments are passed on to `numpy.loadtxt`

    Returns
    -------
    event_dict: dict
        Event dictionary event index -> norting, easting, depth, time, magnitude, eventID

    Raises
    ------
        KeyError: if 'usecol', 'unpack', or 'dtype' are specified as `loadtxt_kwargs`
    """

    kwargs = ["usecol", "unpack", "dtype"]

    if any([kwarg in loadtxt_kwargs for kwarg in kwargs]):
        msg = f"loadtxt_kwargs: {', '.join(kwargs)} are reserved."
        raise KeyError(msg)

    name, time = np.loadtxt(
        filename,
        usecols=(evid_index, time_index),
        unpack=True,
        dtype=str,
        **loadtxt_kwargs,
    )
    north, east, depth, mag = np.loadtxt(
        filename,
        usecols=(north_index, east_index, depth_index, magnitude_index),
        dtype=float,
        unpack=True,
        **loadtxt_kwargs,
    )

    if nameconverter is not None:
        name = map(nameconverter, name)

    if geoconverter is not None:
        north, east, depth = zip(*map(geoconverter, north, east, depth))

    if timeconverter is not None:
        time = map(timeconverter, time)
    else:
        time = map(float, time)

    return [
        core.Event(no, e, d, t, m, na)
        for no, e, d, t, m, na in zip(north, east, depth, time, mag, name)
    ]


def read_event_table(filename: str) -> list[core.Event]:
    """
    Read a relMT event table into list of events

    Parameters
    ----------
    filename : str
        Name of the event table file

    Returns
    -------
    event_list: list
        norting, easting, depth, magnitude, name

    """
    north, east, depth, time, mag = np.loadtxt(
        filename, usecols=(1, 2, 3, 4, 5), unpack=True, dtype=float
    )
    name = np.loadtxt(filename, usecols=6, unpack=True, dtype=str)
    return [
        core.Event(no, e, d, t, m, na)
        for no, e, d, t, m, na in zip(north, east, depth, time, mag, name)
    ]


def make_phase_table(phase_dict: dict[str:(float, float, float)]) -> str:
    """
    Convert phase dictionary to relMT compliant phase table.

    EventIndex Station Phase Arrivaltime  Azimuth Inclination
         (int)  (code) (P/S)   (seconds) (degree)    (degree)

    Arrival time is in the reference frame of the seismic trace.

    Azimuth is degree east of North (X-direction)

    Inclination is degree down from horizontal (in positive Z-direction)

    Parameters
    ----------
    phase_dict: dict
        Phase dictionary
        PhaseID -> Arrivaltime, Azimuth, Inclination

    Returns
    -------
    out : str
        Tabled phase data
    """

    out = "#     EventIndex Station Phase        Arrivaltime  Azimuth "
    out += "Inclination\n"
    out += "#          (int)   (str) (P/S)          (seconds) (degree)    "
    out += "(degree)\n"

    form = "{:>16d} {:>7s} {:>5s} {: 12.6f} {: 8.2f} {: 11.2f}\n"

    for phid, (time, azi, inc) in phase_dict.items():
        event_index, sta, pha = utils.split_phid(phid)
        out += form.format(event_index, sta, pha, time, azi, inc)

    return out


def read_phase_table(filename: str) -> dict[str : tuple[float, float, float]]:
    """
    Read a phase table into phase dictionary

    Parameters
    ----------
    filename : str
        Name of phase file

    Returns
    -------
    phase_dict: dict
        Lookup table phaseID -> (Arrivaltime, azimuth, inclination)
    """
    phids = map(
        utils.join_phid,
        *np.loadtxt(filename, usecols=(0, 1, 2), unpack=True, dtype=str),
    )
    times, azs, incs = np.loadtxt(filename, usecols=(3, 4, 5), unpack=True, dtype=float)

    return {
        phid: core.Phase(time, az, inc)
        for phid, time, az, inc in zip(phids, times, azs, incs)
    }


def make_waveform_arrays(
    traces: Stream,
    phase_dict: dict[str : tuple[float, float, float]],
    twind: float,
    sampling_rate: float,
) -> dict[str:NDArray]:
    """
    Isolate time windows around picks and wrtie to waveform array

    Parameters
    ----------
    traces : obspy.Stream
        Seismic event waveforms
    phase_dict : dict
        Phase dictionary containing phase arrival times
    twind : float
        Time window length around pick (seconds)
    sampling_rate : float
        Sampling rate of the output traces (1/seconds)

    Returns:
    wvdict: dict str(station_phase) -> NDArray
        Station codes and phase identifier to wave trains
    """

    ies, stas, phs = zip(*map(utils.split_phid, phase_dict))
    uies = set(ies)
    ustas = set(stas)
    uphs = set(phs)

    # Initialize output arrays
    nsamp = int(twind * sampling_rate) + 1
    ncha = len(set([tr.stats.channel[-1] for tr in traces]))
    nev = len(uies)
    keys = [f"{sta}_{pha}" for pha in set(phs) for sta in ustas]
    wvd = {key: np.zeros((nev, ncha, nsamp)) for key in keys}

    # Prepare data
    traces.sort(["starttime", "station", "channel"])
    dtmax = max([tr.stats.delta for tr in traces])  # maximum sampling intervall

    # Indices where next set of event observations begins
    t0tr = -np.inf
    nss = []
    ts = []
    for ntr, tr in enumerate(traces):
        ttr = tr.stats.starttime
        if ntr == 0 or abs(ttr - t0tr) > dtmax:
            nss.append(ntr)
            ts.append((ttr.timestamp, tr.stats.endtime.timestamp))
            t0tr = ttr
    nss.append(len(traces))  # explicit last index needed for slicing

    for event_index in uies:
        logger.debug(f"Working on event: {event_index}.")

        # Find traces that contain all picks for this event
        tps = [
            phase_dict[phid][0]
            for phid in phase_dict
            if utils.split_phid(phid)[0] == event_index
        ]
        tpmi = min(tps)
        tpma = max(tps)

        for nt, t in enumerate(ts):
            if tpmi > t[0] and tpma < t[1]:
                evtrs = traces[nss[nt] : nss[nt + 1]]
                break

        for sta in ustas:
            logger.debug(f"Working on station: {sta}.")
            try:
                trs = evtrs.select(station=sta)[
                    0:ncha
                ]  # This should be ncha components
            except IndexError:
                logger.info(
                    f"No or incomplete waveform data for statation: {sta}. Continuing."
                )
                continue

            for ph in uphs:
                logger.debug(f"Working on phase: {ph}.")
                phid = utils.join_phid(event_index, sta, ph)
                key = utils.join_waveid(sta, ph)
                try:
                    tp = UTCDateTime(phase_dict[phid][0])
                except KeyError:
                    logger.info(f"{phid} not in phase dictionary. Continuing.")
                    continue

                # Start and end time.
                tb = tp - twind / 2
                te = tp + twind / 2

                # Iterate over seismogram components
                for ic, tra in enumerate(trs):
                    tr = tra.copy()
                    tr.trim(
                        starttime=tb,
                        endtime=te,
                        pad=True,
                        fill_value=0,
                        nearest_sample=True,
                    )
                    tr.detrend("demean")
                    tr.detrend("simple")  # first and last sample 0 before fft
                    tr.resample(sampling_rate)
                    data = tr.data

                    if len(data) == 0:
                        msg = f"No data for: {phid}. Continuing"
                        logger.warning(msg)
                        continue

                    nm = nsamp - len(data)  # missing samples

                    if nm == 0:
                        wvd[key][event_index, ic, :] = data

                    elif nm > 0:
                        msg = f"{nm} of {nsamp} samples missing for {phid}. "
                        msg += "Padding with zeros at both ends."
                        logger.info(msg)
                        ps = nm // 2  # pad at start
                        pe = len(data) + nm // 2 + nm % 2  # pad at end
                        wvd[key][event_index, ic, ps:pe] = data

                    if nm < 0:
                        msg = f"{-nm} samples too many for {phid}. "
                        msg += "Cropping at both sides."
                        logger.info(msg)
                        ps = -nm // 2  # crop at start
                        pe = -(-nm // 2 + nm % 2)  # crop at end
                        wvd[key][event_index, ic, :] = data[:nsamp]

    return wvd


def read_waveform_file(filename: str) -> dict[str:NDArray]:
    """
    Read a waveform .npy array into waveform dictionary

    Parameters
    ----------
    filename : str
        Name of the waveform file

    Returns
    -------
    wave_array: NDArray
        Event waveform gather of one phase on one station
    """
    return np.load(filename)


def read_config(filename):
    """
    Read a configuration from .yaml file

    Parameters
    ----------
    filename: str
        Name of the configuration file
    Return
    ------
    config: core.Config
        Configuration object
    """
    with open(filename, "r") as fid:
        buf = yaml.safe_load(fid)

    return core.Config(**buf)
