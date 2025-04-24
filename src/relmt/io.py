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

"""In- and output functions"""

import numpy as np
import logging
from relmt import core, mt
from typing import Callable
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


# TODO: Sphinx table not working, likley due to stange indentation somewhe in the docs.
# https://earth.bsc.es/gitlab/es/autosubmit/-/merge_requests/316#note_199986


#    |Station| Northing| Easting |  Depth |
#    |-------|---------|---------|--------|
#    |(code) |(meter)  | (meter) | (meter)|


def make_station_table(station_dict: dict, filename: Path | str | None = None) -> str:
    """Convert station dictionary to relMT compliant station table

    The resulting table has one line per station. Columns are:

    Station, Northing, Easting, Depth
    (code), (meter), (meter),(meter)

    Parameters
    ----------
    station_dict:
        Station dictionary: Code -> core.Station
    filename:
        Write table to this file

    Returns
    -------
    out: str
        Tabeled station data
    """

    # Header
    out = "#Station      Northing       Easting        Depth\n"
    out += "# (code)       (meter)       (meter)      (meter) \n"
    form = "{:>8s} {:>12.3f} {:>12.3f} {:>11.3f}\n"

    for north, east, depth, code in station_dict.values():
        out += form.format(code, north, east, depth)

    if filename is not None:
        with open(filename, "w") as fid:
            fid.write(out)

    return out


def read_station_table(
    filename: str, unpack: bool = False
) -> dict[str, core.Station] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a relMT station table into dictionary structure

    Parameters
    ----------
    filename:
        Name of the station table file
    unpack:
        Return each variable as separate array

    Returns
    -------
    station_dict: dict
        Station code -> norting, easting, depth
    code, north, east, depth: :class:`numpy.ndarray`
        If `unpack=True`
    """
    code = np.loadtxt(filename, usecols=(0), unpack=True, dtype=str)
    (
        north,
        east,
        depth,
    ) = np.loadtxt(filename, usecols=(1, 2, 3), unpack=True, dtype=float)

    if unpack:
        return code, north, east, depth

    return {c: core.Station(n, e, d, c) for c, n, e, d in zip(code, north, east, depth)}


def read_exclude_file(filename: str | Path) -> core.Exclude:
    """
    Read an exclude file

    Parameters
    ----------
    filename:
        Name of the exclude file

    Returns
    -------
    Dictionary with station, event and phase exclusions
    """

    with open(str(filename), "r") as fid:
        excl = yaml.safe_load(fid)

    this_excl = {}
    for key in core.exclude:
        this_excl[key] = excl.get(key, [])

    return core.Exclude(**this_excl)


def save_yaml(filename: str, data: dict):
    """Save data to .yaml file"""
    with open(filename, "w") as fid:
        yaml.safe_dump(data, fid, sort_keys=False)


def make_event_table(
    event_list: list[core.Event], filename: Path | str | None = None
) -> str:
    """
    Convert event dictionary to relMT compliant event table.

    Number Northing Easting   Depth      Time Magnitude       Name
     (int)  (meter) (meter) (meter) (seconds)       (-)      (str)

    Parameters
    ----------
    event_list:
        Seismic event catalog
    filename:
        Write table to this file

    Returns
    -------
    Tabled event data
    """

    # Header
    out = "#Number     Northing      Easting       Depth         Origintime"
    out += " Magnitude             Name\n"
    out += "# (int)      (meter)      (meter)     (meter)          (seconds)"
    out += "       (-)            (str)\n"

    form = "{:>7d} {:>12.3f} {:>12.3f} {:>11.3f} {: 12.6f} {:>9.4f} {:>16s}\n"

    for iev, (north, east, depth, time, mag, name) in enumerate(event_list):
        out += form.format(iev, north, east, depth, time, mag, name)

    if filename is not None:
        with open(str(filename), "w") as fid:
            fid.write(out)

    return out


def make_mt_table(
    mt_dict: dict[int, core.MT] | dict[int, list[core.MT]],
    filename: Path | str | None = None,
) -> str:
    """
    Convert moment tensor dictionary to relMT compliant moment tensor table.

    Number  nn   ee   dd   ne   nd   ed
     (int) (Nm) (Nm) (Nm) (Nm) (Nm) (Nm)

    Parameters
    ----------
    mt_dict:
        Moment Tensor dictionary

        EventIndex ->  core.MT

        or

        EventIndex ->  list[core.MT]

    filename:
        If Write table to this file


    Returns
    -------
    Tabled moment tensor data
    """

    # Header
    out = "#Number            nn            ee            dd            ne"
    out += "            nd            ed\n"
    out += "# (int)          (Nm)          (Nm)          (Nm)          (Nm)"
    out += "           (Nm)         (Nm)\n"

    # Format
    form = "{:>7d} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e}\n"

    try:
        # Each Key holds a list of moment tensors
        out += "".join(
            form.format(iev, *momt) for iev, mts in mt_dict.items() for momt in mts
        )
    except TypeError:
        # Each Key holds a moment tensor
        out += "".join(form.format(iev, *momt) for iev, momt in mt_dict.items())

    if filename is not None:
        with open(filename, "w") as fid:
            fid.write(out)

    return out


def read_mt_table(
    filename: str,
    force_list: bool = False,
    unpack: bool = False,
) -> (
    dict[int, core.MT]
    | dict[int, list[core.MT]]
    | tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    """
    Read a relMT moment tensor table into moment tensor dictionary

    When only one moment tensor is present per event index, return a dict
    of moment tensors. When multiple moment tensors a present (e.g.
    bootstrapping results), return a dict of lists of moment tensors.

    Parameters
    ----------
    filename:
        Name of the moment tensor table file
    force_list:
        Always return a dict of lists. Ignored if `unpack=True`
    unpack:
        Return each variable as separate array. Superceeds `force_list`

    Returns
    -------
    mt_dict: dict, if `unpack=False`
        Moment tensor dictionary

        EventIndex ->  core.MT

        or

        EventIndex ->  list[core.MT]

    event_ids, nn, ee, dd, ne, nd, ed: :class:`numpy.ndarray`, if `unpack=True`
        Event IDs and MT components
    """

    ievs, *mts = np.loadtxt(
        filename,
        unpack=True,
        usecols=(0, 1, 2, 3, 4, 5, 6),
        dtype=float,
        ndmin=2,
    )

    mtarr = np.array(mts).T

    if unpack:
        return ievs.astype(int), *mts

    if force_list or len(set(ievs)) != len(ievs):
        logger.debug("Returning dict of lists of core.MT")
        return {
            int(iev): [core.MT(*mt) for mt in mtarr[ievs == iev, :]]
            for iev in sorted(set(ievs))
        }
    else:
        logger.debug("Returning dict of core.MT")
        return {int(iev): core.MT(*mtarr[ievs == iev, :][0]) for iev in ievs}


def read_event_table(
    filename: str | Path, unpack=False
) -> (
    list[core.Event]
    | tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    """
    Read a relMT event table into list of events

    Parameters
    ----------
    filename:
        Name of the event table file
    unpack:
        Return each variable as separate array.

    Returns
    -------
    event_list: list, if `unpack=False`
        Seismic event catalog

    north, east, depth, magnitude, name: :class:`numpy.ndarray` if `unpack=True`
        Each parameter of the event catalog as an array

    Raises
    ------
    IndexError:
        If event indices in table are not consecutive
    """
    evids, north, east, depth, time, mag = np.loadtxt(
        filename, usecols=(0, 1, 2, 3, 4, 5), unpack=True, dtype=float
    )
    name = np.loadtxt(filename, usecols=6, unpack=True, dtype=str)

    if not np.all(evids == np.arange(len(evids))):
        raise IndexError("Input event list is not consecutive")

    if unpack:
        return north, east, depth, time, name

    return [
        core.Event(no, e, d, t, m, na)
        for no, e, d, t, m, na in zip(north, east, depth, time, mag, name)
    ]


def make_phase_table(
    phase_dict: dict[str, core.Phase], filename: Path | str | None = None
) -> str:
    """
    Convert phase dictionary to relMT compliant phase table.

    EventIndex Station Phase Arrivaltime  Azimuth   Plunge
         (int)  (code) (P/S)   (seconds) (degree) (degree)

    Parameters
    ----------
    phase_dict:
        Phase dictionary
        PhaseID -> :class:`core.Phase`
    filename:
        If not None, write table to this file

    Returns
    -------
    Tabled phase data
    """

    out = "#     EventIndex Station Phase        Arrivaltime  Azimuth "
    out += "Plunge\n"
    out += "#          (int)   (str) (P/S)          (seconds) (degree)    "
    out += "(degree)\n"

    form = "{:>16d} {:>7s} {:>5s} {: 12.6f} {: 8.2f} {: 11.2f}\n"

    for phid, (time, azi, inc) in phase_dict.items():
        event_index, sta, pha = core.split_phaseid(phid)
        out += form.format(event_index, sta, pha, time, azi, inc)

    if filename is not None:
        with open(filename, "w") as fid:
            fid.write(out)

    return out


def read_phase_table(
    filename: str, unpack: bool = False
) -> (
    dict[str, tuple[float, float, float]]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Read a phase table into phase dictionary

    Parameters
    ----------
    filename:
        Name of phase file
    unpack:
        Return each variable as separate array

    Returns
    -------
    phase_dict: dict if `unpack=False`
        Lookup table phaseID -> :class:`relmt.core.Phase`
    phase_id, time, azi, plung: :class:`numpy.ndarray` if `unpack=True`
        The :class:`relmt.core.Phase` attributes as :class:`numpy.ndarray`

    """
    phids = map(
        core.join_phaseid,
        *np.loadtxt(filename, usecols=(0, 1, 2), unpack=True, dtype=str),
    )
    times, azs, incs = np.loadtxt(filename, usecols=(3, 4, 5), unpack=True, dtype=float)

    if unpack:
        return np.array(list(phids)), times, azs, incs

    return {
        phid: core.Phase(time, az, inc)
        for phid, time, az, inc in zip(phids, times, azs, incs)
    }


def read_waveform_array_header(
    station: str,
    phase: str,
    n_align: int = 0,
    directory: str = "",
) -> tuple[np.ndarray, core.Header]:
    """
    Read a waveform array and corresponding header file.

    Defalut values are drawn from default header.

    Parameters
    ----------
    station:
        Station name
    phase:
        Seismic phase, 'P' or 'S'
    n_align:
        Alignment iterration, `0` for input data
    directory:
        Root directory to look for the files

    Returns
    -------
    waveform_array: :class:`numpy.ndarray`
        Event waveform gather of one phase type on one station
    header: :class:`relmt.core.Header`
        Corresponding header dictionary
    """

    wvf = core.file("waveform_array", station, phase, n_align, directory)
    hdrf = core.file("waveform_header", station, phase, n_align, directory)
    default_hdrf = core.file("waveform_header", directory=directory)

    # Load the waveform array...
    wvarr = np.load(wvf)

    # Read in default values from default config, if present

    try:
        hdr = read_header(hdrf, default_name=default_hdrf)
    except FileNotFoundError:
        logger.debug(f"Missing default config: {default_hdrf}")
        hdr = read_header(hdrf)

    return wvarr, hdr


def read_config(filename: str) -> core.Config:
    """
    Read a configuration from .yaml file

    Parameters
    ----------
    filename:
        Name of the configuration file
    Return
    ------
    config: :class:`relmt.core.Config`
        Configuration object
    """

    with open(filename, "r") as fid:
        buf = yaml.safe_load(fid)

    return core.Config(**buf)


def read_header(filename: str, default_name: str | None = None) -> core.Header:
    """
    Read a waveform header from .yaml file

    Parameters
    ----------
    filename:
        Name of the header file
    default_name:
        Name of a file containg default values

    Return
    ------
    Header dictionary
    """
    hdr = core.Header()
    if default_name is not None:
        logger.debug(f"Read default values from file: {default_name}")
        hdr.update_from_file(default_name)

    hdr.update_from_file(filename)

    return hdr


def save_results(filename: str | Path, arr: np.ndarray):
    """Save array to filename using :func:`numpy.savetxt`"""

    try:
        np.savetxt(filename, arr)
    except ValueError:
        np.savetxt(filename, [arr])


def save_amplitudes(
    filename: str,
    table: list[core.P_Amplitude_Ratio] | list[core.S_Amplitude_Ratios],
):
    """Save relative amplitudes to file

    Parameters
    ----------
    filename:
        Name of the output file
    table:
        List of P- or S-amplitude ratios
    """
    ncol = len(table[0])

    if ncol == 5:
        fmt = "{:10s} {:7d} {:7d} {:25.18e} {:7.5f}\n"
        out = "#Station    EventA  EventB  Amplitude_AB             Misfit\n"
    elif ncol == 7:
        fmt = "{:10s} {:7d} {:7d} {:7d} {:25.18e} {:25.18e} {:7.5f}\n"
        out = "#Station    EventA  EventB  EventC  Amplitude_ABC             "
        out += "Amplitude_ACB            Misfit\n"
    else:
        msg = "Found {ncol} index columns, but only 5 or 7 are allowed."
        raise IndexError(msg)

    out += "".join(fmt.format(*line) for line in table)

    with open(filename, "w") as fid:
        fid.write(out)


def read_amplitudes(filename: str, phase: str, unpack: bool = False):
    """
    Load relative amplitudes from file

    Parameters
    ----------
    filename:
        Name of the input file
    phase:
        Seismic phase to consider
    unpack:
        Return each variable as separate array

    Returns
    -------
    amplitudes: if `unpack=False`
        Lists of the :class:`relmt.core.P_Amplitude_Ratio` or
        :class:`relmt.core.S_Amplitude_Ratios`
    station, a, b, amplitude, misfit: if `unpack=True` and `phase=P`
        Arrays of the :class:`relmt.core.P_Amplitude_Ratio` attribute
    station, a, b, c, amp_abc, amb_acb, misfit: if `unpack=True` and `phase=S`
        Arrays of the :class:`relmt.core.S_Amplitude_Ratios` attributes

    Raises
    ------
    ValueError:
        If 'phase' is not `P` or `S`
    """
    stas = np.loadtxt(filename, usecols=0, dtype=str)

    if phase.upper() == "P":
        X = np.loadtxt(filename, usecols=(1, 2, 3, 4), ndmin=2)

        if unpack:
            return stas, *X.T

        return [
            core.P_Amplitude_Ratio(sta, ia, ib, amp, mis)
            for sta, ia, ib, amp, mis in zip(
                stas, *X[:, :2].T.astype(int), X[:, 2], X[:, 3]
            )
        ]

    elif phase.upper() == "S":
        X = np.loadtxt(filename, usecols=(1, 2, 3, 4, 5, 6), ndmin=2)

        if unpack:
            return stas, *X.T

        return [
            core.S_Amplitude_Ratios(sta, ia, ib, ic, amp_abc, amp_acb, mis)
            for sta, ia, ib, ic, amp_abc, amp_acb, mis in zip(
                stas, *X[:, :3].T.astype(int), *X[:, [3, 4]].T, X[:, 5]
            )
        ]

    else:
        raise ValueError("'phase' must be 'P' or 'S'")


def make_gmt_meca_table(
    moment_tensors: list[core.MT] | dict[int, core.MT],
    event_list: list[core.Event] | None = None,
    geoconverter: Callable | None = None,
    filename: str | Path | None = None,
    **savetxt_kwargs: dict,
) -> np.ndarray:
    """Return input compatible with Generic Mapping Tools :func:`meca` function

    Parameters
    ----------
    moment_tensors:
        relMT moment tensors
    event_list:
        The seismic event catalog from which to source location
    geoconverter:
        Function that accepts event north, east, depth coordinates and converts
        into desired output coordianted (e.g. longitude, latitude, depth)
    filename:
        Save output table to file
    savetxt_kwargs:
        When saving to file, keyword arguments are passed on to
        :func:`numpy.savetxt`

    Returns
    -------
    Data input for GMT or PyGMT
    """

    out = []

    # Iterate over dict or list
    def list_dict_iter(obj):
        return obj if isinstance(obj, dict) else range(len(obj))

    for imt in list_dict_iter(moment_tensors):
        momt = moment_tensors[imt]

        north, east, depth = (0.0, 0.0, 0.0)
        if event_list is not None:
            ev = event_list[imt]
            north, east, depth = ev.east, ev.north, ev.depth

        if geoconverter is not None:
            north, east, depth = geoconverter(north, east, depth)

        exp = min(np.log10(np.abs(np.array([*momt]))).astype(int))

        mrr, mtt, mff, mrt, mrf, mtf = np.array([*mt.ned2rtf(*momt)]) / 10.0**exp

        exp -= 7  # Nm -> dyne cm

        out.append([north, east, depth, mrr, mtt, mff, mrt, mrf, mtf, exp])

    outarr = np.array(out)

    if filename is not None:
        np.savetxt(filename, outarr, **savetxt_kwargs)

    return outarr


def read_ext_event_table(
    filename: str,
    north_index: int,
    east_index: int,
    depth_index: int,
    time_index: int,
    magnitude_index: int,
    name_index: int,
    geoconverter: Callable | None = None,
    timeconverter: Callable | None = None,
    nameconverter: Callable | None = None,
    loadtxt_kwargs: dict = {},
) -> list[core.Event]:
    """
    Read an external event table into an event dictionary.

    Parameters
    ----------
    filename:
        Name of the event table file
    north_index, east_index, depth_index, time_index, magnitude_index, name_index:
        Column indices of the event northing, easting, depth, time, magnitude, and name
    geoconverter:
        Function that takes north, east and depth as arguments and returns local
        northing and easting and depth coordinates in meters (e.g. interpret
        north and east as latitude and longitude, and return UTM northing and
        easting; or convert kilometer depth to meter)
    timeconverter:
        Function that takes time string as argument and returns time in seconds
        as a float (e.g. epoch timestamp. Must be consistent with the reference
        frame of the waveforms)
    nameconverter:
        Function that takes a string as argument and returns a user defined
        event name string.
    loadtxt_kwargs: dict
        Additional keyword arguments are passed on to :func:`numpy.loadtxt`

    Returns
    -------
    List of norting, easting, depth, time, magnitude, name

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
        usecols=(name_index, time_index),
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


def read_ext_mt_table(
    filename: str,
    nn_ee_dd_ne_nd_ed_indices: tuple[int, int, int, int, int, int],
    name_index: int | None = None,
    event_list: list[core.Event] | None = None,
    mtconverter: Callable | None = None,
    exponent_index: int | None = None,
    nameconverter: Callable | None = None,
    loadtxt_kwargs: dict = {},
) -> dict[int, core.MT]:
    """
    Read an external moment tensor table into a moment tensor dictionary.

    Parameters
    ----------
    filename:
        Name of the event table file
    nn_ee_dd_ne_nd_ed_indices:
        Column indices of the `nn`, `ee`, `dd`, `ne`, `nd`, and `ed` components
        of the moment tensor
    name_index:
        Column indices of the event name. If None, do not attempt to match event
        in 'event_list'
    event_list:
        List of core.Event tuples. When given, only moment tensors that have a
        matching event name are considered
    mtconverter:
        Function that takes six moment teonsor components (in the order given in
        nn_ee_dd_ne_nd_ed_indices) and, if provided, a seventh exponent argument
        and returns moment tensor components in nn, ee, dd, ne, nd, ed
        convention
    exponent_index:
        Column indices of possible field holding the base-10 exponent. If
        present, we pass this on as last argument to `mt_converter`. Ignored, if
        `mt_converter=None`
    nameconverter:
        Function that takes a string as argument and returns a user defined
        event name string. Only event names that match a name in event_list are considered.
    loadtxt_kwargs:
        Additional keyword arguments are passed on to :func:`numpy.loadtxt`

    Returns
    -------
    Event dictionary event index -> core.MT

    Raises
    ------
        KeyError: if 'usecol', 'unpack', or 'dtype' are specified as `loadtxt_kwargs`
    """

    kwargs = ["usecol", "unpack", "dtype"]

    if any([kwarg in loadtxt_kwargs for kwarg in kwargs]):
        msg = f"loadtxt_kwargs: {', '.join(kwargs)} are reserved."
        raise KeyError(msg)

    if name_index is not None:
        names = np.loadtxt(
            filename,
            usecols=name_index,
            dtype=str,
            **loadtxt_kwargs,
        )
    else:
        names = np.array([])

    # nn, ee, dd, ne, nd, ed = np.loadtxt(
    mt_components = np.loadtxt(
        filename,
        usecols=nn_ee_dd_ne_nd_ed_indices,
        dtype=float,
        # unpack=True,
        **loadtxt_kwargs,
    )

    # If necessary, convert event names
    if nameconverter is not None and name_index is not None:
        if len(names.shape) > 1:
            names = np.array([nameconverter(*name) for name in names])
        else:
            names = np.array(list(map(nameconverter, names)))

    # If necessary, convert moment tensor elements
    if mtconverter is not None:
        if exponent_index is None:
            # nn, ee, dd, ne, nd, ed = zip(*map(mtconverter, nn, ee, dd, ne, nd, ed))
            mt_components = np.array(
                [mtconverter(*mt_comps) for mt_comps in mt_components]
            )
        else:
            # We need an exponent, so read it here
            exp = np.loadtxt(
                filename,
                usecols=exponent_index,
                dtype=float,
                **loadtxt_kwargs,
            )
            # mt_components = map(mtconverter, nn, ee, dd, ne, nd, ed, exp)
            mt_components = np.array(
                [mtconverter(*mt_comps, ex) for mt_comps, ex in zip(mt_components, exp)]
            )

    # Populate dictionary only with known events
    mt_dict = {}
    if event_list is not None and name_index is not None:
        for n, ev in enumerate(event_list):
            if ev.name in names:
                # Find the MT with the event's name
                i = np.nonzero(names == ev.name)[0][0]
                mt_dict[n] = core.MT(*mt_components[i])
    else:
        for n, mt_component in enumerate(mt_components):
            mt_dict[n] = core.MT(*mt_component)

    return mt_dict


def read_phase_nll_hypfile(
    filename: str,
    event_index: int,
    subtract_residual: bool,
    subtract_stationterm: bool,
    minimum_ray_quality: int = 0,
) -> dict[str, core.Phase]:
    """
    Read arrival times and take-off angles from NonLinLoc .hyp file.

    Arrival times can be corrected for location residual and station term.

    Ray take-off azimuth is output in degree east of North (X-direction)

    Ray take-off plunge is output degree down from horizontal (in positive
    Z-direction)

    Parameters
    ----------
    filename:
        Name of the NonLinLoc hypocenter (.hyp) file
    event_index:
        Event index of this set of phase observations. Must correspond to event
        index in event dictionary.
    subtract_residual:
        Subtract localization residual from the arrival time
    subtract_stationterm:
        Subtract station term from the arrival time
    minimum_ray_quality:
        Minumum ray quality to use for take-off angles (0=unreliable, 10=best)

    Returns
    -------
    Lookup table phase ID -> (Arrivaltime, azimuth, plunge)
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
            t = datetime.strptime(date + hourmin, "%Y%m%d%H%M").replace(
                tzinfo=ZoneInfo("UTC")
            ).timestamp() + float(sec)
        except ValueError:
            msg = (
                f"Could not convert pick time to datetime: {date + hourmin}. Skipping."
            )
            logger.warning(msg)
            continue

        if subtract_residual:
            t -= float(line[16])
        if subtract_stationterm:
            t -= float(line[26])

        if int(line[25]) >= minimum_ray_quality:
            # seen from the source, it's the back-azimuth
            azi = (float(line[23]) - 180) % 360
            plu = 90 - float(line[24])  # NLL has 0=down
        else:
            azi = plu = np.nan

        sta = str(line[0])
        pha = str(line[4])

        phid = core.join_phaseid(event_index, sta, pha)
        outd[phid] = core.Phase(t, azi, plu)

    return outd
