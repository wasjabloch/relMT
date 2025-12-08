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
from relmt import core, mt
from typing import Callable
import yaml
from scipy.io import loadmat
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from warnings import filterwarnings

logger = core.register_logger(__name__)

# Ignore warnings about changed np.loadtxt behvaior.
# May be removed in the future, when warning is not issued any more by numpy
filterwarnings("ignore", "Input line 1 contained no data")

# TODO: Sphinx table not working, likley due to stange indentation somewhe in the docs.
# https://earth.bsc.es/gitlab/es/autosubmit/-/merge_requests/316#note_199986


#    |Station| Northing| Easting |  Depth |
#    |-------|---------|---------|--------|
#    |(code) |(meter)  | (meter) | (meter)|


def write_station_table(station_dict: dict, filename: Path | str | None = None) -> str:
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

    template = core.exclude

    try:
        with open(str(filename), "r") as fid:
            excl = yaml.safe_load(fid)
    except FileNotFoundError:
        logger.info(
            f"No exclude file found: {filename}. "
            "Assuming there is nothing to exclude."
        )
        return template.copy()

    this_excl = core.Exclude()
    for key in template:
        this_excl[key] = excl.get(key, [])

    return this_excl


def save_yaml(filename: str, data: dict, format_bandpass=False):
    """Save data to .yaml file"""

    # Format for bandpass yaml files, by ChatGPT
    class BandpassDumper(yaml.SafeDumper):
        pass

    def represent_list(dumper, data):
        # flow_style=True → [a, b], flow_style=False → block style
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    def represent_float(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:float", format(data, ".5g"))

    BandpassDumper.add_representer(float, represent_float)
    BandpassDumper.add_representer(list, represent_list)

    dumper = yaml.dumper.SafeDumper
    if format_bandpass:
        dumper = BandpassDumper

    with open(filename, "w") as fid:
        yaml.dump(data, fid, sort_keys=False, Dumper=dumper)


def read_yaml(filename: str) -> dict:
    """Read data from .yaml file"""
    with open(filename, "r") as fid:
        buf = yaml.safe_load(fid)
    return buf


def write_event_table(
    event_dict: dict[int, core.Event], filename: Path | str | None = None
) -> str:
    """
    Convert event dictionary to relMT compliant event table.

    Number Northing Easting   Depth      Time Magnitude       Name
     (int)  (meter) (meter) (meter) (seconds)       (-)      (str)

    Parameters
    ----------
    event_dict:
        Seismic event catalog
    filename:
        Write table to this file

    Returns
    -------
    Tabled event data
    """

    # Header
    out = "#Number     Northing      Easting       Depth"
    out += "               Origintime Magnitude                 Name\n"
    out += "# (int)      (meter)      (meter)     (meter)          (seconds)"
    out += "       (-)            (str)\n"

    form = "{:>7d} {:>12.3f} {:>12.3f} {:>11.3f} {: 18.6f} {:>9.4f} {:>20s}\n"

    for iev, (north, east, depth, time, mag, name) in event_dict.items():
        out += form.format(iev, north, east, depth, time, mag, name)

    if filename is not None:
        with open(str(filename), "w") as fid:
            fid.write(out)

    return out


def write_mt_table(
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
    dict[int, core.Event]
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
    event_dict: list, if `unpack=False`
        Seismic event catalog

    evid, north, east, depth, magnitude, name: :class:`numpy.ndarray` if `unpack=True`
        Each parameter of the event catalog as an array

    Raises
    ------
    IndexError:
        If event indices in table are not unique
    """
    evids, north, east, depth, time, mag = np.loadtxt(
        filename,
        usecols=(0, 1, 2, 3, 4, 5),
        unpack=True,
        dtype=float,
        ndmin=2,
    )
    name = np.loadtxt(filename, usecols=6, unpack=True, dtype=str, ndmin=1)

    evids = evids.astype(int, copy=False)

    if len(set(evids)) != len(evids):
        uniq, cnt = np.unique(evids, return_counts=True)
        msg = "Some event IDs are not unique: "
        msg += ", ".join(f"{val}" for val in uniq[cnt > 1])
        raise IndexError(msg)

    if unpack:
        return evids, north, east, depth, time, mag, name

    return {
        evid: core.Event(no, e, d, t, m, na)
        for evid, no, e, d, t, m, na in zip(evids, north, east, depth, time, mag, name)
    }


def write_phase_table(
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
    matlab: bool = False,
    defaults: bool = True,
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
    matlab:
        Assume MATALB waveform files. Provide the name of the variable holding
        the waveform array in as header keyword `variable_name`.
    defaults:
        Attempt to pull in default vaules from a `default-hdr.yaml` file in the
        same directory

    Returns
    -------
    waveform_array: :class:`numpy.ndarray`
        Event waveform gather of one phase type on one station
    header: :class:`relmt.core.Header`
        Corresponding header dictionary
    """

    dir = Path(directory)

    wvf = core.file("waveform_array", station, phase, n_align, directory=dir)
    hdrf = core.file("waveform_header", station, phase, n_align, directory=dir)

    loader = np.load
    if matlab:
        wvf = wvf.with_suffix(".mat")
        loader = loadmat

    # Load the waveform array...
    try:
        wvarr = loader(wvf)
    except FileNotFoundError as e:
        logger.debug(f"File not found: {wvf}")
        raise e

    # Read in default values from default config, if present
    if defaults:
        default_hdrf = core.file("waveform_header", n_align=n_align, directory=dir)
        try:
            hdr = read_header(hdrf, default_name=default_hdrf)
        except FileNotFoundError:
            logger.debug(f"Missing default config: {default_hdrf}")
            hdr = read_header(hdrf)
    else:
        hdr = read_header(hdrf)

    if (vname := hdr["variable_name"]) is not None:
        wvarr = wvarr[vname]

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

        try:
            hdr.update_from_file(default_name)
            logger.debug(f"Read default values from header file: {default_name}")
        except FileNotFoundError:
            logger.debug(f"No default header file found: {default_name}")

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
    more_data: list[np.ndarray] = [],
    more_names: list[str] = [""],
    more_formats: list[str] | None = None,
):
    """Save relative amplitudes to file

    Parameters
    ----------
    filename:
        Name of the output file
    table:
        List of P- or S-amplitude ratios
    more_data:
        ``(len(table),)`` arrays holding additional data columns
    more_names:
        Corresponding variable names
    more_formats:
        Corresponding format specifiers
    """

    try:
        ncol = len(table[0])
    except IndexError:
        logger.error("Table is empty. No data to be saved.")
        return

    if ncol == 10:
        fmt = "{:10s} {:7d} {:7d} {:20.13e} {:7.5f} {:11.5f} {:6.4f} {:6.4f} "
        fmt += "{:8.2e} {:8.2e} "
        out = "#Station    EventA  EventB  Amplitude_AB        Misfit  "
        out += "Correlation Sigma1 Sigma2 Highpass Lowpass "
    elif ncol == 13:
        fmt = "{:10s} {:7d} {:7d} {:7d} {:20.13e} {:20.13e} {:7.5f} "
        fmt += "{:11.5f} {:6.4f} {:6.4f} {:6.4f} {:8.2e} {:8.2e} "
        out = "#Station    EventA  EventB  EventC  Amplitude_ABC        "
        out += "Amplitude_ACB       Misfit  Correlation Sigma1 Sigma2 Sigma3 "
        out += "Highpass Lowpass "
    else:
        msg = f"Found {ncol} index columns, but only 9 or 11 are allowed."
        raise IndexError(msg)

    if more_formats is not None:
        fmt += " ".join(more_formats)
    else:
        fmt += " ".join(["{:20.13e}"] * len(more_data))
    fmt += "\n"

    out += " ".join(more_names)
    out += "\n"

    # Make a stack and transform to iterate
    if len(more_data) > 0:
        more_arr = np.vstack(more_data).T
        out += "".join(fmt.format(*line, *more) for line, more in zip(table, more_arr))
    else:
        out += "".join(fmt.format(*line) for line in table)

    with open(filename, "w") as fid:
        fid.write(out)


def save_mt_result_summary(
    filename: str | Path,
    event_dict: dict[int, core.Event],
    mt_dict: dict[int, core.MT],
    gaps: dict[int, np.ndarray] = {},
    links: dict[int, tuple[int, int]] = {},
    misfits: dict[int, float] = {},
    correlations: dict[int, float] = {},
    moment_rms: dict[int, float] = {},
    amplitude_rms: dict[int, float] = {},
    bootstrap_rms: dict[int, float] = {},
    bootstrap_kagan: dict[int, float] = {},
):
    """Combine moment tensor dictionary and event table and write out resut table"""

    evd = event_dict

    # Event number and name
    arrays = [np.array([evn for evn in mt_dict])]
    arrays += np.hsplit(np.array([momt for momt in mt_dict.values()]), 6)

    arrays += [np.array([evd[evn].name for evn in mt_dict])]

    # A priori floats
    arrays += np.hsplit(
        np.array(
            [
                (
                    evd[evn].north,
                    evd[evn].east,
                    evd[evn].depth,
                    evd[evn].time,
                    evd[evn].mag,
                    mt.magnitude_of_moment(mt.moment_of_vector(momt)),
                    gaps.get(evn, [np.nan])[0],  # First and second azimuthal gap
                    (
                        gaps.get(evn, [np.nan])[1]
                        if len(gaps.get(evn, [])) > 1
                        else np.nan
                    ),
                    float(links.get(evn, [np.nan])[0]),  # P-links
                    float(links.get(evn, [np.nan, np.nan])[1]),  # S-links
                    misfits.get(evn, np.nan),
                    correlations.get(evn, np.nan),
                    moment_rms.get(evn, np.nan),
                    amplitude_rms.get(evn, np.nan),
                    bootstrap_rms.get(evn, np.nan),
                    bootstrap_kagan.get(evn, np.nan),
                )
                for evn, momt in mt_dict.items()
            ]
        ),
        16,
    )

    headers = [
        "#   Event",
        "           nn",
        "           ee",
        "           dd",
        "           ne",
        "           nd",
        "           ed",
        "                Name",
        "       North",
        "        East",
        "       Depth",
        "              Time",
        "   Ml",
        "   Mw",
        "Gap1",
        "Gap2",
        "  P-links",
        "  S-links",
        "  Misfit",
        "Correlation",
        "MomentRMS",
        "AmplitudeRMS",
        "BootstrapRMS",
        "KaganRMS",
    ]

    fmts = (
        "%9s %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e %20s %12.3f %12.3f %12.3f "
        "%18.6f %5.2f %5.2f %4.0f %4.0f %9.0f %9.0f %8.4f %11.5f %9.2e %12.2e %12.6f %8.3f"
    ).split()

    write_formatted_table(arrays, fmts, headers, filename)


def write_formatted_table(
    arrays: list[np.ndarray],
    formatters: list[str],
    headers: list[str],
    outfile: str,
    delim=" ",
):
    """
    Write a mixed-type table to a text file.

    Parameters
    ----------
    arrays:
        Each array is one column; all must have the same length and be of
        `dtype` `int`, `float` or `str`-like
    formatters:
        A printf-style format specifier for each column, e.g. ['%s','%.2f','%04d'].
    headers:
        Column names, same length as arrays. Will be joined with `delim`
    outfile:
        Path to write, or an already-open file handle.
    delim:
        Field delimiter.
    """

    # Inspired by ChatGPT 4o-mini-high
    # Sanity checks
    ncols = len(arrays)
    if not (len(formatters) == ncols == len(headers)):
        msg = "'arrays', 'formatters' and 'headers' must all have the same length"
        raise ValueError(msg)

    # ensure all arrays are 1D and same length
    lengths = [arr.shape[0] for arr in arrays]
    if len({*lengths}) != 1:
        raise ValueError("All arrays must have the same number of rows")
    nrows = lengths[0]

    # Create a structured array for output
    dtype_fields = []
    proc_cols = []

    for idx, arr in enumerate(arrays):
        fmt = formatters[idx]

        # treat anything with '%s' or non-numeric dtype as “string” column
        if fmt.endswith("s") or arr.dtype.kind in ("U", "S", "O"):
            # convert all values to Python str
            col_str = arr.astype(str)
            # find max string length
            maxlen = max(len(x) for x in col_str)
            # Unicode string field of exactly maxlen characters
            dtype_fields.append((f"f{idx}", f"<U{maxlen}"))
            proc_cols.append(col_str)

        else:
            # numeric column: preserve or cast dtype
            if arr.dtype.kind not in ("i", "f"):
                arr = arr.astype(float)
            dtype_fields.append((f"f{idx}", arr.dtype))
            proc_cols.append(arr)

    # build structured dtype
    dtypes = np.dtype(dtype_fields)

    # --- 3) Build a list of row-tuples ---
    # note: list(...) so numpy.savetxt can infer shape
    # TODO: is this really necessary? Appears like a massive bottle neck
    rows = [tuple(col[i] for col in proc_cols) for i in range(nrows)]

    # --- 4) Convert to structured array and write ---
    structured_arr = np.array(rows, dtype=dtypes)
    np.savetxt(
        outfile,
        structured_arr,
        fmt=formatters,
        delimiter=delim,
        header=delim.join(headers),
        comments="",  # no leading '#' on header
    )


def read_combinations(filename: str | Path) -> set[tuple[int, int]]:
    """
    Read a table of event combinations

    Each pair will be sorted to allow for


    Parameters
    ----------
    filename:
        Name of the input file
    Returns
    -------
    ``(combinations, 2)`` array of event combinations
    """
    combs = np.loadtxt(filename, usecols=(0, 1), dtype=int)
    combs.sort(axis=-1)
    return set(tuple(comb) for comb in combs)


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
    station, a, b, amplitude, misfit, correlation, sigma1, sigma2, higpass,
    lowpass: if `unpack=True` and `phase=P`
        Arrays of the :class:`relmt.core.P_Amplitude_Ratio` attribute
    station, a, b, c, amp_abc, amb_acb, misfit, correlation, sigma1, sigma2,
    sigma3, higpass, lowpass: if `unpack=True` and `phase=S`
        Arrays of the :class:`relmt.core.S_Amplitude_Ratios` attributes

    Raises
    ------
    ValueError:
        If 'phase' is not `P` or `S`
    """
    stas = np.loadtxt(filename, usecols=0, dtype=str)

    if phase.upper() == "P":
        events = np.loadtxt(filename, usecols=(1, 2), dtype=int, ndmin=2)
        X = np.loadtxt(filename, usecols=(3, 4, 5, 6, 7, 8, 9), ndmin=2)

        if unpack:
            return stas, *events.T, *X.T

        return [
            core.P_Amplitude_Ratio(sta, ia, ib, *floats)
            for sta, ia, ib, floats in zip(
                stas,
                *events.T,
                X,
            )
        ]

    elif phase.upper() == "S":
        events = np.loadtxt(filename, usecols=(1, 2, 3), dtype=int, ndmin=2)
        X = np.loadtxt(filename, usecols=(4, 5, 6, 7, 8, 9, 10, 11, 12), ndmin=2)

        if unpack:
            return stas, *events.T, *X.T

        return [
            core.S_Amplitude_Ratios(sta, ia, ib, ic, *floats)
            for sta, ia, ib, ic, floats in zip(
                stas,
                *events.T,
                X,
            )
        ]

    else:
        raise ValueError("'phase' must be 'P' or 'S'")


def write_gmt_meca_table(
    moment_tensors: list[core.MT] | dict[int, core.MT],
    event_dict: dict[int, core.Event] | None = None,
    geoconverter: Callable | None = None,
    filename: str | Path | None = None,
    **savetxt_kwargs: dict,
) -> np.ndarray:
    """Return input compatible with Generic Mapping Tools :func:`meca` function

    Parameters
    ----------
    moment_tensors:
        relMT moment tensors
    event_dict:
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
        if event_dict is not None:
            ev = event_dict[imt]
            north, east, depth = ev.north, ev.east, ev.depth

        if geoconverter is not None:
            north, east, depth = geoconverter(north, east, depth)

        exp = min(np.log10(np.abs(np.array([*momt]))).astype(int))

        mrr, mtt, mff, mrt, mrf, mtf = np.array([*mt.ned2rtf(*momt)]) / 10.0**exp

        exp -= 7  # Nm -> dyne cm

        out.append([east, north, depth, mrr, mtt, mff, mrt, mrf, mtf, exp])

    outarr = np.array(out)

    if filename is not None:
        np.savetxt(filename, outarr, **savetxt_kwargs)

    return outarr


def _make_hypodd_cc_table(
    evpair_dd: dict[str, np.ndarray],
    cc: dict[str, np.ndarray],
    filename: str | Path | None = None,
    exclude: core.Exclude | None = None,
) -> str:
    """Make HypoDD compliant delay time file"""
    pass


#    for wvid in evpair_dd:
#        sta, pha = core.split_waveid(wvid)
#        iev_dt = evpair_dd[wvid]
#        ccij = cc[wvid]
#        evpairs = iev_dt[:, :2].astype(int)
#        dts = iev_dt[:, 2]
#
#
#        ccs = [ccij[evlist.index(evp[0]), evlist.index(evp[1])] for evp in evpairs]
#
#        return {
#            evpair: (station, dt, cc, phase)
#            for evpair, dt, cc in zip(evpairs, dts, ccs)
#        }


def read_velocity_model(filename: str | Path, has_kilometer=False) -> np.ndarray:
    """Read a velocity  model from file

    The space-seperated file holds three columns: Depth (m), Vp (m/s), Vs (m/s).
    If the last column is absent, we assume a constant Vp/Vs ratio of 1.73.

    Parameters
    ----------
    filename:
        Name of the event table file
    has_kilometer:
        Input file has depth in km and velocities in km/s.

    Returns
    -------
    ``(layers, 3)`` table of Depth (m), Vp (m/s), Vs (m/s)
    """

    try:
        buf = np.loadtxt(filename)
    except ValueError:
        buf = np.loadtxt(filename, delimiter=",")

    if buf.shape[1] == 2:
        # Compute Vp from Vs
        buf = np.hstack((buf, buf[:, 1][:, np.newaxis] / 1.73))
    elif (ncol := buf.shape[1]) != 3:
        msg = f"File has {ncol} columns. Expected 2 or 3."
        raise IndexError(msg)

    if has_kilometer:
        buf *= 1e3

    return buf


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
) -> dict[int, core.Event]:
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
    The seismic event catalog

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

    return {
        evid: core.Event(no, e, d, t, m, na)
        for evid, (no, e, d, t, m, na) in enumerate(
            zip(north, east, depth, time, mag, name)
        )
    }


def read_ext_mt_table(
    filename: str,
    nn_ee_dd_ne_nd_ed_indices: tuple[int, int, int, int, int, int],
    name_index: int | None = None,
    event_dict: dict[int, core.Event] | None = None,
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
        in 'event_dict'
    event_dict:
        Dictionary of core.Event tuples. When given, only moment tensors that have a
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
    if event_dict is not None and name_index is not None:
        for n, ev in event_dict.items():
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
