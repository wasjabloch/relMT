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

"""Naming conventions and configuration classes for relMT"""

import logging
import yaml
import inspect
import os
from functools import wraps
from collections.abc import Iterator, Generator, Callable
from typing import TypedDict
from pathlib import Path

from collections import namedtuple

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

# Naming conventions for output files
# Files that are depend on cluster
basenames = {
    # Cluster
    "config": ("Central configuration file", "config.yaml"),
    "exclude": ("Excluded stations, events and phases", "exclude.yaml"),
    "event": (
        "Seismic event locations, magnitudes and names",
        "events.txt",
    ),
    "station": ("Station codes and locations", "stations.txt"),
    "phase": (
        "Seismic phase arrival times and take-off angles",
        "phases.txt",
    ),
    "reference_mt": (
        "Reference moment tensor elements",
        "reference_mt.txt",
    ),
    "relative_mt": (
        "Relative moment tensor elements",
        "relative_mt.txt",
    ),
    "mt_summary": (
        "Summary of moment tensor results",
        "mt_summary.txt",
    ),
    "bootstrap_mt": (
        "Relative moment tensors from bootstrap subsampling",
        "bootstrap_mts.txt",
    ),
    "bootstrap_statistics": (
        "Diagnostic values extracted from bootstrap statistics",
        "bootstrap_stats.txt",
    ),
    "amplitude_matrix": (
        "Left hand side of the linear system",
        "amplitude_matrix.npz",
    ),
    "amplitude_data_vector": (
        "Right hand side of the linear system",
        "amplitude_data_vector.npy",
    ),
    "amplitude_scale": (
        "Event scaling vector applied to linear system",
        "amplitude_scale.npy",
    ),
    "bandpass": (
        "Optimal filter pass bands per event per station",
        "bandpass.yaml",
    ),
}


# Files that are depend on cluster, phase
basenames_phase = {
    "amplitude_observation": (
        "Amplitude ratios and misfits for event pairs/triplets on a station",
        "amplitudes.txt",
    ),
    "amplitude_summary": (
        "Amplitude observation with posterior misfit measures",
        "amplitude_summary.txt",
    ),
}


# Files that are depend on cluster, station, phase
basenames_phase_station = {
    "waveform_array": ("Phase waveforms on one station", "wvarr.npy"),
    "waveform_header": (
        "Waveform meta data. Without station and phase argument: default-hdr.yaml",
        "hdr.yaml",
    ),
    "combination": (
        "Two column list of allowed event combinations.",
        "combinations.txt",
    ),
    "mccc_time_shift": (
        "Time shifts from Multi-channel cross correlation",
        "dt_cc.txt",
    ),
    "mccc_lag_times": (
        "Pair wise lag times from Multi-channel cross correlation",
        "lag_times.txt",
    ),
    "cc_matrix": ("Cross-correlation matrix", "cc.txt"),
    "cc_vector": ("Averaged cross-correlation vector", "c.txt"),
    "pca_time_shift": ("Time shifts from principal component analysis", "dt_pca.txt"),
    "pca_objective": ("Value of principal component objective function", "phi.txt"),
}


# Suffix to add to cleaned amplitude observations
clean_amplitude_suffix = "-qced"
synthetic_amplitude_suffix = "-synthetic"


def file(
    file_id: str,
    station: str = "",
    phase: str = "",
    n_align: int | None = None,
    directory: str = "",
    suffix: str = "",
):
    """Path to a runtime file, following naming conventions

    Parameters
    ----------

    file_id:
        The file to be accessed, one of:
    PLACEHOLDER FOR FILES

        Alternatively, give a file basename ending in '.'`extension`.  The
        basename will be pre- and suffixed, and sorted into data, align, or
        amplitude folder, if 'iteration' is 0, >0, or None, respectively
    station:
        Station name, required for:
        STATION FILES

    phase:
        Seismic phase, required for:
        PHASE FILES

    n_align:
        Alignment iteration, required for:
        PHASE FILES

    directory:
        Name of the root directory
    suffix:
        String to append to the basename, before the file ending. A single
        leading '-' will prepended (but given leading '-' stripped).

    Returns
    -------
    fpath: Path
        Path to the file
    """

    folder = Path(directory)

    if phase is not None:
        if phase not in "PSps":
            raise ValueError(f"'phase' must be 'P' or 'S', not: {phase}")
        phase = phase.upper()

    # Sort in the right subfolder
    if file_id == "config" or file_id == "exclude":
        # Config and exclude in root folder
        pass

    elif file_id in ["event", "station", "phase", "reference_mt"]:
        folder /= "data"

    elif file_id in [
        "relative_mt",
        "bootstrap_mt",
        "mt_summary",
        "bootstrap_statistics",
        "amplitude_summary",
    ]:
        folder /= "result"

    elif (
        file_id.startswith("amplitude_")
        or file_id.startswith("moment_")
        or file_id.startswith("bandpass")
    ):
        folder /= "amplitude"

    elif file_id.startswith("waveform_") or file_id == "combination":
        if n_align is None or n_align == 0:
            folder /= "data"
        elif n_align > 0:
            folder /= f"align{n_align}"
        else:
            msg = f"'n_align' must be >= 0, not: '{n_align}'.\n"
            raise ValueError(msg)

    elif "." in file_id:
        if n_align is not None:
            if n_align == 0:
                folder /= "data"
            elif n_align > 0:
                folder /= f"align{n_align}"
            else:
                raise ValueError(f"'n_align' must be >= 0, not: {n_align}")
        else:
            folder /= "amplitude"

    else:
        folder /= f"align{n_align}"

    # Assemble the name
    if file_id in basenames:
        fpath = Path(basenames[file_id][1])

    elif file_id in basenames_phase:
        if not phase:
            raise ValueError("'phase' is required")
        fpath = Path(f"{phase}-{basenames_phase[file_id][1]}")

    elif file_id in basenames_phase_station:
        if not phase or not station:
            if file_id == "waveform_header":
                fpath = Path(f"default-{basenames_phase_station[file_id][1]}")
            else:
                raise ValueError("'phase' and 'station' are required")
        else:
            fpath = Path(
                f"{join_waveid(station,phase)}-{basenames_phase_station[file_id][1]}"
            )

    elif "." in file_id:
        # Custom file ID, with suffix
        if phase and not station:
            fpath = Path(f"{phase}-{file_id}")
        elif station and not phase:
            fpath = Path(f"{station}-{file_id}")
        elif station and phase:
            fpath = Path(f"{join_waveid(station, phase)}-{file_id}")
        else:
            fpath = Path(file_id)

    else:
        raise ValueError(f"Unrecognized 'file_id': {file_id}")

    if suffix:
        suffix = "-" + suffix.lstrip("-")

    path = folder / str(fpath).replace(fpath.suffix, suffix + fpath.suffix)

    logger.debug(f"File is: {path}")
    return path


def init(directory: str | Path | None = None):
    """Initialize a working directory"""

    if directory is not None:
        directory = Path(directory)
        try:
            os.mkdir(directory)
            logger.debug(f"Made directory: {directory}")
        except FileExistsError:
            logger.debug(f"{directory} exists. Continuing.")

    else:
        directory = Path()

    for subdir in ["data", "align1", "align2", "amplitude", "result"]:
        try:
            os.mkdir(directory / subdir)
            logger.debug(f"Made directory: {directory / subdir}")
        except FileExistsError:
            logger.debug(f"{directory / subdir} exists. Continuing.")

    # Write the default files
    hdrf = file("waveform_header", directory=directory)
    conff = file("config", directory=directory)
    exclf = file("exclude", directory=directory)

    try:
        Config().to_file(conff)
    except FileExistsError:
        logger.debug(f"{conff} exists. Continuing.")

    try:
        Header().to_file(hdrf)
    except FileExistsError:
        logger.debug(f"{hdrf} exists. Continuing.")

    if not exclf.exists():
        with open(exclf, "w") as fid:
            yaml.safe_dump(exclude, fid)
    else:
        logger.debug(f"{exclf} exists. Continuing.")
    logger.info(f"Working directory is complete: {directory}")


# Now fill actual file names and description into the doc
file.__doc__ = file.__doc__.replace(
    "    PLACEHOLDER FOR FILES",
    "\n".join(f"            * '{k}': {v[0]}" for k, v in basenames.items())
    + "\n"
    + "\n".join(f"            * '{k}': {v[0]}" for k, v in basenames_phase.items())
    + "\n"
    + "\n".join(
        f"            * '{k}': {v[0]}" for k, v in basenames_phase_station.items()
    ),
)

file.__doc__ = file.__doc__.replace(
    "PHASE FILES", ", ".join(f"{key}" for key in basenames_phase)
)

file.__doc__ = file.__doc__.replace(
    "STATION FILES", ", ".join(f"{key}" for key in basenames_phase_station)
)


### Prefixes


def join_phaseid(event_index: int, station: str, phase: str) -> str:
    """Join event, station and phase names to phase identifier"""
    return f"{event_index}_{station}_{phase}"


def split_phaseid(phase_id: str) -> tuple[int, str, str]:
    """Split phase identifier into eventID, station and phase"""
    event_index, station, phase = phase_id.split("_")[:3]
    return int(event_index), station, phase


def join_waveid(station: str, phase: str) -> str:
    """Join station and phase names to waveform identifier"""
    return f"{station}_{phase}"


def split_waveid(wave_id: str) -> tuple[str, str]:
    """Split waveform identifier into station and phase"""
    station, phase = wave_id.split("_")[:2]
    return station, phase


# Iterators
def iterate_waveid(stations: list[str]) -> Iterator[str]:
    """Yield P and S waveform identifiers for all stations"""
    for sta in sorted(stations):
        for pha in "PS":
            yield join_waveid(sta, pha)


def iterate_event_pair(nev: int, event_list: list[int] | range | None = None):
    """
    Yield event pairs and corresponding indices into event_list

    Parameters
    ----------
    nev:
        Total number of events  (= highest event index)
    event_list:
        Reduced set of events to iterate (Default: range(nev))

    Yields
    ------
    a, b: int
        Combinations of a and b
    ia, ib: int if `event_list` is not `None`
        Indices of a and b in `event_list`
    """

    if event_list is None:
        for a in range(nev - 1):
            for b in range(a + 1, nev):
                yield (a, b)
    else:
        for a in range(nev - 1):
            for b in range(a + 1, nev):
                ia = event_list[a]
                ib = event_list[b]
                yield (ia, ib)


def iterate_event_triplet(nev: int, event_list: list[int] | range | None = None):
    """
    Yield event triplets and corresponding indices into event_list

    Parameters
    ----------
    nev:
        Total number of events
    event_list:
        Reduced set of events to iterate (Default: range(nev))

    Yields
    ------
    a, b, c: int
        Indices of combinations of a, b, and c
    ia, ib, ic: int if `event_list` is not `None`
        Indices of a, b and c in `event_list`

    """

    if event_list is None:
        for a in range(nev - 2):
            for b in range(a + 1, nev - 1):
                for c in range(b + 1, nev):
                    yield (a, b, c)

    else:
        for a in range(nev - 2):
            for b in range(a + 1, nev - 1):
                for c in range(b + 1, nev):
                    try:
                        ia = event_list[a]
                        ib = event_list[b]
                        ic = event_list[c]
                        yield (ia, ib, ic)
                    except ValueError:
                        continue


def ijk_ccvec(ns: int) -> Generator[tuple[int, int, int]]:
    """Translate linear index n to cross correlation triplet indices ijk

    Yields all three permutations (i, j, k), (k, i, j) and (j, k i)
    """

    for i in range(ns - 2):
        for j in range(i + 1, ns - 1):
            for k in range(j + 1, ns):
                yield (i, j, k)
                yield (k, i, j)
                yield (j, k, i)


### Objects

Station = namedtuple("Station", ["north", "east", "depth", "name"])
Station.__doc__ = """Coordinates and name of one seismic station

    Attributes
    ----------
    north, east, depth: float
        Coordinates (meter)
    name: str
        Unique name used for referencing
    """

Event = namedtuple("Event", ["north", "east", "depth", "time", "mag", "name"])
Event.__doc__ = """Coordinates, time, magnitude and name of one seismic event

    Attributes
    ----------
    north, east, depth: float
        Coordinates (meter)
    time: float
        Origin time (seconds)
    mag: float
        Magnitude
    name: str
        Unique name used for referencing
    """

Phase = namedtuple("Phase", ["time", "azimuth", "plunge"])
Phase.__doc__ = """Arrival time and take-off angle of one seismic phase

    Attributes
    ----------
    time: float
        Arrival time (seconds)
    azimuth: float
        Take-off angle azimuth (degree east of North)
    plunge: float
        Take-off angle plunge (degree down from horizontal)
    """
# TODO: implement polarities
# Phase = namedtuple("Phase", ["time", "azimuth", "plunge", "polarity"])
# polarity: str
# First motion polarity ('+', '-', or 'o')

MT = namedtuple("MT", ["nn", "ee", "dd", "ne", "nd", "ed"])
MT.__doc__ = """A seismic moment tensor

    Attributes
    ----------
    nn, ee, dd, ne, nd, ed
        Components in North-East-Down coordinates (Nm)
    """

P_Amplitude_Ratio = namedtuple(
    "P_Amplitude_Ratio",
    [
        "station",
        "event_a",
        "event_b",
        "amp_ab",
        "misfit",
        "correlation",
        "sigma1",
        "sigma2",
        "highpass",
        "lowpass",
    ],
)
P_Amplitude_Ratio.__doc__ = """P-wave amplitude ratio observation

    Attributes
    ----------
    station: str
        Unique station name
    event_a, event_b: int
        Event indices
    amp_ab: float
        Amplitude ratio
    misfit: float
        Reconstruction misfit
    correlation: float
        Correlation coefficient of the reconstruction
    sigma1, sigma2: float
        First and second singular value of the seismogram decomposition
    highpass, lowpass: float
        Filter corners at which the amplitude as astimated
    """

S_Amplitude_Ratios = namedtuple(
    "S_Amplitude_Ratios",
    [
        "station",
        "event_a",
        "event_b",
        "event_c",
        "amp_abc",
        "amp_acb",
        "misfit",
        "correlation",
        "sigma1",
        "sigma2",
        "sigma3",
        "highpass",
        "lowpass",
    ],
)

S_Amplitude_Ratios.__doc__ = """S-wave amplitude ratio observations

    Attributes
    ----------
    station: str
        Unique station name
    event_a, event_b, event_c: int
        Event indices
    amp_abc: float
        Amplitude of event `b` in `a` assuming the third event is `c`
    amp_acb: float
        Amplitude of event `c` in `a` assuming the third event is `b`
    misfit: float
        Reconstruction misfit
    correlation: float
        Correlation coefficient of the reconstruction
    sigma1, sigma2, sigma3: float
        First, second and third singular value of the seismogram decomposition
    highpass, lowpass: float
        Filter corners at which the amplitude as astimated
    """


class Exclude(TypedDict, total=True):
    """
    Observations to exclude from processing
    """

    station: list[str]
    """Station names"""

    event: list[int]
    """Events by index"""

    waveform: list[str]
    """Phases ('P' or 'S') on a station, given as Waveform ID (e.g ``STA_P``)"""

    phase_manual: list[str]
    """Phase observation given as Phase ID (e.g. ``0_STA_P``) of one event on
    one station to exclude from waveform alignment"""

    phase_auto_nodata: list[str]
    """Phase observation to exclude due to absent or corrupt data."""

    phase_auto_snr: list[str]
    """Phase observation to exclude due to low signal-to-noise ratio."""

    phase_auto_cc: list[str]
    """Phase observation to exclude due to correlation coefficient."""

    phase_auto_ecn: list[str]
    """Phase observation to exclude due to low expansion coefficient norm."""


# The one exclude dictionary we are going to use.
exclude = Exclude(
    station=[],
    event=[],
    waveform=[],
    phase_manual=[],
    phase_auto_nodata=[],
    phase_auto_snr=[],
    phase_auto_cc=[],
    phase_auto_ecn=[],
)

# Attributes set in the global configuration file
_config_args_comments = {
    "event_file": (
        "str",
        """
Path to the seismic event caltalog""",
    ),
    "station_file": (
        "str",
        """
Path to the station location file, e.g. 'data/stations.txt'""",
    ),
    "phase_file": (
        "str",
        """
Path to the phase file, e.g. 'data/phases.txt'""",
    ),
    "reference_mt_file": (
        "str",
        """
Path to the reference moment tensor file, e.g. 'data/reference_mt.txt'""",
    ),
    "amplitude_suffix": (
        "str",
        """
Suffix (read/write) of amplitude files of this run""",
    ),
    "result_suffix": (
        "str",
        """
Suffix (read/write) of result files of this run""",
    ),
    "compute_synthetics": (
        "bool",
        """
Compute synthetic amplitudes for each moment tensor in 'reference_mt_file'""",
    ),
    "solve_synthetics": (
        "bool",
        """
Compute moment tensors from synthetic, not meassured amplitudes""",
    ),
    "reference_mts": (
        "list",
        """
Event indices of the reference moment tensors to use""",
    ),
    "reference_weight": (
        "float",
        """
Weight of the reference moment tensor""",
    ),
    "mt_constraint": (
        "str",
        """
Constrain the moment tensor. 'none' or 'deviatoric'""",
    ),
    "amplitude_measure": (
        "str",
        """
Method to meassure relative amplitudes. 'indirect': Estimate relative amplitude
as the ratio of principal seismogram contributions to each seismogram.
'direct': Compare each event combination seperatly.""",
    ),
    "amplitude_filter": (
        "str",
        """
Filter method to apply for amplitude measure. 'manual': Use 'highpass' and
'lowpass' of the waveform header files. 'auto': compute filter corners using the
'auto_' options""",
    ),
    "auto_lowpass_method": (
        "str",
        """
Method to estimate lowpass filter that eliminates the source time function.
'duration': Filter by 1/source duration of event magnitude. 'corner':
estimate corner frequency from stress drop of event magnitude. Requires
'auto_lowpass_stressdrop_range'""",
    ),
    "auto_lowpass_stressdrop_range": (
        "list",
        """
When estimating the lowpass frequency of an event as the corner frequency,
assume a stressdrop within this range (Pa).""",
    ),
    "auto_bandpass_snr_target": (
        "float",
        """
If supplied, include frequencies with this signal-to-noise ratio to optimal
bandpass filter. If not supplied, do not attempt to optimize passband.""",
    ),
    "min_dynamic_range": (
        "float",
        """
Minimum ratio (dB) of low- / highpass filter bandwidth in an amplitude ratio
measurement""",
    ),
    "min_amplitude_misfit": (
        "float",
        """
Minimum misfit to assign a full weight of 1. Weights are scaled lineraly from
`min_amplitude mistfit` = 1 to `max_amplitude_misfit` = `min_amplitude_weight`"
""",
    ),
    "max_amplitude_misfit": (
        "float",
        """
Maximum misfit allowed for amplitude reconstruction""",
    ),
    "min_amplitude_weight": (
        "float",
        """
Weight assigned to the maxumum amplitude misfit""",
    ),
    "max_s_sigma1": (
        "float",
        """
Maximum first normalized singular value to allow for an S-wave reconstruction. A
value of 1 indicates that S-waveform adheres to rank 1 rather than rank 2 model.
The relative amplitudes Babc and Bacb are then not lineraly independent.""",
    ),
    "max_magnitude_difference": (
        "float",
        """
Maximum difference in magnitude between two events to allow an amplitude
measurement.""",
    ),
    "max_event_distance": (
        "float",
        """
Maximum distance (m) between two events to include measurement in linear system """,
    ),
    "min_equations": (
        "int",
        """
Minimum number of equations required to constrain one moment tensor""",
    ),
    "max_gap": (
        "float",
        """
Maximum azimuthal gap allowed for one moment tensor""",
    ),
    "bootstrap_samples": (
        "int",
        """
Number of samples to draw for calculating uncertainties""",
    ),
    "ncpu": (
        "int",
        """
Number of threads to use for parallel computations""",
    ),
}


class Config:
    __doc__ = "Configuration for `relMT`\n\n"
    __doc__ += "Parameters\n"
    __doc__ += "----------\n"
    __doc__ += "\n"
    __doc__ += "".join(
        f"{key}:\n    {doc}\n" for key, (_, doc) in _config_args_comments.items()
    )
    __doc__ += "\n"
    __doc__ += "Raises\n"
    __doc__ += "------\n"
    __doc__ += "KeyError:\n    If unknown keywords are present\n"
    __doc__ += "TypeError:\n    If input value is of wrong type\n"
    __doc__ += "\n"

    # Valid arguments for this class (different for header)
    _valid_args = _config_args_comments

    def __init__(
        self,
        event_file: str | None = None,
        station_file: str | None = None,
        phase_file: str | None = None,
        reference_mt_file: str | None = None,
        amplitude_suffix: str = "",
        result_suffix: str = "",
        compute_synthetics: bool | None = None,
        solve_synthetics: bool | None = None,
        reference_mts: list[int] | None = None,
        mt_constraint: str | None = None,
        reference_weight: float | None = None,
        amplitude_measure: str | None = None,
        amplitude_filter: str | None = None,
        auto_lowpass_method: str | None = None,
        auto_lowpass_stressdrop_range: tuple[float, float] | None = None,
        auto_bandpass_snr_target: float | None = None,
        min_dynamic_range: float | None = None,
        min_amplitude_misfit: float | None = None,
        max_amplitude_misfit: float | None = None,
        min_amplitude_weight: float | None = None,
        max_s_sigma1: float | None = None,
        max_magnitude_difference: float | None = None,
        max_event_distance: float | None = None,
        min_equations: int | None = None,
        max_gap: float | None = None,
        bootstrap_samples: int | None = None,
        ncpu: int | None = None,
    ):
        for key, value in locals().items():
            if key != "self":
                self[key] = value

    def __setitem__(self, key, value):
        # Only defined keys are allowed
        if key not in self._valid_args:
            msg = f"Found unknown key: '{key}'. Ignoring."
            logger.warning(msg)
            return

        if value is None:
            # None suffixes become empty stings
            if key.endswith("_suffix"):
                value = ""
            else:
                # Don't attempt to cast not-set values
                self.__setattr__(key, value)
                return

        # If not None, get type from _config_attr_comments
        for attr in self._valid_args:
            if key == "auto_lowpass_stressdrop_range":
                value = [float(value[0]), float(value[1])]
            elif key == attr:
                typ = __builtins__[self._valid_args[attr][0]]
                # Cast input to type
                try:
                    value = typ(value)
                except ValueError:
                    msg = f"Unable to cast value '{value}' of '{key}' to type: {typ}"
                    raise TypeError(msg)

        self._check_choices(key, value)
        self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __iter__(self):
        """Yield configuration keys which are not None"""
        for key in self._valid_args:
            if self[key] is not None:
                yield key

    def __repr__(self):
        out = f"{__name__}.Config(\n"
        out += "\n".join(f"    {key}={value}," for key, value in self.items())
        out += "\n"
        out += ")"
        return out

    def __str__(self):
        out = "# relMT configuration\n"
        for key in self._valid_args:
            # Print the comment
            out += "\n# ".join(self._valid_args[key][1].split("\n"))
            out += "\n"
            out += "# (" + self._valid_args[key][0] + ")\n"

            # Print the key, value pair
            out += f"{key}: {self[key] if self[key] is not None else ''}\n"
        return out

    def __len__(self):
        """Number of set parameters"""
        return len(self.__dict__)

    def __eq__(self, other):
        """Both Config have same length and all elements are equal"""
        return len(self) == len(other) and all(
            sv == ov for sv, ov in zip(self.values(), other.values())
        )

    def _check_choices(self, key, value):
        """Check arguments that allow for a set of choices"""

        if value is None:
            return

        if key == "mt_constraint" and value not in ["none", "deviatoric"]:
            raise ValueError(
                f"Unknown 'mt_constraint': {value}. " "Must be 'none' or 'deviatoric'."
            )
        if key == "amplitude_measure" and value not in ["direct", "indirect"]:
            raise ValueError(
                f"Unknown 'amplitude_measure': {value}. "
                "Must be 'direct' or 'indirect'."
            )
        if key == "amplitude_filter" and value not in ["manual", "auto"]:
            raise ValueError(
                f"Unknown 'amplitude_filter': {value}. " "Must be 'manual' or 'auto'."
            )
        if key == "auto_lowpass_method" and value not in ["duration", "corner"]:
            raise ValueError(
                f"Unknown 'auto_lowpass_method': {value}. "
                "Must be 'duration' or 'corner'."
            )
        if key == "phase" and value not in ["P", "S"]:
            raise ValueError(f"Unknown 'phase': {value}. Must be 'P' or 'S'.")

        return

    def to_file(self, filename: str | Path, overwrite: bool = False):
        """
        Save configuration to .yaml file

        Parameters
        ----------
        filename:
            Name of the file. File ending '.yaml' will be appended if absent.
        overwrite:
            Overwrite file in case it exists

        Raises
        ------
        FileExistsError:
            When file exists and overwrite is `False`
        """

        filename = str(filename)

        if not filename.endswith(".yaml"):
            filename += ".yaml"

        if Path(filename).exists() and not overwrite:
            raise FileExistsError(
                f"Set 'overwrite=True' if you wish to overwrite: {filename}"
            )

        buf = self.__str__()
        with open(filename, "w") as fid:
            fid.write(buf)

        logger.info(f"Configuration written to: {filename}")

    def update_from_file(self, filename: str | Path, overwrite: bool = True):
        """
        Update configuration from .yaml file

        Parameters
        ----------
        filename:
            Name of the configuration file
        overwrite:
            When keyword is already present in config, overwrite its value from
            file
        """
        with open(str(filename), "r") as fid:
            buf = yaml.safe_load(fid)

        for key, value in buf.items():
            if key not in self.keys() or self[key] is None:
                self[key] = value
            elif overwrite and value is not None:
                self[key] = value

        return self

    def update(self, other: dict):
        """
        Add and, if present, replace configuration keys

        Parameters
        ----------
        other:
            Dictionary holding valid key, value pairs
        """
        for key, value in other.items():
            self[key] = value

    def keys(self):
        return [key for key in self.__dict__.keys() if self[key] is not None]

    def items(self):
        return [
            (key, value)
            for key, value in self.__dict__.items()
            if self[key] is not None
        ]

    def values(self):
        return [value for value in self.__dict__.values() if value is not None]

    def get(self, value, default):
        return_value = self.__dict__.get(value, default)
        if return_value is None and default is not None:
            # We found a None, which Config should only return if explicitly
            # asked for
            return_value = default
        return return_value

    def kwargs(self, function: Callable):
        """
        Return only the keyword arguments that are expected by function
        """

        # Arguments defined in function signature
        fun_args = inspect.signature(function).parameters.keys()

        return {key: value for key, value in self.items() if key in fun_args}


# Attributes that must be present in station header file
_header_args_comments = {
    "station": (
        "str",
        """
Station code""",
    ),
    "phase": (
        "str",
        """
Seismic phase type to consider ('P' or 'S')""",
    ),
    "variable_name": (
        "str",
        """
Optional variable name that holds the waveform array""",
    ),
    "components": (
        "str",
        """
One-character component names ordered as in the waveform array, as one string
(e.g. 'ZNE')""",
    ),
    "sampling_rate": (
        "float",
        """
Sampling rate of the seismic waveform (Hertz)""",
    ),
    "events": (
        "list",
        """
Event indices corresponding to the first dimension of the waveform array.""",
    ),
    "data_window": (
        "float",
        """
Time window symmetric about the phase pick (i.e. pick is near the central
sample) (seconds)""",
    ),
    "phase_start": (
        "float",
        """
Start of the phase window before the arrival time pick (negative seconds before
pick).""",
    ),
    "phase_end": (
        "float",
        """
End of the phase window after the arrival time pick (seconds after pick).""",
    ),
    "taper_length": (
        "float",
        """
Combined length of taper that is applied at both ends beyond the phase window.
(seconds)""",
    ),
    "highpass": (
        "float",
        """
Common high-pass filter corner of the waveform (Hertz)""",
    ),
    "lowpass": (
        "float",
        """
Common low-pass filter corner of the waveform (Hertz)""",
    ),
    "null_threshold": (
        "float",
        """
Regard absolute amplitudes at and below this value as null""",
    ),
    "min_signal_noise_ratio": (
        "float",
        """
Minimum allowed signal-to-noise ratio (dB) of signals for event exclusion""",
    ),
    "min_correlation": (
        "float",
        """
Minimum allowed absolute averaged correlation coefficient of a waveform for
event exclusion""",
    ),
    "min_expansion_coefficient_norm": (
        "float",
        """
Minimum allowed norm of the principal component expansion coefficients
contributing to the waveform reconstruction for event exclusion""",
    ),
    "combinations_from_file": (
        "bool",
        """
Read combinations from file names STATION_PHASE-combination.txt""",
    ),
}


class Header(Config):
    __doc__ = "Waveform Header for relMT\n\n"
    __doc__ += "Parameters\n"
    __doc__ += "----------\n"
    __doc__ += "\n"
    __doc__ += "".join(
        f"{key}:\n    {doc}\n" for key, (_, doc) in _header_args_comments.items()
    )
    __doc__ += "\n"
    __doc__ += "Raises\n"
    __doc__ += "------\n"
    __doc__ += "KeyError\n    If unknown keywords are present\n"
    __doc__ += "TypeError\n    If input value is of wrong type\n"
    __doc__ += "\n"

    # Valid arguments for this class (different for header)
    _valid_args = _header_args_comments

    def __init__(
        self,
        station: str | None = None,
        phase: str | None = None,
        components: str | None = None,
        variable_name: str | None = None,
        sampling_rate: float | None = None,
        events: list[int] | None = None,
        data_window: float | None = None,
        phase_start: float | None = None,
        phase_end: float | None = None,
        taper_length: float | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        null_threshold: float | None = None,
        min_signal_noise_ratio: float | None = None,
        min_correlation: float | None = None,
        min_expansion_coefficient_norm: float | None = None,
        combinations_from_file: bool | None = None,
    ):
        for key, value in locals().items():
            if key != "self":
                self[key] = value

    def __repr__(self):
        out = f"{__name__}.Header(\n"
        out += "\n".join(f"    {key}={value}," for key, value in self.items())
        out += "\n"
        out += ")"
        return out

    def __str__(self):
        out = "# relMT waveform header\n"
        for key in self._valid_args:
            # Print the comment
            out += "\n# ".join(self._valid_args[key][1].split("\n"))
            out += "\n"
            out += "# (" + self._valid_args[key][0] + ")\n"

            # Print the key, value pair
            out += f"{key}: {self[key] if self[key] is not None else ''}\n"
        return out

    def validate(self):
        """Check if all arguments are valid"""
        if (
            self["phase_start"] is not None
            and self["phase_end"] is not None
            and self["phase_start"] >= self["phase_end"]
        ):
            raise ValueError(
                f"'phase_start' ({self['phase_start']}) must be smaller than "
                f"'phase_end' ({self['phase_end']})."
            )
        if (hp := self["highpass"]) is not None and (lp := self["lowpass"]) is not None:
            if hp > lp:
                raise ValueError(
                    "'lowpass' ({lp} Hz) must be larger than 'highpass' ({hp} Hz)"
                )
        return True


# TODO: Only make an 'extra' optional dependency
def _module_hint(module_name: str) -> str:
    """Reuturn hint on how to get module working"""

    dep = {
        "matplotlib": "plot",
        "multitaper": "spec",
        "obspy": "obspy",
        "pyrocko": "pyrocko",
        "utm": "geo",
    }

    msg = f"Could not import {module_name}.\n"
    msg += f"Please install relMT with optional {dep} dependencies:\n"
    msg += f"pip install .[{dep}]"

    return msg


_all_args_comments = {**_config_args_comments, **_header_args_comments}


def _doc_config_args(func):
    """Insert documentation of common arguments to function"""
    fun_args = inspect.getfullargspec(func).args

    @wraps(func)
    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    # Index of 'Returns' block of documentation
    iret = func.__doc__.find("    Returns")

    # Copy function until end of 'Parameter' block
    wrap.__doc__ = func.__doc__[: iret - 1]

    # Search function arguments for common ones
    for fun_arg in fun_args:
        if fun_arg in _all_args_comments:
            # Insert common argument documentation
            _, doc = _all_args_comments[fun_arg]
            wrap.__doc__ += f"    {fun_arg}:\n        {doc}\n"

    # copy remainder of doc
    wrap.__doc__ += func.__doc__[iret - 1 :]

    return wrap
