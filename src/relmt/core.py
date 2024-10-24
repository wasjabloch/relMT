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

import logging
from relmt import utils
import yaml

from collections import namedtuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(utils.logsh)

_config_attr_comments = {
    "sampling_rate": ("float", "Sampling rate of the seismic waveform (Hertz)"),
    "data_window": (
        "float",
        "Symmetric time window around the initiallly cut out phase data (seconds)",
    ),
    "phase_start": (
        "float",
        "Start of the phase window before the arrival time pick (negative seconds before pick).",
    ),
    "phase_end": (
        "float",
        "End of the phase window after the arrival time pick (seconds after pick).",
    ),
    "taper_length": (
        "float",
        "Combined length of taper that is applied at both ends beyond the phase window. (seconds)",
    ),
    "highpass": ("float", "High-pass filter corner (Hertz)"),
    "lowpass": ("float", "Low-pass filter corner (Hertz)"),
    "exclude_events": ("list", "List of event indices to exclude from processing"),
}


class Config:
    __doc__ = "Configuration for relMT\n\n"
    __doc__ += "Parameters\n"
    __doc__ += "----------\n"
    __doc__ += "\n"
    __doc__ += "".join(
        f"{key}: ({typ})\n    {doc}\n"
        for key, (typ, doc) in _config_attr_comments.items()
    )
    __doc__ += "\n"
    __doc__ += "Raises\n"
    __doc__ += "------\n"
    __doc__ += "KeyError if unknown keywords are present\n"
    __doc__ += "TypeError if input value is of wrong type\n"
    __doc__ += "\n"

    def __init__(
        self,
        sampling_rate: float | None = None,
        data_window: float | None = None,
        phase_start: float | None = None,
        phase_end: float | None = None,
        taper_length: float | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        exclude_events: list[int] | None = None,
    ):
        for key, value in locals().items():
            if key != "self" and key is not None:
                self[key] = value

    def __setitem__(self, key, value):
        # Only defined keys are allowed
        if key not in _config_attr_comments:
            msg = "Key must be one of: " + ", ".join(_config_attr_comments.keys())
            raise KeyError(msg)

        # None value has NoneType
        if value is None:
            self.__setattr__(key, value)
            return

        # If not None, get type from _config_slot_comments
        for attr in _config_attr_comments:
            typ = __builtins__[_config_attr_comments[attr][0].strip("| None")]
            if key == attr:
                # Cast input to type
                try:
                    value = typ(value)
                except ValueError:
                    msg = f"Unable to cast value '{value}' of '{key}' to type: {typ}"
                    raise TypeError(msg)
        self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __repr__(self):
        out = f"{__name__}.Config(\n"
        out += "\n".join(f"    {key}={value}," for key, value in self.items())
        out += "\n"
        out += ")"
        return out

    def __str__(self):
        out = "# relMT configuration\n"
        for key in _config_attr_comments:
            # Print the comment
            out += "\n"
            out += "# " + _config_attr_comments[key][1] + "\n"
            out += "# (" + _config_attr_comments[key][0] + ")\n"

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

    def to_file(self, filename):
        """
        Save configutation to .yaml file

        Parameters
        ----------
        filename: str | None
            Name of the file. File ending '.yaml' will be appended if absent.
        """

        if not filename.endswith(".yaml"):
            filename += ".yaml"

        buf = self.__str__()
        with open(filename, "w") as fid:
            fid.write(buf)

        logger.info(f"Configuration written to: {filename}")

    def from_file(self, filename: str, update_from_file=True):
        """
        Read a configuration from .yaml file

        Parameters
        ----------
        filename: str
            Name of the configuration file
        update_from_file: bool
            When keyword is already present in config, overwrite its value from
            file
        """
        with open(filename, "r") as fid:
            buf = yaml.safe_load(fid)

        for key, value in buf.items():
            if key not in self.keys() or self[key] is None:
                self[key] = value
            elif update_from_file and value is not None:
                self[key] = value

        return self

    def update(self, other: dict):
        """
        Add and, if present, replace configuration keys

        Parameters
        ----------
        other: dict
            Dictionary holding valid key, value pairs
        """
        for key, value in other.items():
            self[key] = value

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()


Coordinate = namedtuple("Coordinate", ["north", "east", "depth"])

Event = namedtuple("Event", ["north", "east", "depth", "time", "mag", "name"])

Phase = namedtuple("Phase", ["time", "azimuth", "inclination"])
