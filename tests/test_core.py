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
"""
Test the in- and output functions
"""

from relmt import core
import tempfile

import pytest


def test_config_init_empty():
    # Test if an empty config is produced correctly
    config = core.Config()
    assert all(value is None for value in config.values())


def test_config_init_floats():
    # Test if floats are handled correctly
    float_dict = {
        key: 1.0
        for key, value in core._config_attr_comments.items()
        if value[0] == "float"
    }
    config = core.Config(**float_dict)
    assert all(value == 1.0 for value in config.values() if value is not None)


def test_config_init_lists():
    # Test if lists are handled correctly
    list_dict = {
        key: []
        for key, value in core._config_attr_comments.items()
        if value[0] == "list"
    }
    config = core.Config(**list_dict)
    assert all(value == [] for value in config.values() if value is not None)


def test_config_init_type_error():
    # Test if an arror is raised if given wrong type
    float_dict = {
        key: "a"
        for key, value in core._config_attr_comments.items()
        if value[0] == "float"
    }
    with pytest.raises(TypeError):
        _ = core.Config(**float_dict)

def test_config_init_key_error():
    # Test if an arror is raised if given wrong key
    config = core.Config()
    with pytest.raises(KeyError):
        config["foo"] = "bar"


def test_config_to_from_file():
    filename = "myconfig"
    filename2 = "myconfig.yaml"
    confdict = {
        key: 1.0
        for key, value in core._config_attr_comments.items()
        if value[0] == "float"
    }
    confdict.update(
        {
            key: []
            for key, value in core._config_attr_comments.items()
            if value[0] == "list"
        }
    )
    config1 = core.Config(**confdict)
    with tempfile.TemporaryDirectory() as tempdir:
        config1.to_file(filename=str(tempdir + filename))
        config2 = core.Config().from_file(str(tempdir + filename2))
    assert config1 == config2

def test_config_update_from_file():
    # Test if parameters are updated from file on default
    filename = "myconfig.yaml"
    config1 = core.Config(sampling_rate=1, data_window=1)
    with tempfile.TemporaryDirectory() as tempdir:
        config1.to_file(filename=str(tempdir + filename))
        config2 = core.Config(data_window=2, taper_length=2).from_file(
            str(tempdir + filename)
        )
    assert config2["sampling_rate"] == 1.0
    assert config2["data_window"] == 1.0
    assert config2["taper_length"] == 2.0

def test_config_update_not_from_file():
    # Test if parameters kept in object with option
    filename = "myconfig.yaml"
    config1 = core.Config(sampling_rate=1, data_window=1)
    with tempfile.TemporaryDirectory() as tempdir:
        config1.to_file(filename=str(tempdir + filename))
        config2 = core.Config(data_window=2, taper_length=2).from_file(
            str(tempdir + filename), update_from_file=False
        )
    assert config2["sampling_rate"] == 1.0
    assert config2["data_window"] == 2.0
    assert config2["taper_length"] == 2.0
