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

from relmt import core, default
from pathlib import Path
import tempfile

import pytest


def test_config_init_empty():
    # Test if an empty config is produced correctly
    config = core.Config()
    assert all(value is None or value == "" for value in config.values())


def test_config_init_floats():
    # Test if floats are handled correctly
    float_dict = {
        key: 1.0
        for key, value in core._config_args_comments.items()
        if value[0] == "float"
    }
    config = core.Config(**float_dict)
    assert all(config[key] == 1.0 for key in float_dict)


def test_config_init_lists():
    # Test if lists are handled correctly
    list_dict = {
        key: [0, 1]
        for key, value in core._config_args_comments.items()
        if value[0] == "list"
    }
    config = core.Config(**list_dict)
    assert all(config[key] == [0, 1] for key in list_dict)


def test_config_init_type_error():
    # Test if an arror is raised if given wrong type
    float_dict = {
        key: "a"
        for key, value in core._config_args_comments.items()
        if value[0] == "float"
    }
    with pytest.raises(TypeError):
        _ = core.Config(**float_dict)


def test_config_to_from_file():
    filename = "myconfig"
    filename2 = "myconfig.yaml"
    confdict = {
        key: 1.0
        for key, value in core._config_args_comments.items()
        if value[0] == "float"
    }
    confdict.update(
        {
            key: [0, 1]
            for key, value in core._config_args_comments.items()
            if value[0] == "list"
        }
    )
    config1 = core.Config(**confdict)
    with tempfile.TemporaryDirectory() as tempdir:
        config1.to_file(filename=str(tempdir + filename))
        config2 = core.Config().update_from_file(str(tempdir + filename2))
    assert config1 == config2


def test_config_update_from_file():
    # Test if parameters are updated from file on default
    filename = "myconfig.yaml"
    config1 = core.Config(reference_mts=[1], bootstrap_samples=1)
    with tempfile.TemporaryDirectory() as tempdir:
        config1.to_file(filename=str(tempdir + filename))
        config2 = core.Config(bootstrap_samples=2, reference_weight=2).update_from_file(
            str(tempdir + filename)
        )
    assert config2["reference_mts"] == [1]
    assert config2["bootstrap_samples"] == 1
    assert config2["reference_weight"] == 2.0


def test_config_update_not_from_file():
    # Test if parameters kept in object with option
    filename = "myconfig.yaml"
    config1 = core.Config(reference_mts=[1], bootstrap_samples=1)
    with tempfile.TemporaryDirectory() as tempdir:
        config1.to_file(filename=str(tempdir + filename))
        config2 = core.Config(bootstrap_samples=2, reference_weight=2).update_from_file(
            str(tempdir + filename), overwrite=False
        )

        # Try to write the file again...
        with pytest.raises(FileExistsError):
            config1.to_file(filename=str(tempdir + filename))

    assert config2["reference_mts"] == [1]
    assert config2["bootstrap_samples"] == 2
    assert config2["reference_weight"] == 2.0


def test_config_unpack():
    # Test if config object is upacked correctly
    keys = ["max_amplitude_misfit", "bootstrap_samples"]
    conf_dict = {**core.Config(max_amplitude_misfit=99.0, bootstrap_samples=99)}
    assert all(conf_dict[key] == 99.0 for key in keys)


def test_config_test_sanity():
    # Test if arguments with an ill choice are raise
    choice_args = [
        "mt_constraint",
        "amplitude_measure",
        "amplitude_filter",
        "auto_lowpass_method",
    ]

    for arg in choice_args:
        with pytest.raises(ValueError):
            core.Config()[arg] = "invalid"


def test_config_file():
    # Test if config file directories are returned correctly
    assert core.file("config") == Path("config.yaml")


def test_config_iter():
    # Test if config iterates correctly
    given = dict(
        max_amplitude_misfit=99.0,
        bootstrap_samples=99.0,
        amplitude_suffix="amp",
        result_suffix="res",
        qc_suffix="qc",
    )
    notgiven = [
        key for key in core._config_args_comments.keys() if key not in given.keys()
    ]
    conf = core.Config(**given)
    for key in conf:
        assert key in given.keys()
        assert key not in notgiven


def test_config_repr():
    given = dict(
        max_amplitude_misfit=99.0,
        bootstrap_samples=99.0,
        amplitude_suffix="amp",
        result_suffix="res",
        qc_suffix="qc",
    )
    notgiven = [
        key for key in core._config_args_comments.keys() if key not in given.keys()
    ]
    conf = core.Config(**given)
    for key in given:
        assert key in conf.__repr__()
    for key in notgiven:
        assert key not in conf.__repr__()


def test_config_update():
    give = dict(max_amplitude_misfit=99.0, bootstrap_samples=99.0)
    conf = core.Config()
    conf.update(give)
    for key in give:
        assert key in conf.keys()


def test_config_get():
    conf = core.Config(max_amplitude_misfit=99.0, bootstrap_samples=99.0)
    assert conf.get("max_amplitude_misfit", None) == 99.0
    assert conf.get("ncpu", None) == None
    assert conf.get("ncpu", 1) == 1

    # Test the case, where a None was set, but we want a value
    conf["ncpu"] = None
    assert conf.get("ncpu", 1) == 1


def test_config_kwargs():
    # Assert that only arguments needed for function are returned
    def myfun(max_amplitude_misfit):
        return

    conf = core.Config(max_amplitude_misfit=99.0, bootstrap_samples=99.0)
    assert conf.kwargs(myfun) == dict(max_amplitude_misfit=99.0)
    assert "bootstrap_samples" not in conf.kwargs(myfun)


def test_join_split_waveid():
    keys = ("A", "P")
    wvid = core.join_waveid(*keys)
    assert keys == core.split_waveid(wvid)


def test_file_basenames():
    # Test if files requiring no argument are handeled correctly

    # Get all keywords that require no argument
    for arg in core.basenames:
        ans = core.file(arg)
        assert isinstance(ans, Path)
        assert str(ans).endswith(core.basenames[arg][1])


def test_file_basenames_phase():
    # Test if files requiring a phase argument are handeled correctly

    # Get all keywords that require a phase argument
    for arg in core.basenames_phase:
        ans = core.file(arg, phase="P")
        assert isinstance(ans, Path)
        assert str(ans).endswith(core.basenames_phase[arg][1])

        # Make sure an error is raised when no or a wrong phase is given
        with pytest.raises(ValueError):
            ans = core.file(arg)

        with pytest.raises(ValueError):
            ans = core.file(arg, phase="X")


def test_file_basenames_phase_station():
    # Test if files requiring a phase and station argument are handeled correctly

    # Get all keywords that require a phase and station argument
    for arg in core.basenames_phase_station:
        ans = core.file(arg, phase="P", station="STA1")
        assert isinstance(ans, Path)
        assert str(ans).endswith(core.basenames_phase_station[arg][1])

        # Waveform header should produce a default without arguments
        if arg == "waveform_header":
            ans = core.file(arg)
            str(ans) == "default-hdr.yaml"
            continue

        # Make sure an error is raised when no or a wrong phase is given
        with pytest.raises(ValueError):
            _ = core.file(arg, phase="P")

        with pytest.raises(ValueError):
            _ = core.file(arg, station="STA1")

        with pytest.raises(ValueError):
            _ = core.file(arg, phase="X", station="STA1")

        with pytest.raises(ValueError):
            _ = core.file(arg)


def test_file_n_align():
    # Test if the n_align keyword is handled correctly
    ans = core.file("waveform_array", phase="P", station="STA1")
    assert str(ans).startswith("data")

    ans = core.file("waveform_array", phase="P", station="STA1", n_align=0)
    assert str(ans).startswith("data")

    ans = core.file("waveform_array", phase="P", station="STA1", n_align=1)
    assert str(ans).startswith("align1")


def test_file_own_filename():
    # Test if the n_align keyword is handled correctly
    ans = core.file("myfile.dat", phase="P", station="STA1")
    assert str(ans).endswith("STA1_P-myfile.dat")

    ans = core.file("myfile.dat", phase="P")
    assert str(ans).endswith("P-myfile.dat")

    ans = core.file("myfile.dat", station="STA1")
    assert str(ans).endswith("STA1-myfile.dat")

    ans = core.file("myfile.dat", n_align=0)
    assert str(ans).startswith("data")

    ans = core.file("myfile.dat", n_align=1)
    assert str(ans).startswith("align1")

    ans = core.file("myfile.dat", n_align=None)
    assert str(ans).startswith("amplitude")


def test_file_wrong_keyword():
    with pytest.raises(ValueError):
        _ = core.file("invalid")

    with pytest.raises(ValueError):
        _ = core.file("waveform_array", n_align=-1)

    with pytest.raises(ValueError):
        _ = core.file("myfile.dat", n_align=-1)


def test_file_suffix():
    ans = core.file("myfile.dat", suffix="mysuffix")
    assert str(ans).endswith("myfile-mysuffix.dat")


def test_iterate_waveid():
    wvids = list(core.iterate_waveid(["STA1"]))
    assert pytest.approx(wvids) == ["STA1_P", "STA1_S"]


def test_iterate_event_pair():
    pairs = list(core.iterate_event_pair(3))
    assert pytest.approx(pairs) == [(0, 1), (0, 2), (1, 2)]

    pairs = list(core.iterate_event_pair(3, event_list=[0, 2, 4]))
    assert pytest.approx(pairs) == [(0, 2), (0, 4), (2, 4)]


def test_iterate_event_triplet():
    triplets = list(core.iterate_event_triplet(4))
    assert pytest.approx(triplets) == [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]

    triplets = list(core.iterate_event_triplet(4, event_list=[0, 1, 3, 4]))
    assert pytest.approx(triplets) == [
        (0, 1, 3),
        (0, 1, 4),
        (0, 3, 4),
        (1, 3, 4),
    ]


def test_header():
    # Set up a header and try to access all agruments
    hdrkws = dict(
        station="STA1",
        phase="P",
        events_=[0, 1, 2],
        components="NEZ",
        sampling_rate=100.0,
        data_window=12.0,
        phase_start=0.0,
        phase_end=1.0,
        highpass=0.5,
        lowpass=2,
    )
    hdr = core.Header(**hdrkws)
    for kw in hdrkws:
        assert hdrkws[kw] == hdr[kw]


def test_default_config():
    dconf = default.config

    # Assert all keys are present
    for key in core._config_args_comments:
        assert dconf[key] is not None


def test_default_header():
    dhead = default.header

    # Assert all keys are present
    for key in core._header_args_comments:
        assert dhead[key] is not None


def test_default_exclude():
    dex = core.exclude

    # Default exclude file should be all empty
    for key in dex:
        assert dex[key] == []


def test_doc_config_args():
    @core._doc_config_args
    def myfun(sampling_rate=None):
        """
        Example function that does not need to document the argument explicitly

        Parameters
        ----------

        Returns
        -------
        """
        return sampling_rate

    assert "sampling_rate" in myfun.__doc__

    # Try to call the function
    assert myfun() is None
