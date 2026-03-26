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
Test the main functions
"""

from relmt import main, core, default, io, amp
from pathlib import Path
import numpy as np
import shutil
import pytest


try:
    # This is what IPython likes
    from . import conftest
except ImportError:
    # This is what Pytest likes
    import conftest


def test_main_amplitdue(synthetic_aligned_waveforms):
    """
    Test the main amplitude function
    """

    # For S-wave, an accuracy of 1% can only be archived when excluding at a
    # sigma_1 threshold of 0.8. This is because for many waveforms, Babc and
    # Bacb are not lineraily independent and so can take any value, depending on
    # the exact initialization of the problem. As waveform data is filtered
    # before measuring the amplitdues, the observed and synthetic data are
    # initialized slighly different, leading to drastically different outcomes
    # for both realizations of the (almost) same data. Increasing s1_thrs to
    # 0.85 requires increasing srel to ~3.

    srel = 0.01
    s1_thres = 0.8

    # Get the synthetic aligned waveforms
    mtd = {
        n: momt
        for n, momt in enumerate(
            [
                conftest.mt_rl_strike_slip_north_south_m0(1),
                conftest.mt_rl_strike_slip_north_south_m0(2),
                conftest.mt_ll_strike_slip_north_south_m0(3),
                conftest.mt_ll_strike_slip_north_south_m0(1),
                conftest.mt_normal_dip30_north_m0(2),
                conftest.mt_normal_dip30_east_m0(3),
                conftest.mt_reverse_dip30_east_m0(1),
                conftest.mt_reverse_dip30_north_m0(2),
            ]
        )
    }

    # Events in a 10m circle at 1000m depth
    evl = conftest.event_circle(len(mtd), 10, 1000)

    # 10 Events at the surface, at 1000m distance
    std = conftest.station_circle(50)

    phd = conftest.phases(std, evl)

    # Temporary path, array dictionary, header dictionary
    path, arrd, hdrd = synthetic_aligned_waveforms(mtd, evl, std, phd, 0.0)

    config = core.Config(**default.config)
    config["event_file"] = str(path / "data" / "events.txt")
    config["station_file"] = str(path / "data" / "stations.txt")
    config["amplitude_filter"] = "manual"
    config["amplitude_measure"] = "indirect"

    main.amplitude_entry(config, path, 0)

    # Read the created amplitude observation
    pamps = io.read_amplitudes(
        core.file("amplitude_observation", phase="P", directory=path), "P"
    )
    samps = io.read_amplitudes(
        core.file("amplitude_observation", phase="S", directory=path), "S"
    )

    samps = [amp for amp in samps if amp.sigma1 < s1_thres]

    # Read out the observation indices in order
    p_pairs = [(amp.station, amp.event_a, amp.event_b) for amp in pamps]
    s_triplets = [(amp.station, amp.event_a, amp.event_b, amp.event_c) for amp in samps]

    # Create synthetic data in the same order
    p_syn, p_sig = amp.synthetic_p(mtd, evl, std, phd, p_pairs)

    s_syn, order, s_sig = amp.synthetic_s(
        mtd, evl, std, phd, s_triplets, keep_order=True
    )

    # S-wave degenerecy may as well occurr in the synthetic data.
    s_syn = s_syn[s_sig[:, 0] < s1_thres]

    samps = [amp for n, amp in enumerate(samps) if s_sig[n, 0] < s1_thres]
    s_sig = s_sig[s_sig[:, 0] < s1_thres]

    # Read out the amplitude observation
    p_obs = [amp.amp_ab for amp in pamps]
    s_obs0 = [amp.amp_abc for amp in samps]
    s_obs1 = [amp.amp_acb for amp in samps]

    # Check if they agree
    assert (order == [0, 1, 2]).all()

    # Uncertainty due to filter applied in the oversvation
    assert pytest.approx(p_syn, rel=0.001) == p_obs

    # Large uncertainties due to unhandeld degenerecies
    assert pytest.approx(s_syn[:, 0], rel=srel) == s_obs0
    assert pytest.approx(s_syn[:, 1], rel=srel) == s_obs1


# Code below implemented with help of Codex GPT-5.4.
#
# Initial prompt:
# in tests/test_main.py, please implement a full set of tests fore each of the
# *_entry functions in src/relmt/main.py . When data input is needed, please use
# the minimal raw data example provided in the directory tests/data/muji-mini.

MUJI_MINI = Path(__file__).parent / "data" / "muji-mini"


class DummyFigure:
    def __init__(self):
        self.saved = []
        self.title = None

    def savefig(self, filename):
        filename = Path(filename)
        filename.write_text("dummy figure")
        self.saved.append(filename)

    def suptitle(self, title):
        self.title = title


def _make_config(project: Path) -> core.Config:
    config = core.Config(**dict(default.config.items()))
    config.update(
        {
            "event_file": str(project / "data" / "events.txt"),
            "station_file": str(project / "data" / "stations.txt"),
            "phase_file": str(project / "data" / "phases.txt"),
            "reference_mt_file": str(project / "data" / "reference_mts.txt"),
            "reference_mts": [7508],
            "reference_weight": 1000.0,
            "amplitude_suffix": "testamp",
            "admit_suffix": "testadm",
            "result_suffix": "testres",
            "amplitude_filter": "manual",
            "amplitude_measure": "indirect",
            "min_equations": 1,
            "max_gap": 360.0,
            "max_s_sigma1": 1.1,
            "max_amplitude_misfit": 1.0e9,
            "max_s_amplitude_misfit": 1.0e9,
            "max_magnitude_difference": 1.0e9,
            "max_event_distance": 1.0e9,
            "max_s_equations": int(1e6),
            "bootstrap_samples": 0,
            "ncpu": 1,
            "keep_events": [7508],
        }
    )
    return config


def _save_path(fig: DummyFigure) -> Path:
    assert fig.saved
    return fig.saved[-1]


def _run_amplitude_admit_solve(config: core.Config, project: Path) -> None:
    main.amplitude_entry(config, project, 0, overwrite=True)
    main.admit_entry(config, project)
    main.solve_entry(config, project, do_predict=False, iteration=0)


@pytest.fixture
def muji_mini_project(tmp_path) -> Path:
    project = tmp_path / "muji-mini"
    core.init(project)
    shutil.copytree(MUJI_MINI, project, dirs_exist_ok=True)
    return project


@pytest.fixture
def muji_config(muji_mini_project: Path) -> core.Config:
    return _make_config(muji_mini_project)


def test_align_entry_writes_aligned_waveforms(
    muji_mini_project, muji_config, monkeypatch
):
    monkeypatch.chdir(muji_mini_project)

    for do in [(True, True), (True, False), (False, True)]:
        do_mccc, do_pca = do

        main.align_entry(
            muji_config,
            muji_mini_project,
            iteration=0,
            do_mccc=do_mccc,
            do_pca=do_pca,
            overwrite=True,
        )

        outfiles = list((muji_mini_project / "align1").glob("*-wvarr.npy"))
        assert outfiles


def test_exclude_entry_marks_zero_trace_as_no_data(
    muji_mini_project, muji_config, monkeypatch
):
    monkeypatch.chdir(muji_mini_project)

    arrf = muji_mini_project / "data" / "EP03_P-wvarr.npy"
    hdrf = muji_mini_project / "data" / "EP03_P-hdr.yaml"

    arr = np.load(arrf)
    hdr = io.read_header(hdrf)
    arr[0, :, :] = 0.0
    np.save(arrf, arr)

    main.exclude_entry(
        muji_config,
        iteration=0,
        overwrite=True,
        directory=muji_mini_project,
        do_nodata=True,
    )

    excl = io.read_exclude_file(muji_mini_project / "exclude.yaml")
    phaseid = core.join_phaseid(hdr["events_"][0], "EP03", "P")
    assert phaseid in excl["phase_auto_nodata"]


def test_amplitude_entry_writes_observation_files(muji_mini_project, muji_config):
    main.amplitude_entry(muji_config, muji_mini_project, 0, overwrite=True)

    pampf = core.file(
        "amplitude_observation",
        phase="P",
        directory=muji_mini_project,
        suffix=muji_config["amplitude_suffix"],
    )
    sampf = core.file(
        "amplitude_observation",
        phase="S",
        directory=muji_mini_project,
        suffix=muji_config["amplitude_suffix"],
    )

    assert len(io.read_amplitudes(pampf, "P")) > 0
    assert len(io.read_amplitudes(sampf, "S")) > 0


@pytest.mark.parametrize(
    (
        "auto_lowpass_method",
        "fixed_lowpass",
        "stressdrop_range",
        "snr_target",
        "expected_lowpass",
    ),
    [
        ("duration", None, [1.0e6, 1.0e8], None, None),
        ("corner", None, [2.0e5, 4.0e7], None, None),
        ("fixed", 1.75, [1.0e6, 1.0e6], 0.0, 1.75),
    ],
)
def test_amplitude_entry_passes_auto_filter_config(
    muji_mini_project,
    muji_config,
    monkeypatch,
    auto_lowpass_method,
    fixed_lowpass,
    stressdrop_range,
    snr_target,
    expected_lowpass,
):
    monkeypatch.chdir(muji_mini_project)

    muji_config.update(
        {
            "amplitude_filter": "auto",
            "amplitude_suffix": f"auto-{auto_lowpass_method}",
            "auto_lowpass_method": auto_lowpass_method,
            "fixed_lowpass": fixed_lowpass,
            "auto_lowpass_stressdrop_range": stressdrop_range,
            "auto_bandpass_snr_target": snr_target,
        }
    )

    main.amplitude_entry(muji_config, muji_mini_project, 0, overwrite=True)

    bandpassf = core.file(
        "bandpass",
        directory=muji_mini_project,
        suffix=muji_config["amplitude_suffix"],
    )
    assert bandpassf.exists()

    bandpassd = io.read_yaml(bandpassf)
    assert bandpassd
    assert set(bandpassd) == set(core.iterate_waveid(muji_mini_project, 0))

    for event_bandpass in bandpassd.values():
        assert event_bandpass
        for corners in event_bandpass.values():
            assert len(corners) == 2
            assert np.isfinite(corners).all()
            assert 0.0 < corners[0]
            assert 0.0 < corners[1]
            if expected_lowpass is not None:
                assert corners[1] <= expected_lowpass + 1.0e-6

    pampf = core.file(
        "amplitude_observation",
        phase="P",
        directory=muji_mini_project,
        suffix=muji_config["amplitude_suffix"],
    )
    sampf = core.file(
        "amplitude_observation",
        phase="S",
        directory=muji_mini_project,
        suffix=muji_config["amplitude_suffix"],
    )
    assert len(io.read_amplitudes(pampf, "P")) > 0
    assert len(io.read_amplitudes(sampf, "S")) > 0


def test_admit_entry_writes_admitted_observations(muji_mini_project, muji_config):
    main.amplitude_entry(muji_config, muji_mini_project, 0, overwrite=True)
    main.admit_entry(muji_config, muji_mini_project)

    pampf = core.file(
        "amplitude_observation",
        phase="P",
        directory=muji_mini_project,
        suffix=f"{muji_config['amplitude_suffix']}-{muji_config['admit_suffix']}",
    )
    sampf = core.file(
        "amplitude_observation",
        phase="S",
        directory=muji_mini_project,
        suffix=f"{muji_config['amplitude_suffix']}-{muji_config['admit_suffix']}",
    )

    assert len(io.read_amplitudes(pampf, "P")) > 0
    assert len(io.read_amplitudes(sampf, "S")) > 0


def test_solve_entry_writes_result_files(muji_mini_project, muji_config):
    _run_amplitude_admit_solve(muji_config, muji_mini_project)

    mtsuf = (
        f"{muji_config['amplitude_suffix']}-{muji_config['admit_suffix']}"
        f"-{muji_config['result_suffix']}"
    )
    mtfile = core.file("relative_mt", directory=muji_mini_project, suffix=mtsuf)
    sumfile = core.file("mt_summary", directory=muji_mini_project, suffix=mtsuf)

    assert len(io.read_mt_table(mtfile)) > 0
    assert sumfile.exists()


def test_plot_alignment_entry_saves_figure(muji_mini_project, muji_config, monkeypatch):
    fig = DummyFigure()
    call = {}

    def fake_alignment(*args):
        call["args"] = args
        return fig, None

    monkeypatch.setattr(main.plot, "alignment", fake_alignment)

    saveas = muji_mini_project / "alignment.png"
    main.plot_alignment_entry(
        muji_mini_project / "data" / "EP03_P-wvarr.npy",
        config=muji_config,
        do_exclude=True,
        sort="name",
        highlight_events=[7508],
        cc_method="calculate",
        saveas=saveas,
        confirm=False,
    )

    assert _save_path(fig) == saveas
    assert len(call["args"][6]) > 0


def test_plot_spectra_entry_saves_figure(muji_mini_project, monkeypatch):
    fig = DummyFigure()
    call = {}

    def fake_spectra(*args):
        call["args"] = args
        return fig, None

    monkeypatch.setattr(main.plot, "spectra", fake_spectra)

    saveas = muji_mini_project / "spectra.png"
    main.plot_spectra_entry(
        muji_mini_project / "data" / "EP03_P-wvarr.npy",
        highlight=[7508],
        integrate=True,
        saveas=saveas,
    )

    assert _save_path(fig) == saveas
    assert call["args"][3] == [7508]


def test_plot_connections_entry_saves_figure(muji_mini_project, muji_config):

    # Make amplitude observations
    main.amplitude_entry(muji_config, muji_mini_project, 0, overwrite=True)

    # Read them
    pampf = core.file(
        "amplitude_observation",
        phase="P",
        directory=muji_mini_project,
        suffix=muji_config["amplitude_suffix"],
    )
    sampf = core.file(
        "amplitude_observation",
        phase="S",
        directory=muji_mini_project,
        suffix=muji_config["amplitude_suffix"],
    )

    # Plot and save
    saveas = muji_mini_project / "amplitude-connections.png"
    main.plot_connections_entry(
        pampf,
        sfile=sampf,
        highlight=[7508],
        saveas=saveas,
    )

    # Test if a figure was written
    assert saveas.exists()


def test_plot_mt_entry_saves_figure(muji_mini_project, muji_config, monkeypatch):
    fig = DummyFigure()
    call = {}

    def fake_mt_matrix(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return fig, None

    monkeypatch.setattr(main.plot, "mt_matrix", fake_mt_matrix)

    saveas = muji_mini_project / "mt.png"
    main.plot_mt_entry(
        muji_mini_project / "data" / "reference_mts.txt",
        muji_config,
        highlight=[7508],
        overlay_dc_at=0.5,
        sort_by="number",
        color_by="mag",
        saveas=saveas,
    )

    assert _save_path(fig) == saveas
    assert call["args"][1] == [7508]
    assert call["kwargs"]["overlay_dc_at"] == 0.5
