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
Test the linear system functions
"""

from relmt import ls, core, mt, amp
from scipy.sparse import coo_matrix
import numpy as np

import pytest

try:
    # This is what IPython likes
    from . import conftest
except ImportError:
    # This is what Pytest likes
    import conftest

r2 = np.sqrt(2)


def test_gamma():

    # Coordinate system is North-East-Down
    # Azimuth is degree east of north
    # Plunge is degree down from horizontal

    # Vector must have unit length
    for inc in range(-90, 90, 10):
        for azi in range(0, 360, 30):
            assert pytest.approx(1.0) == np.sum(ls.gamma(azi, inc) ** 2)

    # Some coordinate conventions
    assert pytest.approx((1.0, 0.0, 0.0)) == ls.gamma(0.0, 0.0)
    assert pytest.approx((0.0, 1.0, 0.0)) == ls.gamma(90.0, 0.0)
    assert pytest.approx((0.0, -1.0, 0.0)) == ls.gamma(-90.0, 0.0)
    assert pytest.approx((0.0, -1.0, 0.0)) == ls.gamma(270.0, 0.0)
    assert pytest.approx((-1.0, 0.0, 0.0)) == ls.gamma(180.0, 0.0)
    assert pytest.approx((0.0, 0.0, 1.0)) == ls.gamma(0, 90.0)
    assert pytest.approx((0.0, 0.0, 1.0)) == ls.gamma(90, 90.0)
    assert pytest.approx((0.0, 0.0, -1.0)) == ls.gamma(270, -90.0)
    assert pytest.approx((0.5, np.sin(60.0 * np.pi / 180), 0)) == ls.gamma(
        60.0,
        0,
    )
    assert pytest.approx((0, np.cos(30.0 * np.pi / 180), -0.5)) == ls.gamma(90.0, -30)


def test_directional_coefficient_general_p():
    # Calculated looking in Plourde & Bostock (2019) Eq. A2, not in the code
    assert pytest.approx((1, 0, 0, 0, 0, 0)) == ls.directional_coefficient_general_p(
        [1, 0, 0]
    )
    assert pytest.approx((0, 1, 0, 0, 0, 0)) == ls.directional_coefficient_general_p(
        [0, 1, 0]
    )
    assert pytest.approx((0, 0, 1, 0, 0, 0)) == ls.directional_coefficient_general_p(
        [0, 0, 1]
    )
    assert pytest.approx((2, 2, 0, 4, 0, 0)) == ls.directional_coefficient_general_p(
        [r2, r2, 0]
    )
    assert pytest.approx((0, 2, 2, 0, 0, 4)) == ls.directional_coefficient_general_p(
        [0, r2, r2]
    )
    assert pytest.approx((2, 0, 2, 0, 4, 0)) == ls.directional_coefficient_general_p(
        [r2, 0, r2]
    )


def test_directional_coefficient_general_s():
    # Calculated looking in Plourde & Bostock (2019) Eq. A3, not in the code
    gs1, gs2, gs3 = ls.directional_coefficient_general_s([1, 0, 0])
    assert pytest.approx((0, 0, 0, 0, 0, 0)) == gs1
    assert pytest.approx((0, 0, 0, 1, 0, 0)) == gs2
    assert pytest.approx((0, 0, 0, 0, 1, 0)) == gs3

    gs1, gs2, gs3 = ls.directional_coefficient_general_s([0, 1, 0])
    assert pytest.approx((0, 0, 0, 1, 0, 0)) == gs1
    assert pytest.approx((0, 0, 0, 0, 0, 0)) == gs2
    assert pytest.approx((0, 0, 0, 0, 0, 1)) == gs3

    gs1, gs2, gs3 = ls.directional_coefficient_general_s([0, 0, 1])
    assert pytest.approx((0, 0, 0, 0, 1, 0)) == gs1
    assert pytest.approx((0, 0, 0, 0, 0, 1)) == gs2
    assert pytest.approx((0, 0, 0, 0, 0, 0)) == gs3


def test_directional_coefficient_deviatoric_p():
    # Calculated looking in Plourde & Bostock (2019) Eq. A8, not in the code
    assert pytest.approx((1, 0, 0, 0, 0)) == ls.directional_coefficient_deviatoric_p(
        [1, 0, 0]
    )
    assert pytest.approx((0, 1, 0, 0, 0)) == ls.directional_coefficient_deviatoric_p(
        [0, 1, 0]
    )
    assert pytest.approx((-1, -1, 0, 0, 0)) == ls.directional_coefficient_deviatoric_p(
        [0, 0, 1]
    )
    assert pytest.approx((2, 2, 4, 0, 0)) == ls.directional_coefficient_deviatoric_p(
        [r2, r2, 0]
    )
    assert pytest.approx((-2, 0, 0, 0, 4)) == ls.directional_coefficient_deviatoric_p(
        [0, r2, r2]
    )
    assert pytest.approx((0, -2, 0, 4, 0)) == ls.directional_coefficient_deviatoric_p(
        [r2, 0, r2]
    )


def test_directional_coefficient_deviatoric_s():
    # Calculated looking in Plourde & Bostock (2019) Eq. A9, not in the code
    gs1, gs2, gs3 = ls.directional_coefficient_deviatoric_s([1, 0, 0])
    assert pytest.approx((0, 0, 0, 0, 0)) == gs1
    assert pytest.approx((0, 0, 1, 0, 0)) == gs2
    assert pytest.approx((0, 0, 0, 1, 0)) == gs3

    gs1, gs2, gs3 = ls.directional_coefficient_deviatoric_s([0, 1, 0])
    assert pytest.approx((0, 0, 1, 0, 0)) == gs1
    assert pytest.approx((0, 0, 0, 0, 0)) == gs2
    assert pytest.approx((0, 0, 0, 0, 1)) == gs3

    gs1, gs2, gs3 = ls.directional_coefficient_deviatoric_s([0, 0, 1])
    assert pytest.approx((0, 0, 0, 1, 0)) == gs1
    assert pytest.approx((0, 0, 0, 0, 1)) == gs2
    assert pytest.approx((0, 0, 0, 0, 0)) == gs3


def test_distance_ratio():
    # Station => x---------a-b <= events
    assert pytest.approx(0.9) == ls.distance_ratio((0, 90, 0), (0, 100, 0), (0, 0, 0))

    # Station => x---------b-a <= events
    assert pytest.approx(1.1) == ls.distance_ratio((0, 110, 0), (0, 100, 0), (0, 0, 0))


def test_p_equation():
    # Make a set of observation along y-axis. Test if correct line is produced
    Aab = 1e-3
    ampl = core.P_Amplitude_Ratio("A", 0, 1, Aab, 0, 0.9, 0.1, 0.5, 20.0)
    sta = core.Station(0, 10, 0, "A")
    events = [core.Event(0, 0, 0, 0, 1, "0"), core.Event(0, 0, 0, 0, 2, "1")]
    in_events = list(range(len(events)))
    phd = {"0_A_P": core.Phase(1, 90, 0), "1_A_P": core.Phase(1, 90, 0)}

    for mt_elements in [5, 6]:

        expect = np.zeros(len(events) * mt_elements)
        expect[1] = -1  # -gamma(event 0)
        expect[mt_elements + 1] = Aab  # gamma(event 1) * Aab * r (=1)

        line = ls.p_equation(ampl, in_events, sta, events, phd, mt_elements)
        pytest.approx(line) == expect

    ampl = core.P_Amplitude_Ratio("A", 0, 1, 1 / Aab, 0, 0.9, 0.1, 0.5, 20.0)
    for mt_elements in [5, 6]:

        expect = np.zeros(len(events) * mt_elements)
        expect[1] = Aab  # Positions of elements reversed
        expect[mt_elements + 1] = -1

        line = ls.p_equation(ampl, in_events, sta, events, phd, mt_elements)
        pytest.approx(line) == expect


def test_s_equations():
    # Make a set of observation along y-axis. Test if correct line is produced
    Babc = 1e-3
    Bacb = 2e3
    ampl = core.S_Amplitude_Ratios(
        "A", 0, 1, 2, Babc, Bacb, 0, 0.8, 0.1, 0.1, 0.5, 20.0
    )
    sta = core.Station(0, 10, 0, "A")
    events = [
        core.Event(0, 0, 0, 0, 1, "0"),
        core.Event(0, 0, 0, 0, 2, "1"),
        core.Event(0, 0, 0, 0, 2, "2"),
    ]
    in_events = list(range(len(events)))
    phd = {
        "0_A_P": core.Phase(1, 90, 0),
        "1_A_P": core.Phase(1, 90, 0),
        "2_A_P": core.Phase(1, 90, 0),
    }

    for mt_elements in [5, 6]:

        # Indices where directional coefficient == 1
        # Obs! Internal sorting of directional coefficients selects gs1 and gs3
        # Obs! because gs2 is all zeros
        i0 = 3
        i1 = 5
        if mt_elements == 5:
            i0 = 2
            i1 = 4

        # OBS! For ray along y gs2 is all 0

        expect0 = np.zeros(len(events) * mt_elements)
        expect0[i0] = -1  # -gamma(event 0)
        expect0[mt_elements + i0] = Babc  # gamma(event 1) * Babc * r (=1)
        expect0[2 * mt_elements + i0] = Bacb  # gamma(event 2) * Bacb * r (=1)

        expect1 = np.zeros(len(events) * mt_elements)
        expect1[i1] = -1  # -gamma(event 0)
        expect1[mt_elements + i1] = Babc  # gamma(event 1) * Aab * r (=1)
        expect1[mt_elements + i1] = Bacb  # gamma(event 1) * Aab * r (=1)

        lines = ls.s_equations(ampl, in_events, sta, events, phd, mt_elements)
        pytest.approx(lines[0]) == expect0
        pytest.approx(lines[1]) == expect1


def test_weight_misfit():
    amp = core.P_Amplitude_Ratio("A", 0, 1, 1e-3, 0.5, 0.9, 0.1, 0.5, 20.0)
    assert pytest.approx(0.5) == ls.weight_misfit(amp, 0.0, 1.0, 0.0, "P")
    assert pytest.approx(0.0) == ls.weight_misfit(amp, 0.0, 0.1, 0.0, "P")
    assert pytest.approx(0.75) == ls.weight_misfit(amp, 0.0, 2.0, 0.0, "P")

    amp = core.S_Amplitude_Ratios(
        "A", 0, 1, 2, 1e-3, 2e-3, 0.5, 0.8, 0.1, 0.1, 0.5, 20.0
    )
    assert pytest.approx([0.5, 0.5]) == ls.weight_misfit(amp, 0.0, 1.0, 0.0, "S")
    assert pytest.approx([0.0, 0.0]) == ls.weight_misfit(amp, 0.0, 0.1, 0.0, "S")
    assert pytest.approx([0.75, 0.75]) == ls.weight_misfit(amp, 0.0, 2.0, 0.0, "S")

    with pytest.raises(ValueError):
        ls.weight_misfit(amp, 0.0, 1.0, 0.0, "X")


def test_weight_s_amplitude():
    amp = core.S_Amplitude_Ratios(
        "A", 0, 1, 2, 1e-3, 2e-3, 0.5, 0.8, 0.1, 0.1, 0.5, 20.0
    )
    assert pytest.approx([1.0, 1.0]) == ls.weight_s_amplitude(amp)
    amp = core.S_Amplitude_Ratios(
        "A", 0, 1, 2, -1e3, 2e-3, 0.5, 0.8, 0.1, 0.1, 0.5, 20.0
    )
    assert pytest.approx([1e-3, 1e-3]) == ls.weight_s_amplitude(amp)
    amp = core.S_Amplitude_Ratios(
        "A", 0, 1, 2, -1e3, 1e4, 0.5, 0.8, 0.1, 0.1, 0.5, 20.0
    )
    assert pytest.approx([1e-4, 1e-4]) == ls.weight_s_amplitude(amp)


def test_homogenous_amplitude_equations():

    # Set up a minimal P & S observation
    Aab = 5e-1
    Babc = 1e-3
    Bacb = 2e3
    pamps = [core.P_Amplitude_Ratio("A", 0, 1, Aab, 0.0, 0.9, 0.1, 0.5, 20.0)]
    samps = [
        core.S_Amplitude_Ratios("A", 0, 1, 2, Babc, Bacb, 0.0, 0.8, 0.1, 0.1, 0.5, 20.0)
    ]

    # Co-located events, Station to the east (along y)
    stad = {"A": core.Station(0, 10, 0, "A")}
    evl = [
        core.Event(0, 0, 0, 0, 1, "0"),
        core.Event(0, 0, 0, 0, 2, "1"),
        core.Event(0, 0, 0, 0, 3, "2"),
    ]
    phd = {
        "0_A_P": core.Phase(1, 90, 0),
        "1_A_P": core.Phase(1, 90, 0),
        "2_A_P": core.Phase(1, 90, 0),
    }

    in_events = list(range(len(evl)))

    # Test deviatoric and full MT
    for constraint in ["deviatoric", "none"]:
        mt_elements = ls.mt_elements(constraint)

        # Directional coefficients (see directional coefficient tests above)
        if mt_elements == 6:
            pcoef = np.array((0, 1, 0, 0, 0, 0))
            scoef0 = np.array((0, 0, 0, 1, 0, 0))
            scoef1 = np.array((0, 0, 0, 0, 0, 0))
            scoef2 = np.array((0, 0, 0, 0, 0, 1))
        if mt_elements == 5:
            pcoef = np.array((0, 1, 0, 0, 0))
            scoef0 = np.array((0, 0, 1, 0, 0))
            scoef1 = np.array((0, 0, 0, 0, 0))
            scoef2 = np.array((0, 0, 0, 0, 1))

        # Call the function
        A, b = ls.homogenous_amplitude_equations(
            pamps, samps, in_events, stad, evl, phd, constraint
        )

        # Expected equations, left hand side
        assert A[0, :mt_elements] == pytest.approx(-pcoef)
        assert A[0, mt_elements : 2 * mt_elements] == pytest.approx(Aab * pcoef)
        assert A[1, :mt_elements] == pytest.approx(-scoef0)
        assert A[1, mt_elements : 2 * mt_elements] == pytest.approx(Babc * scoef0)
        assert A[1, 2 * mt_elements : 3 * mt_elements] == pytest.approx(Bacb * scoef0)
        assert A[2, :mt_elements] == pytest.approx(-scoef2)
        assert A[2, mt_elements : 2 * mt_elements] == pytest.approx(Babc * scoef2)
        assert A[2, 2 * mt_elements : 3 * mt_elements] == pytest.approx(Bacb * scoef2)

        # Expected equations, right hand side
        assert pytest.approx(b) == np.zeros(len(evl))[:, np.newaxis]

        # Try setting the coefficients
        A, b = ls.homogenous_amplitude_equations(
            pamps,
            samps,
            in_events,
            stad,
            evl,
            phd,
            constraint,
            s_coefficients=(1, 2),
        )

        assert A[1, mt_elements : 2 * mt_elements] == pytest.approx(Babc * scoef1)
        assert A[1, 2 * mt_elements : 3 * mt_elements] == pytest.approx(Bacb * scoef1)
        assert A[2, :mt_elements] == pytest.approx(-scoef2)
        assert A[2, mt_elements : 2 * mt_elements] == pytest.approx(Babc * scoef2)
        assert A[2, 2 * mt_elements : 3 * mt_elements] == pytest.approx(Bacb * scoef2)


def test_reference_mt_equations():
    refmt = core.MT(-3.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    # Events in system
    nev = 2
    ref_evs = [0]
    refmt_dict = {0: refmt}

    for const, mt_elements in zip(["none", "deviatoric"], [6, 5]):
        A, b = ls.reference_mt_equations(ref_evs, refmt_dict, nev, const)

        assert pytest.approx(A[:, :mt_elements]) == np.eye(mt_elements)

        if const == "none":
            assert pytest.approx(b) == np.array(refmt)[:, np.newaxis]
        else:
            # eliminate M33 from the system
            refmt = np.array(refmt)
            refmt[2] = refmt[3]
            refmt[3] = refmt[4]
            refmt[4] = refmt[5]
            refmt = refmt[:-1]
            assert pytest.approx(b) == np.array(refmt)[:, np.newaxis]


def test_reference_mt_norm():
    mt_elements = 5
    ev_norm = np.array([0] * mt_elements + [1] * mt_elements + [2] * mt_elements)
    ref_mts = [1]
    exp = np.array([1] * mt_elements)[:, np.newaxis]
    assert pytest.approx(exp) == ls.reference_mt_event_norm(
        ev_norm, ref_mts, mt_elements
    )


def test_norm_event_median_amplitude():
    # Event amplitudes
    evs = [
        [1, 2, 3],
        [10, 20, 30],
        [100, 200, 300],
        [-10, 5, 5],
    ]
    mt_elements = 6
    neq = 3  # Equations
    nev = len(evs)  # Events
    mat = np.zeros((neq, nev * mt_elements))

    for nev, ev in enumerate(evs):
        mat[0, nev * mt_elements + 0] = ev[0]
        mat[1, nev * mt_elements + 1] = ev[1]
        mat[2, nev * mt_elements + 3] = ev[2]

    result = ls.norm_event_median_amplitude(mat, nmt=mt_elements)
    for nev, ev in enumerate(evs):
        assert result[nev * mt_elements : (nev + 1) * mt_elements] == pytest.approx(
            [1 / np.median(np.abs(ev))] * mt_elements
        )

    result = ls.norm_event_median_amplitude(mat, nmt=mt_elements, n_homogenous=2)
    for nev, ev in enumerate(evs):
        assert result[nev * mt_elements : (nev + 1) * mt_elements] == pytest.approx(
            [1 / np.median(np.abs(ev[:2]))] * mt_elements
        )

    with pytest.raises(IndexError):
        ls.norm_event_median_amplitude(mat, nmt=5)


def test_condition_homogenous_matrix_by_norm():
    mt_elements = 6
    neq = 3  # Equations
    nev = 4  # Events
    mat = np.zeros((neq, nev * mt_elements))

    mat[0, 0] = 1
    mat[1, [3, 5]] = [1, np.sqrt(2)]
    mat[2, [2, 5, 10, 11]] = [3, 3, 3, 3]

    assert pytest.approx(
        np.array([1, 1 / np.sqrt(3), 1 / 6])[:, np.newaxis]
    ) == ls.condition_homogenous_matrix_by_norm(mat)

    # Constrain homogenous part
    assert pytest.approx(
        np.array([1, 1 / np.sqrt(3), 1])[:, np.newaxis]
    ) == ls.condition_homogenous_matrix_by_norm(mat, n_homogenous=2)


def test_solve_lsmr():
    # Test if a set of moment tensors is recovered correctly

    # We have four events
    nev = 4

    # ... and 24 stations
    nsta = 24

    # The reference moment tensor and its weight
    ref_mts = [3]
    refmt_weight = 1000

    for constraint in ["none", "deviatoric"]:
        mt_elements = ls.mt_elements(constraint)

        epi_dist = 100e3  # 100 km epicentral distance
        elev = -10e3  # 10 km elevation
        dist = np.sqrt(epi_dist**2 + elev**2)

        evl, stad, phd = conftest.make_events_stations_phases(nev, nsta, epi_dist, elev)
        in_events = list(range(len(evl)))

        M0 = [mt.moment_of_magnitude(ev.mag) for ev in evl]

        # Moment tensors in tuple notation
        mtd = {
            0: core.MT(0, 0, 0, M0[0], 0, 0),  # DC: P -> NE
            1: core.MT(0, 0, 0, 0, M0[1], 0),  # DC: P -> NZ
            2: core.MT(0, 0, 0, 0, 0, M0[2]),  # DC: P -> EZ
            3: core.MT(
                M0[3] / 3, M0[3] / 3, -2 / 3 * M0[3], M0[3] / 3, M0[3] / 3, M0[3] / 3
            ),  # A DC with something on each component
        }

        # Displacement observations
        # TODO: Find out what's the difference to conftest.make_radiation
        up, us = conftest.old_make_radiation(phd, mtd, dist)

        # Combination indices
        ab = [(a, b) for a in range(nev - 1) for b in range(a + 1, nev)]
        abc = [
            (a, b, c)
            for a in range(nev - 2)
            for b in range(a + 1, nev - 1)
            for c in range(b + 1, nev)
        ]

        p_amplitudes = []
        s_amplitudes = []
        for ist in range(nsta):

            # Apply misfits from 0 to 1, by station
            misfit = ist / nsta

            # P amplitude ratios
            for a, b in ab:
                Aab, sigma = amp.pca_amplitude_2p(up[[a, b], ist, :])
                p_amplitudes.append(
                    core.P_Amplitude_Ratio(
                        str(ist), a, b, Aab, misfit, *sigma, 0.5, 20.0
                    )
                )

            # S amplitude ratios
            for a, b, c in abc:
                Babc, Bacb, iord, sigma = amp.pca_amplitude_3s(us[[a, b, c], ist, :])
                s_amplitudes.append(
                    core.S_Amplitude_Ratios(
                        str(ist),
                        *np.array([a, b, c])[iord],
                        Babc,
                        Bacb,
                        misfit,
                        *sigma,
                        0.5,
                        20.0,
                    )
                )

        # Try if applying a weight by line does not change the result
        iapply_line_norm = ["none", "evnorm", "misweight", "ampweight"]

        for iapply in iapply_line_norm:
            # Build homogenos part of linear system
            Ah, bh = ls.homogenous_amplitude_equations(
                p_amplitudes, s_amplitudes, in_events, stad, evl, phd, constraint
            )

            # Build inhomogenous equations
            Ai, bi = ls.reference_mt_equations(ref_mts, mtd, len(evl), constraint)

            # Collect and apply weights ...
            # ... of the homogenous system
            ev_norm = ls.norm_event_median_amplitude(Ah, mt_elements)
            Ah *= ev_norm

            # Test different weights applied by row
            if iapply == "evnorm":
                eq_norm = ls.condition_homogenous_matrix_by_norm(Ah)
                Ah *= eq_norm

            elif iapply == "misweight":
                min_misfit = 0.0
                max_misfit = 1.0
                min_weight = 0.0
                mis_weights = np.vstack(
                    [
                        ls.weight_misfit(amp, min_misfit, max_misfit, min_weight, "P")
                        for amp in p_amplitudes
                    ]
                    + [
                        ls.weight_misfit(amp, min_misfit, max_misfit, min_weight, "S")
                        for amp in s_amplitudes
                    ]
                )
                Ah *= mis_weights

            elif iapply == "ampweight":
                amp_weights = np.vstack(
                    [1.0 for _ in p_amplitudes]
                    + [ls.weight_s_amplitude(amp) for amp in s_amplitudes]
                )
                Ah *= amp_weights

            elif iapply == "none":
                pass

            else:
                raise ValueError(f"Unknown iapply: {iapply}")

            # Collect and apply weights of the inhomogenous system
            mean_moment = mt.mean_moment([mtd[iev] for iev in ref_mts])
            refev_norm = ls.reference_mt_event_norm(ev_norm, ref_mts, mt_elements)
            Ai *= refmt_weight
            bi *= refmt_weight / mean_moment / refev_norm

            # Scale of resulting relative moment tensors
            ev_scale = mean_moment * ev_norm

            A = coo_matrix(np.vstack((Ah, Ai))).tocsc()
            b = np.vstack((bh, bi))

            # Invert and save results
            m, _ = ls.solve_lsmr(A, b, ev_scale)

            outmts = {
                i: momt
                for i, momt in enumerate(mt.mt_tuples(m, constraint))
                if any(momt)
            }

            # Assert that the moment tensors correlate perfectly enough
            for imt in range(nev):
                ccp, ccs = mt.correlation(mtd[imt], outmts[imt])
                assert pytest.approx(ccp) == 1.0
                assert pytest.approx(ccs) == 1.0


def test_mt_elements_invalid_constraint():
    with pytest.raises(ValueError):
        ls.mt_elements("invalid")


def test_unpack_residuals():

    # 2 P residuals, 2 x 2 S residuals, 2 reference residuals
    residuals = [0, 0, 1, 1, 1, 1, 2, 2]
    p_res, s_res, r_res = ls.unpack_resiudals(residuals, 2, 1, 2)

    assert pytest.approx(p_res) == [0, 0]
    assert pytest.approx(s_res) == [[1, 1], [1, 1]]
    assert pytest.approx(r_res) == [2, 2]


def test_bootstrap_lsmr():
    # Test the bootstrapping function

    # Run parameters
    nboot = 2  # Boostrap samples to draw
    seed = 0

    # Parameters of the linear system
    n_mts = 4  # Number of moment tensors
    mt_elements = 6  # Each MT has ... elements
    ncol = n_mts * mt_elements  # Columns of the linear system

    peq = 50  # P-, ...
    seq = 100  # S-, and ...
    req = mt_elements  # reference-equations

    Ah = np.zeros((peq + seq, ncol))
    for ieq in range(peq + seq):
        # Place some event observations in the matrix.
        nev = ieq % n_mts
        Ah[ieq, nev * mt_elements : (nev + 1) * mt_elements] = np.ones(mt_elements)

    Ai = np.zeros((req, ncol))
    for i in range(mt_elements):
        Ai[i, i] = 1

    neq = peq + seq + req  # total equations
    A = coo_matrix(np.vstack((Ah, Ai))).tocsc()  # Identity matrix
    b = np.arange(neq)[:, np.newaxis]  # Data vector
    ev_scale = np.ones(ncol)  # Unit scales

    # Try the sequential method
    mboot = ls.bootstrap_lsmr(A, b, ev_scale, peq, seq, nboot, seed, 1)

    assert mboot.shape == (nboot, ncol)

    # TODO: assert parallel result is the same as sequential
    # mboot2 = ls.bootstrap_lsmr(A, b, ev_scale, peq, seq, nboot, seed, 2)


# Created by Cvscode Co-pilot
def test_solve_irls_sparse():
    G = coo_matrix(np.eye(3)).tocsc()
    d = np.array([1, 2, 3])
    m, r = ls.solve_irls_sparse(G, d)
    assert pytest.approx(m) == d
    assert pytest.approx(r) == 0
