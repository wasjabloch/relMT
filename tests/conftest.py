# Functions to create synthetic data for testing purposes

import numpy as np
from relmt import core, angle, utils, mt, signal, io

import pytest

rng = np.random.default_rng(12345)


def mt_rl_strike_slip_north_south_m0(magnitude):
    # North striking right lateral magnitude 0
    M0 = mt.moment_of_magnitude(magnitude)
    return core.MT(0.0, 0.0, 0.0, M0, 0.0, 0.0)


def mt_ll_strike_slip_north_south_m0(magnitude):
    # North striking left lateral magnitude 0
    M0 = mt.moment_of_magnitude(magnitude)
    return core.MT(0.0, 0.0, 0.0, -M0, 0.0, 0.0)


def mt_normal_dip30_east_m0(magnitude):
    # 30 deg east dipping normal fault
    M0 = mt.moment_of_magnitude(magnitude)
    cos30 = np.cos(np.radians(30))
    return core.MT(0.0, cos30 * M0, -cos30 * M0, 0.0, 0.0, -M0 / 2)


def mt_reverse_dip30_east_m0(magnitude):
    # 30 deg east dipping reverse fault
    M0 = mt.moment_of_magnitude(magnitude)
    cos30 = np.cos(np.radians(30))
    return core.MT(0.0, -cos30 * M0, cos30 * M0, 0.0, 0.0, M0 / 2)


def mt_normal_dip30_north_m0(magnitude):
    # 30 deg north dipping normal fault
    M0 = mt.moment_of_magnitude(magnitude)
    cos30 = np.cos(np.radians(30))
    return core.MT(cos30 * M0, 0.0, -cos30 * M0, 0.0, -M0 / 2, 0.0)


def mt_reverse_dip30_north_m0(magnitude):
    # 30 deg north dipping reverse fault
    M0 = mt.moment_of_magnitude(magnitude)
    cos30 = np.cos(np.radians(30))
    return core.MT(-cos30 * M0, 0.0, cos30 * M0, 0.0, M0 / 2, 0.0)


def event_circle(number: int = 8, distance: float = 0.0, depth: float = 0.0):
    # 8 events in a circle, at distance from origin
    return [
        core.Event(
            distance * np.sin(n * 2 * np.pi / number),
            distance * np.cos(n * 2 * np.pi / number),
            depth,
            float(n),
            0.0,
            str(n),
        )
        for n in range(number)
    ]


def station_circle(number: int = 8, distance: float = 1000.0, depth: float = 0.0):
    # 8 stations in a circle, at 1000 m distance from origin
    return {
        f"STA{n}": core.Station(
            distance * np.sin(n * 2 * np.pi / number),
            distance * np.cos(n * 2 * np.pi / number),
            depth,
            f"STA{n}",
        )
        for n in range(8)
    }


def phases(stations, events, vp=6000, vs=4000):
    # Synthetic phase readings in a constant velocity model
    phs = "PS"

    phd = {}
    for ista in stations:
        for iev, ev in enumerate(events):
            sta = stations[ista]
            for ph in phs:
                phid = core.join_phaseid(iev, ista, ph)
                azi = angle.azimuth(ev.north, ev.east, sta.north, sta.east)
                plu = angle.plunge(
                    ev.north, ev.east, ev.depth, sta.north, sta.east, sta.depth
                )
                dist = utils.cartesian_distance(
                    ev.north, ev.east, ev.depth, sta.north, sta.east, sta.depth
                )
                time = dist / vp if ph == "P" else dist / vs
                phd[phid] = core.Phase(time, azi, plu)

    return phd


@pytest.fixture
def synthetic_aligned_waveforms(tmp_path):
    def _create(mt_dict, event_list, station_dict, phase_dict, noise_level=0.0):
        # Create the waveform arrays
        arrd = {
            wvid: rng.normal(size=(len(event_list), 3, 512), scale=noise_level)
            for wvid in core.iterate_waveid(station_dict)
        }

        # And the corresponding headers
        hdrd = {
            core.join_waveid(sta, pha): core.Header(
                station=sta,
                phase=pha,
                components="NEZ",
                sampling_rate=100,
                events=list(range(len(event_list))),
                data_window=5.12,
                phase_start=0,
                phase_end=2,
                taper_length=0.5,
                highpass=0.2,
                lowpass=10.0,
            )
            for sta in station_dict
            for pha in "PS"
        }

        # Now create synthetic data for each phase observation
        for phid in phase_dict:
            iev, sta, pha = core.split_phaseid(phid)
            wvid = core.join_waveid(sta, pha)
            azi = phase_dict[phid].azimuth
            plu = phase_dict[phid].plunge
            dist = utils.cartesian_distance(
                event_list[iev].north,
                event_list[iev].east,
                event_list[iev].depth,
                station_dict[sta].north,
                station_dict[sta].east,
                station_dict[sta].depth,
            )
            if pha == "P":
                amps = mt.p_radiation(
                    mt.mt_array(mt_dict[iev]),
                    azi,
                    plu,
                    dist,
                    3600,  # Density in kg/m3
                    6000,  # P-wave velocity in m/s
                )[:, np.newaxis]
            elif pha == "S":
                amps = mt.s_radiation(
                    mt.mt_array(mt_dict[iev]),
                    azi,
                    plu,
                    dist,
                    3600,  # Density in kg/m3
                    4000,  # S-wave velocity in m/s
                )[:, np.newaxis]

            # Save them into the corresponding array. Add on top of pre-allocated noise
            arrd[wvid][iev, :, :] += (
                signal.make_wavelet(512, 100 / 5, "sin", 100 / 6, 0, 100) * amps
            )

        # Now save everyting to disc
        datadir = tmp_path / "data"
        datadir.mkdir()

        io.make_event_table(event_list, core.file("event", directory=tmp_path))
        io.make_station_table(station_dict, core.file("station", directory=tmp_path))
        io.save_yaml(core.file("exclude", directory=tmp_path), core.exclude)

        for wvid in core.iterate_waveid(station_dict):
            sta, pha = core.split_waveid(wvid)
            hdrf = core.file("waveform_header", sta, pha, directory=tmp_path)
            arrf = core.file("waveform_array", sta, pha, directory=tmp_path)
            hdrd[wvid].to_file(hdrf)
            np.save(arrf, arrd[wvid])

        # But still return the original data, just in case
        return tmp_path, arrd, hdrd

    return _create


def make_events_stations_phases(nev: int, nsta: int, epi_dist: float, elev: float):
    """
    Make event, station and phase tables. Events are at origin, stations in a
    circle above the events

    Paramaters
    ----------
    nev: int
        Number of events
    nsta: int
        Number of stations
    epi_dist:
        Epicentral distance from event to station (meter)
    elev:
        Elevation of stations above events (meter)


    Returns
    -------
    ev_list: list of core.Event
        `nev` events located at the origin (0, 0, 0)
    stad:
        Station dictionary, located in a circle around origin
    phd:
        Phase dictionary. Arrival time not set, but azimuth and plunge
    """

    # Station coordinates

    stazi = np.arange(1, 361, 360 / nsta)  # nsta stations in a circle
    stplu = np.tan(epi_dist / elev) * 180 / np.pi

    # Station dictionary
    stad = {
        str(ista): core.Station(
            np.cos(azi * np.pi / 180) * epi_dist,
            np.sin(azi * np.pi / 180) * epi_dist,
            elev,
            str(ista),
        )
        for ista, azi in enumerate(stazi)
    }

    # All events co-located at the origin
    evl = [core.Event(0, 0, 0, 0, 1, iev) for iev in range(nev)]  # Event coordinates

    phd = {
        core.join_phaseid(iev, st, ph): core.Phase(0, stazi[st], stplu)
        for iev in range(nev)
        for st in range(nsta)
        for ph in "PS"
    }

    return evl, stad, phd


def make_radiation(phd, mtd, dist):
    """
    Create a radiation pattern of phases in `phd` from moment tensors in `mtd`

    Parameters
    ----------
    phd:
        Phase dictionary
    mtd:
        Moment tensor dictionary
    dist: float
        Constant distance from source to receiver (meter)

    returns
    -------
    up, us
        ``(len(mtd), events*stations, 3)`` P- and S-displacement at receiver
    """
    # Medium paramters (should cancel out)
    rho = 3.6e3  # 3600 kg/m3
    alpha = 6000  # m/s
    beta = 4500  # m/s

    stations = sorted(set(core.split_phaseid(phk)[1] for phk in phd))

    # P amplitudes (MTs x stations x component)
    up = np.array(
        [
            [
                mt.p_radiation(
                    mt.mt_array(m),
                    phd[core.join_phaseid(iev, st, "P")].azimuth,
                    phd[core.join_phaseid(iev, st, "P")].plunge,
                    dist,
                    rho,
                    alpha,
                )
                for st in stations
            ]
            for iev, m in sorted(mtd.items())
        ]
    )

    # S amplitudes
    us = np.array(
        [
            [
                mt.s_radiation(
                    mt.mt_array(m),
                    phd[core.join_phaseid(iev, st, "S")].azimuth,
                    phd[core.join_phaseid(iev, st, "S")].plunge,
                    dist,
                    rho,
                    beta,
                )
                for st in stations
            ]
            for iev, m in sorted(mtd.items())
        ]
    )

    return up, us


def old_make_radiation(phd, mtd, dist):
    """
    Create a radiation pattern of phases in `phd` from moment tensors in `mtd`

    Parameters
    ----------
    phd:
        Phase dictionary
    mtd:
        Moment tensor dictionary
    dist: float
        Constant distance from source to receiver (meter)

    returns
    -------
    up, us
        P- and S-displacement at receiver
    """
    # Medium paramters (should cancel out)
    rho = 3.6e3  # 3600 kg/m3
    alpha = 6000  # m/s
    beta = 4500  # m/s

    # P amplitudes (MTs x stations x component)
    up = np.array(
        [
            [
                mt.p_radiation(mt.mt_array(m), ph.azimuth, ph.plunge, dist, rho, alpha)
                for phk, ph in phd.items()
                if core.split_phaseid(phk)[2] == "P"
            ]
            for m in mtd.values()
        ]
    )

    # S amplitudes
    us = np.array(
        [
            [
                mt.s_radiation(mt.mt_array(m), ph.azimuth, ph.plunge, dist, rho, beta)
                for phk, ph in phd.items()
                if core.split_phaseid(phk)[2] == "S"
            ]
            for m in mtd.values()
        ]
    )

    return up, us
