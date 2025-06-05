# Functions to create synthetic data for testing purposes

import numpy as np
from relmt import core, mt


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
