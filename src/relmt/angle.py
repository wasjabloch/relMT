"""
Functions used to compute take-off angles from 1D velocity model

Copied from SKHASH

Significant portions of the functions in this file are based on the Fortran HASH
code originally written by Jeanne L. Hardebeck & Peter M. Shearer, and all of it
is inspired by their work. Please cite the appropriate references if you use
this code.
"""

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

import numpy as np
import logging
from relmt import core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(core.logsh)


def hash_plunge_table(
    depth_velocity: np.ndarray,
    depths: np.ndarray,
    distances: np.ndarray,
    nray: int,
):
    """
    Create tables of takeoff plunge angles given a 1D velocity model.

    Modified from Public Domain code SKHASH (Skoumal, Hardebeck & Shearer, 2024;
    Hardebeck & Shearer, 2002) by courtesey of USGS.

    Parameters
    ----------
    depth_velocity:
        ``(layers, 2)`` array of depth and seismic velocity (arbitrary consitent
        units, e.g. m and m/s or km and km/s)
    depths:
        Array of depths of the seimic source (unit as above)
    dists:
        Array of source-receiver distances (unit as above). Must start at 0-range
    nray:
        Number of rays traced

    Retruns:
    --------
    ``(distance, depth)`` lookup table of takeoff plunge angles
    """

    # TODO: consider station at depth (borehole)

    if depth_velocity[:, 0][-1] < depths[-1]:
        depth_velocity = np.vstack((depth_velocity, depth_velocity[-1, :]))
        depth_velocity[-1, 0] = depths[-1] + 1
        depth_velocity[-1, 1] = depth_velocity[-1, 1] + 0.001

    ndel = len(distances)
    ndep = len(depths)

    table = np.zeros((ndel, ndep)) - 999

    pmin = 0

    # Code internally works in km/s. Let's respect that
    z = depth_velocity[:, 0]
    alpha = depth_velocity[:, 1]

    nz = len(z)
    z = np.hstack((z, z[nz - 1]))
    alpha = np.hstack((alpha, alpha[nz - 1]))

    # Re-sample velocity model onto depth-vector
    for i in range(nz - 1, 0, -1):
        for idep in range(ndep - 1, -1, -1):
            if (z[i - 1] <= (depths[idep] - 0.00001)) & (
                z[i] >= (depths[idep] + 0.00001)
            ):
                z = np.insert(z, i, z[i - 1])
                alpha = np.insert(alpha, i, alpha[i - 1])
                z[i] = depths[idep]
                frac = (z[i] - z[i - 1]) / (z[i + 1] - z[i - 1])
                alpha[i] = alpha[i - 1] + frac * (alpha[i + 1] - alpha[i - 1])

    # do the ray tracing
    slow = 1 / alpha
    pmax = slow[0]
    pstep = (pmax - pmin) / nray

    npmax = int((pmax + pstep / 2 - pmin) / pstep) + 1

    depxcor = np.zeros((npmax, ndep))
    depucor = np.zeros((npmax, ndep))
    deptcor = np.zeros((npmax, ndep))

    # Overwritten in the next line? TODO: test delete
    tmp_ind = np.where(depths == 0)[0]

    tmp_ind = np.where(depths != 0)[0]
    depxcor[:, tmp_ind] = -999
    deptcor[:, tmp_ind] = -999

    ptab = np.linspace(pmin, pmin + pstep * (npmax - 1), num=npmax)

    # Layer thicknesses and slownesses
    h_array = z[1:] - z[:-1]
    utop = slow[:-1]
    ubot = slow[1:]

    """LAYERTRACE equivalent"""
    dx = np.zeros((npmax, len(utop)))
    dt = np.zeros((npmax, len(utop)))
    irtr = np.zeros((npmax, len(utop)), dtype=int)

    qs = np.zeros((npmax, len(utop)))
    qs[:] = np.nan
    qr = np.zeros((npmax, len(utop)))
    qr[:] = np.nan
    ytop = utop - ptab[:, np.newaxis]
    ytop_pos_flag = ytop > 0
    qs[ytop_pos_flag] = (
        ytop[ytop_pos_flag] * (utop + ptab[:, np.newaxis])[ytop_pos_flag]
    )
    qs[ytop_pos_flag] = np.sqrt(qs[ytop_pos_flag])

    qr = np.arctan2(qs, ptab[:, np.newaxis])

    b = np.ma.divide(-np.log(ubot / utop), h_array)
    b = b.filled(np.nan)

    # integral at upper limit, 1/b factor omitted until end
    etau = qs - qr * ptab[:, np.newaxis]
    ex = qr

    # check lower limit to see if we have turning point
    ybot = ubot - ptab[:, np.newaxis]

    # if turning point, then no contribution from bottom point
    y_subzero_flag = ybot <= 0
    y_greaterzero_flag = ybot > 0
    irtr[y_subzero_flag] = 2
    irtr[y_greaterzero_flag] = 1

    irtr[~ytop_pos_flag] = 0

    dx[y_subzero_flag] = ex[y_subzero_flag]
    dx = dx / b
    dtau = etau / b
    dt[y_subzero_flag] = (
        dtau[y_subzero_flag] + (dx * ptab[:, np.newaxis])[y_subzero_flag]
    )  # converts tau to t

    q = np.zeros(ybot.shape)
    q[:] = np.nan
    q[y_greaterzero_flag] = ybot[y_greaterzero_flag] * (
        (ubot + ptab[:, np.newaxis])[y_greaterzero_flag]
    )
    qs = np.sqrt(q)

    qr = np.arctan2(qs, ptab[:, np.newaxis])
    etau = etau - qs + ptab[:, np.newaxis] * qr
    ex = ex - qr

    exb = ex / b
    dtau = etau / b
    dx[y_greaterzero_flag] = exb[y_greaterzero_flag]
    dt[y_greaterzero_flag] = (
        dtau[y_greaterzero_flag] + (exb * ptab[:, np.newaxis])[y_greaterzero_flag]
    )

    # Ensures values after ray has turned are nan
    x = (irtr == 0) | (irtr == 2)
    idx = np.arange(npmax), x.argmax(axis=1)
    tmp = x[idx] == True

    idx_1 = idx[0][tmp], idx[1][tmp] + 1
    tmp_1 = idx_1[1] < (len(utop) - 1)
    idx_1 = idx_1[0][tmp_1], idx_1[1][tmp_1]
    for xx in range(len(idx_1[0])):
        row = idx_1[0][xx]
        col = idx_1[1][xx]
        dx[row, col:] = np.nan
        dt[row, col:] = np.nan

    # Distance table, travel time table
    deltab = np.nansum(dx, axis=1) * 2
    tttab = np.nansum(dt, axis=1) * 2

    idx_2 = idx[0][tmp], idx[1][tmp]
    tmp_2 = idx_2[1] < (len(utop) - 1)
    idx_2 = idx_2[0][tmp_2], idx_2[1][tmp_2]
    for xx in range(len(idx_2[0])):
        row = idx_2[0][xx]
        col = idx_2[1][xx]
        dx[row, col:] = np.nan
        dt[row, col:] = np.nan

    depxcor = np.cumsum(dx, axis=1)
    deptcor = np.cumsum(dt, axis=1)
    output_col_ind = np.where(np.isin(z, depths))[0] - 1
    depxcor = depxcor[:, output_col_ind]
    deptcor = deptcor[:, output_col_ind]
    depxcor[:, 0] = 0
    deptcor[:, 0] = 0
    depucor[:] = slow[output_col_ind + 1]
    depucor[np.isnan(depxcor)] = -999
    depxcor[np.isnan(depxcor)] = -999
    deptcor[np.isnan(deptcor)] = -999

    x = np.diff(depxcor, axis=0) <= 0
    idx = x.argmax(axis=0) + 1
    # TODO: The line below caused IndexError for certain velocity models
    # Because x is one smaller than depxor along dimension 0
    # (the +1 above points correctly into idx, but not into tmp
    # tmp = x[idx, np.arange(ndep)] == False

    # So we substract that one index to find the one that should be set to the
    # maximum ray index. This should be safe.
    tmp = x[idx - 1, np.arange(ndep)] == False
    idx[tmp] = npmax - 1

    for idep in range(ndep):
        # upgoing rays from source
        xsave_up = depxcor[: (idx[idep]), idep]
        tsave_up = deptcor[: (idx[idep]), idep]
        usave_up = depucor[: (idx[idep]), idep]
        psave_up = -1 * ptab[: (idx[idep])]

        # downgoing rays from source
        down_idx = np.where((depxcor[:, idep] != -999) & (deltab != -999))[0][::-1]
        xsave_down = deltab[down_idx] - depxcor[down_idx, idep]
        tsave_down = tttab[down_idx] - deptcor[down_idx, idep]
        usave_down = depucor[down_idx, idep]
        psave_down = ptab[down_idx]

        # Merges upgoing and downgoing ray arrays
        xsave = np.hstack([xsave_up, xsave_down])
        tsave = np.hstack([tsave_up, tsave_down])
        usave = np.hstack([usave_up, usave_down])
        psave = np.hstack([psave_up, psave_down])

        # Now search the rays closest to the querry distances
        scr1 = np.zeros(ndel)
        for idel in range(1, ndel):
            del_x = distances[idel]
            ind = np.where((xsave[:-1] <= del_x) & (xsave[1:] >= del_x))[0] + 1

            frac = (del_x - xsave[ind - 1]) / (xsave[ind] - xsave[ind - 1])
            t1 = tsave[ind - 1] + frac * (tsave[ind] - tsave[ind - 1])

            min_ind = ind[np.argmin(t1)]

            scr1[idel] = psave[min_ind] / usave[min_ind]

        angle = np.rad2deg(np.arcsin(scr1))

        angle_flag = angle >= 0
        angle *= -1

        # SKHASH convention
        angle[angle_flag] += 180

        table[:, idep] = angle

    if distances[0] == 0:
        # SKHASH convention
        table[0, :] = 0.0

    # relMT convention
    table -= 90.0

    return table


def clean_vmodel(vmodel: np.ndarray) -> np.ndarray:
    """Clean velocity model as to adhere to create_takeoff_angle convention

    Function courtesy of the SKHASH developers

    Parameters
    ----------
    vmodel:
        ``(layers, 3)`` table holding depth,(m) P and S-wave velocity (m/s)

    Returns
    -------
    ``(layers, 3)`` tabel with monotonically increasing velocities
    """

    if len(vmodel) == 1:
        msg = (
            "Velocity model has only a single velocity. "
            "There must be at least two points."
        )
        raise IndexError(msg)
    else:
        if not (all(np.diff(vmodel[:, 0]) >= 0)):
            msg = (
                "The velocity model is expected to be ordered in terms of "
                "increasing depth. Ordering it now. "
                "Unexpected results may occurr."
            )
            logger.warning(msg)

            vm_sort_ind = np.argsort(vmodel[:, 0])
            vmodel = vmodel[vm_sort_ind, :]

        if np.any(idup := (np.diff(vmodel[:, 0]) == 0)):
            msg = (
                "Duplicate velocities for a given depth. Removing layers: "
                + ", ".join(map(str, idup.nonzero()[0]))
            )
            logger.warning(msg)
            vmodel = np.delete(
                vmodel,
                np.where(idup)[0],
                axis=0,
            )
    if vmodel[:, 0].max() < 1000:
        msg = (
            "Velocity model only reaches down to "
            f"{vmodel[:, 0].max()} m. Is this intentional?"
        )
        logger.warning(msg)

    if vmodel[:, 1].max() < 1000:
        msg = (
            "Maximum velocity is only "
            f"{vmodel[:, 1].max()} m/s. Is this intentional?"
        )
        logger.warning(msg)

    # If there are constant velocity layers, merge those rows
    drop_constant_vel_ind = np.where(np.diff(vmodel[:, 1]) <= 0)[0] + 1
    if len(drop_constant_vel_ind) > 0:
        msg = (
            "Constant velocities for layers: "
            + ", ".join(map(str, drop_constant_vel_ind))
            + ". Deleting."
        )
        logger.warning(msg)
        vmodel = np.delete(vmodel, drop_constant_vel_ind, axis=0)

    return vmodel
