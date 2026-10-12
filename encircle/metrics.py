"""Mission metrics computed from a recorded trajectory."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from encircle.controller import ORBIT_HEIGHT, VERTICAL, orbit_coordinates, orbit_frame
from encircle.simulate import Trajectory


def evaluate(
    traj: Trajectory,
    radius: float,
    radial_tolerance: float = 0.10,
    spacing_tolerance: float = 0.10,
    hold_fraction: float = 0.95,
    steady_fraction: float = 0.2,
    height: float = ORBIT_HEIGHT,
    axis=VERTICAL,
) -> dict:
    """Convergence time, orbit accuracy, spacing uniformity and safety.

    A frame counts as converged when every drone is within
    ``radial_tolerance * radius`` of the orbit and every angular gap is within
    ``spacing_tolerance`` of the ideal gap 2 pi / N. The convergence time is
    the first time after which at least ``hold_fraction`` of the remaining
    frames are converged, so that short excursions caused by gusts do not
    count as a failure. Steady-state averages use the last
    ``steady_fraction`` of the run.

    For a 3D trajectory the orbit lies in the plane normal to ``axis``, with
    its centre ``height`` above the target (the defaults are those of the
    controller). The distance from the orbit radius is measured in that
    plane, and a converged drone must also be within
    ``radial_tolerance * radius`` of the plane. The result then also holds
    the RMS height above the plane and the largest vertical speed, and
    ``max_speed_mps`` is the largest horizontal speed.
    """
    time = np.asarray(traj.time)
    pos = np.asarray(traj.pos)
    n, dim = pos.shape[1:]
    rel = pos - np.asarray(traj.target)[:, None, :]
    if dim == 3:
        rel = rel - height * orbit_frame(axis)[2]
    planar, zeta = (np.asarray(x) for x in orbit_coordinates(jnp.asarray(rel), axis))
    rho = np.linalg.norm(planar, axis=-1)

    converged = np.all(np.abs(rho - radius) < radial_tolerance * radius, axis=1)
    converged &= np.all(np.abs(zeta) < radial_tolerance * radius, axis=1)
    gaps = None
    if n > 1:
        theta = np.sort(np.arctan2(planar[..., 1], planar[..., 0]), axis=1)
        gaps = np.diff(np.concatenate([theta, theta[:, :1] + 2 * math.pi], axis=1), axis=1)
        ideal = 2 * math.pi / n
        converged &= np.all(np.abs(gaps - ideal) < spacing_tolerance * ideal, axis=1)

    # Fraction of converged frames from each frame to the end of the run.
    remaining = np.cumsum(converged[::-1])[::-1] / np.arange(len(time), 0, -1)
    holds = np.flatnonzero(remaining >= hold_fraction)
    convergence_time = float(time[holds[0]]) if holds.size else math.inf

    steady = slice(int(len(time) * (1 - steady_fraction)), None)
    vel = np.asarray(traj.vel)
    speed = np.linalg.norm(vel[..., :2], axis=-1)
    result = {
        "drones": n,
        "convergence_time_s": convergence_time,
        "radial_rms_m": float(np.sqrt(np.mean((rho[steady] - radius) ** 2))),
        "spacing_cv": math.nan,
        "min_distance_m": math.nan,
        "max_speed_mps": float(speed.max()),
    }
    if n > 1:
        result["spacing_cv"] = float(np.mean(gaps[steady].std(axis=1) / gaps[steady].mean(axis=1)))
        pairwise = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
        pairwise[:, np.arange(n), np.arange(n)] = np.inf
        result["min_distance_m"] = float(pairwise.min())
    if dim == 3:
        result["height_rms_m"] = float(np.sqrt(np.mean(zeta[steady] ** 2)))
        result["max_vertical_speed_mps"] = float(np.abs(vel[..., 2]).max())
    return result
