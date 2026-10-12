"""Closed-loop simulation of drones as particles, using the ParticleGym physics core.

A drone is a point mass whose velocity relaxes to the commanded velocity with
the response time ``tau`` of its velocity controller, plus random velocity
fluctuations that model gusts:

    dv/dt = (u - v) / tau + sqrt(2 wind^2 / tau) xi(t).

This is the Langevin equation with drift velocity ``u`` that ParticleGym
integrates exactly with its BAOAB scheme (``particlegym.physics.langevin_baoab``).
The same equation is used in 2D and in 3D; the flight controller is assumed to
balance gravity, so the weight of the drone does not appear.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from particlegym import physics as pp

from encircle.controller import EncircleController


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A mission: drones take off from a launch area, fly to the target and orbit it.

    Args:
        n_drones: Number of drones.
        duration: Simulated time (s).
        dt: Integration time step (s).
        substeps: Integration steps per control step (control period dt * substeps).
        tau: Response time of the drones' velocity controller (s).
        wind: Standard deviation of random velocity fluctuations (m/s).
        launch: Centre of the launch area (m); the target starts at the origin.
        launch_spacing: Distance between drones at take-off (m).
        target_velocity: Velocity of the target (m/s).
        dim: Number of dimensions, 2 or 3. In 3D, ``launch`` and
            ``target_velocity`` may have two entries (a launch area at the
            altitude of the target, a target that moves horizontally) or three.
    """

    n_drones: int = 8
    duration: float = 120.0
    dt: float = 0.02
    substeps: int = 5
    tau: float = 0.5
    wind: float = 0.0
    launch: tuple[float, ...] = (-100.0, 0.0)
    launch_spacing: float = 4.0
    target_velocity: tuple[float, ...] = (0.0, 0.0)
    dim: int = 2

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        as_vector(self.launch, self.dim)
        as_vector(self.target_velocity, self.dim)

    @property
    def control_period(self) -> float:
        return self.dt * self.substeps

    def launch_positions(self) -> np.ndarray:
        """Take-off positions (N, d) on a horizontal square grid around ``launch``."""
        columns = math.ceil(math.sqrt(self.n_drones))
        index = np.arange(self.n_drones)
        grid = np.stack([index % columns, index // columns], axis=1).astype(float)
        grid -= grid.mean(axis=0)
        offsets = np.zeros((self.n_drones, self.dim))
        offsets[:, :2] = self.launch_spacing * grid
        return as_vector(self.launch, self.dim) + offsets


def as_vector(value, dim: int) -> np.ndarray:
    """``value`` as a vector with ``dim`` entries; a missing third entry is 0."""
    vector = np.asarray(value, dtype=float)
    if vector.shape == (2,) and dim == 3:
        vector = np.append(vector, 0.0)
    if vector.shape != (dim,):
        raise ValueError(f"expected a vector with {dim} entries, got {value!r}")
    return vector


class Trajectory(NamedTuple):
    """Recorded once per control step."""

    time: jax.Array  # (T,)
    pos: jax.Array  # (T, N, d)
    vel: jax.Array  # (T, N, d)
    command: jax.Array  # (T, N, d)
    target: jax.Array  # (T, d)


def simulate(scenario: Scenario, controller: EncircleController, key=None) -> Trajectory:
    """Run the closed loop: controller -> commanded velocity -> particle dynamics."""
    key = jax.random.key(0) if key is None else key
    box = pp.Box((-1e4,) * scenario.dim, (1e4,) * scenario.dim)
    field = pp.ForceField(box)  # drones do not touch: separation is the controller's job
    state = pp.create_state(scenario.launch_positions(), mass=1.0, radius=0.5)
    target_vel = jnp.asarray(as_vector(scenario.target_velocity, scenario.dim), state.pos.dtype)
    gamma = 1.0 / scenario.tau
    kT = scenario.wind**2  # unit mass: velocity fluctuations with standard deviation `wind`
    steps = round(scenario.duration / scenario.control_period)

    def control_step(carry, step_key):
        state, time = carry
        target = time * target_vel
        command = controller(state.pos, target, target_vel)

        def substep(s, k):
            s, _ = pp.langevin_baoab(
                s, field, scenario.dt, gamma=gamma, kT=kT, drift=command, key=k
            )
            return s, None

        state, _ = jax.lax.scan(substep, state, jax.random.split(step_key, scenario.substeps))
        time = time + scenario.control_period
        return (state, time), Trajectory(time, state.pos, state.vel, command, time * target_vel)

    @jax.jit
    def run(state, key):
        carry = (state, jnp.zeros((), state.pos.dtype))
        return jax.lax.scan(control_step, carry, jax.random.split(key, steps))[1]

    return run(state, key)
