"""Reinforcement-learning environment: drones learn to encircle a target.

Functional and JAX-native, in the style of ParticleGym environments::

    env = Encircle(n_max=8)
    params = env.default_params
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, action, params)

``reset`` and ``step`` are pure functions: compile them with ``jax.jit`` and
run thousands of environments at once with ``jax.vmap``. Every episode draws
the number of active drones between ``n_min`` and ``n_max``; the other drones
are switched off with the alive mask, so a single policy with shared
parameters learns 1 to 8 drones at once. Observations use relative positions
and the nearest neighbours only, so they do not depend on the number of drones.

Actions are velocity commands in units of ``max_speed``. With
``residual=True`` (the default) they are added to the command of the
encirclement controller with phase balancing and repulsion switched off
(guidance field, lag compensation, radial feedback, target velocity). A zero
action then already flies every drone onto the orbit, and the policy only
has to learn the coordination: spacing and collision avoidance.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from particlegym import physics as pp

from encircle.controller import EncircleController, angular_gaps
from encircle.simulate import Scenario


class EncircleParams(NamedTuple):
    """Episode parameters (a pytree: they can be traced and batched).

    Attributes:
        radius: Standoff radius R (m).
        speed: Desired orbit speed v0 (m/s).
        wind: Standard deviation of random velocity fluctuations (m/s).
        target_speed: Speed of the target, which moves in a random direction (m/s).
        radial_weight: Weight of the orbit-radius error in the reward.
        speed_weight: Weight of the orbit-speed error in the reward.
        spacing_weight: Weight of the spacing error in the reward.
        collision_weight: Penalty for being closer than the safe distance.
        max_steps: Episode length in control steps.
    """

    radius: float = 20.0
    speed: float = 5.0
    wind: float = 0.0
    target_speed: float = 0.0
    radial_weight: float = 1.0
    speed_weight: float = 0.5
    spacing_weight: float = 0.5
    collision_weight: float = 5.0
    max_steps: int = 600


class EncircleState(NamedTuple):
    particles: pp.ParticleState
    target: jax.Array  # (2,)
    target_vel: jax.Array  # (2,)
    step: jax.Array


class Encircle:
    """Drones that take off from a launch area, fly to a target and orbit it.

    Args:
        n_max: Largest number of drones (array size).
        n_min: Smallest number of drones drawn at reset.
        max_speed: Largest commanded speed (m/s); actions are scaled by it.
        tau: Response time of the drones' velocity controller (s).
        dt: Integration time step (s).
        substeps: Integration steps per control step.
        safe_distance: Separation below which the collision penalty applies (m).
        launch_distance: Distance from the target to the launch area (m).
        launch_spacing: Distance between drones at take-off (m).
        neighbors: Number of nearest neighbours in each observation.
        residual: Add actions to the guidance-field command instead of using
            them as the full command.
    """

    def __init__(
        self,
        n_max: int = 8,
        n_min: int = 1,
        max_speed: float = 8.0,
        tau: float = 0.5,
        dt: float = 0.02,
        substeps: int = 5,
        safe_distance: float = 3.0,
        launch_distance: float = 100.0,
        launch_spacing: float = 4.0,
        neighbors: int = 3,
        residual: bool = True,
    ) -> None:
        if not 1 <= n_min <= n_max:
            raise ValueError("need 1 <= n_min <= n_max")
        self.n_max, self.n_min = int(n_max), int(n_min)
        self.max_speed, self.tau = float(max_speed), float(tau)
        self.dt, self.substeps = float(dt), int(substeps)
        self.safe_distance = float(safe_distance)
        self.launch_distance = float(launch_distance)
        self.neighbors = int(neighbors)
        self.residual = bool(residual)
        self.field = pp.ForceField(pp.Box((-1e4, -1e4), (1e4, 1e4)))
        grid = Scenario(n_drones=n_max, launch=(0.0, 0.0), launch_spacing=launch_spacing)
        self._launch_offsets = grid.launch_positions()
        self.observation_size = 4 + 5 * self.neighbors + 1

    @property
    def default_params(self) -> EncircleParams:
        return EncircleParams()

    @property
    def control_period(self) -> float:
        return self.dt * self.substeps

    def reset(self, key, params: EncircleParams | None = None):
        """Start an episode: a random number of drones at a launch area in a random direction."""
        params = self.default_params if params is None else params
        count_key, launch_key, target_key = jax.random.split(key, 3)
        n_active = jax.random.randint(count_key, (), self.n_min, self.n_max + 1)
        angle = jax.random.uniform(launch_key, (), minval=0.0, maxval=2 * math.pi)
        launch = self.launch_distance * jnp.stack([jnp.cos(angle), jnp.sin(angle)])
        offsets = jnp.asarray(self._launch_offsets, launch.dtype)
        alive = jnp.arange(self.n_max) < n_active
        particles = pp.create_state(launch + offsets, mass=1.0, radius=0.5, alive=alive)
        heading = jax.random.uniform(target_key, (), minval=0.0, maxval=2 * math.pi)
        target_vel = params.target_speed * jnp.stack([jnp.cos(heading), jnp.sin(heading)])
        state = EncircleState(
            particles=particles,
            target=jnp.zeros(2, launch.dtype),
            target_vel=target_vel.astype(launch.dtype),
            step=jnp.zeros((), jnp.int32),
        )
        return self.observe(state, params), state

    def step(self, key, state: EncircleState, action, params: EncircleParams | None = None):
        """One control step. ``action`` has shape (n_max, 2), entries in [-1, 1]."""
        params = self.default_params if params is None else params
        p = state.particles
        command = self.max_speed * jnp.clip(jnp.asarray(action, p.pos.dtype), -1.0, 1.0)
        if self.residual:
            base = EncircleController(
                radius=params.radius,
                speed=params.speed,
                max_speed=self.max_speed,
                phase_gain=0.0,
                repulsion_speed=0.0,
                response_time=self.tau,
            )
            command = command + base(p.pos, state.target, state.target_vel, p.alive)
        norm = jnp.linalg.norm(command, axis=1, keepdims=True)
        command = command * jnp.minimum(1.0, self.max_speed / jnp.maximum(norm, 1e-12))
        command = jnp.where(p.alive[:, None], command, 0.0)

        def substep(s, k):
            s, _ = pp.langevin_baoab(
                s,
                self.field,
                self.dt,
                gamma=1.0 / self.tau,
                kT=params.wind**2,
                drift=command,
                key=k,
            )
            return s, None

        particles, _ = jax.lax.scan(substep, p, jax.random.split(key, self.substeps))
        state = EncircleState(
            particles=particles,
            target=state.target + self.control_period * state.target_vel,
            target_vel=state.target_vel,
            step=state.step + 1,
        )
        reward = self.reward(state, params)
        done = state.step >= params.max_steps
        info = {"n_active": jnp.sum(particles.alive)}
        return self.observe(state, params), state, reward, done, info

    def observe(self, state: EncircleState, params: EncircleParams) -> jax.Array:
        """Observations (n_max, observation_size); rows of inactive drones are zero.

        Per drone: position and velocity relative to the target (scaled by R
        and max_speed), then for each of the nearest neighbours its relative
        position, relative velocity and a presence flag, and finally the
        fraction of active drones.
        """
        p = state.particles
        alive = p.alive
        own = jnp.concatenate(
            [(p.pos - state.target) / params.radius, (p.vel - state.target_vel) / self.max_speed],
            axis=1,
        )
        diff = p.pos[None, :, :] - p.pos[:, None, :]  # neighbour j relative to drone i
        pair = alive[:, None] & alive[None, :] & ~jnp.eye(self.n_max, dtype=bool)
        dist = jnp.where(pair, jnp.linalg.norm(diff, axis=-1), jnp.inf)
        nearest = jnp.argsort(dist, axis=1)[:, : self.neighbors]
        present = jnp.take_along_axis(pair, nearest, axis=1)
        rel_pos = jnp.take_along_axis(diff, nearest[..., None], axis=1) / params.radius
        rel_vel = (p.vel[nearest] - p.vel[:, None, :]) / self.max_speed
        neighbours = jnp.concatenate([rel_pos, rel_vel, present[..., None].astype(own.dtype)], -1)
        neighbours = jnp.where(present[..., None], neighbours, 0.0)
        fraction = jnp.full((self.n_max, 1), jnp.sum(alive) / self.n_max, own.dtype)
        obs = jnp.concatenate([own, neighbours.reshape(self.n_max, -1), fraction], axis=1)
        return jnp.where(alive[:, None], obs, 0.0)

    def reward(self, state: EncircleState, params: EncircleParams) -> jax.Array:
        """Per-drone reward (n_max,): 0 is perfect, inactive drones get 0."""
        p = state.particles
        alive = p.alive
        rel = p.pos - state.target
        rho = jnp.linalg.norm(rel, axis=1)
        outward = rel / jnp.maximum(rho, 1e-12)[:, None]
        tangent = jnp.stack([-outward[:, 1], outward[:, 0]], axis=1)
        orbit_speed = jnp.sum((p.vel - state.target_vel) * tangent, axis=1)

        radial = -jnp.minimum(jnp.abs(rho / params.radius - 1.0), 2.0)
        speed = -jnp.abs(orbit_speed - params.speed) / params.speed
        count = jnp.sum(alive)
        ideal = 2 * math.pi / jnp.maximum(count, 1)
        gap_ahead, _ = angular_gaps(jnp.arctan2(rel[:, 1], rel[:, 0]), alive)
        spacing = jnp.where(count > 1, -jnp.abs(gap_ahead - ideal) / ideal, 0.0)
        pair = alive[:, None] & alive[None, :] & ~jnp.eye(self.n_max, dtype=bool)
        dist = jnp.linalg.norm(p.pos[:, None, :] - p.pos[None, :, :], axis=-1)
        too_close = jnp.any(pair & (dist < self.safe_distance), axis=1)

        reward = (
            params.radial_weight * radial
            + params.speed_weight * speed
            + params.spacing_weight * spacing
            - params.collision_weight * too_close
        )
        return jnp.where(alive, reward, 0.0)
