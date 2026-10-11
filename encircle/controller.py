"""Encirclement controller: Lyapunov guidance vector field with phase balancing.

Every drone receives the commanded velocity

    u_i = v_i + tau a_i + u_rep,i + v_target,    v_i = s_i f(r_i),

where ``r_i`` is the drone's position relative to the target and ``f`` is the
unit Lyapunov guidance vector field for standoff tracking (Lawrence, Frew &
Pisano 2008; Frew, Lawrence & Morris 2008). The field converges to the circle
of radius R and circulates counterclockwise. ``s_i`` is the drone's speed,
raised or lowered near the circle so that the drones spread evenly around it.
``a_i = (v_i . grad) v_i`` is the acceleration needed to follow the field; the
feed-forward term ``tau a_i`` compensates the lag of a velocity controller with
response time ``tau``, which would otherwise widen the orbit by about
``omega tau R``. ``u_rep,i`` is a short-range repulsion that keeps drones
apart. The same rule works for any number of drones.

In particle terms the command is the drift velocity of the Langevin equation
integrated by ParticleGym: the drone relaxes to it with the response time of
its velocity controller.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp


def guidance_field(rel: jax.Array, radius) -> jax.Array:
    """Unit Lyapunov guidance vectors (N, 2) at positions ``rel`` relative to the target.

    Outside the circle the field points inwards, inside it points outwards, and
    on the circle it is tangential (counterclockwise). Its magnitude is 1
    everywhere except at the target itself.
    """
    x, y = rel[:, 0], rel[:, 1]
    r2 = x * x + y * y
    r = jnp.sqrt(jnp.maximum(r2, 1e-12))
    scale = 1.0 / (r * (r2 + radius * radius))
    fx = -scale * (x * (r2 - radius * radius) + 2.0 * r * radius * y)
    fy = -scale * (y * (r2 - radius * radius) - 2.0 * r * radius * x)
    return jnp.stack([fx, fy], axis=1)


def angular_gaps(theta: jax.Array, active: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Angles to the next active drone ahead and behind, counterclockwise.

    Returns two (N,) arrays; inactive drones get 0. With a single active drone
    both gaps are 0.
    """
    n = theta.shape[0]
    order = jnp.argsort(jnp.where(active, theta, jnp.inf))  # active drones first, by angle
    count = jnp.maximum(jnp.sum(active), 1)
    rank = jnp.zeros(n, jnp.int32).at[order].set(jnp.arange(n, dtype=jnp.int32))
    ahead = order[jnp.mod(rank + 1, count)]
    behind = order[jnp.mod(rank - 1, count)]
    two_pi = 2.0 * math.pi
    gap_ahead = jnp.mod(theta[ahead] - theta, two_pi)
    gap_behind = jnp.mod(theta - theta[behind], two_pi)
    return jnp.where(active, gap_ahead, 0.0), jnp.where(active, gap_behind, 0.0)


@dataclasses.dataclass(frozen=True)
class EncircleController:
    """Commanded velocities that make drones encircle a target.

    Args:
        radius: Standoff radius R (m).
        speed: Nominal orbit speed v0 (m/s).
        max_speed: Saturation of the commanded speed (m/s).
        phase_gain: Speed change per metre of spacing imbalance (1/s).
        max_speed_change: Largest phase-balancing speed change, as a fraction of v0.
        safe_distance: Distance below which drones repel each other (m).
        repulsion_speed: Repulsion speed at zero separation (m/s).
        response_time: Response time tau of the drones' velocity controller (s),
            used for the feed-forward term; 0 disables it.
        radial_gain: Extra radial feedback near the circle (1/s). The field
            alone pulls drones back to the orbit at the rate v0 / R; this term
            stiffens the orbit against gusts.
    """

    radius: float = 20.0
    speed: float = 5.0
    max_speed: float = 8.0
    phase_gain: float = 0.5
    max_speed_change: float = 0.5
    safe_distance: float = 3.0
    repulsion_speed: float = 4.0
    response_time: float = 0.5
    radial_gain: float = 0.5

    def __call__(
        self,
        pos: jax.Array,
        target: jax.Array,
        target_vel: jax.Array,
        active: jax.Array | None = None,
    ) -> jax.Array:
        """Commanded velocities (N, 2) for drones at ``pos`` (N, 2)."""
        n = pos.shape[0]
        active = jnp.ones(n, bool) if active is None else active
        rel = pos - target
        rho = jnp.linalg.norm(rel, axis=1)

        # Phase balancing: speed up when the gap ahead is larger than the gap
        # behind. It acts only near the circle.
        theta = jnp.arctan2(rel[:, 1], rel[:, 0])
        gap_ahead, gap_behind = angular_gaps(theta, active)
        near = jnp.exp(-(((rho - self.radius) / (0.5 * self.radius)) ** 2))
        change = self.phase_gain * self.radius * 0.5 * (gap_ahead - gap_behind) * near
        limit = self.max_speed_change * self.speed
        speed = self.speed + jnp.clip(change, -limit, limit)

        # Field velocity and, by automatic differentiation, the acceleration of
        # a drone that follows it (the derivative of the field along itself).
        def field(r):
            return speed[:, None] * guidance_field(r, self.radius)

        velocity = field(rel)
        _, acceleration = jax.jvp(field, (rel,), (velocity,))
        command = velocity + self.response_time * acceleration
        outward = rel / jnp.maximum(rho, 1e-12)[:, None]
        command = command + (self.radial_gain * (self.radius - rho) * near)[:, None] * outward
        command = command + self.repulsion(pos, active) + target_vel
        norm = jnp.linalg.norm(command, axis=1, keepdims=True)
        command = command * jnp.minimum(1.0, self.max_speed / jnp.maximum(norm, 1e-12))
        return jnp.where(active[:, None], command, 0.0)

    def repulsion(self, pos: jax.Array, active: jax.Array) -> jax.Array:
        """Velocity (N, 2) pushing drones apart once closer than ``safe_distance``."""
        diff = pos[:, None, :] - pos[None, :, :]
        dist = jnp.linalg.norm(diff + jnp.eye(pos.shape[0])[..., None], axis=-1)
        pair = active[:, None] & active[None, :] & ~jnp.eye(pos.shape[0], dtype=bool)
        strength = self.repulsion_speed * jnp.maximum(1.0 - dist / self.safe_distance, 0.0)
        push = jnp.where(pair[..., None], (strength / dist)[..., None] * diff, 0.0)
        return jnp.sum(push, axis=1)
