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

The controller works in 2D and in 3D. In 3D the orbit is a circle of radius R
in the plane normal to ``axis``, with its centre ``height`` above the target
along that axis; by default it is a horizontal circle above the target. The
rule above then acts on the components of ``r_i`` in the orbit plane, and a
proportional term brings the drone to the plane,

    v_i = s_i f(r_i in the plane) - clip(k_h zeta_i) n,

where ``zeta_i`` is the height of the drone above the orbit plane and ``n`` is
the orbit axis. Horizontal and vertical speeds are limited separately, as for
a multirotor. With all drones in the orbit plane, the 3D controller gives the
commands of the 2D controller.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

VERTICAL = (0.0, 0.0, 1.0)
# Default height of a 3D orbit above the target (m).
ORBIT_HEIGHT = 10.0


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


def orbit_frame(axis=VERTICAL) -> np.ndarray:
    """Right-handed orthonormal frame of a 3D orbit: rows ``e1``, ``e2`` and the axis ``n``.

    ``e1`` and ``e2`` span the orbit plane, and drones circulate from ``e1``
    towards ``e2``, counterclockwise when seen from the tip of ``n``. For a
    vertical axis the frame is the x, y and z axes.
    """
    n = np.asarray(axis, dtype=float)
    if n.shape != (3,) or not np.linalg.norm(n) > 0:
        raise ValueError("axis must be a non-zero vector with three components")
    n = n / np.linalg.norm(n)
    # Start from the x axis, or from the y axis if the orbit axis is close to x.
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, n) * n
    e1 = e1 / np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1), n])


def orbit_coordinates(vectors: jax.Array, axis=VERTICAL) -> tuple[jax.Array, jax.Array]:
    """Components of ``vectors`` (..., d) in the orbit plane (..., 2) and along its axis (...).

    In 2D the plane is the whole space and the component along the axis is 0.
    """
    if vectors.shape[-1] == 2:
        return vectors, jnp.zeros(vectors.shape[:-1], vectors.dtype)
    e1, e2, n = jnp.asarray(orbit_frame(axis), vectors.dtype)
    return jnp.stack([vectors @ e1, vectors @ e2], axis=-1), vectors @ n


def from_orbit_coordinates(planar: jax.Array, along: jax.Array, dim: int, axis=VERTICAL):
    """Inverse of :func:`orbit_coordinates`: vectors (N, dim) from their components."""
    if dim == 2:
        return planar
    e1, e2, n = jnp.asarray(orbit_frame(axis), planar.dtype)
    return planar[:, :1] * e1 + planar[:, 1:] * e2 + along[:, None] * n


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
        height: 3D only: height of the orbit centre above the target, along
            ``axis`` (m).
        axis: 3D only: normal of the orbit plane; vertical for a horizontal orbit.
        height_gain: 3D only: rate at which the height above the orbit plane
            decays (1/s).
        max_vertical_speed: 3D only: saturation of the vertical speed, and of
            the speed of the approach to the orbit plane (m/s). ``max_speed``
            then limits the horizontal speed.
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
    height: float = ORBIT_HEIGHT
    axis: tuple[float, float, float] = VERTICAL
    height_gain: float = 0.5
    max_vertical_speed: float = 3.0

    def __call__(
        self,
        pos: jax.Array,
        target: jax.Array,
        target_vel: jax.Array,
        active: jax.Array | None = None,
    ) -> jax.Array:
        """Commanded velocities (N, d) for drones at ``pos`` (N, d), with d = 2 or 3."""
        n, dim = pos.shape
        active = jnp.ones(n, bool) if active is None else active
        rel = pos - self.orbit_centre(target)
        planar, zeta = orbit_coordinates(rel, self.axis)
        rho = jnp.linalg.norm(planar, axis=1)

        # Phase balancing: speed up when the gap ahead is larger than the gap
        # behind. It acts only near the circle.
        theta = jnp.arctan2(planar[:, 1], planar[:, 0])
        gap_ahead, gap_behind = angular_gaps(theta, active)
        half = 0.5 * self.radius
        near = jnp.exp(-(((rho - self.radius) / half) ** 2) - (zeta / half) ** 2)
        change = self.phase_gain * self.radius * 0.5 * (gap_ahead - gap_behind) * near
        limit = self.max_speed_change * self.speed
        speed = self.speed + jnp.clip(change, -limit, limit)

        # Field velocity and, by automatic differentiation, the acceleration of
        # a drone that follows it (the derivative of the field along itself).
        def field(r):
            in_plane, above = orbit_coordinates(r, self.axis)
            circulation = speed[:, None] * guidance_field(in_plane, self.radius)
            fastest = self.max_vertical_speed
            approach = -jnp.clip(self.height_gain * above, -fastest, fastest)
            return from_orbit_coordinates(circulation, approach, dim, self.axis)

        velocity = field(rel)
        _, acceleration = jax.jvp(field, (rel,), (velocity,))
        command = velocity + self.response_time * acceleration
        outward = planar / jnp.maximum(rho, 1e-12)[:, None]
        outward = from_orbit_coordinates(outward, jnp.zeros_like(rho), dim, self.axis)
        command = command + (self.radial_gain * (self.radius - rho) * near)[:, None] * outward
        command = command + self.repulsion(pos, active) + target_vel
        return jnp.where(active[:, None], self.saturate(command), 0.0)

    def orbit_centre(self, target: jax.Array) -> jax.Array:
        """Centre of the orbit: the target in 2D, ``height`` above it along ``axis`` in 3D."""
        if target.shape[-1] == 2:
            return target
        return target + self.height * jnp.asarray(orbit_frame(self.axis)[2], target.dtype)

    def saturate(self, command: jax.Array) -> jax.Array:
        """Limit commands (N, d) to ``max_speed``, and in 3D to ``max_vertical_speed``.

        In 3D ``max_speed`` limits the horizontal speed.
        """
        horizontal = command[:, :2]
        norm = jnp.linalg.norm(horizontal, axis=1, keepdims=True)
        horizontal = horizontal * jnp.minimum(1.0, self.max_speed / jnp.maximum(norm, 1e-12))
        if command.shape[1] == 2:
            return horizontal
        limit = self.max_vertical_speed
        return jnp.concatenate([horizontal, jnp.clip(command[:, 2:], -limit, limit)], axis=1)

    def repulsion(self, pos: jax.Array, active: jax.Array) -> jax.Array:
        """Velocity (N, d) pushing drones apart once closer than ``safe_distance``."""
        diff = pos[:, None, :] - pos[None, :, :]
        dist = jnp.linalg.norm(diff + jnp.eye(pos.shape[0])[..., None], axis=-1)
        pair = active[:, None] & active[None, :] & ~jnp.eye(pos.shape[0], dtype=bool)
        strength = self.repulsion_speed * jnp.maximum(1.0 - dist / self.safe_distance, 0.0)
        push = jnp.where(pair[..., None], (strength / dist)[..., None] * diff, 0.0)
        return jnp.sum(push, axis=1)
