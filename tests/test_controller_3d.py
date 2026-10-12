"""The encirclement controller in 3D: orbit frame, planar limit, altitude and tilted orbits."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from encircle import (
    EncircleController,
    Scenario,
    evaluate,
    orbit_coordinates,
    orbit_frame,
    simulate,
)
from encircle.controller import from_orbit_coordinates

TILTED = (0.0, math.sin(math.radians(30.0)), math.cos(math.radians(30.0)))


def simulate_one(controller, **kwargs):
    return simulate(Scenario(dim=3, **kwargs), controller, jax.random.key(1))


@pytest.mark.parametrize("axis", [(0, 0, 1), (0, 0, -2), (1, 0, 0), (0, 1, 0), TILTED, (3, -1, 2)])
def test_orbit_frame_is_orthonormal_and_right_handed(axis):
    frame = orbit_frame(axis)
    e1, e2, n = frame
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(e1, e2), n, atol=1e-12)
    np.testing.assert_allclose(n, np.asarray(axis) / np.linalg.norm(axis), atol=1e-12)


def test_orbit_frame_of_a_horizontal_orbit_is_the_coordinate_axes():
    np.testing.assert_array_equal(orbit_frame(), np.eye(3))
    with pytest.raises(ValueError):
        orbit_frame((0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        orbit_frame((0.0, 1.0))


def test_orbit_coordinates_can_be_inverted():
    vectors = jnp.asarray(np.random.default_rng(0).normal(size=(50, 3)))
    planar, along = orbit_coordinates(vectors, TILTED)
    np.testing.assert_allclose(along, vectors @ np.asarray(TILTED), atol=1e-12)
    np.testing.assert_allclose(np.sum(planar**2, axis=1) + along**2, np.sum(vectors**2, axis=1))
    back = from_orbit_coordinates(planar, along, 3, TILTED)
    np.testing.assert_allclose(back, vectors, atol=1e-12)
    # In 2D the plane is the whole space.
    flat, zero = orbit_coordinates(vectors[:, :2])
    np.testing.assert_array_equal(flat, vectors[:, :2])
    np.testing.assert_array_equal(zero, 0.0)


def test_command_is_tangential_on_the_orbit_and_returns_drones_to_its_plane():
    # The guidance field alone: no lag compensation, radial feedback or coordination.
    field = EncircleController(
        height=10.0, response_time=0.0, radial_gain=0.0, phase_gain=0.0, repulsion_speed=0.0
    )
    target = jnp.array([3.0, -2.0, 1.0])
    on_orbit = target + jnp.array([20.0, 0.0, 10.0])
    pos = jnp.stack([on_orbit, on_orbit + jnp.array([0.0, 0.0, 4.0]), target])
    command = field(pos, target, jnp.zeros(3))
    np.testing.assert_allclose(command[0], [0.0, 5.0, 0.0], atol=1e-12)  # counterclockwise
    np.testing.assert_allclose(command[1], [0.0, 5.0, -2.0], atol=1e-12)  # 4 m too high
    assert command[2, 2] == 3.0  # 10 m too low: climbs at the largest vertical speed

    moving = field(pos, target, jnp.array([1.0, 0.5, -0.25]))
    np.testing.assert_allclose(moving[:2] - command[:2], [[1.0, 0.5, -0.25]] * 2, atol=1e-12)


def test_speeds_are_limited_separately_in_the_horizontal_and_vertical():
    controller = EncircleController(max_speed=8.0, max_vertical_speed=3.0)
    command = jnp.array([[30.0, 40.0, 10.0], [3.0, 4.0, -1.0], [0.0, 0.0, -7.0]])
    np.testing.assert_allclose(
        controller.saturate(command), [[4.8, 6.4, 3.0], [3.0, 4.0, -1.0], [0.0, 0.0, -3.0]]
    )


def test_drones_in_the_orbit_plane_fly_as_in_2d():
    key = jax.random.key(1)
    flat = simulate(Scenario(n_drones=8), EncircleController(), key)
    level = simulate(Scenario(n_drones=8, dim=3), EncircleController(height=0.0), key)
    assert level.pos.shape == (*flat.pos.shape[:2], 3)
    np.testing.assert_allclose(level.pos[..., :2], flat.pos, rtol=0, atol=1e-9)
    np.testing.assert_allclose(level.vel[..., :2], flat.vel, rtol=0, atol=1e-9)
    np.testing.assert_allclose(level.pos[..., 2], 0.0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 4, 8])
def test_drones_climb_to_the_orbit_and_spread_evenly(n):
    controller = EncircleController(height=10.0)
    trajectory = simulate_one(controller, n_drones=n)
    result = evaluate(trajectory, 20.0, height=10.0)
    # After one control step of 0.1 s the vertical speed is still relaxing to 3 m/s:
    # z = v (t - tau (1 - exp(-t / tau))).
    climbed = 3.0 * (0.1 - 0.5 * (1 - math.exp(-0.1 / 0.5)))
    np.testing.assert_allclose(trajectory.pos[0, :, 2], climbed, atol=1e-3)
    np.testing.assert_allclose(trajectory.pos[-1, :, 2], 10.0, atol=0.01)
    assert result["convergence_time_s"] < 60.0
    assert result["radial_rms_m"] < 0.2
    assert result["height_rms_m"] < 0.01
    assert result["max_speed_mps"] <= 8.0 + 1e-6
    assert result["max_vertical_speed_mps"] <= 3.0 + 1e-6
    if n > 1:
        assert result["spacing_cv"] < 0.01
        assert result["min_distance_m"] > 2.0


def test_gusts_and_a_climbing_target_are_tolerated():
    trajectory = simulate_one(EncircleController(), wind=0.3, target_velocity=(2.0, 0.0, 1.0))
    result = evaluate(trajectory, 20.0)
    np.testing.assert_allclose(trajectory.target[-1], [240.0, 0.0, 120.0], rtol=1e-6)
    assert result["convergence_time_s"] < 80.0
    assert result["radial_rms_m"] < 0.5
    assert result["height_rms_m"] < 0.5
    assert result["spacing_cv"] < 0.05
    assert result["min_distance_m"] > 1.5


def test_tilted_orbit():
    controller = EncircleController(axis=TILTED)
    trajectory = simulate_one(controller)
    result = evaluate(trajectory, 20.0, axis=TILTED)
    assert result["convergence_time_s"] < 60.0
    assert result["radial_rms_m"] < 0.2
    assert result["height_rms_m"] < 0.01
    assert result["spacing_cv"] < 0.01
    # The orbit centre is 10 m from the target along the axis, and the circle
    # of radius 20 m is tilted by 30 degrees: altitudes span 10 cos 30 -+ 10 m.
    altitude = np.asarray(trajectory.pos[-300:, :, 2])
    centre = 10.0 * math.cos(math.radians(30.0))
    np.testing.assert_allclose(
        [altitude.min(), altitude.max()], [centre - 10, centre + 10], atol=0.2
    )
    # With the horizontal orbit assumed, the same trajectory is far from the orbit.
    assert evaluate(trajectory, 20.0)["height_rms_m"] > 5.0


def test_scenarios_accept_two_or_three_components():
    np.testing.assert_array_equal(Scenario(dim=3).launch_positions()[:, 2], 0.0)
    np.testing.assert_array_equal(
        Scenario(dim=3).launch_positions()[:, :2], Scenario().launch_positions()
    )
    raised = Scenario(dim=3, launch=(-100.0, 0.0, 5.0), n_drones=3).launch_positions()
    assert raised.shape == (3, 3)
    np.testing.assert_array_equal(raised[:, 2], 5.0)
    for kwargs in ({"dim": 4}, {"launch": (0.0, 0.0, 1.0)}, {"dim": 3, "target_velocity": (1.0,)}):
        with pytest.raises(ValueError):
            Scenario(**kwargs)
