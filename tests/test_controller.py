"""The encirclement controller: field geometry, orbit accuracy, spacing and safety."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from encircle import (
    EncircleController,
    Scenario,
    angular_gaps,
    evaluate,
    guidance_field,
    simulate,
)


def simulate_one(controller, n, **kwargs):
    return simulate(Scenario(n_drones=n, **kwargs), controller, jax.random.key(1))


def test_guidance_field_has_unit_speed_and_circulates_counterclockwise():
    rng = np.random.default_rng(0)
    rel = jnp.asarray(rng.uniform(-80, 80, size=(200, 2)))
    np.testing.assert_allclose(np.linalg.norm(guidance_field(rel, 20.0), axis=1), 1.0)

    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    tangent = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    on_circle = guidance_field(jnp.asarray(20.0 * unit), 20.0)
    np.testing.assert_allclose(on_circle, tangent, atol=1e-12)
    outside = guidance_field(jnp.asarray(40.0 * unit), 20.0)
    inside = guidance_field(jnp.asarray(10.0 * unit), 20.0)
    assert np.all(np.sum(np.asarray(outside) * unit, axis=1) < 0)  # inwards
    assert np.all(np.sum(np.asarray(inside) * unit, axis=1) > 0)  # outwards


def test_angular_gaps_skip_inactive_drones():
    theta = jnp.array([0.0, 1.0, 2.5, 3.0, -2.0])
    active = jnp.array([True, True, False, True, True])
    ahead, behind = angular_gaps(theta, active)
    # Active drones in counterclockwise order: -2.0, 0.0, 1.0, 3.0.
    np.testing.assert_allclose(ahead, [1.0, 2.0, 0.0, 2 * math.pi - 5.0, 2.0])
    np.testing.assert_allclose(behind, [2.0, 1.0, 0.0, 2.0, 2 * math.pi - 5.0])


def test_feed_forward_removes_the_orbit_bias_of_the_velocity_lag():
    lagging = evaluate(
        simulate_one(EncircleController(response_time=0.0, radial_gain=0.0), n=1), 20.0
    )
    compensated = evaluate(simulate_one(EncircleController(), n=1), 20.0)
    assert lagging["radial_rms_m"] > 1.0  # about omega * tau * R = 2.5 m
    assert compensated["radial_rms_m"] < 0.2


@pytest.mark.parametrize("n", [2, 5, 8])
def test_drones_spread_evenly_without_collisions(n):
    result = evaluate(simulate_one(EncircleController(), n=n), 20.0)
    assert result["convergence_time_s"] < 60.0
    assert result["spacing_cv"] < 0.01
    assert result["radial_rms_m"] < 0.2
    assert result["min_distance_m"] > 2.0
    assert result["max_speed_mps"] <= 8.0 + 1e-6


def test_gusts_and_a_moving_target_are_tolerated():
    trajectory = simulate_one(EncircleController(), n=8, wind=0.3, target_velocity=(2.0, 0.0))
    result = evaluate(trajectory, 20.0)
    assert result["convergence_time_s"] < 80.0
    assert result["radial_rms_m"] < 0.5
    assert result["spacing_cv"] < 0.05
    assert result["min_distance_m"] > 1.5
