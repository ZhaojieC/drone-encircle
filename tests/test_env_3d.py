"""The reinforcement-learning environment in 3D: shapes, altitude, rewards and actions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from particlegym import physics as pp

from encircle import Encircle, EncircleParams, EncircleState


def test_reset_starts_at_the_altitude_of_the_target():
    env = Encircle(dim=3)
    assert env.observation_size == 6 + 7 * 3 + 1
    params = EncircleParams(target_speed=2.0, target_climb=1.0)
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    alive = np.asarray(state.particles.alive)
    assert obs.shape == (8, 28)
    assert state.particles.pos.shape == (8, 3)
    np.testing.assert_array_equal(state.particles.pos[:, 2], 0.0)
    np.testing.assert_array_equal(state.target, 0.0)
    np.testing.assert_allclose(np.linalg.norm(state.target_vel[:2]), 2.0)
    assert state.target_vel[2] == 1.0
    # The orbit centre is 10 m above the target, so the drones start 0.5 R below it.
    np.testing.assert_allclose(np.asarray(obs)[alive, 2], -0.5)
    assert np.all(np.asarray(obs)[~alive] == 0.0)


def test_step_compiles_and_batches():
    env = Encircle(dim=3)
    keys = jax.random.split(jax.random.key(0), 16)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.zeros((16, 8, 3))
    obs, states, reward, done, info = jax.jit(jax.vmap(env.step))(keys, states, actions)
    assert obs.shape == (16, 8, env.observation_size)
    assert reward.shape == (16, 8) and done.shape == (16,)
    np.testing.assert_array_equal(states.step, 1)
    alive = np.asarray(states.particles.alive)
    assert np.all(np.asarray(reward)[~alive] == 0.0)
    assert np.all(np.asarray(states.particles.pos)[alive][:, 2] > 0.0)  # climbing
    np.testing.assert_array_equal(info["n_active"], alive.sum(axis=1))


def _ideal_state(env, n_active, params):
    """Drones evenly spaced on the orbit, flying at the orbit speed."""
    angles = 2 * math.pi * np.arange(env.n_max) / n_active
    unit = np.stack([np.cos(angles), np.sin(angles), np.zeros(env.n_max)], axis=1)
    tangent = np.stack([-unit[:, 1], unit[:, 0], np.zeros(env.n_max)], axis=1)
    particles = pp.create_state(
        params.radius * unit + [0.0, 0.0, params.height],
        vel=params.speed * tangent,
        alive=np.arange(env.n_max) < n_active,
    )
    return EncircleState(particles, jnp.zeros(3), jnp.zeros(3), jnp.zeros((), jnp.int32))


def test_reward_is_zero_on_the_ideal_orbit_and_counts_height_errors():
    env = Encircle(dim=3)
    params = EncircleParams(height=15.0)
    ideal = _ideal_state(env, 5, params)
    np.testing.assert_allclose(env.reward(ideal, params), 0.0, atol=1e-9)
    grad = jax.grad(
        lambda pos: env.reward(
            ideal._replace(particles=ideal.particles._replace(pos=pos)), params
        )[0]
    )(ideal.particles.pos)
    assert np.all(np.isfinite(np.asarray(grad)))

    # 4 m above the orbit: the distance to the orbit is 0.2 R.
    high = ideal.particles._replace(pos=ideal.particles.pos.at[0, 2].add(4.0))
    reward = np.asarray(env.reward(ideal._replace(particles=high), params))
    np.testing.assert_allclose(reward[0], -0.2 * params.radial_weight, atol=1e-9)
    np.testing.assert_allclose(reward[1:], 0.0, atol=1e-9)

    # 3 m outside and 4 m above: 5 m from the orbit.
    off = ideal.particles.pos.at[0].add(jnp.array([3.0, 0.0, 4.0]))
    reward = np.asarray(
        env.reward(ideal._replace(particles=ideal.particles._replace(pos=off)), params)
    )
    np.testing.assert_allclose(reward[0], -0.25 * params.radial_weight, atol=1e-9)

    # Directly above each other, closer than the safe distance.
    stacked = ideal.particles.pos.at[1].set(ideal.particles.pos[0] + jnp.array([0.0, 0.0, 2.0]))
    reward = np.asarray(
        env.reward(ideal._replace(particles=ideal.particles._replace(pos=stacked)), params)
    )
    assert reward[0] <= -params.collision_weight and reward[1] <= -params.collision_weight


def test_zero_residual_climbs_and_follows_the_guidance_field_to_the_orbit():
    env = Encircle(dim=3, n_min=1, n_max=1)
    params = EncircleParams(height=12.0)
    _, state = env.reset(jax.random.key(3), params)

    def body(s, k):
        _, s, reward, _, _ = env.step(k, s, jnp.zeros((1, 3)), params)
        return s, reward[0]

    state, reward = jax.jit(
        lambda s: jax.lax.scan(body, s, jax.random.split(jax.random.key(4), 400))
    )(state)
    rel = state.particles.pos[0] - state.target
    assert abs(float(jnp.linalg.norm(rel[:2])) - params.radius) < 1.0
    assert abs(float(rel[2]) - 12.0) < 0.05
    assert float(reward[-1]) > float(reward[0])


def test_full_command_mode_scales_vertical_actions_by_the_vertical_speed_limit():
    env = Encircle(dim=3, n_min=2, n_max=2, residual=False, max_vertical_speed=2.0)
    _, state = env.reset(jax.random.key(5))
    action = jnp.array([[1.0, 0.0, 1.0], [0.0, -0.5, -0.5]])
    _, state, _, _, _ = env.step(jax.random.key(6), state, action)
    expected = np.array([[8.0, 0.0, 2.0], [0.0, -4.0, -1.0]]) * (1 - np.exp(-0.1 / 0.5))
    np.testing.assert_allclose(state.particles.vel, expected, rtol=1e-9)


def test_the_target_moves_and_climbs():
    env = Encircle(dim=3)
    params = EncircleParams(target_speed=2.0, target_climb=-0.5)
    _, state = env.reset(jax.random.key(7), params)
    _, state, _, _, _ = env.step(jax.random.key(8), state, jnp.zeros((8, 3)), params)
    np.testing.assert_allclose(np.linalg.norm(state.target[:2]), 0.2)
    np.testing.assert_allclose(state.target[2], -0.05)


def test_configuration():
    assert Encircle(dim=3, axis=(0.0, 0.0, 2.0)).axis == (0.0, 0.0, 1.0)
    assert Encircle().observation_size == 20  # the 2D environment is unchanged
    with pytest.raises(ValueError):
        Encircle(dim=4)
    with pytest.raises(ValueError):
        Encircle(dim=3, axis=(0.0, 0.0, 0.0))
