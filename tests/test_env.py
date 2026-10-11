"""The reinforcement-learning environment: API, masking, batching and rewards."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from particlegym import physics as pp

from encircle import Encircle, EncircleParams, EncircleState


def test_reset_draws_a_number_of_drones_and_masks_the_rest():
    env = Encircle(n_max=8, n_min=1)
    reset = jax.jit(env.reset)
    counts = []
    for seed in range(20):
        obs, state = reset(jax.random.key(seed))
        alive = np.asarray(state.particles.alive)
        counts.append(alive.sum())
        assert obs.shape == (8, env.observation_size)
        assert np.all(np.asarray(obs)[~alive] == 0.0)
        distance = np.linalg.norm(np.asarray(state.particles.pos), axis=1)
        assert np.all(distance > 80.0)  # launched far from the target
    assert min(counts) >= 1 and max(counts) <= 8 and len(set(counts)) > 3


def test_step_compiles_and_batches():
    env = Encircle()
    keys = jax.random.split(jax.random.key(0), 16)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.zeros((16, 8, 2))
    obs, states, reward, done, info = jax.jit(jax.vmap(env.step))(keys, states, actions)
    assert obs.shape == (16, 8, env.observation_size)
    assert reward.shape == (16, 8) and done.shape == (16,)
    np.testing.assert_array_equal(states.step, 1)
    alive = np.asarray(states.particles.alive)
    assert np.all(np.asarray(reward)[~alive] == 0.0)
    np.testing.assert_array_equal(info["n_active"], alive.sum(axis=1))


def _ideal_state(env, n_active, params):
    """Drones evenly spaced on the orbit, flying at the orbit speed."""
    angles = 2 * math.pi * np.arange(env.n_max) / n_active
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    tangent = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    particles = pp.create_state(
        params.radius * unit,
        vel=params.speed * tangent,
        alive=np.arange(env.n_max) < n_active,
    )
    return EncircleState(particles, jnp.zeros(2), jnp.zeros(2), jnp.zeros((), jnp.int32))


def test_reward_is_zero_on_the_ideal_orbit_and_negative_elsewhere():
    env = Encircle()
    params = EncircleParams()
    ideal = _ideal_state(env, 5, params)
    reward = np.asarray(env.reward(ideal, params))
    np.testing.assert_allclose(reward, 0.0, atol=1e-9)

    moved = ideal.particles._replace(pos=ideal.particles.pos.at[0].multiply(1.2))
    reward = np.asarray(env.reward(ideal._replace(particles=moved), params))
    assert reward[0] < -0.2  # 20 % off the orbit radius
    np.testing.assert_allclose(reward[5:], 0.0)  # inactive drones

    crowded = ideal.particles._replace(pos=ideal.particles.pos.at[1].set(ideal.particles.pos[0]))
    reward = np.asarray(env.reward(ideal._replace(particles=crowded), params))
    assert reward[0] <= -params.collision_weight and reward[1] <= -params.collision_weight


def test_zero_residual_follows_the_guidance_field_to_the_orbit():
    env = Encircle(n_min=1, n_max=1)
    params = EncircleParams()
    _, state = env.reset(jax.random.key(3), params)

    def body(s, k):
        _, s, reward, _, _ = env.step(k, s, jnp.zeros((1, 2)), params)
        return s, reward[0]

    state, reward = jax.jit(
        lambda s: jax.lax.scan(body, s, jax.random.split(jax.random.key(4), 400))
    )(state)
    rho = float(jnp.linalg.norm(state.particles.pos[0] - state.target))
    assert abs(rho - params.radius) < 1.0
    assert float(reward[-1]) > float(reward[0])


def test_full_command_mode_uses_actions_as_velocities():
    env = Encircle(n_min=2, n_max=2, residual=False)
    _, state = env.reset(jax.random.key(5))
    _, state, _, _, _ = env.step(jax.random.key(6), state, jnp.array([[1.0, 0.0], [0.0, -0.5]]))
    velocity = np.asarray(state.particles.vel)
    expected = np.array([[8.0, 0.0], [0.0, -4.0]]) * (1 - np.exp(-0.1 / 0.5))
    np.testing.assert_allclose(velocity, expected, rtol=1e-9)
