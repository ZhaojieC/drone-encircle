"""Throughput of the reinforcement-learning environment with batched environments.

Each environment step is one 0.1 s control step (5 physics substeps) for up to
8 drones. Every timing covers about 200,000 environment steps, so that it is
long enough to be accurate; the best of three timings after compilation is
reported.

    python scripts/bench_env.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from encircle import Encircle

TOTAL_STEPS = 200_000


def main() -> None:
    env = Encircle()
    print(f"JAX {jax.__version__} on {jax.devices()[0].platform.upper()}")
    print("| Parallel environments | Environment steps per second |")
    print("|---|---|")
    for batch in (1, 64, 512, 4096):
        steps = max(20, TOTAL_STEPS // batch)
        keys = jax.random.split(jax.random.key(0), batch)
        _, states = jax.jit(jax.vmap(env.reset))(keys)
        actions = jnp.zeros((batch, env.n_max, 2))
        step = jax.vmap(env.step)

        @jax.jit
        def rollout(states, keys, actions=actions, step=step, steps=steps, batch=batch):
            def body(s, k):
                return step(jax.random.split(k, batch), s, actions)[1], None

            return jax.lax.scan(body, states, jax.random.split(keys[0], steps))[0]

        rollout(states, keys).particles.pos.block_until_ready()
        best = np.inf
        for _ in range(3):
            start = time.perf_counter()
            rollout(states, keys).particles.pos.block_until_ready()
            best = min(best, time.perf_counter() - start)
        print(f"| {batch} | {batch * steps / best:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
