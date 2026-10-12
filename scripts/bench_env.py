"""Throughput of the reinforcement-learning environment with batched environments.

Each environment step is one 0.1 s control step (5 physics substeps) for up to
8 drones. Every timing covers at least 200,000 environment steps and lasts at
least 0.5 s, so that it is long enough to be accurate (on a GPU the rollout is
made longer until it does); the best of three timings after compilation is
reported.

    python scripts/bench_env.py [--dim 3] [--batches 1 64 512 4096]
"""

import argparse
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from encircle import Encircle

MIN_STEPS = 200_000  # environment steps per timing
MIN_SECONDS = 0.5  # shortest timing that is reported


def best_time(env: Encircle, batch: int, steps: int) -> float:
    """Shortest of three wall times (s) of a compiled rollout of `steps` control steps."""
    keys = jax.random.split(jax.random.key(0), batch)
    _, states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.zeros((batch, env.n_max, env.dim))
    step = jax.vmap(env.step)

    @jax.jit
    def rollout(states, keys):
        def body(s, k):
            return step(jax.random.split(k, batch), s, actions)[1], None

        return jax.lax.scan(body, states, jax.random.split(keys[0], steps))[0]

    rollout(states, keys).particles.pos.block_until_ready()
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        rollout(states, keys).particles.pos.block_until_ready()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 64, 512, 4096))
    args = parser.parse_args()

    env = Encircle(dim=args.dim)
    print(f"JAX {jax.__version__} on {jax.devices()[0].platform.upper()}, {args.dim}D")
    print("| Parallel environments | Environment steps per second |")
    print("|---|---|")
    for batch in args.batches:
        steps = max(20, MIN_STEPS // batch)
        best = best_time(env, batch, steps)
        while best < MIN_SECONDS:
            steps = math.ceil(1.5 * steps * MIN_SECONDS / best)
            best = best_time(env, batch, steps)
        print(f"| {batch} | {batch * steps / best:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
