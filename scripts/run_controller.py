"""Run the encirclement controller for 1 to 8 drones and harder cases; print a Markdown table.

python scripts/run_controller.py [--dim 3] [--json results.json]
"""

import argparse
import json
import math

import jax

from encircle import EncircleController, Scenario, evaluate, simulate

TILT = math.radians(30.0)


def cases(dim: int) -> list[tuple[str, Scenario, EncircleController]]:
    """Name, scenario and controller of every case of the table."""
    controller = EncircleController()
    names = [f"{n} drone{'s' if n > 1 else ''}" for n in range(1, 9)]
    rows = [(name, {"n_drones": n}, controller) for n, name in enumerate(names, start=1)]
    rows += [
        ("8 drones, gusts 0.3 m/s", {"wind": 0.3}, controller),
        ("8 drones, target moving at 2 m/s", {"target_velocity": (2.0, 0.0)}, controller),
        (
            "8 drones, gusts and moving target",
            {"wind": 0.3, "target_velocity": (2.0, 0.0)},
            controller,
        ),
    ]
    if dim == 3:
        climbing = {"target_velocity": (2.0, 0.0, 1.0)}
        tilted = EncircleController(axis=(0.0, math.sin(TILT), math.cos(TILT)))
        rows += [
            ("8 drones, target moving at 2 m/s and climbing at 1 m/s", climbing, controller),
            ("8 drones, gusts, moving and climbing target", {"wind": 0.3, **climbing}, controller),
            ("8 drones, orbit tilted by 30°", {}, tilted),
        ]
    return [(name, Scenario(dim=dim, **kwargs), c) for name, kwargs, c in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()

    columns = ["Case", "Convergence (s)", "Radial RMS (m)"]
    columns += ["Height RMS (m)"] if args.dim == 3 else []
    columns += ["Spacing CV", "Min. separation (m)", "Max. speed (m/s)"]
    columns += ["Max. vertical speed (m/s)"] if args.dim == 3 else []
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    results = []
    for name, scenario, controller in cases(args.dim):
        trajectory = simulate(scenario, controller, jax.random.key(1))
        metrics = evaluate(
            trajectory, controller.radius, height=controller.height, axis=controller.axis
        )
        results.append({"case": name, **metrics})
        alone = scenario.n_drones == 1
        cells = [name, f"{metrics['convergence_time_s']:.1f}", f"{metrics['radial_rms_m']:.2f}"]
        cells += [f"{metrics['height_rms_m']:.2f}"] if args.dim == 3 else []
        cells += [
            "n/a" if alone else f"{metrics['spacing_cv']:.3f}",
            "n/a" if alone else f"{metrics['min_distance_m']:.1f}",
            f"{metrics['max_speed_mps']:.1f}",
        ]
        cells += [f"{metrics['max_vertical_speed_mps']:.1f}"] if args.dim == 3 else []
        print("| " + " | ".join(cells) + " |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
