"""Run the encirclement controller for 1 to 8 drones and harder cases; print a Markdown table.

python scripts/run_controller.py [--json results.json]
"""

import argparse
import json

import jax

from encircle import EncircleController, Scenario, evaluate, simulate

CASES = [(f"{n} drone{'s' if n > 1 else ''}", Scenario(n_drones=n)) for n in range(1, 9)]
CASES += [
    ("8 drones, gusts 0.3 m/s", Scenario(wind=0.3)),
    ("8 drones, target moving at 2 m/s", Scenario(target_velocity=(2.0, 0.0))),
    ("8 drones, gusts and moving target", Scenario(wind=0.3, target_velocity=(2.0, 0.0))),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()

    controller = EncircleController()
    results = []
    print(
        "| Case | Convergence (s) | Radial RMS (m) | Spacing CV | "
        "Min. separation (m) | Max. speed (m/s) |"
    )
    print("|---|---|---|---|---|---|")
    for name, scenario in CASES:
        trajectory = simulate(scenario, controller, jax.random.key(1))
        metrics = evaluate(trajectory, controller.radius)
        results.append({"case": name, **metrics})
        spacing = "n/a" if scenario.n_drones == 1 else f"{metrics['spacing_cv']:.3f}"
        separation = "n/a" if scenario.n_drones == 1 else f"{metrics['min_distance_m']:.1f}"
        print(
            f"| {name} | {metrics['convergence_time_s']:.1f} | {metrics['radial_rms_m']:.2f} | "
            f"{spacing} | {separation} | {metrics['max_speed_mps']:.1f} |"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
