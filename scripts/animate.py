"""Animate 1, 2, 4 and 8 drones flying to a target and encircling it (GIF).

python scripts/animate.py [--output encircle.gif] [--seconds 60]
"""

import argparse

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from encircle import EncircleController, Scenario, simulate

# Reference palette (light mode): one series colour for the drones, the next
# categorical slot for the target, ink and hairline tokens for everything else.
SURFACE = "#fcfcfb"
DRONE = "#2a78d6"
TRAIL = "#86b6ef"
TARGET = "#eb6834"
ORBIT = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
COUNTS = (1, 2, 4, 8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="encircle.gif")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--frame-interval", type=float, default=0.5, help="simulated seconds")
    args = parser.parse_args()

    controller = EncircleController()
    runs = [
        simulate(Scenario(n_drones=n, duration=args.seconds), controller, jax.random.key(1))
        for n in COUNTS
    ]
    period = Scenario().control_period
    stride = max(1, round(args.frame_interval / period))
    trail = round(5.0 / period)  # show the last 5 s of each path
    frames = range(0, len(runs[0].time), stride)

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), dpi=72)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.05, hspace=0.18)
    clock = fig.text(0.02, 0.95, "", color=INK, fontsize=13, fontweight="bold")
    fig.text(
        0.98,
        0.95,
        "drones (blue) fly to the target (orange) and orbit it at R = 20 m",
        color=INK_2,
        ha="right",
        fontsize=10,
    )

    artists = []
    circle = np.linspace(0, 2 * np.pi, 200)
    for ax, n, run in zip(axes.flat, COUNTS, runs, strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-108, 30)
        ax.set_ylim(-34, 34)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#e1e0d9")
            spine.set_linewidth(0.8)
        ax.set_title(f"{n} drone{'s' if n > 1 else ''}", color=INK, loc="left", fontsize=11)
        ax.plot(20 * np.cos(circle), 20 * np.sin(circle), color=ORBIT, lw=1.0, zorder=1)
        target = ax.plot([], [], "o", color=TARGET, ms=8, mec=SURFACE, mew=2, zorder=4)[0]
        paths = [ax.plot([], [], color=TRAIL, lw=1.5, zorder=2)[0] for _ in range(n)]
        dots = ax.plot([], [], "o", color=DRONE, ms=7, mec=SURFACE, mew=1.5, zorder=3)[0]
        artists.append((run, target, paths, dots))
    # Scale bar in the first panel.
    axes.flat[0].plot([-100, -80], [-29, -29], color=MUTED, lw=1.5)
    axes.flat[0].text(-90, -27, "20 m", color=MUTED, ha="center", va="bottom", fontsize=9)

    def update(frame):
        clock.set_text(f"t = {float(runs[0].time[frame]):4.1f} s")
        changed = [clock]
        for run, target, paths, dots in artists:
            pos = np.asarray(run.pos)
            tgt = np.asarray(run.target[frame])
            target.set_data([tgt[0]], [tgt[1]])
            start = max(0, frame - trail)
            for i, path in enumerate(paths):
                path.set_data(pos[start : frame + 1, i, 0], pos[start : frame + 1, i, 1])
            dots.set_data(pos[frame, :, 0], pos[frame, :, 1])
            changed += [target, dots, *paths]
        return changed

    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save(args.output, writer=PillowWriter(fps=12))
    print(f"wrote {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
