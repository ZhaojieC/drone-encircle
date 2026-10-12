"""Write a 3D encirclement run as files that OVITO can render.

Writes, into the output directory, the trajectory of the drones and the
target (scene.xyz, extended XYZ; species D and T), the orbit as a thin torus
(orbit.obj), its vertical projection on the ground (footprint.obj) and a
ground plane at the altitude of the target (ground.obj).
scripts/render_ovito.py turns them into a video.

    python scripts/export_scene.py --output scene [--drones 8] [--seconds 45]
"""

import argparse
import pathlib

import jax
import numpy as np

from encircle import EncircleController, Scenario, orbit_frame, simulate


def write_obj(path, vertices: np.ndarray, quads: np.ndarray) -> None:
    """Write a mesh of quadrilaterals as triangles."""
    with open(path, "w") as f:
        for x, y, z in vertices:
            f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
        for a, b, c, d in quads + 1:
            f.write(f"f {a} {b} {c}\nf {a} {c} {d}\n")


def torus(centre, frame, radius: float, tube: float, segments: int = 180, sides: int = 10):
    """Vertices and quadrilaterals of a torus around the axis ``frame[2]``."""
    e1, e2, n = frame
    u = 2 * np.pi * np.arange(segments) / segments
    v = 2 * np.pi * np.arange(sides) / sides
    outward = np.cos(u)[:, None] * e1 + np.sin(u)[:, None] * e2  # (segments, 3)
    ring = (radius + tube * np.cos(v))[None, :, None] * outward[:, None, :]
    vertices = centre + ring + tube * np.sin(v)[None, :, None] * n
    i, j = np.meshgrid(np.arange(segments), np.arange(sides), indexing="ij")
    i1, j1 = (i + 1) % segments, (j + 1) % sides
    quads = np.stack([i * sides + j, i1 * sides + j, i1 * sides + j1, i * sides + j1], axis=-1)
    return vertices.reshape(-1, 3), quads.reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="scene", help="output directory")
    parser.add_argument("--drones", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--every", type=int, default=3, help="control steps per frame")
    parser.add_argument("--drone-radius", type=float, default=1.5, help="drawn size (m)")
    args = parser.parse_args()

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    controller = EncircleController()
    scenario = Scenario(n_drones=args.drones, duration=args.seconds, wind=args.wind, dim=3)
    trajectory = simulate(scenario, controller, jax.random.key(1))
    pos, target = np.asarray(trajectory.pos), np.asarray(trajectory.target)
    time = np.asarray(trajectory.time)

    lo, hi = pos.min(axis=(0, 1)) - 25.0, pos.max(axis=(0, 1)) + 25.0
    cell = " ".join(f"{x:.6g}" for x in np.diag(hi - lo).ravel())
    origin = " ".join(f"{x:.6g}" for x in lo)
    header = (
        f'Lattice="{cell}" Origin="{origin}" '
        'Properties=species:S:1:pos:R:3:radius:R:1:id:I:1 pbc="F F F"'
    )
    with open(out / "scene.xyz", "w") as f:
        for k in range(0, len(time), args.every):
            f.write(f"{args.drones + 1}\n{header} Time={time[k]:.6g}\n")
            for i, (x, y, z) in enumerate(pos[k]):
                f.write(f"D {x:.5f} {y:.5f} {z:.5f} {args.drone_radius} {i}\n")
            x, y, z = target[k]
            f.write(f"T {x:.5f} {y:.5f} {z:.5f} {1.5 * args.drone_radius} {args.drones}\n")

    frame = orbit_frame(controller.axis)
    centre = target[-1] + controller.height * frame[2]
    write_obj(out / "orbit.obj", *torus(centre, frame, controller.radius, 0.2))
    # The ground is drawn one drone radius below the launch altitude, so that
    # the drones rest on it before take-off.
    z = -args.drone_radius
    vertices, quads = torus(centre, frame, controller.radius, 0.2)
    vertices[:, 2] = z + 0.05 + 0.02 * (vertices[:, 2] - centre[2])  # flattened onto the ground
    write_obj(out / "footprint.obj", vertices, quads)
    corners = np.array(
        [[lo[0], lo[1], z], [hi[0], lo[1], z], [hi[0], hi[1], z], [lo[0], hi[1], z]]
    )
    write_obj(out / "ground.obj", corners, np.array([[0, 1, 2, 3]]))
    frames = len(range(0, len(time), args.every))
    print(f"{frames} frames of {args.every * scenario.control_period:.2f} s written to {out}/")


if __name__ == "__main__":
    main()
