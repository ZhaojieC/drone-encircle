"""Render a scene written by scripts/export_scene.py with OVITO (Tachyon renderer).

Drones are blue, the target is red, the orbit is the dark ring (the grey ring
is its projection on the ground), and thin lines trace the paths of the
drones. The camera is fixed. Run with a Python that has the `ovito` package;
without a display:

    QT_QPA_PLATFORM=offscreen python scripts/render_ovito.py scene [--still FRAME]

The frames are written to scene/frames/; join them with

    ffmpeg -framerate 15 -i scene/frames/frame%04d.png -c:v libx264 -pix_fmt yuv420p scene.mp4
"""

import argparse
import pathlib

import numpy as np
from ovito.data import TriangleMesh
from ovito.io import import_file
from ovito.modifiers import GenerateTrajectoryLinesModifier
from ovito.vis import TachyonRenderer, Viewport

BLUE, RED = (0.0, 0.3, 1.0), (1.0, 0.0, 0.0)
MESHES = {
    "orbit.obj": (0.2, 0.2, 0.2),
    "footprint.obj": (0.55, 0.55, 0.55),
    "ground.obj": (1.0, 1.0, 1.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", help="directory written by export_scene.py")
    parser.add_argument("--still", type=int, default=None, help="render one frame to a PNG")
    parser.add_argument("--size", type=int, nargs=2, default=(960, 540))
    parser.add_argument("--camera", type=float, nargs=3, default=(-42.0, -150.0, 95.0))
    parser.add_argument("--look-at", type=float, nargs=3, default=(-42.0, 0.0, 0.0))
    parser.add_argument("--fov", type=float, default=34.0, help="field of view (degrees)")
    args = parser.parse_args()
    scene = pathlib.Path(args.scene)

    pipeline = import_file(str(scene / "scene.xyz"), multiple_frames=True)

    def colour(frame, data):
        data.cell_.vis.enabled = False
        is_target = np.asarray(data.particles["Particle Identifier"]) == data.particles.count - 1
        data.particles_.create_property("Color", data=np.where(is_target[:, None], RED, BLUE))

    pipeline.modifiers.append(colour)
    trails = GenerateTrajectoryLinesModifier(only_selected=False)
    pipeline.modifiers.append(trails)
    trails.vis.width = 0.3
    trails.vis.color = (0.45, 0.6, 1.0)
    trails.vis.upto_current_time = True
    pipeline.add_to_scene()

    for name, colour_ in MESHES.items():
        mesh_pipeline = import_file(str(scene / name))
        mesh = next(o for o in mesh_pipeline.compute().objects if isinstance(o, TriangleMesh))
        mesh.vis.color = colour_
        mesh.vis.highlight_edges = False
        mesh_pipeline.add_to_scene()

    viewport = Viewport(type=Viewport.Type.Perspective)
    camera, target = np.array(args.camera), np.array(args.look_at)
    viewport.camera_pos = tuple(camera)
    viewport.camera_dir = tuple(target - camera)
    viewport.fov = np.radians(args.fov)
    # Without ambient occlusion and shadows: both smear the thin path lines over the ground.
    renderer = TachyonRenderer(ambient_occlusion=False, shadows=False, antialiasing_samples=8)
    if args.still is not None:
        viewport.render_image(
            filename=str(scene / f"still{args.still:04d}.png"),
            size=tuple(args.size),
            frame=args.still,
            renderer=renderer,
        )
    else:
        (scene / "frames").mkdir(exist_ok=True)
        viewport.render_anim(
            filename=str(scene / "frames" / "frame.png"), size=tuple(args.size), renderer=renderer
        )


if __name__ == "__main__":
    main()
