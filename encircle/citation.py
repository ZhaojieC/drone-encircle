# drone-encircle, https://github.com/ZhaojieC/drone-encircle (MIT license).
# In published work, please cite ParticleGym and the papers in the README's References.
"""The works to cite when using drone-encircle.

:func:`remind` prints them once per process, like the citation reminder of
LAMMPS: the first time an :class:`~encircle.Encircle` environment is created
or :func:`~encircle.simulate` runs. Set the environment variable
``DRONE_ENCIRCLE_CITE=0`` to turn it off.
"""

from __future__ import annotations

import os
import sys

REFERENCES = (
    "ParticleGym, the particle dynamics: Z. Chai, https://github.com/ZhaojieC/particlegym",
    "the guidance vector field: D. A. Lawrence, E. W. Frew and W. J. Pisano, "
    "J. Guidance, Control, and Dynamics 31(5), 1220 (2008)",
    "standoff tracking of moving targets: E. W. Frew, D. A. Lawrence and S. Morris, "
    "J. Guidance, Control, and Dynamics 31(2), 290 (2008)",
)

_reminded = False


def remind() -> None:
    """Print the works to cite, once per process, unless ``DRONE_ENCIRCLE_CITE=0``."""
    global _reminded
    if _reminded or os.environ.get("DRONE_ENCIRCLE_CITE", "1").strip() == "0":
        return
    _reminded = True
    lines = "\n".join(f"- {reference}" for reference in REFERENCES)
    print(
        "drone-encircle: if you publish results obtained with it, please cite\n"
        f"{lines}\n"
        "The README lists the full references. Set DRONE_ENCIRCLE_CITE=0 to hide this note.",
        file=sys.stderr,
    )
