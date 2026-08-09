"""What one registration costs in seconds, measured rather than guessed.

Out of the gate because the number is a property of the machine and not of the
tree. A wall clock assertion in the suite fails on a loaded build agent and
passes on an idle one, and neither run says anything about the code. What the
suite does assert is the part that is the same everywhere:
``Registration.correlations``, which is arithmetic on the parameters.

The requirement this directory is named for is a machine that is otherwise idle.
A timing taken while something else is running measures the something else.

Run it with:

    uv run python harness/quiet-machine/registration_cost.py

It prints one row per field size, the environment it ran in, and nothing else.
Paste the output into `docs/registration.md` together with the machine, so a
sweep design can be costed from a number somebody actually observed.
"""

from __future__ import annotations

# The pin has to be applied before the numerical stack loads, because a BLAS
# reads its thread count once, when it is imported. Every other import in this
# file is therefore below it.
from gutachten.determinism import pin_threads

DETERMINISM = pin_threads()

import platform  # noqa: E402
import time  # noqa: E402

import numpy  # noqa: E402
import scipy  # noqa: E402

from gutachten.compare.register import RegistrationParameters, angles, register  # noqa: E402
from gutachten.surface import AxisOrientation, LengthUnit, Surface  # noqa: E402
from gutachten.synth import SurfaceParameters, generate  # noqa: E402

SIZES = (192, 384, 768)
SPACING_UM = 4.0

SETTINGS = RegistrationParameters(
    grid=6,
    minimum_valid=0.5,
    rotation_range_deg=10.0,
    rotation_step_deg=2.0,
    translation_limit=8,
)


def a_pair(size: int) -> tuple[Surface, Surface]:
    """A matching pair of the stated size, with only the striae on it."""
    settings = SurfaceParameters(
        rows=size,
        columns=size,
        pixel_spacing_um=SPACING_UM,
        striae_angle_deg=0.0,
        striae_spacing_um=40.0,
        form_depth_um=0.0,
        firing_pin_depth_um=0.0,
        drag_mark_depth_um=0.0,
        noise_um=0.0,
        seed=20260809,
    )
    first = generate(settings, source_id=11)
    second = generate(settings, source_id=11, rotation_deg=6.0, translation_px=(2.0, -3.0))

    def wrap(heights: numpy.ndarray, source: str) -> Surface:
        return Surface(
            heights=heights,
            spacing_y=SPACING_UM,
            spacing_x=SPACING_UM,
            unit=LengthUnit.MICROMETRE,
            orientation=AxisOrientation.Y_DOWN,
            source=source,
        )

    return wrap(second.heights_um, "subject"), wrap(first.heights_um, "reference")


def main() -> None:
    print(f"mode: {DETERMINISM.mode.value}, threads pinned to {DETERMINISM.threads}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"processor: {platform.processor() or 'not reported by the platform'}")
    print(f"numpy {numpy.__version__}, scipy {scipy.__version__}")
    print(
        f"grid={SETTINGS.grid} minimum_valid={SETTINGS.minimum_valid} "
        f"rotation_range_deg={SETTINGS.rotation_range_deg} "
        f"rotation_step_deg={SETTINGS.rotation_step_deg} "
        f"translation_limit={SETTINGS.translation_limit} "
        f"angles={len(angles(SETTINGS))}"
    )
    print()
    print("field      correlations   seconds   ms per correlation")
    for size in SIZES:
        subject, reference = a_pair(size)
        # One run rather than a best of several. What a sweep will pay is what
        # one search takes on a machine doing this and nothing else, and a
        # minimum over repeats is a number nobody's sweep will observe.
        start = time.perf_counter()
        found = register(subject, reference, SETTINGS)
        elapsed = time.perf_counter() - start
        each = 1000.0 * elapsed / found.correlations
        print(f"{size:4d}x{size:<4d} {found.correlations:12d} {elapsed:9.2f} {each:20.2f}")


if __name__ == "__main__":
    main()
