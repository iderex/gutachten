"""The one run the determinism proofs are taken over, in one place.

Two tests need the same run: the byte comparison that asserts two runs of it
agree, and the recording that asserts three platforms agree to a declared
tolerance. Written once here because a fixture that drifts between them turns a
cross-platform disagreement into a question about which of two chains produced
which number.

The chain is the ``every-step`` profile, which is the profile that exists to run
every step this tree registers. Reading it from the file rather than assembling
a chain here is what makes the coverage assertion mean something: a step added
to the tree has to be added to a profile before the suite is green at all, and
this fixture then runs it without anybody remembering to come back here.

The surface is generated, sized so the whole run takes well under a second, and
its seed is fixed and recorded. It is not the smallest surface that would work.
A field small enough that the bandpass and the levelling have nothing to bite on
would compare two runs of a chain that did almost nothing, and would agree
across platforms for the wrong reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import EnvironmentRecord, RunManifest, record_run
from gutachten.profile import load
from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.registry import REGISTRY

#: Fixed and written down rather than taken from a clock, because the whole
#: point of the run is that it is the same run every time.
SEED = 20260809

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "every-step.json"

#: Held still deliberately. The recorded numbers below are numbers about this
#: environment as much as about this code, and a run that changed either would
#: be a different measurement recorded under the same name.
ENVIRONMENT = EnvironmentRecord(
    software_version="0.0.0", dependencies=(("numpy", "recorded-elsewhere"),)
)


def the_input() -> Surface:
    """The surface the chain runs on."""
    generated = generate(SurfaceParameters(seed=SEED))
    return Surface(
        heights=generated.heights_um,
        spacing_y=generated.parameters.pixel_spacing_um,
        spacing_x=generated.parameters.pixel_spacing_um,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def the_chain() -> list[str]:
    """The identifiers the fixture runs, in order."""
    return [step.identifier for step in load(PROFILE, REGISTRY).steps]


def one_run() -> tuple[Surface, RunManifest]:
    """The whole run: generate, preprocess, record."""
    profile = load(PROFILE, REGISTRY)
    return record_run(
        role="scan-a",
        surface=the_input(),
        profile=profile.record(),
        chain=profile.chain(),
        registry=REGISTRY,
        seed=SEED,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=ENVIRONMENT,
    )


def summary(surface: Surface) -> dict[str, float]:
    """The numbers a cross-platform comparison is made over.

    Four rather than one. A mean alone hides a sign flip in half the field
    against the other half, and the extremes are where a difference in a library
    build shows up first because they are where the arithmetic ran furthest.
    ``measured`` is a count rather than a height: a platform that dropped one
    more sample than another has disagreed about the shape of the mask, which
    would move every other number here for a reason that is not arithmetic.
    """
    heights = surface.heights[np.isfinite(surface.heights)]
    return {
        "measured": float(heights.size),
        "mean_um": float(heights.mean()),
        "deviation_um": float(heights.std()),
        "lowest_um": float(heights.min()),
        "highest_um": float(heights.max()),
    }
