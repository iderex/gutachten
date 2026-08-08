"""The same seed gives the same surface, and this is where "same" got measured.

The unit tests show that two calls in one process agree. That is the easy half.
The half that matters for a sensitivity study is whether a surface generated on
a Linux runner is the surface generated on Windows, because otherwise a recorded
result cannot be reproduced by anyone who is not on the machine that recorded it.

It is not, and that was measured rather than assumed. One canonical surface
hashed to three different digests on the three platforms of the build job:

    windows  69543b5f358cbd46ac45c8a2abf5cfae5b582e28b8ff196a1ca60ead498311eb
    ubuntu   d6d7fd38de94a6a0f07df8419499835ede6a03e92a78c217ac1fe63058c34fb4
    macos    eac324e689a444bdefa2b28f1d761986b5f80c7f99159d41f778b2104759c8c5

Every draw comes from a `numpy.random.Generator`, which is bit-reproducible by
design, so the divergence is not in the random numbers. It is in the
transcendental functions: sine, cosine and the square root inside `hypot` are
supplied by the platform's maths library and by whichever vector path NumPy
takes, and those agree to within an ulp rather than exactly.

So this test compares against a recorded surface at a stated tolerance instead of
comparing digests, and the tolerance is what makes the statement honest. What is
still refused is any change to the generator that moves a height by more than a
picometre, which is every change anyone would make on purpose. What is no longer
claimed is bit-identity, because it does not hold.

Whether bit-identity is required here, and what it would cost to get it, is not
this issue's to settle. It is the determinism issue's, and this measurement is
the evidence it needs.

To accept a deliberate change to the generator, regenerate the recording in the
same commit, so the recording and the reason for it arrive together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gutachten.synth import SurfaceParameters, generate
from tests.support.tolerance import assert_close

RECORDING = Path(__file__).with_name("synthetic_surface.npy")

# Fixed here rather than taken from the defaults, so that changing a default
# does not silently change what this test is about.
CANONICAL = SurfaceParameters(
    rows=64,
    columns=64,
    pixel_spacing_um=4.0,
    striae_depth_um=2.0,
    striae_spacing_um=40.0,
    striae_angle_deg=12.0,
    form_depth_um=12.0,
    firing_pin_radius_um=60.0,
    firing_pin_depth_um=25.0,
    drag_mark_width_um=32.0,
    drag_mark_depth_um=6.0,
    noise_um=0.3,
    edge_dropout_fraction=0.05,
    seed=20260808,
)

# A picometre, in the micrometre units the surface is in. Nine orders of
# magnitude below the smallest feature the generator produces, and six orders
# above the last-place disagreement between two platforms' maths libraries on
# values of this size. A real change to the generator lands far outside it; a
# change of platform lands far inside.
PLATFORM_TOLERANCE_UM = 1e-9


def test_the_canonical_surface_matches_its_recording() -> None:
    recorded = np.load(RECORDING)
    observed = generate(CANONICAL).heights_um

    assert observed.shape == recorded.shape, (
        f"the canonical surface changed shape, from {recorded.shape} to {observed.shape}"
    )

    # Where there is no measurement is compared exactly and separately. A
    # missing sample that became a number, or the reverse, is not a small
    # numerical difference and no tolerance should be able to absorb it.
    assert np.array_equal(np.isnan(observed), np.isnan(recorded)), (
        "the pattern of missing data in the canonical surface moved, which is a "
        "change in what was measured rather than in a measured value"
    )

    present = np.isfinite(recorded)
    assert_close(
        observed[present],
        recorded[present],
        what="canonical synthetic surface against its recording",
        atol=PLATFORM_TOLERANCE_UM,
    )
