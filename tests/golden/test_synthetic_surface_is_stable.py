"""The same seed gives the same surface, on every platform.

The unit tests show that two calls in one process agree. That is the easy half.
The half that matters for a sensitivity study is that a surface generated on a
Linux runner is the same surface as one generated on Windows, because otherwise
a recorded result cannot be reproduced by anyone who is not on the machine that
recorded it.

This is a golden test rather than an assertion about equality, so the recording
travels with the repository and the three platform jobs each compare against the
same bytes. A digest is used instead of the array because the array is a hundred
and fifty kilobytes and the thing being checked is whether any byte moved.

To accept a deliberate change to the generator, regenerate the digest in the same
commit, so the recording and the reason for it arrive together. Regenerating it
to make a red run go green is the failure this test exists against.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from gutachten.synth import SurfaceParameters, generate

RECORDING = Path(__file__).with_name("synthetic_surface_digest.txt")

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


def digest_of(parameters: SurfaceParameters) -> str:
    surface = generate(parameters)
    array = surface.heights_um
    # Byte order is stated rather than inherited, so a big endian machine would
    # be compared against the same bytes as a little endian one instead of
    # failing for a reason that has nothing to do with the generator.
    return hashlib.sha256(array.astype(">f8").tobytes()).hexdigest()


def test_the_canonical_surface_matches_its_recording() -> None:
    recorded = [
        line.strip()
        for line in RECORDING.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(recorded) == 1, f"{RECORDING.name} should hold exactly one digest"

    observed = digest_of(CANONICAL)
    assert observed == recorded[0], (
        f"the canonical synthetic surface no longer hashes to its recording. "
        f"Recorded {recorded[0]}, generated {observed}. Either the generator "
        f"changed, which means updating {RECORDING.name} in the same commit, or "
        f"the same seed no longer gives the same surface on this platform, which "
        f"means a recorded result cannot be reproduced anywhere else and is the "
        f"more serious of the two."
    )
