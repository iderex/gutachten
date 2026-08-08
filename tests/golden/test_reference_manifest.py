"""A whole manifest, recorded, so the schema cannot move quietly.

The schema version is supposed to move when a field that affects a result is
added or when the meaning of one changes. Nothing can judge whether it moved for
the right reason, and this test does not try. What it does is make the version a
recorded byte rather than a number the suite reads back out of the same constant
it is checking, so a field added without the version moving arrives in a review
as a diff showing both facts at once.

It also holds the serialisation itself. Two manifests from two runs should
differ only where the runs differ, which is a property of key order and of
formatting as much as of content, and neither is visible from an assertion about
one field at a time.

To accept a deliberate change, update the recording in the same commit that
causes it.
"""

from __future__ import annotations

import json
from pathlib import Path

from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import (
    EnvironmentRecord,
    FileRecord,
    ProfileRecord,
    RunManifest,
    StepRecord,
)

RECORDING = Path(__file__).with_name("reference_manifest.json")


def a_reference_run() -> RunManifest:
    """One complete manifest, with every field populated by hand.

    Written out here rather than imported from the unit tests, because a
    recording that moves when a shared fixture moves records the fixture rather
    than the schema.
    """
    return RunManifest(
        inputs=(FileRecord(role="scan-a", sha256="a" * 64),),
        profile=ProfileRecord(name="published", version="1"),
        steps=(
            StepRecord(identifier="level", version="2", parameters=(("model", "plane"),)),
            StepRecord(
                identifier="bandpass",
                version="1",
                parameters=(("lower_um", 25.0), ("upper_um", 250.0)),
            ),
        ),
        seed=20260808,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(
            software_version="0.0.0",
            dependencies=(("numpy", "2.1.0"), ("scipy", "1.14.0")),
        ),
        outputs=(FileRecord(role="surface", sha256="b" * 64),),
    )


def serialised(manifest: RunManifest) -> str:
    """The one serialisation this recording is made in.

    ``sort_keys`` is off deliberately. The schema decides the order of its own
    keys and the recording is what proves that order is stable; sorting here
    would hide a schema that had stopped deciding it.
    """
    return json.dumps(manifest.to_dict(), indent=2, sort_keys=False) + "\n"


def test_a_complete_manifest_matches_its_recording() -> None:
    observed = serialised(a_reference_run())
    recorded = RECORDING.read_text(encoding="utf-8")

    assert observed == recorded, (
        f"the serialised manifest has moved away from {RECORDING.name}. If the change "
        f"is deliberate, update the recording in the same commit, and check that the "
        f"schema version moved with it where the change touches what a field means."
    )
