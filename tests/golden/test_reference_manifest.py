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

from pathlib import Path

from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import (
    ComparisonRecord,
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

    The two steps are one that measured nothing and one that measured something,
    because those are the two shapes a step record has and a recording holding
    only one of them would let the other move without a diff. The rejection step
    is the case the outcomes field exists for: a sweep varies ``threshold`` and
    reads ``rejected_samples`` to see how much surface the threshold took.
    """
    return RunManifest(
        inputs=(FileRecord(role="scan-a", sha256="a" * 64),),
        profile=ProfileRecord(name="published", version="1"),
        steps=(
            StepRecord(identifier="level", version="2", parameters=(("model", "plane"),)),
            StepRecord(
                identifier="reject-outliers",
                version="1",
                parameters=(
                    ("criterion", "median-absolute-deviation"),
                    ("neighbourhood", 40.0),
                    ("threshold", 5.0),
                ),
                outcomes=(("measured_samples", 8192), ("rejected_samples", 137)),
            ),
        ),
        seed=20260808,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(
            software_version="0.0.0",
            dependencies=(("numpy", "2.1.0"), ("scipy", "1.14.0")),
        ),
        outputs=(FileRecord(role="surface", sha256="b" * 64),),
        comparison=ComparisonRecord(
            method="cell-correlation",
            version="1",
            parameters=(("grid", 8), ("minimum_valid", 0.5)),
        ),
    )


def test_a_complete_manifest_matches_its_recording() -> None:
    # `to_text`, the writer a run actually uses, rather than a serialisation
    # written out here. A recording made by the test's own copy of the writer
    # records the copy, and the two then drift in the direction where the
    # recording keeps passing.
    observed = a_reference_run().to_text()
    recorded = RECORDING.read_text(encoding="utf-8")

    assert observed == recorded, (
        f"the serialised manifest has moved away from {RECORDING.name}. If the change "
        f"is deliberate, update the recording in the same commit, and check that the "
        f"schema version moved with it where the change touches what a field means."
    )
