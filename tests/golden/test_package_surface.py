"""A golden test: a whole thing compared against a recorded output.

Most defects in a numerical pipeline are not crashes. They are a number, or a
shape, or a set of names that quietly changed, and the run that produced them
exited zero. A golden test is the layer that notices that, and it works by
recording the whole output rather than by asserting the parts somebody thought
to check.

There is no pipeline yet, so the whole thing recorded here is the importable
surface of the package: every module and subpackage under ``gutachten``. That is
a real recording of a real output and it fails for the real reason a golden test
fails, which is that something changed and the change was not deliberate. When
the pipeline exists, its recorded runs join this directory and this test stays as
the one that guards the layout.

To accept a deliberate change, update the recording in the same commit that
causes it, so the diff shows both the cause and the consequence.
"""

from __future__ import annotations

import pkgutil
from pathlib import Path

import gutachten

RECORDING = Path(__file__).with_name("package_surface.txt")


def observed_surface() -> list[str]:
    names = [gutachten.__name__]
    names += [
        module.name
        for module in pkgutil.walk_packages(gutachten.__path__, prefix="gutachten.")
    ]
    return sorted(names)


def test_the_importable_surface_matches_the_recording() -> None:
    observed = observed_surface()
    recorded = [
        line.strip()
        for line in RECORDING.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    missing = sorted(set(recorded) - set(observed))
    added = sorted(set(observed) - set(recorded))

    assert not missing and not added, (
        f"the importable surface of gutachten has moved away from its recording. "
        f"Gone: {missing}. New: {added}. If this change is deliberate, update "
        f"{RECORDING.name} in the same commit, so the recording and the reason "
        f"for it arrive together."
    )
