"""What the same run produces on Linux, macOS and Windows.

This is the measurement leg. The cross-platform tolerance this project promises
has never been measured, which `docs/determinism.md` says in those words, and a
number written here before the three platforms have been read would be the hope
that document refuses.

So this reports and enforces nothing. It runs the fixture, prints what this
platform produced at full precision, and skips, so the three matrix legs of one
workflow run carry the three readings between them. The enforcing test replaces
this one in the same branch, with the tolerance taken from what came back.

A skip is used rather than a print because the gate runs `pytest` with `-ra`,
which reports a skip reason and captures the stdout of a passing test. A run
that covered less than the whole suite says so here, which is the rule this
repository already holds for every other skip.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

from tests.golden.determinism_fixture import one_run, summary


def test_what_this_platform_produces() -> None:
    surface, _manifest = one_run()
    observed = summary(surface)

    reading = ", ".join(f"{name}={value!r}" for name, value in sorted(observed.items()))
    pytest.skip(
        f"measurement leg for issue 27, enforcing nothing yet. "
        f"platform={platform.system()} machine={platform.machine()} "
        f"python={sys.version.split()[0]} numpy={np.__version__} :: {reading}"
    )
