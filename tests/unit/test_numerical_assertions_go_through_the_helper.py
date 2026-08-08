"""The rule that every numerical assertion goes through one helper, enforced.

Written as a check rather than as a sentence in a document, because a sentence
does not refuse anything. What it refuses is a test comparing floating point
values with a library call that carries its own default tolerance: those
defaults are invisible at the call site, they differ between libraries, and a
test that inherits one is a test whose tolerance nobody chose.

The scan is over the test tree only. Source under ``src/`` is not a test and is
not asserted against.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
HELPER = TESTS / "support" / "tolerance.py"

# Each pattern is a numerical comparison that carries a tolerance the call site
# does not state. The name is what the failure message reports.
FORBIDDEN = {
    "pytest.approx": re.compile(r"\bapprox\s*\("),
    "numpy.testing.assert_allclose": re.compile(r"\bassert_allclose\s*\("),
    "numpy.testing.assert_almost_equal": re.compile(r"\bassert_almost_equal\s*\("),
    "numpy.testing.assert_array_almost_equal": re.compile(r"\bassert_array_almost_equal\s*\("),
    "math.isclose": re.compile(r"\bisclose\s*\("),
}


def test_no_test_compares_numbers_outside_the_tolerance_helper() -> None:
    offences: list[str] = []
    for path in sorted(TESTS.rglob("*.py")):
        if path == HELPER or path == Path(__file__).resolve():
            # The helper is where the comparison is allowed to happen, and this
            # file has to hold the patterns in order to look for them.
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for name, pattern in FORBIDDEN.items():
                if pattern.search(line):
                    relative = path.relative_to(TESTS.parent).as_posix()
                    offences.append(f"{relative}:{lineno} uses {name}")

    assert not offences, (
        "numerical comparisons must go through tests.support.tolerance.assert_close, "
        "which takes the tolerance as an argument and states it in the failure "
        "message. Found:\n  " + "\n  ".join(offences)
    )
