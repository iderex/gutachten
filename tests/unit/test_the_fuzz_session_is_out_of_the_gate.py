"""The fuzz session stays out of the default run, and the exclusion is checked.

The session damages a container until a time budget runs out. That is minutes of
wall clock and a different set of inputs every time, so on the merge path it
would slow every pull request down and still give a green that says less than it
looks like it says. What belongs in the gate is what the session produces: a
finding becomes a fixture in the conformance corpus.

None of that survives on its own. The exclusion is two lines of configuration, a
marker and a deselection, and either can be deleted by somebody who wanted a
quick answer from a local run. So this is what refuses their removal, and it
reads the configuration file rather than the runner's state, because a test
asking the running session whether it was deselected is a test that cannot run
when the answer is no.

What it cannot see is a scheduled workflow that stopped running. A workflow that
was disabled, or whose cron never fires, leaves this file green and the session
never runs at all. That gap is real, nothing in the tree can read it, and
`tests/fuzz/README.md` says where the schedule is declared so a reader can check
it against the runs on the repository.
"""

from __future__ import annotations

import itertools
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
SESSION = ROOT / "tests" / "fuzz"

configuration = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"][
    "ini_options"
]


def test_the_fuzz_marker_is_declared() -> None:
    """An undeclared marker is refused by --strict-markers, and then deselected by nothing."""
    declared = [entry.split(":", 1)[0] for entry in configuration["markers"]]
    assert "fuzz" in declared, (
        f"the markers declared are {declared}. Without the fuzz marker the deselection below "
        "selects nothing and the session runs on every pull request."
    )


def test_the_default_run_deselects_the_fuzz_session() -> None:
    options = configuration["addopts"]
    pairs = list(itertools.pairwise(options))
    assert ("-m", "not fuzz") in pairs, (
        f"the configured options are {options} and none of them deselects the fuzz marker. "
        "The session would then run on the merge path, where it adds minutes and a "
        "different set of inputs every time."
    )


def test_the_session_marks_itself() -> None:
    """A module in the directory carrying no marker is one the deselection misses."""
    unmarked = sorted(
        path.name
        for path in SESSION.glob("test_*.py")
        if "pytestmark = pytest.mark.fuzz" not in path.read_text(encoding="utf-8")
    )
    assert not unmarked, (
        f"{unmarked} sit in the fuzz directory and do not mark themselves, so the "
        "deselection does not reach them and they run in the gate."
    )
