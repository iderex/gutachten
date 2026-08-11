"""The parity record is refused when it stops describing the workflows in the tree.

`docs/parity.json` says what quality target this repository is held to, what was
decided about each check in it, and what check names this repository produces. The
last of those is the part a maintainer reads when deciding which checks hold a
merge, and it is the part that rots first. A job renamed in a workflow leaves the
name in the record pointing at nothing, and a workflow added leaves a check
running that the record never mentions. Both are silent, and the second is worse:
a set chosen from a list that was missing an entry is a set with a hole in it.

So the names are derived from the workflow files here rather than trusted, in
both directions.

## What the reader below can and cannot see

It reads the workflow files as text. There is no YAML parser in this project's
dependency graph, and adding a second parser for one check is a cost the record
does not justify, so what is here matches the shape these files actually have:
`jobs:` at column zero, a job identifier two spaces in, an optional `name:` four
spaces in, and a matrix expanded from the `include:` list. It refuses rather than
guesses where that shape does not hold, because a reader that silently finds no
jobs would turn this whole module green while checking nothing.

One check name in the record comes from nowhere in this tree. The code scanning
run named `zizmor` is created by the SARIF upload rather than by a job, so
nothing here can derive it. It is declared with the source it comes from, and
the test below asserts that it declares one rather than asserting it exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "docs" / "parity.json"
PAGE = ROOT / "docs" / "parity.md"
WORKFLOWS = ROOT / ".github" / "workflows"

DECISIONS = ("ported", "adapted", "not-applicable")

#: A job identifier: two spaces, a name, a colon, nothing else on the line.
JOB = re.compile(r"^ {2}([A-Za-z0-9_-]+):\s*$")
#: The job's own `name:`, four spaces in. A step's `name:` sits deeper and a
#: workflow's sits at column zero, so neither is matched here.
JOB_NAME = re.compile(r"^ {4}name:\s*(.+?)\s*$")
#: What a matrix substitution looks like inside a job name.
PLACEHOLDER = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")

record = json.loads(RECORD.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Workflow:
    """One workflow file, read for the two things the record claims about it."""

    path: Path
    on_pull_request: bool
    check_names: tuple[str, ...]

    @property
    def relative(self) -> str:
        return self.path.relative_to(ROOT).as_posix()


def _block(lines: list[str], key: str) -> list[str]:
    """The lines under a column-zero key, up to the next column-zero key."""
    inside: list[str] = []
    collecting = False
    for line in lines:
        if line.startswith(f"{key}:"):
            collecting = True
            continue
        if collecting and line and not line[0].isspace() and not line.startswith("#"):
            break
        if collecting:
            inside.append(line)
    return inside


def _expand(name: str, body: list[str]) -> tuple[str, ...]:
    """A job name with its matrix substitutions replaced by the declared values."""
    keys = PLACEHOLDER.findall(name)
    if not keys:
        return (name,)
    expanded = [name]
    for key in keys:
        values = re.findall(rf"^\s+{re.escape(key)}:\s*(\S+)\s*$", "\n".join(body), re.MULTILINE)
        assert values, (
            f"the job name {name!r} substitutes matrix.{key} and no line in the job declares "
            f"{key}. The check names this repository produces cannot be derived from it."
        )
        expanded = [
            candidate.replace("${{ matrix." + key + " }}", value)
            for candidate in expanded
            for value in values
        ]
    return tuple(expanded)


def read(path: Path) -> Workflow:
    lines = path.read_text(encoding="utf-8").splitlines()
    triggers = _block(lines, "on")
    assert triggers, f"{path.name} declares no triggers this reader can find under `on:`"
    jobs = _block(lines, "jobs")
    assert jobs, f"{path.name} declares no jobs this reader can find under `jobs:`"

    names: list[str] = []
    identifier: str | None = None
    body: list[str] = []
    for line in [*jobs, "  end-of-file:"]:
        found = JOB.match(line)
        if not found:
            body.append(line)
            continue
        if identifier is not None:
            declared = [JOB_NAME.match(entry) for entry in body]
            named = next((match.group(1) for match in declared if match), identifier)
            names.extend(_expand(named, body))
        identifier, body = found.group(1), []
    assert names, f"{path.name} has a jobs block and no job this reader could name"

    return Workflow(
        path=path,
        on_pull_request=any(line.startswith("  pull_request:") for line in triggers),
        check_names=tuple(names),
    )


workflows = tuple(read(path) for path in sorted(WORKFLOWS.glob("*.yml")))
produced = {name: flow for flow in workflows for name in flow.check_names}
declared = {entry["name"]: entry for entry in record["check_names"]}


def test_the_reader_found_every_workflow_file() -> None:
    """A glob matching nothing turns every comparison below into one of two empty sets."""
    files = sorted(path.name for path in WORKFLOWS.glob("*.yml"))
    assert len(files) == len(workflows) and len(files) > 1, (
        f"{WORKFLOWS} holds {files} and the reader produced {len(workflows)} workflows."
    )


def test_every_declared_check_name_is_produced_by_the_workflow_it_names() -> None:
    wrong = {
        name: entry["workflow"]
        for name, entry in declared.items()
        if entry["workflow"] is not None
        and (name not in produced or produced[name].relative != entry["workflow"])
    }
    assert not wrong, (
        f"{wrong} name a workflow that does not produce them. A record naming a check that no "
        "longer exists is a required set with a name in it that nothing will ever report."
    )


def test_every_check_name_the_workflows_produce_is_declared() -> None:
    missing = sorted(set(produced) - set(declared))
    assert not missing, (
        f"{missing} are produced by the workflows and are absent from {RECORD.name}. A check "
        "the record does not list is a check nobody chooses for or against."
    )


def test_a_check_name_with_no_workflow_declares_where_it_comes_from() -> None:
    """Nothing in this tree can derive a check run created by an upload rather than by a job."""
    undeclared = sorted(
        name
        for name, entry in declared.items()
        if entry["workflow"] is None and not entry.get("source")
    )
    assert not undeclared, (
        f"{undeclared} name no workflow and no source, so the record asserts a check run exists "
        "and offers a reader nothing to check that against."
    )


def test_whether_a_check_runs_on_a_pull_request_matches_its_workflow() -> None:
    wrong = {
        name: entry["on_pull_request"]
        for name, entry in declared.items()
        if name in produced and produced[name].on_pull_request != entry["on_pull_request"]
    }
    assert not wrong, (
        f"{wrong} disagree with the triggers of the workflow that produces them. Only a check "
        "that runs on a pull request can hold a merge, so this is the column the decision is "
        "read out of."
    )


def test_the_decisions_cover_the_target_set_exactly() -> None:
    required = list(record["target"]["required"])
    decided = [entry["context"] for entry in record["decisions"]]
    assert decided == required, (
        f"the target set is {required} and the decisions cover {decided}. A check in the target "
        "with no decision is a check dropped by accident, which is what this record exists "
        "against."
    )


def test_every_decision_is_one_of_the_three_words_and_carries_a_reason() -> None:
    wrong = {
        entry["context"]: entry["decision"]
        for entry in record["decisions"]
        if entry["decision"] not in DECISIONS or not entry.get("reason", "").strip()
    }
    assert not wrong, (
        f"{wrong} carry a decision outside {DECISIONS} or no reason. A deviation with no "
        "reasoning is indistinguishable from a check that was forgotten."
    )


def test_every_check_still_to_be_built_names_an_issue() -> None:
    """The four ported guards predate the first issue on this board and name none.

    Everything else is work, and work with no issue is work nobody can argue
    with afterwards.
    """
    entries = [
        *(entry for entry in record["decisions"] if entry["decision"] == "adapted"),
        *record["no_counterpart_there"],
        *record["not_required_there"],
    ]
    silent = sorted(
        entry.get("context", entry.get("name", "")) for entry in entries if not entry["issues"]
    )
    assert not silent, (
        f"{silent} are adapted or have no counterpart on the target board, and name no issue."
    )


def test_every_check_name_a_decision_claims_is_one_this_repository_produces() -> None:
    entries = [*record["decisions"], *record["no_counterpart_there"], *record["not_required_there"]]
    claimed = {name for entry in entries for name in entry["produces"]}
    unknown = sorted(claimed - set(declared))
    assert not unknown, (
        f"{unknown} are named as produced by a decision and are not in the list of check names. "
        "A decision pointing at a check that does not exist reads as one that is in force."
    )


def test_the_page_agrees_with_the_record_on_the_counts_it_quotes() -> None:
    """A count written into prose once and left behind is the failure this repository names."""
    page = PAGE.read_text(encoding="utf-8")
    counts = {
        "required checks in the target set": len(record["target"]["required"]),
        "ported unchanged": sum(
            1 for entry in record["decisions"] if entry["decision"] == "ported"
        ),
        "adapted": sum(1 for entry in record["decisions"] if entry["decision"] == "adapted"),
        "check names this repository produces": len(record["check_names"]),
        "of those, on a pull request": sum(
            1 for entry in record["check_names"] if entry["on_pull_request"]
        ),
    }
    rows = dict(re.findall(r"^\| ([^|]+?) \| (\d+) \|$", page, re.MULTILINE))
    wrong = {
        subject: (count, rows.get(subject))
        for subject, count in counts.items()
        if rows.get(subject) != str(count)
    }
    assert not wrong, (
        f"{PAGE.name} states {wrong} as (record, page) for the counts it quotes. A count "
        "written once and left behind is what this repository calls restated-not-referenced."
    )
