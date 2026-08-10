"""Pull request hygiene for this board, in two tiers that mean different things.

Three of the rules here transfer from anywhere and refuse a class of defect that
has nothing to do with the language. Two are specific to what has to move
together in this tree, and they are the reason this is a check rather than a
convention: a parameter added to a transform and not added to the profiles, and
a manifest schema whose meaning moved while its version did not, are both silent.
Nothing crashes, the suite goes green, and every run recorded before the change
now describes something else.

## The two tiers, and why the line is where it is

A rule with near zero false positives refuses. A rule that encodes a convention
warns and never refuses, because a check that reds a reasonable pull request
teaches people to route around checks, and a routed-around check protects
nothing at all. The size rule is the clearest case: a sweep results table or a
new transform legitimately exceeds any line cap anybody would set, so it warns.

The tier is a property of the rule and is written beside it. Promoting one is a
change to this file that a reader can see, which is the point: the first time a
warning is inconvenient somebody will propose promoting it, and that argument
should happen in a diff.

## Every rule states its own reason where it fires

A refusal that prints only a pattern gets worked around, because the person
reading it has no way to tell a rule that matters from a rule somebody liked.
Each rule below carries the failure it prevents in its own text, and that text
is what the check prints.

## What this reads, and what it therefore cannot see

The pull request body, the messages of the commits the pull request adds, the
list of paths that changed, and the base and head content of individual files.
It does not run the suite and it does not import the package. A rule that needed
either belongs in `tests/`, where the gate already runs it, rather than here.

`review` takes everything it reads as data, so the rules are exercised against
constructed near misses in `tests/unit/test_pr_hygiene.py` rather than by
pushing deliberately broken branches at the board.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

REFUSES = "refuses"
WARNS = "warns"

#: An issue reference, as this board writes one: a bare number, or a link to an
#: issue on this repository. Both spellings appear in the landed history.
ISSUE_REFERENCE = re.compile(r"#\d+|github\.com/iderex/gutachten/issues/\d+")

#: The characters a commit message may contain. Printable ASCII plus the two
#: whitespace characters a message is laid out with. A message is read in a
#: terminal, in an email, in a web page and by `git log` on a machine whose
#: locale nobody chose, and a character that renders in one of those and not the
#: others turns the record of why a change was made into a guess.
ALLOWED = frozenset(chr(code) for code in range(0x20, 0x7F)) | {"\n", "\t"}

#: Where the transforms live, and where the parameter sets that drive them live.
TRANSFORMS = "src/gutachten/transforms/"
PROFILES = "profiles/"

#: The recorded manifest, and the module whose constant says what schema it is
#: an instance of.
GOLDEN_MANIFEST = "tests/golden/reference_manifest.json"
MANIFEST_MODULE = "src/gutachten/manifest.py"
SCHEMA_VERSION = re.compile(r"^SCHEMA_VERSION\s*=\s*(\d+)", re.MULTILINE)

#: The line count above which a change is called large. Inherited rather than
#: derived here, and deliberately not measured against the diffs already on this
#: board, because a threshold chosen while looking at the changes it will judge
#: is a threshold chosen to let them through. It is revisited when there is a
#: sweep results table to measure against, and it warns rather than refuses
#: precisely because that day is coming.
LARGE_CHANGE_LINES = 400


@dataclass(frozen=True)
class Commit:
    """One commit the pull request adds, as the rules read it."""

    sha: str
    message: str
    merge: bool = False

    @property
    def subject(self) -> str:
        return self.message.splitlines()[0] if self.message.splitlines() else ""


@dataclass(frozen=True)
class Finding:
    """One rule firing, with the reason it exists in its own words."""

    rule: str
    tier: str
    detail: str
    reason: str

    def rendered(self) -> str:
        mark = "REFUSED" if self.tier == REFUSES else "WARNING"
        return f"{mark} {self.rule}: {self.detail}\n    {self.reason}"


@dataclass(frozen=True)
class Change:
    """Everything the rules are allowed to read, gathered once.

    ``at_base`` and ``at_head`` return a file's content or ``None`` where the
    file does not exist on that side. They are functions rather than a mapping
    so that a rule reads only the files it is about, and so a test can supply
    two dictionaries instead of a repository.
    """

    body: str
    commits: tuple[Commit, ...]
    paths: tuple[str, ...]
    lines_changed: int
    at_base: Callable[[str], str | None] = lambda path: None
    at_head: Callable[[str], str | None] = lambda path: None
    #: Set where a rule could not read what it needed. A rule that cannot see
    #: its subject says so rather than passing, because a check that goes green
    #: when it read nothing is worse than one that is absent.
    unreadable: list[str] = field(default_factory=list)

    def touching(self, prefix: str) -> tuple[str, ...]:
        return tuple(path for path in self.paths if path.startswith(prefix))


def the_body_names_an_issue(change: Change) -> Iterable[Finding]:
    if ISSUE_REFERENCE.search(change.body):
        return
    yield Finding(
        rule="body-names-an-issue",
        tier=REFUSES,
        detail="the pull request body names no issue",
        reason=(
            "Every change starts as an issue, and the body is where a reader finds what "
            "was wrong, what the evidence is and what done means. A change that names "
            "none is a change whose reason lives only in whoever wrote it."
        ),
    )


def every_commit_names_an_issue(change: Change) -> Iterable[Finding]:
    """The rule this board words differently from where it was inherited.

    It was inherited as a rule about the commit SUBJECT. Measured against this
    board, that rule refuses everything that has ever landed here:

        git log origin/main --no-merges --format=%s | grep -c '#[0-9]'
        0

    while 24 of the 29 messages carry the reference in the body. So the rule is
    applied to the whole message. What it is actually for, which is that a
    commit read on its own leads back to the argument for it, is unmoved, and a
    rule that refused every pull request on the board would be routed around
    within a day.

    Merge commits are exempt. Their message is generated by the forge and
    records an integration rather than a change.
    """
    for commit in change.commits:
        if commit.merge or ISSUE_REFERENCE.search(commit.message):
            continue
        yield Finding(
            rule="every-commit-names-an-issue",
            tier=REFUSES,
            detail=f"{commit.sha[:9]} names no issue: {commit.subject!r}",
            reason=(
                "A commit is read on its own, in a bisect or a blame, long after the pull "
                "request it arrived in has scrolled away. One that names no issue leads a "
                "reader nowhere."
            ),
        )


def commit_messages_stay_inside_the_character_set(change: Change) -> Iterable[Finding]:
    for commit in change.commits:
        outside = sorted({character for character in commit.message if character not in ALLOWED})
        if not outside:
            continue
        yield Finding(
            rule="message-inside-the-character-set",
            tier=REFUSES,
            detail=(
                f"{commit.sha[:9]} carries {[hex(ord(character)) for character in outside]}, "
                "which is outside printable ASCII"
            ),
            reason=(
                "A commit message is read in a terminal, in an email and by a tool on a "
                "machine whose locale nobody chose. A character that renders in one of "
                "those and not the others turns the record of why a change was made into a "
                "guess, and the bidirectional ones can make it read as its own opposite."
            ),
        )


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        named = decorator.func
        name = named.id if isinstance(named, ast.Name) else getattr(named, "attr", "")
        if name != "dataclass":
            continue
        for keyword in decorator.keywords:
            frozen = isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            if keyword.arg == "frozen" and frozen:
                return True
    return False


def parameter_fields(source: str) -> dict[str, frozenset[str]]:
    """The declared fields of every frozen dataclass in ``source``.

    Read with `ast` rather than off the diff. A field added inside a hunk that
    also moved a dozen lines around it is invisible to a line-based reading, and
    that is exactly the change this rule is for.
    """
    found: dict[str, frozenset[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and _is_frozen_dataclass(node):
            found[node.name] = frozenset(
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
            )
    return found


def a_new_parameter_reaches_a_profile(change: Change) -> Iterable[Finding]:
    """The silent default this whole plan is built against.

    A parameter added to a step and not added to the profiles is a parameter
    with no value in any recorded chain. The sweep cannot move it, the manifest
    does not name it, and the step reads whatever it reads. The suite refuses
    this too, at profile load, and the two are not redundant: this one says so
    on the pull request, with the field named, before anybody has read a
    traceback out of a test run.

    It reads Python and only Python. The prefix holds a README and will hold
    more prose than that, and handing one of those to a parser produced a
    refusal under `could-not-read`, which is the escalation this file keeps for
    a rule that could not see a subject it was supposed to judge. A markdown
    file is not that subject. Two refusals that mean opposite things are
    indistinguishable in the message a reader gets, and the cheap way out of the
    wrong one is to stop editing the README.
    """
    if change.touching(PROFILES):
        return
    for path in (item for item in change.touching(TRANSFORMS) if item.endswith(".py")):
        before, after = change.at_base(path), change.at_head(path)
        if after is None:
            continue
        if before is None:
            before = ""
        try:
            was, now = parameter_fields(before), parameter_fields(after)
        except SyntaxError as broken:
            change.unreadable.append(f"{path}: {broken}")
            continue
        for record, fields in now.items():
            added = sorted(fields - was.get(record, frozenset()))
            if not added:
                continue
            yield Finding(
                rule="a-new-parameter-reaches-a-profile",
                tier=REFUSES,
                detail=f"{path} adds {added} to {record} and nothing under {PROFILES} changed",
                reason=(
                    "A parameter no profile sets is a parameter no recorded run names and no "
                    "sweep can move, which is the silent default this project exists "
                    "against. Add the value to every profile that runs the step, with where "
                    "it came from."
                ),
            )


def _key_paths(value: object, prefix: str = "") -> set[str]:
    """Every key path in a parsed JSON document, without its values.

    The shape rather than the content. A recorded number that moved is a run
    that changed; a recorded key that moved is a schema that changed, and only
    the second is what a version exists to mark.
    """
    found: set[str] = set()
    if isinstance(value, dict):
        for key, inner in value.items():
            here = f"{prefix}.{key}" if prefix else str(key)
            found.add(here)
            found |= _key_paths(inner, here)
    elif isinstance(value, list):
        for item in value:
            found |= _key_paths(item, f"{prefix}[]")
    return found


def a_schema_move_carries_a_version_move(change: Change) -> Iterable[Finding]:
    """A manifest that means something different under the same version.

    Every run recorded before such a change goes on parsing and now describes
    something else, and nothing anywhere says so. The recorded manifest is read
    for its key set rather than its values, because a value that moved is a run
    that differed and a key that moved is the schema.
    """
    if GOLDEN_MANIFEST not in change.paths:
        return
    before, after = change.at_base(GOLDEN_MANIFEST), change.at_head(GOLDEN_MANIFEST)
    if before is None or after is None:
        change.unreadable.append(f"{GOLDEN_MANIFEST}: missing on one side of the change")
        return
    try:
        moved = _key_paths(json.loads(before)) ^ _key_paths(json.loads(after))
    except json.JSONDecodeError as broken:
        change.unreadable.append(f"{GOLDEN_MANIFEST}: {broken}")
        return
    if not moved:
        return

    versions = []
    for content in (change.at_base(MANIFEST_MODULE), change.at_head(MANIFEST_MODULE)):
        found = SCHEMA_VERSION.search(content or "")
        versions.append(found.group(1) if found else None)
    if versions[0] != versions[1] and None not in versions:
        return
    yield Finding(
        rule="a-schema-move-carries-a-version-move",
        tier=REFUSES,
        detail=(
            f"{GOLDEN_MANIFEST} moves {sorted(moved)} and SCHEMA_VERSION stays at {versions[1]}"
        ),
        reason=(
            "A manifest that means something different under the same schema version "
            "breaks every run recorded before the change, silently: they parse, and they "
            "now describe something else. Move SCHEMA_VERSION in the same change."
        ),
    )


def a_changed_transform_carries_a_changed_test(change: Change) -> Iterable[Finding]:
    """A warning, and it stays one.

    A transform can legitimately change without its tests changing, when what
    moved was a docstring or a message. Refusing that would be a rule people
    learn to defeat with a whitespace edit to a test file, which is worse than
    the warning it replaced.
    """
    changed = [path for path in change.touching(TRANSFORMS) if path.endswith(".py")]
    if not changed:
        return
    if any(path.startswith("tests/") and path.endswith(".py") for path in change.paths):
        return
    yield Finding(
        rule="a-changed-transform-carries-a-changed-test",
        tier=WARNS,
        detail=f"{changed} changed and no test file did",
        reason=(
            "Most defects in a numerical step are a number that quietly moved rather than "
            "a crash. A change to a step with no change to what watches it is the shape "
            "that gets through. This warns rather than refuses, because a message or a "
            "docstring is a legitimate change that owes no test."
        ),
    )


def a_large_change_is_called_large(change: Change) -> Iterable[Finding]:
    if change.lines_changed <= LARGE_CHANGE_LINES:
        return
    yield Finding(
        rule="a-large-change-is-called-large",
        tier=WARNS,
        detail=f"{change.lines_changed} lines changed, above {LARGE_CHANGE_LINES}",
        reason=(
            "A change too large to hold in one reading gets approved rather than read. "
            "This warns and never refuses: a sweep results table, a new transform with its "
            "proofs, or a recorded run legitimately exceeds any cap, and a check that reds "
            "those teaches people to route around checks."
        ),
    )


RULES = (
    the_body_names_an_issue,
    every_commit_names_an_issue,
    commit_messages_stay_inside_the_character_set,
    a_new_parameter_reaches_a_profile,
    a_schema_move_carries_a_version_move,
    a_changed_transform_carries_a_changed_test,
    a_large_change_is_called_large,
)


def review(change: Change) -> list[Finding]:
    """Every finding, in rule order, refusals and warnings together."""
    return [finding for rule in RULES for finding in rule(change)]


def report(findings: Sequence[Finding], unreadable: Sequence[str]) -> tuple[str, int]:
    """What the check prints, and what it exits with."""
    lines = []
    refusals = [finding for finding in findings if finding.tier == REFUSES]
    warnings = [finding for finding in findings if finding.tier == WARNS]

    for finding in refusals + warnings:
        lines.append(finding.rendered())
    for problem in unreadable:
        lines.append(
            f"REFUSED could-not-read: {problem}\n    A rule that cannot see what it judges "
            "says so rather than passing."
        )

    if not lines:
        lines.append(f"All {len(RULES)} rules ran and none of them fired.")
    elif not refusals and not unreadable:
        lines.append(
            f"{len(warnings)} warning(s) and nothing refused, so this check passes. A "
            "warning is a rule that encodes a convention; the tier of each rule is set in "
            ".github/pr_hygiene.py and moving one is a change a reader can see."
        )
    return "\n\n".join(lines), (1 if refusals or unreadable else 0)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True, encoding="utf-8"
    ).stdout


def _at(revision: str) -> Callable[[str], str | None]:
    def read(path: str) -> str | None:
        try:
            return _git("show", f"{revision}:{path}")
        except subprocess.CalledProcessError:
            return None

    return read


def change_from_git(base: str, head: str, body: str) -> Change:
    """What the workflow hands the rules, read out of the repository."""
    commits = []
    for line in _git("log", "--format=%H %P", f"{base}..{head}").splitlines():
        sha, _, parents = line.partition(" ")
        commits.append(
            Commit(
                sha=sha,
                message=_git("log", "-1", "--format=%B", sha),
                merge=len(parents.split()) > 1,
            )
        )

    names = _git("diff", "--name-only", f"{base}...{head}").splitlines()
    paths = tuple(path for path in names if path)

    lines_changed = 0
    for line in _git("diff", "--numstat", f"{base}...{head}").splitlines():
        counts = line.split("\t")[:2]
        # A binary file reports "-" for both, which is not a line count and is
        # not silently read as zero lines of a change that happened.
        lines_changed += sum(int(count) for count in counts if count.isdigit())
    return Change(
        body=body,
        commits=tuple(commits),
        paths=paths,
        lines_changed=lines_changed,
        at_base=_at(base),
        at_head=_at(head),
    )


def main() -> int:
    base = os.environ.get("BASE_SHA", "")
    head = os.environ.get("HEAD_SHA", "")
    if not base or not head:
        print("BASE_SHA and HEAD_SHA are required; this check reads a range, not a tree.")
        return 1
    change = change_from_git(base, head, os.environ.get("PR_BODY", ""))
    text, code = report(review(change), change.unreadable)
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
