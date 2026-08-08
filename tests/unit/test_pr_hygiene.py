"""The pull request hygiene rules, each against a near miss and a neighbour.

The rules are exercised as functions over constructed data rather than by
pushing deliberately broken branches at the board. That is the only way to prove
the ones that refuse: a fixture that trips exactly one rule, and a neighbour one
edit away from it that trips nothing, cannot be produced by a real pull request
without leaving it behind.

`.github/pr_hygiene.py` is not importable as part of the package and is not
meant to be. It is loaded here by path, which is also the statement that it
ships with the repository and not with the wheel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

HYGIENE = Path(__file__).resolve().parents[2] / ".github" / "pr_hygiene.py"


def _loaded() -> ModuleType:
    specification = importlib.util.spec_from_file_location("pr_hygiene", HYGIENE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules.setdefault("pr_hygiene", module)
    specification.loader.exec_module(module)
    return module


hygiene = _loaded()


A_PARAMETER_RECORD = """
from dataclasses import dataclass

@dataclass(frozen=True)
class EdgeParameters:
    width: float
    criterion: str
"""

THE_SAME_RECORD_WITH_ONE_MORE_FIELD = """
from dataclasses import dataclass

@dataclass(frozen=True)
class EdgeParameters:
    width: float
    criterion: str
    dilation: float
"""

A_MANIFEST = '{"schema_version": 3, "steps": [{"identifier": "level", "version": "1"}]}'
THE_SAME_MANIFEST_WITH_ONE_MORE_KEY = (
    '{"schema_version": 3, "steps": [{"identifier": "level", "version": "1", "outcomes": {}}]}'
)
THE_SAME_MANIFEST_WITH_A_MOVED_VALUE = (
    '{"schema_version": 3, "steps": [{"identifier": "bandpass", "version": "1"}]}'
)

A_MODULE_AT_VERSION = "SCHEMA_VERSION = 3\n"
A_MODULE_AT_THE_NEXT_VERSION = "SCHEMA_VERSION = 4\n"


def a_change(
    *,
    body: str = "Closes #114.",
    commits: tuple[object, ...] = (),
    paths: tuple[str, ...] = (),
    lines_changed: int = 1,
    base: dict[str, str] | None = None,
    head: dict[str, str] | None = None,
) -> object:
    """A change that trips nothing, to be broken one field at a time.

    Every fixture below starts from this, so each one is the single edit that
    made a rule fire rather than a fixture assembled to fail.
    """
    at_base = dict(base or {})
    at_head = dict(head or {})
    return hygiene.Change(
        body=body,
        commits=tuple(commits) or (hygiene.Commit(sha="a" * 40, message="A change.\n\nSee #114."),),
        paths=paths,
        lines_changed=lines_changed,
        at_base=at_base.get,
        at_head=at_head.get,
    )


def rules_that_fired(change: object) -> list[str]:
    return [finding.rule for finding in hygiene.review(change)]


def test_a_change_that_trips_nothing_trips_nothing() -> None:
    # The neighbour every fixture below is one edit away from. Without it, a
    # rule that fired on everything would look like a rule that works.
    findings = hygiene.review(a_change())
    assert findings == []
    text, code = hygiene.report(findings, [])
    assert code == 0
    assert "none of them fired" in text


def test_a_body_that_names_no_issue_is_refused() -> None:
    assert rules_that_fired(a_change(body="Tidies things up.")) == ["body-names-an-issue"]


def test_a_body_that_links_the_issue_instead_of_numbering_it_passes() -> None:
    # Both spellings appear in the landed history, so a rule that took only one
    # would refuse work that is already the convention.
    assert (
        rules_that_fired(a_change(body="See https://github.com/iderex/gutachten/issues/114")) == []
    )


def test_a_commit_that_names_no_issue_is_refused() -> None:
    silent = hygiene.Commit(sha="b" * 40, message="Fix the thing.\n")
    assert rules_that_fired(a_change(commits=(silent,))) == ["every-commit-names-an-issue"]


def test_a_merge_commit_that_names_no_issue_is_not_refused() -> None:
    # Its message is generated and it records an integration rather than a
    # change. This is the near miss for the rule above: one field apart.
    merge = hygiene.Commit(sha="c" * 40, message="Merge main into a branch\n", merge=True)
    assert rules_that_fired(a_change(commits=(merge,))) == []


def test_a_message_outside_the_character_set_is_refused() -> None:
    # A right-to-left override, which is the character that can make a message
    # read as its own opposite. Built with chr() rather than written out: this
    # repository's unicode guard refuses that byte in a tracked file, which is
    # the same rule one layer down, and a fixture may not smuggle past it.
    override = chr(0x202E)
    smuggled = hygiene.Commit(sha="d" * 40, message=f"Refuse {override}the thing. See #114.\n")
    assert rules_that_fired(a_change(commits=(smuggled,))) == ["message-inside-the-character-set"]


def test_a_message_that_is_printable_ascii_with_tabs_and_newlines_passes() -> None:
    laid_out = hygiene.Commit(sha="e" * 40, message="A change. See #114.\n\n\tindented\n")
    assert rules_that_fired(a_change(commits=(laid_out,))) == []


def test_a_parameter_added_without_a_profile_update_is_refused() -> None:
    fired = hygiene.review(
        a_change(
            paths=("src/gutachten/transforms/edge.py", "tests/unit/transforms/test_edge.py"),
            base={"src/gutachten/transforms/edge.py": A_PARAMETER_RECORD},
            head={"src/gutachten/transforms/edge.py": THE_SAME_RECORD_WITH_ONE_MORE_FIELD},
        )
    )
    assert [finding.rule for finding in fired] == ["a-new-parameter-reaches-a-profile"]
    assert "dilation" in fired[0].detail
    assert "EdgeParameters" in fired[0].detail


def test_the_same_parameter_added_with_a_profile_update_passes() -> None:
    assert (
        rules_that_fired(
            a_change(
                paths=(
                    "src/gutachten/transforms/edge.py",
                    "tests/unit/transforms/test_edge.py",
                    "profiles/every-step.json",
                ),
                base={"src/gutachten/transforms/edge.py": A_PARAMETER_RECORD},
                head={"src/gutachten/transforms/edge.py": THE_SAME_RECORD_WITH_ONE_MORE_FIELD},
            )
        )
        == []
    )


def test_a_transform_changed_without_adding_a_parameter_is_not_refused() -> None:
    # The near miss for the rule above. A step whose message or docstring moved
    # owes no profile edit, and a rule that refused it would be a rule people
    # satisfy with a whitespace change to a JSON file.
    assert (
        rules_that_fired(
            a_change(
                paths=("src/gutachten/transforms/edge.py", "tests/unit/transforms/test_edge.py"),
                base={"src/gutachten/transforms/edge.py": A_PARAMETER_RECORD},
                head={"src/gutachten/transforms/edge.py": A_PARAMETER_RECORD + "\n# a comment\n"},
            )
        )
        == []
    )


def test_a_schema_key_that_moved_without_the_version_is_refused() -> None:
    fired = hygiene.review(
        a_change(
            paths=("tests/golden/reference_manifest.json", "src/gutachten/manifest.py"),
            base={
                "tests/golden/reference_manifest.json": A_MANIFEST,
                "src/gutachten/manifest.py": A_MODULE_AT_VERSION,
            },
            head={
                "tests/golden/reference_manifest.json": THE_SAME_MANIFEST_WITH_ONE_MORE_KEY,
                "src/gutachten/manifest.py": A_MODULE_AT_VERSION,
            },
        )
    )
    assert [finding.rule for finding in fired] == ["a-schema-move-carries-a-version-move"]
    assert "outcomes" in fired[0].detail


def test_the_same_key_moving_with_the_version_passes() -> None:
    assert (
        rules_that_fired(
            a_change(
                paths=("tests/golden/reference_manifest.json", "src/gutachten/manifest.py"),
                base={
                    "tests/golden/reference_manifest.json": A_MANIFEST,
                    "src/gutachten/manifest.py": A_MODULE_AT_VERSION,
                },
                head={
                    "tests/golden/reference_manifest.json": THE_SAME_MANIFEST_WITH_ONE_MORE_KEY,
                    "src/gutachten/manifest.py": A_MODULE_AT_THE_NEXT_VERSION,
                },
            )
        )
        == []
    )


def test_a_recorded_value_that_moved_is_not_a_schema_move() -> None:
    # The near miss that decides whether this rule is usable. A recorded run
    # that differs is a run that differed; only a key that moved is the schema.
    # A rule that refused a changed value would fire on every legitimate update
    # to the recording and would be switched off within a week.
    assert (
        rules_that_fired(
            a_change(
                paths=("tests/golden/reference_manifest.json",),
                base={
                    "tests/golden/reference_manifest.json": A_MANIFEST,
                    "src/gutachten/manifest.py": A_MODULE_AT_VERSION,
                },
                head={
                    "tests/golden/reference_manifest.json": THE_SAME_MANIFEST_WITH_A_MOVED_VALUE,
                    "src/gutachten/manifest.py": A_MODULE_AT_VERSION,
                },
            )
        )
        == []
    )


def test_a_transform_changed_with_no_test_changed_warns_and_passes() -> None:
    findings = hygiene.review(
        a_change(
            paths=("src/gutachten/transforms/edge.py", "profiles/every-step.json"),
            base={"src/gutachten/transforms/edge.py": A_PARAMETER_RECORD},
            head={"src/gutachten/transforms/edge.py": A_PARAMETER_RECORD + "\n# a comment\n"},
        )
    )
    assert [finding.rule for finding in findings] == ["a-changed-transform-carries-a-changed-test"]
    assert findings[0].tier == hygiene.WARNS
    text, code = hygiene.report(findings, [])
    assert code == 0
    assert "nothing refused" in text


def test_a_large_change_warns_and_passes() -> None:
    findings = hygiene.review(a_change(lines_changed=hygiene.LARGE_CHANGE_LINES * 3))
    assert [finding.rule for finding in findings] == ["a-large-change-is-called-large"]
    assert findings[0].tier == hygiene.WARNS
    text, code = hygiene.report(findings, [])
    assert code == 0
    assert "This check passes" in text or "nothing refused" in text


def test_a_change_exactly_at_the_threshold_does_not_warn() -> None:
    assert rules_that_fired(a_change(lines_changed=hygiene.LARGE_CHANGE_LINES)) == []


def test_every_finding_carries_the_reason_it_exists() -> None:
    # The clause asking that each rule states its own reason in its output
    # rather than only in the issue. A refusal that prints a pattern and no
    # reason gets worked around by whoever hits it.
    seen = set()
    for finding in hygiene.review(
        a_change(
            body="nothing here",
            commits=(hygiene.Commit(sha="f" * 40, message="Quiet.\n"),),
            lines_changed=hygiene.LARGE_CHANGE_LINES * 2,
        )
    ):
        seen.add(finding.rule)
        assert len(finding.reason.split()) > 15
        assert finding.reason in finding.rendered()
        assert finding.detail in finding.rendered()
    assert seen == {
        "body-names-an-issue",
        "every-commit-names-an-issue",
        "a-large-change-is-called-large",
    }


def test_the_two_tiers_are_separated_in_what_the_check_exits_with() -> None:
    # The clause asking for the tiers to be separated. A warning that failed the
    # check and a refusal that did not would both make the tier a label.
    refusing = hygiene.review(a_change(body="nothing here"))
    warning = hygiene.review(a_change(lines_changed=hygiene.LARGE_CHANGE_LINES * 2))
    assert hygiene.report(refusing, [])[1] == 1
    assert hygiene.report(warning, [])[1] == 0
    assert hygiene.report(refusing + warning, [])[1] == 1


def test_a_rule_that_cannot_read_its_subject_refuses_rather_than_passing() -> None:
    # A check that goes green because it read nothing is worse than one that is
    # absent, because it is quoted afterwards as having found nothing.
    change = a_change(
        paths=("src/gutachten/transforms/edge.py",),
        base={"src/gutachten/transforms/edge.py": A_PARAMETER_RECORD},
        head={"src/gutachten/transforms/edge.py": "def broken(:\n"},
    )
    hygiene.review(change)
    text, code = hygiene.report([], change.unreadable)
    assert code == 1
    assert "could-not-read" in text


def test_the_check_refuses_a_run_that_was_given_no_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.delenv("HEAD_SHA", raising=False)
    assert hygiene.main() == 1
