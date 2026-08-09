"""The declared ranges cover every parameter a sweep can reach, and nothing else.

A sensitivity result is worth exactly as much as the range it was taken over, so
`docs/ranges.json` is an artefact rather than a note, and this is what refuses it
when it drifts. The failure it exists against is silent in both directions. A
parameter added to a step and left out of the declaration is a parameter the
sweep cannot move, and a sweep that never moved it reports a stability the
pipeline does not have. A declaration naming a parameter nothing answers to is a
range somebody wrote once and a reader counts as covered.

## Why the rules are functions over data

`problems` reads a declaration, the reachable parameters and the counts a
document states, all as plain values. So every rule is exercised against a
constructed near miss below rather than by breaking the real file and watching
the suite, and a rule that could never fire is visible as a near miss nobody
could write. The same shape as `.github/pr_hygiene.py`, for the same reason.

## What is reachable is derived, not listed

The registry is the authority for the preprocessing parameters, and the
comparison parameter records are found by inspecting the two modules that hold
them, so a record added beside them is covered without an edit here. Reading only
the registry would pass a declaration covering the preprocessing and none of the
scoring, which is the failure `test_the_comparison_parameters_are_reached` names.

What none of this checks is whether a bound is the right one. That is a
judgement, it is what the review is for, and the `source` field exists so a
reader can see which bounds nobody outside this repository has ever stated.
"""

from __future__ import annotations

import dataclasses
import json
import re
import types
import typing
from collections import Counter
from enum import Enum
from pathlib import Path

import gutachten.transforms  # noqa: F401  (importing registers the steps)
from gutachten.compare import cmc, register
from gutachten.transforms.registry import REGISTRY

#: The declaration and the document that argues for it.
RANGES = Path(__file__).resolve().parents[2] / "docs" / "ranges.json"
DOCUMENT = RANGES.with_suffix(".md")

#: The closed vocabulary a bound's source comes from. Four words: a value stated
#: in a published method, a range stated in a standard, the physical limit of the
#: measurement, or a judgement. A fifth word would let a bound escape the count
#: that says how much of the file is this repository's own opinion.
SOURCES = ("published", "standardised", "physical", "judgement")

#: The modules whose parameter records the comparison stage reads. They are not
#: transforms and register nowhere, so the registry does not reach them.
COMPARISON = (register, cmc)

#: A row of the count table in the document, as it is written there.
COUNT_ROW = re.compile(r"^\|\s*`(\w+)`\s*\|\s*(\d+)\s*\|", re.MULTILINE)


@dataclasses.dataclass(frozen=True)
class Finding:
    """One rule firing, named by the rule so a near miss can assert on it."""

    rule: str
    detail: str


def optional(annotation: object) -> tuple[bool, object]:
    """Whether ``annotation`` admits None, and what is left when it does not."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        rest = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(rest) != len(typing.get_args(annotation)):
            return True, rest[0]
    return False, annotation


def bounds(entry: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    """Every countable bound of one parameter, under the name it is reported by.

    An interval contributes two, a set contributes one per admissible value, and
    a nullable parameter contributes one more. That is the unit the counts in the
    document are taken in.
    """
    found: list[tuple[str, dict[str, object]]] = []
    if entry.get("kind") == "interval":
        for name in ("lower", "upper"):
            if isinstance(entry.get(name), dict):
                found.append((name, entry[name]))  # type: ignore[arg-type]
    for value in entry.get("values", ()):  # type: ignore[union-attr]
        found.append((f"values[{value.get('value')!r}]", value))
    if "null" in entry:
        found.append(("null", entry["null"]))  # type: ignore[arg-type]
    return found


def problems(
    declaration: dict[str, dict[str, object]],
    reachable: dict[str, object],
    stated: dict[str, int],
) -> list[Finding]:
    """Every way the declaration, the parameters and the document disagree."""
    found: list[Finding] = []

    for key in sorted(set(reachable) - set(declaration)):
        found.append(
            Finding(
                "undeclared-parameter",
                f"{key} can be reached by a sweep and no range is declared for it. A "
                "parameter with no declared range cannot be swept, and a sweep that "
                "never moved it reports a stability the pipeline does not have.",
            )
        )
    for key in sorted(set(declaration) - set(reachable)):
        found.append(
            Finding(
                "dangling-declaration",
                f"{key} is declared and nothing takes it. A range for a parameter that "
                "does not exist is a line a reader counts as covered.",
            )
        )

    for key in sorted(set(declaration) & set(reachable)):
        found.extend(_one(key, declaration[key], reachable[key]))

    counted = Counter(
        bound["source"] for entry in declaration.values() for _, bound in bounds(entry)
    )
    for word in SOURCES:
        if word not in stated:
            found.append(
                Finding(
                    "class-missing-from-the-table",
                    f"the document states no count for {word!r}, and a class left out of "
                    "the table is a class a reader reads as empty.",
                )
            )
    for word, number in sorted(stated.items()):
        if counted.get(word, 0) != number:
            found.append(
                Finding(
                    "counts-disagree",
                    f"the document states {number} {word!r} bounds and the declaration "
                    f"holds {counted.get(word, 0)}.",
                )
            )
    return found


def _one(key: str, entry: dict[str, object], annotation: object) -> list[Finding]:
    """The rules about one parameter's declaration."""
    found: list[Finding] = []
    for label, bound in bounds(entry):
        if bound.get("source") not in SOURCES:
            found.append(
                Finding(
                    "source-outside-the-vocabulary",
                    f"{key}.{label} names {bound.get('source')!r}, outside {list(SOURCES)}. "
                    "A fifth word is a bound escaping the count that says how much of "
                    "this file is this repository's own judgement.",
                )
            )
        if not str(bound.get("where", "")).strip():
            found.append(
                Finding(
                    "bound-without-a-sentence",
                    f"{key}.{label} says nothing about where it came from, which is the "
                    "half of a bound a reader can argue with.",
                )
            )

    nullable, inner = optional(annotation)
    if ("null" in entry) != nullable:
        found.append(
            Finding(
                "null-declared-wrongly",
                f"{key} admits None: {nullable}, and its declaration carries a null: "
                f"{'null' in entry}. A nullable parameter whose null is undeclared has a "
                "state the design cannot reach, and a null declared for a parameter that "
                "refuses one is a cell that cannot run.",
            )
        )

    enumerated = isinstance(inner, type) and issubclass(inner, Enum)
    categorical = inner is bool or inner is str or enumerated
    wanted = "set" if categorical else "interval"
    if entry.get("kind") != wanted:
        found.append(
            Finding(
                "shape-disagrees-with-the-parameter",
                f"{key} takes {inner} and is declared as {entry.get('kind')!r} rather than "
                f"{wanted!r}. A word or a flag has no bound to be, and writing one as an "
                "interval reads as a declared range and sweeps nothing.",
            )
        )
        return found

    if wanted == "interval":
        lower, upper = entry["lower"]["value"], entry["upper"]["value"]  # type: ignore[index]
        if not lower < upper:
            found.append(
                Finding(
                    "interval-runs-downwards",
                    f"{key} declares a lower bound of {lower} and an upper of {upper}.",
                )
            )
        if inner is int and not (isinstance(lower, int) and isinstance(upper, int)):
            found.append(
                Finding(
                    "fractional-bound-on-a-whole-number",
                    f"{key} takes a whole number and its bounds are {lower!r} and "
                    f"{upper!r}. A design stepping between two fractions produces cells "
                    "the parameter record refuses.",
                )
            )
        return found

    values = [value["value"] for value in entry["values"]]  # type: ignore[index]
    if len(values) != len({repr(value) for value in values}):
        found.append(Finding("repeated-value", f"{key} declares a value twice: {values}"))
    if inner is bool and sorted(values, key=repr) != [False, True]:
        found.append(
            Finding(
                "flag-with-one-value",
                f"{key} is a flag and declares {values}. A flag whose set holds one value "
                "is a setting held fixed while the report says it was moved.",
            )
        )
    if isinstance(inner, type) and issubclass(inner, Enum):
        admissible = {member.value for member in inner}
        outside = sorted(str(value) for value in values if value not in admissible)
        if outside:
            found.append(
                Finding(
                    "value-outside-the-enumeration",
                    f"{key} declares {outside}, outside {sorted(admissible)}.",
                )
            )
    return found


def declaration() -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = json.loads(RANGES.read_text(encoding="utf-8"))[
        "parameters"
    ]
    return loaded


def reachable() -> dict[str, object]:
    """Every parameter a sweep can reach, keyed as the declaration keys it.

    Transform parameters are keyed by the identifier the manifest names the step
    by. Comparison parameters are keyed by their module, because the two rules in
    `cmc` read different subsets of one set of names and a key per record would
    declare `down_threshold` twice.
    """
    found: dict[str, object] = {}
    for transform in REGISTRY:
        hints = typing.get_type_hints(transform.parameters_type)
        for field in dataclasses.fields(transform.parameters_type):
            found[f"{transform.identifier}.{field.name}"] = hints[field.name]
    for module in COMPARISON:
        for name in sorted(dir(module)):
            record = getattr(module, name)
            if not (isinstance(record, type) and dataclasses.is_dataclass(record)):
                continue
            if not name.endswith("Parameters") or record.__module__ != module.__name__:
                continue
            hints = typing.get_type_hints(record)
            for field in dataclasses.fields(record):
                key = f"compare.{module.__name__.rsplit('.', 1)[1]}.{field.name}"
                found.setdefault(key, hints[field.name])
    return found


def stated_counts() -> dict[str, int]:
    return {word: int(number) for word, number in COUNT_ROW.findall(DOCUMENT.read_text("utf-8"))}


def test_the_declared_ranges_agree_with_the_parameters_and_the_document() -> None:
    found = problems(declaration(), reachable(), stated_counts())
    assert not found, "\n".join(f"{finding.rule}: {finding.detail}" for finding in found)


def test_the_comparison_parameters_are_reached() -> None:
    """The registry is not the whole reachable set, and reading only it would pass.

    The search settings and the identification parameters are swept for the same
    reason the preprocessing ones are, and they register nowhere. The high
    variant's own parameter is named because it lives on the second record in
    that module, so a check that found one record and stopped would miss it.
    """
    keys = reachable()
    assert "compare.register.rotation_step_deg" in keys
    assert "compare.cmc.correlation_threshold" in keys
    assert "compare.cmc.high_tolerance" in keys


def test_the_document_states_a_count_for_every_class() -> None:
    assert set(stated_counts()) == set(SOURCES)


# The near misses. Each one is a declaration that is wrong in exactly one way,
# and each asserts the rule that names that way, so a rule silently deleted
# reddens here rather than passing as a file nobody violated.

_SENTENCE = "a sentence, because a bound without one cannot be argued with"


def _interval(**over: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "kind": "interval",
        "lower": {"value": 1.0, "source": "judgement", "where": _SENTENCE},
        "upper": {"value": 2.0, "source": "judgement", "where": _SENTENCE},
    }
    entry.update(over)
    return entry


def _rules(declared: dict[str, dict[str, object]], takes: dict[str, object]) -> set[str]:
    counted = Counter(bound["source"] for entry in declared.values() for _, bound in bounds(entry))
    stated = {word: counted.get(word, 0) for word in SOURCES}
    return {finding.rule for finding in problems(declared, takes, stated)}


def test_a_parameter_with_no_declared_range_is_refused() -> None:
    assert "undeclared-parameter" in _rules({}, {"step.width": float})


def test_a_declaration_nothing_answers_to_is_refused() -> None:
    assert "dangling-declaration" in _rules({"step.gone": _interval()}, {})


def test_a_source_outside_the_four_words_is_refused() -> None:
    entry = _interval(lower={"value": 1.0, "source": "obvious", "where": _SENTENCE})
    assert "source-outside-the-vocabulary" in _rules({"s.w": entry}, {"s.w": float})


def test_a_bound_with_no_sentence_is_refused() -> None:
    entry = _interval(upper={"value": 2.0, "source": "judgement", "where": "   "})
    assert "bound-without-a-sentence" in _rules({"s.w": entry}, {"s.w": float})


def test_an_interval_declared_for_a_word_is_refused() -> None:
    """The near miss the issue this file answers names by itself.

    Six of these parameters take a word or a flag. A bound of zero to one written
    for one of them reads as a declared range and sweeps nothing.
    """
    entry = _interval(
        lower={"value": 0.0, "source": "judgement", "where": _SENTENCE},
        upper={"value": 1.0, "source": "judgement", "where": _SENTENCE},
    )
    assert "shape-disagrees-with-the-parameter" in _rules({"s.model": entry}, {"s.model": str})


def test_a_set_declared_for_a_number_is_refused() -> None:
    entry = {
        "kind": "set",
        "values": [{"value": 1.0, "source": "judgement", "where": _SENTENCE}],
    }
    assert "shape-disagrees-with-the-parameter" in _rules({"s.w": entry}, {"s.w": float})


def test_a_nullable_parameter_with_no_null_declared_is_refused() -> None:
    assert "null-declared-wrongly" in _rules({"s.w": _interval()}, {"s.w": float | None})


def test_a_null_declared_for_a_parameter_that_refuses_one_is_refused() -> None:
    entry = _interval(null={"source": "judgement", "where": _SENTENCE})
    assert "null-declared-wrongly" in _rules({"s.w": entry}, {"s.w": float})


def test_an_interval_running_downwards_is_refused() -> None:
    entry = _interval(
        lower={"value": 9.0, "source": "judgement", "where": _SENTENCE},
        upper={"value": 2.0, "source": "judgement", "where": _SENTENCE},
    )
    assert "interval-runs-downwards" in _rules({"s.w": entry}, {"s.w": float})


def test_a_fractional_bound_on_a_whole_number_parameter_is_refused() -> None:
    assert "fractional-bound-on-a-whole-number" in _rules({"s.n": _interval()}, {"s.n": int})


def test_a_flag_whose_set_holds_one_value_is_refused() -> None:
    entry = {
        "kind": "set",
        "values": [{"value": True, "source": "published", "where": _SENTENCE}],
    }
    assert "flag-with-one-value" in _rules({"s.exclude": entry}, {"s.exclude": bool})


def test_a_value_outside_the_enumeration_is_refused() -> None:
    entry = {
        "kind": "set",
        "values": [{"value": "mean", "source": "judgement", "where": _SENTENCE}],
    }
    assert "value-outside-the-enumeration" in _rules(
        {"s.consensus": entry}, {"s.consensus": cmc.ConsensusRule}
    )


def test_a_value_declared_twice_is_refused() -> None:
    entry = {
        "kind": "set",
        "values": [
            {"value": "median", "source": "judgement", "where": _SENTENCE},
            {"value": "median", "source": "judgement", "where": _SENTENCE},
        ],
    }
    assert "repeated-value" in _rules({"s.consensus": entry}, {"s.consensus": cmc.ConsensusRule})


def test_counts_that_disagree_with_the_declaration_are_refused() -> None:
    declared = {"s.w": _interval()}
    stated = dict.fromkeys(SOURCES, 0) | {"judgement": 7}
    assert "counts-disagree" in {
        finding.rule for finding in problems(declared, {"s.w": float}, stated)
    }


def test_a_class_left_out_of_the_table_is_refused() -> None:
    declared = {"s.w": _interval()}
    stated = {"judgement": 2}
    assert "class-missing-from-the-table" in {
        finding.rule for finding in problems(declared, {"s.w": float}, stated)
    }
