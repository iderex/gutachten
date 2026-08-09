"""A design is refused where it could not run, and its cells do not depend on order.

Every refusal below is a near miss over the design that actually runs in this
suite, changed in exactly one way. A design is a file somebody edits, so the ways
it can be wrong are the ways a person gets one wrong, and each of them is cheaper
to meet when the file is read than in the middle of a sweep that has been running
for a day.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import gutachten.transforms  # noqa: F401  (importing registers the steps)
from gutachten.sweep.design import Bound, DesignError, load, load_ranges
from gutachten.transforms.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[3]
SMALL = Path(__file__).resolve().parent / "small.json"
RANGES = ROOT / "docs" / "ranges.json"
PROFILE = ROOT / "profiles" / "published-chain.json"


@pytest.fixture(scope="module")
def ranges() -> dict[str, Bound]:
    return load_ranges(RANGES)


def written(directory: Path, name: str, **changes: Any) -> Path:
    """The design that runs here, changed in the ways named, written elsewhere.

    The profile is rewritten to an absolute path because a design resolves it
    relative to itself, and these copies do not sit beside `profiles/`.
    """
    declared = json.loads(SMALL.read_text(encoding="utf-8"))
    declared["profile"] = str(PROFILE)
    declared["name"] = name
    declared.update(changes)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    return path


def test_the_design_this_suite_runs_loads(ranges: dict[str, Bound]) -> None:
    design = load(SMALL, REGISTRY, ranges)
    assert design.name == "small"
    assert [varied.parameter for varied in design.varied] == [
        "trim-edge.width",
        "compare.register.grid",
    ]
    assert [pair.kind for pair in design.pairs] == ["matching", "non-matching"]


def test_every_cell_names_every_varied_parameter(ranges: dict[str, Bound]) -> None:
    """A one-at-a-time design still writes a complete assignment.

    The alternative is a row saying what one parameter was and leaving the reader
    to infer the rest from the design, which is the inference a results table
    exists to make unnecessary.
    """
    design = load(SMALL, REGISTRY, ranges)
    names = {varied.parameter for varied in design.varied}
    for cell in design.cells():
        assert {parameter for parameter, _ in cell.assignment} == names


def test_the_base_cell_appears_once(ranges: dict[str, Bound]) -> None:
    """Each arm of a one-at-a-time design passes through the base.

    Without the deduplication the base is computed once per varied parameter and
    every summary taken over the table counts it that many times.
    """
    design = load(SMALL, REGISTRY, ranges)
    base = tuple(
        sorted((varied.parameter, design.base(varied.parameter)) for varied in design.varied)
    )
    at_base = [cell for cell in design.cells() if cell.assignment == base]
    assert len(at_base) == len(design.pairs)


def test_the_identifiers_do_not_depend_on_the_order_they_were_generated_in(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    """The identifier is what a resumed run recognises a completed cell by.

    An identifier depending on generation order would make a resumed run
    recompute everything while reporting that it had reused nothing, which reads
    exactly like a fresh run.
    """
    first = load(SMALL, REGISTRY, ranges)
    reversed_vary = list(reversed(json.loads(SMALL.read_text(encoding="utf-8"))["vary"]))
    second = load(written(tmp_path, "small", vary=reversed_vary), REGISTRY, ranges)
    assert {cell.identifier for cell in first.cells()} == {
        cell.identifier for cell in second.cells()
    }


def test_a_full_factorial_design_crosses_every_value(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    design = load(written(tmp_path, "crossed", generator="full-factorial"), REGISTRY, ranges)
    assert len(design.cells()) == 2 * 2 * len(design.pairs)


def test_a_parameter_with_no_declared_range_cannot_be_swept(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    """The clause #81 asks for, at the place a sweep would have moved it.

    A parameter absent from the declaration has no bounds anybody argued over, so
    a design moving it would take its axis from whoever wrote the design, inside
    the change that produced the numbers.
    """
    path = written(tmp_path, "unranged", vary=[{"parameter": "trim-edge.nonsense", "points": 2}])
    with pytest.raises(DesignError, match="no declared range"):
        load(path, REGISTRY, ranges)


def test_a_value_outside_the_declared_range_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    path = written(
        tmp_path, "outside", vary=[{"parameter": "trim-edge.width", "values": [40.0, 4000.0]}]
    )
    with pytest.raises(DesignError, match="declared range does not admit"):
        load(path, REGISTRY, ranges)


def test_a_parameter_varied_over_one_value_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    path = written(tmp_path, "still", vary=[{"parameter": "trim-edge.width", "values": [40.0]}])
    with pytest.raises(DesignError, match="has not been varied"):
        load(path, REGISTRY, ranges)


def test_a_value_visited_twice_is_refused(ranges: dict[str, Bound], tmp_path: Path) -> None:
    path = written(
        tmp_path, "twice", vary=[{"parameter": "trim-edge.width", "values": [40.0, 40.0]}]
    )
    with pytest.raises(DesignError, match="visit a value twice"):
        load(path, REGISTRY, ranges)


def test_a_design_that_varies_nothing_is_refused(ranges: dict[str, Bound], tmp_path: Path) -> None:
    with pytest.raises(DesignError, match="varies nothing"):
        load(written(tmp_path, "flat", vary=[]), REGISTRY, ranges)


def test_a_parameter_declared_varied_twice_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    entry = {"parameter": "trim-edge.width", "values": [40.0, 80.0]}
    with pytest.raises(DesignError, match="varied twice"):
        load(written(tmp_path, "doubled", vary=[entry, entry]), REGISTRY, ranges)


def test_a_parameter_of_a_step_the_profile_does_not_run_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    """The chain being swept is the reproduction chain, which masks nothing.

    A design moving a masking parameter around it would report a flat curve for a
    step that never ran, which reads as a robust step rather than as an absent one.
    """
    path = written(
        tmp_path,
        "unrun",
        vary=[{"parameter": "mask-marks.exclude_drag", "values": [True, False]}],
    )
    with pytest.raises(DesignError, match="does not run it"):
        load(path, REGISTRY, ranges)


def test_a_count_of_points_across_a_set_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    path = written(tmp_path, "counted", vary=[{"parameter": "level.model", "points": 2}])
    with pytest.raises(DesignError, match="no interval to step along"):
        load(path, REGISTRY, ranges)


def test_a_number_with_neither_values_nor_points_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    path = written(tmp_path, "unsaid", vary=[{"parameter": "trim-edge.width"}])
    with pytest.raises(DesignError, match="neither which values"):
        load(path, REGISTRY, ranges)


def test_points_across_a_whole_number_range_that_repeat_are_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    """A grid of three to twelve has ten values and a design may not ask for more.

    Rounding two points onto one whole number is a cell visited twice, which every
    summary over the table counts twice.
    """
    path = written(tmp_path, "crowded", vary=[{"parameter": "compare.register.grid", "points": 40}])
    with pytest.raises(DesignError, match="repeat a value"):
        load(path, REGISTRY, ranges)


def test_a_set_with_no_values_named_visits_the_whole_admissible_set(
    ranges: dict[str, Bound],
) -> None:
    bound = ranges["level.model"]
    assert bound.kind == "set"
    assert set(bound.values) == {"plane", "polynomial", "sphere"}


def test_a_generator_this_runner_does_not_offer_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    path = written(tmp_path, "sobol", generator="low-discrepancy")
    with pytest.raises(DesignError, match="generator"):
        load(path, REGISTRY, ranges)


def test_a_surface_parameter_the_design_leaves_out_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    """The generator carries defaults and a design may not lean on them.

    A surface parameter nobody stated is one nobody chose for the sweep, and the
    noise level it silently takes decides how many cells clear a correlation
    threshold.
    """
    surface = json.loads(SMALL.read_text(encoding="utf-8"))["surface"]
    del surface["noise_um"]
    with pytest.raises(DesignError, match="the generator takes"):
        load(written(tmp_path, "partial", surface=surface), REGISTRY, ranges)


def test_a_seed_stated_for_the_surface_is_refused(ranges: dict[str, Bound], tmp_path: Path) -> None:
    surface = json.loads(SMALL.read_text(encoding="utf-8"))["surface"] | {"seed": 1}
    with pytest.raises(DesignError, match="the seed excepted"):
        load(written(tmp_path, "seeded", surface=surface), REGISTRY, ranges)


def test_a_pair_of_a_kind_this_runner_cannot_make_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    pairs = [{"name": "casework", "kind": "unknown-source", "seed": 1}]
    with pytest.raises(DesignError, match="the kinds are"):
        load(written(tmp_path, "unknown", pairs=pairs), REGISTRY, ranges)


def test_a_design_with_no_pair_is_refused(ranges: dict[str, Bound], tmp_path: Path) -> None:
    with pytest.raises(DesignError, match="declares no pair"):
        load(written(tmp_path, "unpaired", pairs=[]), REGISTRY, ranges)


def test_two_pairs_under_one_name_are_refused(ranges: dict[str, Bound], tmp_path: Path) -> None:
    pairs = [
        {"name": "same-source", "kind": "matching", "seed": 1},
        {"name": "same-source", "kind": "non-matching", "seed": 2},
    ]
    with pytest.raises(DesignError, match="declared as"):
        load(written(tmp_path, "renamed", pairs=pairs), REGISTRY, ranges)


def test_a_design_whose_recorded_name_is_not_its_file_name_is_refused(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    path = written(tmp_path, "onedisk")
    declared = json.loads(path.read_text(encoding="utf-8"))
    declared["name"] = "in-the-file"
    path.write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(DesignError, match="records the name"):
        load(path, REGISTRY, ranges)


def test_a_design_that_is_not_json_says_so(ranges: dict[str, Bound], tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(DesignError, match="not readable as JSON"):
        load(path, REGISTRY, ranges)


def test_a_count_of_points_steps_across_the_declared_range(
    ranges: dict[str, Bound], tmp_path: Path
) -> None:
    """Both kinds of interval, because they are stepped differently.

    A whole number parameter is stepped in whole numbers, so a design never
    produces a cell the parameter record refuses halfway through a sweep.
    """
    path = written(
        tmp_path,
        "stepped",
        vary=[
            {"parameter": "trim-edge.width", "points": 3},
            {"parameter": "compare.register.grid", "points": 3},
        ],
    )
    design = load(path, REGISTRY, ranges)
    values = {varied.parameter: varied.values for varied in design.varied}
    assert values["trim-edge.width"] == (4.0, 202.0, 400.0)
    assert values["compare.register.grid"] == (3, 8, 12)


def test_a_bound_admits_a_null_only_where_one_is_declared(ranges: dict[str, Bound]) -> None:
    """The null is a state a design may visit and no interval contains it."""
    assert ranges["level.robust_tuning"].admits(None) is True
    assert ranges["trim-edge.width"].admits(None) is False
    assert ranges["trim-edge.width"].admits("wide") is False
    assert ranges["level.model"].admits("plane") is True
