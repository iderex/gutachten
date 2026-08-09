"""The preregistered design, checked against the ranges it is drawn over.

A preregistration is worth what it costs to change quietly. This file is what
makes changing it cost something: the design names a coordinate for every
parameter with a declared range and no others, every mapping is one of the four
words the page argues for, every controlled coordinate names the parameter that
controls it, and the counts written into the page match the file.

What it cannot check is whether the design is a good one. The sample size is
arithmetic on a measured cost and the arithmetic is checked here; whether
sixty four base samples over forty coordinates is enough to estimate an
interaction term is a judgement, the page states the expectation that it is not,
and the run in #87 is what settles it.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs" / "sensitivity-design.json"
RANGES = ROOT / "docs" / "ranges.json"
PAGE = ROOT / "docs" / "sensitivity-design.md"

#: How a drawn unit coordinate becomes a parameter value. A closed vocabulary,
#: because the page argues for each of the four and a fifth word would be a
#: mapping nobody argued for.
MAPPINGS = ("direct", "controlled", "mode-split", "snapped")

#: The estimator's cost in evaluations, as the page states it: N(k+2) for the
#: first order and total order indices together.
EVALUATIONS_PER_SAMPLE = 2

design = json.loads(DESIGN.read_text(encoding="utf-8"))
ranges = json.loads(RANGES.read_text(encoding="utf-8"))["parameters"]
coordinates = {entry["parameter"]: entry for entry in design["coordinates"]}


def test_every_parameter_with_a_declared_range_has_a_coordinate() -> None:
    """A parameter left out of the design is one held still without saying so.

    That is the criticism this design is written against, one level up: a
    parameter held at its tuned value cannot afterwards be reported as
    unimportant.
    """
    missing = sorted(set(ranges) - set(coordinates))
    assert not missing, (
        f"{missing} have a declared plausible range and no coordinate in the design. A "
        "parameter the design does not move is held at its base and cannot be reported on."
    )


def test_no_coordinate_names_a_parameter_with_no_declared_range() -> None:
    stray = sorted(set(coordinates) - set(ranges))
    assert not stray, (
        f"{stray} are coordinates of the design and docs/ranges.json declares no range for "
        "them. The ranges are the axis every index is computed along."
    )


def test_no_parameter_is_named_twice() -> None:
    named = [entry["parameter"] for entry in design["coordinates"]]
    assert len(set(named)) == len(named), f"a parameter appears twice: {sorted(named)}"


def test_every_mapping_is_one_of_the_four_the_page_argues_for() -> None:
    stray = sorted({entry["mapping"] for entry in coordinates.values()} - set(MAPPINGS))
    assert not stray, f"{stray} are mappings the page does not argue for; the four are {MAPPINGS}"


def test_a_controlled_or_snapped_coordinate_names_what_decides_it() -> None:
    """A coordinate decided by another one, with no other one named, decides itself."""
    offences = [
        name
        for name, entry in coordinates.items()
        if entry["mapping"] in ("controlled", "snapped")
        and coordinates.get(str(entry.get("by"))) is None
    ]
    assert not offences, (
        f"{offences} are decided by another coordinate and name none that the design moves. "
        "A controlling parameter outside the design is one nobody swept."
    )


def test_every_mapping_that_is_not_direct_says_why() -> None:
    offences = sorted(
        name
        for name, entry in coordinates.items()
        if entry["mapping"] != "direct" and not str(entry.get("why", "")).strip()
    )
    assert not offences, (
        f"{offences} depart from the direct mapping and give no reason. A departure written "
        "in silence is the one a reader cannot argue with."
    )


def test_only_a_nullable_parameter_takes_the_null_splitting_mappings() -> None:
    """A mode split or a controlled null is a statement about a null that exists.

    Declaring one for a parameter whose range admits no null would reserve half
    the coordinate for a value the step refuses.
    """
    offences = sorted(
        name
        for name, entry in coordinates.items()
        if entry["mapping"] in ("controlled", "mode-split") and "null" not in ranges[name]
    )
    assert not offences, (
        f"{offences} take a mapping that decides a null and their declared range admits none"
    )


def test_the_null_share_leaves_both_modes_reachable() -> None:
    share = design["null_share"]
    assert 0.0 < share < 1.0, (
        f"the null share is {share}. At nothing or at one, one of the two modes a mode split "
        "exists to compare is never drawn, and the design would report on a mode it never ran."
    )


def test_the_base_sample_is_a_power_of_two() -> None:
    """The balance a Sobol sequence is chosen for holds at powers of two."""
    samples = design["base_samples"]
    assert samples > 0 and samples & (samples - 1) == 0, (
        f"the base sample is {samples}, which is not a power of two. Between them the "
        "sequence loses the balance it was chosen for."
    )


def test_the_cell_count_in_the_page_is_the_arithmetic_of_the_design() -> None:
    """The line a reader costs the sweep from, derived rather than remembered."""
    samples = design["base_samples"]
    width = len(coordinates) + EVALUATIONS_PER_SAMPLE
    pairs = sum(design["pairs"].values())
    evaluations = samples * width
    cells = evaluations * pairs
    page = PAGE.read_text(encoding="utf-8")
    stated = re.search(rf"N=\s*{samples}\s+evaluations=\s*(\d+)\s+cells=\s*(\d+)", page)
    assert stated is not None, (
        f"the page states no line for N={samples}, and the sample size is the number every "
        "other figure in it is divided out of."
    )
    assert (int(stated.group(1)), int(stated.group(2))) == (evaluations, cells), (
        f"the page states {stated.group(1)} evaluations and {stated.group(2)} cells for "
        f"N={samples}; the design's own numbers give {evaluations} and {cells}."
    )


def test_the_mapping_counts_in_the_page_match_the_file() -> None:
    """A count written once and left behind is what this repository calls drift.

    It lands hardest here, on the table saying how much of the space is sampled
    directly and how much through a mapping somebody chose.
    """
    counted = collections.Counter(entry["mapping"] for entry in coordinates.values())
    page = PAGE.read_text(encoding="utf-8")
    for mapping in MAPPINGS:
        row = re.search(rf"^\|\s*`{re.escape(mapping)}`\s*\|\s*(\d+)\s*\|", page, re.MULTILINE)
        assert row is not None, f"the page has no table row for the {mapping!r} mapping"
        assert int(row.group(1)) == counted[mapping], (
            f"the page says {row.group(1)} coordinates are {mapping!r} and the file has "
            f"{counted[mapping]}"
        )


def test_the_page_names_the_base_sample_the_design_declares() -> None:
    """The one number a preregistration is easiest to move quietly.

    The budget table states a line for several sample sizes, so the arithmetic
    check above stays green for any of them. What pins which one was chosen is
    the sentence, and this is what refuses a design whose sample moved while the
    page went on naming the old one.
    """
    samples = design["base_samples"]
    page = PAGE.read_text(encoding="utf-8")
    assert f"the base sample is {samples}" in page, (
        f"the design declares {samples} base samples and the page says no such thing. A "
        "sample size that moved after the results were seen is exactly what preregistering "
        "one is against, and a table of several sizes does not name the chosen one."
    )


def test_the_page_states_the_coordinate_count_the_file_holds() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert f"coordinates: {len(coordinates)}" in page
    assert f"declared ranges: {len(ranges)}" in page
