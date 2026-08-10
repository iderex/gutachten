"""The two score distributions, what the set behind them supports, and every refusal.

The estimates themselves are arithmetic and would be dull to assert one number
at a time. What is worth proving is the part that decides whether a number gets
printed at all: that the bound counts sources and not pairs, and that a score
the reference set never saw comes out as the limit of the set rather than as a
very large ratio with nothing behind it.

Every refusal here was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import math

import pytest

from gutachten.conclusions import conclusion_words
from gutachten.stats.distributions import (
    BoundedLogRatio,
    Comparison,
    Proposition,
    ReferenceSet,
    estimate,
    hold_within,
    supportable_range,
)
from tests.support.tolerance import assert_close

CELLS = 20

# Eight firearms, and every ordered pair of distinct ones compared once, which
# is how a reference set of this kind is actually assembled. It produces many
# more comparisons than sources, which is the gap every bound here turns on.
FIREARMS = tuple(f"firearm-{index}" for index in range(8))


def a_reference_set(name: str = "synthetic-eight") -> ReferenceSet:
    """A set whose pair count and source count are far apart, deliberately.

    Twenty-eight different-source comparisons over eight firearms, and eight
    same-source comparisons, one per firearm. A bound read off the pair count
    would be more than three times the one read off the sources.
    """
    comparisons = []
    for index, firearm in enumerate(FIREARMS):
        comparisons.append(Comparison(score=14 + index % 3, source_a=firearm, source_b=firearm))
    for first in range(len(FIREARMS)):
        for second in range(first + 1, len(FIREARMS)):
            comparisons.append(
                Comparison(
                    score=(first + second) % 4,
                    source_a=FIREARMS[first],
                    source_b=FIREARMS[second],
                )
            )
    return ReferenceSet(name=name, cells=CELLS, comparisons=tuple(comparisons))


def test_both_distributions_are_estimated_over_the_declared_support() -> None:
    reference = a_reference_set()

    same = estimate(reference, Proposition.SAME_SOURCE)
    different = estimate(reference, Proposition.DIFFERENT_SOURCE)

    for one in (same, different):
        # The declared support and not the observed one, so two sets produce
        # distributions of one length that can be read side by side.
        assert len(one.probability) == CELLS + 1
        assert_close(sum(one.probability), 1.0, what=f"{one.proposition.value} total", atol=1e-12)
        assert one.reference_set == "synthetic-eight"

    # The observed frequency and nothing else. Three of the eight same-source
    # comparisons scored 14, so the estimate at 14 is three eighths exactly.
    assert_close(same.probability_of(14), 3 / 8, what="p(14 | same source)", atol=1e-12)
    # Unsmoothed: a score nobody saw is zero rather than something small.
    assert_close(same.probability_of(0), 0.0, what="p(0 | same source)", atol=0.0)
    assert_close(different.probability_of(14), 0.0, what="p(14 | different source)", atol=0.0)


def test_the_pair_count_and_the_source_count_are_both_reported_and_differ() -> None:
    # They are different numbers and the smaller one governs, so a reader has to
    # meet both. Twenty-eight comparisons drawn from eight firearms is eight
    # independent units, and the gap is the whole reason the next test exists.
    different = estimate(a_reference_set(), Proposition.DIFFERENT_SOURCE)

    assert different.counts.pairs == 28
    assert different.counts.sources == 8


def test_the_supportable_range_is_computed_from_the_sources_and_not_from_the_pairs() -> None:
    # The near miss, and it is the one somebody writes without noticing: the
    # pair count is the number in front of you, it is larger, and using it gives
    # a ceiling of 1.45 in place of 0.90. That is a factor of three and a half on
    # a ratio, in the direction that flatters the method.
    bounds = supportable_range(a_reference_set())

    assert_close(bounds.most_positive, math.log10(8), what="ceiling", atol=1e-12)
    assert_close(bounds.most_negative, -math.log10(8), what="floor", atol=1e-12)
    assert bounds.different_source.pairs == 28
    assert bounds.different_source.sources == 8
    # Stated as the size of the mistake rather than as an inequality, because
    # the number a reader wants is how far the wrong ceiling would have been.
    assert_close(
        math.log10(28) - bounds.most_positive,
        math.log10(3.5),
        what="how far above the sources' ceiling the pairs' ceiling would sit",
        atol=1e-12,
    )


def test_a_score_the_set_never_saw_is_reported_as_the_bound_and_not_as_a_number() -> None:
    # This is the failure mode that makes this class of method contentious. The
    # score is common under one proposition and unobserved under the other, so
    # the raw ratio is an infinity, and a system that printed it would report a
    # ratio of ten to the anything from a probability nobody measured.
    reference = a_reference_set()
    same = estimate(reference, Proposition.SAME_SOURCE)
    different = estimate(reference, Proposition.DIFFERENT_SOURCE)
    bounds = supportable_range(reference)

    over = same.probability_of(14)
    under = different.probability_of(14)
    assert over > 0.0
    assert under == 0.0
    raw = math.inf

    held = hold_within(bounds, raw)

    assert held.at_the_bound
    assert_close(held.value, math.log10(8), what="held ratio", atol=1e-12)
    assert held.sources_behind_the_bound == 8
    assert "as far as reference set" in held.statement()
    assert "synthetic-eight" in held.statement()


def test_a_ratio_the_set_carries_is_reported_as_itself() -> None:
    # The bound is not a cap applied to everything. A value inside the range
    # comes back unchanged and says nothing about a limit, or the limit would
    # read as a property of every result.
    bounds = supportable_range(a_reference_set())

    held = hold_within(bounds, 0.5)

    assert not held.at_the_bound
    assert_close(held.value, 0.5, what="reported ratio", atol=1e-12)
    assert "as far as" not in held.statement()


def test_the_bound_holds_in_the_negative_direction_too() -> None:
    # A ratio far below one is a statement about the same-source data, and it is
    # that side's source count that has to carry it.
    bounds = supportable_range(a_reference_set())

    held = hold_within(bounds, -math.inf)

    assert held.at_the_bound
    assert_close(held.value, -math.log10(8), what="held ratio", atol=1e-12)
    assert held.sources_behind_the_bound == 8
    assert "below" in held.statement()


def test_no_statement_this_module_prints_says_what_a_comparison_showed() -> None:
    # Neither exemption mark is used, so this passes on the words themselves.
    bounds = supportable_range(a_reference_set())

    for held in (hold_within(bounds, 0.5), hold_within(bounds, math.inf)):
        assert conclusion_words(held.statement(), source="distributions") == []


def test_a_set_resting_on_one_source_says_so_in_its_diagnostics() -> None:
    # The diagnostic that matters most, and the case it is for: a set that looks
    # like six comparisons and is really one firearm compared against five
    # others. The pair count hides it and the source count does not catch it
    # either, because six sources is six sources.
    lopsided = ReferenceSet(
        name="one-firearm-against-the-rest",
        cells=CELLS,
        comparisons=(
            Comparison(score=12, source_a="a", source_b="a"),
            Comparison(score=1, source_a="a", source_b="b"),
            Comparison(score=2, source_a="a", source_b="c"),
            Comparison(score=3, source_a="a", source_b="d"),
            Comparison(score=1, source_a="a", source_b="e"),
            Comparison(score=2, source_a="a", source_b="f"),
        ),
    )

    different = estimate(lopsided, Proposition.DIFFERENT_SOURCE)
    spread = estimate(a_reference_set(), Proposition.DIFFERENT_SOURCE)

    assert different.counts.sources == 6
    assert_close(
        different.diagnostics.largest_source_share, 1.0, what="largest source share", atol=1e-12
    )
    # The set assembled the ordinary way sits well below one, so the number is
    # one that moves rather than one that is always high.
    assert_close(
        spread.diagnostics.largest_source_share, 7 / 28, what="largest source share", atol=1e-12
    )


def test_the_diagnostics_report_what_was_seen_and_what_was_not() -> None:
    different = estimate(a_reference_set(), Proposition.DIFFERENT_SOURCE)

    assert different.diagnostics.observed_low == 0
    assert different.diagnostics.observed_high == 3
    assert different.diagnostics.distinct_scores == 4
    # Seventeen of the twenty-one possible scores were never seen. High for any
    # real set, reported rather than reassured about.
    assert_close(
        different.diagnostics.unobserved_support_share,
        17 / 21,
        what="unobserved share of the support",
        atol=1e-12,
    )


def test_a_score_outside_the_declared_support_is_refused() -> None:
    with pytest.raises(ValueError, match="above the cell count"):
        ReferenceSet(
            name="too-large",
            cells=4,
            comparisons=(
                Comparison(score=5, source_a="a", source_b="a"),
                Comparison(score=1, source_a="a", source_b="b"),
            ),
        )
    reading = estimate(a_reference_set(), Proposition.SAME_SOURCE)
    with pytest.raises(ValueError, match="outside the support"):
        reading.probability_of(CELLS + 1)


def test_a_set_holding_only_one_proposition_is_refused() -> None:
    # Both propositions or no ratio. A set supplying one of them would hand the
    # other whatever the code happened to do with an empty sequence, which for a
    # division is a crash somewhere further down and for a sum is zero.
    with pytest.raises(ValueError, match="holds no different-source comparison"):
        ReferenceSet(
            name="same-source-only",
            cells=CELLS,
            comparisons=(Comparison(score=3, source_a="a", source_b="a"),),
        )
    with pytest.raises(ValueError, match="holds no same-source comparison"):
        ReferenceSet(
            name="different-source-only",
            cells=CELLS,
            comparisons=(Comparison(score=3, source_a="a", source_b="b"),),
        )


def test_an_unnamed_reference_set_or_an_unusable_cell_count_is_refused() -> None:
    with pytest.raises(ValueError, match="must be named"):
        ReferenceSet(name="  ", cells=CELLS, comparisons=a_reference_set().comparisons)
    with pytest.raises(ValueError, match="positive integer"):
        ReferenceSet(name="no-cells", cells=0, comparisons=a_reference_set().comparisons)
    # True is an int in Python and would otherwise declare a support of one cell.
    with pytest.raises(ValueError, match="positive integer"):
        ReferenceSet(name="boolean", cells=True, comparisons=a_reference_set().comparisons)  # type: ignore[arg-type]


def test_a_comparison_without_a_usable_score_or_a_named_source_is_refused() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        Comparison(score=3.0, source_a="a", source_b="b")  # type: ignore[arg-type]
    # True is an int and would otherwise be recorded as a score of one.
    with pytest.raises(TypeError, match="must be an integer"):
        Comparison(score=True, source_a="a", source_b="b")
    with pytest.raises(ValueError, match="cannot be negative"):
        Comparison(score=-1, source_a="a", source_b="b")
    with pytest.raises(ValueError, match="source_b must name a source"):
        Comparison(score=1, source_a="a", source_b="   ")


def test_a_ratio_that_is_not_a_number_is_refused_rather_than_held_at_a_bound() -> None:
    # An infinity is a result beyond the set and is held. A not-a-number is a
    # computation that failed, and holding it would print the failure at the
    # bound where it reads exactly like a strong result.
    bounds = supportable_range(a_reference_set())

    with pytest.raises(ValueError, match="computation that failed"):
        hold_within(bounds, math.nan)


def test_a_probability_read_with_something_that_is_not_a_score_is_refused() -> None:
    reading = estimate(a_reference_set(), Proposition.SAME_SOURCE)

    with pytest.raises(TypeError, match="a score is an integer"):
        reading.probability_of(3.0)  # type: ignore[arg-type]


def test_the_bounded_result_carries_the_set_it_came_from() -> None:
    # The design constraint from the milestone, at this level: the number cannot
    # be built without the identity of the set behind it, so it cannot be
    # printed without it either.
    bounds = supportable_range(a_reference_set(name="named-set"))

    held = hold_within(bounds, 0.25)

    assert isinstance(held, BoundedLogRatio)
    assert held.reference_set == "named-set"
    assert "named-set" in held.statement()
