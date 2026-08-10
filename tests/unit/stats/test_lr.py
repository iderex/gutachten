"""The ratio, its interval, and the barrier against quoting it alone.

The arithmetic here is short. What is worth proving is the part that decides
what a reader ends up holding: that the number cannot be printed without the
propositions and the set behind it, and that taking sources out of the reference
set makes the interval worse rather than better.

Every refusal here was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gutachten.conclusions import conclusion_words
from gutachten.stats.distributions import Comparison, ReferenceSet, supportable_range
from gutachten.stats.lr import COVERAGE, LogRatio, Propositions, log_ratio
from tests.support.tolerance import assert_close

CELLS = 20
RESAMPLES = 400
SEED = 20260810

PROPOSITIONS = Propositions(
    numerator="the two surfaces were produced by one firearm",
    denominator="the two surfaces were produced by two firearms from the same population",
    population="firearms of this model, consecutively manufactured",
)

# A score both propositions were seen at, so the ratio is a finite number and
# the interval is about sampling rather than about the bound. Overlap is what a
# real reference set has, and a fixture without it would only ever exercise the
# infinite case.
OVERLAPPING = 8


#: Seen under the same-source proposition and under no other. The infinite case.
ONLY_SAME_SOURCE = OVERLAPPING + 2


def a_reference_set(firearms: int = 10, name: str = "synthetic-overlap") -> ReferenceSet:
    """A set whose two distributions overlap, over a stated number of firearms.

    Half the same-source comparisons and half the different-source ones sit at
    ``OVERLAPPING``, which is where every assertion below is made. The half is
    deliberate and it was calibrated rather than guessed: with a small share of
    the sources carrying the score, a resample that dropped all of them is
    likelier than the tail the interval is cut at, and the endpoint comes out
    infinite for that reason rather than for anything about the estimate. The
    first shape this file used put three sources in ten at the score and the
    lower endpoint was an infinity in about one run in forty.

    The pattern is arithmetic on the index rather than drawn, so the set is the
    same set on every platform and the interval is a property of the resampling.
    """
    names = [f"firearm-{index}" for index in range(firearms)]
    comparisons = [
        Comparison(
            score=OVERLAPPING if index % 2 == 0 else ONLY_SAME_SOURCE,
            source_a=name_,
            source_b=name_,
        )
        for index, name_ in enumerate(names)
    ]
    for first in range(firearms):
        for second in range(first + 1, firearms):
            comparisons.append(
                Comparison(
                    score=OVERLAPPING if (first + second) % 2 == 0 else OVERLAPPING - 2,
                    source_a=names[first],
                    source_b=names[second],
                )
            )
    return ReferenceSet(name=name, cells=CELLS, comparisons=tuple(comparisons))


def a_ratio(reference: ReferenceSet | None = None, score: int = OVERLAPPING) -> LogRatio:
    return log_ratio(
        a_reference_set() if reference is None else reference,
        score,
        propositions=PROPOSITIONS,
        resamples=RESAMPLES,
        seed=SEED,
    )


def test_a_ratio_and_an_interval_are_produced_for_a_score() -> None:
    result = a_ratio()

    assert result.score == OVERLAPPING
    assert result.reference_set == "synthetic-overlap"
    assert_close(result.coverage, COVERAGE, what="the interval coverage", atol=0.0)
    # An interval that contains its own point estimate, which is the least a
    # bootstrap interval has to do and the first thing an off-by-one in the
    # order statistics breaks.
    assert result.low.value <= result.point.value <= result.high.value
    assert result.resamples == RESAMPLES
    assert result.resamples_discarded == 0
    # Both counts travel with the result, because the smaller one governs.
    assert result.same_source.pairs == 10
    assert result.same_source.sources == 10
    assert result.different_source.pairs == 45
    assert result.different_source.sources == 10


def test_the_result_cannot_be_formatted_without_its_propositions_and_reference_set() -> None:
    # The design constraint of this issue. A format specification is exactly how
    # somebody prints the ratio on its own, so it raises rather than producing
    # that line, and the plain form carries everything.
    result = a_ratio()

    with pytest.raises(ValueError, match="drops the propositions and the reference set"):
        f"{result:.2f}"
    with pytest.raises(ValueError, match="drops the propositions and the reference set"):
        format(result, ">40")

    whole = f"{result}"
    assert PROPOSITIONS.numerator in whole
    assert PROPOSITIONS.denominator in whole
    assert PROPOSITIONS.population in whole
    assert "synthetic-overlap" in whole


def test_a_ratio_without_stated_propositions_cannot_be_built_at_all() -> None:
    # The barrier is at construction as well as at printing, or a caller would
    # simply pass two empty strings and print the whole statement with nothing
    # in it.
    with pytest.raises(ValueError, match="numerator proposition must be stated"):
        Propositions(numerator="   ", denominator="two firearms", population="this model")
    with pytest.raises(ValueError, match="population proposition must be stated"):
        Propositions(numerator="one firearm", denominator="two firearms", population="")
    # The near miss is two propositions that are the same sentence, which gives
    # a ratio of one by construction and prints 0.00 like a real result.
    with pytest.raises(ValueError, match="the same sentence"):
        Propositions(numerator="one firearm", denominator="one firearm ", population="this model")


def test_the_interval_widens_as_the_reference_set_is_subsampled() -> None:
    # The property this issue asks to be demonstrated. Fewer sources is less
    # evidence and the interval has to say so.
    #
    # Measured on the width before the bound is applied, and that is not a
    # convenience. The bound is itself a function of the source count, so taking
    # sources away tightens it, and the held interval can narrow while the
    # estimate underneath it is getting worse. That is the reason the unheld
    # width is carried at all.
    widths = [a_ratio(a_reference_set(firearms=count)).unheld_width for count in (24, 14, 8)]

    assert widths[0] < widths[1] < widths[2], widths

    # The reason the width above is the unheld one, stated as the fact it rests
    # on rather than as a claim about which direction the held interval moved on
    # this particular fixture: the bound tightens as sources are taken away, so
    # a held endpoint is not a thing to read a trend off.
    assert (
        supportable_range(a_reference_set(firearms=8)).most_positive
        < supportable_range(a_reference_set(firearms=24)).most_positive
    )


def test_the_same_seed_gives_the_same_interval_and_a_different_one_does_not() -> None:
    # A bootstrap with an unrecorded seed is an interval nobody else gets. The
    # seed is required and it decides the answer, which is checked in both
    # directions so this cannot pass against a resampler that ignores it.
    reference = a_reference_set()
    first = log_ratio(reference, OVERLAPPING, propositions=PROPOSITIONS, resamples=200, seed=1)
    again = log_ratio(reference, OVERLAPPING, propositions=PROPOSITIONS, resamples=200, seed=1)
    other = log_ratio(reference, OVERLAPPING, propositions=PROPOSITIONS, resamples=200, seed=2)

    assert_close(first.unheld_width, again.unheld_width, what="the interval width", atol=0.0)
    assert first.unheld_width != other.unheld_width


def test_an_endpoint_beyond_what_the_set_supports_is_reported_at_the_bound() -> None:
    # A score seen under one proposition and not the other gives an infinite
    # raw ratio, which is the case the whole bound exists for. It comes out as
    # the limit of the set with the source count that set it, not as a number.
    result = a_ratio(score=ONLY_SAME_SOURCE)

    assert result.point.at_the_bound
    assert_close(result.point.value, math.log10(10), what="the held ratio", atol=1e-12)
    assert result.point.sources_behind_the_bound == 10
    assert math.isinf(result.unheld_width)


def test_a_score_seen_under_neither_proposition_is_refused_rather_than_bounded() -> None:
    # Zero divided by zero. Reporting a bound for it would say the set had
    # something to say about a score it never saw, which is the opposite of what
    # the bound is for.
    with pytest.raises(ValueError, match="seen under neither proposition"):
        a_ratio(score=CELLS)


def test_a_bootstrap_of_one_resample_is_refused() -> None:
    # One resample gives an interval of zero width, which reads as certainty.
    with pytest.raises(ValueError, match="at least two resamples"):
        log_ratio(a_reference_set(), OVERLAPPING, propositions=PROPOSITIONS, resamples=1, seed=SEED)
    with pytest.raises(ValueError, match="at least two resamples"):
        log_ratio(
            a_reference_set(), OVERLAPPING, propositions=PROPOSITIONS, resamples=True, seed=SEED
        )


def test_a_score_that_is_not_an_integer_is_refused() -> None:
    with pytest.raises(TypeError, match="is an integer"):
        log_ratio(a_reference_set(), 8.0, propositions=PROPOSITIONS, resamples=RESAMPLES, seed=SEED)  # type: ignore[arg-type]


def test_the_whole_statement_says_nothing_about_what_a_comparison_showed() -> None:
    # A ratio is not a conclusion and the line that prints it must not read as
    # one. No exemption mark is used, so this passes on the words themselves.
    assert conclusion_words(str(a_ratio()), source="lr") == []


def test_the_statement_carries_both_counts_so_the_smaller_one_is_visible() -> None:
    whole = str(a_ratio())

    assert "10 same-source comparisons over 10 sources" in whole
    assert "45 different-source comparisons over 10 sources" in whole


def a_two_firearm_set() -> ReferenceSet:
    """The smallest set a bootstrap over sources can be asked about.

    Two firearms, one same-source comparison each and one different-source
    comparison between them. Half the draws of two sources with replacement pick
    one firearm twice, and the different-source comparison then has no weight at
    all, so half the resamples say nothing about a ratio.
    """
    return ReferenceSet(
        name="two-firearms",
        cells=CELLS,
        comparisons=(
            Comparison(score=5, source_a="a", source_b="a"),
            Comparison(score=5, source_a="b", source_b="b"),
            Comparison(score=3, source_a="a", source_b="b"),
        ),
    )


def test_a_resample_that_empties_a_proposition_is_discarded_and_counted() -> None:
    # Discarded rather than absorbed. A draw with no different-source comparison
    # in it has no denominator, and counting it as a zero would be inventing an
    # observation at the end of the distribution the whole bound is about.
    result = log_ratio(a_two_firearm_set(), 5, propositions=PROPOSITIONS, resamples=200, seed=7)

    assert result.resamples_discarded == 105
    assert result.resamples == 200


def test_a_set_too_small_for_a_bootstrap_over_sources_is_refused_rather_than_narrowed() -> None:
    # The seed is searched rather than chosen: with two resamples of this set,
    # both draws land on one firearm about one time in four, and seed 4 is the
    # first that does. What is being proven is that the answer is a refusal
    # naming the set rather than an interval computed some other way, because
    # the other way available is resampling pairs and that is the failure #93
    # calls the more consequential one in the milestone.
    with pytest.raises(ValueError, match="too small in sources for a bootstrap over sources"):
        log_ratio(a_two_firearm_set(), 5, propositions=PROPOSITIONS, resamples=2, seed=4)


def a_pair_resampled_width(
    reference: ReferenceSet, score: int, *, resamples: int, seed: int
) -> float:
    """The interval width this module would produce if it resampled pairs.

    Written out here rather than offered as an option in the source, because it
    is the thing the design refuses and an option is how a refused thing gets
    used. It draws comparisons with replacement, which treats the many pairs
    from one firearm as independent of each other.
    """
    generator = np.random.default_rng(seed)
    comparisons = reference.comparisons
    drawn = []
    for _ in range(resamples):
        picked = [
            comparisons[int(index)]
            for index in generator.integers(0, len(comparisons), len(comparisons))
        ]
        same = [item.score for item in picked if item.source_a == item.source_b]
        different = [item.score for item in picked if item.source_a != item.source_b]
        if not same or not different:
            continue
        over = same.count(score) / len(same)
        under = different.count(score) / len(different)
        if over == 0.0 or under == 0.0:
            continue
        drawn.append(math.log10(over / under))
    ordered = sorted(drawn)
    tail = (1.0 - COVERAGE) / 2.0
    last = len(ordered) - 1
    return ordered[math.ceil((1.0 - tail) * last)] - ordered[math.floor(tail * last)]


def a_clustered_set(firearms: int = 10, scans: int = 4) -> ReferenceSet:
    """A set with several scans of each firearm, all of them scoring alike.

    This is the structure the resampling argument is actually about: many
    comparisons from one firearm, correlated with each other because they are
    that firearm. Every scan of a firearm carries its score, so removing the
    firearm removes all of them together.
    """
    names = [f"firearm-{index}" for index in range(firearms)]
    comparisons = [
        Comparison(
            score=OVERLAPPING if index % 2 == 0 else ONLY_SAME_SOURCE,
            source_a=name_,
            source_b=name_,
        )
        for index, name_ in enumerate(names)
        for _ in range(scans)
    ]
    for first in range(firearms):
        for second in range(first + 1, firearms):
            comparisons.append(
                Comparison(
                    score=OVERLAPPING if (first + second) % 2 == 0 else OVERLAPPING - 2,
                    source_a=names[first],
                    source_b=names[second],
                )
            )
    return ReferenceSet(name="clustered", cells=CELLS, comparisons=tuple(comparisons))


def test_resampling_pairs_would_give_a_narrower_interval_than_resampling_sources() -> None:
    # The choice this module makes, shown rather than asserted. Forty
    # same-source comparisons drawn from ten firearms are not forty independent
    # observations, and drawing them independently invents evidence that is not
    # there. Too narrow is worse than absent, because it reads as precision.
    #
    # The demonstration needs a set where a firearm's comparisons actually
    # correlate, which is why it does not use the fixture the rest of this file
    # runs on. That set has one comparison per firearm, so there is no
    # within-firearm structure for either scheme to get right or wrong, and
    # measured on it the two intervals come out 0.71 by source against 0.76 by
    # pair: the pair scheme is the wider one there, for the unrelated reason
    # that it also resamples how many comparisons land in each proposition. The
    # argument is about clustering and it is only visible where there is some.
    reference = a_clustered_set()

    by_source = log_ratio(
        reference, OVERLAPPING, propositions=PROPOSITIONS, resamples=RESAMPLES, seed=SEED
    ).unheld_width
    by_pair = a_pair_resampled_width(reference, OVERLAPPING, resamples=RESAMPLES, seed=SEED)

    assert by_pair < by_source, (by_pair, by_source)
