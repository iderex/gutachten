"""The score based likelihood ratio, with the interval its reference set supports.

Given a score, the two distributions estimated in
:mod:`gutachten.stats.distributions` and the reference set they were estimated
on, this produces the base ten log ratio and a bootstrap interval on it. The
propositions and the identity of the reference set travel inside the result, and
there is no route to the number that leaves them behind.

## Why the result type is shaped against being quoted

The way this quantity gets misused is not by being computed wrongly. It is by
being lifted out of its context and quoted alone, and a ratio quoted without its
propositions is a number about nothing: the same score gives a different ratio
against a different relevant population, and the reader cannot tell which one
they are holding.

So ``LogRatio`` refuses a format specification. ``f"{result}"`` prints the whole
statement and ``f"{result:.2f}"`` raises, because a format specification is
precisely how somebody prints the number on its own. That is a small barrier
rather than a guarantee - the fields are still reachable by anybody who wants
them - and it is aimed at the accident rather than at the determined.

## The bootstrap resamples sources, not pairs

This is the most consequential choice here and it is decided in #93. The pairs
drawn from one firearm are not independent of each other: they share its
manufacturing history, its wear and its scan session. Resampling pairs treats
them as though they were, and produces an interval that is too narrow by a wide
margin, which is worse than reporting no interval at all because it reads as
precision.

So a resample draws sources with replacement and takes the comparisons those
sources support. A same-source comparison enters with the multiplicity of its
one source; a different-source comparison enters with the product of its two,
which is the ordinary cluster bootstrap applied to a structure whose units are
the two ends of each comparison. A draw that ends up with no comparison under
one of the propositions says nothing about a ratio and is discarded, and how
many were discarded is reported rather than absorbed.

## The reference set may not touch the pair being evaluated

A ratio whose distributions were fitted on the case in front of the reader is
circular. It is a well known failure and it is easy to reintroduce by accident
when the available data is small, so :func:`log_ratio` takes the pair being
evaluated and refuses a reference set that touches either of its sources.

The refusal is by source and not by pair, and that is the whole point of it. A
reference set containing the evaluated pair itself is the version anybody would
catch. The version that actually happens is a reference set containing a
*different* pair from the same firearm: it looks disjoint at the level of pairs,
it passes any check written over pairs, and the firearm's own marks are then on
both sides of the comparison.

## Two intervals, and why both are here

Every endpoint is held inside what the reference set supports, by
:func:`gutachten.stats.distributions.hold_within`, and that held interval is the
one to report. It is also not the one that shows sampling uncertainty moving,
because the bound is itself a function of how many sources the set has: take
sources away and the bound tightens, so the held interval can narrow while the
estimate underneath it is getting worse.

``unheld_width`` is therefore carried beside it. That is the width before the
bound is applied, it is what widens when the reference set is subsampled, and it
is named for what it is so that nobody reports it as the interval.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from gutachten.stats.distributions import (
    BoundedLogRatio,
    Counts,
    Proposition,
    ReferenceSet,
    hold_within,
    supportable_range,
)

__all__ = [
    "EvaluatedPair",
    "LogRatio",
    "NotDisjoint",
    "Propositions",
    "check_disjoint",
    "log_ratio",
]

#: The interval reported. Written here rather than taken as an argument, because
#: an interval whose coverage is chosen per call is an interval two results
#: cannot be compared across, and the number a reader assumes is this one.
COVERAGE = 0.95


@dataclass(frozen=True)
class Propositions:
    """What the ratio is a ratio between, and over which population.

    All three are required and none may be blank. A ratio computed against one
    population and read as though it were computed against another is the most
    common way this quantity misleads, so the population is a field of the
    result rather than a sentence in a document somebody may not have read.
    """

    numerator: str
    denominator: str
    population: str

    def __post_init__(self) -> None:
        for name in ("numerator", "denominator", "population"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"the {name} proposition must be stated, got {value!r}. A ratio whose "
                    "propositions are not written down is a number a reader cannot place, "
                    "and leaving one blank is how it stops being written down."
                )
        if self.numerator.strip() == self.denominator.strip():
            raise ValueError(
                "the two propositions are the same sentence. A ratio between one "
                "proposition and itself is one by construction, and a result reading 0.00 "
                "for that reason is indistinguishable from a result that measured nothing."
            )


class NotDisjoint(Exception):
    """A reference set touches a source of the pair being evaluated."""


@dataclass(frozen=True)
class EvaluatedPair:
    """The comparison in front of the reader, by the sources of its two surfaces.

    Required wherever a ratio is computed. A reference set is only disjoint from
    something, and a function that could be called without naming that something
    would be a function that checks disjointness against nothing.
    """

    source_a: str
    source_b: str

    def __post_init__(self) -> None:
        for name in ("source_a", "source_b"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"the evaluated pair's {name} must name a source, got {value!r}. An "
                    "unnamed source matches nothing in a reference set, so the "
                    "disjointness check would pass by having nothing to compare."
                )

    @property
    def sources(self) -> frozenset[str]:
        return frozenset({self.source_a, self.source_b})


def check_disjoint(reference: ReferenceSet, evaluated: EvaluatedPair) -> None:
    """Refuse a reference set that touches either source of ``evaluated``.

    By source rather than by pair. The reference set is what the two
    distributions were estimated on, so a firearm appearing in both the set and
    the comparison puts its own marks on both sides of the ratio, whether or not
    the particular pair is the same one.
    """
    touching = sorted(
        {
            source
            for comparison in reference.comparisons
            for source in comparison.sources & evaluated.sources
        }
    )
    if touching:
        raise NotDisjoint(
            f"reference set {reference.name!r} contains comparisons from {touching}, which "
            f"is where the pair being evaluated came from. A ratio whose distributions "
            "were estimated on the case in front of the reader is circular, and this is "
            "refused by source rather than by pair because the reference set does not have "
            "to contain the same pair to be the same firearm."
        )


@dataclass(frozen=True)
class LogRatio:
    """A base ten log ratio, its interval, and everything needed to read either.

    ``point``, ``low`` and ``high`` are each held inside what the reference set
    supports. ``unheld_width`` is the width of the interval before the bound was
    applied and is a diagnostic rather than something to report.
    """

    score: int
    reference_set: str
    propositions: Propositions
    evaluated: EvaluatedPair
    point: BoundedLogRatio
    low: BoundedLogRatio
    high: BoundedLogRatio
    unheld_width: float
    coverage: float
    resamples: int
    resamples_discarded: int
    same_source: Counts
    different_source: Counts

    def __str__(self) -> str:
        """The whole statement. There is no shorter one.

        Every part a reader needs to place the number is here, in one string,
        because the alternative is a caller assembling it from fields and
        leaving out whichever one is inconvenient.
        """
        return (
            f"log10 ratio {self.point.value:.2f}, "
            f"{self.coverage:.0%} interval {self.low.value:.2f} to {self.high.value:.2f}, "
            f"for a score of {self.score} congruent cells. "
            f"Numerator: {self.propositions.numerator}. "
            f"Denominator: {self.propositions.denominator}. "
            f"Relevant population: {self.propositions.population}. "
            f"Evaluated pair: {self.evaluated.source_a} against "
            f"{self.evaluated.source_b}, neither of them in the reference set. "
            f"Reference set {self.reference_set!r}: "
            f"{self.same_source.pairs} same-source comparisons over "
            f"{self.same_source.sources} sources, "
            f"{self.different_source.pairs} different-source comparisons over "
            f"{self.different_source.sources} sources."
        )

    def __format__(self, specification: str) -> str:
        """Refuse a format specification, which is how the number gets quoted alone.

        ``f"{result}"`` gives the whole statement. ``f"{result:.2f}"`` is
        somebody printing the ratio without its propositions and without the set
        it was estimated on, and it raises instead of producing that line.
        """
        if specification:
            raise ValueError(
                f"a log ratio was formatted with {specification!r}, which prints the "
                "number and drops the propositions and the reference set it means "
                "nothing without. Format it with no specification, which prints all of "
                "them, or read the field you need and carry the rest with it."
            )
        return str(self)


def _sources(reference: ReferenceSet) -> tuple[str, ...]:
    """Every source in the set, sorted, so a draw is reproducible from a seed."""
    found: set[str] = set()
    for comparison in reference.comparisons:
        found |= comparison.sources
    return tuple(sorted(found))


def _weighted_log_ratio(
    reference: ReferenceSet, score: int, multiplicity: Counter[str] | None
) -> float | None:
    """The raw log ratio at ``score`` under a source multiplicity, or nothing.

    ``multiplicity`` of ``None`` is the set as it stands, which is every source
    once. Nothing is returned where a draw left a proposition with no comparison
    at all, or where the score was seen under neither: both are draws that say
    nothing about a ratio, and averaging them in as a zero would be inventing an
    observation.
    """
    totals = {proposition: 0 for proposition in Proposition}
    at_score = {proposition: 0 for proposition in Proposition}
    for comparison in reference.comparisons:
        if multiplicity is None:
            weight = 1
        elif comparison.source_a == comparison.source_b:
            weight = multiplicity[comparison.source_a]
        else:
            weight = multiplicity[comparison.source_a] * multiplicity[comparison.source_b]
        if weight == 0:
            continue
        totals[comparison.proposition] += weight
        if comparison.score == score:
            at_score[comparison.proposition] += weight

    if not all(totals.values()):
        return None

    over = at_score[Proposition.SAME_SOURCE] / totals[Proposition.SAME_SOURCE]
    under = at_score[Proposition.DIFFERENT_SOURCE] / totals[Proposition.DIFFERENT_SOURCE]
    if over == 0.0 and under == 0.0:
        return None
    if under == 0.0:
        return math.inf
    if over == 0.0:
        return -math.inf
    return math.log10(over / under)


def log_ratio(
    reference: ReferenceSet,
    score: int,
    *,
    propositions: Propositions,
    evaluated: EvaluatedPair,
    resamples: int,
    seed: int,
) -> LogRatio:
    """The log ratio at ``score`` on ``reference``, with a bootstrap interval.

    ``resamples`` and ``seed`` are both required and neither has a default. A
    bootstrap with an unrecorded seed is an interval nobody else gets, and a
    resample count chosen inside this function would be a number decided once
    and then invisible in every result that carried it.
    """
    if isinstance(score, bool) or not isinstance(score, int):
        raise TypeError(f"a score is a count of congruent cells and is an integer, got {score!r}")
    check_disjoint(reference, evaluated)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 2:
        raise ValueError(
            f"a bootstrap needs at least two resamples, got {resamples!r}. One resample "
            "produces an interval of zero width, which reads as certainty."
        )

    point = _weighted_log_ratio(reference, score, None)
    if point is None:
        raise ValueError(
            f"a score of {score} was seen under neither proposition in reference set "
            f"{reference.name!r}. A ratio there would be zero divided by zero, and "
            "reporting a bound for it would say the set had something to say about a "
            "score it never saw."
        )

    sources = _sources(reference)
    generator = np.random.default_rng(seed)
    drawn: list[float] = []
    discarded = 0
    for _ in range(resamples):
        indices = generator.integers(0, len(sources), size=len(sources))
        multiplicity = Counter(sources[int(index)] for index in indices)
        resampled = _weighted_log_ratio(reference, score, multiplicity)
        if resampled is None:
            discarded += 1
            continue
        drawn.append(resampled)

    if not drawn:
        raise ValueError(
            f"every one of {resamples} resamples of reference set {reference.name!r} left "
            "a proposition with nothing in it, so there is no interval. The set is too "
            "small in sources for a bootstrap over sources, which is the honest reading "
            "rather than a reason to resample pairs instead."
        )

    # Order statistics rather than an interpolated quantile. A resample of a
    # score seen under one proposition and not the other is a legitimate
    # infinity, and interpolating between an infinity and a finite value is a
    # subtraction of infinities, which produces not-a-number and would put a
    # missing endpoint where a wide one belongs.
    ordered = sorted(drawn)
    tail = (1.0 - COVERAGE) / 2.0
    last = len(ordered) - 1
    low_raw = ordered[math.floor(tail * last)]
    high_raw = ordered[math.ceil((1.0 - tail) * last)]

    bounds = supportable_range(reference)
    return LogRatio(
        score=score,
        reference_set=reference.name,
        propositions=propositions,
        evaluated=evaluated,
        point=hold_within(bounds, point),
        low=hold_within(bounds, low_raw),
        high=hold_within(bounds, high_raw),
        unheld_width=(
            high_raw - low_raw if math.isfinite(low_raw) and math.isfinite(high_raw) else math.inf
        ),
        coverage=COVERAGE,
        resamples=resamples,
        resamples_discarded=discarded,
        same_source=reference.counts(Proposition.SAME_SOURCE),
        different_source=reference.counts(Proposition.DIFFERENT_SOURCE),
    )
