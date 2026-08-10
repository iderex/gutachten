"""What a score does under each proposition, estimated on a declared reference set.

A score based likelihood ratio is the ratio of two probabilities of one score:
its probability under the proposition that both surfaces came from one source,
and its probability under the proposition that they came from two. This module
estimates those two distributions and says what the set they were estimated on
can and cannot support. The ratio itself is #97.

## The propositions are named, not implied

``Proposition`` has two members and the reference set carries which comparisons
belong to each. A ratio computed against one population and read as though it
were computed against another is the most common way this quantity misleads, so
the population is a field of the estimate rather than a sentence in a document
somebody may not have read.

## The estimator, and its assumptions

The observed distribution over the integer support, with no parametric family
and no smoothing. The score is a count of congruent cells out of a known number
of cells, so the support is ``0`` to ``cells`` and is finite and known before any
data arrives. What the estimator does is count how often each value occurred and
divide.

Two assumptions, both of them load bearing.

The comparisons in the set are exchangeable within their proposition. Nothing
here weights one comparison above another, so a set assembled to over-represent
one firearm produces a distribution about that firearm.

A value never observed is estimated at zero rather than at something small. That
is the honest reading of the data and it is deliberately not smoothed, because
smoothing invents mass in exactly the tail this module exists to be careful
about: a fitted tail produces a very large ratio with no evidence behind it, and
the number that comes out looks the same as one with evidence behind it. The
zero is instead handled by the supportable range below, which refuses to report
a ratio the set cannot carry rather than reporting one from a probability nobody
measured.

No parametric family is used, so the check of a fit against the empirical
distribution is not made here. That is an absence rather than a check that
passed. What is reported instead is whether the set is sufficient for what is
being asked of it, in ``Diagnostics``, and the numbers are printed rather than
compared against a threshold this module chose.

## Sources govern, and pairs are the number that looks larger

Both counts are reported, and they are different numbers. A set of two hundred
comparisons drawn from eight firearms carries eight independent units, not two
hundred: the comparisons from one firearm share its manufacturing history, its
wear and its scan session. Every bound below is computed from the source count
for that reason, and the pair count is reported beside it so a reader can see the
gap rather than infer it.

## What the set can support

The smallest probability a set of ``k`` independent sources can tell apart from
zero is ``1 / k``, which is one source in ``k``. Nothing in such a set separates
a probability of ``1 / k`` from a probability of one in ten thousand, so a ratio
larger than ``k`` is a statement the set cannot make. The bound is therefore
``k`` in each direction, with ``k`` the source count under the proposition in the
denominator, and it is computed from the set rather than chosen.

A raw ratio beyond it is reported as beyond it. ``BoundedLogRatio`` carries the
bound, says that it was reached, and its wording is a statement about the limit
of the reference set rather than a number that reads as a measurement.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "BoundedLogRatio",
    "Comparison",
    "Counts",
    "Diagnostics",
    "Estimate",
    "Proposition",
    "ReferenceSet",
    "SupportableRange",
    "estimate",
    "hold_within",
    "supportable_range",
]


class Proposition(Enum):
    """The two propositions a score is given a probability under.

    Written as the source relationship rather than as an outcome, because that is
    what the reference set actually labels: two surfaces either came from one
    source or they did not, and that is a fact about how the set was assembled.
    """

    SAME_SOURCE = "same-source"
    DIFFERENT_SOURCE = "different-source"


@dataclass(frozen=True)
class Comparison:
    """One scored comparison, with the source each of its two surfaces came from.

    The sources are carried rather than a single boolean, because every bound in
    this module counts independent units and a boolean cannot say that forty
    comparisons came from four firearms.
    """

    score: int
    source_a: str
    source_b: str

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise TypeError(
                f"a score is a count of congruent cells and must be an integer, got "
                f"{self.score!r}. A float here is a score that went through an average "
                "somewhere, and the support this module counts over is the integers."
            )
        if self.score < 0:
            raise ValueError(f"a score cannot be negative, got {self.score}")
        for name, value in (("source_a", self.source_a), ("source_b", self.source_b)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} must name a source, got {value!r}. An unnamed source cannot "
                    "be counted as an independent unit and cannot be held out of a "
                    "reference set, and both are things this module does with it."
                )

    @property
    def proposition(self) -> Proposition:
        """Which proposition this comparison is an observation under."""
        if self.source_a == self.source_b:
            return Proposition.SAME_SOURCE
        return Proposition.DIFFERENT_SOURCE

    @property
    def sources(self) -> frozenset[str]:
        """The sources this comparison rests on, as a set of one or of two."""
        return frozenset({self.source_a, self.source_b})


@dataclass(frozen=True)
class Counts:
    """How much of a reference set stands behind one proposition.

    Both numbers are here because they are different and the smaller one governs.
    ``pairs`` is what a reader sees first and ``sources`` is what every bound is
    computed from.
    """

    pairs: int
    sources: int


@dataclass(frozen=True)
class Diagnostics:
    """Whether the set is sufficient for what is being asked of it.

    These are reported and not compared against a threshold. A threshold chosen
    here would be a judgement made once by whoever wrote this file and then
    applied silently to every set afterwards, and the judgement depends on what
    the estimate is for.

    ``largest_source_share`` is the one worth reading first: it is the fraction
    of the comparisons that involve the single source appearing in the most of
    them. A distribution whose share is near one is a distribution about one
    firearm carrying the name of a population.

    ``unobserved_support_share`` is the fraction of the possible scores that were
    never seen. It is high for any real set and that is not a defect; it is the
    reason the supportable range exists, and it is here so a reader meets the
    number rather than the reassurance.
    """

    observed_low: int
    observed_high: int
    distinct_scores: int
    largest_source_share: float
    unobserved_support_share: float


@dataclass(frozen=True)
class ReferenceSet:
    """The scored comparisons a distribution is estimated on, and what it is called.

    ``name`` is carried through into everything computed from the set. A ratio
    that can be printed without saying what it was estimated on is a ratio that
    will be quoted without it.

    ``cells`` is how many cells the score is a count out of, so the support is
    ``0`` to ``cells`` inclusive. It is declared rather than taken from the
    largest score seen: a set in which no comparison ever reached the maximum
    would otherwise report a support that its own data defined, and two sets
    would then have supports of different widths for no reason a reader could
    see.
    """

    name: str
    cells: int
    comparisons: tuple[Comparison, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"a reference set must be named, got {self.name!r}. The identity of the "
                "set travels with every number computed from it, and an unnamed set "
                "produces numbers nobody can check against anything."
            )
        if isinstance(self.cells, bool) or not isinstance(self.cells, int) or self.cells < 1:
            raise ValueError(
                f"the cell count must be a positive integer, got {self.cells!r}. It is the "
                "upper end of the support and a support of nothing is not a distribution."
            )
        for comparison in self.comparisons:
            if comparison.score > self.cells:
                raise ValueError(
                    f"a comparison scores {comparison.score} congruent cells out of "
                    f"{self.cells}. A score above the cell count is a score from a "
                    "different comparison stage, and averaging the two would produce a "
                    "distribution over a support neither of them has."
                )
        for proposition in Proposition:
            if not self.under(proposition):
                raise ValueError(
                    f"the reference set {self.name!r} holds no {proposition.value} "
                    "comparison. A ratio needs a probability under both propositions, and "
                    "a set supplying one of them would give the other whatever the code "
                    "happened to do with an empty sequence."
                )

    def under(self, proposition: Proposition) -> tuple[Comparison, ...]:
        """The comparisons that are observations under ``proposition``."""
        return tuple(item for item in self.comparisons if item.proposition is proposition)

    def counts(self, proposition: Proposition) -> Counts:
        """The pair count and the independent source count under ``proposition``."""
        selected = self.under(proposition)
        sources: set[str] = set()
        for comparison in selected:
            sources |= comparison.sources
        return Counts(pairs=len(selected), sources=len(sources))


@dataclass(frozen=True)
class Estimate:
    """One estimated distribution, with what it was estimated on and how well.

    ``probability`` is indexed by score: entry ``i`` is the estimated probability
    of a score of ``i``, and the tuple runs the whole declared support so that
    two estimates from two sets are the same length and can be read side by side.
    """

    reference_set: str
    proposition: Proposition
    probability: tuple[float, ...]
    counts: Counts
    diagnostics: Diagnostics

    def probability_of(self, score: int) -> float:
        """The estimated probability of ``score``, or zero where it was never seen."""
        if isinstance(score, bool) or not isinstance(score, int):
            raise TypeError(f"a score is an integer, got {score!r}")
        if not 0 <= score < len(self.probability):
            raise ValueError(
                f"a score of {score} is outside the support 0 to "
                f"{len(self.probability) - 1}, which is the cell count the reference set "
                f"{self.reference_set!r} declared. A probability read off the end of a "
                "distribution is a probability of nothing."
            )
        return self.probability[score]


def estimate(reference: ReferenceSet, proposition: Proposition) -> Estimate:
    """Estimate the score distribution under ``proposition`` on ``reference``.

    The observed distribution over the declared support, unsmoothed. What the
    data did not show is estimated at zero, and what to do about that is
    :func:`supportable_range` rather than a prior invented here.
    """
    selected = reference.under(proposition)
    counted = Counter(item.score for item in selected)
    total = len(selected)
    probability = tuple(counted.get(score, 0) / total for score in range(reference.cells + 1))

    appearances: Counter[str] = Counter()
    for comparison in selected:
        for source in comparison.sources:
            appearances[source] += 1
    observed = sorted(counted)

    return Estimate(
        reference_set=reference.name,
        proposition=proposition,
        probability=probability,
        counts=reference.counts(proposition),
        diagnostics=Diagnostics(
            observed_low=observed[0],
            observed_high=observed[-1],
            distinct_scores=len(observed),
            largest_source_share=max(appearances.values()) / total,
            unobserved_support_share=1.0 - len(observed) / (reference.cells + 1),
        ),
    )


@dataclass(frozen=True)
class SupportableRange:
    """How far from zero a log ratio may go before the set stops carrying it.

    Base ten throughout, because that is the scale this quantity is read on and a
    bound quoted in one base and read in another is off by a factor of two and a
    bit in the direction that flatters.

    ``most_positive`` is set by the different-source source count and
    ``most_negative`` by the same-source one, because a ratio above one is a
    statement about how rare the score is under the different-source proposition
    and it is that proposition's data that has to carry it.
    """

    reference_set: str
    most_negative: float
    most_positive: float
    same_source: Counts
    different_source: Counts


def supportable_range(reference: ReferenceSet) -> SupportableRange:
    """What ``reference`` can support, computed from how many sources it holds.

    ``k`` independent sources cannot tell a probability of ``1 / k`` from a
    smaller one, because one source in ``k`` is the finest the set divides. So a
    ratio beyond ``k`` rests on a probability the set never measured, and
    ``log10(k)`` is where the reporting stops.
    """
    same = reference.counts(Proposition.SAME_SOURCE)
    different = reference.counts(Proposition.DIFFERENT_SOURCE)
    return SupportableRange(
        reference_set=reference.name,
        most_negative=-math.log10(same.sources),
        most_positive=math.log10(different.sources),
        same_source=same,
        different_source=different,
    )


@dataclass(frozen=True)
class BoundedLogRatio:
    """A base ten log ratio held inside what the reference set supports.

    ``value`` is the number to report. ``at_the_bound`` says whether the raw
    result reached or passed the limit, in which case ``value`` is the limit and
    is a statement about the reference set rather than about the comparison.
    """

    value: float
    at_the_bound: bool
    reference_set: str
    sources_behind_the_bound: int

    def statement(self) -> str:
        """The line to print, which cannot read as a measurement when it is not one.

        At the bound the wording carries the direction, the limit and how many
        sources set it, so a reader meeting the number alone still meets what
        limits it. It is written without either of the marks in
        ``gutachten.conclusions``: neither applies to a statement about one
        comparison, and reaching for one of them where it does not apply is the
        switch that design is avoiding.
        """
        if not self.at_the_bound:
            return f"log10 ratio {self.value:.2f} on reference set {self.reference_set!r}"
        direction = "above" if self.value > 0 else "below"
        return (
            f"log10 ratio {direction} {self.value:+.2f}, which is as far as reference set "
            f"{self.reference_set!r} reaches on {self.sources_behind_the_bound} sources"
        )


def hold_within(bounds: SupportableRange, log_ratio: float) -> BoundedLogRatio:
    """Report ``log_ratio`` where the set carries it, and the bound where it does not.

    ``log_ratio`` may be an infinity, which is what an unobserved score produces
    and is the case this exists for. It is held at the bound like any other value
    beyond it rather than special-cased, because a probability of zero and a
    probability too small for the set to have measured are the same statement
    about the set.
    """
    if math.isnan(log_ratio):
        raise ValueError(
            "a log ratio of not-a-number is a computation that failed rather than a "
            "result beyond the reference set, and holding it at a bound would report "
            "the failure as a measurement."
        )
    if log_ratio >= bounds.most_positive:
        return BoundedLogRatio(
            value=bounds.most_positive,
            at_the_bound=True,
            reference_set=bounds.reference_set,
            sources_behind_the_bound=bounds.different_source.sources,
        )
    if log_ratio <= bounds.most_negative:
        return BoundedLogRatio(
            value=bounds.most_negative,
            at_the_bound=True,
            reference_set=bounds.reference_set,
            sources_behind_the_bound=bounds.same_source.sources,
        )
    return BoundedLogRatio(
        value=log_ratio,
        at_the_bound=False,
        reference_set=bounds.reference_set,
        sources_behind_the_bound=0,
    )
