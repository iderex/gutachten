"""The congruent matching cells rule, with its four identification parameters exposed.

The registration in `gutachten.compare.register` says where each cell of one
face matched on the other and at what orientation. This module turns that set of
answers into a count. A cell pair is congruent when its displacement down, its
displacement across and its orientation all sit within thresholds of the
consensus across cells, and when its correlation clears a threshold of its own.
Those four numbers are the identification parameters, and the score is how many
cells met all four.

## Why the four thresholds are parameters here and not constants

They are usually quoted as fixed values, in the same sentence as the method, as
though they were part of the algorithm rather than choices made on a dataset.
They are free in exactly the way the preprocessing parameters are free: each one
moves how many cells are counted, the count is the score, and the score is what a
decision is read off. A method whose thresholds are published as constants has
had a sensitivity analysis done for it once, somewhere, on data nobody can see.
So they are settings, they reach the manifest, and the sweep moves them with
everything else.

## The consensus is a rule, and the rule has parameters too

Where the cells agree is not a fact waiting to be read off; it is whatever the
chosen summary says it is. Two summaries are implemented.

`median` takes the middle value of each of the three quantities. It needs
nothing declared beyond itself and it moves smoothly as cells are added.

`histogram-mode` bins each quantity at a declared width and takes the centre of
the fullest bin. That is closer to what the published descriptions do, and the
bin width is the parameter easiest to overlook in the whole method: it decides
which cells fall into one bin and therefore which cells count as agreeing.

The bin edges are fixed to the origin rather than to the data, so bin `k` covers
`[k*w, (k+1)*w)` whatever the values happen to be, and the same set of cells bins
the same way in every run. The cost of that choice is stated rather than hidden:
a cluster straddling an edge is split between two bins, and the mode then lands
on a half of it. Recentring the grid on the data would move the edges instead,
which trades a reproducible answer for one that follows the sample.

A tie between two equally full bins goes to the lower one. Something has to
decide it, and a rule written down is better than whichever bin the iteration
reached first.

The consensus is taken over every cell the registration returned, not over the
cells that already cleared the correlation threshold. Those are different rules
and the second one is circular in a way worth avoiding: the cells that agree
best would then decide what agreement means. This module implements the first
and does not offer the second.

## What comes out beside the score

How many cells were eligible, how many were discarded, and what the consensus
values were. A score of eight means one thing out of forty cells and another out
of nine, and a count reported without its denominator cannot be read at all.

The four rejection counts are reported separately and they overlap: a cell
failing three conditions is counted in three of them. They do not sum to the
number discarded and they are not meant to. What they are for is saying which
condition is doing the work, which is the first thing a sweep over these four
numbers has to answer.

## The high variant, and how it differs

The variant reads the count of agreeing cells as a function of orientation
instead of picking one consensus orientation and judging every cell against it.
For each orientation the search visited, the cells that best match there are
judged against that orientation's own displacement consensus; the busiest
orientation sets a peak; every orientation within a declared number of cells of
that peak is kept; and the score is how many distinct cells agreed at any of
them.

What it is for is the case where the consensus is poorly determined. Two nearly
equal peaks make the single-consensus rule a coin toss between them, and the
cells at the losing orientation are discarded for having agreed with each other
at the wrong angle. The variant keeps both.

Its parameter set is not the original four plus extras. It reads three of them,
and it has no rotation threshold at all, because within one orientation every
cell carries that orientation and a threshold on it would compare each cell
against itself. What the original rule does with a threshold, this one does with
the shape of the distribution. That is a finding about the two rules rather than
a simplification of one.

Both rules run off one registration, which is what makes the comparison between
them a comparison of decision rules rather than of two pipelines. Where they
disagree, `disagreements` names the cells rather than the difference of two
totals: two rules reaching the same count out of different cells have not
agreed, and a subtraction reports that as agreement.

**Where this departs from the published description, and it is not settled here.**
The published variant discusses the high region as a contiguous run of
orientations around the peak. This implementation keeps every orientation within
tolerance of the peak whether or not it is contiguous with it. Nothing in this
tree can say which is right: no real data is readable here yet, the difference
only shows on a distribution with two separated peaks, and the reproduction
check against the published result is #77. The choice is written down so that a
disagreement about a number is traceable to it.

## What this does not decide

What a count of congruent cells means. There is no threshold here above which a
pair is called anything, and there will not be one in this module: the weight of
evidence milestone turns a score into a likelihood ratio, and #101 refuses the
words this module would otherwise be tempted into.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from gutachten.compare.register import (
    CellRegistration,
    Registration,
    RegistrationParameters,
    register,
)
from gutachten.compare.register import record as search_record
from gutachten.manifest import ComparisonRecord
from gutachten.surface import Surface

__all__ = [
    "METHOD",
    "METHOD_HIGH",
    "VERSION",
    "VERSION_HIGH",
    "CmcParameters",
    "Consensus",
    "ConsensusRule",
    "HighCmcParameters",
    "HighScore",
    "Score",
    "disagreements",
    "record",
    "record_high",
    "score",
    "score_high",
    "score_pair",
    "score_pair_high",
]

#: What a manifest names this stage by, and the version of what it computes. The
#: registration is part of the method rather than a stage beside it, so one
#: record carries the settings of both and a manifest cannot describe a count
#: without describing the search that produced the answers counted.
METHOD = "congruent-matching-cells"
VERSION = "1"

#: The high variant, under its own name rather than as a flag on the one above.
#: The two read different sets of parameters, so one record covering both would
#: have to carry settings the rule that produced it never looked at.
METHOD_HIGH = "high-congruent-matching-cells"
VERSION_HIGH = "1"


class ConsensusRule(Enum):
    """How the agreed displacement and orientation are arrived at."""

    #: The middle value of each quantity.
    MEDIAN = "median"
    #: The centre of the fullest bin, at a declared bin width.
    HISTOGRAM_MODE = "histogram-mode"


@dataclass(frozen=True)
class CmcParameters:
    """The four identification parameters and the consensus rule.

    ``down_threshold`` and ``across_threshold`` are distances in samples and
    ``rotation_threshold_deg`` is an angle, each the largest departure from the
    consensus a cell may have and still count. ``correlation_threshold`` is the
    correlation a cell has to clear.

    ``translation_bin`` and ``rotation_bin_deg`` are the histogram widths. They
    are read only by ``histogram-mode`` and have to be absent under ``median``,
    so a manifest cannot carry a bin width beside a rule that never looked at it.
    """

    down_threshold: float
    across_threshold: float
    rotation_threshold_deg: float
    correlation_threshold: float
    consensus: ConsensusRule
    translation_bin: float | None = None
    rotation_bin_deg: float | None = None


@dataclass(frozen=True)
class HighCmcParameters:
    """What the high variant is told, which is not the same set as the rule above.

    ``high_tolerance`` is how many cells below the busiest orientation an
    orientation may be and still be treated as part of the peak. It is the
    parameter the variant adds.

    There is no rotation threshold and no rotation bin here, and their absence is
    the shape of the variant rather than an omission. Within one orientation
    every cell carries that orientation, so an agreement threshold on rotation
    would compare each cell against itself and reject nothing. What the original
    rule does with a threshold, this one does with the shape of the count across
    orientations.
    """

    down_threshold: float
    across_threshold: float
    correlation_threshold: float
    consensus: ConsensusRule
    high_tolerance: int
    translation_bin: float | None = None


@dataclass(frozen=True)
class Consensus:
    """Where the cells agreed, and what said so."""

    down: float
    across: float
    rotation_deg: float
    rule: ConsensusRule


@dataclass(frozen=True)
class Score:
    """The count of congruent cells and everything needed to read it.

    ``eligible`` is how many cells the registration returned. A cell that could
    not be registered never reaches this rule and is not counted in either
    direction, which is why this number is reported rather than the grid.

    ``failed_correlation``, ``failed_down``, ``failed_across`` and
    ``failed_rotation`` are how many eligible cells each condition rejected on
    its own. They overlap and do not sum to ``discarded``.

    ``cells`` names which cells were counted, not only how many. Two rules that
    reach the same total out of different cells have not agreed, and a count on
    its own cannot say so.

    ``rule`` is which rule produced this, carried in the result rather than left
    to whatever called it, so anything reporting a score can say what it is a
    score of.
    """

    congruent: int
    eligible: int
    discarded: int
    consensus: Consensus
    failed_correlation: int
    failed_down: int
    failed_across: int
    failed_rotation: int
    cells: tuple[tuple[int, int], ...]
    rule: str = METHOD


def _checked(parameters: CmcParameters) -> tuple[float, float, float, float, float, float]:
    """Refuse a rule that could not count anything, or that states what it will not read."""
    thresholds: dict[str, float] = {}
    for name, value in (
        ("down_threshold", parameters.down_threshold),
        ("across_threshold", parameters.across_threshold),
        ("rotation_threshold_deg", parameters.rotation_threshold_deg),
        ("correlation_threshold", parameters.correlation_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number, got {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
        thresholds[name] = float(value)

    for name in ("down_threshold", "across_threshold", "rotation_threshold_deg"):
        if thresholds[name] < 0.0:
            raise ValueError(
                f"{name} is how far from the consensus a cell may sit and still count, so "
                f"it cannot be negative, got {thresholds[name]!r}."
            )

    correlation = thresholds["correlation_threshold"]
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(
            f"correlation_threshold is {correlation!r} and a correlation lies between -1 "
            "and 1. Above one no cell can ever be congruent and below minus one every "
            "cell clears it, and in both cases the score stops being about the surfaces."
        )

    if not isinstance(parameters.consensus, ConsensusRule):
        raise TypeError(
            f"the consensus rule must be a ConsensusRule, got {parameters.consensus!r}. "
            "Where the cells agree is a choice and there is no default for it."
        )

    bins = (
        ("translation_bin", parameters.translation_bin),
        ("rotation_bin_deg", parameters.rotation_bin_deg),
    )
    if parameters.consensus is ConsensusRule.HISTOGRAM_MODE:
        absent = [name for name, value in bins if value is None]
        if absent:
            raise ValueError(
                f"the {ConsensusRule.HISTOGRAM_MODE.value!r} rule needs {absent} and this "
                "record leaves them null. The bin width decides which cells fall into one "
                "bin and therefore which cells count as agreeing, so a rule that had to "
                "choose one for itself would be the parameter this project exists against."
            )
        widths: list[float] = []
        for width_name, width in bins:
            if isinstance(width, bool) or not isinstance(width, int | float):
                raise TypeError(f"{width_name} must be a number, got {width!r}")
            if not math.isfinite(float(width)) or float(width) <= 0.0:
                raise ValueError(
                    f"{width_name} must be a positive finite width, got {width!r}. A bin of "
                    "nothing puts every cell in a bin of its own and the fullest of them "
                    "holds one."
                )
            widths.append(float(width))
        translation_bin, rotation_bin = widths[0], widths[1]
    else:
        present = [name for name, value in bins if value is not None]
        if present:
            raise ValueError(
                f"the {parameters.consensus.value!r} rule does not read {present}, and "
                "this record states them. A manifest carrying both leaves a reader unable "
                "to say which of the two decided the consensus the cells were judged "
                "against."
            )
        # Recorded as absent rather than as a number the rule ignored.
        translation_bin, rotation_bin = math.nan, math.nan

    return (
        thresholds["down_threshold"],
        thresholds["across_threshold"],
        thresholds["rotation_threshold_deg"],
        correlation,
        translation_bin,
        rotation_bin,
    )


def record(search: RegistrationParameters, rule: CmcParameters) -> ComparisonRecord:
    """The manifest entry for a score made with these settings.

    Both halves in one record, because the count depends on the search as much as
    on the thresholds: the same four numbers over a coarser rotation step count
    different cells. A manifest holding the thresholds and not the search would
    describe a score nobody could reproduce.
    """
    down, across, rotation, correlation, translation_bin, rotation_bin = _checked(rule)
    # Taken off the search's own record rather than read out of the parameter
    # object here, so the two stages cannot come to disagree about what the
    # search recorded, and so the search's refusals run before a score does.
    searched = search_record(search).parameters
    return ComparisonRecord(
        method=METHOD,
        version=VERSION,
        parameters=tuple(
            sorted(
                (
                    *searched,
                    ("across_threshold", across),
                    ("consensus", rule.consensus.value),
                    ("correlation_threshold", correlation),
                    ("down_threshold", down),
                    ("rotation_bin_deg", None if math.isnan(rotation_bin) else rotation_bin),
                    ("rotation_threshold_deg", rotation),
                    ("translation_bin", None if math.isnan(translation_bin) else translation_bin),
                )
            )
        ),
    )


def _mode(values: np.ndarray, width: float) -> float:
    """The middle of the fullest bin's members, binning ``values`` at ``width``.

    The edges are fixed to the origin rather than to the data, so the same set of
    values bins the same way in every run and on every machine. What that costs
    is a cluster that straddles an edge, which is split between two bins.

    The answer is the median of what fell in the fullest bin and not the centre
    of the bin itself. A bin centre carries the arbitrary phase of the grid into
    the consensus: sixteen cells all reporting six degrees would agree exactly
    and be summarised as seven, and every cell would then be judged against half
    a bin width of something no cell said. The width decides which cells form the
    cluster, which is what it is for, and the cells decide where the cluster is.
    """
    indices = np.floor(values / width).astype(np.int64)
    occupied, counts = np.unique(indices, return_counts=True)
    # ``unique`` returns the bin indices in ascending order and ``argmax``
    # returns the first maximum, so the lowest of two equally full bins wins.
    fullest = int(occupied[int(np.argmax(counts))])
    return float(np.median(values[indices == fullest]))


def _consensus(
    matches: tuple[CellRegistration, ...],
    rule: ConsensusRule,
    translation_bin: float,
    rotation_bin: float,
) -> Consensus:
    down = np.array([match.down for match in matches], dtype=np.float64)
    across = np.array([match.across for match in matches], dtype=np.float64)
    rotation = np.array([match.rotation_deg for match in matches], dtype=np.float64)

    if rule is ConsensusRule.MEDIAN:
        return Consensus(
            down=float(np.median(down)),
            across=float(np.median(across)),
            rotation_deg=float(np.median(rotation)),
            rule=rule,
        )
    return Consensus(
        down=_mode(down, translation_bin),
        across=_mode(across, translation_bin),
        rotation_deg=_mode(rotation, rotation_bin),
        rule=rule,
    )


def score(registration: Registration, parameters: CmcParameters) -> Score:
    """How many of the registered cells agree, and everything needed to read that."""
    down_threshold, across_threshold, rotation_threshold, correlation_threshold, tb, rb = _checked(
        parameters
    )

    matches = registration.matches
    if not matches:
        raise ValueError(
            "the registration returned no cell, so there is nothing to take a consensus "
            "over. A score of zero congruent cells and a comparison that could not be "
            "made are different results and this rule will not report one as the other."
        )

    agreed = _consensus(matches, parameters.consensus, tb, rb)

    counted: list[tuple[int, int]] = []
    failed = {"correlation": 0, "down": 0, "across": 0, "rotation": 0}
    for match in matches:
        checks = {
            "correlation": match.correlation >= correlation_threshold,
            "down": abs(match.down - agreed.down) <= down_threshold,
            "across": abs(match.across - agreed.across) <= across_threshold,
            "rotation": abs(match.rotation_deg - agreed.rotation_deg) <= rotation_threshold,
        }
        for name, held in checks.items():
            if not held:
                failed[name] += 1
        if all(checks.values()):
            counted.append((match.row, match.column))

    congruent = len(counted)
    return Score(
        cells=tuple(counted),
        congruent=congruent,
        eligible=len(matches),
        discarded=len(matches) - congruent,
        consensus=agreed,
        failed_correlation=failed["correlation"],
        failed_down=failed["down"],
        failed_across=failed["across"],
        failed_rotation=failed["rotation"],
    )


def _checked_high(parameters: HighCmcParameters) -> tuple[float, float, float, float, int]:
    """Refuse a high variant whose settings do not describe one.

    The three it shares with the original rule are refused by the original
    rule's own check, handed a record carrying them, so one condition has one
    message wherever it fires. The rotation threshold that check also asks for
    is supplied here as zero and is not read by anything downstream, because
    this variant has no rotation threshold to refuse.
    """
    shared = CmcParameters(
        down_threshold=parameters.down_threshold,
        across_threshold=parameters.across_threshold,
        rotation_threshold_deg=0.0,
        correlation_threshold=parameters.correlation_threshold,
        consensus=parameters.consensus,
        translation_bin=parameters.translation_bin,
        # The original rule refuses a histogram record with no rotation bin, and
        # this variant has none to give, so the shared check is handed the
        # translation width for both. Nothing reads the rotation one here.
        rotation_bin_deg=parameters.translation_bin,
    )
    down, across, _, correlation, translation_bin, _ = _checked(shared)

    tolerance = parameters.high_tolerance
    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise TypeError(
            f"high_tolerance is a number of cells, so it must be a whole one, got {tolerance!r}."
        )
    if tolerance < 0:
        raise ValueError(
            f"high_tolerance is how far below the busiest orientation an orientation may "
            f"be and still count, so it cannot be negative, got {tolerance!r}."
        )
    return down, across, correlation, translation_bin, tolerance


@dataclass(frozen=True)
class HighScore:
    """The high variant's count, and the distribution it was read off.

    ``per_angle`` is how many cells agreed at each orientation on its own, in the
    order the orientations were searched. It is reported rather than reduced to
    its peak because the shape of it is the thing the variant is about: a
    matching pair puts a spike on one orientation and a non-matching pair does
    not, and a reader given only the peak cannot tell those apart.

    ``high_angles_deg`` is the orientations within ``high_tolerance`` cells of
    the peak, and ``cells`` names the cells counted at any of them.
    """

    congruent: int
    eligible: int
    discarded: int
    peak: int
    high_angles_deg: tuple[float, ...]
    per_angle: tuple[tuple[float, int], ...]
    cells: tuple[tuple[int, int], ...]
    rule: str = METHOD_HIGH


def _agreeing_at(
    matches: tuple[CellRegistration, ...],
    consensus_rule: ConsensusRule,
    translation_bin: float,
    down_threshold: float,
    across_threshold: float,
    correlation_threshold: float,
) -> tuple[tuple[int, int], ...]:
    """Which cells agree at one orientation, judged against that orientation's own consensus.

    The rotation is not compared with anything. Every cell here carries the
    orientation this call is about, so a comparison of it against a consensus
    over the same one value can only ever hold.
    """
    if not matches:
        return ()
    agreed = _consensus(matches, consensus_rule, translation_bin, translation_bin)
    return tuple(
        (match.row, match.column)
        for match in matches
        if match.correlation >= correlation_threshold
        and abs(match.down - agreed.down) <= down_threshold
        and abs(match.across - agreed.across) <= across_threshold
    )


def score_high(registration: Registration, parameters: HighCmcParameters) -> HighScore:
    """The high variant: how many cells agree anywhere near the busiest orientation.

    Read off the count of agreeing cells as a function of orientation rather than
    off one consensus. Where the consensus is poorly determined, the original
    rule's single orientation is a coin toss between two nearly equal peaks and
    the cells at the losing one are thrown away; here every orientation within
    ``high_tolerance`` cells of the busiest is kept and the cells counted at any
    of them are the score.
    """
    down, across, correlation, translation_bin, tolerance = _checked_high(parameters)

    if not registration.matches:
        raise ValueError(
            "the registration returned no cell, so there is nothing to take a consensus "
            "over. A score of zero congruent cells and a comparison that could not be "
            "made are different results and this rule will not report one as the other."
        )

    at_each: list[tuple[float, tuple[tuple[int, int], ...]]] = [
        (
            angle,
            _agreeing_at(matches, parameters.consensus, translation_bin, down, across, correlation),
        )
        for angle, matches in registration.by_angle
    ]
    per_angle = tuple((angle, len(agreeing)) for angle, agreeing in at_each)
    peak = max(count for _, count in per_angle)

    if peak == 0:
        # No orientation had a cell agreeing with any other, so there is no peak
        # to be near. Reporting every orientation as within tolerance of a peak
        # of nothing would name the whole search as the region of agreement.
        return HighScore(
            congruent=0,
            eligible=len(registration.matches),
            discarded=len(registration.matches),
            peak=0,
            high_angles_deg=(),
            per_angle=per_angle,
            cells=(),
        )

    high = [angle for angle, count in per_angle if count >= peak - tolerance]
    counted: set[tuple[int, int]] = set()
    for angle, agreeing in at_each:
        if angle in high:
            counted.update(agreeing)

    return HighScore(
        congruent=len(counted),
        eligible=len(registration.matches),
        discarded=len(registration.matches) - len(counted),
        peak=peak,
        high_angles_deg=tuple(high),
        per_angle=per_angle,
        cells=tuple(sorted(counted)),
    )


def record_high(search: RegistrationParameters, rule: HighCmcParameters) -> ComparisonRecord:
    """The manifest entry for a score made by the high variant.

    A separate record under a separate method name rather than a flag on the
    original one. The two rules read different sets of parameters, and a record
    that carried the union of them would say a rotation threshold decided a
    count that never read one.
    """
    down, across, correlation, translation_bin, tolerance = _checked_high(rule)
    searched = search_record(search).parameters
    return ComparisonRecord(
        method=METHOD_HIGH,
        version=VERSION_HIGH,
        parameters=tuple(
            sorted(
                (
                    *searched,
                    ("across_threshold", across),
                    ("consensus", rule.consensus.value),
                    ("correlation_threshold", correlation),
                    ("down_threshold", down),
                    ("high_tolerance", tolerance),
                    (
                        "translation_bin",
                        None if rule.translation_bin is None else translation_bin,
                    ),
                )
            )
        ),
    )


def disagreements(
    original: Score, high: HighScore
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """The cells one rule counted and the other did not, both ways round.

    Enumerated rather than reduced to a difference of two totals. Two rules can
    reach the same count out of different cells, which is a disagreement a
    subtraction reports as none.
    """
    first = set(original.cells)
    second = set(high.cells)
    return tuple(sorted(first - second)), tuple(sorted(second - first))


def score_pair(
    subject: Surface,
    reference: Surface,
    search: RegistrationParameters,
    rule: CmcParameters,
) -> tuple[Score, ComparisonRecord]:
    """Register the pair and count the cells that agree, with the record of both.

    The record comes back beside the score rather than being left for a caller to
    assemble, so a count cannot reach a manifest under settings that are not the
    ones it was produced with.
    """
    return score(register(subject, reference, search), rule), record(search, rule)


def score_pair_high(
    subject: Surface,
    reference: Surface,
    search: RegistrationParameters,
    rule: HighCmcParameters,
) -> tuple[HighScore, ComparisonRecord]:
    """The same, under the high variant."""
    return (
        score_high(register(subject, reference, search), rule),
        record_high(search, rule),
    )
