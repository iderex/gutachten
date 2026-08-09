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
    "VERSION",
    "CmcParameters",
    "Consensus",
    "ConsensusRule",
    "Score",
    "record",
    "score",
    "score_pair",
]

#: What a manifest names this stage by, and the version of what it computes. The
#: registration is part of the method rather than a stage beside it, so one
#: record carries the settings of both and a manifest cannot describe a count
#: without describing the search that produced the answers counted.
METHOD = "congruent-matching-cells"
VERSION = "1"


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
    """

    congruent: int
    eligible: int
    discarded: int
    consensus: Consensus
    failed_correlation: int
    failed_down: int
    failed_across: int
    failed_rotation: int


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

    congruent = 0
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
            congruent += 1

    return Score(
        congruent=congruent,
        eligible=len(matches),
        discarded=len(matches) - congruent,
        consensus=agreed,
        failed_correlation=failed["correlation"],
        failed_down=failed["down"],
        failed_across=failed["across"],
        failed_rotation=failed["rotation"],
    )


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
