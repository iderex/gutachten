"""Rejecting outliers, with the criterion, its threshold and its reach declared.

An optical measurement of a cartridge case carries samples that are not surface:
a dropout read back as an extreme height, a speck of dust, a specular return off
a steep wall. Every published chain removes them, usually with a threshold on a
robust measure of spread, and that threshold is normally a round number nobody
sourced. Tight, it removes real surface; loose, it leaves spikes that the later
filtering spreads over their neighbourhoods.

So the measure of spread, the threshold and whether the test is made against the
whole surface or against a neighbourhood are all parameters, they are all in the
manifest, and the sweep can move all three.

## Detecting a hole and filling one are two decisions

A rejected sample becomes missing. It is not replaced, not interpolated, and not
smoothed over. If interpolation is wanted it is a separate transform with its
own parameters, because a step that both found a hole and filled it records one
decision where two were made, and the sweep can then no longer ask what the
filling was worth on its own.

## How much surface this discarded is recorded

The count of rejected samples and the count of measured samples it was taken
from both go into the provenance, as outcomes rather than as parameters: they
are what the run found, not what it was told. A sensitivity report saying a
threshold moved the score, without saying that the threshold removed a fifth of
the surface on the way, is missing the more interesting half of its own result.

## The two criteria, and why the weaker one is offered

``Dispersion.MEDIAN_ABSOLUTE_DEVIATION`` is the robust measure and is what the
published chains use. ``Dispersion.STANDARD_DEVIATION`` is the one that is
reached for first and is inflated by the very spikes it is being used to find,
so a few large outliers raise the threshold until they pass it. It is offered
because a sweep that cannot run the weaker rule cannot report what the stronger
one is worth, and this project's whole argument is about what these choices are
worth.

## Global or in a neighbourhood

``neighbourhood`` is a radius as a length in the surface's own unit, or
``None`` for a test against the whole surface. A length rather than a number of
samples, for the reason the edge trim states: a neighbourhood in samples is a
different physical region on every instrument, and the sensitivity study would
then be sweeping the instrument along with the step.

The neighbourhood is a rectangle rather than a disc, because the two axes have
their own spacings and a disc in samples is an ellipse on the surface.

A window with no measured sample in it, and a surface whose spread is zero, both
leave the centre of the window alone. Not by a special case: the comparison is
``deviation > threshold * spread``, and a comparison against not-a-number is
false, while a spread of zero rejects only what differs from the centre at all.
Both are stated here because the second is a sharp rule that will surprise
somebody, and neither is a branch.

## The cost, measured

The neighbourhood test runs a median over every window, which is two passes of
``scipy.ndimage.generic_filter`` and is the expensive route rather than a
convolution. On this machine, with a nine by nine window:

    $ uv run python -c "import time, numpy as np; from scipy import ndimage; \\
      a = np.random.default_rng(0).normal(size=(256, 256)); t = time.perf_counter(); \\
      ndimage.generic_filter(a, np.nanmedian, footprint=np.ones((9, 9), dtype=bool)); \\
      print(time.perf_counter() - t)"
    1.48

So a scan of a few hundred thousand samples costs seconds rather than
milliseconds in this step. That is a number a sweep design has to be costed
against, and it is here rather than discovered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage

from gutachten.surface import Surface
from gutachten.transforms.base import Parameters, SurfaceProperty, record_for
from gutachten.transforms.registry import REGISTRY

__all__ = ["Dispersion", "OutlierParameters", "RejectOutliers"]


class Dispersion(Enum):
    """How the spread the threshold is counted in is measured."""

    #: The median of the absolute deviations from the median. Robust: a spike
    #: barely moves it, which is what lets it find one.
    MEDIAN_ABSOLUTE_DEVIATION = "median-absolute-deviation"
    #: The ordinary standard deviation, which the spikes inflate.
    STANDARD_DEVIATION = "standard-deviation"


@dataclass(frozen=True)
class OutlierParameters:
    """What counts as an outlier here.

    ``criterion`` is the name of a :class:`Dispersion`. ``threshold`` is how
    many of those spreads a sample may sit from the centre before it is
    rejected. ``neighbourhood`` is the radius of the region the centre and the
    spread are taken over, as a length in the surface's own unit, or ``None``
    for the whole surface.
    """

    criterion: str
    threshold: float
    neighbourhood: float | None


class RejectOutliers:
    """Mark samples that sit too far from their surroundings as missing."""

    identifier = "reject-outliers"
    version = "1"
    parameters_type = OutlierParameters
    produces = frozenset({SurfaceProperty.OUTLIERS_MARKED})
    requires = frozenset[SurfaceProperty]()
    # A bandpass spreads a spike over the width of its kernel, so a rejection
    # made afterwards finds a smear rather than a sample and takes the real
    # surface around it with the artefact.
    refuses = frozenset({SurfaceProperty.FILTERED})

    def apply(self, surface: Surface, parameters: Parameters) -> Surface:
        # The pipeline checks the record type before the chain starts and
        # `record_for` checks it again on the way out. This one narrows the type
        # for the checker and refuses before a single field has been read.
        if not isinstance(parameters, OutlierParameters):
            raise TypeError(
                f"{self.identifier!r} was handed {type(parameters).__name__} rather than "
                "OutlierParameters, so nothing here has read a field off it"
            )
        criterion = _criterion(parameters)
        threshold = _threshold(parameters)

        observed = ~surface.missing
        measured = int(np.count_nonzero(observed))
        if measured == 0:
            raise ValueError(
                "every sample of this surface is missing, so there is nothing to reject "
                "and nothing to measure a spread over. A step that ran over an empty "
                "surface would record itself as having rejected none."
            )

        if parameters.neighbourhood is None:
            centre, spread = _over_the_whole_surface(surface.heights, criterion)
        else:
            centre, spread = _over_a_neighbourhood(surface, parameters.neighbourhood, criterion)

        # A comparison against not-a-number is false, so a window that held no
        # measured sample leaves its centre alone without a branch saying so.
        rejected = observed & (np.abs(surface.heights - centre) > threshold * spread)
        removed = int(np.count_nonzero(rejected))
        if removed == measured:
            raise ValueError(
                f"a threshold of {threshold} {criterion.value}s rejects all {measured} "
                "measured samples of this surface. Every number after this step would be "
                "taken over an empty array, which most reductions answer with "
                "not-a-number rather than with an error."
            )

        heights = np.array(surface.heights, dtype=np.float64, copy=True)
        heights[rejected] = np.nan
        record = record_for(self, parameters).with_outcomes(
            rejected_samples=removed, measured_samples=measured
        )
        return surface.with_transform(record, heights)


def _criterion(parameters: OutlierParameters) -> Dispersion:
    """The criterion named, refusing a name that is not one.

    A misspelling would otherwise fall through to whichever branch was written
    as the default, and the manifest would record a word that decided nothing.
    """
    try:
        return Dispersion(parameters.criterion)
    except ValueError:
        known = ", ".join(sorted(item.value for item in Dispersion))
        raise ValueError(
            f"{parameters.criterion!r} is not a criterion this step knows; it takes one of {known}"
        ) from None


def _threshold(parameters: OutlierParameters) -> float:
    """The threshold, refusing one that cannot mean a number of spreads."""
    threshold = parameters.threshold
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(f"the threshold must be a number of spreads, got {threshold!r}")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            f"the threshold must be a positive finite number of spreads, got {threshold!r}. "
            "A threshold of nothing rejects every sample that is not exactly at the centre, "
            "and a threshold that is not a number rejects none while recording that a "
            "rejection ran."
        )
    return float(threshold)


def _over_the_whole_surface(
    heights: np.ndarray, criterion: Dispersion
) -> tuple[np.ndarray, np.ndarray]:
    """The centre and the spread of every measured sample, as two scalars.

    Returned as arrays so the caller compares against them the same way it
    compares against the per-sample fields the neighbourhood route produces.
    """
    measured = heights[np.isfinite(heights)]
    return np.asarray(_centre(measured, criterion)), np.asarray(_spread(measured, criterion))


def _over_a_neighbourhood(
    surface: Surface, radius: float, criterion: Dispersion
) -> tuple[np.ndarray, np.ndarray]:
    """The centre and the spread of each sample's own surroundings.

    Two passes over the surface rather than one. The spread is measured about
    the centre of the same window, so the centre has to exist before the spread
    can be taken, and no single pass produces both.
    """
    footprint = _footprint(radius, surface)
    centre = _windowed(surface.heights, footprint, lambda values: _centre(values, criterion))
    spread = _windowed(surface.heights, footprint, lambda values: _spread(values, criterion))
    return centre, spread


def _footprint(radius: float, surface: Surface) -> np.ndarray:
    """A rectangle of ``radius`` either side of a sample, in whole samples.

    Refuses a radius that reaches no neighbour along an axis. A window one
    sample wide in a direction takes its centre and its spread from a line, and
    a step asked for with a neighbourhood it cannot see is a step that records
    itself as having run against surroundings it never had.
    """
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"the neighbourhood radius must be a positive finite length, got {radius!r}. "
            "Pass None for a test against the whole surface, which is a different "
            "measurement and is recorded as one."
        )
    down = round(radius / surface.spacing_y)
    across = round(radius / surface.spacing_x)
    if down < 1 or across < 1:
        raise ValueError(
            f"a radius of {radius} {surface.unit.value} rounds to {down} rows and "
            f"{across} columns at this surface's spacings, so the neighbourhood would "
            "reach no neighbour along at least one axis and every sample would be "
            "compared against a line through itself."
        )
    shape = (down * 2 + 1, across * 2 + 1)  # structural: a radius either side of the sample
    return np.ones(shape, dtype=bool)


def _windowed(
    heights: np.ndarray, footprint: np.ndarray, of: Callable[[np.ndarray], float]
) -> np.ndarray:
    """``of`` applied to the measured samples inside each window.

    Outside the array is not-a-number rather than a reflection or an edge value.
    A reflected sample is a measurement that was never made, and a spike near
    the boundary would be compared against its own mirror image.
    """
    filtered = ndimage.generic_filter(
        heights,
        lambda values: of(values[np.isfinite(values)]),
        footprint=footprint,
        mode="constant",
        cval=np.nan,
    )
    return np.asarray(filtered)


def _centre(measured: np.ndarray, criterion: Dispersion) -> float:
    """Where the criterion says the middle of ``measured`` is.

    The median for the robust criterion and the mean for the other, so each
    threshold is counted from the centre its own spread is measured about.
    Mixing them would put a number of median absolute deviations from the mean
    into the manifest, which is not a quantity.
    """
    if measured.size == 0:
        # numpy answers an empty reduction with not-a-number and a warning, and
        # this suite turns warnings into errors. A window with nothing measured
        # in it is the ordinary case at the edge of a masked region, not a
        # failure.
        return float("nan")
    if criterion is Dispersion.MEDIAN_ABSOLUTE_DEVIATION:
        return float(np.median(measured))
    return float(np.mean(measured))


def _spread(measured: np.ndarray, criterion: Dispersion) -> float:
    """How far ``measured`` is spread out, in the criterion's own terms."""
    if measured.size == 0:
        return float("nan")
    if criterion is Dispersion.MEDIAN_ABSOLUTE_DEVIATION:
        return float(np.median(np.abs(measured - np.median(measured))))
    return float(np.std(measured))


REGISTRY.register(RejectOutliers())
