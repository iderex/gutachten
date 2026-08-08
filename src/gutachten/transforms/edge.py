"""Cutting away the edge, with how much declared rather than set by eye.

Optical measurement of a cartridge case falls off near the boundary of the
measured region and near steep walls, where the instrument's numerical aperture
stops returning valid data. Every published chain cuts that region away, how
much is cut is normally set by eye, and it changes directly how much of the
breech face survives into the comparison. So it is a parameter, it is in the
manifest, and the sweep can move it.

## The two criteria, and why both

``EdgeCriterion.FRAME`` cuts a band of the declared width in from the boundary
of the array. That is the case where the instrument simply stopped measuring at
the edge of its field.

``EdgeCriterion.GROW`` widens the region that is already missing by the declared
width. That is the case near a steep wall, where the samples that came back are
not trustworthy for some distance around the ones that did not. A frame cut
would not reach an invalid patch in the middle of the surface, and growing alone
would not touch an edge that came back with plausible looking values.

They are separate parameters and not a single cleverer rule, because a sweep
that moves one number and watches a score cannot interpret a number that means
two different things depending on where on the surface it is applied.

## Marked missing, not deleted

The array keeps its dimensions and the removed samples become not-a-number. A
step that deleted rows would move every coordinate downstream of it, so a
registration found on a trimmed surface would not be a registration on the one
the operator is looking at. It also means a later step can still tell where the
surface was, which the masking steps need.

## The width is a length, not a number of samples

Declared in the surface's own unit and converted using its spacing. A parameter
in samples would mean a different physical distance on every instrument, and the
sensitivity study would then be sweeping the instrument as well as the step.
The conversion rounds to the nearest whole sample, and a width that rounds to
none is refused rather than silently doing nothing: a sweep is entitled to
assume that a step it asked for happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage

from gutachten.surface import Surface
from gutachten.transforms.base import Parameters, SurfaceProperty, record_for
from gutachten.transforms.registry import REGISTRY

__all__ = ["EdgeCriterion", "EdgeParameters", "TrimEdge"]


class EdgeCriterion(Enum):
    """Where the width is measured from."""

    #: In from the boundary of the array.
    FRAME = "frame"
    #: Out from whatever is already missing.
    GROW = "grow"


@dataclass(frozen=True)
class EdgeParameters:
    """How much of the edge to remove, and what to measure it from.

    ``width`` is a length in the surface's own unit, and ``criterion`` is the
    name of an :class:`EdgeCriterion`. A name rather than the enum itself
    because a parameter record is written into the manifest as plain data, and a
    field that cannot be serialised is a field that cannot be recorded.
    """

    width: float
    criterion: str


class TrimEdge:
    """Mark the edge of the measured region as missing."""

    identifier = "trim-edge"
    version = "1"
    parameters_type = EdgeParameters
    produces = frozenset({SurfaceProperty.EDGES_TRIMMED})
    requires = frozenset[SurfaceProperty]()
    # Trimming after a bandpass filter removes a region whose values have
    # already spread into the surface around it, so the trim takes away the
    # symptom and leaves the cause.
    refuses = frozenset({SurfaceProperty.FILTERED})

    def apply(self, surface: Surface, parameters: Parameters) -> Surface:
        # The pipeline checks the record type before the chain starts and
        # `record_for` checks it again on the way out. This one is what narrows
        # the type for the checker, and it refuses before a single field has
        # been read, which the one on the way out cannot do. Its wording says
        # which of the three refused, so a test can tell them apart.
        if not isinstance(parameters, EdgeParameters):
            raise TypeError(
                f"{self.identifier!r} was handed {type(parameters).__name__} rather than "
                "EdgeParameters, so nothing here has read a field off it"
            )
        criterion = _criterion(parameters)
        down = _samples(parameters.width, surface.spacing_y, "spacing_y")
        across = _samples(parameters.width, surface.spacing_x, "spacing_x")

        heights = np.array(surface.heights, dtype=np.float64, copy=True)
        if criterion is EdgeCriterion.FRAME:
            heights[:down, :] = np.nan
            heights[-down:, :] = np.nan
            heights[:, :across] = np.nan
            heights[:, -across:] = np.nan
        else:
            heights[_grown(surface.missing, down, across)] = np.nan

        if not np.any(np.isfinite(heights)):
            raise ValueError(
                f"trimming {parameters.width} {surface.unit.value} from a "
                f"{surface.shape[0]}x{surface.shape[1]} surface leaves nothing measured. "
                "A step that removes the whole surface is a parameter set nobody meant, "
                "and every number after it would be taken over an empty array."
            )

        return surface.with_transform(record_for(self, parameters), heights)


def _criterion(parameters: EdgeParameters) -> EdgeCriterion:
    """The criterion named, refusing a name that is not one.

    A misspelled criterion would otherwise fall through to whichever branch was
    written as the default, and the manifest would record a word that did not
    decide anything.
    """
    try:
        return EdgeCriterion(parameters.criterion)
    except ValueError:
        known = ", ".join(sorted(item.value for item in EdgeCriterion))
        raise ValueError(
            f"{parameters.criterion!r} is not a criterion this step knows; it takes one of {known}"
        ) from None


def _samples(width: float, spacing: float, named: str) -> int:
    """``width`` as a whole number of samples at ``spacing``, refusing a width that vanishes."""
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(
            f"the width to remove must be a positive finite length, got {width!r}. A "
            "step asked for with nothing to remove is a step that should not be in the "
            "chain, and recording it as one hides that."
        )
    count = round(width / spacing)
    if count < 1:
        raise ValueError(
            f"a width of {width} rounds to no samples at a {named} of {spacing}, so this "
            "step would record itself as having run and change nothing. A sweep is "
            "entitled to assume a step it asked for happened."
        )
    return count


def _grown(missing: np.ndarray, down: int, across: int) -> np.ndarray:
    """``missing`` widened by ``down`` rows and ``across`` columns.

    A rectangular neighbourhood rather than a disc, because the two axes have
    their own spacings and a disc in samples is an ellipse on the surface.
    """
    shape = (down * 2 + 1, across * 2 + 1)  # structural: a radius either side of the sample
    return np.asarray(ndimage.binary_dilation(missing, structure=np.ones(shape, dtype=bool)))


REGISTRY.register(TrimEdge())
