"""Masking the drag mark and the extractor mark, with excluding them a setting.

Both are real toolmarks and both are excluded from breech face comparison in the
published chains. That exclusion is a choice rather than a fact, and it is one
nobody has costed.

A drag mark is produced as the case moves against the breech face, so its
position depends on the mechanism's motion. It carries information and it is
also unstable between firings. An extractor mark is made by a different part of
the mechanism and sits near the rim. Excluding either is defensible; what
excluding it is worth on the same data is not known.

So this step takes where each mark is as parameters and whether to exclude it as
a separate one. The sweep moves the two booleans, the board can then report what
excluding a drag mark costs and buys, and not excluding one is a reachable
configuration rather than a change to the code.

## Where the mark is, and whether to remove it, are two different facts

The geometry is declared whether or not the mark is excluded, and it is recorded
either way. Where the drag mark on a particular scan lies is a property of that
scan; whether this run cut it out is a property of the run. A manifest that
recorded the geometry only when the mask was applied could not tell a surface
with no drag mark from one whose drag mark was kept, and those are different
inputs to the same score.

## A step that excludes nothing is a configuration, not a mistake

The edge trim refuses a width that removes no sample, because a step asked for
with nothing to remove is a step that should not be in the chain. This step is
the opposite case and deliberately so: excluding neither mark is the
configuration the whole comparison is against, and a sweep has to be able to
visit it. What is refused is asking for an exclusion that then covers no sample,
because that is the silent no-op the trim's rule is really about.

## Marked missing, not deleted

The array keeps its dimensions and the excluded samples become not-a-number, for
the reason the edge trim gives: a step that deleted rows would move every
coordinate downstream of it, so a registration found on a masked surface would
not be a registration on the surface an operator is looking at.

## How much each region took is recorded

The count of samples each region masked goes into the provenance as an outcome,
so a sweep comparing the two settings can say what the exclusion cost in surface
as well as what it did to the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gutachten.surface import Surface
from gutachten.transforms.base import Parameters, SurfaceProperty, record_for
from gutachten.transforms.registry import REGISTRY

__all__ = ["MarkParameters", "MaskMarks"]


@dataclass(frozen=True)
class MarkParameters:
    """Where the two marks are, and whether each one is cut out.

    Lengths are in the surface's own unit and angles are in degrees.

    The drag mark is a straight band. ``drag_position`` is the distance from the
    centre of the field to the middle of the band, measured across the band
    rather than along it, so it means the same thing at every angle.
    ``drag_angle_deg`` is the direction the band runs, from the column axis.

    The extractor mark is a disc. ``extractor_distance`` and
    ``extractor_angle_deg`` say where its centre sits relative to the centre of
    the field, in polar form, because an extractor mark is described by how far
    out on the rim it is and in which direction rather than by two offsets.
    """

    drag_width: float
    drag_position: float
    drag_angle_deg: float
    exclude_drag: bool

    extractor_radius: float
    extractor_distance: float
    extractor_angle_deg: float
    exclude_extractor: bool


class MaskMarks:
    """Cut out the declared mark regions, or leave them, as the settings say."""

    identifier = "mask-marks"
    version = "1"
    parameters_type = MarkParameters
    produces = frozenset({SurfaceProperty.MASKED})
    requires = frozenset[SurfaceProperty]()
    # The canonical case the ordering rules exist for. A bandpass spreads a
    # region into its neighbourhood, so a mask applied afterwards removes a
    # region whose contents have already leaked into the surface around it: the
    # symptom goes and the cause stays, and the run exits zero.
    refuses = frozenset({SurfaceProperty.FILTERED})

    def apply(self, surface: Surface, parameters: Parameters) -> Surface:
        # The pipeline checks the record type before the chain starts and
        # `record_for` checks it again on the way out. This one narrows the type
        # for the checker and refuses before a single field has been read.
        if not isinstance(parameters, MarkParameters):
            raise TypeError(
                f"{self.identifier!r} was handed {type(parameters).__name__} rather than "
                "MarkParameters, so nothing here has read a field off it"
            )
        _check(parameters)

        measured = ~surface.missing
        if not np.any(measured):
            raise ValueError(
                "every sample of this surface is missing, so there is nothing left for a "
                "mask to exclude. A step that ran over an empty surface would record "
                "itself as having masked one."
            )

        y_um, x_um = _coordinates(surface)
        drag = _band(y_um, x_um, parameters) if parameters.exclude_drag else _nothing(y_um)
        extractor = (
            _disc(y_um, x_um, parameters) if parameters.exclude_extractor else _nothing(y_um)
        )
        _refuse_an_exclusion_that_covers_nothing(drag, extractor, measured, parameters)

        excluded = (drag | extractor) & measured
        if int(np.count_nonzero(excluded)) == int(np.count_nonzero(measured)):
            raise ValueError(
                "the declared mark regions cover every measured sample of this surface. "
                "Every number after this step would be taken over an empty array, which "
                "most reductions answer with not-a-number rather than with an error."
            )

        heights = np.array(surface.heights, dtype=np.float64, copy=True)
        heights[excluded] = np.nan
        record = record_for(self, parameters).with_outcomes(
            drag_samples=int(np.count_nonzero(drag & measured)),
            extractor_samples=int(np.count_nonzero(extractor & measured)),
        )
        return surface.with_transform(record, heights)


def _check(parameters: MarkParameters) -> None:
    """Refuse a geometry that does not describe a region.

    Every field is checked whether or not its region is excluded. The geometry
    is recorded either way, and a manifest carrying a position that is not a
    number describes a scan nobody can reconstruct.
    """
    for name, value in (
        ("drag_width", parameters.drag_width),
        ("extractor_radius", parameters.extractor_radius),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a length, got {value!r}")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be a positive finite length, got {value!r}. A region of no "
                "width is not a region, and recording one says a mark was described when "
                "it was not."
            )

    for name, value in (
        ("drag_position", parameters.drag_position),
        ("drag_angle_deg", parameters.drag_angle_deg),
        ("extractor_distance", parameters.extractor_distance),
        ("extractor_angle_deg", parameters.extractor_angle_deg),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {value!r}")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    for name, setting in (
        ("exclude_drag", parameters.exclude_drag),
        ("exclude_extractor", parameters.exclude_extractor),
    ):
        if not isinstance(setting, bool):
            raise TypeError(
                f"{name} must be true or false, got {setting!r}. It is the setting the "
                "sweep moves, and a value that is neither would be recorded as one of "
                "them without saying which."
            )


def _coordinates(surface: Surface) -> tuple[np.ndarray, np.ndarray]:
    """Physical coordinates in the surface's own unit, centred on the field."""
    rows, columns = surface.shape
    down = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2) * surface.spacing_y
    across = (np.arange(columns, dtype=np.float64) - (columns - 1) / 2) * surface.spacing_x
    y_um, x_um = np.meshgrid(down, across, indexing="ij")
    return np.asarray(y_um), np.asarray(x_um)


def _nothing(like: np.ndarray) -> np.ndarray:
    """No region at all, in the shape of the field."""
    return np.zeros(like.shape, dtype=bool)


def _band(y_um: np.ndarray, x_um: np.ndarray, parameters: MarkParameters) -> np.ndarray:
    """The straight band the drag mark parameters describe.

    The position is measured across the band, as the signed distance from the
    centre of the field along the band's own normal. Measuring it along a fixed
    axis instead would make the same number mean a different place as soon as
    the angle moved, and the sweep is going to move the angle.
    """
    angle = math.radians(parameters.drag_angle_deg)
    across = y_um * math.cos(angle) - x_um * math.sin(angle)
    return np.asarray(np.abs(across - parameters.drag_position) <= parameters.drag_width / 2)


def _disc(y_um: np.ndarray, x_um: np.ndarray, parameters: MarkParameters) -> np.ndarray:
    """The disc the extractor mark parameters describe."""
    angle = math.radians(parameters.extractor_angle_deg)
    centre_y = parameters.extractor_distance * math.cos(angle)
    centre_x = parameters.extractor_distance * math.sin(angle)
    return np.asarray(np.hypot(y_um - centre_y, x_um - centre_x) <= parameters.extractor_radius)


def _refuse_an_exclusion_that_covers_nothing(
    drag: np.ndarray,
    extractor: np.ndarray,
    measured: np.ndarray,
    parameters: MarkParameters,
) -> None:
    """Refuse an exclusion that was asked for and reaches no measured sample.

    Not excluding a mark is a configuration this step exists to make reachable,
    and it is not this refusal. Asking for an exclusion that then does nothing
    is the silent no-op, and a sweep comparing the two settings would see two
    identical surfaces and record them under different parameters.
    """
    for name, region, asked in (
        ("drag mark", drag, parameters.exclude_drag),
        ("extractor mark", extractor, parameters.exclude_extractor),
    ):
        if asked and not np.any(region & measured):
            raise ValueError(
                f"the {name} region was asked to be excluded and covers no measured sample "
                "of this surface, so the run would record an exclusion that changed "
                "nothing. A sweep comparing the two settings would then see one surface "
                "under two parameter sets."
            )


REGISTRY.register(MaskMarks())
