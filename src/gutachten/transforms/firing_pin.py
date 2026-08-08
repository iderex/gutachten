"""Masking the firing pin impression, with the region declared or detected.

The firing pin impression is a deep crater near the centre of the primer. It is
excluded from breech face comparison because it carries the marks of a different
part of the mechanism, and because its depth dominates any correlation it is
left inside. How much genuine breech face area survives next to it depends
entirely on where its edge is judged to be, and the published chains do not
agree on how to judge that: some use a fixed radius, some detect a circle, and
some have somebody draw the region by hand.

So the method is a parameter and the region is recorded whichever way it was
arrived at. A sweep over the detection parameters is only interpretable if the
region each cell produced is in the manifest, because the thing that moved the
score is the region and not the threshold that chose it.

## The three methods, and why only two are here

``FiringPinMethod.FIXED`` masks a circle the profile states. Nothing is
recovered, and the region in the provenance is the one that was asked for.

``FiringPinMethod.DETECTED`` finds the crater on the surface it is handed and
records what it found. This is the method the gate proves against ground truth,
because it is the only one of the three that has a recovered value to compare
with a truth at all: a fixed radius recovers nothing, so a test asserting that
the recovery is close to the truth would be asserting that a number equals
itself.

A hand drawn region is deliberately not implemented. It cannot be exercised by
an automated test at all, and every test the gate runs, runs unattended. Adding
it would mean either a step nothing here can prove or a gate that waits for a
mouse. What it owes, if it is ever wanted, is a way of getting a drawn region
into a profile as data, at which point it is the fixed method with a polygon
instead of a circle rather than a third way of running this step.

## Why the radius comes from the area and not from a circle fit

A crater edge on a real scan is ragged, and the samples nearest it are the ones
the instrument is least sure about. Fitting a circle to that boundary weights
exactly those samples hardest. Counting the samples inside and converting the
area to the radius of the circle of the same area weights every sample equally,
which is the more honest reduction of a region that is not exactly a circle to a
single number. It also degrades sensibly: a crater that is slightly oval reports
the radius of the circle covering the same surface, which is what the mask is
going to remove anyway.

## Holes inside the crater are filled before it is measured

A step that ran earlier may already have marked samples inside the crater as
missing, and a missing sample is not a sample that is above the threshold. Left
alone, those holes would be subtracted from the area and the recovered radius
would come out smaller than the truth, so the mask would leave a ring of crater
wall behind. The region is therefore closed before it is measured. That is the
case that actually arises here rather than a hypothetical one: the extractor
mark in ``profiles/every-step.json`` sits inside the crater, and mask-marks runs
before this step.

## The dilation is a parameter and not a few pixels

The crater wall is not vertical and the samples just outside the detected edge
are contaminated by it. How far out to go is exactly the sort of number that
gets set once early and never revisited, and it trades genuine breech face
signal against contamination. It is a length in the surface's own unit, it is
recorded, and the sweep can move it. Zero is a reachable configuration and not a
mistake: masking the detected region and nothing more is one of the settings
this step exists to let somebody compare against.

## Marked missing, not deleted

The array keeps its dimensions and the masked samples become not-a-number, for
the reason the edge trim gives: a step that deleted rows would move every
coordinate downstream of it, so a registration found on a masked surface would
not be a registration on the surface an operator is looking at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage

from gutachten.surface import Surface
from gutachten.transforms.base import Parameters, SurfaceProperty, record_for
from gutachten.transforms.registry import REGISTRY

__all__ = ["MUST_BE_POSITIVE", "FiringPinMethod", "FiringPinParameters", "MaskFiringPin", "Region"]

#: The fields a value of zero or less makes meaningless, and why each one. Kept
#: as data beside the check rather than as two more branches inside it, because
#: a field added to the record with no entry here is a field nobody refused.
MUST_BE_POSITIVE = {
    "declared_radius": (
        "A circle of no radius is not a region, and recording one says a firing pin was "
        "described when it was not."
    ),
    "depth_threshold": (
        "It is how far below the surface's own median a sample has to sit before it counts "
        "as crater, and a threshold of nothing counts half the surface."
    ),
}


class FiringPinMethod(Enum):
    """How the region to mask is arrived at."""

    #: The profile states the circle and this step masks it.
    FIXED = "fixed"
    #: The crater is found on the surface and what was found is recorded.
    DETECTED = "detected"


@dataclass(frozen=True)
class Region:
    """A circle on the surface, in the surface's own unit.

    The centre is measured from the centre of the field, positive along each
    axis in the direction the array's indices increase, which is the convention
    the surface's orientation record names.
    """

    centre_y: float
    centre_x: float
    radius: float


@dataclass(frozen=True)
class FiringPinParameters:
    """Which method, what it needs, and how far past the edge to go.

    Lengths are in the surface's own unit. The fields belonging to the method
    that is not in use are null rather than absent: a profile says which state
    each field is in, and a reader of a manifest can see that the fixed circle
    was not merely forgotten.

    ``method`` is the name of a :class:`FiringPinMethod` rather than the member
    itself, because a parameter record is written into the manifest as plain
    data and a field that cannot be serialised is a field that cannot be
    recorded.

    ``depth_threshold`` is how far below the surface's own median a sample has
    to sit before it counts as crater. It is stated as a depth rather than as a
    height, because a threshold in absolute height would mean a different thing
    on every scan and on every step of a chain that has removed form.
    """

    method: str
    declared_centre_y: float | None
    declared_centre_x: float | None
    declared_radius: float | None
    depth_threshold: float | None
    dilation: float


class MaskFiringPin:
    """Mask the firing pin impression, and record the region that was masked."""

    identifier = "mask-firing-pin"
    version = "1"
    parameters_type = FiringPinParameters
    produces = frozenset({SurfaceProperty.MASKED})
    requires = frozenset[SurfaceProperty]()
    # The same case mask-marks refuses, and for the same reason. A bandpass
    # spreads the crater into its neighbourhood, so a mask applied afterwards
    # takes away the crater and leaves what the crater did to the surface around
    # it, while the run exits zero.
    refuses = frozenset({SurfaceProperty.FILTERED})

    def apply(self, surface: Surface, parameters: Parameters) -> Surface:
        # The pipeline checks the record type before the chain starts and
        # `record_for` checks it again on the way out. This one narrows the type
        # for the checker and refuses before a single field has been read.
        if not isinstance(parameters, FiringPinParameters):
            raise TypeError(
                f"{self.identifier!r} was handed {type(parameters).__name__} rather than "
                "FiringPinParameters, so nothing here has read a field off it"
            )
        method = _method(parameters)
        _check(parameters, method)

        measured = ~surface.missing
        if not np.any(measured):
            raise ValueError(
                "every sample of this surface is missing, so there is no crater to find "
                "and nothing to mask. A step that ran over an empty surface would record "
                "itself as having masked one."
            )

        y_um, x_um = _coordinates(surface)
        if method is FiringPinMethod.DETECTED:
            region = _detected(surface, y_um, x_um, measured, parameters)
        else:
            region = _declared(parameters)

        dilated = region.radius + parameters.dilation
        inside = np.hypot(y_um - region.centre_y, x_um - region.centre_x) <= dilated
        masked = inside & measured
        if not np.any(masked):
            raise ValueError(
                f"the region this step would mask, centred at ({region.centre_y}, "
                f"{region.centre_x}) with a radius of {dilated} {surface.unit.value}, "
                "covers no measured sample of this surface. The run would record a firing "
                "pin as having been removed from a surface it was never on."
            )
        if int(np.count_nonzero(masked)) == int(np.count_nonzero(measured)):
            raise ValueError(
                "the region this step would mask covers every measured sample of this "
                "surface. Every number after it would be taken over an empty array, which "
                "most reductions answer with not-a-number rather than with an error."
            )

        heights = np.array(surface.heights, dtype=np.float64, copy=True)
        heights[masked] = np.nan
        record = record_for(self, parameters).with_outcomes(
            centre_y=region.centre_y,
            centre_x=region.centre_x,
            radius=region.radius,
            masked_radius=dilated,
            masked_samples=int(np.count_nonzero(masked)),
        )
        return surface.with_transform(record, heights)


def _method(parameters: FiringPinParameters) -> FiringPinMethod:
    """The method named, refusing a name that is not one.

    A misspelled method would otherwise fall through to whichever branch was
    written last, and the manifest would record a word that decided nothing.
    """
    try:
        return FiringPinMethod(parameters.method)
    except ValueError:
        known = ", ".join(sorted(item.value for item in FiringPinMethod))
        raise ValueError(
            f"{parameters.method!r} is not a method this step knows; it takes one of "
            f"{known}. A hand drawn region is not one of them, because every test the "
            "gate runs, runs unattended."
        ) from None


def _check(parameters: FiringPinParameters, method: FiringPinMethod) -> None:
    """Refuse a record whose fields do not match the method it names.

    Both directions are refused. A fixed circle carried alongside a detection
    threshold is a record where a reader cannot tell which of the two decided
    the region, and a detection run that also states a radius reads afterwards
    as though the radius had been used.
    """
    if isinstance(parameters.dilation, bool) or not isinstance(parameters.dilation, (int, float)):
        raise TypeError(f"dilation must be a length, got {parameters.dilation!r}")
    if not math.isfinite(parameters.dilation) or parameters.dilation < 0.0:
        raise ValueError(
            f"dilation must be a finite length of zero or more, got {parameters.dilation!r}. "
            "Zero is the configuration that masks the region and nothing further, and a "
            "negative one would shrink the mask inside the crater it is there to remove."
        )

    fixed_fields: tuple[tuple[str, float | None], ...] = (
        ("declared_centre_y", parameters.declared_centre_y),
        ("declared_centre_x", parameters.declared_centre_x),
        ("declared_radius", parameters.declared_radius),
    )
    detected_fields: tuple[tuple[str, float | None], ...] = (
        ("depth_threshold", parameters.depth_threshold),
    )

    if method is FiringPinMethod.FIXED:
        wanted, unwanted = fixed_fields, detected_fields
    else:
        wanted, unwanted = detected_fields, fixed_fields

    absent = [name for name, value in wanted if value is None]
    if absent:
        raise ValueError(
            f"the {method.value!r} method needs {absent} and this record leaves them null. "
            "A method that had to invent one of its own inputs would be reading a value "
            "nobody recorded, which is the thing a parameter record exists to prevent."
        )
    present = [name for name, value in unwanted if value is not None]
    if present:
        raise ValueError(
            f"the {method.value!r} method does not read {present}, and this record states "
            "them. A manifest carrying both leaves a reader unable to say which of the two "
            "decided the region that was masked."
        )

    given: dict[str, float] = {}
    for name, value in wanted:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        given[name] = float(value)

    for name, why in MUST_BE_POSITIVE.items():
        if name in given and given[name] <= 0.0:
            raise ValueError(f"{name} must be a positive length, got {given[name]!r}. {why}")


def _coordinates(surface: Surface) -> tuple[np.ndarray, np.ndarray]:
    """Physical coordinates in the surface's own unit, centred on the field."""
    rows, columns = surface.shape
    down = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2) * surface.spacing_y
    across = (np.arange(columns, dtype=np.float64) - (columns - 1) / 2) * surface.spacing_x
    y_um, x_um = np.meshgrid(down, across, indexing="ij")
    return np.asarray(y_um), np.asarray(x_um)


def _declared(parameters: FiringPinParameters) -> Region:
    """The circle the profile stated. ``_check`` has already refused a null."""
    return Region(
        centre_y=float(parameters.declared_centre_y or 0.0),
        centre_x=float(parameters.declared_centre_x or 0.0),
        radius=float(parameters.declared_radius or 0.0),
    )


def _detected(
    surface: Surface,
    y_um: np.ndarray,
    x_um: np.ndarray,
    measured: np.ndarray,
    parameters: FiringPinParameters,
) -> Region:
    """Find the crater and say where it is and how big it is.

    The threshold is relative to the surface's own median, so it means the same
    thing before and after a step that has moved the whole surface up or down.
    The median is taken over the measured samples only, because a missing sample
    is not a low one.
    """
    threshold = float(parameters.depth_threshold or 0.0)
    heights = np.asarray(surface.heights, dtype=np.float64)
    median = float(np.median(heights[measured]))
    deep = measured & (heights < median - threshold)
    if not np.any(deep):
        raise ValueError(
            f"no sample of this surface sits more than {threshold} {surface.unit.value} "
            f"below its median of {median}, so there is no crater here to detect. Either "
            "the threshold is deeper than the impression or this surface has none, and the "
            "two are not the same thing: say which by looking before widening it."
        )

    # Closing the region before measuring it. A hole is a sample an earlier step
    # already marked missing, and subtracting it from the area would report a
    # radius smaller than the crater, leaving a ring of wall inside the mask.
    filled = np.asarray(ndimage.binary_fill_holes(deep))
    labelled, count = ndimage.label(filled)
    sizes = np.bincount(np.asarray(labelled).ravel())
    # Label zero is everything that is not in a region.
    largest = int(np.argmax(sizes[1:])) + 1
    region = np.asarray(labelled) == largest

    area = float(np.count_nonzero(region)) * surface.spacing_y * surface.spacing_x
    radius = math.sqrt(area / math.pi)
    centre_y = float(np.mean(y_um[region]))
    centre_x = float(np.mean(x_um[region]))

    offset = math.hypot(centre_y, centre_x)
    if offset > radius:
        raise ValueError(
            f"the deepest region on this surface is centred {offset} {surface.unit.value} "
            f"from the centre of the field and its own radius is {radius}, so the centre of "
            "the field is not inside it. A firing pin impression is near the centre of the "
            f"primer, and masking this region would remove {count} region(s) worth of "
            "something else while leaving the impression in place."
        )
    return Region(centre_y=centre_y, centre_x=centre_x, radius=radius)


REGISTRY.register(MaskFiringPin())
