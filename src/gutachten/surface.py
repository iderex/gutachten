"""The internal representation of a measured surface.

A surface is a height array plus the things that make it mean something: the
sample spacing, the units those numbers are in, the orientation convention, and
a mask marking where there is no measurement. Everything downstream of the
reader works on this type and never on a bare array, because the most expensive
error available in this project is a millimetre treated as a micrometre, and an
array on its own cannot refuse it.

## What a surface is here, and why this shape

**Heights are float64 and missing data is not-a-number.** A zero is a legitimate
height. A pipeline that cannot tell an absent sample from a height of zero will
level a surface against a hole and produce a number that looks fine. So absence
has exactly one representation, it is checked for by `numpy.isnan`, and nothing
downstream has to know which sentinel a particular reader chose. An infinity is
refused at construction: it is neither a height nor an absence, and letting one
through means every statistic over the surface silently becomes an infinity too.

**One length unit covers the heights and both spacings.** Not one unit per axis
and a second for the heights. Two units on one object is precisely the confusion
this type exists to remove, and a surface whose heights are in micrometres and
whose spacing is in millimetres would pass every type check and produce an
aspect ratio wrong by a thousand. Converting to a single internal unit at read
time, and keeping the unit the file declared in the provenance record, is the
reader's job and is issue #45.

**The unit and the spacing have no defaults and construction is refused without
them.** Assuming micrometres is the assumption everybody makes and nobody writes
down. A default here would be that assumption, spelled in a place where it never
gets read.

**Orientation is recorded rather than assumed.** Which way the row axis
increases is a convention, image processing and metrology disagree about it, and
two surfaces compared across the disagreement produce a low score that looks
like a genuine non-match rather than like a defect.

**The type is immutable and the height array is copied on the way in.** A frozen
dataclass whose array field is still writeable is not immutable, because the
caller keeps a reference to the array it handed over. So the array is copied and
the copy is marked read-only. That costs one array copy per surface, which is
paid once at construction against a class of defect that is invisible until a
score comes out wrong.

**A provenance chain is carried, not derived.** Each entry names a transform and
its version. A surface that has been levelled and filtered says so, rather than
arriving as an anonymous array whose history lives in whoever ran it. What a
transform *is* is issue #49; this module holds only the record that one was
applied.

Equality is identity, deliberately. Two float arrays are equal within a
tolerance or not at all, the tolerance depends on what is being asked, and a
dataclass `__eq__` would either raise on the array comparison or invent an
answer. Comparisons in the suite go through `assert_close` in
`tests/support/tolerance.py`, where the tolerance is an argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import numpy as np

__all__ = [
    "AxisOrientation",
    "LengthUnit",
    "ParameterValue",
    "Surface",
    "TransformRecord",
]


class LengthUnit(Enum):
    """A length unit a scan may declare.

    The members are the units the exchange format and the instruments in this
    field actually emit. A unit outside this set is refused at the door rather
    than mapped onto the nearest member, because a reader that guesses is the
    failure this type is here to prevent.
    """

    NANOMETRE = "nm"
    MICROMETRE = "um"
    MILLIMETRE = "mm"
    METRE = "m"

    @property
    def micrometres(self) -> float:
        """How many micrometres one of this unit is.

        Micrometres are the internal unit: a fired cartridge case carries
        features of a few micrometres on a face of a few millimetres, so this is
        the unit in which the numbers a reader prints are readable without an
        exponent.
        """
        return _MICROMETRES_PER_UNIT[self]


# Written out per member rather than computed from a metre factor, so the
# micrometre case is exactly 1.0 and the millimetre case is exactly 1000.0
# rather than whatever a division of two inexact binary literals produces.
# `LengthUnit.micrometres` raises `KeyError` for a member missing from this map,
# and a test asserts every member is present, so adding a unit and forgetting
# its size is caught at the addition rather than at the first surface read in
# that unit.
_MICROMETRES_PER_UNIT: Final[dict[LengthUnit, float]] = {
    LengthUnit.NANOMETRE: 0.001,
    LengthUnit.MICROMETRE: 1.0,
    LengthUnit.MILLIMETRE: 1000.0,
    LengthUnit.METRE: 1000000.0,
}


class AxisOrientation(Enum):
    """Which way the row axis of the height array increases.

    Row zero is the first row of the array in both cases. What differs is what
    that means on the object: under ``Y_DOWN`` the row index increases in the
    direction an image viewer draws downwards, which is what the instruments
    write and what an image processing library assumes; under ``Y_UP`` it
    increases in the direction a plot of the surface would call up, which is the
    metrology convention.

    Neither is correct in general and this project does not pick one for a file
    to have meant. It records what the file declared, so that a comparison
    between two surfaces that disagree is a refusable condition rather than a
    low score.
    """

    Y_DOWN = "y-down"
    Y_UP = "y-up"


#: What a transform parameter may be in a provenance record. Deliberately narrow:
#: a record has to survive being written into a file and read back by somebody
#: else, and an arbitrary object does not.
ParameterValue = str | int | float | bool | None


@dataclass(frozen=True)
class TransformRecord:
    """One transform, already applied, as it appears in a surface's provenance.

    The parameters are a sorted tuple of pairs rather than a mapping, for two
    reasons. A frozen dataclass holding a dict is not frozen, because the dict
    is still writeable through the reference the caller kept. And iteration over
    an unordered container is a source of run-to-run difference, which is the
    thing the determinism work in #25 is about; sorting once here means a
    provenance chain written twice is written the same way twice.
    """

    name: str
    version: str
    parameters: tuple[tuple[str, ParameterValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a transform record needs a name")
        if not self.version:
            raise ValueError(
                f"transform {self.name!r} has no version; a record without one "
                "cannot say which implementation produced the surface"
            )
        keys = [key for key, _ in self.parameters]
        if sorted(keys) != keys:
            raise ValueError(
                f"transform {self.name!r} has parameters out of order: {keys}; "
                "build the record with TransformRecord.of, which sorts them"
            )
        if len(set(keys)) != len(keys):
            raise ValueError(f"transform {self.name!r} names a parameter twice: {keys}")

    @classmethod
    def of(cls, name: str, version: str, **parameters: ParameterValue) -> TransformRecord:
        """Build a record, sorting the parameters by name."""
        return cls(name=name, version=version, parameters=tuple(sorted(parameters.items())))


@dataclass(frozen=True, eq=False)
class Surface:
    """A height field with everything needed to know what its numbers mean.

    ``spacing_y`` is the distance between neighbouring rows and ``spacing_x``
    the distance between neighbouring columns, both in ``unit``, which is also
    the unit the heights are in.

    ``source`` identifies where the surface came from: the name of the scan, the
    identifier the public database knows it by, or the parameters of a generated
    one. It is carried so that a result can be traced back to an input, and it
    is refused empty because a surface nobody can trace is a surface nobody can
    check.
    """

    #: Height, row by column, in ``unit``. Not-a-number marks the absence of a
    #: measurement, which is a different thing from a height of zero. The array
    #: is a read-only copy of whatever was handed to the constructor.
    heights: np.ndarray
    spacing_y: float
    spacing_x: float
    unit: LengthUnit
    orientation: AxisOrientation
    source: str
    provenance: tuple[TransformRecord, ...] = field(default=())

    def __post_init__(self) -> None:
        if not isinstance(self.unit, LengthUnit):
            raise TypeError(
                f"a surface needs a declared length unit, got {self.unit!r}; "
                "there is no default, because assuming micrometres is the "
                "assumption nobody writes down"
            )
        if not isinstance(self.orientation, AxisOrientation):
            raise TypeError(
                f"a surface needs a declared axis orientation, got {self.orientation!r}"
            )

        for name, spacing in (("spacing_y", self.spacing_y), ("spacing_x", self.spacing_x)):
            if not isinstance(spacing, (int, float)) or isinstance(spacing, bool):
                raise TypeError(f"{name} must be a number in {self.unit.value}, got {spacing!r}")
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError(
                    f"{name} must be a positive finite distance in {self.unit.value}, "
                    f"got {spacing!r}"
                )

        if not isinstance(self.heights, np.ndarray):
            raise TypeError(f"heights must be a numpy array, got {type(self.heights).__name__}")
        if self.heights.ndim != 2:
            raise ValueError(
                f"a surface is a two dimensional height field, got {self.heights.ndim} "
                f"dimensions with shape {self.heights.shape}"
            )
        if self.heights.size == 0:
            raise ValueError(f"a surface with no samples is not a surface: {self.heights.shape}")
        if self.heights.dtype != np.float64:
            # Coercing here would silently turn an integer array of millimetres,
            # or a float32 array carrying its own rounding, into something that
            # looks like a measurement in float64. The caller converts and says
            # so.
            raise TypeError(
                f"heights must be float64, got {self.heights.dtype}; convert at the "
                "reader and record that the conversion happened"
            )
        if np.isinf(self.heights).any():
            raise ValueError(
                "heights contain an infinity; absence of a measurement is not-a-number "
                "and an infinite height is neither a measurement nor an absence"
            )

        if not self.source:
            raise ValueError("a surface needs a source identity")

        if not isinstance(self.provenance, tuple) or not all(
            isinstance(entry, TransformRecord) for entry in self.provenance
        ):
            raise TypeError(
                f"provenance must be a tuple of TransformRecord, got {self.provenance!r}"
            )

        # The copy is what makes this type immutable rather than merely frozen.
        # Without it the caller still holds a writeable reference to the array
        # and can move a height after every check above has passed.
        frozen_heights = self.heights.copy()
        frozen_heights.flags.writeable = False
        object.__setattr__(self, "heights", frozen_heights)

    @property
    def shape(self) -> tuple[int, int]:
        """Rows and columns."""
        rows, columns = self.heights.shape
        return int(rows), int(columns)

    @property
    def missing(self) -> np.ndarray:
        """A boolean mask, true where there is no measurement."""
        mask: np.ndarray = np.isnan(self.heights)
        return mask

    @property
    def observed(self) -> np.ndarray:
        """The heights that were measured, as a flat array, for statistics."""
        observed: np.ndarray = self.heights[~self.missing]
        return observed

    def with_transform(self, record: TransformRecord, heights: np.ndarray) -> Surface:
        """The surface a transform produced, with ``record`` appended.

        The only way provenance grows. A transform that returns a bare array and
        leaves the caller to remember what it did is a transform whose effect is
        invisible in the output, which is the thing this project is arguing
        against.
        """
        if not isinstance(record, TransformRecord):
            raise TypeError(f"a provenance entry must be a TransformRecord, got {record!r}")
        return Surface(
            heights=heights,
            spacing_y=self.spacing_y,
            spacing_x=self.spacing_x,
            unit=self.unit,
            orientation=self.orientation,
            source=self.source,
            provenance=(*self.provenance, record),
        )
