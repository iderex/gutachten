"""Searching for the registration between two surfaces over translation and rotation.

The cell correlation in `gutachten.compare.cells` answers where a tile of one
face matches best on another one at a single orientation. Two cartridge cases
fired by the same mechanism are almost never at the same orientation, so that
answer on its own is the answer to a question nobody asked. This module runs the
same correlation once per candidate rotation and keeps, for each cell, the
orientation and the displacement where it matched best.

What comes out is per cell and stops there. Whether the cells agree with one
another, and how many agreeing cells make a score, is the rule in #73. Reporting
a single registration for the pair here would settle that question in passing,
and the disagreement between cells is the quantity the whole method is built on.

## Which surface is turned

The reference, once per angle, and never the cells. A cell is resampled onto a
grid it was not measured on, and resampling smooths exactly the fine structure
the correlation is looking at, so a chain that interpolated every tile at every
angle would be measuring its own interpolation as much as the surface. The
subject is divided once, its samples are the ones the instrument produced, and
the cost of the resampling is paid on the surface the tiles are slid over.

The resampling is bilinear and a rotated sample counts as a measurement only
where every sample the interpolation drew on was measured. The usual shortcut is
to interpolate through the missing samples and take whatever comes out, which
invents heights around the rim of every hole and around the whole edge of the
field. Here the weight of the measured samples is resampled alongside the
heights, and anything short of the full weight becomes not-a-number. That
shrinks the surface slightly at every boundary, which is the honest outcome: a
sample nobody measured stays a sample nobody measured after the field is turned.

Turning a field inside its own array also loses its corners. That loss is real
and it is not hidden: those samples come back as absent, they lower the overlap
of any placement that reaches them, and a placement whose overlap falls under
the declared proportion is not scored at all.

## Why the translation is bounded, and what goes wrong when it is not

The correlation is taken over the placements where a tile lies wholly inside the
other array. Without a bound that is the whole array, and a tile cut from the
edge of the subject then cannot be placed at the displacement it actually moved
by, because that placement hangs off the side. Measured on a 192 by 192 pair
carrying a known displacement of two samples down and three across, a three by
three grid recovered it on the four cells that could reach it and found a
spurious peak on the other five; the same construction with the bound in place
recovered it on all nine. The failure is worst exactly where a real face has
most of its usable surface, because the interior is where the firing pin
impression is.

So the reference is padded by the bound before the search and the placements
outside it are discarded, which makes every displacement inside the bound
reachable from every cell. The padding is absent measurement rather than zero
for the same reason the resampling refuses to invent one: a tile hanging over
the edge correlates against nothing there and its overlap says so.

The bound is a declared parameter and not a margin chosen inside this file. A
number that decides which registrations are reachable belongs in the manifest
with the others, and this one has the same double edge the rotation range has:
too tight and a genuine displacement is outside the search, too loose and a
non-matching pair gets more placements to find a spurious peak in.

## What the parameters are and why each is one

The rotation range and the rotation step, because a step too coarse steps over
the angle a genuine match sits at, and a range too wide hands a non-matching
pair more orientations to find a peak at. That second effect runs straight from
a search setting to a false positive rate, which makes it one of the more
interesting things this project can put a number on.

The translation bound, for the reason above.

The grid and the minimum measured proportion, which are the cell parameters and
are carried through unchanged rather than restated, so the two stages cannot
drift apart on the two numbers they share.

The step has to divide the range. A range of five degrees searched in steps of
two is a search over four, six or nothing depending on how the arithmetic is
rounded, and a manifest recording the range would then name something that was
not searched.

## Determinism, and the tie nobody chose

Every draw here is arithmetic; nothing samples. Two things still decide an
answer where the numbers do not, and both are fixed rather than left to fall out
of the iteration order. Among the placements of one cell at one angle, the first
in row-then-column order wins a tie, which is what `numpy.nanargmax` does. Among
angles, the first searched wins a tie, because the comparison that replaces a
cell's best is strictly greater. The angles are searched from the negative end
of the range upwards, so a pair that correlates identically at two orientations
reports the same one on every run and on every machine.

## What the cost is, and where the number lives

`Registration.correlations` is how many cell-by-angle correlations were
evaluated. It is arithmetic on the parameters, it is the same on every machine,
and it is the quantity a sweep design multiplies up. What one of them costs in
seconds is not: that is a property of the machine, so it is measured in
`docs/registration.md` with the command that produced it and the machine it was
produced on, and it is deliberately not written into a manifest, where a wall
clock reading would make two runs of one configuration differ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate

from gutachten.compare.cells import Cell, CellParameters, _comparable, correlate, divide
from gutachten.manifest import ComparisonRecord
from gutachten.surface import Surface

__all__ = [
    "METHOD",
    "VERSION",
    "CellRegistration",
    "Registration",
    "RegistrationParameters",
    "angles",
    "record",
    "register",
    "rotate_heights",
]

#: What a manifest names this stage by, and the version of what it computes. The
#: version moves when the same input would produce a different number, which is
#: the transform rule applied to a stage that is not a transform.
METHOD = "cell-registration"
VERSION = "1"

#: How much of the interpolation weight has to come from measured samples for a
#: resampled sample to be a measurement. Bilinear weights over a wholly measured
#: neighbourhood sum to one, so anything below this is a stencil that reached
#: into a hole or over the edge of the field. The slack is for the float64
#: arithmetic in the resampling and for nothing else.
_FULL_WEIGHT = 1.0 - 1e-9

#: How far the step is allowed to miss dividing the range before the range is
#: refused as one the search does not cover, as a fraction of the step.
_DIVIDES = 1e-9


@dataclass(frozen=True)
class RegistrationParameters:
    """Everything the search is told, with nothing left implicit.

    ``grid`` and ``minimum_valid`` are the cell parameters and mean there what
    they mean in `gutachten.compare.cells`. ``rotation_range_deg`` is searched
    both ways from zero and ``rotation_step_deg`` is the spacing between the
    orientations tried. ``translation_limit`` is the largest displacement in
    samples, along either axis, that a cell may be matched at.
    """

    grid: int
    minimum_valid: float
    rotation_range_deg: float
    rotation_step_deg: float
    translation_limit: int

    @property
    def cells(self) -> CellParameters:
        """The two numbers this stage shares with the cell correlation."""
        return CellParameters(grid=self.grid, minimum_valid=self.minimum_valid)


@dataclass(frozen=True)
class CellRegistration:
    """Where one cell matched best over every orientation searched.

    ``down`` and ``across`` are the displacement from where the cell sat on its
    own surface, in samples, and ``rotation_deg`` is the orientation of the
    reference at which that displacement was found. ``overlap`` is how many
    samples both surfaces measured at that placement, which is what the
    correlation was actually taken over.
    """

    row: int
    column: int
    down: int
    across: int
    rotation_deg: float
    correlation: float
    overlap: int


@dataclass(frozen=True)
class Registration:
    """What the search found, and what it cost to find it.

    ``matches`` holds one entry per cell that came back with a correlation, in
    row then column order. A cell that met no placement with enough overlap is
    absent rather than present with a number standing in for one, because a cell
    that could not be registered and a cell that registered badly are different
    inputs to the rule that counts them.

    ``by_angle`` holds the same thing one orientation at a time: for each angle
    searched, where each cell matched best at that angle alone. ``matches`` is
    the best of those per cell and is derived from it. Both are kept because a
    decision rule that only ever sees the best-over-angles answer cannot ask how
    the count of agreeing cells varies with orientation, which is what the high
    variant in #75 is about, and re-running the search per angle to recover it
    would pay for the whole search again.

    ``correlations`` is how many cell-by-angle correlations were evaluated. It is
    the same number on every machine and it is what a sweep design is costed
    from.
    """

    matches: tuple[CellRegistration, ...]
    by_angle: tuple[tuple[float, tuple[CellRegistration, ...]], ...]
    angles_deg: tuple[float, ...]
    correlations: int


def _checked(parameters: RegistrationParameters) -> tuple[float, float, int]:
    """Refuse a search whose settings do not describe a search.

    The grid and the minimum proportion are refused by the cell parameters they
    are, and are not checked a second time here: two places refusing one
    condition is two messages a reader has to reconcile when one of them fires.
    """
    for name, value in (
        ("rotation_range_deg", parameters.rotation_range_deg),
        ("rotation_step_deg", parameters.rotation_step_deg),
    ):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number of degrees, got {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number of degrees, got {value!r}")

    span = float(parameters.rotation_range_deg)
    step = float(parameters.rotation_step_deg)

    if span < 0.0:
        raise ValueError(
            f"the rotation range is searched both ways from zero, so it is a half width "
            f"and cannot be negative, got {span!r}."
        )
    if span >= 180.0:
        raise ValueError(
            f"a rotation range of {span!r} degrees searches past a half turn in both "
            "directions, so the same orientation is searched twice under two names and "
            "which of them a cell reports is decided by the tie break rather than by the "
            "surface."
        )
    if step <= 0.0:
        raise ValueError(
            f"the rotation step must be a positive number of degrees, got {step!r}. A step "
            "of nothing does not reach the end of any range."
        )
    if span > 0.0 and step > span:
        raise ValueError(
            f"a rotation step of {step!r} degrees over a range of {span!r} searches no "
            "orientation but zero, which is a range of zero recorded as something else."
        )

    steps = span / step
    if abs(steps - round(steps)) > _DIVIDES:
        raise ValueError(
            f"a rotation step of {step!r} degrees does not divide a range of {span!r}, so "
            f"the search stops at {round(steps) * step!r} degrees and the manifest would "
            "record a range that was never reached."
        )

    limit = parameters.translation_limit
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            f"the translation limit must be a whole number of samples, got {limit!r}. Half "
            "a sample is not a placement this search can visit."
        )
    if limit < 0:
        raise ValueError(
            f"the translation limit is a distance in samples and cannot be negative, got {limit!r}."
        )
    return span, step, limit


def angles(parameters: RegistrationParameters) -> tuple[float, ...]:
    """The orientations the search will try, from the negative end upwards.

    Built as a whole number of steps either side of zero rather than by adding
    the step repeatedly, so the end points are the range to the last digit and
    zero is exactly zero. An accumulated sum drifts, and a search whose last
    angle is 9.999999999999998 degrees is a search whose manifest says ten.
    """
    span, step, _ = _checked(parameters)
    count = round(span / step)
    return tuple(float(index * step) for index in range(-count, count + 1))


def record(parameters: RegistrationParameters) -> ComparisonRecord:
    """The manifest entry for a search made with these parameters.

    Built here rather than assembled by a caller, so the numbers that decide
    which registrations are reachable cannot reach a result without reaching the
    record of it.
    """
    span, step, limit = _checked(parameters)
    cells = parameters.cells
    return ComparisonRecord(
        method=METHOD,
        version=VERSION,
        parameters=(
            ("grid", cells.grid),
            ("minimum_valid", float(cells.minimum_valid)),
            ("rotation_range_deg", span),
            ("rotation_step_deg", step),
            ("translation_limit", limit),
        ),
    )


def rotate_heights(heights: np.ndarray, degrees: float) -> np.ndarray:
    """``heights`` turned by ``degrees`` about the centre of its own array.

    Positive is the same sense as `gutachten.synth.generate`'s ``rotation_deg``,
    which is the direction the generator moves a striation pattern in. That
    convention is the one this project's own ground truth is built in, and it is
    the opposite of the underlying library's, so the angle is negated here in one
    place rather than at every call. A test compares this function against a
    pattern the generator built at the same angle, so a sign flip is a red suite
    rather than a registration that reports the wrong way round.

    The array keeps its shape. What rotates out of it is lost and comes back as
    absent, which is what it is.
    """
    values = np.where(np.isnan(heights), 0.0, heights).astype(np.float64)
    weight = (~np.isnan(heights)).astype(np.float64)

    turned = rotate(values, -float(degrees), reshape=False, order=1, mode="constant", cval=0.0)
    carried = rotate(weight, -float(degrees), reshape=False, order=1, mode="constant", cval=0.0)

    # Dividing by the carried weight would fill the rim of every hole with a
    # height assembled out of the measured samples next to it. Those heights are
    # this module's own invention and they would correlate.
    result: np.ndarray = np.where(carried >= _FULL_WEIGHT, turned, np.nan)
    return result


def _best_placement(
    cell: Cell, padded: np.ndarray, minimum: float, limit: int
) -> tuple[int, int, float, int] | None:
    """Where ``cell`` matched best inside the bound, or nothing if nowhere did.

    The placements outside the bound are removed before the maximum is taken
    rather than after. Taking the maximum first and rejecting it afterwards
    reports no match for a cell that has a perfectly good one inside the bound,
    which is a cell silently dropped from the count that decides the result.
    """
    correlation, overlap = correlate(cell.heights, padded, minimum)

    down = np.arange(correlation.shape[0])[:, None] - limit - cell.top
    across = np.arange(correlation.shape[1])[None, :] - limit - cell.left
    inside = (np.abs(down) <= limit) & (np.abs(across) <= limit)

    bounded = np.where(inside, correlation, np.nan)
    if not np.any(np.isfinite(bounded)):
        return None

    at = np.unravel_index(int(np.nanargmax(bounded)), bounded.shape)
    return (
        int(at[0]) - limit - cell.top,
        int(at[1]) - limit - cell.left,
        float(bounded[at]),
        int(overlap[at]),
    )


def register(
    subject: Surface, reference: Surface, parameters: RegistrationParameters
) -> Registration:
    """Search every orientation in the range for where each cell of ``subject`` sits.

    The reference is turned and the subject's cells are not, so the tiles being
    correlated carry the samples the instrument produced rather than samples
    this module interpolated.
    """
    _span, _step, limit = _checked(parameters)
    _comparable(subject, reference)

    cells = parameters.cells
    minimum = float(cells.minimum_valid)
    tiles = divide(subject, cells)
    usable = [tile for tile in tiles if tile.measured >= minimum * tile.heights.size]
    if not usable:
        raise ValueError(
            f"no cell of a {cells.grid}x{cells.grid} grid carries {minimum} of its samples "
            f"measured, so there is nothing to register. The subject surface has "
            f"{int(np.count_nonzero(~subject.missing))} measured samples of "
            f"{subject.heights.size}."
        )

    searched = angles(parameters)
    best: dict[tuple[int, int], CellRegistration] = {}
    per_angle: list[tuple[float, tuple[CellRegistration, ...]]] = []
    evaluated = 0

    for angle in searched:
        turned = rotate_heights(np.asarray(reference.heights), angle)
        padded = np.pad(turned, limit, mode="constant", constant_values=np.nan)
        here: list[CellRegistration] = []
        for tile in usable:
            evaluated += 1
            found = _best_placement(tile, padded, minimum, limit)
            if found is None:
                continue
            down, across, correlation, overlap = found
            at_this_angle = CellRegistration(
                row=tile.row,
                column=tile.column,
                down=down,
                across=across,
                rotation_deg=angle,
                correlation=correlation,
                overlap=overlap,
            )
            here.append(at_this_angle)
            key = (tile.row, tile.column)
            standing = best.get(key)
            # Strictly greater, so the first angle searched holds a tie. A pair
            # that correlates identically at two orientations then reports the
            # same one on every run rather than the one the loop happened to
            # reach last.
            if standing is None or correlation > standing.correlation:
                best[key] = at_this_angle
        per_angle.append((angle, tuple(here)))

    if not best:
        raise ValueError(
            f"every one of the {len(usable)} usable cells was searched at each of "
            f"{len(searched)} orientations and not one of them came back with a "
            f"correlation. Either no placement within {limit} samples leaves {minimum} of "
            "a cell overlapping measured surface, or nothing varies enough at any of them "
            "for a correlation to mean anything."
        )

    return Registration(
        matches=tuple(best[key] for key in sorted(best)),
        by_angle=tuple(per_angle),
        angles_deg=searched,
        correlations=evaluated,
    )
