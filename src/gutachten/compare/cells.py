"""Dividing a surface into cells and correlating each one against another surface.

The comparison this project reproduces asks, for each tile of one breech face,
where that tile best matches on the other one and how well. This module answers
that for a single orientation. Searching over rotation as well is #71, and the
rule that turns a set of answers into a count of agreeing cells is #73.

## What is a parameter here and why

The grid, because how finely a face is divided decides how many cells carry a
signal and how many carry noise, and the published descriptions fix it without
showing what it costs.

The minimum proportion of measured samples a cell must carry, because a cell
that is mostly missing correlates against whatever is left and produces a high
number from very little surface. That threshold discards cells, so it moves the
count of cells the decision is made over, which is the quantity the whole method
turns on. It belongs in the sweep with everything else.

The same threshold also bounds the overlap at a placement. The quantity it
protects is the same one in both cases, a correlation taken over too few
samples, and two separately tunable numbers for one failure would be two ways of
saying what one says.

## Missing samples are handled, not substituted

A masked normalised cross correlation, so a sample nobody measured contributes
nothing to any of the sums. The usual shortcut is to fill the missing samples
with zero, or with the surface mean, and correlate as if the array were
complete. That does not leave the answer alone: the filled region is a constant
patch, it correlates with the constant patches on the other surface, and the
score it adds is a score for a region where there is no surface at all. What
that shortcut is worth is measured in this module's tests rather than assumed
either way.

The six sums the correlation needs are each a linear cross correlation, so each
is one call to a fast transform rather than a loop over placements. The
arithmetic is the standard masked form: every sum is taken over the samples both
surfaces measured at a given placement, and the means and the variances are
recomputed for each placement from those same sums, because a mean taken over
the whole cell is not the mean of the part of it that overlapped.

## What this does not decide

Whether a cell's best correlation is high enough to mean anything, what
agreement between cells is, and how many agreeing cells make a score are all the
rule in #73. This module reports where each cell matched best and how well, and
stops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from gutachten.manifest import ComparisonRecord
from gutachten.surface import Surface

__all__ = [
    "METHOD",
    "VERSION",
    "Cell",
    "CellMatch",
    "CellParameters",
    "compare",
    "correlate",
    "divide",
    "record",
]

#: What a manifest names this stage by, and the version of what it computes. The
#: version moves when the same input would produce a different number, which is
#: the transform rule applied to a stage that is not a transform.
METHOD = "cell-correlation"
VERSION = "1"

#: When a variance reached by subtracting one large sum from another is the
#: arithmetic's own noise rather than a spread. Both sums are positive and of
#: the same size, so their difference carries the digits float64 has left after
#: the cancellation, and below this fraction of the sums themselves the two
#: agreed to every digit there was. What that means for a surface is measured in
#: this module's tests: a constant patch has no spread and gets no correlation
#: rather than a correlation between two patterns that are not there.
_MEANINGFUL = 1e-12


@dataclass(frozen=True)
class CellParameters:
    """How finely the face is divided, and how much of a cell has to be there.

    ``grid`` is the number of cells along each axis. ``minimum_valid`` is the
    proportion of a cell's samples that must have been measured, both for the
    cell to be used at all and for a placement of it to be scored.
    """

    grid: int
    minimum_valid: float


@dataclass(frozen=True, eq=False)
class Cell:
    """One tile of a surface, and where on that surface it came from.

    ``top`` and ``left`` are the position of the tile's first sample in the
    surface it was cut from, in samples. They are what a displacement is
    measured against, so a match is reported as how far the tile moved rather
    than as where it landed.
    """

    row: int
    column: int
    top: int
    left: int
    heights: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = self.heights.shape
        return int(rows), int(columns)

    @property
    def measured(self) -> int:
        """How many of the tile's samples carry a measurement."""
        return int(np.count_nonzero(~np.isnan(self.heights)))


@dataclass(frozen=True)
class CellMatch:
    """Where one cell matched best on the other surface, and how well.

    ``down`` and ``across`` are the displacement from where the cell sat on its
    own surface, in samples. ``overlap`` is how many samples both surfaces
    measured at that placement, which is what the correlation was actually taken
    over.
    """

    row: int
    column: int
    down: int
    across: int
    correlation: float
    overlap: int


def _checked(parameters: CellParameters) -> tuple[int, float]:
    grid = parameters.grid
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 1:
        raise ValueError(
            f"the grid must be a whole number of cells along an axis, got {grid!r}. A "
            "grid of nothing divides the face into no cells and the comparison has "
            "nothing to count."
        )
    minimum = parameters.minimum_valid
    if isinstance(minimum, bool) or not isinstance(minimum, int | float):
        raise TypeError(f"the minimum valid proportion must be a number, got {minimum!r}")
    minimum = float(minimum)
    if not np.isfinite(minimum) or not 0.0 < minimum <= 1.0:
        raise ValueError(
            f"the minimum valid proportion must be above zero and at most one, got "
            f"{minimum!r}. At zero a cell with nothing measured in it is compared against "
            "the other surface, and whatever number comes back is a number about nothing."
        )
    return grid, minimum


def record(parameters: CellParameters) -> ComparisonRecord:
    """The manifest entry for a comparison made with these parameters.

    Built here rather than assembled by a caller, so the two numbers that decide
    how many cells enter the count cannot reach a result without reaching the
    record of it.
    """
    grid, minimum = _checked(parameters)
    return ComparisonRecord(
        method=METHOD,
        version=VERSION,
        parameters=(("grid", grid), ("minimum_valid", minimum)),
    )


def divide(surface: Surface, parameters: CellParameters) -> tuple[Cell, ...]:
    """Cut ``surface`` into a grid of tiles, in row then column order.

    The tiles differ in size by at most one sample where the grid does not
    divide the surface, rather than the remainder being dropped. A dropped
    remainder is a strip of the face that no cell ever covers, and on a scan
    whose edge has already been trimmed that strip is real surface.
    """
    grid, _ = _checked(parameters)
    rows, columns = surface.shape
    if grid > min(rows, columns):
        raise ValueError(
            f"a grid of {grid} on a {rows}x{columns} surface would cut a cell with no "
            "samples in it. The grid is a number of cells along an axis, not a size."
        )

    down = np.array_split(np.arange(rows), grid)
    across = np.array_split(np.arange(columns), grid)
    heights = np.asarray(surface.heights)

    return tuple(
        Cell(
            row=row,
            column=column,
            top=int(down[row][0]),
            left=int(across[column][0]),
            heights=heights[
                int(down[row][0]) : int(down[row][-1]) + 1,
                int(across[column][0]) : int(across[column][-1]) + 1,
            ],
        )
        for row in range(grid)
        for column in range(grid)
    )


def _slide(over: np.ndarray, tile: np.ndarray) -> np.ndarray:
    """``sum(over[i + k, j + l] * tile[k, l])`` for every placement of ``tile``.

    One transform rather than a loop over placements. The tile is reversed on
    both axes because a convolution reverses what it is handed and a correlation
    does not.
    """
    result: np.ndarray = fftconvolve(over, tile[::-1, ::-1], mode="valid")
    return result


def correlate(
    cell: np.ndarray, other: np.ndarray, minimum_valid: float
) -> tuple[np.ndarray, np.ndarray]:
    """The masked correlation of ``cell`` at every placement inside ``other``.

    Returns the correlation at each placement and the number of samples both
    arrays measured there. A placement whose overlap falls below
    ``minimum_valid`` of the cell, or where either side is flat over the
    overlap, has no correlation and comes back as not-a-number, because zero is
    a correlation and "not computed" is not.
    """
    if cell.ndim != 2 or other.ndim != 2:
        raise ValueError("a cell and the surface it is searched on are both two dimensional")
    if cell.shape[0] > other.shape[0] or cell.shape[1] > other.shape[1]:
        raise ValueError(
            f"a cell of {cell.shape} does not fit inside a surface of {other.shape}, so "
            "there is no placement of it to score."
        )

    cell_mask = (~np.isnan(cell)).astype(np.float64)
    other_mask = (~np.isnan(other)).astype(np.float64)
    cell_values = np.where(np.isnan(cell), 0.0, cell).astype(np.float64)
    other_values = np.where(np.isnan(other), 0.0, other).astype(np.float64)

    overlap = _slide(other_mask, cell_mask)
    # The transform reaches these through cancelling sums, so a count that is a
    # whole number by construction comes back a little either side of one.
    counted = np.rint(overlap)
    enough = counted >= max(1.0, np.ceil(minimum_valid * cell.size))

    # Every sum is taken over the overlap only, which is what makes the mean and
    # the variance below the ones of the part that overlapped rather than of the
    # whole cell.
    with np.errstate(invalid="ignore", divide="ignore"):
        divisor = np.where(enough, counted, np.nan)
        both = _slide(other_values, cell_values)
        other_sum = _slide(other_values, cell_mask)
        cell_sum = _slide(other_mask, cell_values)
        other_squares = _slide(other_values * other_values, cell_mask)
        cell_squares = _slide(other_mask, cell_values * cell_values)

        numerator = both - other_sum * cell_sum / divisor
        # A variance reached as a difference of two large sums comes out
        # not-a-number where that difference goes slightly negative, and the
        # comparison below turns that into a placement with no correlation,
        # which is what an overlap with nothing varying in it is.
        spread = _spread(other_squares, other_sum, divisor) * _spread(
            cell_squares, cell_sum, divisor
        )
        correlation = np.where(
            spread > 0.0, numerator / np.where(spread > 0.0, spread, 1.0), np.nan
        )

    # A correlation cannot exceed one, and the subtractions above can leave it a
    # rounding error outside. A number above one in a report is one a reader
    # cannot make sense of.
    correlation = np.where(enough, np.clip(correlation, -1.0, 1.0), np.nan)
    return correlation, counted.astype(np.int64)


def _spread(squares: np.ndarray, total: np.ndarray, count: np.ndarray) -> np.ndarray:
    """The standard deviation over an overlap, or zero where there is not one.

    The variance is a difference of two positive sums of the same size, so what
    survives the subtraction is whatever digits were left after the cancellation.
    Where that difference is a fraction of the sums smaller than float64 can
    resolve, the two sums agreed to every digit they had and there is no spread
    to report, which is a different statement from a spread of nearly nothing.
    Reporting the difference there produces a correlation between two patterns
    that are not present.
    """
    raw = squares - total**2 / count
    return np.sqrt(np.where(raw > _MEANINGFUL * squares, raw, 0.0))


def _comparable(subject: Surface, reference: Surface) -> None:
    """Refuse two surfaces whose numbers do not mean the same thing.

    The expensive error available here is a millimetre treated as a micrometre,
    and a correlation between two surfaces at different sampling intervals is a
    correlation between two different fields of view that reports as a number
    about one.
    """
    if subject.unit is not reference.unit:
        raise ValueError(
            f"the surfaces declare {subject.unit.value} and {reference.unit.value}. "
            "Comparing them would put two length scales in one correlation."
        )
    if subject.orientation is not reference.orientation:
        raise ValueError(
            f"the surfaces declare {subject.orientation.value} and "
            f"{reference.orientation.value}. A displacement measured between them would "
            "have the wrong sign and nothing downstream could tell."
        )
    if (subject.spacing_y, subject.spacing_x) != (reference.spacing_y, reference.spacing_x):
        raise ValueError(
            f"the surfaces are sampled at {subject.spacing_y}x{subject.spacing_x} and "
            f"{reference.spacing_y}x{reference.spacing_x}. A cell from one covers a "
            "different amount of face on the other, so a displacement in samples means "
            "two things at once."
        )


def compare(
    subject: Surface, reference: Surface, parameters: CellParameters
) -> tuple[CellMatch, ...]:
    """Where each usable cell of ``subject`` matches best on ``reference``.

    A cell carrying less than the declared proportion of measured samples is not
    compared at all, and a cell with no placement that meets the same proportion
    produces no match. Both are dropped rather than scored, because a cell
    reported with a correlation taken over a handful of samples enters the count
    that decides the result on the same footing as one taken over a whole tile.
    """
    grid, minimum = _checked(parameters)
    _comparable(subject, reference)

    tiles = divide(subject, parameters)
    usable = [cell for cell in tiles if cell.measured >= minimum * cell.heights.size]
    if not usable:
        raise ValueError(
            f"no cell of a {grid}x{grid} grid carries {minimum} of its samples measured, "
            f"so there is nothing to correlate. The subject surface has "
            f"{int(np.count_nonzero(~subject.missing))} measured samples of "
            f"{subject.heights.size}."
        )

    matches: list[CellMatch] = []
    for cell in usable:
        correlation, overlap = correlate(cell.heights, np.asarray(reference.heights), minimum)
        if not np.any(np.isfinite(correlation)):
            continue
        flat = int(np.nanargmax(correlation))
        at = np.unravel_index(flat, correlation.shape)
        matches.append(
            CellMatch(
                row=cell.row,
                column=cell.column,
                down=int(at[0]) - cell.top,
                across=int(at[1]) - cell.left,
                correlation=float(correlation[at]),
                overlap=int(overlap[at]),
            )
        )

    if not matches:
        raise ValueError(
            f"every one of the {len(usable)} usable cells was searched and not one of "
            "them came back with a correlation. Either the two surfaces measure regions "
            f"that meet in less than {minimum} of a cell, or nothing varies enough at any "
            "placement for a correlation to mean anything, which is what a surface whose "
            "heights are large next to how much they vary looks like from here."
        )
    return tuple(matches)
