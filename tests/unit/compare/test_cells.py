"""The cell division and the masked correlation, against constructions.

The correlation is checked against a direct computation of the same definition
rather than against a recorded number, because a recorded number records
whatever the code did on the day it was written. The offsets are checked against
a generator that was told what offset to build in.

Every refusal here was removed in turn and the suite watched go red.
"""

from __future__ import annotations

import numpy as np
import pytest

from gutachten.compare.cells import (
    METHOD,
    VERSION,
    CellParameters,
    compare,
    correlate,
    divide,
    record,
)
from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import EnvironmentRecord, ProfileRecord, record_run
from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, matching_pair, non_matching_pair
from gutachten.transforms.bandpass import BandpassParameters, RobustGaussianBandpass
from gutachten.transforms.pipeline import Step
from gutachten.transforms.registry import Registry
from tests.support.tolerance import assert_close

ROWS = 256
COLUMNS = 256
SPACING_UM = 4.0
#: How far the second surface's striae were moved, along the axis they vary in.
SHIFT = 3

#: The striae run along the column axis, so the pattern is a function of the
#: column alone. A displacement along the rows moves nothing, and no comparison
#: of striae can recover it. The tests below assert the component that is
#: determined and say so where the other one is not.
STRIAE_ANGLE_DEG = 0.0
STRIAE_SPACING_UM = 40.0


def parameters(**overrides: object) -> SurfaceParameters:
    settings: dict[str, object] = {
        "rows": ROWS,
        "columns": COLUMNS,
        "pixel_spacing_um": SPACING_UM,
        "striae_angle_deg": STRIAE_ANGLE_DEG,
        "striae_spacing_um": STRIAE_SPACING_UM,
        # The form, the firing pin impression and the drag mark do not move when
        # the striae do, so a surface carrying them correlates on features that
        # are in the same place on both and reports no displacement. A real
        # chain removes them before this stage; the step that removes the firing
        # pin impression is #58, so they are switched off in the generator here
        # rather than pretended away.
        "form_depth_um": 0.0,
        "firing_pin_depth_um": 0.0,
        "drag_mark_depth_um": 0.0,
        "noise_um": 0.0,
        "seed": 20260808,
    }
    settings.update(overrides)
    return SurfaceParameters(**settings)  # type: ignore[arg-type]


def as_surface(heights: np.ndarray, *, spacing: float = SPACING_UM) -> Surface:
    return Surface(
        heights=heights,
        spacing_y=spacing,
        spacing_x=spacing,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def a_matching_pair(**overrides: object) -> tuple[Surface, Surface]:
    first, second = matching_pair(
        parameters(**overrides), translation_px=(0.0, float(SHIFT)), rotation_deg=0.0
    )
    return as_surface(np.asarray(first.heights_um)), as_surface(np.asarray(second.heights_um))


def by_definition(cell: np.ndarray, other: np.ndarray, minimum_valid: float) -> np.ndarray:
    """The same correlation, computed one placement at a time from its definition.

    Slow and obvious. What it is for is that the fast route reaches the answer
    through six cross correlations and a subtraction of large sums, which is
    where a masked correlation goes wrong, and a fast route checked only against
    itself is checked against nothing.
    """
    rows = other.shape[0] - cell.shape[0] + 1
    columns = other.shape[1] - cell.shape[1] + 1
    floor = max(1.0, np.ceil(minimum_valid * cell.size))
    out = np.full((rows, columns), np.nan)

    for i in range(rows):
        for j in range(columns):
            window = other[i : i + cell.shape[0], j : j + cell.shape[1]]
            both = ~np.isnan(window) & ~np.isnan(cell)
            if np.count_nonzero(both) < floor:
                continue
            here = window[both] - window[both].mean()
            there = cell[both] - cell[both].mean()
            spread = np.sqrt(float((here**2).sum()) * float((there**2).sum()))
            if spread > 0.0:
                out[i, j] = float((here * there).sum() / spread)
    return out


def test_the_fast_correlation_agrees_with_the_definition() -> None:
    rng = np.random.default_rng(20260808)
    other = rng.normal(size=(14, 13))
    cell = rng.normal(size=(5, 4))
    other[np.asarray(rng.random(other.shape)) < 0.2] = np.nan
    cell[2, 1] = np.nan
    cell[0, 3] = np.nan

    fast, overlap = correlate(cell, other, 0.5)
    slow = by_definition(cell, other, 0.5)

    assert fast.shape == slow.shape
    # Where one route declines to answer and the other does not, the two do not
    # agree about which placements were scored, which no comparison of the
    # values would show.
    assert np.array_equal(np.isfinite(fast), np.isfinite(slow))
    scored = np.isfinite(slow)
    assert_close(
        fast[scored],
        slow[scored],
        what="the masked correlation against its definition",
        atol=1e-12,
    )
    assert int(overlap.max()) <= cell.size


def test_the_correlation_of_a_cell_with_itself_is_one() -> None:
    rng = np.random.default_rng(1)
    cell = rng.normal(size=(6, 7))

    values, _ = correlate(cell, cell, 1.0)

    assert values.shape == (1, 1)
    assert_close(values[0, 0], 1.0, what="a cell correlated against itself", atol=1e-12)


def a_shifted_pair(
    scale: float, *, size: int = 8, margin: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """A cell and a surface carrying it, both offset from zero by ``scale``.

    The offset is what puts the arithmetic under strain: every sum the
    correlation is built from grows with it while the quantity being measured
    does not, so the subtraction that recovers a variance loses digits.
    """
    rng = np.random.default_rng(20260808)
    cell = scale + rng.normal(size=(size, size))
    other = scale + rng.normal(size=(size + 2 * margin, size + 2 * margin))
    other[margin : margin + size, margin : margin + size] = cell
    return cell, other


def test_a_correlation_that_rounded_outside_its_range_is_brought_back_in() -> None:
    # Heights a hundred times their own variation away from zero, which is an
    # ordinary unlevelled surface. The subtraction loses a few digits and the
    # correlation lands just outside one, which is a rounding error rather than
    # a broken computation and is brought back in. A number above one in a
    # report is a number a reader cannot make sense of.
    cell, other = a_shifted_pair(100.0)

    values, _ = correlate(cell, other, 1.0)

    assert float(np.nanmax(values)) <= 1.0
    assert_close(
        float(np.nanmax(values)), 1.0, what="the correlation of a cell against itself", atol=1e-9
    )


def test_an_overlap_that_is_a_whole_number_is_treated_as_one() -> None:
    # The overlap is counted through the same transform as everything else, so a
    # count that is 256 by construction arrives as 255.99999999999997. Compared
    # against a floor of 256 without rounding, a placement covering the whole
    # cell is discarded for being one part in a thousand million short of
    # covering the whole cell.
    rng = np.random.default_rng(3)
    cell = rng.normal(size=(16, 16))

    values, overlap = correlate(cell, cell, 1.0)

    assert values.shape == (1, 1)
    assert np.isfinite(values[0, 0])
    assert int(overlap[0, 0]) == cell.size


def test_an_overlap_with_nothing_varying_in_it_has_no_correlation() -> None:
    # A correlation needs both sides to vary. Where one does not, the variance
    # is a subtraction of two sums that agree to every digit they have, and what
    # comes out is the arithmetic's own noise. Dividing by the square root of
    # that produces a correlation between two patterns that are not there.
    flat = np.full((5, 5), 7.0)
    rng = np.random.default_rng(4)
    other = rng.normal(size=(9, 9))

    values, _ = correlate(flat, other, 1.0)

    assert not np.any(np.isfinite(values))


def test_the_grid_covers_every_sample_of_the_surface_exactly_once() -> None:
    # A remainder dropped rather than distributed is a strip of face no cell
    # covers, and on a scan whose edge has been trimmed that strip is surface.
    surface = as_surface(np.arange(37 * 41, dtype=np.float64).reshape(37, 41))

    seen = np.zeros(surface.shape, dtype=np.int64)
    for cell in divide(surface, CellParameters(grid=5, minimum_valid=0.5)):
        rows, columns = cell.shape
        seen[cell.top : cell.top + rows, cell.left : cell.left + columns] += 1

    assert int(seen.min()) == 1
    assert int(seen.max()) == 1


def test_a_matching_pair_reports_the_displacement_it_was_built_with() -> None:
    # The clause asking for high correlation at the known offset. The generator
    # was told to move the striae by SHIFT columns and no interpolation was
    # introduced, so the correlation at that placement is one rather than nearly
    # one, and the assertion is against the construction.
    subject, reference = a_matching_pair()
    settings = CellParameters(grid=4, minimum_valid=0.5)

    reachable = [
        cell for cell in divide(subject, settings) if cell.left + SHIFT + cell.shape[1] <= COLUMNS
    ]
    matches = {(match.row, match.column): match for match in compare(subject, reference, settings)}

    assert len(reachable) == 12
    for cell in reachable:
        match = matches[(cell.row, cell.column)]
        assert match.across == SHIFT
        assert_close(
            match.correlation,
            1.0,
            what=f"the best correlation of cell {cell.row},{cell.column}",
            atol=1e-9,
        )


def test_the_correlation_falls_away_from_the_placement_that_matches() -> None:
    # The clause asking for low correlation elsewhere. Half a striation spacing
    # away the pattern is in antiphase, which is the strongest statement
    # available about a periodic surface: the correlation is not merely small
    # there, it is the negative of itself.
    subject, reference = a_matching_pair()
    settings = CellParameters(grid=4, minimum_valid=0.5)
    cell = next(item for item in divide(subject, settings) if (item.row, item.column) == (1, 1))

    values, _ = correlate(cell.heights, np.asarray(reference.heights), 0.5)
    along = values[cell.top]
    half_a_spacing = round(STRIAE_SPACING_UM / SPACING_UM / 2)

    assert_close(along[cell.left + SHIFT], 1.0, what="the correlation where it matches", atol=1e-9)
    assert float(along[cell.left + SHIFT + half_a_spacing]) < -0.9
    assert float(along[cell.left + SHIFT - half_a_spacing]) < -0.9


def test_one_cell_s_correlation_does_not_separate_a_match_from_a_non_match() -> None:
    # Worth asserting rather than assuming, because it is the reason the method
    # counts cells that agree on a displacement instead of thresholding a
    # correlation, and a reader meeting a correlation of 0.99 in an output would
    # otherwise take it for a match. Both numbers here were measured by running
    # this test.
    settings = CellParameters(grid=4, minimum_valid=0.5)
    subject, reference = a_matching_pair()
    first, second = non_matching_pair(parameters())
    unrelated = compare(
        as_surface(np.asarray(first.heights_um)),
        as_surface(np.asarray(second.heights_um)),
        settings,
    )
    matching = compare(subject, reference, settings)

    assert max(match.correlation for match in unrelated) > 0.98

    # What does separate them is agreement. The pair built from one source puts
    # most of its cells on one displacement; the pair built from two sources
    # spreads them.
    def largest_agreement(matches: tuple[object, ...]) -> int:
        displacements = [match.across for match in matches]  # type: ignore[attr-defined]
        return max(displacements.count(value) for value in set(displacements))

    assert largest_agreement(matching) == 12
    assert largest_agreement(unrelated) == 4


def test_substituting_for_a_missing_sample_moves_the_correlation() -> None:
    # The clause asking for the difference as a number. Filling the missing
    # samples with zero and correlating as though the array were complete is the
    # usual shortcut. Measured here at the placement the surfaces actually match
    # at, so the two treatments are compared on one placement rather than on
    # whichever placement each of them liked best.
    subject, reference = a_matching_pair(edge_dropout_fraction=0.12)
    settings = CellParameters(grid=4, minimum_valid=0.3)
    other = np.asarray(reference.heights)
    filled_other = np.where(np.isnan(other), 0.0, other)

    differences = []
    for cell in divide(subject, settings):
        if cell.measured < 0.3 * cell.heights.size:
            continue
        if cell.left + SHIFT + cell.shape[1] > COLUMNS:
            continue
        at = (cell.top, cell.left + SHIFT)
        explicit, _ = correlate(cell.heights, other, 0.3)
        substituted, _ = correlate(
            np.where(np.isnan(cell.heights), 0.0, cell.heights), filled_other, 0.3
        )
        differences.append((float(explicit[at]), float(substituted[at])))

    assert len(differences) == 10
    for explicit_value, _ in differences:
        assert_close(
            explicit_value, 1.0, what="the correlation where the surfaces match", atol=1e-9
        )

    largest = max(explicit_value - filled for explicit_value, filled in differences)
    smallest = min(explicit_value - filled for explicit_value, filled in differences)
    # Cells that lie entirely inside the measured region meet no missing sample
    # and the two treatments agree exactly. Cells that straddle the missing edge
    # do not, and there the shortcut takes 0.0477 off a correlation of one.
    assert_close(smallest, 0.0, what="the difference where no sample is missing", atol=1e-12)
    assert_close(
        largest, 0.047732, what="the difference where the cell meets missing samples", atol=5e-6
    )


def test_the_grid_and_the_threshold_reach_the_manifest() -> None:
    # The clause asking for both parameters in the manifest. Recorded through
    # the run record rather than asserted on the record object, so what is
    # checked is the text a reader will have.
    registry = Registry()
    registry.register(RobustGaussianBandpass())
    settings = CellParameters(grid=6, minimum_valid=0.35)
    subject, _ = a_matching_pair()

    _, manifest = record_run(
        role="input",
        surface=subject,
        profile=ProfileRecord(name="a-profile", version="1"),
        chain=[
            Step(
                identifier="bandpass",
                parameters=BandpassParameters(
                    short_cutoff=20.0, long_cutoff=120.0, robust_tuning=None, robust_passes=None
                ),
            )
        ],
        registry=registry,
        seed=0,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
        comparison=record(settings),
    )

    text = manifest.to_text()
    assert f'"method": "{METHOD}"' in text
    assert f'"version": "{VERSION}"' in text
    assert '"grid": 6' in text
    assert '"minimum_valid": 0.35' in text


def test_a_run_that_compared_nothing_says_so_rather_than_leaving_the_field_out() -> None:
    registry = Registry()
    registry.register(RobustGaussianBandpass())
    subject, _ = a_matching_pair()

    _, manifest = record_run(
        role="input",
        surface=subject,
        profile=ProfileRecord(name="a-profile", version="1"),
        chain=[
            Step(
                identifier="bandpass",
                parameters=BandpassParameters(
                    short_cutoff=20.0, long_cutoff=120.0, robust_tuning=None, robust_passes=None
                ),
            )
        ],
        registry=registry,
        seed=0,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
    )

    assert '"comparison": null' in manifest.to_text()


@pytest.mark.parametrize("grid", [0, -1, True, 2.5])
def test_a_grid_that_is_not_a_count_of_cells_is_refused(grid: object) -> None:
    surface = as_surface(np.zeros((8, 8)))

    with pytest.raises(ValueError, match="whole number of cells"):
        divide(surface, CellParameters(grid=grid, minimum_valid=0.5))  # type: ignore[arg-type]


def test_a_grid_finer_than_the_surface_is_refused() -> None:
    surface = as_surface(np.zeros((4, 9)))

    with pytest.raises(ValueError, match="no samples in it"):
        divide(surface, CellParameters(grid=5, minimum_valid=0.5))


@pytest.mark.parametrize("minimum", [0.0, -0.1, 1.5, float("nan")])
def test_a_threshold_outside_its_range_is_refused(minimum: float) -> None:
    surface = as_surface(np.zeros((8, 8)))

    with pytest.raises(ValueError, match="above zero and at most one"):
        divide(surface, CellParameters(grid=2, minimum_valid=minimum))


def test_a_threshold_that_is_not_a_number_is_refused() -> None:
    surface = as_surface(np.zeros((8, 8)))

    with pytest.raises(TypeError, match="must be a number"):
        divide(surface, CellParameters(grid=2, minimum_valid=True))  # type: ignore[arg-type]


def test_a_cell_larger_than_the_surface_it_is_searched_on_is_refused() -> None:
    with pytest.raises(ValueError, match="does not fit inside"):
        correlate(np.zeros((5, 5)), np.zeros((4, 9)), 0.5)


def test_a_one_dimensional_array_is_refused() -> None:
    with pytest.raises(ValueError, match="two dimensional"):
        correlate(np.zeros(5), np.zeros((4, 9)), 0.5)


def test_two_surfaces_in_different_units_are_not_compared() -> None:
    # The expensive error available in this project is a length scale nobody
    # checked.
    subject, reference = a_matching_pair()
    other = Surface(
        heights=np.asarray(reference.heights),
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MILLIMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )

    with pytest.raises(ValueError, match="two length scales"):
        compare(subject, other, CellParameters(grid=4, minimum_valid=0.5))


def test_two_surfaces_with_opposite_orientations_are_not_compared() -> None:
    subject, reference = a_matching_pair()
    other = Surface(
        heights=np.asarray(reference.heights),
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_UP,
        source="synthetic",
    )

    with pytest.raises(ValueError, match="wrong sign"):
        compare(subject, other, CellParameters(grid=4, minimum_valid=0.5))


def test_two_surfaces_at_different_sampling_intervals_are_not_compared() -> None:
    subject, reference = a_matching_pair()
    other = as_surface(np.asarray(reference.heights), spacing=SPACING_UM / 2.0)

    with pytest.raises(ValueError, match="means two things at once"):
        compare(subject, other, CellParameters(grid=4, minimum_valid=0.5))


def test_a_surface_with_no_usable_cell_is_refused() -> None:
    subject, reference = a_matching_pair()
    empty = as_surface(np.full(subject.shape, np.nan))

    with pytest.raises(ValueError, match="nothing to correlate"):
        compare(empty, reference, CellParameters(grid=4, minimum_valid=0.5))


def test_a_cell_with_no_placement_that_overlaps_enough_is_dropped() -> None:
    # Two surfaces whose measured regions barely meet. The cells exist, the
    # placements exist, and none of them overlaps enough to be scored, which is
    # a different state from a low correlation and is not reported as one.
    subject, reference = a_matching_pair()
    hollow = np.full(reference.shape, np.nan)
    hollow[:4, :4] = np.asarray(reference.heights)[:4, :4]

    with pytest.raises(ValueError, match="not one of them came back"):
        compare(subject, as_surface(hollow), CellParameters(grid=4, minimum_valid=0.9))
