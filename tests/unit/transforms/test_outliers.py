"""Outlier rejection, against a surface whose spikes were put there by this file.

The count of injected spikes is known because it was asked for, so "removed the
spikes" is a comparison against a construction rather than against a judgement
about what looks like a spike.

Every count quoted here was measured by running the step at this commit, and
every refusal was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.outliers import OutlierParameters, RejectOutliers
from gutachten.transforms.pipeline import OrderingError, Step, check_chain
from gutachten.transforms.registry import REGISTRY, Registry
from tests.support.tolerance import assert_close
from tests.unit.transforms.declared_example import Scale, ScaleParameters
from tests.unit.transforms.undeclared_example import Clip, ClipParameters

ROWS = 48
COLUMNS = 48
SPACING_UM = 4.0
#: How many samples this file lifts out of the surface, and by how much.
SPIKES = 25
SPIKE_HEIGHT_UM = 60.0
ROBUST = "median-absolute-deviation"
ORDINARY = "standard-deviation"


def spike_positions() -> np.ndarray:
    """Where the spikes go, drawn once from a seeded generator."""
    drawn = np.random.default_rng(7).choice(ROWS * COLUMNS, size=SPIKES, replace=False)
    flat = np.zeros(ROWS * COLUMNS, dtype=bool)
    flat[drawn] = True
    return flat.reshape(ROWS, COLUMNS)


def a_spiked_surface() -> tuple[Surface, np.ndarray]:
    """A generated surface with ``SPIKES`` samples lifted, and where they are."""
    generated = generate(
        SurfaceParameters(
            rows=ROWS,
            columns=COLUMNS,
            pixel_spacing_um=SPACING_UM,
            seed=20260808,
        )
    )
    heights = np.array(generated.heights_um, dtype=np.float64, copy=True)
    spikes = spike_positions()
    heights[spikes] += SPIKE_HEIGHT_UM
    return as_surface(heights), spikes


def as_surface(heights: np.ndarray) -> Surface:
    return Surface(
        heights=heights,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def settings(criterion: str, threshold: float, neighbourhood: float | None) -> OutlierParameters:
    return OutlierParameters(criterion=criterion, threshold=threshold, neighbourhood=neighbourhood)


def rejected_count(result: Surface) -> int:
    return int(dict(result.provenance[-1].outcomes)["rejected_samples"])  # type: ignore[arg-type]


def test_the_surface_carries_the_spikes_this_file_put_in_it() -> None:
    # Asserted rather than assumed. Every count below is stated against this
    # construction, so a generator or a draw that put them somewhere else would
    # leave those tests passing against the wrong surface.
    surface, spikes = a_spiked_surface()

    assert int(np.count_nonzero(spikes)) == SPIKES
    assert not np.any(surface.missing)


def test_the_stated_threshold_removes_every_spike_and_a_looser_one_removes_fewer() -> None:
    surface, spikes = a_spiked_surface()

    tight = RejectOutliers().apply(surface, settings(ROBUST, 3.0, None))
    loose = RejectOutliers().apply(surface, settings(ROBUST, 6.0, None))

    # Both find all twenty five, measured at this commit.
    assert int(np.count_nonzero(tight.missing & spikes)) == SPIKES
    assert int(np.count_nonzero(loose.missing & spikes)) == SPIKES
    # And the tight one takes ninety three samples of real surface with them,
    # which is the cost the sweep exists to put a number on: 118 against 25.
    assert rejected_count(tight) == 118
    assert rejected_count(loose) == SPIKES
    assert rejected_count(loose) < rejected_count(tight)


def test_the_neighbourhood_and_the_whole_surface_are_different_measurements() -> None:
    # Two routes through the step, and a test that only ran one of them would
    # pass for an implementation that ignored the parameter.
    surface, spikes = a_spiked_surface()

    whole = RejectOutliers().apply(surface, settings(ROBUST, 3.0, None))
    nearby = RejectOutliers().apply(surface, settings(ROBUST, 3.0, 12.0))

    # 118 against 29, measured at this commit. Compared against its own
    # surroundings a striation is not unusual; compared against the whole
    # surface it is, because the surface's spread is dominated by the form.
    assert rejected_count(whole) == 118
    assert rejected_count(nearby) == 29
    assert int(np.count_nonzero(nearby.missing & spikes)) == SPIKES


def test_the_criterion_the_spikes_inflate_lets_them_through_and_the_robust_one_does_not() -> None:
    # This is why the weaker criterion is offered at all. A sweep that could not
    # run it could not report what the robust one is worth.
    surface, spikes = a_spiked_surface()

    robust = RejectOutliers().apply(surface, settings(ROBUST, 12.0, None))
    ordinary = RejectOutliers().apply(surface, settings(ORDINARY, 12.0, None))

    # Measured at this commit: all twenty five against none. Twenty five samples
    # out of two thousand raise the standard deviation until twelve of them
    # clears the very spikes it was set against.
    assert int(np.count_nonzero(robust.missing & spikes)) == SPIKES
    assert rejected_count(ordinary) == 0


def test_a_rejected_sample_becomes_missing_and_is_never_replaced() -> None:
    # Detecting a hole and filling one are two decisions, and a step that made
    # both would record one.
    surface, _ = a_spiked_surface()

    result = RejectOutliers().apply(surface, settings(ROBUST, 3.0, None))

    gone = result.missing & ~surface.missing
    assert np.all(np.isnan(result.heights[gone]))
    kept = ~result.missing
    assert_close(
        result.heights[kept],
        surface.heights[kept],
        what="the heights this step kept",
        # Exactly zero. The step marks samples missing and computes nothing, so
        # a surviving height that moved at all is a defect and not drift.
        atol=0.0,
    )
    assert result.shape == surface.shape


def test_the_count_of_rejected_samples_is_recorded_with_what_it_was_taken_from() -> None:
    # A count on its own does not say how much surface was discarded. The
    # denominator is what turns it into a proportion a sweep can report.
    surface, _ = a_spiked_surface()

    result = RejectOutliers().apply(surface, settings(ROBUST, 3.0, None))

    entry = result.provenance[-1]
    assert entry.name == "reject-outliers"
    assert dict(entry.parameters) == {
        "criterion": ROBUST,
        "neighbourhood": None,
        "threshold": 3.0,
    }
    assert dict(entry.outcomes) == {
        "measured_samples": ROWS * COLUMNS,
        "rejected_samples": 118,
    }
    # And it agrees with the surface it describes, rather than being a number
    # the step wrote down beside what it did.
    assert int(np.count_nonzero(result.missing)) == 118


def test_the_samples_already_missing_are_not_counted_as_rejected() -> None:
    # A step that reported the whole mask as its own work would tell a sweep
    # that a threshold discarded surface an earlier step had already removed.
    surface, _ = a_spiked_surface()
    holed = np.array(surface.heights, dtype=np.float64, copy=True)
    holed[: ROWS // 4, :] = np.nan

    result = RejectOutliers().apply(as_surface(holed), settings(ROBUST, 3.0, None))

    entry = dict(result.provenance[-1].outcomes)
    assert entry["measured_samples"] == ROWS * COLUMNS - (ROWS // 4) * COLUMNS
    newly = int(np.count_nonzero(result.missing & ~np.isnan(holed)))
    assert entry["rejected_samples"] == newly


def test_a_window_with_nothing_measured_in_it_leaves_its_centre_alone() -> None:
    # The comparison is against not-a-number there, which is false, so this
    # holds without a branch saying so. It is asserted because the day somebody
    # rewrites the comparison is the day it stops holding silently.
    # A band three columns wide against a neighbourhood one sample either side,
    # so the window centred on the middle column holds no measured sample at
    # all. Narrower and every window still catches a neighbour, and the case
    # this is about never arises.
    heights = np.zeros((ROWS, COLUMNS))
    heights[:, COLUMNS // 2 - 1 : COLUMNS // 2 + 2] = np.nan
    heights[ROWS // 2, 0] = SPIKE_HEIGHT_UM

    result = RejectOutliers().apply(as_surface(heights), settings(ROBUST, 3.0, SPACING_UM))

    # The empty windows judge nothing, and the spike three columns away from
    # them is still found, so the step did not simply stop.
    assert bool(result.missing[ROWS // 2, 0])
    assert rejected_count(result) == 1


def test_a_criterion_this_step_does_not_know_is_refused_naming_what_it_takes() -> None:
    surface, _ = a_spiked_surface()

    with pytest.raises(ValueError, match="not a criterion this step knows; it takes one of"):
        RejectOutliers().apply(surface, settings("MAD", 3.0, None))


def test_a_threshold_that_is_not_a_positive_finite_count_of_spreads_is_refused() -> None:
    surface, _ = a_spiked_surface()

    for threshold in (0.0, -3.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite number of spreads"):
            RejectOutliers().apply(surface, settings(ROBUST, threshold, None))


def test_a_threshold_that_is_not_a_number_is_refused() -> None:
    # `True` is an `int` to Python and would be read as a threshold of one.
    surface, _ = a_spiked_surface()

    with pytest.raises(TypeError, match="must be a number of spreads"):
        RejectOutliers().apply(surface, settings(ROBUST, True, None))  # type: ignore[arg-type]


def test_a_setting_that_rejects_every_measured_sample_is_refused() -> None:
    # Every number after this step would be taken over an empty array, which
    # most reductions answer with not-a-number rather than with an error.
    # Two heights in equal numbers. The median falls between them, so no sample
    # sits at the centre, and every one of them is a whole median absolute
    # deviation away from it. Half a deviation therefore rejects the lot.
    two_valued = np.where((np.indices((ROWS, COLUMNS)).sum(axis=0) % 2).astype(bool), 1.0, -1.0)

    with pytest.raises(ValueError, match="rejects all"):
        RejectOutliers().apply(as_surface(two_valued), settings(ROBUST, 0.5, None))


def test_a_surface_with_no_measured_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="every sample of this surface is missing"):
        RejectOutliers().apply(
            as_surface(np.full((ROWS, COLUMNS), np.nan)), settings(ROBUST, 3.0, None)
        )


def test_a_neighbourhood_that_is_not_a_positive_finite_length_is_refused() -> None:
    surface, _ = a_spiked_surface()

    for radius in (0.0, -4.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="must be a positive finite length"):
            RejectOutliers().apply(surface, settings(ROBUST, 3.0, radius))


def test_a_neighbourhood_that_reaches_no_neighbour_is_refused() -> None:
    # Silently comparing every sample against a line through itself is the
    # failure. A sweep is entitled to assume a neighbourhood it asked for
    # existed.
    surface, _ = a_spiked_surface()

    with pytest.raises(ValueError, match="would reach no neighbour"):
        RejectOutliers().apply(surface, settings(ROBUST, SPACING_UM / 4, None if False else 0.4))


def test_the_two_axes_are_measured_at_their_own_spacings() -> None:
    # The fixture above is square in both spacing and shape, so a step reading
    # one spacing for both axes would pass every other test in this file.
    heights = np.zeros((ROWS, COLUMNS))
    anisotropic = Surface(
        heights=heights,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM / 4,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )

    # Half a row spacing reaches no neighbouring row, and two column spacings
    # across. One physical length, two answers, and the refusal names the row
    # axis rather than passing because the column axis was satisfied.
    with pytest.raises(ValueError, match="rounds to 0 rows and 2 columns"):
        RejectOutliers().apply(anisotropic, settings(ROBUST, 3.0, SPACING_UM / 2))


def test_the_step_refuses_a_record_that_is_not_its_own_before_reading_a_field() -> None:
    # `record_for` refuses the same mistake on the way out, by which point the
    # step has done its work.
    surface, _ = a_spiked_surface()

    class NotOurs:
        criterion = ROBUST
        threshold = 3.0
        neighbourhood = None

    with pytest.raises(TypeError, match="rather than OutlierParameters, so nothing here has read"):
        RejectOutliers().apply(surface, NotOurs())  # type: ignore[arg-type]


def test_rejecting_after_a_filtering_step_is_refused_by_the_pipeline() -> None:
    # A bandpass spreads a spike over the width of its kernel, so a rejection
    # made afterwards finds a smear and takes the surface around it as well.
    registry = Registry()
    registry.register(Scale())
    registry.register(Clip())
    registry.register(RejectOutliers())
    chain = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=1.0)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=1.0)),
        Step(identifier="reject-outliers", parameters=settings(ROBUST, 3.0, None)),
    ]

    with pytest.raises(OrderingError, match="refuses a surface that is filtered"):
        check_chain(chain, registry)


def test_importing_the_package_is_what_registers_the_step() -> None:
    # In a fresh interpreter, because this module has already imported the step
    # directly and that import is what registers it.
    program = (
        "import gutachten.transforms; "
        "from gutachten.transforms.registry import REGISTRY; "
        "print(','.join(REGISTRY.identifiers()))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "reject-outliers" in completed.stdout.strip().split(",")
    assert REGISTRY["reject-outliers"].version == "1"
