"""The registration search, against a pair whose rotation and shift are known.

The ground truth here is a construction rather than a recorded output. A pattern
that varies in one direction only cannot say how far it moved along that
direction, so a single striation field would leave half of the displacement
undetermined and the test would have to say so. Two striation fields crossed at
a right angle do not have that gap, and the generator applies a rotation and a
translation to the coordinates it evaluates the pattern on rather than to a
finished image, so the sum of two fields carried through the same rotation and
translation is exactly the rigid transform of their sum, with no interpolation
anywhere in the truth.

Every refusal here was removed in turn and the suite watched go red.
"""

from __future__ import annotations

import numpy as np
import pytest

from gutachten.compare.cells import CellParameters
from gutachten.compare.cells import compare as compare_cells
from gutachten.compare.register import (
    METHOD,
    VERSION,
    RegistrationParameters,
    angles,
    record,
    register,
    rotate_heights,
)
from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import EnvironmentRecord, ProfileRecord, record_run
from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.bandpass import BandpassParameters, RobustGaussianBandpass
from gutachten.transforms.pipeline import Step
from gutachten.transforms.registry import Registry
from tests.support.tolerance import assert_close

SIZE = 192
SPACING_UM = 4.0

#: The two striation fields the ground truth is built from. They are crossed at
#: a right angle and given different spacings, so the crossed pattern is not
#: periodic in either direction at the period of one of them.
ALONG_SPACING_UM = 40.0
ACROSS_SPACING_UM = 56.0
ALONG_SOURCE = 11
ACROSS_SOURCE = 22

#: What the pair carries. The rotation is a whole number of steps of the search
#: below, because a search that cannot visit the truth is a search this test
#: would be measuring the step size of.
ROTATION_DEG = 6.0
TRANSLATION_PX = (2.0, -3.0)

#: A cell of the subject that moved by ``TRANSLATION_PX`` is found on the
#: reference that far the other way, so the displacement a match reports is the
#: negative of the one the generator applied.
EXPECTED_DOWN = -int(TRANSLATION_PX[0])
EXPECTED_ACROSS = -int(TRANSLATION_PX[1])


def field_parameters(*, striae_angle_deg: float, striae_spacing_um: float) -> SurfaceParameters:
    """One striation field and nothing else on it.

    The form, the firing pin impression and the drag mark are switched off
    because the generator does not move them when it moves the striae, so a
    surface carrying them is not a rigid transform of the surface it is supposed
    to be one of. The noise is off for the same reason: it is drawn per surface
    and does not transform.
    """
    return SurfaceParameters(
        rows=SIZE,
        columns=SIZE,
        pixel_spacing_um=SPACING_UM,
        striae_angle_deg=striae_angle_deg,
        striae_spacing_um=striae_spacing_um,
        form_depth_um=0.0,
        firing_pin_depth_um=0.0,
        drag_mark_depth_um=0.0,
        noise_um=0.0,
        seed=20260809,
    )


def crossed_heights(
    *, rotation_deg: float = 0.0, translation_px: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Two striation fields at a right angle, both carried by the same transform."""
    along = generate(
        field_parameters(striae_angle_deg=0.0, striae_spacing_um=ALONG_SPACING_UM),
        source_id=ALONG_SOURCE,
        rotation_deg=rotation_deg,
        translation_px=translation_px,
    )
    across = generate(
        field_parameters(striae_angle_deg=90.0, striae_spacing_um=ACROSS_SPACING_UM),
        source_id=ACROSS_SOURCE,
        rotation_deg=rotation_deg,
        translation_px=translation_px,
    )
    heights: np.ndarray = np.asarray(along.heights_um) + np.asarray(across.heights_um)
    return heights


def as_surface(heights: np.ndarray, *, source: str = "synthetic") -> Surface:
    return Surface(
        heights=heights,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source=source,
    )


def a_transformed_pair() -> tuple[Surface, Surface]:
    """The subject, which carries the transform, and the reference, which does not."""
    subject = as_surface(
        crossed_heights(rotation_deg=ROTATION_DEG, translation_px=TRANSLATION_PX),
        source="subject",
    )
    reference = as_surface(crossed_heights(), source="reference")
    return subject, reference


def settings(**overrides: object) -> RegistrationParameters:
    values: dict[str, object] = {
        "grid": 3,
        "minimum_valid": 0.5,
        "rotation_range_deg": 10.0,
        "rotation_step_deg": 2.0,
        "translation_limit": 6,
    }
    values.update(overrides)
    return RegistrationParameters(**values)  # type: ignore[arg-type]


def test_the_rotation_and_the_displacement_are_recovered_on_every_cell() -> None:
    # The clause asking for a pair with a known translation and rotation to be
    # registered to within a stated tolerance of the truth. The tolerance on the
    # rotation is zero because the truth is one of the angles searched, and the
    # tolerance on the displacement is zero because a displacement is a whole
    # number of samples here and a match either lands on it or does not.
    subject, reference = a_transformed_pair()

    found = register(subject, reference, settings())

    assert len(found.matches) == 9
    assert_close(
        [match.rotation_deg for match in found.matches],
        [ROTATION_DEG] * 9,
        what="the orientation each cell registered at",
        atol=0.0,
    )
    assert [(match.down, match.across) for match in found.matches] == [
        (EXPECTED_DOWN, EXPECTED_ACROSS)
    ] * 9
    assert_close(
        [match.correlation for match in found.matches],
        [1.0] * 9,
        what="the correlation each cell registered at",
        atol=0.05,
    )


def test_the_bound_on_the_translation_is_what_makes_the_edge_cells_reachable() -> None:
    # What the bound is worth, as a number, against the same pair. The
    # correlation is taken over placements where a tile lies wholly inside the
    # other array, so without the bound a tile cut from the edge cannot be
    # placed where it actually moved to. The unbounded route is the cell
    # correlation run directly on the reference turned to the right angle, which
    # is what this module would be if it did not pad.
    subject, reference = a_transformed_pair()
    turned = as_surface(rotate_heights(np.asarray(reference.heights), ROTATION_DEG))

    unbounded = compare_cells(subject, turned, CellParameters(grid=3, minimum_valid=0.5))
    on_the_truth = [
        match
        for match in unbounded
        if (match.down, match.across) == (EXPECTED_DOWN, EXPECTED_ACROSS)
    ]

    assert len(unbounded) == 9
    assert len(on_the_truth) == 4

    bounded = register(subject, reference, settings())
    assert len(bounded.matches) == 9
    assert all(
        (match.down, match.across) == (EXPECTED_DOWN, EXPECTED_ACROSS) for match in bounded.matches
    )


def test_no_cell_is_reported_at_a_displacement_outside_the_declared_bound() -> None:
    # The bound the manifest records is the bound the search kept to. The pair
    # here has moved further than the limit allows, so the placement that
    # correlates best of all is outside it, and what has to come back is the
    # best placement inside. A search that padded the reference and then took
    # the maximum over everything would report the displacement it found at
    # three samples across under a record saying it looked no further than one.
    subject, reference = a_transformed_pair()
    limit = 1

    found = register(subject, reference, settings(translation_limit=limit))

    assert found.matches
    outside = [
        (match.down, match.across)
        for match in found.matches
        if abs(match.down) > limit or abs(match.across) > limit
    ]
    assert not outside, f"reported outside a bound of {limit} sample: {outside}"


def test_turning_a_field_agrees_with_the_pattern_the_generator_built_at_that_angle() -> None:
    # The sign convention, pinned against the construction rather than against
    # this module's own output. A flip here would report every registration the
    # wrong way round while every other test in this file still passed, because
    # they compare the search against a truth this same function produced.
    base = crossed_heights()
    truth = crossed_heights(rotation_deg=ROTATION_DEG)

    turned = rotate_heights(base, ROTATION_DEG)

    both = np.isfinite(turned) & np.isfinite(truth)
    assert int(np.count_nonzero(both)) > SIZE * SIZE // 2
    # The truth is evaluated analytically at every sample and the turned field
    # is interpolated between samples, so the two differ by the interpolation
    # error of a bilinear resampling of a pattern with ten samples to a period.
    # It is stated here rather than hidden in a default: 0.2 micrometres on a
    # crossed pattern whose peak to peak is 4.
    assert_close(
        turned[both],
        truth[both],
        what="a turned field against the pattern the generator built at that angle",
        atol=0.2,
    )


def test_turning_the_other_way_is_not_the_same_thing() -> None:
    # The near miss the test above exists for. A sign flip is one character and
    # it produces a field that is smooth, correctly shaped and turned the wrong
    # way, which nothing downstream would notice.
    base = crossed_heights()
    truth = crossed_heights(rotation_deg=ROTATION_DEG)

    wrong_way = rotate_heights(base, -ROTATION_DEG)

    both = np.isfinite(wrong_way) & np.isfinite(truth)
    worst = float(np.max(np.abs(wrong_way[both] - truth[both])))
    assert worst > 1.0


def test_a_resampled_sample_is_a_measurement_only_where_every_sample_it_drew_on_was() -> None:
    # Interpolating through a hole invents heights around its rim, and those
    # heights correlate. Here the hole grows by the reach of the interpolation
    # instead.
    heights = np.zeros((32, 32))
    heights[16, 16] = np.nan

    turned = rotate_heights(heights, 5.0)

    assert np.isnan(turned[16, 16])
    # The reach of a bilinear stencil is one sample, so the hole grows and does
    # not spread across the field.
    grown = int(np.count_nonzero(np.isnan(turned[8:24, 8:24])))
    assert 1 < grown <= 9


def test_turning_a_field_by_nothing_leaves_it_alone() -> None:
    rng = np.random.default_rng(20260809)
    heights = rng.normal(size=(24, 24))

    turned = rotate_heights(heights, 0.0)

    assert_close(turned, heights, what="a field turned by no angle", atol=0.0)


def test_the_angles_reach_both_ends_of_the_range_and_pass_through_zero() -> None:
    searched = angles(settings(rotation_range_deg=10.0, rotation_step_deg=2.5))

    assert len(searched) == 9
    assert searched == tuple(sorted(searched))
    # Built from a whole number of steps rather than by accumulating one, so the
    # ends are the range to the last digit and zero is exactly zero.
    assert searched[0] == -10.0
    assert searched[-1] == 10.0
    assert 0.0 in searched


def test_a_range_of_nothing_searches_one_orientation() -> None:
    # Not searching over rotation is a configuration a sweep has to be able to
    # visit, not a mistake, so it is reachable rather than refused.
    assert angles(settings(rotation_range_deg=0.0, rotation_step_deg=2.0)) == (0.0,)


def test_the_cost_is_the_cells_times_the_angles_and_is_the_same_on_two_runs() -> None:
    subject, reference = a_transformed_pair()
    parameters = settings()

    first = register(subject, reference, parameters)
    second = register(subject, reference, parameters)

    assert first.correlations == 9 * len(angles(parameters))
    assert first.correlations == 99
    assert first.angles_deg == second.angles_deg
    # The whole result, not a summary of it. A search that reported the same
    # count and a different registration would be a search whose answer moved.
    assert first.matches == second.matches


def test_a_cell_that_cannot_be_registered_is_absent_rather_than_reported_at_zero() -> None:
    # A subject whose measured surface sits in one corner. The cells with
    # nothing in them are not compared at all, and what comes back is shorter
    # than the grid rather than padded out with cells carrying a correlation
    # standing in for one they do not have.
    subject, reference = a_transformed_pair()
    holed = np.asarray(subject.heights).copy()
    holed[SIZE // 3 :, :] = np.nan
    subject = as_surface(holed, source="subject")

    found = register(subject, reference, settings())

    assert 0 < len(found.matches) < 9
    assert {(match.row, match.column) for match in found.matches} == {(0, 0), (0, 1), (0, 2)}


def test_the_search_parameters_reach_the_manifest() -> None:
    # The clause asking for the range and the step in the manifest. Recorded
    # through the run record rather than asserted on the record object, so what
    # is checked is the text a reader will have.
    registry = Registry()
    registry.register(RobustGaussianBandpass())
    subject, _ = a_transformed_pair()

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
        comparison=record(settings(grid=6, minimum_valid=0.35)),
    )

    text = manifest.to_text()
    assert f'"method": "{METHOD}"' in text
    assert f'"version": "{VERSION}"' in text
    assert '"grid": 6' in text
    assert '"minimum_valid": 0.35' in text
    assert '"rotation_range_deg": 10.0' in text
    assert '"rotation_step_deg": 2.0' in text
    assert '"translation_limit": 6' in text


def test_a_range_the_step_does_not_divide_is_refused() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        angles(settings(rotation_range_deg=5.0, rotation_step_deg=2.0))


def test_a_negative_range_is_refused() -> None:
    with pytest.raises(ValueError, match="half width"):
        angles(settings(rotation_range_deg=-4.0))


def test_a_range_past_a_half_turn_is_refused() -> None:
    with pytest.raises(ValueError, match="half turn"):
        angles(settings(rotation_range_deg=180.0, rotation_step_deg=1.0))


def test_a_step_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="positive number of degrees"):
        angles(settings(rotation_step_deg=0.0))


def test_a_step_wider_than_the_range_is_refused() -> None:
    with pytest.raises(ValueError, match="searches no orientation but zero"):
        angles(settings(rotation_range_deg=2.0, rotation_step_deg=5.0))


def test_a_range_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(TypeError, match="rotation_range_deg"):
        angles(settings(rotation_range_deg="ten"))


def test_a_step_that_is_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="finite number of degrees"):
        angles(settings(rotation_step_deg=float("inf")))


def test_a_translation_limit_that_is_not_a_whole_number_of_samples_is_refused() -> None:
    with pytest.raises(TypeError, match="whole number of samples"):
        angles(settings(translation_limit=2.5))


def test_a_translation_limit_given_as_a_boolean_is_refused() -> None:
    # ``True`` is an integer in Python and a limit of one sample is a plausible
    # setting, so this passes every check that asks only whether it is a whole
    # number.
    with pytest.raises(TypeError, match="whole number of samples"):
        angles(settings(translation_limit=True))


def test_a_negative_translation_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        angles(settings(translation_limit=-1))


def test_two_surfaces_that_do_not_measure_the_same_way_are_refused() -> None:
    # Delegated to the cell correlation, which holds the three conditions and
    # their messages. Asserted here because a search that skipped the check
    # would correlate two length scales and report a number about one.
    subject, reference = a_transformed_pair()
    in_millimetres = Surface(
        heights=np.asarray(reference.heights).copy(),
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MILLIMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="reference",
    )

    with pytest.raises(ValueError, match="two length scales"):
        register(subject, in_millimetres, settings())


def test_a_subject_with_no_usable_cell_is_refused() -> None:
    subject, reference = a_transformed_pair()
    empty = np.full_like(np.asarray(subject.heights), np.nan)
    empty[0, 0] = 1.0

    with pytest.raises(ValueError, match="nothing to register"):
        register(as_surface(empty, source="subject"), reference, settings())


def test_a_pair_that_meets_nowhere_inside_the_bound_is_refused() -> None:
    # Both surfaces are measured, both carry usable cells, and no placement
    # within the bound leaves enough of a cell over measured surface. A search
    # that returned an empty result here would be a comparison reporting no
    # cells rather than one saying it could not be made.
    subject, reference = a_transformed_pair()
    hollow = np.asarray(reference.heights).copy()
    hollow[:, :] = np.nan

    with pytest.raises(ValueError, match="not one of them came back"):
        register(subject, as_surface(hollow, source="reference"), settings())
