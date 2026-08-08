"""The surface type, and every construction it refuses.

The refusals are the point of this file. A surface that accepts a height array
with no unit, or with a spacing of zero, or with the caller still holding a
writeable reference to its heights, is an envelope that carries the same defects
a bare array carries and costs a class definition on top.

Each test here names one refusal. Deleting the corresponding check in
``src/gutachten/surface.py`` reddens exactly the test that names it, which is
the property that makes these guards rather than decoration.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gutachten.surface import AxisOrientation, LengthUnit, Surface, TransformRecord
from tests.support.tolerance import assert_close

# The tolerance for a value that was handed to the constructor and read back
# out. Nothing computes it and nothing rounds it, so anything other than zero
# would be admitting a change this suite is not looking for.
CARRIED_THROUGH = 0.0


def a_surface(**overrides: object) -> Surface:
    """A valid surface, with one thing changed at a time by the caller."""
    arguments: dict[str, object] = {
        "heights": np.zeros((4, 5), dtype=np.float64),
        "spacing_y": 4.0,
        "spacing_x": 2.5,
        "unit": LengthUnit.MICROMETRE,
        "orientation": AxisOrientation.Y_DOWN,
        "source": "test",
    }
    arguments.update(overrides)
    return Surface(**arguments)  # type: ignore[arg-type]


def test_a_surface_carries_its_spacing_unit_orientation_and_source() -> None:
    surface = a_surface()

    assert_close(surface.spacing_y, 4.0, what="row spacing as constructed", atol=CARRIED_THROUGH)
    assert_close(surface.spacing_x, 2.5, what="column spacing as constructed", atol=CARRIED_THROUGH)
    assert surface.unit is LengthUnit.MICROMETRE
    assert surface.orientation is AxisOrientation.Y_DOWN
    assert surface.source == "test"
    assert surface.provenance == ()
    assert surface.shape == (4, 5)


def test_a_surface_with_no_declared_unit_is_refused() -> None:
    with pytest.raises(TypeError, match="declared length unit"):
        a_surface(unit=None)


def test_a_surface_with_a_unit_that_is_not_a_length_unit_is_refused() -> None:
    with pytest.raises(TypeError, match="declared length unit"):
        a_surface(unit="um")


def test_a_surface_with_no_declared_orientation_is_refused() -> None:
    with pytest.raises(TypeError, match="declared axis orientation"):
        a_surface(orientation=None)


def test_a_surface_with_no_spacing_is_refused() -> None:
    with pytest.raises(TypeError, match="spacing_y must be a number"):
        a_surface(spacing_y=None)
    with pytest.raises(TypeError, match="spacing_x must be a number"):
        a_surface(spacing_x=None)


def test_a_spacing_that_is_a_boolean_is_refused() -> None:
    # `True` is an int in Python and would otherwise pass the numeric check and
    # arrive as a spacing of one, in whatever unit was declared.
    with pytest.raises(TypeError, match="spacing_y must be a number"):
        a_surface(spacing_y=True)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_spacing_that_is_not_a_positive_finite_distance_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="positive finite distance"):
        a_surface(spacing_x=bad)


def test_heights_that_are_not_an_array_are_refused() -> None:
    with pytest.raises(TypeError, match="heights must be a numpy array"):
        a_surface(heights=[[0.0, 1.0], [2.0, 3.0]])


@pytest.mark.parametrize("shape", [(6,), (2, 3, 4)])
def test_heights_that_are_not_two_dimensional_are_refused(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="two dimensional height field"):
        a_surface(heights=np.zeros(shape, dtype=np.float64))


def test_heights_with_no_samples_are_refused() -> None:
    with pytest.raises(ValueError, match="no samples"):
        a_surface(heights=np.zeros((0, 4), dtype=np.float64))


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_heights_that_are_not_float64_are_refused(dtype: type) -> None:
    with pytest.raises(TypeError, match="heights must be float64"):
        a_surface(heights=np.zeros((3, 3), dtype=dtype))


def test_heights_containing_an_infinity_are_refused() -> None:
    heights = np.zeros((3, 3), dtype=np.float64)
    heights[1, 1] = np.inf

    with pytest.raises(ValueError, match="infinity"):
        a_surface(heights=heights)


def test_a_surface_with_no_source_identity_is_refused() -> None:
    with pytest.raises(ValueError, match="source identity"):
        a_surface(source="")


def test_a_provenance_chain_that_is_not_transform_records_is_refused() -> None:
    with pytest.raises(TypeError, match="tuple of TransformRecord"):
        a_surface(provenance=("levelling",))
    with pytest.raises(TypeError, match="tuple of TransformRecord"):
        a_surface(provenance=[TransformRecord.of("levelling", "1")])


def test_missing_data_is_not_a_number_and_is_not_a_height_of_zero() -> None:
    heights = np.zeros((2, 3), dtype=np.float64)
    heights[0, 1] = np.nan

    surface = a_surface(heights=heights)

    assert surface.missing.tolist() == [[False, True, False], [False, False, False]]
    assert surface.observed.shape == (5,)
    assert_close(
        surface.observed,
        np.zeros(5),
        what="the measured heights beside a missing sample",
        atol=CARRIED_THROUGH,
    )


def test_a_field_cannot_be_reassigned() -> None:
    surface = a_surface()

    with pytest.raises(dataclasses.FrozenInstanceError):
        surface.spacing_y = 8.0  # type: ignore[misc]


def test_the_height_array_cannot_be_written_to() -> None:
    surface = a_surface()

    with pytest.raises(ValueError, match="read-only"):
        surface.heights[0, 0] = 1.0


def test_moving_the_array_that_was_handed_in_does_not_move_the_surface() -> None:
    heights = np.zeros((3, 3), dtype=np.float64)
    surface = a_surface(heights=heights)

    heights[1, 1] = 1000.0

    assert_close(
        surface.heights,
        np.zeros((3, 3)),
        what="surface heights after the caller's array was written to",
        atol=CARRIED_THROUGH,
    )


def test_a_transform_extends_the_provenance_chain_and_leaves_the_original_alone() -> None:
    surface = a_surface()
    record = TransformRecord.of("level", "2", model="plane")

    levelled = surface.with_transform(record, np.ones((4, 5), dtype=np.float64))

    assert levelled.provenance == (record,)
    assert surface.provenance == ()
    assert levelled.unit is surface.unit
    assert levelled.orientation is surface.orientation
    assert levelled.source == surface.source
    assert_close(
        levelled.spacing_y,
        surface.spacing_y,
        what="row spacing across a transform",
        atol=CARRIED_THROUGH,
    )

    twice = levelled.with_transform(TransformRecord.of("filter", "1"), levelled.heights)
    assert [entry.name for entry in twice.provenance] == ["level", "filter"]


def test_a_transform_that_is_not_a_record_is_refused() -> None:
    surface = a_surface()

    with pytest.raises(TypeError, match="must be a TransformRecord"):
        surface.with_transform("level", np.ones((4, 5), dtype=np.float64))  # type: ignore[arg-type]


def test_a_transform_record_needs_a_name_and_a_version() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        TransformRecord.of("", "1")
    with pytest.raises(ValueError, match="has no version"):
        TransformRecord.of("level", "")


def test_a_transform_record_sorts_its_parameters() -> None:
    record = TransformRecord.of("bandpass", "3", upper_um=250.0, lower_um=25.0)

    assert [key for key, _ in record.parameters] == ["lower_um", "upper_um"]


def test_a_transform_record_refuses_parameters_out_of_order() -> None:
    # Two runs writing one provenance chain in two orders are two records that
    # do not compare equal, which is the thing the sorting exists to prevent.
    with pytest.raises(ValueError, match="out of order"):
        TransformRecord(name="bandpass", version="3", parameters=(("upper", 1.0), ("lower", 2.0)))


def test_a_transform_record_refuses_a_parameter_named_twice() -> None:
    with pytest.raises(ValueError, match="names a parameter twice"):
        TransformRecord(name="bandpass", version="3", parameters=(("cut", 1.0), ("cut", 2.0)))


def test_a_record_carries_what_the_step_found_apart_from_what_it_was_told() -> None:
    # A parameter is an input a sweep varies and a re-run reproduces; an outcome
    # is a measurement of what the run did. A reader who cannot tell them apart
    # cannot use either.
    record = TransformRecord.of("reject-outliers", "1", threshold=3.0).with_outcomes(
        rejected=118, measured=2304
    )

    assert dict(record.parameters) == {"threshold": 3.0}
    assert dict(record.outcomes) == {"measured": 2304, "rejected": 118}
    assert [key for key, _ in record.outcomes] == ["measured", "rejected"]


def test_a_record_with_no_outcomes_says_so_by_carrying_none() -> None:
    # Most steps measure nothing worth recording, and an empty tuple is what
    # says that rather than an absent field a reader has to know about.
    assert TransformRecord.of("level", "1", model="plane").outcomes == ()


def test_a_transform_record_refuses_outcomes_out_of_order() -> None:
    with pytest.raises(ValueError, match="has outcomes out of order"):
        TransformRecord(
            name="reject-outliers", version="1", outcomes=(("rejected", 1), ("measured", 2))
        )


def test_a_transform_record_refuses_an_outcome_named_twice() -> None:
    with pytest.raises(ValueError, match="names a outcome twice"):
        TransformRecord(
            name="reject-outliers", version="1", outcomes=(("rejected", 1), ("rejected", 2))
        )


def test_a_name_cannot_be_both_a_parameter_and_an_outcome() -> None:
    # The failure is a reader who cannot tell whether a number was asked for or
    # found, in a record whose whole purpose is to answer that.
    with pytest.raises(ValueError, match="as both a parameter and an outcome"):
        TransformRecord(
            name="reject-outliers",
            version="1",
            parameters=(("threshold", 3.0),),
            outcomes=(("threshold", 4.0),),
        )


def test_outcomes_are_written_once_and_not_added_to() -> None:
    # Two answers to one question with nothing to say which of them the run
    # produced.
    written = TransformRecord.of("reject-outliers", "1", threshold=3.0).with_outcomes(rejected=1)

    with pytest.raises(ValueError, match="already records outcomes"):
        written.with_outcomes(rejected=2)


def test_every_length_unit_declares_its_size() -> None:
    # The near miss: a unit added to the enum and not to the size map. Without
    # this test the omission surfaces as a KeyError on the first file read in
    # that unit, which is a long way from where the mistake was made.
    missing: list[str] = []
    for unit in LengthUnit:
        try:
            size = unit.micrometres
        except KeyError:
            missing.append(unit.name)
        else:
            assert size > 0.0, f"{unit.name} declares a size of {size} micrometres"

    assert not missing, f"length units with no declared size in micrometres: {missing}"


def test_the_unit_sizes_are_the_sizes_they_claim() -> None:
    assert_close(
        [unit.micrometres for unit in LengthUnit],
        [0.001, 1.0, 1000.0, 1000000.0],
        what="micrometres per length unit",
        atol=CARRIED_THROUGH,
    )
