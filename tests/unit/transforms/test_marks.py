"""The mark masking, against a generated drag mark whose position is known.

The generator digs the groove from parameters, so where it is is known because
it was asked for. The extractor mark has no counterpart in the generator, so its
region is compared against the disc its own parameters describe, which is still
a comparison against a construction rather than against a judgement about what
looks like a mark.

Every count quoted here was measured by running the step at this commit, and
every refusal was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import EnvironmentRecord, ProfileRecord, record_run
from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.bandpass import BandpassParameters, RobustGaussianBandpass
from gutachten.transforms.level import LevelParameters, RemoveForm
from gutachten.transforms.marks import MarkParameters, MaskMarks
from gutachten.transforms.pipeline import OrderingError, Step, check_chain, run_chain
from gutachten.transforms.registry import REGISTRY, Registry
from tests.support.tolerance import assert_close

ROWS = 64
COLUMNS = 64
SPACING_UM = 4.0
DRAG_WIDTH_UM = 30.0
DRAG_DEPTH_UM = 6.0
#: Where the generator puts the groove, as a fraction of the row extent.
DRAG_FRACTION = 0.25
#: The same place in the coordinates this step works in, which is the field
#: centred on itself. Written out from the generator's own expression rather
#: than as the number it comes to, so a generator that moved it moves this too.
DRAG_CENTRE_UM = (DRAG_FRACTION - 0.5) * (ROWS * SPACING_UM)

EXTRACTOR_RADIUS_UM = 40.0
EXTRACTOR_DISTANCE_UM = 100.0
EXTRACTOR_ANGLE_DEG = 90.0


def a_surface(*, drag_depth_um: float = DRAG_DEPTH_UM) -> Surface:
    generated = generate(
        SurfaceParameters(
            rows=ROWS,
            columns=COLUMNS,
            pixel_spacing_um=SPACING_UM,
            drag_mark_width_um=DRAG_WIDTH_UM,
            drag_mark_position=DRAG_FRACTION,
            drag_mark_depth_um=drag_depth_um,
            noise_um=0.0,
            seed=20260808,
        )
    )
    return as_surface(np.asarray(generated.heights_um))


def as_surface(heights: np.ndarray) -> Surface:
    return Surface(
        heights=heights,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def coordinates() -> tuple[np.ndarray, np.ndarray]:
    down = (np.arange(ROWS, dtype=np.float64) - (ROWS - 1) / 2) * SPACING_UM
    across = (np.arange(COLUMNS, dtype=np.float64) - (COLUMNS - 1) / 2) * SPACING_UM
    y_um, x_um = np.meshgrid(down, across, indexing="ij")
    return np.asarray(y_um), np.asarray(x_um)


def settings(
    *,
    exclude_drag: bool,
    exclude_extractor: bool,
    drag_angle_deg: float = 0.0,
    drag_position: float = DRAG_CENTRE_UM,
) -> MarkParameters:
    return MarkParameters(
        drag_width=DRAG_WIDTH_UM,
        drag_position=drag_position,
        drag_angle_deg=drag_angle_deg,
        exclude_drag=exclude_drag,
        extractor_radius=EXTRACTOR_RADIUS_UM,
        extractor_distance=EXTRACTOR_DISTANCE_UM,
        extractor_angle_deg=EXTRACTOR_ANGLE_DEG,
        exclude_extractor=exclude_extractor,
    )


def test_the_generator_dug_the_groove_where_this_file_says_it_did() -> None:
    # Asserted rather than assumed. Every expectation below names
    # DRAG_CENTRE_UM, so a generator that put the groove somewhere else would
    # leave them passing against the wrong region.
    y_um, _ = coordinates()
    groove = np.abs(y_um - DRAG_CENTRE_UM) <= DRAG_WIDTH_UM / 2

    with_groove = np.asarray(a_surface().heights)
    without = np.asarray(a_surface(drag_depth_um=0.0).heights)

    assert_close(
        with_groove[groove] - without[groove],
        np.full(int(np.count_nonzero(groove)), -DRAG_DEPTH_UM),
        what="how much deeper the groove is than the same surface without it",
        # Exactly. The generator subtracts a constant inside the band.
        atol=0.0,
    )
    assert_close(
        with_groove[~groove],
        without[~groove],
        what="the surface outside the groove, with the groove and without it",
        atol=0.0,
    )


def test_the_setting_that_asks_for_the_drag_mark_removes_exactly_it() -> None:
    y_um, _ = coordinates()
    groove = np.abs(y_um - DRAG_CENTRE_UM) <= DRAG_WIDTH_UM / 2

    result = MaskMarks().apply(a_surface(), settings(exclude_drag=True, exclude_extractor=False))

    assert np.array_equal(result.missing, groove)
    # Eight rows of sixty four, measured at this commit.
    assert int(np.count_nonzero(result.missing)) == 512


def test_the_setting_that_does_not_ask_leaves_the_drag_mark_in_place() -> None:
    # This is the configuration the whole comparison is against, and it has to
    # be reachable by a parameter rather than by a change to the code.
    surface = a_surface()

    result = MaskMarks().apply(surface, settings(exclude_drag=False, exclude_extractor=False))

    assert not np.any(result.missing)
    assert_close(
        result.heights,
        surface.heights,
        what="a surface run through the step with neither mark excluded",
        # Exactly. Nothing was asked for and nothing was done.
        atol=0.0,
    )


def test_the_two_regions_are_separate_settings_and_separate_counts() -> None:
    surface = a_surface()
    step = MaskMarks()

    both = step.apply(surface, settings(exclude_drag=True, exclude_extractor=True))
    extractor_only = step.apply(surface, settings(exclude_drag=False, exclude_extractor=True))

    # 512 and 286, measured at this commit, and they do not overlap on this
    # surface, so the two together are their sum.
    assert dict(both.provenance[-1].outcomes) == {"drag_samples": 512, "extractor_samples": 286}
    assert int(np.count_nonzero(both.missing)) == 798
    assert dict(extractor_only.provenance[-1].outcomes) == {
        "drag_samples": 0,
        "extractor_samples": 286,
    }


def test_the_band_is_measured_across_itself_so_the_angle_means_something() -> None:
    # A position measured along a fixed axis would name a different place as
    # soon as the angle moved, and the sweep is going to move the angle.
    y_um, x_um = coordinates()
    angle = np.deg2rad(30.0)
    across = y_um * np.cos(angle) - x_um * np.sin(angle)
    expected = np.abs(across - DRAG_CENTRE_UM) <= DRAG_WIDTH_UM / 2

    result = MaskMarks().apply(
        a_surface(),
        settings(exclude_drag=True, exclude_extractor=False, drag_angle_deg=30.0),
    )

    assert np.array_equal(result.missing, expected)
    # 480 against the 512 the same band covers along the axis, measured at this
    # commit. Turning a band that is offset from the centre swings part of it
    # off the corner of the field, so the count moves as well as the region, and
    # a position measured along a fixed axis would have moved the region
    # somewhere else again.
    assert int(np.count_nonzero(result.missing)) == 480


def test_the_extractor_region_is_the_disc_its_parameters_describe() -> None:
    y_um, x_um = coordinates()
    angle = np.deg2rad(EXTRACTOR_ANGLE_DEG)
    centre_y = EXTRACTOR_DISTANCE_UM * np.cos(angle)
    centre_x = EXTRACTOR_DISTANCE_UM * np.sin(angle)
    disc = np.hypot(y_um - centre_y, x_um - centre_x) <= EXTRACTOR_RADIUS_UM

    result = MaskMarks().apply(a_surface(), settings(exclude_drag=False, exclude_extractor=True))

    assert np.array_equal(result.missing, disc)


def test_the_geometry_is_recorded_whether_or_not_the_mark_was_excluded() -> None:
    # Where the drag mark on a scan lies is a property of the scan; whether this
    # run cut it out is a property of the run. A manifest recording the geometry
    # only when the mask was applied could not tell a surface with no drag mark
    # from one whose drag mark was kept.
    kept = MaskMarks().apply(a_surface(), settings(exclude_drag=False, exclude_extractor=False))

    assert dict(kept.provenance[-1].parameters) == {
        "drag_angle_deg": 0.0,
        "drag_position": DRAG_CENTRE_UM,
        "drag_width": DRAG_WIDTH_UM,
        "exclude_drag": False,
        "exclude_extractor": False,
        "extractor_angle_deg": EXTRACTOR_ANGLE_DEG,
        "extractor_distance": EXTRACTOR_DISTANCE_UM,
        "extractor_radius": EXTRACTOR_RADIUS_UM,
    }


def test_both_configurations_run_end_to_end_and_the_setting_reaches_the_manifest() -> None:
    # The clause asking for both configurations to run end to end. The chain is
    # mask, level, filter, which is the order the ordering rules allow and the
    # one a real run would use.
    registry = Registry()
    registry.register(MaskMarks())
    registry.register(RemoveForm())
    registry.register(RobustGaussianBandpass())
    recorded = {}

    for excluding in (True, False):
        chain = [
            Step(
                identifier="mask-marks",
                parameters=settings(exclude_drag=excluding, exclude_extractor=False),
            ),
            Step(
                identifier="level",
                parameters=LevelParameters(
                    model="polynomial", order=2, robust_tuning=None, robust_passes=None
                ),
            ),
            Step(
                identifier="bandpass",
                parameters=BandpassParameters(
                    short_cutoff=16.0,
                    long_cutoff=120.0,
                    robust_tuning=None,
                    robust_passes=None,
                ),
            ),
        ]
        result, manifest = record_run(
            role="input",
            surface=a_surface(),
            profile=ProfileRecord(name="a-profile", version="1"),
            chain=chain,
            registry=registry,
            seed=0,
            determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
            environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
        )
        recorded[excluding] = (result, manifest)
        assert [step.identifier for step in manifest.steps] == ["mask-marks", "level", "bandpass"]
        assert dict(manifest.steps[0].parameters)["exclude_drag"] is excluding

    excluded_result, excluded_manifest = recorded[True]
    kept_result, kept_manifest = recorded[False]

    # Two runs that differ in one setting, and they differ in the surface as
    # well as in the record. A step whose setting did nothing would produce one
    # surface under two manifests.
    assert int(np.count_nonzero(excluded_result.missing)) == 512
    assert int(np.count_nonzero(kept_result.missing)) == 0
    assert excluded_manifest.to_text() != kept_manifest.to_text()
    assert '"exclude_drag": true' in excluded_manifest.to_text()
    assert '"exclude_drag": false' in kept_manifest.to_text()


def test_an_exclusion_that_covers_no_measured_sample_is_refused() -> None:
    # Not excluding a mark is a configuration this step exists to make
    # reachable, and it is not this refusal. Asking for an exclusion that then
    # does nothing is the silent no-op, and a sweep comparing the two settings
    # would see one surface under two parameter sets.
    far_away = ROWS * SPACING_UM

    with pytest.raises(ValueError, match="drag mark region was asked to be excluded"):
        MaskMarks().apply(
            a_surface(),
            settings(exclude_drag=True, exclude_extractor=False, drag_position=far_away),
        )


def test_an_extractor_exclusion_that_covers_no_measured_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="extractor mark region was asked to be excluded"):
        MaskMarks().apply(
            a_surface(),
            MarkParameters(
                drag_width=DRAG_WIDTH_UM,
                drag_position=DRAG_CENTRE_UM,
                drag_angle_deg=0.0,
                exclude_drag=False,
                extractor_radius=EXTRACTOR_RADIUS_UM,
                extractor_distance=ROWS * SPACING_UM * 2,
                extractor_angle_deg=EXTRACTOR_ANGLE_DEG,
                exclude_extractor=True,
            ),
        )


def test_a_region_that_covers_the_whole_surface_is_refused() -> None:
    with pytest.raises(ValueError, match="cover every measured sample"):
        MaskMarks().apply(
            a_surface(),
            MarkParameters(
                drag_width=ROWS * SPACING_UM * 4,
                drag_position=0.0,
                drag_angle_deg=0.0,
                exclude_drag=True,
                extractor_radius=EXTRACTOR_RADIUS_UM,
                extractor_distance=EXTRACTOR_DISTANCE_UM,
                extractor_angle_deg=EXTRACTOR_ANGLE_DEG,
                exclude_extractor=False,
            ),
        )


def test_a_surface_with_no_measured_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="every sample of this surface is missing"):
        MaskMarks().apply(
            as_surface(np.full((ROWS, COLUMNS), np.nan)),
            settings(exclude_drag=True, exclude_extractor=False),
        )


def test_a_region_with_no_width_is_refused_whether_or_not_it_is_excluded() -> None:
    # Checked whether or not the region is excluded, because the geometry is
    # recorded either way and a manifest carrying a width of nothing describes a
    # scan nobody can reconstruct.
    for width in (0.0, -30.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="drag_width must be a positive finite length"):
            MaskMarks().apply(
                a_surface(),
                MarkParameters(
                    drag_width=width,
                    drag_position=DRAG_CENTRE_UM,
                    drag_angle_deg=0.0,
                    exclude_drag=False,
                    extractor_radius=EXTRACTOR_RADIUS_UM,
                    extractor_distance=EXTRACTOR_DISTANCE_UM,
                    extractor_angle_deg=EXTRACTOR_ANGLE_DEG,
                    exclude_extractor=False,
                ),
            )


def test_a_radius_that_is_not_a_length_is_refused() -> None:
    with pytest.raises(TypeError, match="extractor_radius must be a length"):
        MaskMarks().apply(
            a_surface(),
            MarkParameters(
                drag_width=DRAG_WIDTH_UM,
                drag_position=DRAG_CENTRE_UM,
                drag_angle_deg=0.0,
                exclude_drag=False,
                extractor_radius=True,  # type: ignore[arg-type]
                extractor_distance=EXTRACTOR_DISTANCE_UM,
                extractor_angle_deg=EXTRACTOR_ANGLE_DEG,
                exclude_extractor=False,
            ),
        )


def test_a_position_or_an_angle_that_is_not_a_finite_number_is_refused() -> None:
    with pytest.raises(ValueError, match="drag_position must be finite"):
        MaskMarks().apply(
            a_surface(),
            settings(exclude_drag=False, exclude_extractor=False, drag_position=float("nan")),
        )
    with pytest.raises(TypeError, match="drag_angle_deg must be a number"):
        MaskMarks().apply(
            a_surface(),
            settings(exclude_drag=False, exclude_extractor=False, drag_angle_deg=True),  # type: ignore[arg-type]
        )


def test_a_setting_that_is_not_true_or_false_is_refused() -> None:
    # It is the setting the sweep moves, and a value that is neither would be
    # recorded as one of them without saying which.
    with pytest.raises(TypeError, match="exclude_drag must be true or false"):
        MaskMarks().apply(
            a_surface(),
            MarkParameters(
                drag_width=DRAG_WIDTH_UM,
                drag_position=DRAG_CENTRE_UM,
                drag_angle_deg=0.0,
                exclude_drag=1,  # type: ignore[arg-type]
                extractor_radius=EXTRACTOR_RADIUS_UM,
                extractor_distance=EXTRACTOR_DISTANCE_UM,
                extractor_angle_deg=EXTRACTOR_ANGLE_DEG,
                exclude_extractor=False,
            ),
        )


def test_the_step_refuses_a_record_that_is_not_its_own_before_reading_a_field() -> None:
    class NotOurs:
        drag_width = DRAG_WIDTH_UM

    with pytest.raises(TypeError, match="rather than MarkParameters, so nothing here has read"):
        MaskMarks().apply(a_surface(), NotOurs())  # type: ignore[arg-type]


def test_masking_after_a_filtering_step_is_refused_by_the_pipeline() -> None:
    # The canonical case the ordering rules exist for. The filter spreads the
    # region into its neighbourhood, so the mask afterwards takes the symptom
    # and leaves the cause.
    registry = Registry()
    registry.register(MaskMarks())
    registry.register(RobustGaussianBandpass())
    chain = [
        Step(
            identifier="bandpass",
            parameters=BandpassParameters(
                short_cutoff=16.0, long_cutoff=120.0, robust_tuning=None, robust_passes=None
            ),
        ),
        Step(
            identifier="mask-marks",
            parameters=settings(exclude_drag=True, exclude_extractor=False),
        ),
    ]

    with pytest.raises(OrderingError, match="refuses a surface that is filtered"):
        check_chain(chain, registry)


def test_the_missing_samples_that_were_already_there_are_not_counted_as_this_step_s() -> None:
    heights = np.asarray(a_surface().heights).copy()
    heights[:8, :] = np.nan
    registry = Registry()
    registry.register(MaskMarks())

    result = run_chain(
        [
            Step(
                identifier="mask-marks",
                parameters=settings(exclude_drag=True, exclude_extractor=False),
            )
        ],
        registry,
        as_surface(heights),
    )

    outcomes = dict(result.provenance[-1].outcomes)
    # The groove is eight rows starting at row eight, so the first eight rows of
    # the field overlap none of it, and the count is the groove alone.
    assert outcomes["drag_samples"] == 512


def test_importing_the_package_is_what_registers_the_step() -> None:
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

    assert "mask-marks" in completed.stdout.strip().split(",")
    assert REGISTRY["mask-marks"].version == "1"
