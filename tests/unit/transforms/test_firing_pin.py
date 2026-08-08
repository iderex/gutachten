"""The firing pin masking, against a generated impression of known size.

The generator digs the crater from parameters, so its centre and its radius are
known because they were asked for rather than because somebody measured them
afterwards and agreed. That is what makes the recovery leg here a comparison
against a truth instead of a comparison against another estimate.

Every number quoted in a comment below was measured by running the step at this
commit, and every refusal was deleted in turn and the suite watched go red.
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
from gutachten.transforms.firing_pin import FiringPinParameters, MaskFiringPin
from gutachten.transforms.marks import MarkParameters, MaskMarks
from gutachten.transforms.pipeline import OrderingError, Step, check_chain, run_chain
from gutachten.transforms.registry import REGISTRY, Registry
from tests.support.tolerance import assert_close

ROWS = 128
COLUMNS = 128
SPACING_UM = 8.0

#: The impression the generator digs, which is the truth this file compares
#: against. Written out from the generator's own defaults rather than as numbers
#: that happen to agree with them.
FIRING_PIN_RADIUS_UM = SurfaceParameters().firing_pin_radius_um
FIRING_PIN_DEPTH_UM = SurfaceParameters().firing_pin_depth_um

#: Between the 12 micrometre peak to peak form and the 25 micrometre impression,
#: so it cannot be tripped by the form alone.
DEPTH_THRESHOLD_UM = 10.0
DILATION_UM = 8.0

#: The tolerance the recovery is held to, and where it comes from. The region is
#: resolved to whole samples, so no method reading this grid can place its edge
#: better than half a sampling interval, and a tolerance tighter than that would
#: be a statement about this particular draw. It is derived from the grid rather
#: than read off the measurement: the deviation actually observed is 0.468
#: micrometres in the radius and zero in the centre, which is what a tolerance
#: chosen to make the test pass would have been set just above.
RECOVERY_TOLERANCE_UM = SPACING_UM / 2


def a_surface(**overrides: float | int) -> Surface:
    generated = generate(
        SurfaceParameters(
            rows=ROWS,
            columns=COLUMNS,
            pixel_spacing_um=SPACING_UM,
            seed=20260808,
            **overrides,  # type: ignore[arg-type]
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


def detected(
    *,
    depth_threshold: float | None = DEPTH_THRESHOLD_UM,
    dilation: float = DILATION_UM,
) -> FiringPinParameters:
    return FiringPinParameters(
        method="detected",
        declared_centre_y=None,
        declared_centre_x=None,
        declared_radius=None,
        depth_threshold=depth_threshold,
        dilation=dilation,
    )


def fixed(
    *,
    centre_y: float | None = 0.0,
    centre_x: float | None = 0.0,
    radius: float | None = FIRING_PIN_RADIUS_UM,
    dilation: float = 0.0,
) -> FiringPinParameters:
    return FiringPinParameters(
        method="fixed",
        declared_centre_y=centre_y,
        declared_centre_x=centre_x,
        declared_radius=radius,
        depth_threshold=None,
        dilation=dilation,
    )


def outcomes_of(surface: Surface) -> dict[str, object]:
    return dict(surface.provenance[-1].outcomes)


def test_the_generator_dug_the_impression_this_file_compares_against() -> None:
    # The truth checked before anything is asserted against it. A generator that
    # stopped digging a crater would otherwise make every recovery test below
    # pass or fail for a reason that has nothing to do with this step.
    surface = a_surface(noise_um=0.0)
    y_um, x_um = coordinates()
    inside = np.hypot(y_um, x_um) <= FIRING_PIN_RADIUS_UM * 0.9  # structural: well inside the wall
    outside = np.hypot(y_um, x_um) >= FIRING_PIN_RADIUS_UM * 1.1  # structural: well outside it
    assert float(np.max(surface.heights[inside])) < float(np.min(surface.heights[outside]))


def test_the_recovered_centre_and_radius_are_the_ones_that_were_generated() -> None:
    result = MaskFiringPin().apply(a_surface(), detected())
    found = outcomes_of(result)
    assert_close(
        [found["centre_y"], found["centre_x"]],
        [0.0, 0.0],
        what="the recovered centre of the firing pin impression, against the generated one",
        atol=RECOVERY_TOLERANCE_UM,
    )
    assert_close(
        found["radius"],
        FIRING_PIN_RADIUS_UM,
        what="the recovered radius of the firing pin impression, against the generated one",
        atol=RECOVERY_TOLERANCE_UM,
    )


def test_the_impression_is_what_gets_masked_and_the_surface_beside_it_survives() -> None:
    result = MaskFiringPin().apply(a_surface(), detected())
    y_um, x_um = coordinates()
    radius = np.hypot(y_um, x_um)
    # Half a sampling interval either side of the wall is where the staircase
    # the grid makes of a circle lives, so the assertion is made away from it.
    well_inside = radius <= FIRING_PIN_RADIUS_UM - SPACING_UM
    well_outside = radius >= FIRING_PIN_RADIUS_UM + DILATION_UM + SPACING_UM
    assert bool(np.all(result.missing[well_inside]))
    assert not bool(np.any(result.missing[well_outside]))


def test_the_dilation_changes_the_masked_area() -> None:
    # 1272, 1396 and 1672 samples at this commit. What matters is that the
    # parameter moves the area at all: a dilation that was a constant inside the
    # step would give three identical counts and a sweep over it would report
    # that the step is insensitive to something that decides how much breech
    # face survives.
    counts = [
        int(np.count_nonzero(MaskFiringPin().apply(a_surface(), detected(dilation=d)).missing))
        for d in (0.0, DILATION_UM, DILATION_UM * 3)
    ]
    assert counts[0] < counts[1] < counts[2]

    # And the smaller mask is contained in the larger, so what grew is the edge
    # rather than some other region the step chose differently.
    small = MaskFiringPin().apply(a_surface(), detected(dilation=0.0)).missing
    large = MaskFiringPin().apply(a_surface(), detected(dilation=DILATION_UM)).missing
    assert bool(np.all(large[small]))


def test_a_hole_inside_the_impression_does_not_shrink_the_recovered_radius() -> None:
    # The near miss, and it is not hypothetical: profiles/every-step.json runs
    # mask-marks first and its extractor disc sits inside the crater. Without
    # closing the region before measuring it, the recovered radius comes out at
    # 155.44 against a truth of 160.0, which is 4.56 outside a tolerance of 4.0
    # and would leave a ring of crater wall inside the mask. It fails by half a
    # micrometre, which is the kind of near miss worth having: a hole one size
    # smaller would pass, and the guard is still the reason this one does not.
    masked_first = MaskMarks().apply(
        a_surface(),
        MarkParameters(
            drag_width=30.0,
            drag_position=-256.0,
            drag_angle_deg=0.0,
            exclude_drag=True,
            extractor_radius=40.0,
            extractor_distance=100.0,
            extractor_angle_deg=90.0,
            exclude_extractor=True,
        ),
    )
    found = outcomes_of(MaskFiringPin().apply(masked_first, detected()))
    assert_close(
        found["radius"],
        FIRING_PIN_RADIUS_UM,
        what="the recovered radius with the extractor mark already cut out of the crater",
        atol=RECOVERY_TOLERANCE_UM,
    )


def test_the_fixed_method_masks_the_circle_it_was_told_and_recovers_nothing() -> None:
    stated = FIRING_PIN_RADIUS_UM / 2
    result = MaskFiringPin().apply(a_surface(), fixed(radius=stated))
    found = outcomes_of(result)
    assert found["radius"] == stated
    assert found["masked_radius"] == stated
    y_um, x_um = coordinates()
    radius = np.hypot(y_um, x_um)
    assert bool(np.all(result.missing[radius <= stated - SPACING_UM]))
    assert not bool(np.any(result.missing[radius >= stated + SPACING_UM]))


def test_the_region_that_was_masked_is_recorded_beside_the_parameters_that_chose_it() -> None:
    # The provenance carries both halves and keeps them apart: what the run was
    # told, and what it found. A sweep over depth_threshold is only interpretable
    # if the region each cell produced is recorded, because the thing that moved
    # the score is the region and not the threshold that chose it.
    #
    # The run manifest carries the parameters and not the outcomes, so the
    # recovered region reaches the surface's provenance and stops there. That is
    # a property of gutachten.manifest.StepRecord rather than of this step, and
    # it is written up in the issue this step was built from.
    surface = a_surface()
    chain = [Step(identifier="mask-firing-pin", parameters=detected())]
    result, manifest = record_run(
        role="input",
        surface=surface,
        profile=ProfileRecord(name="a-test", version="1"),
        chain=chain,
        registry=REGISTRY,
        seed=0,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
    )
    record = result.provenance[-1]
    assert record.name == "mask-firing-pin"
    assert [key for key, _ in record.outcomes] == [
        "centre_x",
        "centre_y",
        "masked_radius",
        "masked_samples",
        "radius",
    ]
    assert dict(record.parameters)["depth_threshold"] == DEPTH_THRESHOLD_UM
    assert dict(record.parameters)["dilation"] == DILATION_UM

    step = manifest.steps[0]
    assert step.identifier == "mask-firing-pin"
    assert dict(step.parameters)["method"] == "detected"
    assert '"mask-firing-pin"' in manifest.to_text()


def test_a_method_this_step_does_not_know_is_refused() -> None:
    with pytest.raises(ValueError, match="is not a method this step knows"):
        MaskFiringPin().apply(
            a_surface(),
            FiringPinParameters(
                method="hand-drawn",
                declared_centre_y=None,
                declared_centre_x=None,
                declared_radius=None,
                depth_threshold=DEPTH_THRESHOLD_UM,
                dilation=DILATION_UM,
            ),
        )


def test_a_method_whose_own_fields_are_null_is_refused() -> None:
    with pytest.raises(ValueError, match="needs \\['declared_radius'\\]"):
        MaskFiringPin().apply(a_surface(), fixed(radius=None))
    with pytest.raises(ValueError, match="needs \\['depth_threshold'\\]"):
        MaskFiringPin().apply(a_surface(), detected(depth_threshold=None))


def test_a_record_carrying_the_other_method_s_fields_is_refused() -> None:
    # Both directions. A record holding a fixed circle and a detection threshold
    # at once leaves a reader of the manifest unable to say which of the two
    # decided the region that was removed.
    with pytest.raises(ValueError, match="does not read \\['depth_threshold'\\]"):
        MaskFiringPin().apply(
            a_surface(),
            FiringPinParameters(
                method="fixed",
                declared_centre_y=0.0,
                declared_centre_x=0.0,
                declared_radius=FIRING_PIN_RADIUS_UM,
                depth_threshold=DEPTH_THRESHOLD_UM,
                dilation=0.0,
            ),
        )
    with pytest.raises(ValueError, match="does not read \\['declared_radius'\\]"):
        MaskFiringPin().apply(
            a_surface(),
            FiringPinParameters(
                method="detected",
                declared_centre_y=None,
                declared_centre_x=None,
                declared_radius=FIRING_PIN_RADIUS_UM,
                depth_threshold=DEPTH_THRESHOLD_UM,
                dilation=0.0,
            ),
        )


def test_a_dilation_that_is_not_a_length_of_zero_or_more_is_refused() -> None:
    with pytest.raises(TypeError, match="dilation must be a length"):
        MaskFiringPin().apply(a_surface(), detected(dilation="wide"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dilation must be a length"):
        MaskFiringPin().apply(a_surface(), detected(dilation=True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite length of zero or more"):
        MaskFiringPin().apply(a_surface(), detected(dilation=-1.0))
    with pytest.raises(ValueError, match="finite length of zero or more"):
        MaskFiringPin().apply(a_surface(), detected(dilation=float("nan")))


def test_a_declared_field_that_is_not_a_finite_number_is_refused() -> None:
    with pytest.raises(TypeError, match="declared_centre_y must be a number"):
        MaskFiringPin().apply(a_surface(), fixed(centre_y="middle"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="declared_radius must be a number"):
        MaskFiringPin().apply(a_surface(), fixed(radius=True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="declared_centre_x must be finite"):
        MaskFiringPin().apply(a_surface(), fixed(centre_x=float("inf")))


def test_a_radius_or_a_threshold_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="declared_radius must be a positive length"):
        MaskFiringPin().apply(a_surface(), fixed(radius=0.0))
    with pytest.raises(ValueError, match="depth_threshold must be a positive length"):
        MaskFiringPin().apply(a_surface(), detected(depth_threshold=-1.0))


def test_a_surface_with_nothing_that_deep_is_refused_rather_than_masked_anyway() -> None:
    # Deeper than the impression the generator dug. The refusal says the two
    # cases are different, because widening the threshold until something is
    # found is exactly how a step ends up masking the form.
    with pytest.raises(ValueError, match="no crater here to detect"):
        MaskFiringPin().apply(a_surface(), detected(depth_threshold=FIRING_PIN_DEPTH_UM * 10))


def test_a_deepest_region_that_is_not_near_the_centre_is_refused() -> None:
    # A firing pin impression is near the centre of the primer. A deep region in
    # a corner is something else, and masking it would remove the something else
    # and leave the impression in place.
    heights = np.zeros((ROWS, COLUMNS), dtype=np.float64)
    heights[:8, :8] = -FIRING_PIN_DEPTH_UM  # structural: a corner, in samples
    with pytest.raises(ValueError, match="the centre of the field is not inside it"):
        MaskFiringPin().apply(as_surface(heights), detected())


def test_a_region_that_covers_no_measured_sample_is_refused() -> None:
    away = ROWS * SPACING_UM
    with pytest.raises(ValueError, match="covers no measured sample"):
        MaskFiringPin().apply(a_surface(), fixed(centre_y=away, centre_x=away, radius=SPACING_UM))


def test_a_region_that_covers_the_whole_surface_is_refused() -> None:
    with pytest.raises(ValueError, match="covers every measured sample"):
        MaskFiringPin().apply(a_surface(), fixed(radius=ROWS * SPACING_UM))


def test_a_surface_with_no_measured_sample_is_refused() -> None:
    empty = np.full((ROWS, COLUMNS), np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="every sample of this surface is missing"):
        MaskFiringPin().apply(as_surface(empty), detected())


def test_the_step_refuses_a_record_that_is_not_its_own_before_reading_a_field() -> None:
    with pytest.raises(TypeError, match="rather than FiringPinParameters"):
        MaskFiringPin().apply(a_surface(), BandpassParameters(20.0, 120.0, None, None))


def test_masking_after_a_filtering_step_is_refused_by_the_pipeline() -> None:
    # The ordering rule this step declares. A bandpass spreads the crater into
    # the surface around it, so a mask applied afterwards removes the crater and
    # leaves what the crater did to its neighbourhood, and the run exits zero.
    chain = [
        Step(
            identifier="bandpass",
            parameters=BandpassParameters(
                short_cutoff=20.0, long_cutoff=120.0, robust_tuning=None, robust_passes=None
            ),
        ),
        Step(identifier="mask-firing-pin", parameters=detected()),
    ]
    with pytest.raises(OrderingError, match="refuses a surface that is filtered"):
        check_chain(chain, REGISTRY)


def test_the_step_runs_as_a_chain_and_records_itself_once() -> None:
    registry = Registry()
    registry.register(MaskFiringPin())
    registry.register(RobustGaussianBandpass())
    result = run_chain(
        [Step(identifier="mask-firing-pin", parameters=detected())],
        registry,
        a_surface(),
    )
    assert [record.name for record in result.provenance] == ["mask-firing-pin"]


def test_importing_the_package_is_what_registers_the_step() -> None:
    # The registry is what the manifest resolver, the sweep and the constants
    # audit all read, so a step reachable only by importing its own module is a
    # step all three are blind to.
    source = "import gutachten.transforms; from gutachten.transforms.registry import REGISTRY; "
    source += "print('mask-firing-pin' in REGISTRY)"
    finished = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )
    assert finished.stdout.strip() == "True"
