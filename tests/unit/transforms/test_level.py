"""The levelling step, against a surface whose tilt and curvature are known.

The generator is asked for a bowl and nothing else, and the tilt is added here
by an expression this file also uses to state what levelling has to remove. So
every expectation below is a comparison against a construction rather than
against a guess about what a real scan looks like.

Every residual quoted here was measured by running the step at this commit, and
each refusal was deleted in turn and the suite watched go red.
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
from gutachten.transforms.level import LevelParameters, RemoveForm
from gutachten.transforms.pipeline import OrderingError, Step, check_chain
from gutachten.transforms.registry import REGISTRY, Registry
from tests.support.tolerance import assert_close
from tests.unit.transforms.declared_example import Scale, ScaleParameters
from tests.unit.transforms.undeclared_example import Clip, ClipParameters

ROWS = 48
COLUMNS = 48
SPACING_UM = 4.0
#: Peak to peak depth of the bowl the generator is asked for.
FORM_DEPTH_UM = 12.0
#: The tilt this file adds, as a height per unit of length along each axis.
TILT_DOWN = 0.05
TILT_ACROSS = -0.03


def coordinates() -> tuple[np.ndarray, np.ndarray]:
    """The same centred physical grid the step fits over."""
    down = (np.arange(ROWS, dtype=np.float64) - (ROWS - 1) / 2) * SPACING_UM
    across = (np.arange(COLUMNS, dtype=np.float64) - (COLUMNS - 1) / 2) * SPACING_UM
    y_um, x_um = np.meshgrid(down, across, indexing="ij")
    return np.asarray(y_um), np.asarray(x_um)


def a_bowl() -> np.ndarray:
    """A generated surface carrying the form and nothing else.

    The striae, the firing pin and the drag mark are asked for at zero depth and
    the noise at zero, so what comes back is the generator's quadratic bowl on
    its own. A residual stated against anything else would be a residual against
    the striae as much as against the levelling.
    """
    generated = generate(
        SurfaceParameters(
            rows=ROWS,
            columns=COLUMNS,
            pixel_spacing_um=SPACING_UM,
            form_depth_um=FORM_DEPTH_UM,
            striae_depth_um=0.0,
            firing_pin_depth_um=0.0,
            drag_mark_depth_um=0.0,
            noise_um=0.0,
            seed=20260808,
        )
    )
    return np.asarray(generated.heights_um)


def a_tilted_bowl() -> np.ndarray:
    y_um, x_um = coordinates()
    return a_bowl() + TILT_DOWN * y_um + TILT_ACROSS * x_um


def as_surface(heights: np.ndarray) -> Surface:
    return Surface(
        heights=heights,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def ordinary(model: str, order: int | None) -> LevelParameters:
    return LevelParameters(model=model, order=order, robust_tuning=None, robust_passes=None)


def levelled(heights: np.ndarray, parameters: LevelParameters) -> np.ndarray:
    return np.asarray(RemoveForm().apply(as_surface(heights), parameters).heights)


def test_the_generated_surface_is_the_bowl_this_file_says_it_is() -> None:
    # Asserted rather than assumed. Every residual below is stated against this
    # shape, so a generator that made a different one would leave those tests
    # passing against the wrong surface.
    y_um, x_um = coordinates()
    radius = np.hypot(y_um, x_um)
    expected = FORM_DEPTH_UM * (radius / radius.max()) ** 2

    assert_close(
        a_bowl(),
        expected,
        what="the generated bowl against the quadratic this file expects",
        # Exactly zero. Both sides are the same arithmetic on the same grid.
        atol=0.0,
    )


def test_the_second_order_polynomial_removes_the_tilt_and_the_curvature() -> None:
    # The bowl is quadratic by construction and the tilt is linear, so a second
    # order polynomial spans both exactly and what is left is the solve's own
    # rounding.
    result = levelled(a_tilted_bowl(), ordinary("polynomial", 2))

    assert_close(
        result,
        np.zeros_like(result),
        what="what a second order polynomial leaves of a known tilt and a known bowl",
        # Measured at 1.8e-14 micrometres at this commit, over heights of order
        # ten micrometres. The bound is a decimal order above it so that ordinary
        # floating point differences between platforms do not redden the gate,
        # and it is still ten decimal orders below anything this pipeline calls a
        # feature.
        atol=1e-13,
    )


def test_the_plane_removes_the_tilt_and_leaves_the_curvature() -> None:
    # Two separate claims, and the second is the one that matters: a plane is
    # offered because it removes less, and a test that only checked the tilt had
    # gone would pass for a step that removed everything.
    tilted = levelled(a_tilted_bowl(), ordinary("plane", None))
    untilted = levelled(a_bowl(), ordinary("plane", None))

    assert_close(
        tilted,
        untilted,
        what="a plane levelled surface with the tilt against one without it",
        # Measured at 8.9e-15 micrometres at this commit. The tilt is gone in
        # both, so what remains is the same bowl either way.
        atol=1e-13,
    )
    # And the bowl survives it: 11.99 micrometres peak to valley against the
    # 12.0 the generator was asked for, measured at this commit.
    assert float(tilted.max() - tilted.min()) > FORM_DEPTH_UM / 2


def test_each_model_leaves_the_residual_measured_for_it() -> None:
    # The three models are here because each removes a different amount, and
    # this is where that is a number rather than a sentence. Each bound was
    # measured at this commit by running the step.
    heights = a_tilted_bowl()
    cases = (
        # A plane cannot reach a quadratic bowl, and 2.64 micrometres is how
        # much of one it leaves behind on this surface.
        (ordinary("plane", None), 2.636, 0.001),
        # Exact, to the solve's own rounding.
        (ordinary("polynomial", 2), 0.0, 1e-13),
        # A sphere is not the paraboloid the generator makes, and 0.0139
        # micrometres is what that difference is worth here. It is two decimal
        # orders below the plane and far above the polynomial, which is the
        # ordering the sensitivity study is going to be reporting.
        (ordinary("sphere", None), 0.01393, 0.00001),
    )

    for parameters, expected, bound in cases:
        residual = float(np.sqrt(np.mean(levelled(heights, parameters) ** 2)))
        assert_close(
            residual,
            expected,
            what=f"the root mean square left by the {parameters.model!r} model",
            atol=bound,
        )


def test_the_fit_ignores_the_masked_samples_and_the_difference_is_measured() -> None:
    # Not-a-number does not survive a least squares solve, so the shape this
    # error actually takes is a mask that something upstream filled with a
    # plausible constant. Both fits are made and the distance between them is
    # reported, so this is a measurement of the error rather than an assertion
    # that it is absent.
    masked = a_bowl()
    masked[: ROWS // 4, : COLUMNS // 4] = np.nan
    filled = np.where(np.isnan(masked), 0.0, masked)
    observed = ~np.isnan(masked)

    ignoring = levelled(masked, ordinary("polynomial", 2))
    including = levelled(filled, ordinary("polynomial", 2))

    assert_close(
        ignoring[observed],
        np.zeros_like(ignoring[observed]),
        what="what the fit leaves where it excluded the masked samples",
        # Measured at 7.1e-15 micrometres at this commit.
        atol=1e-13,
    )
    # 3.95 micrometres, measured at this commit, against a bowl 12 micrometres
    # deep. A third of the form left behind, from a mask filled with a number
    # nobody would look at twice.
    worst = float(np.max(np.abs(including[observed] - ignoring[observed])))
    assert_close(
        worst,
        3.947,
        what="how far a fit that read the filled mask is from one that did not",
        atol=0.001,
    )


def test_the_missing_samples_stay_missing_and_the_shape_does_not_move() -> None:
    masked = a_bowl()
    masked[: ROWS // 4, : COLUMNS // 4] = np.nan
    surface = as_surface(masked)

    result = RemoveForm().apply(surface, ordinary("polynomial", 2))

    assert result.shape == surface.shape
    assert np.array_equal(result.missing, surface.missing)


def test_a_robust_fit_resists_a_spike_that_an_ordinary_fit_follows() -> None:
    # A dropout read as an extreme height is what an optical instrument
    # produces, and least squares is defined by how hard it chases one.
    spiked = a_bowl()
    corner = (slice(5, 8), slice(5, 8))
    spiked[corner] += 500.0
    clean = np.ones(spiked.shape, dtype=bool)
    clean[corner] = False

    plain = levelled(spiked, ordinary("polynomial", 2))
    robust = levelled(
        spiked, LevelParameters(model="polynomial", order=2, robust_tuning=1.5, robust_passes=5)
    )

    assert_close(
        robust[clean],
        np.zeros_like(robust[clean]),
        what="what a robust fit leaves of the bowl outside the spike",
        # Measured at 1.1e-07 micrometres at this commit. Not zero: Huber
        # weights the spike down rather than discarding it, and what is left is
        # the weight it kept.
        atol=1e-06,
    )
    # 24.8 micrometres, measured at this commit. Twice the depth of the bowl the
    # step was asked to remove, from nine samples out of two thousand.
    assert_close(
        float(np.max(np.abs(plain[clean]))),
        24.79,
        what="what an ordinary fit leaves of the bowl outside the spike",
        atol=0.01,
    )


def test_a_surface_with_nothing_left_to_remove_survives_a_robust_fit() -> None:
    # An exact fit leaves residuals whose scale is zero, and a Huber band of
    # zero width puts every sample at zero weight. The next solve would then
    # have no data in it at all, which is a division nobody sees until a real
    # surface happens to be flat.
    flat = np.zeros((ROWS, COLUMNS))

    result = levelled(
        flat, LevelParameters(model="plane", order=None, robust_tuning=1.5, robust_passes=3)
    )

    assert_close(
        result,
        flat,
        what="a flat surface levelled robustly",
        # Exactly zero. There is nothing to remove and nothing was removed.
        atol=0.0,
    )


def test_the_step_records_the_model_and_the_order_it_ran_with() -> None:
    result = RemoveForm().apply(as_surface(a_bowl()), ordinary("polynomial", 2))

    entry = result.provenance[-1]
    assert entry.name == "level"
    assert entry.version == "1"
    assert dict(entry.parameters) == {
        "model": "polynomial",
        "order": 2,
        "robust_passes": None,
        "robust_tuning": None,
    }


def test_the_model_and_the_order_reach_the_manifest() -> None:
    # The provenance entry above is on the surface. What a sweep and a re-run
    # read is the manifest, and this is the assertion that the two do not part
    # company.
    registry = Registry()
    registry.register(RemoveForm())
    chain = [Step(identifier="level", parameters=ordinary("polynomial", 3))]

    _, manifest = record_run(
        role="input",
        surface=as_surface(a_tilted_bowl()),
        profile=ProfileRecord(name="a-profile", version="1"),
        chain=chain,
        registry=registry,
        seed=0,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
    )

    step = manifest.steps[-1]
    assert step.identifier == "level"
    assert dict(step.parameters)["model"] == "polynomial"
    assert dict(step.parameters)["order"] == 3
    assert '"order": 3' in manifest.to_text()


def test_a_model_this_step_does_not_know_is_refused_naming_what_it_takes() -> None:
    with pytest.raises(ValueError, match="not a form model this step knows; it takes one of"):
        RemoveForm().apply(as_surface(a_bowl()), ordinary("Plane", None))


def test_an_order_recorded_against_a_model_that_has_none_is_refused() -> None:
    # A number in the manifest that decided nothing is a number the sweep will
    # move while reporting that the step is insensitive to it.
    for model in ("plane", "sphere"):
        with pytest.raises(ValueError, match="has no order to choose"):
            RemoveForm().apply(as_surface(a_bowl()), ordinary(model, 2))


def test_a_polynomial_with_no_order_is_refused() -> None:
    with pytest.raises(ValueError, match="names no order"):
        RemoveForm().apply(as_surface(a_bowl()), ordinary("polynomial", None))


def test_a_first_order_polynomial_is_refused_and_points_at_the_plane() -> None:
    # One form reachable under two names would be visited twice by a sweep, and
    # two rows of the sensitivity table would then differ only in the word
    # recorded against them.
    with pytest.raises(ValueError, match="a first order polynomial is a plane"):
        RemoveForm().apply(as_surface(a_bowl()), ordinary("polynomial", 1))


def test_an_order_below_the_first_is_refused() -> None:
    with pytest.raises(ValueError, match="must be at least the first"):
        RemoveForm().apply(as_surface(a_bowl()), ordinary("polynomial", 0))


def test_an_order_that_is_not_a_whole_number_is_refused() -> None:
    # `True` is an `int` to Python and would be read as a first order fit.
    for order in (2.0, True):
        with pytest.raises(TypeError, match="must be a whole number"):
            RemoveForm().apply(as_surface(a_bowl()), ordinary("polynomial", order))  # type: ignore[arg-type]


def test_half_a_robustness_setting_is_refused() -> None:
    # A tuning constant nothing reweights and a count of reweightings with no
    # constant to reweight by are each half a setting, and each would run as an
    # ordinary fit while the manifest recorded something else.
    halves = (
        LevelParameters(model="plane", order=None, robust_tuning=1.5, robust_passes=None),
        LevelParameters(model="plane", order=None, robust_tuning=None, robust_passes=3),
    )
    for parameters in halves:
        with pytest.raises(ValueError, match="disagree about whether"):
            RemoveForm().apply(as_surface(a_bowl()), parameters)


def test_a_tuning_constant_that_is_not_a_positive_finite_width_is_refused() -> None:
    for tuning in (0.0, -1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite number of residual scales"):
            RemoveForm().apply(
                as_surface(a_bowl()),
                LevelParameters(model="plane", order=None, robust_tuning=tuning, robust_passes=3),
            )


def test_a_tuning_constant_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(TypeError, match="robust_tuning must be a number"):
        RemoveForm().apply(
            as_surface(a_bowl()),
            LevelParameters(model="plane", order=None, robust_tuning=True, robust_passes=3),
        )


def test_a_pass_count_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="robust_passes must be at least one"):
        RemoveForm().apply(
            as_surface(a_bowl()),
            LevelParameters(model="plane", order=None, robust_tuning=1.5, robust_passes=0),
        )


def test_a_pass_count_that_is_not_a_whole_number_is_refused() -> None:
    for passes in (3.0, True):
        with pytest.raises(TypeError, match="robust_passes must be a whole number"):
            RemoveForm().apply(
                as_surface(a_bowl()),
                LevelParameters(model="plane", order=None, robust_tuning=1.5, robust_passes=passes),  # type: ignore[arg-type]
            )


def test_a_surface_with_no_measured_sample_is_refused() -> None:
    # Every reduction over an empty array answers with not-a-number rather than
    # with an error, so the step would record itself as having run.
    with pytest.raises(ValueError, match="every sample of this surface is missing"):
        RemoveForm().apply(as_surface(np.full((ROWS, COLUMNS), np.nan)), ordinary("plane", None))


def test_a_surface_of_one_sample_determines_no_form() -> None:
    with pytest.raises(ValueError, match="has no extent in either"):
        RemoveForm().apply(as_surface(np.zeros((1, 1))), ordinary("plane", None))


def test_observed_samples_that_lie_on_one_line_are_refused() -> None:
    # A rank deficient solve returns the minimum norm answer rather than
    # failing, and that answer is a form nobody asked for subtracted from the
    # whole field.
    one_row = a_bowl()
    one_row[1:, :] = np.nan

    with pytest.raises(ValueError, match="do not determine a polynomial of order 2"):
        RemoveForm().apply(as_surface(one_row), ordinary("polynomial", 2))


def test_a_surface_with_no_curvature_determines_no_sphere() -> None:
    y_um, x_um = coordinates()

    with pytest.raises(ValueError, match="do not determine a sphere"):
        RemoveForm().apply(
            as_surface(TILT_DOWN * y_um + TILT_ACROSS * x_um), ordinary("sphere", None)
        )


def test_a_sphere_that_does_not_reach_the_corners_of_the_field_is_refused() -> None:
    # The alternative is a clamp, and a clamped sphere puts a flat rim on the
    # levelled surface that no later step can tell from measured data.
    y_um, x_um = coordinates()
    radius = np.hypot(y_um, x_um)
    steep = 100.0 * (radius / radius.max()) ** 2

    with pytest.raises(ValueError, match="does not reach every sample of the field"):
        RemoveForm().apply(as_surface(steep), ordinary("sphere", None))


def test_the_step_refuses_a_record_that_is_not_its_own_before_reading_a_field() -> None:
    # `record_for` refuses the same mistake on the way out, by which point the
    # fit has been made. The wording asserted here is the one only the check at
    # the door produces, so this cannot pass on the later refusal.
    class NotOurs:
        model = "plane"
        order = None
        robust_tuning = None
        robust_passes = None

    with pytest.raises(TypeError, match="rather than LevelParameters, so nothing here has read"):
        RemoveForm().apply(as_surface(a_bowl()), NotOurs())  # type: ignore[arg-type]


def test_the_two_axes_are_measured_at_their_own_spacings() -> None:
    # The fixture above is square in both spacing and shape, so a step reading
    # one spacing for both axes would pass every other test in this file.
    y_um, x_um = coordinates()
    # The tilt is written against the physical grid at the true spacings, so a
    # step that assumed the column spacing equalled the row spacing would fit
    # the wrong gradient across the columns and leave it behind.
    heights = TILT_DOWN * y_um + TILT_ACROSS * (x_um / 2)
    anisotropic = Surface(
        heights=heights,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM / 2,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )

    result = RemoveForm().apply(anisotropic, ordinary("plane", None))

    assert_close(
        result.heights,
        np.zeros_like(result.heights),
        what="what a plane leaves of a tilt written against the true spacings",
        # Exact to the solve's rounding; the surface is a plane and the model is
        # a plane.
        atol=1e-13,
    )


def test_levelling_after_a_filtering_step_is_refused_by_the_pipeline() -> None:
    # A bandpass has already removed the long wavelength content a form model is
    # fitted to, so the fit afterwards is a fit to what the filter left at the
    # edges of its band.
    registry = Registry()
    registry.register(Scale())
    registry.register(Clip())
    registry.register(RemoveForm())
    chain = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=1.0)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=1.0)),
        Step(identifier="level", parameters=ordinary("plane", None)),
    ]

    with pytest.raises(OrderingError, match="refuses a surface that is filtered"):
        check_chain(chain, registry)


def test_importing_the_package_is_what_registers_the_step() -> None:
    # In a fresh interpreter, because this module has already imported the step
    # directly and that import is what registers it. What has to hold is that a
    # caller who imported only the package sees a complete registry, since the
    # manifest resolver, the sweep and the constants audit all read it.
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

    assert "level" in completed.stdout.strip().split(",")
    assert REGISTRY["level"].version == "1"
