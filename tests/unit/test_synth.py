"""One test per generated feature, asserting the feature is there with the
parameters it was asked for.

A generator nobody checks is a generator whose ground truth is a comment. If the
striae are not at the spacing that was requested, every preprocessing test built
on this module measures the wrong thing and each one still passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from gutachten.synth import (
    SurfaceParameters,
    generate,
    matching_pair,
    non_matching_pair,
)
from tests.support.tolerance import assert_close

# A field large enough that a spatial frequency is measurable in it and small
# enough that the suite stays quick.
BASE = SurfaceParameters(
    rows=192,
    columns=192,
    pixel_spacing_um=4.0,
    striae_depth_um=2.0,
    striae_spacing_um=40.0,
    form_depth_um=12.0,
    firing_pin_radius_um=120.0,
    firing_pin_depth_um=25.0,
    drag_mark_width_um=32.0,
    drag_mark_depth_um=6.0,
    noise_um=0.0,
    seed=20260808,
)


STRIAE_ONLY = {"form_depth_um": 0.0, "firing_pin_depth_um": 0.0, "drag_mark_depth_um": 0.0}


def dominant_period_um(surface) -> float:
    """Recover the striation period from the surface, by measurement.

    The mean of each row is removed first so that a constant offset does not
    take the spectrum's first bin. This function is the test's own instrument
    and is deliberately blunt: it recovers a period from an image without being
    told what period to expect.
    """
    heights = np.nan_to_num(surface.heights_um, nan=0.0)
    heights = heights - heights.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(heights, axis=1)).mean(axis=0)
    spectrum[0] = 0.0
    peak = int(np.argmax(spectrum))
    return heights.shape[1] * surface.parameters.pixel_spacing_um / peak


def test_the_striae_are_at_the_spacing_that_was_asked_for() -> None:
    for requested in (32.0, 40.0, 48.0, 64.0):
        surface = generate(
            BASE.__class__(**{**BASE.__dict__, **STRIAE_ONLY, "striae_spacing_um": requested})
        )
        measured = dominant_period_um(surface)
        # The pattern is a band of components spread by eight percent either
        # side of the requested frequency, so the strongest bin can sit that far
        # off before anything is wrong, and one FFT bin at this field size adds
        # a few percent more. Twelve percent is those two numbers and not a
        # number chosen to make the test pass. A pattern at half or twice the
        # requested spacing fails this by a wide margin, which is the mistake
        # worth catching.
        assert_close(
            measured,
            requested,
            what=f"recovered striation period for a requested {requested} um",
            atol=0.0,
            rtol=0.12,
        )


def test_the_striae_have_the_depth_that_was_asked_for() -> None:
    quiet = BASE.__class__(**{**BASE.__dict__, **STRIAE_ONLY, "striae_depth_um": 3.0})
    surface = generate(quiet)
    observed = surface.observed
    peak_to_peak = float(observed.max() - observed.min())
    # The generator rescales the pattern to the requested peak to peak, so this
    # is exact rather than approximate and the tolerance says so.
    assert_close(
        peak_to_peak,
        3.0,
        what="peak to peak height of a surface carrying only striae",
        atol=1e-12,
    )


def test_the_firing_pin_region_is_present_at_the_radius_that_was_asked_for() -> None:
    surface = generate(BASE)
    heights = surface.heights_um
    rows, columns = np.indices(heights.shape)
    spacing = BASE.pixel_spacing_um
    radius_um = np.hypot(rows - (BASE.rows - 1) / 2.0, columns - (BASE.columns - 1) / 2.0) * spacing

    well_inside = radius_um < BASE.firing_pin_radius_um * 0.6
    well_outside = radius_um > BASE.firing_pin_radius_um * 1.4
    step = float(heights[well_outside].mean() - heights[well_inside].mean())

    # The step across the boundary is the firing pin depth plus whatever the
    # form contributes between the two rings, so this asserts the impression is
    # present and of the right order rather than exact to the micrometre.
    assert step > BASE.firing_pin_depth_um * 0.8, (
        f"no firing pin impression found: the mean height outside the requested "
        f"radius is only {step} um above the mean height inside it, against a "
        f"requested depth of {BASE.firing_pin_depth_um} um"
    )


def test_the_drag_mark_is_present_at_the_width_that_was_asked_for() -> None:
    plain = BASE.__class__(
        **{
            **BASE.__dict__,
            "striae_depth_um": 0.0,
            "form_depth_um": 0.0,
            "firing_pin_depth_um": 0.0,
        }
    )
    surface = generate(plain)
    profile = surface.heights_um.mean(axis=1)
    grooved = profile < -plain.drag_mark_depth_um / 2.0
    measured_width_um = float(np.count_nonzero(grooved)) * plain.pixel_spacing_um
    assert_close(
        measured_width_um,
        plain.drag_mark_width_um,
        what="width of the drag mark measured off the row profile",
        atol=plain.pixel_spacing_um,
    )


def test_the_form_is_present_with_the_depth_that_was_asked_for() -> None:
    bowl = BASE.__class__(
        **{
            **BASE.__dict__,
            "striae_depth_um": 0.0,
            "firing_pin_depth_um": 0.0,
            "drag_mark_depth_um": 0.0,
            "form_depth_um": 20.0,
        }
    )
    surface = generate(bowl)
    heights = surface.heights_um
    centre = float(heights[bowl.rows // 2, bowl.columns // 2])
    corner = float(heights[0, 0])
    assert_close(
        corner - centre,
        20.0,
        what="peak to peak height of the form, corner against centre",
        # The centre sample sits half a pixel off the exact centre on an even
        # grid, where the quadratic is a fraction of a nanometre above zero.
        atol=1e-3,
    )


def test_the_noise_has_the_standard_deviation_that_was_asked_for() -> None:
    flat = BASE.__class__(
        **{
            **BASE.__dict__,
            "striae_depth_um": 0.0,
            "form_depth_um": 0.0,
            "firing_pin_depth_um": 0.0,
            "drag_mark_depth_um": 0.0,
            "noise_um": 0.5,
        }
    )
    surface = generate(flat)
    measured = float(np.std(surface.observed))
    # Over 192 by 192 independent samples the sample standard deviation is
    # within a fraction of a percent of the population value, so five percent is
    # loose enough to never flake and tight enough to catch a wrong scale.
    assert_close(
        measured,
        0.5,
        what="standard deviation of the generated noise",
        atol=0.0,
        rtol=0.05,
    )


def test_edge_dropout_removes_the_fraction_that_was_asked_for() -> None:
    lossy = BASE.__class__(**{**BASE.__dict__, "edge_dropout_fraction": 0.1})
    surface = generate(lossy)
    heights = surface.heights_um

    assert np.all(np.isnan(heights[0, :])), "the requested edge dropout left the first row measured"
    assert np.all(np.isnan(heights[:, 0])), (
        "the requested edge dropout left the first column measured"
    )
    assert np.all(np.isfinite(heights[lossy.rows // 2, lossy.columns // 2])), (
        "the edge dropout reached the centre of the field"
    )

    kept_rows = lossy.rows - 2 * round(lossy.rows * 0.1)
    kept_columns = lossy.columns - 2 * round(lossy.columns * 0.1)
    assert_close(
        float(np.count_nonzero(np.isfinite(heights))),
        float(kept_rows * kept_columns),
        what="number of measured samples left after the requested edge dropout",
        atol=0.0,
    )


def test_a_surface_with_no_dropout_has_no_missing_data() -> None:
    surface = generate(BASE)
    assert np.all(np.isfinite(surface.heights_um)), (
        "a surface generated with no edge dropout still carries missing data"
    )


def test_the_same_seed_gives_the_same_surface() -> None:
    first = generate(BASE)
    second = generate(BASE)
    assert np.array_equal(first.heights_um, second.heights_um), (
        "two surfaces generated from identical parameters are not identical"
    )


def test_a_different_seed_gives_a_different_surface() -> None:
    other = BASE.__class__(**{**BASE.__dict__, "seed": BASE.seed + 1, "noise_um": 0.2})
    assert not np.array_equal(generate(BASE).heights_um, generate(other).heights_um)


def test_a_matching_pair_shares_its_striae_and_a_non_matching_pair_does_not() -> None:
    # Striae only. The form, the firing pin and the drag mark are identical in
    # every surface these parameters produce, so leaving them in would have both
    # pairs correlating above 0.9 on the shared shape and the test would pass
    # while proving nothing about the striae. That was measured, not assumed.
    noisy = BASE.__class__(**{**BASE.__dict__, **STRIAE_ONLY, "noise_um": 0.2})

    same_a, same_b = matching_pair(noisy, translation_px=(0.0, 0.0), rotation_deg=0.0)
    other_a, other_b = non_matching_pair(noisy)

    def striae_correlation(first, second) -> float:
        # Compare the high frequency content only, since the form, the firing
        # pin and the drag mark are identical in every surface here and would
        # correlate whatever the striae did.
        def detrended(surface):
            band = surface.heights_um[-48:, :]
            return (band - band.mean(axis=1, keepdims=True)).ravel()

        left, right = detrended(first), detrended(second)
        return float(np.corrcoef(left, right)[0, 1])

    matched = striae_correlation(same_a, same_b)
    unmatched = striae_correlation(other_a, other_b)

    # Measured separation on this construction is about 0.9 against about
    # 0.2. The thresholds sit inside that gap on both sides, so the test fails
    # if the two classes move towards each other rather than only if they swap.
    assert matched > 0.8, f"a pair built from one source correlates only {matched}"
    assert unmatched < 0.5, f"a pair built from two sources correlates {unmatched}"
    assert same_a.source_id == same_b.source_id
    assert other_a.source_id != other_b.source_id


def test_a_matching_pair_records_the_offset_it_was_given() -> None:
    _, moved = matching_pair(BASE, translation_px=(5.0, -3.0), rotation_deg=11.0)
    assert moved.translation_px == (5.0, -3.0)
    assert_close(
        moved.rotation_deg,
        11.0,
        what="rotation recorded on the moved half of a matching pair",
        atol=0.0,
    )


def test_a_rotation_moves_the_striae_and_leaves_the_firing_pin_alone() -> None:
    upright = generate(BASE)
    turned = generate(BASE, rotation_deg=30.0)

    centre = (BASE.rows // 2, BASE.columns // 2)
    assert_close(
        turned.heights_um[centre],
        upright.heights_um[centre] - 0.0,
        what="height at the centre of the firing pin, rotated against upright",
        atol=BASE.striae_depth_um,
    )
    assert not np.array_equal(upright.heights_um, turned.heights_um), (
        "rotating the striae changed nothing"
    )


def test_a_parameter_set_that_cannot_be_represented_is_refused() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        SurfaceParameters(pixel_spacing_um=10.0, striae_spacing_um=10.0)


def test_a_negative_noise_level_is_refused() -> None:
    with pytest.raises(ValueError, match="noise"):
        SurfaceParameters(noise_um=-1.0)


def test_an_edge_dropout_that_would_leave_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="middle"):
        SurfaceParameters(edge_dropout_fraction=0.5)


def test_a_field_too_small_to_be_a_surface_is_refused() -> None:
    with pytest.raises(ValueError, match="not a surface"):
        SurfaceParameters(rows=4, columns=4)


def test_a_non_matching_pair_refuses_to_use_one_source_twice() -> None:
    with pytest.raises(ValueError, match="two different sources"):
        non_matching_pair(BASE, other_source_id=BASE.seed)
