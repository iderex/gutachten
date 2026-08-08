"""The bandpass, against surfaces whose spatial frequency content is known.

A sinusoid of a stated wavelength is what the transmission characteristic is
about, so the fixture here is a sinusoid rather than a generated breech face.
The amplitude at that wavelength is recovered by projecting onto it, which is
exact for the frequency asked about and does not care what else is present.

Every number quoted here was measured by running the step at this commit, and
every refusal was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.transforms import bandpass
from gutachten.transforms.bandpass import (
    ALPHA,
    STANDARD,
    BandpassParameters,
    RobustGaussianBandpass,
    band_transmission,
    transmission,
)
from gutachten.transforms.pipeline import Step, run_chain
from gutachten.transforms.registry import REGISTRY, Registry
from tests.support.tolerance import assert_close

ROWS = 192
COLUMNS = 192
SPACING_UM = 2.0
SHORT_UM = 20.0
LONG_UM = 120.0
#: Where the mask goes in the tests that use one, and it is well inside the
#: field so the edge of the field is not what those tests are measuring.
HOLE = (slice(80, 112), slice(80, 112))


def columns_um() -> np.ndarray:
    """The physical position of each sample along the column axis."""
    across = np.arange(COLUMNS, dtype=np.float64) * SPACING_UM
    _, x_um = np.meshgrid(np.arange(ROWS, dtype=np.float64) * SPACING_UM, across, indexing="ij")
    return np.asarray(x_um)


def a_sinusoid(wavelength_um: float) -> np.ndarray:
    """A unit amplitude sinusoid running across the columns."""
    return np.asarray(np.sin(2 * math.pi * columns_um() / wavelength_um))


def as_surface(heights: np.ndarray) -> Surface:
    return Surface(
        heights=heights,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def settings(tuning: float | None = None, passes: int | None = None) -> BandpassParameters:
    return BandpassParameters(
        short_cutoff=SHORT_UM,
        long_cutoff=LONG_UM,
        robust_tuning=tuning,
        robust_passes=passes,
    )


def filtered(heights: np.ndarray, parameters: BandpassParameters) -> np.ndarray:
    return np.asarray(RobustGaussianBandpass().apply(as_surface(heights), parameters).heights)


def amplitude_at(field: np.ndarray, wavelength_um: float) -> float:
    """The amplitude of ``field`` at ``wavelength_um``, over the middle third.

    The middle third rather than the whole field. The kernel is truncated at the
    boundary and the weighted mean there is taken over fewer samples, which is
    the correct answer for a surface that stops and the wrong place to measure a
    transmission.
    """
    core = (slice(ROWS // 3, 2 * ROWS // 3), slice(COLUMNS // 3, 2 * COLUMNS // 3))
    wavenumber = 2 * math.pi / wavelength_um
    position = columns_um()[core]
    values = field[core]
    cosine = 2 * float(np.mean(values * np.cos(wavenumber * position)))
    sine = 2 * float(np.mean(values * np.sin(wavenumber * position)))
    return float(math.hypot(cosine, sine))


def test_the_weighting_constant_is_what_puts_the_cutoff_at_half() -> None:
    # The specified value is a consequence of the constant rather than a number
    # beside it, and this is where that is asserted rather than assumed. If
    # ALPHA is ever typed as a decimal, this is what notices.
    assert_close(
        transmission(SHORT_UM, SHORT_UM),
        0.5,
        what="the smoothing transmission at its own cutoff wavelength",
        # Exact to the arithmetic: exp(-pi * ln(2) / pi) is exp(-ln 2).
        atol=1e-15,
    )
    assert_close(
        ALPHA,
        math.sqrt(math.log(2) / math.pi),
        what="the Gaussian weighting constant",
        atol=0.0,
    )


def test_the_measured_transmission_matches_the_specified_one_at_each_cutoff() -> None:
    # The clause this issue turns on. The specified value is computed from the
    # weighting rather than transcribed, so this compares a measurement against
    # a derivation.
    for wavelength in (SHORT_UM, LONG_UM):
        heights = a_sinusoid(wavelength)
        band = filtered(heights, settings())

        measured = amplitude_at(band, wavelength) / amplitude_at(heights, wavelength)
        assert_close(
            measured,
            band_transmission(wavelength, SHORT_UM, LONG_UM),
            what=f"the transmission of the band at {wavelength} micrometres",
            # Measured at 1.0e-10 and 3.9e-11 at this commit. The bound is two
            # decimal orders above, which is still seven below anything that
            # would show in a score.
            atol=1e-8,
        )


def test_the_transmission_follows_the_characteristic_across_the_band_and_outside_it() -> None:
    # Two cutoffs on their own would pass for a filter that happened to halve
    # everything. These are inside the band, below the short cutoff and above
    # the long one.
    for wavelength in (45.0, 8.0, 400.0):
        heights = a_sinusoid(wavelength)
        band = filtered(heights, settings())

        measured = amplitude_at(band, wavelength) / amplitude_at(heights, wavelength)
        assert_close(
            measured,
            band_transmission(wavelength, SHORT_UM, LONG_UM),
            what=f"the transmission of the band at {wavelength} micrometres",
            atol=1e-8,
        )
    # And the middle of the band is where most of a sinusoid survives, against
    # 1.3 per cent below the short cutoff and 5.9 per cent above the long one.
    assert_close(
        band_transmission(45.0, SHORT_UM, LONG_UM),
        0.8648,
        what="the specified transmission in the middle of the band",
        atol=0.0001,
    )


def test_a_masked_region_puts_no_step_at_its_own_boundary() -> None:
    # The failure the issue names. A constant surface has nothing in any band,
    # so anything this filter returns for one is an artefact and its size is
    # readable directly.
    constant = np.full((ROWS, COLUMNS), 7.0)
    holed = np.array(constant, dtype=np.float64, copy=True)
    holed[HOLE] = np.nan
    measured = ~np.isnan(holed)

    band = filtered(holed, settings())

    assert_close(
        band[measured],
        np.zeros_like(band[measured]),
        what="what the filter leaves of a constant surface with a hole in it",
        # Measured at 6.2e-15 micrometres at this commit, which is the same
        # number the identical surface without a hole produces.
        atol=1e-12,
    )


def test_filling_the_mask_instead_is_what_the_artefact_looks_like() -> None:
    # Measured rather than described, and on the same surface as the test above,
    # so the two numbers are comparable: 6.2e-15 against 3.95.
    constant = np.full((ROWS, COLUMNS), 7.0)
    filled = np.array(constant, dtype=np.float64, copy=True)
    filled[HOLE] = 0.0

    band = filtered(filled, settings())

    assert_close(
        float(np.max(np.abs(band))),
        3.952,
        what="the step a mask filled with zero leaves in the band",
        atol=0.001,
    )


def test_a_mask_does_not_reach_the_surface_far_away_from_it() -> None:
    # The other half of handling the mask explicitly. A region that contributes
    # nothing must also take nothing away from samples the kernel never joins
    # to it.
    heights = a_sinusoid(45.0)
    holed = np.array(heights, dtype=np.float64, copy=True)
    holed[HOLE] = np.nan
    far = (slice(8, 40), slice(8, 40))

    whole = filtered(heights, settings())
    with_hole = filtered(holed, settings())

    assert_close(
        with_hole[far],
        whole[far],
        what="the band far from a mask, with the mask and without it",
        # Measured at 1.5e-08 micrometres at this commit. Not zero: the kernel
        # is truncated where the weighting stops changing a float64 sum, and
        # the residue is that truncation rather than the mask.
        atol=1e-06,
    )


def test_the_robust_filter_keeps_a_spike_out_of_the_band_and_the_linear_one_does_not() -> None:
    # What the robustness is for. One sample lifted, and the band around it read
    # against the band the same surface produces without the spike.
    clean = a_sinusoid(45.0)
    spiked = np.array(clean, dtype=np.float64, copy=True)
    spiked[ROWS // 2, COLUMNS // 2] += 200.0
    around = np.zeros((ROWS, COLUMNS), dtype=bool)
    around[ROWS // 2 - 4 : ROWS // 2 + 5, COLUMNS // 2 - 4 : COLUMNS // 2 + 5] = True
    around[ROWS // 2, COLUMNS // 2] = False

    reference = filtered(clean, settings())
    linear = filtered(spiked, settings())
    robust = filtered(spiked, settings(tuning=4.0, passes=3))

    # 7.61 against 0.021 micrometres, measured at this commit.
    assert_close(
        float(np.max(np.abs(linear[around] - reference[around]))),
        7.611,
        what="what one spike puts into the band under the linear filter",
        atol=0.001,
    )
    assert_close(
        float(np.max(np.abs(robust[around] - reference[around]))),
        0.0210,
        what="what one spike puts into the band under the robust filter",
        atol=0.0001,
    )


def test_what_the_robustness_costs_the_transmission_is_measured_not_assumed() -> None:
    # The reweighting is not linear, so the specified characteristic is the
    # linear filter's and a robust run does not have it. Stated as a number
    # here rather than left for somebody to find in a score.
    heights = a_sinusoid(SHORT_UM)

    linear = amplitude_at(filtered(heights, settings()), SHORT_UM)
    robust = amplitude_at(filtered(heights, settings(tuning=4.0, passes=3)), SHORT_UM)
    scale = amplitude_at(heights, SHORT_UM)

    assert_close(
        linear / scale,
        0.5,
        what="the linear transmission at the short cutoff",
        atol=1e-08,
    )
    assert_close(
        robust / scale,
        0.480074,
        what="the transmission at the short cutoff after three robust passes",
        # Two percentage points below the specified value. Measured at this
        # commit with a tuning constant of four.
        atol=1e-05,
    )


def test_a_surface_with_no_measured_sample_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="every sample of this surface is missing"):
        RobustGaussianBandpass().apply(as_surface(np.full((ROWS, COLUMNS), np.nan)), settings())


def test_the_missing_samples_stay_missing_and_the_shape_does_not_move() -> None:
    heights = a_sinusoid(45.0)
    heights[HOLE] = np.nan
    surface = as_surface(heights)

    result = RobustGaussianBandpass().apply(surface, settings())

    assert result.shape == surface.shape
    assert np.array_equal(result.missing, surface.missing)


def test_a_cutoff_that_is_not_a_positive_finite_wavelength_is_refused() -> None:
    heights = a_sinusoid(45.0)

    for bad in (0.0, -20.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="short_cutoff must be a positive finite wavelength"):
            filtered(
                heights,
                BandpassParameters(
                    short_cutoff=bad, long_cutoff=LONG_UM, robust_tuning=None, robust_passes=None
                ),
            )
        with pytest.raises(ValueError, match="long_cutoff must be a positive finite wavelength"):
            filtered(
                heights,
                BandpassParameters(
                    short_cutoff=SHORT_UM, long_cutoff=bad, robust_tuning=None, robust_passes=None
                ),
            )


def test_a_cutoff_that_is_not_a_number_is_refused() -> None:
    heights = a_sinusoid(45.0)

    with pytest.raises(TypeError, match="short_cutoff must be a wavelength"):
        filtered(
            heights,
            BandpassParameters(
                short_cutoff=True,  # type: ignore[arg-type]
                long_cutoff=LONG_UM,
                robust_tuning=None,
                robust_passes=None,
            ),
        )


def test_two_cutoffs_that_are_not_a_band_are_refused() -> None:
    # Swapped, the step would return a surface of nearly nothing and record
    # itself as a bandpass.
    heights = a_sinusoid(45.0)

    for short, long in ((LONG_UM, SHORT_UM), (SHORT_UM, SHORT_UM)):
        with pytest.raises(ValueError, match="is not shorter than the long cutoff"):
            filtered(
                heights,
                BandpassParameters(
                    short_cutoff=short, long_cutoff=long, robust_tuning=None, robust_passes=None
                ),
            )


def test_a_cutoff_below_the_sampling_is_refused() -> None:
    # The kernel would be a single tap and the filter would return its input
    # while recording that it had filtered it.
    heights = a_sinusoid(45.0)

    with pytest.raises(ValueError, match="less than one sample"):
        filtered(
            heights,
            BandpassParameters(
                short_cutoff=SPACING_UM / 8,
                long_cutoff=LONG_UM,
                robust_tuning=None,
                robust_passes=None,
            ),
        )


def test_half_a_robustness_setting_is_refused() -> None:
    heights = a_sinusoid(45.0)

    for tuning, passes in ((4.0, None), (None, 3)):
        with pytest.raises(ValueError, match="disagree about whether"):
            filtered(
                heights,
                BandpassParameters(
                    short_cutoff=SHORT_UM,
                    long_cutoff=LONG_UM,
                    robust_tuning=tuning,
                    robust_passes=passes,
                ),
            )


def test_a_tuning_constant_that_cannot_reach_anything_is_refused() -> None:
    heights = a_sinusoid(45.0)

    for tuning in (0.0, -4.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite number of residual scales"):
            filtered(heights, settings(tuning=tuning, passes=3))
    with pytest.raises(TypeError, match="robust_tuning must be a number"):
        filtered(heights, settings(tuning=True, passes=3))  # type: ignore[arg-type]


def test_a_pass_count_that_is_not_at_least_one_whole_reweighting_is_refused() -> None:
    heights = a_sinusoid(45.0)

    with pytest.raises(ValueError, match="robust_passes must be at least one"):
        filtered(heights, settings(tuning=4.0, passes=0))
    for passes in (3.0, True):
        with pytest.raises(TypeError, match="robust_passes must be a whole number"):
            filtered(heights, settings(tuning=4.0, passes=passes))  # type: ignore[arg-type]


def test_a_surface_already_at_its_own_mean_line_keeps_every_sample_at_full_weight() -> None:
    # A residual scale of zero. The reweighting has nothing to find, and a
    # division by it would put every sample at no weight and leave the next
    # smoothing with no data at all.
    result = filtered(np.zeros((ROWS, COLUMNS)), settings(tuning=4.0, passes=2))

    assert_close(
        result,
        np.zeros_like(result),
        what="a flat surface through the robust filter",
        # Exactly zero. There is nothing in any band and nothing was found.
        atol=0.0,
    )


def test_the_transmission_formula_refuses_a_length_that_is_not_one() -> None:
    with pytest.raises(ValueError, match="both positive lengths"):
        transmission(0.0, SHORT_UM)
    with pytest.raises(ValueError, match="both positive lengths"):
        transmission(SHORT_UM, -1.0)


def test_the_step_refuses_a_record_that_is_not_its_own_before_reading_a_field() -> None:
    class NotOurs:
        short_cutoff = SHORT_UM
        long_cutoff = LONG_UM
        robust_tuning = None
        robust_passes = None

    with pytest.raises(TypeError, match="rather than BandpassParameters, so nothing here has read"):
        RobustGaussianBandpass().apply(as_surface(a_sinusoid(45.0)), NotOurs())  # type: ignore[arg-type]


def test_the_code_and_the_documentation_name_the_same_part_of_the_standard() -> None:
    # The done-condition asks for the part implemented to be named in the code
    # and to match what the documentation claims. Two files stating one fact is
    # a fact that drifts, so this is what refuses the drift.
    documentation = Path(__file__).resolve().parents[3] / "docs" / "filtering.md"

    assert STANDARD == "ISO 16610-71:2014"
    assert STANDARD in documentation.read_text(encoding="utf-8")
    assert bandpass.__doc__ is not None
    assert STANDARD in bandpass.__doc__


def test_the_step_records_both_cutoffs_and_runs_in_a_chain() -> None:
    registry = Registry()
    registry.register(RobustGaussianBandpass())
    chain = [Step(identifier="bandpass", parameters=settings())]

    result = run_chain(chain, registry, as_surface(a_sinusoid(45.0)))

    entry = result.provenance[-1]
    assert entry.name == "bandpass"
    assert entry.version == "1"
    assert dict(entry.parameters) == {
        "long_cutoff": LONG_UM,
        "robust_passes": None,
        "robust_tuning": None,
        "short_cutoff": SHORT_UM,
    }


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

    assert "bandpass" in completed.stdout.strip().split(",")
    assert REGISTRY["bandpass"].version == "1"
