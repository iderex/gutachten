"""The robust Gaussian regression bandpass, with both cutoffs declared.

The comparison operates on a band of spatial wavelengths. The short cutoff
removes measurement noise and the long cutoff removes whatever form the
levelling left behind, and both numbers are chosen by the analyst. Method papers
state them as though they were settled. What moving them inside a plausible
range does to a score is the first thing this project should be able to report,
so neither has a default and the sweep can move both.

## Which standard, and which part of it

The filter is the robust areal Gaussian regression filter of

    ISO 16610-71:2014, Geometrical product specifications (GPS) - Filtration -
    Part 71: Robust areal filters: Gaussian regression filters

Areal rather than profile: the profile filters of the same series are
ISO 16610-31 for the robust Gaussian regression case, and a surface is not a
profile. The title and number above were read off the ISO catalogue entry for
the standard and cross-checked against the filtration guide published by the
Physikalisch-Technische Bundesanstalt, which names 16610-31 for profiles and
16610-71 for areal data in the same sentence. They are not quoted from memory.

WHAT IS IMPLEMENTED IS THE ZEROTH ORDER FORM, and this is the disclosure rather
than a footnote to it. The regression is over a locally weighted mean, which is
the degree zero case. The standard's higher degree variant, which combines the
Gaussian kernel with Savitzky-Golay coefficients to retain the shape of a peak,
is not implemented here and this module does not claim it. The text of the
standard is not public, so what could be verified is its number, its title and
the shape of the filter as described in the open literature; the constants the
standard fixes internally could not be read, which is why the one that matters
is a parameter here rather than a number written down as though it had been.

## The kernel, and where its constant comes from

The Gaussian weighting is

    s(x) = exp(-pi * (x / (alpha * cutoff))^2),   alpha = sqrt(ln(2) / pi)

and ``alpha`` is what makes the transmission exactly one half at the cutoff
wavelength, which is the whole point of it: the transmission of that weighting
at a wavelength ``w`` is ``exp(-pi * (alpha * cutoff / w)^2)``, and at
``w = cutoff`` that is ``exp(-ln 2)``. So the specified value the tests compare
against is derived here rather than quoted, and ``alpha`` is computed from
``ln(2)`` and ``pi`` rather than typed as ``0.4697``.

## The band is a difference of two smoothings

Smoothing at the short cutoff leaves everything longer than it. Smoothing at the
long cutoff leaves everything longer than that. The band between them is the
first minus the second, so a wavelength at either cutoff arrives at half its
amplitude and the two ends of the band are symmetric.

## Missing samples are weighted out, never filled

This is the part the issue exists to force into the open. A filter that treats
not-a-number as zero puts a step at the edge of every mask, and the step is a
feature of the same size as the ones being compared. Here the kernel is applied
to the measured samples and to the mask separately, and the result is their
ratio:

    smooth = convolve(height * weight) / convolve(weight)

with the weight zero where there is no measurement. A masked region therefore
contributes nothing and biases nothing, the samples beside it are a weighted
mean of what was actually measured near them, and a constant surface with a hole
in it filters to the same constant. The edge of the field is the same case and
needs no separate rule.

## Robustness is two numbers or neither

``robust_tuning`` and ``robust_passes`` are both present or both absent. Absent,
this is the linear Gaussian regression filter and its transmission is the
specified one. Present, each pass recomputes a Tukey biweight weight from the
residual and folds it into the weighting above, which is what lets a deep valley
or a residual spike stop dragging the mean line toward itself.

The tuning constant is a parameter rather than a number in this file. The
standard fixes one and the standard is not readable here, and a constant taken
from somewhere else and presented as the standard's would be exactly the kind of
unsourced number this project is arguing against. It is also the honest shape:
it changes the result, so the sweep is entitled to it.

A fixed pass count rather than a convergence tolerance, for the reason the
levelling step gives: a tolerance is an undeclared number deciding how many
passes ran, and the count would then depend on the platform's floating point.

## What robustness costs the transmission, measured

The transmission characteristic above is the linear filter's. The reweighting is
not linear, so a robust run does not have it: on a pure sinusoid the residual is
the sinusoid, and the biweight pulls down its own peaks. That is a real property
of the filter rather than a defect, the tests measure how far it moves, and the
number is in ``tests/unit/transforms/test_bandpass.py`` beside the linear one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from gutachten.surface import Surface
from gutachten.transforms.base import Parameters, SurfaceProperty, record_for
from gutachten.transforms.registry import REGISTRY

__all__ = ["ALPHA", "STANDARD", "BandpassParameters", "RobustGaussianBandpass"]

#: The part of ISO 16610 this module implements, in the form a reader can look
#: up. Named here rather than in a comment so a test can assert that the code
#: and the documentation name the same part.
STANDARD = "ISO 16610-71:2014"

#: The constant of the Gaussian weighting, which is what puts the transmission
#: at exactly one half at the cutoff wavelength. Computed rather than typed:
#: written as a decimal it would be a number somebody would eventually have to
#: check, and the check is this expression.
ALPHA = math.sqrt(math.log(2) / math.pi)


@dataclass(frozen=True)
class BandpassParameters:
    """The band, and how the fit that finds it is made.

    ``short_cutoff`` and ``long_cutoff`` are wavelengths, as lengths in the
    surface's own unit. ``robust_tuning`` and ``robust_passes`` are both
    ``None`` for the linear filter and both set for the robust one.
    """

    short_cutoff: float
    long_cutoff: float
    robust_tuning: float | None
    robust_passes: int | None


class RobustGaussianBandpass:
    """Keep the band between two declared cutoff wavelengths."""

    identifier = "bandpass"
    version = "1"
    parameters_type = BandpassParameters
    produces = frozenset({SurfaceProperty.FILTERED})
    requires = frozenset[SurfaceProperty]()
    refuses = frozenset[SurfaceProperty]()

    def apply(self, surface: Surface, parameters: Parameters) -> Surface:
        # The pipeline checks the record type before the chain starts and
        # `record_for` checks it again on the way out. This one narrows the type
        # for the checker and refuses before a single field has been read.
        if not isinstance(parameters, BandpassParameters):
            raise TypeError(
                f"{self.identifier!r} was handed {type(parameters).__name__} rather than "
                "BandpassParameters, so nothing here has read a field off it"
            )
        short, long = _cutoffs(parameters)
        tuning, passes = _robustness(parameters)

        measured = ~surface.missing
        if not np.any(measured):
            raise ValueError(
                "every sample of this surface is missing, so there is nothing to filter. "
                "A step that ran over an empty surface would record itself as having "
                "filtered one."
            )

        heights = np.asarray(surface.heights)
        weight = measured.astype(np.float64)
        for _ in range(passes):
            residual = heights - _smooth(heights, weight, short, surface)
            weight = measured * _biweight(residual[measured], residual, tuning)

        near = _smooth(heights, weight, short, surface)
        far = _smooth(heights, weight, long, surface)
        band = np.where(measured, near - far, np.nan)

        return surface.with_transform(record_for(self, parameters), band)


def _cutoffs(parameters: BandpassParameters) -> tuple[float, float]:
    """The two wavelengths, refusing a pair that is not a band.

    Both are checked before either is used, so a message names the field that is
    wrong rather than the arithmetic that failed on it.
    """
    named = (("short_cutoff", parameters.short_cutoff), ("long_cutoff", parameters.long_cutoff))
    for name, value in named:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a wavelength, got {value!r}")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be a positive finite wavelength, got {value!r}. There is no "
                "default: which band a comparison is made in is the choice this step "
                "exists to expose."
            )

    short = float(parameters.short_cutoff)
    long = float(parameters.long_cutoff)
    if short >= long:
        raise ValueError(
            f"the short cutoff {short} is not shorter than the long cutoff {long}, so the "
            "band between them is empty or inverted. A run with the two swapped would "
            "return a surface of nearly nothing and record itself as a bandpass."
        )
    return short, long


def _robustness(parameters: BandpassParameters) -> tuple[float, int]:
    """The two robustness settings, refusing one without the other.

    Returns the tuning constant and the number of passes. The linear filter
    reports zero passes and never enters the loop, so the tuning constant it
    reports is never read.
    """
    tuning = parameters.robust_tuning
    passes = parameters.robust_passes

    if (tuning is None) != (passes is None):
        raise ValueError(
            f"robust_tuning={tuning!r} and robust_passes={passes!r} disagree about whether "
            "this filter is robust. Both are None for the linear Gaussian regression "
            "filter and both are set for the robust one, because a tuning constant "
            "nothing reweights and a count of reweightings with no constant to reweight "
            "by are each half a setting."
        )
    if tuning is None or passes is None:
        return 0.0, 0

    if isinstance(tuning, bool) or not isinstance(tuning, (int, float)):
        raise TypeError(f"robust_tuning must be a number of residual scales, got {tuning!r}")
    if not np.isfinite(tuning) or tuning <= 0.0:
        raise ValueError(
            f"robust_tuning must be a positive finite number of residual scales, got "
            f"{tuning!r}. It is where the biweight reaches zero, and a reach of nothing "
            "puts every sample at zero weight including the ones the mean line is made of."
        )
    if isinstance(passes, bool) or not isinstance(passes, int):
        raise TypeError(f"robust_passes must be a whole number of reweightings, got {passes!r}")
    if passes < 1:
        raise ValueError(
            f"robust_passes must be at least one, got {passes!r}. A robust filter that "
            "reweights nothing is the linear filter recorded under a name that says it is "
            "not."
        )
    return float(tuning), passes


def _biweight(observed: np.ndarray, residual: np.ndarray, tuning: float) -> np.ndarray:
    """Tukey's biweight of ``residual``, in units of the residual's own scale.

    The scale is the median absolute deviation of the measured residuals,
    normalised so that it estimates the standard deviation of a normal. A scale
    of zero means the mean line already passes through more than half the
    samples exactly, and there is nothing for a reweighting to find; every
    sample keeps full weight rather than the division being made.
    """
    scale = _MAD_TO_SIGMA * float(np.median(np.abs(observed - np.median(observed))))
    if scale == 0.0:
        return np.ones_like(residual)
    scaled = residual / (tuning * scale)
    inside = np.abs(scaled) < 1.0
    return np.asarray(np.where(inside, (1.0 - np.where(inside, scaled, 0.0) ** 2) ** 2, 0.0))


#: What the median absolute deviation has to be multiplied by to estimate the
#: standard deviation of a normal. Fixed by the distribution rather than chosen.
_MAD_TO_SIGMA = 1.4826  # structural: the normal-consistency factor of the median absolute deviation


def _smooth(heights: np.ndarray, weight: np.ndarray, cutoff: float, surface: Surface) -> np.ndarray:
    """The Gaussian weighted mean of ``heights`` at ``cutoff``, given ``weight``.

    The kernel is applied to the weighted heights and to the weights, and the
    result is their ratio. That is what keeps a masked region from contributing
    a value it does not have, and it makes the boundary of the field the same
    case as the boundary of a mask rather than a separate rule.

    Separable, because a two dimensional Gaussian is the product of two one
    dimensional ones and the two axes have their own spacings.
    """
    filled = np.where(np.isfinite(heights), heights, 0.0)
    numerator = _separable(filled * weight, cutoff, surface)
    denominator = _separable(weight, cutoff, surface)
    # A sample whose whole kernel reaches nothing measured has no weighted mean.
    # It stays not-a-number rather than being given the ratio of two zeros, and
    # the caller masks it back out.
    return np.asarray(
        np.where(
            denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), np.nan
        )
    )


def _separable(values: np.ndarray, cutoff: float, surface: Surface) -> np.ndarray:
    """``values`` convolved with the Gaussian weighting along both axes."""
    down = ndimage.convolve1d(
        values, _kernel(cutoff, surface.spacing_y), axis=0, mode="constant", cval=0.0
    )
    across = ndimage.convolve1d(
        down, _kernel(cutoff, surface.spacing_x), axis=1, mode="constant", cval=0.0
    )
    return np.asarray(across)


def _kernel(cutoff: float, spacing: float) -> np.ndarray:
    """The Gaussian weighting, sampled at ``spacing`` and truncated where it vanishes.

    The truncation radius is derived rather than chosen: it is where the
    weighting falls below the smallest number that changes a float64 sum of
    values of order one. Truncating earlier would be a decision about the filter
    that nobody recorded, and truncating later would add terms that cannot move
    the answer.

    Not normalised here. The caller divides by the same kernel applied to the
    weights, so any constant factor cancels, and the normalisation that matters
    is the one that accounts for the missing samples rather than one computed
    over a kernel that assumes none.
    """
    width = ALPHA * cutoff
    reach = width * math.sqrt(math.log(1.0 / np.finfo(np.float64).eps) / math.pi)
    samples = int(reach / spacing)
    if samples < 1:
        raise ValueError(
            f"a cutoff of {cutoff} reaches {reach} at a sample spacing of {spacing}, which "
            "is less than one sample, so the kernel would be a single tap and the filter "
            "would return its input while recording that it had filtered it. The cutoff "
            "has to be longer than the instrument's own sampling."
        )
    offsets = np.arange(-samples, samples + 1, dtype=np.float64) * spacing
    return np.asarray(np.exp(-math.pi * (offsets / width) ** 2))


def transmission(wavelength: float, cutoff: float) -> float:
    """What fraction of a sinusoid of ``wavelength`` a smoothing at ``cutoff`` keeps.

    The specified characteristic of the Gaussian weighting, written as the
    formula rather than as a table, so a test compares a measurement against a
    derivation instead of against a number somebody transcribed. At
    ``wavelength == cutoff`` it is exactly one half.
    """
    if wavelength <= 0.0 or cutoff <= 0.0:
        raise ValueError(
            f"a transmission is asked at a wavelength and a cutoff, both positive lengths; "
            f"got wavelength={wavelength!r} cutoff={cutoff!r}"
        )
    return float(math.exp(-math.pi * (ALPHA * cutoff / wavelength) ** 2))


def band_transmission(wavelength: float, short_cutoff: float, long_cutoff: float) -> float:
    """What fraction of a sinusoid of ``wavelength`` the band between two cutoffs keeps.

    The band is the difference of the two smoothings, so its transmission is the
    difference of theirs. At either cutoff this is one half less what the other
    smoothing happens to keep there, which is why the value the tests compare
    against is not flatly one half and is computed for the pair in use.
    """
    return transmission(wavelength, short_cutoff) - transmission(wavelength, long_cutoff)


REGISTRY.register(RobustGaussianBandpass())
