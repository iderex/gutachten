"""Synthetic breech face impressions with ground truth by construction.

The suite needs surfaces to test against before any real scan is read, and it
needs them offline. Without a generator the only test data is a download, and
the gate would either reach the network or test nothing.

Ground truth here is not an opinion. A surface is built from parameters, so the
striae spacing, the firing pin radius, the noise level and the tilt are known
because they were asked for, not because somebody measured them afterwards and
agreed. That is what lets a preprocessing step be tested for what it claims: a
levelling step is given a surface with a known tilt and has to remove it, a
bandpass is given a known spatial frequency and has to attenuate it by a known
amount.

Two surfaces from one source share their striae, with a translation and a
rotation this module applies and therefore knows. Two from different sources do
not. So matching and non-matching pairs exist by construction, and a comparison
that gets them the wrong way round is caught by a test rather than by a study.

**These are not real scans and nothing here claims they are.** A synthetic
surface is smooth where a real one is not, its noise is independent where real
noise is not, and a step that works on these can still fail on casework. The
reproduction check against real data in a later milestone is where that gap is
measured. This module does not close it and does not pretend to.

Determinism. Every draw comes from a `numpy.random.Generator` built from the
integer seed in the parameters, and no unseeded global state is touched, so the
same parameters give the same array. That is a property the suite asserts rather
than a claim made here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "SurfaceParameters",
    "SyntheticSurface",
    "generate",
    "matching_pair",
    "non_matching_pair",
]


@dataclass(frozen=True)
class SurfaceParameters:
    """Everything that decides a synthetic surface, with nothing implicit.

    Lengths are micrometres and angles are degrees, stated here because a number
    without its unit is the most expensive kind of number in this project.

    A parameter set is frozen so that a surface cannot be produced from one set
    of numbers and reported against another.
    """

    rows: int = 256
    columns: int = 256
    #: Distance on the surface between neighbouring samples.
    pixel_spacing_um: float = 4.0

    #: Peak to peak amplitude of the striae, and the distance between them.
    striae_depth_um: float = 2.0
    striae_spacing_um: float = 40.0
    #: Direction the striae run, measured from the row axis.
    striae_angle_deg: float = 0.0

    #: A gentle bowl, which is the form a levelling step has to remove. This is
    #: the peak to peak height of the quadratic across the whole field.
    form_depth_um: float = 12.0

    #: The circular firing pin impression at the centre of the field.
    firing_pin_radius_um: float = 160.0
    firing_pin_depth_um: float = 25.0

    #: A drag mark: a straight groove of a stated width running across the field.
    drag_mark_width_um: float = 30.0
    drag_mark_depth_um: float = 6.0
    #: Where the drag mark crosses the field, as a fraction of the row extent.
    drag_mark_position: float = 0.25

    #: Standard deviation of the independent measurement noise.
    noise_um: float = 0.2

    #: Fraction of the field, at each edge, with no measurement at all. Real
    #: scans lose data at the edge and every published preprocessing chain
    #: starts by cutting it off, so a surface that cannot express it cannot test
    #: that step.
    edge_dropout_fraction: float = 0.0

    seed: int = 0

    def __post_init__(self) -> None:
        if self.rows < 8 or self.columns < 8:
            raise ValueError(
                f"a surface smaller than 8 by 8 is not a surface: {self.rows}x{self.columns}"
            )
        if self.pixel_spacing_um <= 0.0:
            raise ValueError(f"pixel spacing must be positive, got {self.pixel_spacing_um}")
        if self.striae_spacing_um <= 0.0:
            raise ValueError(f"striae spacing must be positive, got {self.striae_spacing_um}")
        if self.noise_um < 0.0:
            raise ValueError(f"noise cannot be negative, got {self.noise_um}")
        if not 0.0 <= self.edge_dropout_fraction < 0.5:
            raise ValueError(
                f"edge dropout is a fraction of each edge and must leave a middle, "
                f"got {self.edge_dropout_fraction}"
            )
        if self.striae_spacing_um < 2.0 * self.pixel_spacing_um:
            # Below two samples per period the striae are not represented at all,
            # they are aliased into something else. Refusing here means a test
            # asking for an unsamplable spacing fails at the request rather than
            # passing against a pattern that is not the one it named.
            raise ValueError(
                f"striae spacing {self.striae_spacing_um} um is below the Nyquist "
                f"limit for a pixel spacing of {self.pixel_spacing_um} um; the "
                f"pattern asked for cannot be represented on this grid"
            )


@dataclass(frozen=True)
class SyntheticSurface:
    """A generated height field and the parameters that produced it."""

    #: Height in micrometres, row by column. NaN marks the absence of a
    #: measurement, which is a different thing from a height of zero.
    heights_um: np.ndarray
    parameters: SurfaceParameters
    #: Where this surface's striae came from. Two surfaces sharing a source are
    #: a matching pair; two with different sources are not. It is recorded so a
    #: test asserts against the construction rather than against a guess.
    source_id: int
    #: Translation and rotation applied to this surface relative to its source.
    translation_px: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0

    @property
    def observed(self) -> np.ndarray:
        """The heights with missing data removed, for statistics over them."""
        return self.heights_um[np.isfinite(self.heights_um)]


def _coordinates(parameters: SurfaceParameters) -> tuple[np.ndarray, np.ndarray]:
    """Physical coordinates in micrometres, centred on the field."""
    rows = np.arange(parameters.rows, dtype=np.float64) - (parameters.rows - 1) / 2.0
    columns = np.arange(parameters.columns, dtype=np.float64) - (parameters.columns - 1) / 2.0
    row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
    return row_grid * parameters.pixel_spacing_um, column_grid * parameters.pixel_spacing_um


def _striae(
    y_um: np.ndarray,
    x_um: np.ndarray,
    parameters: SurfaceParameters,
    source_id: int,
    translation_px: tuple[float, float],
    rotation_deg: float,
) -> np.ndarray:
    """A striation pattern, rotated and translated by a known amount.

    The phase of every striation comes from the source seed, so two surfaces
    built from one source carry the same pattern and two from different sources
    do not. The rotation and translation are applied to the coordinates rather
    than to the finished image, so no interpolation is introduced and the
    relationship between the two surfaces stays exact.
    """
    angle = math.radians(parameters.striae_angle_deg + rotation_deg)
    shift_y = translation_px[0] * parameters.pixel_spacing_um
    shift_x = translation_px[1] * parameters.pixel_spacing_um

    # Distance along the direction the striae vary in, which is normal to the
    # direction they run.
    across = (x_um - shift_x) * math.cos(angle) + (y_um - shift_y) * math.sin(angle)

    # The pattern is a band of components clustered around the requested
    # spacing, with amplitudes, phases and small frequency offsets all drawn
    # from the source. A single sinusoid with a random phase would not do: two
    # different sources would then differ only in phase, would still correlate
    # near one, and a non-matching pair would be indistinguishable from a
    # matching one. That was measured rather than supposed, and it is the
    # reason for the shape of this function.
    source = np.random.default_rng(source_id)
    components = 16
    spread = 0.08
    base_frequency = 1.0 / parameters.striae_spacing_um
    offsets = source.uniform(-spread, spread, size=components)
    amplitudes = source.uniform(0.3, 1.0, size=components)
    phases = source.uniform(0.0, 2.0 * math.pi, size=components)

    wave = np.zeros_like(across)
    for offset, amplitude, phase in zip(offsets, amplitudes, phases, strict=True):
        wave += amplitude * np.sin(2.0 * math.pi * base_frequency * (1.0 + offset) * across + phase)

    # Rescale to exactly the requested peak to peak, so the depth parameter is
    # true by construction rather than true on average. A sum of components with
    # random phases realises whatever amplitude it realises, and a test asking
    # whether the striae are as deep as they were asked to be would otherwise be
    # testing the draw.
    span = float(wave.max() - wave.min())
    if span == 0.0:
        return np.zeros_like(across)
    return (
        parameters.striae_depth_um * (wave - wave.min()) / span - parameters.striae_depth_um / 2.0
    )


def _form(y_um: np.ndarray, x_um: np.ndarray, parameters: SurfaceParameters) -> np.ndarray:
    """A quadratic bowl of the stated peak to peak depth."""
    radius = np.hypot(y_um, x_um)
    largest = float(np.max(radius))
    if largest == 0.0:
        return np.zeros_like(radius)
    bowl: np.ndarray = parameters.form_depth_um * (radius / largest) ** 2
    return bowl


def _firing_pin(y_um: np.ndarray, x_um: np.ndarray, parameters: SurfaceParameters) -> np.ndarray:
    """A circular impression at the centre, of the stated radius and depth."""
    inside = np.hypot(y_um, x_um) <= parameters.firing_pin_radius_um
    return np.where(inside, -parameters.firing_pin_depth_um, 0.0)


def _drag_mark(y_um: np.ndarray, x_um: np.ndarray, parameters: SurfaceParameters) -> np.ndarray:
    """A straight groove of the stated width, running along the column axis."""
    extent = float(parameters.rows * parameters.pixel_spacing_um)
    centre = (parameters.drag_mark_position - 0.5) * extent
    inside = np.abs(y_um - centre) <= parameters.drag_mark_width_um / 2.0
    return np.where(inside, -parameters.drag_mark_depth_um, 0.0)


def _apply_edge_dropout(heights: np.ndarray, parameters: SurfaceParameters) -> np.ndarray:
    """Replace a border of the field with NaN, as a real scan loses its edge."""
    if parameters.edge_dropout_fraction == 0.0:
        return heights
    down = round(parameters.rows * parameters.edge_dropout_fraction)
    across = round(parameters.columns * parameters.edge_dropout_fraction)
    if down == 0 and across == 0:
        return heights
    out = heights.copy()
    if down:
        out[:down, :] = np.nan
        out[-down:, :] = np.nan
    if across:
        out[:, :across] = np.nan
        out[:, -across:] = np.nan
    return out


def generate(
    parameters: SurfaceParameters,
    *,
    source_id: int | None = None,
    translation_px: tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> SyntheticSurface:
    """Build a surface from ``parameters``.

    ``source_id`` decides the striation pattern and defaults to the seed, so a
    caller who says nothing gets a surface whose striae belong to it alone.
    ``translation_px`` and ``rotation_deg`` move the striae by a known amount
    without moving the form, the firing pin or the drag mark, which is what
    happens when the same breech face marks the same cartridge twice at a
    different orientation.
    """
    identity = parameters.seed if source_id is None else source_id
    y_um, x_um = _coordinates(parameters)

    heights = (
        _form(y_um, x_um, parameters)
        + _striae(y_um, x_um, parameters, identity, translation_px, rotation_deg)
        + _firing_pin(y_um, x_um, parameters)
        + _drag_mark(y_um, x_um, parameters)
    )

    if parameters.noise_um > 0.0:
        # The noise is drawn from the surface's own seed rather than from the
        # source, so two surfaces sharing a source still differ in the way two
        # measurements of one object differ.
        noise_rng = np.random.default_rng((parameters.seed, identity))
        heights = heights + noise_rng.normal(0.0, parameters.noise_um, size=heights.shape)

    heights = _apply_edge_dropout(heights, parameters)

    return SyntheticSurface(
        heights_um=heights,
        parameters=parameters,
        source_id=identity,
        translation_px=translation_px,
        rotation_deg=rotation_deg,
    )


def matching_pair(
    parameters: SurfaceParameters,
    *,
    translation_px: tuple[float, float] = (3.0, -2.0),
    rotation_deg: float = 7.0,
) -> tuple[SyntheticSurface, SyntheticSurface]:
    """Two surfaces of one source, at a stated offset. A known match.

    Produced by an explicit call rather than by two independent draws that
    happen to agree, so a test asserting a match is asserting against the
    construction.
    """
    source_id = parameters.seed
    first = generate(parameters, source_id=source_id)
    second = generate(
        replace(parameters, seed=parameters.seed + 1),
        source_id=source_id,
        translation_px=translation_px,
        rotation_deg=rotation_deg,
    )
    return first, second


def non_matching_pair(
    parameters: SurfaceParameters,
    *,
    other_source_id: int | None = None,
) -> tuple[SyntheticSurface, SyntheticSurface]:
    """Two surfaces of different sources. A known non-match."""
    source_id = parameters.seed
    second_source = source_id + 10_007 if other_source_id is None else other_source_id
    if second_source == source_id:
        raise ValueError("a non-matching pair needs two different sources")
    first = generate(parameters, source_id=source_id)
    second = generate(replace(parameters, seed=parameters.seed + 1), source_id=second_source)
    return first, second
