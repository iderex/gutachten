"""Removing the form, with the model that gets removed declared rather than assumed.

A scan of a cartridge case carries two things nobody wants to compare: the tilt
of how the case sat on the stage, and the curvature of the primer itself. Both
are removed before comparison, and which of them is removed is a choice rather
than a fact. A plane takes the tilt and leaves the curvature. A second order
polynomial takes both, and takes with them whatever genuine low frequency breech
face structure happens to look quadratic. A fitted sphere takes the primer's own
shape and leaves the part of the curvature that is not spherical.

Each removes a different amount of what is arguably real, so the model is a
parameter, it is in the manifest, and the sweep can move it.

## The three models, and why a plane has its own name

``FormModel.PLANE`` is the first order polynomial and is computed as one. It
carries its own name because it is the model people actually mean, and because
naming it costs nothing: a first order polynomial is then refused under
``FormModel.POLYNOMIAL`` and pointed here. Without that refusal one form would
be reachable under two names, a sweep enumerating the parameter space would
visit it twice, and two rows of a sensitivity table would differ only in the
word recorded against them.

``FormModel.SPHERE`` is a genuine sphere and not a paraboloid wearing the name.
It is fitted in the algebraic form, which is linear in the sphere's centre and
radius, and evaluated on the branch that lies nearer the measured heights. The
residual it minimises is therefore the algebraic one rather than the vertical
distance, which is the standard trade and is worth stating: for a shallow cap,
which is what a primer is, the two agree closely, and for a deep one they do
not.

## The order is the polynomial's alone

``order`` is the polynomial's order and is refused for the other two models,
which have no order to choose. Recording ``order`` against a plane would put a
number in the manifest that decided nothing, and a sweep reading the manifest
would move it and report that the step is insensitive to it, which is true and
useless.

## Masked samples are excluded from the fit, and that is not a setting

Fitting over the masked region rather than around it is the quiet error this
step is most likely to be broken by. Not-a-number does not survive a least
squares solve, so a step that did not exclude it would not fit wrongly, it would
fail loudly; the shape the error actually takes is a mask filled with a
plausible constant somewhere upstream, and the fit then leans toward whatever
that constant was. So the observed samples are selected by the surface's own
mask, and a test measures how far the two answers are apart rather than
asserting that they are not.

The form is evaluated at every sample including the missing ones, so the surface
that comes out has the same shape and the same mask as the one that went in. A
missing height minus a number is still missing.

## The boundary, and why there is no rule to declare for it

The fit is global least squares over every observed sample, and the model is
evaluated from the same coefficients everywhere. There is no interior rule and
no separate edge rule for a reader to choose between, so nothing about the
boundary is a parameter here. What the model does at the edge of the field is
what the model does: the polynomial continues, and the sphere is refused
outright where the fitted radius does not span the field, because a sphere
evaluated past its own rim would have to be clamped, and a clamp puts a flat rim
on a levelled surface that no later step can tell from data.

## Robustness is two numbers or neither

``robust_tuning`` and ``robust_passes`` are both present or both absent.
Absent, the fit is ordinary least squares. Present, it is iteratively reweighted
least squares with Huber weights, where ``robust_tuning`` is the width of the
unweighted band in units of the residual scale, and ``robust_passes`` is how
many reweightings are made.

A fixed number of passes rather than a convergence tolerance. A tolerance would
be a number nobody chose that decides how many passes ran, and the count would
then depend on the platform's floating point in a project whose determinism
controls exist to stop exactly that. A declared count runs the same everywhere
and appears in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from gutachten.surface import Surface
from gutachten.transforms.base import Parameters, SurfaceProperty, record_for
from gutachten.transforms.registry import REGISTRY

__all__ = ["FormModel", "LevelParameters", "RemoveForm"]

#: What the median absolute deviation has to be multiplied by to estimate the
#: standard deviation of a normal distribution. It is fixed by the distribution
#: and not by anyone's preference, which is why it is not a parameter.
_MAD_TO_SIGMA = 1.4826  # structural: the normal-consistency factor of the median absolute deviation


class FormModel(Enum):
    """The form removed from the surface."""

    #: The first order polynomial: the tilt and nothing else.
    PLANE = "plane"
    #: A polynomial of the declared order, which must be at least the second.
    POLYNOMIAL = "polynomial"
    #: A sphere fitted to the measured points, evaluated as a height field.
    SPHERE = "sphere"


@dataclass(frozen=True)
class LevelParameters:
    """Which form is removed, and how the fit that finds it is made.

    ``model`` is the name of a :class:`FormModel`. A name rather than the enum
    itself because a parameter record is written into the manifest as plain
    data, and a field that cannot be serialised is a field that cannot be
    recorded.

    ``order`` belongs to the polynomial and is ``None`` for the other two.
    ``robust_tuning`` and ``robust_passes`` are both ``None`` for an ordinary
    least squares fit and both set for a robust one. ``None`` is written
    explicitly at every call site because the record carries no defaults, so a
    run that made an ordinary fit says so rather than being silent about it.
    """

    model: str
    order: int | None
    robust_tuning: float | None
    robust_passes: int | None


class RemoveForm:
    """Fit a form to the measured samples and subtract it."""

    identifier = "level"
    version = "1"
    parameters_type = LevelParameters
    produces = frozenset({SurfaceProperty.LEVELLED})
    requires = frozenset[SurfaceProperty]()
    # A bandpass has already removed the long wavelength content a form model
    # is fitted to, so a fit made afterwards is a fit to what the filter left
    # behind at the edges of the band rather than to the shape of the primer.
    refuses = frozenset({SurfaceProperty.FILTERED})

    def apply(self, surface: Surface, parameters: Parameters) -> Surface:
        # The pipeline checks the record type before the chain starts and
        # `record_for` checks it again on the way out. This one narrows the type
        # for the checker and refuses before a single field has been read, which
        # the one on the way out cannot do.
        if not isinstance(parameters, LevelParameters):
            raise TypeError(
                f"{self.identifier!r} was handed {type(parameters).__name__} rather than "
                "LevelParameters, so nothing here has read a field off it"
            )
        model = _model(parameters)
        degree = _degree(model, parameters.order)
        tuning, passes = _robustness(parameters)

        observed = ~surface.missing
        if not np.any(observed):
            raise ValueError(
                "every sample of this surface is missing, so there is nothing to fit a "
                "form to. A step that levelled an empty surface would record itself as "
                "having run and hand on the array it was given."
            )

        y_um, x_um = _coordinates(surface)
        extent = float(max(np.max(np.abs(y_um)), np.max(np.abs(x_um))))
        if extent <= 0.0:
            raise ValueError(
                f"a {surface.shape[0]}x{surface.shape[1]} surface has no extent in either "
                "direction, so no form is determined by it"
            )

        if model is FormModel.SPHERE:
            form = _sphere_form(y_um, x_um, surface.heights, observed, extent, tuning, passes)
        else:
            form = _polynomial_form(
                y_um, x_um, surface.heights, observed, extent, degree, tuning, passes
            )

        return surface.with_transform(record_for(self, parameters), surface.heights - form)


def _model(parameters: LevelParameters) -> FormModel:
    """The model named, refusing a name that is not one.

    A misspelled model would otherwise fall through to whichever branch was
    written as the default, and the manifest would record a word that did not
    decide anything.
    """
    try:
        return FormModel(parameters.model)
    except ValueError:
        known = ", ".join(sorted(item.value for item in FormModel))
        raise ValueError(
            f"{parameters.model!r} is not a form model this step knows; it takes one of {known}"
        ) from None


def _degree(model: FormModel, order: int | None) -> int:
    """The polynomial degree ``model`` and ``order`` mean between them.

    Refuses an order recorded against a model that has none, and a first order
    polynomial, which is the plane under a second name.
    """
    if model is not FormModel.POLYNOMIAL:
        if order is not None:
            raise ValueError(
                f"the {model.value!r} model has no order to choose and was given "
                f"order={order!r}. Pass order=None, so the manifest does not record a "
                "number that decided nothing and a sweep does not move one."
            )
        # The plane is the first order polynomial and is computed as one. The
        # sphere ignores this and takes its own route.
        return 1

    if order is None:
        raise ValueError(
            "the 'polynomial' model names no order, and the order is what decides how "
            "much of the surface it removes. Pass one, or name the 'plane' model."
        )
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError(f"the polynomial order must be a whole number, got {order!r}")
    if order < 1:
        raise ValueError(
            f"the polynomial order must be at least the first, got {order!r}. An order of "
            "zero removes the mean, which is a different step and is not this one."
        )
    if order == 1:
        raise ValueError(
            "a first order polynomial is a plane. Name the 'plane' model instead, so one "
            "form is not reachable under two names and a sweep enumerating the parameter "
            "space does not visit it twice."
        )
    return order


def _robustness(parameters: LevelParameters) -> tuple[float | None, int]:
    """The two robustness settings, refusing one without the other.

    Returns the tuning constant and the number of passes. The passes are
    meaningless without a tuning constant, so a fit that is not robust reports
    zero of them and never enters the loop.
    """
    tuning = parameters.robust_tuning
    passes = parameters.robust_passes

    if (tuning is None) != (passes is None):
        raise ValueError(
            f"robust_tuning={tuning!r} and robust_passes={passes!r} disagree about whether "
            "this fit is robust. Both are None for an ordinary least squares fit and both "
            "are set for a robust one, because a tuning constant nothing reweights and a "
            "count of reweightings with no constant to reweight by are each half a "
            "setting."
        )
    if tuning is None or passes is None:
        return None, 0

    if isinstance(tuning, bool) or not isinstance(tuning, (int, float)):
        raise TypeError(f"robust_tuning must be a number of residual scales, got {tuning!r}")
    if not np.isfinite(tuning) or tuning <= 0.0:
        raise ValueError(
            f"robust_tuning must be a positive finite number of residual scales, got "
            f"{tuning!r}. It is the half width of the band inside which a residual is "
            "left at full weight, and a band of no width downweights every sample "
            "including the ones the form is made of."
        )
    if isinstance(passes, bool) or not isinstance(passes, int):
        raise TypeError(f"robust_passes must be a whole number of reweightings, got {passes!r}")
    if passes < 1:
        raise ValueError(
            f"robust_passes must be at least one, got {passes!r}. A robust fit that "
            "reweights nothing is an ordinary least squares fit recorded under a name "
            "that says it is not."
        )
    return float(tuning), passes


def _coordinates(surface: Surface) -> tuple[np.ndarray, np.ndarray]:
    """Physical coordinates in the surface's own unit, centred on the field.

    Centred rather than counted from a corner, so the constant term of the fit
    is the height at the middle of the field and the higher terms are not
    carrying an offset that the array indices happened to impose.
    """
    rows, columns = surface.shape
    down = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2) * surface.spacing_y
    across = (np.arange(columns, dtype=np.float64) - (columns - 1) / 2) * surface.spacing_x
    y_um, x_um = np.meshgrid(down, across, indexing="ij")
    return np.asarray(y_um), np.asarray(x_um)


def _polynomial_columns(y: np.ndarray, x: np.ndarray, degree: int) -> list[np.ndarray]:
    """One column per term of a two dimensional polynomial up to ``degree``.

    Grouped by total degree and, within a group, by the power on ``y``, so the
    column order is decided here rather than by the order a set iterated in.
    """
    columns: list[np.ndarray] = []
    for total in range(degree + 1):
        for power_y in range(total + 1):
            columns.append(np.asarray((y**power_y) * (x ** (total - power_y))))
    return columns


def _polynomial_form(
    y_um: np.ndarray,
    x_um: np.ndarray,
    heights: np.ndarray,
    observed: np.ndarray,
    extent: float,
    degree: int,
    tuning: float | None,
    passes: int,
) -> np.ndarray:
    """The fitted polynomial, evaluated at every sample including the missing ones.

    The coordinates are divided by the extent of the field before the terms are
    raised to their powers. A twelfth power of a coordinate in micrometres would
    otherwise range over dozens of orders of magnitude across one design matrix,
    and the solve would be answering a question about floating point rather than
    about the surface.
    """
    columns = _polynomial_columns(y_um / extent, x_um / extent, degree)
    design = np.column_stack([column[observed] for column in columns])
    coefficients = _fit(
        design, heights[observed], tuning, passes, what=f"a polynomial of order {degree}"
    )

    form = np.zeros_like(heights)
    for coefficient, column in zip(coefficients, columns, strict=True):
        form = form + coefficient * column
    return form


def _sphere_form(
    y_um: np.ndarray,
    x_um: np.ndarray,
    heights: np.ndarray,
    observed: np.ndarray,
    extent: float,
    tuning: float | None,
    passes: int,
) -> np.ndarray:
    """The fitted sphere, evaluated as a height field over the whole grid.

    Fitted in the algebraic form. Writing the sphere as

        y^2 + x^2 + z^2 = 2 a y + 2 b x + 2 c z + (R^2 - a^2 - b^2 - c^2)

    makes it linear in the centre and in one combination of the centre and the
    radius, so it is the same weighted solve the polynomial uses and inherits
    the same robustness.

    Every coordinate and the heights with them are divided by the extent of the
    field, which leaves a sphere a sphere and keeps the squared terms of one
    magnitude, and the result is multiplied back at the end.
    """
    y_scaled = y_um / extent
    x_scaled = x_um / extent
    z_scaled = heights / extent

    design = np.column_stack(
        [
            y_scaled[observed],
            x_scaled[observed],
            z_scaled[observed],
            np.ones_like(z_scaled[observed]),
        ]
    )
    values = y_scaled[observed] ** 2 + x_scaled[observed] ** 2 + z_scaled[observed] ** 2
    coefficients = _fit(design, values, tuning, passes, what="a sphere")

    twice_centre_y, twice_centre_x, twice_centre_z, offset = (
        float(value) for value in coefficients
    )
    centre_y = twice_centre_y / 2
    centre_x = twice_centre_x / 2
    centre_z = twice_centre_z / 2
    radius_squared = offset + centre_y**2 + centre_x**2 + centre_z**2

    # One check rather than two. A squared radius that is not positive describes
    # no real sphere at all, and it reaches every sample of the field from below
    # exactly as a real sphere too small for the field does, so the same refusal
    # covers both and there is no second branch that no surface has been found
    # to reach.
    above_axis = radius_squared - (y_scaled - centre_y) ** 2 - (x_scaled - centre_x) ** 2
    if np.any(above_axis < 0.0):
        raise ValueError(
            f"the sphere fitted to these samples has a squared radius of "
            f"{radius_squared * extent**2} and does not reach every sample of the field, "
            f"falling short by up to {-float(np.min(above_axis)) * extent**2} in the same "
            "units, so the form is undefined where it stops. A squared radius that is not "
            "positive is the same refusal for a stronger reason: those samples describe "
            "no real sphere. Evaluating either anyway would need a clamp, and a clamped "
            "sphere puts a flat rim on the levelled surface that no later step can tell "
            "from measured data."
        )

    from_axis = np.sqrt(above_axis)
    # Two branches meet the same sphere. Which one the surface sits on is
    # decided by the data rather than by an assumption about whether a primer is
    # convex, because both occur: a bowl and a dome are the same sphere seen
    # from opposite sides.
    lower = centre_z - from_axis
    upper = centre_z + from_axis
    lower_miss = float(np.sum((z_scaled[observed] - lower[observed]) ** 2))
    upper_miss = float(np.sum((z_scaled[observed] - upper[observed]) ** 2))
    chosen = lower if lower_miss <= upper_miss else upper
    return np.asarray(chosen * extent)


def _fit(
    design: np.ndarray,
    values: np.ndarray,
    tuning: float | None,
    passes: int,
    *,
    what: str,
) -> np.ndarray:
    """Least squares coefficients, reweighted ``passes`` times if asked.

    The weights are Huber's: a residual inside ``tuning`` scales keeps its full
    weight and one outside is weighted down in proportion to how far out it is.
    The scale is re-estimated from the residuals on every pass, because the
    first pass's residuals are the ordinary fit's and are inflated by whatever
    the reweighting is there to remove.
    """
    coefficients = _least_squares(design, values, what=what)
    if tuning is None:
        return coefficients

    for _ in range(passes):
        residual = values - design @ coefficients
        deviation = np.abs(residual - np.median(residual))
        scale = _MAD_TO_SIGMA * float(np.median(deviation))
        if scale == 0.0:
            # More than half the residuals are at the median, which is what an
            # exact fit looks like. There is nothing to downweight, and a limit
            # of zero would put every sample at zero weight and leave the solve
            # with no data at all.
            return coefficients
        limit = tuning * scale
        magnitude = np.abs(residual)
        weight = np.where(magnitude <= limit, 1.0, limit / np.maximum(magnitude, limit))
        root = np.sqrt(weight)
        coefficients = _least_squares(design * root[:, None], values * root, what=what)

    return coefficients


def _least_squares(design: np.ndarray, values: np.ndarray, *, what: str) -> np.ndarray:
    """Solve, refusing a system the observed samples do not determine.

    A rank deficient solve returns the minimum norm answer rather than failing,
    and that answer is a form nobody asked for evaluated over the whole field.
    The case it arrives from is real: an order raised until the surface has
    fewer observed samples than the fit has coefficients, or a mask that leaves
    the observed samples along one line.
    """
    solution, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    wanted = design.shape[1]
    if rank < wanted:
        raise ValueError(
            f"the observed samples do not determine {what}: the fit has {wanted} "
            f"coefficients and the samples span {rank} of them. Either the surface has "
            "too few measured samples for this model, or the ones it has lie on a line."
        )
    return np.asarray(solution)


REGISTRY.register(RemoveForm())
