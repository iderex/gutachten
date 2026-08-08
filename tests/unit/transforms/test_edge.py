"""The edge trim, against a synthetic surface whose invalid border is known.

The generator puts the border there on purpose, so this is a comparison against
a construction rather than against a guess about what a real scan looks like.

Every refusal here was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.edge import EdgeParameters, TrimEdge
from gutachten.transforms.pipeline import OrderingError, Step, check_chain
from gutachten.transforms.registry import REGISTRY, Registry
from tests.support.tolerance import assert_close
from tests.unit.transforms.declared_example import Scale, ScaleParameters
from tests.unit.transforms.undeclared_example import Clip, ClipParameters

ROWS = 40
COLUMNS = 40
SPACING_UM = 4.0
#: Four rows and four columns at each edge, which is what the generator makes
#: of a tenth of a forty sample field.
DROPOUT_FRACTION = 0.1
KNOWN_BORDER = 4


def a_surface(dropout: float = DROPOUT_FRACTION) -> Surface:
    generated = generate(
        SurfaceParameters(
            rows=ROWS,
            columns=COLUMNS,
            pixel_spacing_um=SPACING_UM,
            edge_dropout_fraction=dropout,
            seed=20260808,
        )
    )
    return Surface(
        heights=generated.heights_um,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def frame_mask(rows: int, columns: int, band: int) -> np.ndarray:
    """The band of width ``band`` around the edge of a ``rows`` by ``columns`` field."""
    mask = np.zeros((rows, columns), dtype=bool)
    mask[:band, :] = True
    mask[rows - band :, :] = True
    mask[:, :band] = True
    mask[:, columns - band :] = True
    return mask


def test_the_generated_border_is_where_this_file_says_it_is() -> None:
    # Asserted rather than assumed. Every expectation below is stated in terms
    # of KNOWN_BORDER, so a generator that put its dropout somewhere else would
    # make those tests pass against the wrong region.
    surface = a_surface()

    assert np.array_equal(surface.missing, frame_mask(ROWS, COLUMNS, KNOWN_BORDER))


def test_trimming_the_matching_width_marks_exactly_the_known_border() -> None:
    surface = a_surface()
    width = KNOWN_BORDER * SPACING_UM

    result = TrimEdge().apply(surface, EdgeParameters(width=width, criterion="frame"))

    assert np.array_equal(result.missing, frame_mask(ROWS, COLUMNS, KNOWN_BORDER))


def test_a_wider_and_a_narrower_width_remove_provably_more_and_less() -> None:
    # The parameter has to move the result, or the sweep it exists for is
    # measuring nothing.
    surface = a_surface()
    matching = KNOWN_BORDER * SPACING_UM
    step = TrimEdge()

    narrower = step.apply(surface, EdgeParameters(width=matching - SPACING_UM, criterion="frame"))
    exact = step.apply(surface, EdgeParameters(width=matching, criterion="frame"))
    wider = step.apply(surface, EdgeParameters(width=matching + SPACING_UM, criterion="frame"))

    # Narrower than the border removes nothing the surface had not already lost,
    # because the generator's dropout is still missing underneath it.
    assert int(narrower.missing.sum()) == int(exact.missing.sum())
    assert np.array_equal(wider.missing, frame_mask(ROWS, COLUMNS, KNOWN_BORDER + 1))
    assert int(wider.missing.sum()) > int(exact.missing.sum())


def test_growing_reaches_an_invalid_patch_a_frame_cut_never_touches() -> None:
    # The two criteria are not two spellings of one rule. A hole in the middle
    # is what the frame cut cannot see.
    surface = a_surface(dropout=0.0)
    heights = np.array(surface.heights, copy=True)
    heights[20, 20] = np.nan
    holed = Surface(
        heights=heights,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )
    step = TrimEdge()

    framed = step.apply(holed, EdgeParameters(width=SPACING_UM, criterion="frame"))
    grown = step.apply(holed, EdgeParameters(width=SPACING_UM, criterion="grow"))

    assert not framed.missing[19, 19]
    assert grown.missing[19, 19]
    assert grown.missing[21, 21]
    assert not grown.missing[18, 18]
    # Growing by one sample around one hole reaches its eight neighbours and
    # nothing on the boundary of the field.
    assert int(grown.missing.sum()) == 9


def test_the_step_records_the_width_and_the_criterion_it_ran_with() -> None:
    surface = a_surface()

    result = TrimEdge().apply(surface, EdgeParameters(width=16.0, criterion="frame"))

    entry = result.provenance[-1]
    assert entry.name == "trim-edge"
    assert entry.version == "1"
    assert dict(entry.parameters) == {"criterion": "frame", "width": 16.0}


def test_the_dimensions_do_not_change_and_the_surviving_heights_are_untouched() -> None:
    # Deleting rows would move every coordinate downstream, so a registration
    # found on a trimmed surface would not be one on the surface an operator is
    # looking at.
    surface = a_surface()

    result = TrimEdge().apply(surface, EdgeParameters(width=20.0, criterion="frame"))

    assert result.shape == surface.shape
    kept = ~result.missing
    assert_close(
        result.heights[kept],
        surface.heights[kept],
        what="the heights this step kept",
        # Exactly zero. The step marks samples missing and computes nothing,
        # so a surviving height that moved at all is a defect and not drift.
        atol=0.0,
    )


def test_a_criterion_this_step_does_not_know_is_refused_naming_what_it_takes() -> None:
    # A misspelling would otherwise fall through to whichever branch was written
    # last, and the manifest would record a word that decided nothing.
    surface = a_surface()

    with pytest.raises(ValueError, match="not a criterion this step knows; it takes one of"):
        TrimEdge().apply(surface, EdgeParameters(width=8.0, criterion="Frame"))


def test_a_width_that_rounds_to_no_samples_is_refused() -> None:
    # Silently doing nothing is the failure. A sweep is entitled to assume that
    # a step it asked for happened.
    surface = a_surface()

    with pytest.raises(ValueError, match="rounds to no samples"):
        TrimEdge().apply(surface, EdgeParameters(width=SPACING_UM / 4, criterion="frame"))


def test_a_width_that_is_not_a_positive_length_is_refused() -> None:
    surface = a_surface()
    step = TrimEdge()

    for width in (0.0, -8.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="must be a positive finite length"):
            step.apply(surface, EdgeParameters(width=width, criterion="frame"))


def test_a_width_that_removes_the_whole_surface_is_refused() -> None:
    # Every number after it would be taken over an empty array, which most
    # numpy reductions answer with nan rather than with an error.
    surface = a_surface()

    with pytest.raises(ValueError, match="leaves nothing measured"):
        TrimEdge().apply(surface, EdgeParameters(width=ROWS * SPACING_UM, criterion="frame"))


def test_the_step_refuses_a_record_that_is_not_its_own_before_reading_a_field() -> None:
    # `record_for` refuses the same mistake on the way out, by which point the
    # step has done its work. The wording asserted here is the one that only
    # the check at the door produces, so this cannot pass on the later refusal.
    surface = a_surface()

    class NotOurs:
        width = 8.0
        criterion = "frame"

    with pytest.raises(TypeError, match="rather than EdgeParameters, so nothing here has read"):
        TrimEdge().apply(surface, NotOurs())  # type: ignore[arg-type]


def test_the_two_axes_are_measured_at_their_own_spacings() -> None:
    # The fixture above is square in both spacing and shape, so a step that read
    # one spacing for both axes would pass every other test in this file. A real
    # instrument's two axes are not obliged to agree.
    generated = generate(
        SurfaceParameters(rows=ROWS, columns=COLUMNS, pixel_spacing_um=SPACING_UM, seed=1)
    )
    anisotropic = Surface(
        heights=generated.heights_um,
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM / 2,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )

    result = TrimEdge().apply(anisotropic, EdgeParameters(width=SPACING_UM, criterion="frame"))

    # One sample down at 4 micrometres a row, two across at 2 micrometres a
    # column, for one physical distance.
    assert np.array_equal(result.missing[:, 0], np.ones(ROWS, dtype=bool))
    assert bool(result.missing[10, 1])
    assert not bool(result.missing[10, 2])
    assert bool(result.missing[0, 10])
    assert not bool(result.missing[1, 10])


def test_trimming_after_a_filtering_step_is_refused_by_the_pipeline() -> None:
    # The trim would remove a region whose values the filter has already spread
    # into the surface around it, taking away the symptom and leaving the cause.
    registry = Registry()
    registry.register(Scale())
    registry.register(Clip())
    registry.register(TrimEdge())
    chain = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=1.0)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=1.0)),
        Step(
            identifier="trim-edge",
            parameters=EdgeParameters(width=SPACING_UM, criterion="frame"),
        ),
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

    assert "trim-edge" in completed.stdout.strip().split(",")
    assert REGISTRY["trim-edge"].version == "1"
