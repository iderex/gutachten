"""A chain runs end to end, and a chain that cannot mean anything is refused.

The fixtures next door declare an ordering between them: the scaling step leaves
the surface levelled and refuses one that has been filtered, and the clipping
step requires a levelled surface and leaves it filtered. That is the shape the
real steps have, with masking before filtering rather than after it, and it is
enough to reach every branch of the check.

Every refusal here was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import numpy as np
import pytest

from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.pipeline import OrderingError, Step, check_chain, run_chain
from gutachten.transforms.registry import Registry
from tests.unit.transforms.declared_example import Scale, ScaleParameters
from tests.unit.transforms.undeclared_example import Clip, ClipParameters


def a_registry() -> Registry:
    registry = Registry()
    registry.register(Scale())
    registry.register(Clip())
    return registry


def a_surface() -> Surface:
    return Surface(
        heights=np.ones((3, 4), dtype=np.float64),
        spacing_y=4.0,
        spacing_x=4.0,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="test",
    )


def a_generated_surface() -> Surface:
    """A surface from the generator, which is what a chain actually meets.

    A hand built array of ones is enough to reach a refusal and is not enough to
    show a chain running: it has no form to level, no striae and no missing
    samples, and every one of those is a thing a real step has to survive.
    """
    generated = generate(SurfaceParameters(rows=32, columns=48, seed=20260808))
    return Surface(
        heights=generated.heights_um,
        spacing_y=generated.parameters.pixel_spacing_um,
        spacing_x=generated.parameters.pixel_spacing_um,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def test_a_chain_of_identifiers_and_records_runs_end_to_end_on_a_synthetic_surface() -> None:
    surface = a_generated_surface()
    chain = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=3.0)),
    ]

    result = run_chain(chain, a_registry(), surface)

    assert [entry.name for entry in result.provenance] == ["example-scale", "example-clip"]
    assert [dict(entry.parameters) for entry in result.provenance] == [
        {"factor": 2.0},
        {"factor": 3.0},
    ]
    assert result.shape == surface.shape


def test_a_step_that_needs_something_no_earlier_step_established_is_refused() -> None:
    chain = [Step(identifier="example-clip", parameters=ClipParameters(factor=1.0))]

    with pytest.raises(OrderingError, match="requires the surface to be levelled"):
        run_chain(chain, a_registry(), a_surface())


def test_an_ordering_violation_names_both_transforms() -> None:
    # The case the constraints exist for: a step that refuses a surface an
    # earlier step has already changed. Nothing crashes without this check. The
    # run exits zero and produces a surface that is wrong in a way that looks
    # like data, so the message has to name what made it that way.
    chain = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=1.0)),
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
    ]

    with pytest.raises(OrderingError) as refused:
        run_chain(chain, a_registry(), a_surface())

    message = str(refused.value)
    assert "example-scale" in message
    assert "example-clip" in message
    assert "filtered" in message


def test_the_whole_chain_is_checked_before_any_of_it_runs() -> None:
    # A chain wrong in its third step fails before the first has touched the
    # surface, rather than after two steps of work and a plausible looking
    # intermediate that somebody will save.
    surface = a_surface()
    chain = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=1.0)),
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
    ]

    with pytest.raises(OrderingError):
        run_chain(chain, a_registry(), surface)

    assert surface.provenance == ()


def test_a_chain_can_be_checked_without_running_it() -> None:
    registry = a_registry()
    good = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=3.0)),
    ]

    check_chain(good, registry)

    with pytest.raises(OrderingError):
        check_chain(
            [Step(identifier="example-clip", parameters=ClipParameters(factor=1.0))], registry
        )


def test_an_identifier_the_registry_does_not_hold_is_refused_by_the_registry() -> None:
    chain = [Step(identifier="level", parameters=ScaleParameters(factor=1.0))]

    with pytest.raises(KeyError, match="registered: example-clip, example-scale"):
        run_chain(chain, a_registry(), a_surface())


def test_a_step_handed_the_wrong_parameter_record_stops_the_chain_before_it_starts() -> None:
    # record_for refuses the same mistake from inside a step, which is one step
    # too late: it fires after everything before it has already run. The wrong
    # record here is on the second step, and the first must not have touched the
    # surface when it is refused.
    surface = a_surface()
    chain = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
        Step(identifier="example-clip", parameters=ScaleParameters(factor=1.0)),
    ]

    with pytest.raises(TypeError, match="step 1 runs 'example-clip', which takes"):
        run_chain(chain, a_registry(), surface)

    assert surface.provenance == ()


def test_a_refusal_names_the_step_that_first_made_the_surface_that_way() -> None:
    # Two steps filter, and the question a refusal answers is when the surface
    # became filtered, not which step last agreed that it was.
    class SecondClip(Clip):
        identifier = "example-clip-2"

    registry = a_registry()
    registry.register(SecondClip())
    chain = [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=1.0)),
        Step(identifier="example-clip-2", parameters=ClipParameters(factor=1.0)),
        Step(identifier="example-scale", parameters=ScaleParameters(factor=2.0)),
    ]

    with pytest.raises(OrderingError) as refused:
        run_chain(chain, registry, a_surface())

    assert "example-clip'" in str(refused.value)
    assert "example-clip-2" not in str(refused.value)


def test_a_chain_with_no_steps_is_refused() -> None:
    with pytest.raises(ValueError, match="chain with no steps"):
        run_chain([], a_registry(), a_surface())


def test_a_step_that_does_not_record_itself_is_refused() -> None:
    # The near miss is a step that returns its input unchanged, which is what a
    # step under construction does. Every number afterwards is attributed to a
    # chain that is missing it.
    class Silent(Scale):
        identifier = "example-silent"

        def apply(self, surface: Surface, parameters: ScaleParameters) -> Surface:
            return surface

    registry = Registry()
    registry.register(Silent())
    chain = [Step(identifier="example-silent", parameters=ScaleParameters(factor=1.0))]

    with pytest.raises(RuntimeError, match=r"new provenance entries rather than\s+one"):
        run_chain(chain, registry, a_surface())
