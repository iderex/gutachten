"""The interface, the parameter record rules, and what the registry refuses.

Every refusal here was disabled in turn and the suite watched go red, which is
what separates these from decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.transforms.base import Parameters, Transform, check_parameters, record_for
from gutachten.transforms.registry import REGISTRY, Registry
from tests.unit.transforms.declared_example import Scale, ScaleParameters
from tests.unit.transforms.undeclared_example import Clip


def a_surface() -> Surface:
    return Surface(
        heights=np.ones((3, 4), dtype=np.float64),
        spacing_y=4.0,
        spacing_x=4.0,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="test",
    )


def test_a_transform_satisfies_the_interface_and_a_parameter_record_the_protocol() -> None:
    assert isinstance(Scale(), Transform)
    assert isinstance(ScaleParameters(factor=2.0), Parameters)


def test_applying_a_transform_records_the_parameters_that_ran() -> None:
    surface = a_surface()

    result = Scale().apply(surface, ScaleParameters(factor=3.0))

    assert len(result.provenance) == 1
    entry = result.provenance[0]
    assert entry.name == "example-scale"
    assert entry.version == "1"
    assert dict(entry.parameters) == {"factor": 3.0}
    assert result.shape == surface.shape


def test_a_record_is_refused_for_the_wrong_parameter_type() -> None:
    from tests.unit.transforms.undeclared_example import ClipParameters

    with pytest.raises(TypeError, match="takes ScaleParameters"):
        record_for(Scale(), ClipParameters(factor=1.0))


def test_a_parameter_record_that_is_not_a_dataclass_is_refused() -> None:
    class NotARecord:
        pass

    with pytest.raises(TypeError, match="not a dataclass"):
        check_parameters(NotARecord)


def test_a_parameter_record_that_can_be_changed_after_the_run_is_refused() -> None:
    @dataclass
    class Mutable:
        threshold: float

    with pytest.raises(TypeError, match="not frozen"):
        check_parameters(Mutable)


def test_a_parameter_record_with_a_default_is_refused() -> None:
    # The near miss: a field added with a sensible default, which then travels
    # into every run that does not mention it and is invisible to the sweep.
    @dataclass(frozen=True)
    class WithDefault:
        threshold: float = 0.35

    with pytest.raises(TypeError, match="gives a default to \\['threshold'\\]"):
        check_parameters(WithDefault)


def test_a_parameter_record_with_a_default_factory_is_refused() -> None:
    @dataclass(frozen=True)
    class WithFactory:
        cutoffs: tuple[float, ...] = field(default_factory=tuple)

    with pytest.raises(TypeError, match="gives a default to \\['cutoffs'\\]"):
        check_parameters(WithFactory)


def test_a_parameter_record_with_no_parameters_at_all_is_refused() -> None:
    @dataclass(frozen=True)
    class Empty:
        pass

    with pytest.raises(TypeError, match="declares no parameters"):
        check_parameters(Empty)


def test_registering_a_transform_makes_it_reachable_by_identifier() -> None:
    registry = Registry()

    returned = registry.register(Scale())

    assert isinstance(returned, Scale)
    assert "example-scale" in registry
    assert registry["example-scale"].version == "1"
    assert len(registry) == 1


def test_the_registry_iterates_in_identifier_order_rather_than_registration_order() -> None:
    # Registered late-first on purpose. Registering them in identifier order
    # would leave the two orders identical, and the test would then pass against
    # a registry that had never sorted anything.
    registry = Registry()
    registry.register(Scale())
    registry.register(Clip())

    assert registry.identifiers() == ("example-clip", "example-scale")
    assert [transform.identifier for transform in registry] == ["example-clip", "example-scale"]


def test_an_unknown_identifier_reports_what_is_registered() -> None:
    registry = Registry()
    registry.register(Scale())

    with pytest.raises(KeyError, match="registered: example-scale"):
        registry["level"]


def test_a_transform_with_no_identifier_is_refused() -> None:
    class Nameless(Scale):
        identifier = ""

    with pytest.raises(ValueError, match="declares no identifier"):
        Registry().register(Nameless())


def test_a_transform_with_no_version_is_refused() -> None:
    class Unversioned(Scale):
        version = ""

    with pytest.raises(ValueError, match="declares no version"):
        Registry().register(Unversioned())


def test_two_transforms_under_one_identifier_are_refused() -> None:
    registry = Registry()
    registry.register(Scale())

    class Other(Clip):
        identifier = "example-scale"

    with pytest.raises(ValueError, match="already registered"):
        registry.register(Other())


def test_registration_refuses_a_bad_parameter_record_rather_than_the_first_run() -> None:
    @dataclass(frozen=True)
    class Defaulted:
        threshold: float = 0.35

    class Sloppy(Scale):
        identifier = "example-sloppy"
        parameters_type = Defaulted

    with pytest.raises(TypeError, match="gives a default"):
        Registry().register(Sloppy())


def test_the_shipped_registry_holds_no_step_yet_and_this_file_added_none_to_it() -> None:
    # The fixtures above go into their own registries. A test that registered
    # into REGISTRY would leak into every test that runs after it, and into the
    # audit over the shipped steps.
    assert REGISTRY.identifiers() == ()
