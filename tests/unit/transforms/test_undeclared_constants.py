"""A number a step did not declare is found by reading it, not by reviewing it.

The rule is that a parameter which is not in the record does not exist. This is
where it is refused. Two fixture modules stand beside each other: one that
declares every number it uses, and one with a threshold typed where it was
needed, which is how the rule actually gets broken.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from gutachten.transforms.audit import undeclared_constants
from gutachten.transforms.registry import REGISTRY, Registry
from tests.unit.transforms.declared_example import Scale
from tests.unit.transforms.undeclared_example import Clip

# The same step twice, differing in one comment. `{marker}` is where the
# exemption goes, and the pair either side of it is what shows the marker is
# doing the work rather than the value happening to be allowed.
A_STEP_WITH_ONE_LITERAL = '''\
"""A step written into the test's own directory, so the audit has a file to read."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    factor: float


class Step:
    identifier = "example-written"
    version = "1"
    parameters_type = Parameters

    def apply(self, surface: object, parameters: Parameters) -> object:
        percent = 100.0{marker}
        _ = percent
        return surface
'''


@pytest.fixture
def write_step(tmp_path: Path) -> Iterator[Callable[[str, str], Any]]:
    """Import ``A_STEP_WITH_ONE_LITERAL`` from a real file and return the step.

    The audit reads a transform's source off disk, and it finds that file
    through the module the class was defined in, so the module has to be in
    ``sys.modules`` for the duration. The ones this writes are removed again
    afterwards rather than left where a later test would import them.
    """
    written: list[str] = []

    def factory(name: str, marker: str) -> Any:
        path = tmp_path / f"{name}.py"
        path.write_text(A_STEP_WITH_ONE_LITERAL.format(marker=marker), encoding="utf-8")
        specification = importlib.util.spec_from_file_location(name, path)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        written.append(name)
        specification.loader.exec_module(module)
        return module.Step()

    yield factory

    for name in written:
        sys.modules.pop(name, None)


def test_a_step_that_declares_its_numbers_passes() -> None:
    registry = Registry()
    registry.register(Scale())

    assert undeclared_constants(registry) == []


def test_a_number_typed_where_it_was_needed_is_found_and_named() -> None:
    registry = Registry()
    registry.register(Clip())

    found = undeclared_constants(registry)

    assert [(item.identifier, item.value) for item in found] == [
        ("example-clip", 0.35),
        ("example-clip", 0.35),
        ("example-clip", 0.05),
    ]
    assert "undeclared_example.py" in str(found[0])
    assert "0.35" in str(found[0])
    assert "structural" in str(found[0])


def test_the_report_is_in_source_order_and_not_in_the_order_the_tree_was_walked() -> None:
    # The floor is the shallower literal and sits on the later line, so a walk
    # order that reached it first is what the sort has to undo. Without the
    # sort the report opens on a line the reader has already passed.
    found = undeclared_constants([Clip()])

    assert [item.line for item in found] == sorted(item.line for item in found)
    assert found[-1].value == 0.05


def test_without_the_marker_the_same_literal_is_reported(
    write_step: Callable[[str, str], Any],
) -> None:
    exempt = write_step("marked_step", "  # structural: a percentage is not a knob")
    plain = write_step("unmarked_step", "")

    assert undeclared_constants([exempt]) == []

    found = undeclared_constants([plain])
    assert [item.value for item in found] == [100.0]


def test_a_marker_carrying_no_reason_still_exempts_and_that_is_the_bound(
    write_step: Callable[[str, str], Any],
) -> None:
    # Stated rather than asserted away: the check reads the marker and cannot
    # judge whether the sentence after it is a reason. What the marker buys is
    # that the claim is in the diff for a reader to disagree with.
    bare = write_step("bare_marker_step", "  # structural:")

    assert undeclared_constants([bare]) == []


def test_a_transform_with_no_source_file_cannot_be_audited_and_says_so() -> None:
    # A step assembled at run time has no source for this check to read, so it
    # is refused rather than passing silently, which is what an audit that
    # skipped what it could not open would do.
    source = (
        "class Built:\n"
        "    identifier = 'example-built'\n"
        "    version = '1'\n"
        "    parameters_type = None\n"
        "    def apply(self, surface, parameters):\n"
        "        return surface\n"
    )
    namespace: dict[str, object] = {}
    exec(compile(source, "<assembled at run time>", "exec"), namespace)
    built = namespace["Built"]()  # type: ignore[operator]

    with pytest.raises(TypeError, match="no source file"):
        undeclared_constants([built])


def test_the_shipped_steps_declare_every_number_they_use() -> None:
    # The registry is empty until the transform issues land, and an empty
    # registry proves nothing on its own. What proves the check is the fixture
    # above. This test is what starts failing the moment a real step is added
    # with a constant in it.
    assert undeclared_constants(REGISTRY) == []
