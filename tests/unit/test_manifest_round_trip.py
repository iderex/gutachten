"""A run writes a manifest, and the manifest runs the run again.

That is the property the record is worth anything for. Everything else the
schema does is bookkeeping that only pays off if a manifest fed back in
reproduces what it describes.

Every refusal here was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import (
    EnvironmentRecord,
    FileRecord,
    ProfileRecord,
    VersionMismatch,
    from_dict,
    read,
    record_run,
    rerun,
    resolve,
    surface_digest,
)
from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.base import SurfaceProperty, record_for
from gutachten.transforms.pipeline import Step
from gutachten.transforms.registry import Registry
from tests.unit.transforms.declared_example import Scale, ScaleParameters
from tests.unit.transforms.undeclared_example import Clip, ClipParameters

A_PROFILE = ProfileRecord(name="published", version="1")
AN_ENVIRONMENT = EnvironmentRecord(
    software_version="0.0.0",
    dependencies=(("numpy", "2.1.0"), ("scipy", "1.14.0")),
)
REFERENCE = DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS)


@dataclass(frozen=True)
class WindowParameters:
    """Two parameters, declared out of alphabetical order deliberately.

    `dataclasses.fields` hands them back in declaration order, so a manifest
    that recorded them in that order rather than sorting would come out with
    `upper_um` first, and every comparison of two manifests would then depend on
    which order somebody happened to type the fields in.
    """

    upper_um: float
    lower_um: float


class Window:
    """A second step at a second version, so a recorded version can be wrong."""

    identifier = "example-window"
    version = "2"
    parameters_type = WindowParameters
    produces = frozenset({SurfaceProperty.FILTERED})
    requires = frozenset({SurfaceProperty.LEVELLED})
    refuses = frozenset[SurfaceProperty]()

    def apply(self, surface: Surface, parameters: WindowParameters) -> Surface:
        heights = np.clip(surface.heights, parameters.lower_um, parameters.upper_um)
        return surface.with_transform(record_for(self, parameters), np.asarray(heights))


def a_registry() -> Registry:
    registry = Registry()
    registry.register(Scale())
    registry.register(Clip())
    registry.register(Window())
    return registry


def a_surface() -> Surface:
    generated = generate(SurfaceParameters(rows=24, columns=32, seed=20260808))
    return Surface(
        heights=generated.heights_um,
        spacing_y=generated.parameters.pixel_spacing_um,
        spacing_x=generated.parameters.pixel_spacing_um,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


# The generated surface is a bowl running from about -31 to -12 micrometres, so
# the clipping step saturates at any factor above about 0.011 and floors to zero
# below about 0.004. These two sit between, where the output actually depends on
# the parameter. The first factors this file used were outside that window, and
# every height came out at the clip bound, so three of these tests passed
# against a chain that was ignoring its own parameter.
A_FACTOR = 0.008
ANOTHER_FACTOR = 0.009
# Two factors that both saturate, so the output is the same either side of them.
A_SATURATING_FACTOR = 2.0
ANOTHER_SATURATING_FACTOR = 2.5


def a_chain(factor: float = A_FACTOR) -> list[Step]:
    return [
        Step(identifier="example-scale", parameters=ScaleParameters(factor=factor)),
        Step(identifier="example-clip", parameters=ClipParameters(factor=1.0)),
        Step(
            identifier="example-window",
            parameters=WindowParameters(upper_um=1.0, lower_um=-1.0),
        ),
    ]


def a_run(factor: float = A_FACTOR) -> tuple[Surface, object]:
    return record_run(
        role="scan-a",
        surface=a_surface(),
        profile=A_PROFILE,
        chain=a_chain(factor),
        registry=a_registry(),
        seed=20260808,
        determinism=REFERENCE,
        environment=AN_ENVIRONMENT,
    )


def test_a_completed_run_writes_a_manifest_naming_what_it_ran(tmp_path: Path) -> None:
    surface = a_surface()
    result, manifest = a_run()
    path = tmp_path / "run.json"

    manifest.write(path)

    written = read(path)
    assert written == manifest
    assert [step.identifier for step in written.steps] == [
        "example-scale",
        "example-clip",
        "example-window",
    ]
    # The versions come off the registered steps, not off the chain, and the
    # third one is at a different version so a hard coded "1" would show.
    assert [step.version for step in written.steps] == ["1", "1", "2"]
    # Sorted, so two manifests differ where the runs differ and not where the
    # fields happened to be declared.
    assert written.steps[2].parameters == (("lower_um", -1.0), ("upper_um", 1.0))
    assert written.inputs == (FileRecord(role="scan-a", sha256=surface_digest(surface)),)
    assert written.outputs == (FileRecord(role="surface", sha256=surface_digest(result)),)


def test_feeding_the_manifest_back_in_produces_a_byte_identical_output() -> None:
    # The property the whole record exists for. The digest is over the saved
    # array rather than its raw bytes, so a result of the same size and a
    # different shape is not mistaken for a reproduction.
    result, manifest = a_run()

    again = rerun(manifest, a_registry(), a_surface())

    assert surface_digest(again) == surface_digest(result)
    assert surface_digest(again) == manifest.outputs[0].sha256
    assert np.array_equal(again.heights, result.heights, equal_nan=True)


def differing_lines(left: str, right: str) -> list[tuple[str, str]]:
    """The lines two manifests disagree on, refusing a pair of different lengths.

    Equal lengths first, because a document that gained or lost a line is not a
    document a reader can compare line by line, and a diff of it would report
    every line after the change.
    """
    one = left.splitlines()
    two = right.splitlines()
    assert len(one) == len(two)
    return [(a.strip(), b.strip()) for a, b in zip(one, two, strict=True) if a != b]


def test_two_runs_differing_in_one_parameter_that_changes_nothing_differ_in_one_line() -> None:
    # A reader comparing two manifests by eye is the most common way a
    # difference is actually found, so one changed parameter has to be one
    # changed line and not a reordered document. These two factors both saturate
    # the clipping step, so the output is the same and the parameter line is the
    # only thing that moves.
    _, first = a_run(factor=A_SATURATING_FACTOR)
    _, second = a_run(factor=ANOTHER_SATURATING_FACTOR)

    assert differing_lines(first.to_text(), second.to_text()) == [
        ('"factor": 2.0', '"factor": 2.5')
    ]


def test_a_parameter_that_moves_the_output_moves_the_recorded_hash_and_nothing_else() -> None:
    # The general case, and it is two lines rather than one. A parameter that
    # changes the result has to change the recorded output hash as well, or the
    # manifest would name a result it did not produce. Stated here rather than
    # left to be discovered from a failing comparison.
    _, first = a_run(factor=A_FACTOR)
    _, second = a_run(factor=ANOTHER_FACTOR)

    differing = differing_lines(first.to_text(), second.to_text())

    assert len(differing) == 2
    assert differing[0] == ('"factor": 0.008', '"factor": 0.009')
    assert differing[1][0].startswith('"sha256"')


def test_a_manifest_naming_a_version_the_tree_no_longer_holds_is_refused_by_name() -> None:
    # Silently substituting the current step is how a reproduction becomes a
    # different experiment with the same label, so both versions are named.
    _, manifest = a_run()
    registry = a_registry()

    class Newer(Scale):
        version = "2"

    moved_on = Registry()
    moved_on.register(Newer())
    moved_on.register(Clip())
    moved_on.register(Window())

    resolve(manifest, registry)

    with pytest.raises(VersionMismatch, match=r"version '1'.*holds version '2'"):
        resolve(manifest, moved_on)


def test_a_step_whose_parameters_moved_without_its_version_moving_is_refused() -> None:
    # The version is supposed to move when the behaviour does and nothing can
    # check that somebody remembered. What can be checked is that the names the
    # manifest recorded are the names the step in this tree takes, which catches
    # a renamed or an added parameter under an unchanged version.
    _, manifest = a_run()

    @dataclass(frozen=True)
    class RenamedParameters:
        scale: float

    class Renamed(Scale):
        parameters_type = RenamedParameters

    tree = Registry()
    tree.register(Renamed())
    tree.register(Clip())
    tree.register(Window())

    with pytest.raises(ValueError, match=r"records \['factor'\]"):
        resolve(manifest, tree)


def test_re_running_against_a_different_input_is_refused() -> None:
    # The manifest names its input by hash for exactly this: a scan that
    # changed upstream between two runs recorded under one name.
    _, manifest = a_run()
    other = generate(SurfaceParameters(rows=24, columns=32, seed=1))
    different = Surface(
        heights=other.heights_um,
        spacing_y=other.parameters.pixel_spacing_um,
        spacing_x=other.parameters.pixel_spacing_um,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )

    with pytest.raises(ValueError, match=r"hashes to .* and the manifest names"):
        rerun(manifest, a_registry(), different)


def test_a_re_run_whose_result_moved_under_an_unchanged_version_is_refused() -> None:
    # The case a version number cannot catch: the step changed, the version did
    # not, and the chain, the parameters and the input all still match.
    _, manifest = a_run()

    class Drifted(Scale):
        def apply(self, surface: Surface, parameters: ScaleParameters) -> Surface:
            return super().apply(surface, ScaleParameters(factor=parameters.factor * 1.5))

    drifted = Registry()
    drifted.register(Drifted())
    drifted.register(Clip())
    drifted.register(Window())

    with pytest.raises(ValueError, match="what moved is the code behind a step"):
        rerun(manifest, drifted, a_surface())


def test_a_manifest_missing_a_field_is_refused_naming_it() -> None:
    _, manifest = a_run()
    data = manifest.to_dict()
    del data["seed"]

    with pytest.raises(ValueError, match=r"missing \['seed'\]"):
        from_dict(data)


def test_a_manifest_at_a_schema_version_this_code_does_not_write_is_refused(
    tmp_path: Path,
) -> None:
    _, manifest = a_run()
    data = manifest.to_dict()
    data["schema_version"] = data["schema_version"] + 1
    path = tmp_path / "future.json"
    path.write_text(
        manifest.to_text().replace('"schema_version": 3', '"schema_version": 4'),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="is not the version this code writes"):
        read(path)


def test_a_manifest_written_twice_is_written_byte_identically(tmp_path: Path) -> None:
    _, manifest = a_run()
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"

    manifest.write(first)
    manifest.write(second)

    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()


def test_a_hand_edited_manifest_is_read_back_with_its_mappings_sorted() -> None:
    # A manifest is a text file somebody edits, so the reader cannot assume the
    # order the writer produced. The records refuse an unsorted mapping, so a
    # reader that passed one straight through would refuse a file it should
    # have accepted, which sends somebody to read the source over a key order.
    _, manifest = a_run()
    data = manifest.to_dict()
    data["steps"][2]["parameters"] = {"upper_um": 1.0, "lower_um": -1.0}
    data["environment"]["dependencies"] = {"scipy": "1.14.0", "numpy": "2.1.0"}

    rebuilt = from_dict(data)

    assert rebuilt.steps[2].parameters == (("lower_um", -1.0), ("upper_um", 1.0))
    assert rebuilt.environment.dependencies == (("numpy", "2.1.0"), ("scipy", "1.14.0"))


def test_two_surfaces_with_the_same_bytes_and_different_shapes_hash_differently() -> None:
    # A digest over the raw bytes would call a reshaped result a reproduction.
    # The saved form carries the shape and the dtype, so it does not.
    values = np.linspace(0.0, 1.0, 12, dtype=np.float64)

    def shaped(rows: int, columns: int) -> Surface:
        return Surface(
            heights=values.reshape(rows, columns),
            spacing_y=4.0,
            spacing_x=4.0,
            unit=LengthUnit.MICROMETRE,
            orientation=AxisOrientation.Y_DOWN,
            source="test",
        )

    wide = shaped(2, 6)
    tall = shaped(3, 4)

    assert wide.heights.tobytes() == tall.heights.tobytes()
    assert surface_digest(wide) != surface_digest(tall)
