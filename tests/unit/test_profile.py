"""The shipped profiles, and every way a profile file can be wrong.

Both profiles in ``profiles/`` are loaded and run here rather than being read,
because a profile that parses and cannot run is the failure a reader of the file
cannot see. Every refusal below was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import EnvironmentRecord, record_run
from gutachten.profile import ORIGINS, Profile, ProfileError, load, load_directory
from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.base import SurfaceProperty
from gutachten.transforms.registry import REGISTRY, Registry

#: The directory a release will have to carry, which is #132. Resolved from this
#: file rather than from the working directory, so the suite finds it whichever
#: directory pytest was started in.
SHIPPED = Path(__file__).resolve().parents[2] / "profiles"

#: The field the shipped profiles are written against: the generator's own size
#: and sampling interval. `every-step` places the drag mark by a length from the
#: centre, which is only the generator's groove on a field of this size.
ROWS = 256
COLUMNS = 256
SPACING_UM = 4.0


def a_surface() -> Surface:
    generated = generate(
        SurfaceParameters(
            rows=ROWS,
            columns=COLUMNS,
            pixel_spacing_um=SPACING_UM,
            seed=20260808,
        )
    )
    return Surface(
        heights=np.asarray(generated.heights_um),
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="synthetic",
    )


def shipped() -> tuple[Profile, ...]:
    return load_directory(SHIPPED, REGISTRY)


def written(tmp_path: Path, data: dict[str, Any], *, called: str | None = None) -> Path:
    path = tmp_path / f"{called or data['name']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def as_data(name: str) -> dict[str, Any]:
    """A shipped profile as plain data, to be broken one field at a time.

    The near misses below start from a file that does load, so each one is the
    single edit that made it stop loading rather than a fixture assembled to
    fail.
    """
    parsed: dict[str, Any] = json.loads((SHIPPED / f"{name}.json").read_text(encoding="utf-8"))
    return parsed


def a_source(origin: str = "not-sourced") -> dict[str, str]:
    return {"origin": origin, "where": "a test", "confidence": "none"}


def test_the_repository_ships_more_than_one_profile() -> None:
    # The clause asking for at least two. One profile is a default with a file
    # name, and the argument this directory exists for is that a configuration
    # is one of many rather than the right one.
    names = [profile.name for profile in shipped()]
    assert names == ["every-step", "published-chain"]


def test_every_shipped_profile_runs_end_to_end_and_reaches_the_manifest() -> None:
    # The clauses asking that the profiles run and that the version is in the
    # manifest. Run rather than parsed: a chain that resolves and cannot run is
    # the failure reading the file does not show.
    surface = a_surface()
    for profile in shipped():
        result, manifest = record_run(
            role="input",
            surface=surface,
            profile=profile.record(),
            chain=profile.chain(),
            registry=REGISTRY,
            seed=0,
            determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
            environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
        )
        assert [step.identifier for step in manifest.steps] == [
            step.identifier for step in profile.steps
        ]
        assert manifest.profile.name == profile.name
        assert manifest.profile.version == profile.version
        assert f'"version": "{profile.version}"' in manifest.to_text()
        assert np.any(np.isfinite(result.heights))


def test_the_two_profiles_are_two_configurations_and_not_one() -> None:
    # A second profile that produced the same surface would satisfy the count
    # and none of the reason for it.
    surface = a_surface()
    texts = set()
    for profile in shipped():
        _, manifest = record_run(
            role="input",
            surface=surface,
            profile=profile.record(),
            chain=profile.chain(),
            registry=REGISTRY,
            seed=0,
            determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
            environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
        )
        texts.add(manifest.to_text())
    assert len(texts) == 2


def test_the_reproduction_profile_says_where_every_value_came_from() -> None:
    # The clause asking the reproduction profile to say, at each parameter it
    # could not source, where the value came from. The gap is the result: the
    # count of parameters that trace to a stated value is asserted here, so a
    # profile that quietly promoted an assumption to a citation reds this.
    profile = load(SHIPPED / "published-chain.json", REGISTRY)
    origins = [
        (step.identifier, name, source.origin)
        for step in profile.steps
        for name, source in step.sources
    ]
    assert all(origin in ORIGINS for _, _, origin in origins)
    stated = [(step, name) for step, name, origin in origins if origin == "stated"]
    assert stated == [("bandpass", "short_cutoff"), ("bandpass", "long_cutoff")]
    assert len(origins) == 10


def test_a_profile_that_does_not_set_a_parameter_is_refused(tmp_path: Path) -> None:
    # The clause asking that a profile missing a parameter a registered
    # transform requires reds the build. The near miss is the one somebody
    # actually makes: a field added to a step, and the profiles not touched.
    data = as_data("published-chain")
    del data["steps"][2]["parameters"]["long_cutoff"]

    with pytest.raises(ProfileError) as refusal:
        load(written(tmp_path, data), REGISTRY)

    message = str(refusal.value)
    assert "long_cutoff" in message
    assert "bandpass" in message
    assert "published-chain" in message


def test_a_parameter_the_step_does_not_take_is_refused(tmp_path: Path) -> None:
    # A misspelling reaches nothing and is refused rather than ignored, because
    # a setting somebody believes is in force is worse than one they know they
    # have to add. Written without a provenance entry so that the refusal this
    # asserts is this one and not the one for an undocumented value.
    data = as_data("published-chain")
    data["steps"][2]["parameters"]["cutoff"] = 16.0

    with pytest.raises(ProfileError, match="believes is in force"):
        load(written(tmp_path, data), REGISTRY)


def test_a_profile_field_that_is_empty_is_refused(tmp_path: Path) -> None:
    # A version somebody left blank names two runs the same thing.
    data = as_data("published-chain")
    data["version"] = "  "

    with pytest.raises(ProfileError, match="the profile version"):
        load(written(tmp_path, data), REGISTRY)


def test_a_value_with_no_recorded_provenance_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    del data["steps"][2]["sources"]["short_cutoff"]

    with pytest.raises(ProfileError, match="where the value came from"):
        load(written(tmp_path, data), REGISTRY)


def test_provenance_left_behind_by_the_parameter_it_described_is_refused(tmp_path: Path) -> None:
    # A profile that dropped a parameter and kept its citation would go on
    # reading as though something had been checked.
    data = as_data("published-chain")
    data["steps"][2]["parameters"]["gone"] = 1.0
    del data["steps"][2]["parameters"]["gone"]
    data["steps"][2]["sources"]["gone"] = a_source("stated")

    with pytest.raises(ProfileError, match="gone"):
        load(written(tmp_path, data), REGISTRY)


def test_an_origin_outside_the_vocabulary_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    data["steps"][2]["sources"]["short_cutoff"]["origin"] = "probably"

    with pytest.raises(ProfileError, match="probably"):
        load(written(tmp_path, data), REGISTRY)


def test_a_provenance_entry_that_says_nothing_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    data["steps"][2]["sources"]["short_cutoff"]["confidence"] = "   "

    with pytest.raises(ProfileError, match="confidence"):
        load(written(tmp_path, data), REGISTRY)


def test_a_provenance_entry_missing_a_field_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    del data["steps"][2]["sources"]["short_cutoff"]["confidence"]

    with pytest.raises(ProfileError, match="confidence"):
        load(written(tmp_path, data), REGISTRY)


def test_a_setting_written_as_a_number_is_refused(tmp_path: Path) -> None:
    # True is an integer in Python, so an isinstance check that does not say
    # otherwise lets a flag arrive as 1 and a run that masked nothing exits
    # zero.
    data = as_data("every-step")
    data["steps"][0]["parameters"]["exclude_drag"] = 1

    with pytest.raises(ProfileError, match="true or false"):
        load(written(tmp_path, data), REGISTRY)


def test_a_number_written_as_a_setting_is_refused(tmp_path: Path) -> None:
    data = as_data("every-step")
    data["steps"][3]["parameters"]["order"] = True

    with pytest.raises(ProfileError, match="arrives as 0 or 1"):
        load(written(tmp_path, data), REGISTRY)


def test_a_whole_number_reaches_a_length_as_a_length(tmp_path: Path) -> None:
    # JSON has one number type and a length written without a point arrives as
    # an integer. Refusing it would make the file format harder than it needs to
    # be; carrying it into a float64 pipeline as an int is what is refused
    # everywhere else, so it is converted here and asserted to have been.
    data = as_data("every-step")
    data["steps"][1]["parameters"]["width"] = 40

    profile = load(written(tmp_path, data), REGISTRY)
    width = profile.steps[1].parameters.width  # type: ignore[attr-defined]
    assert isinstance(width, float)


def test_a_string_where_a_number_belongs_is_refused(tmp_path: Path) -> None:
    data = as_data("every-step")
    data["steps"][1]["parameters"]["width"] = "40"

    with pytest.raises(ProfileError, match="width"):
        load(written(tmp_path, data), REGISTRY)


def test_a_null_the_field_does_not_admit_is_refused(tmp_path: Path) -> None:
    # Absent and unset are different states and the record says which one a
    # field has. A null written into a field that has no null is neither.
    data = as_data("every-step")
    data["steps"][1]["parameters"]["criterion"] = None

    with pytest.raises(ProfileError, match="null"):
        load(written(tmp_path, data), REGISTRY)


def test_a_chain_that_cannot_run_in_its_order_is_refused_at_load(tmp_path: Path) -> None:
    # A profile is text somebody edits and a sweep permutes what it says, so an
    # impossible order arrives by editing rather than by misunderstanding. It is
    # refused when the file is read rather than when the fifth step is reached.
    data = as_data("published-chain")
    data["steps"] = [data["steps"][2], data["steps"][0], data["steps"][1]]

    with pytest.raises(Exception, match="refuses a surface that is filtered"):
        load(written(tmp_path, data), REGISTRY)


def test_a_step_this_tree_does_not_register_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    data["steps"][0]["transform"] = "crop"

    with pytest.raises(ProfileError, match="crop"):
        load(written(tmp_path, data), REGISTRY)


def test_the_recorded_name_and_the_file_name_have_to_agree(tmp_path: Path) -> None:
    data = as_data("published-chain")

    with pytest.raises(ProfileError, match="copied file"):
        load(written(tmp_path, data, called="published-chain-copy"), REGISTRY)


def test_a_profile_missing_a_field_of_its_own_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    del data["description"]

    with pytest.raises(ProfileError, match="description"):
        load(written(tmp_path, data, called="published-chain"), REGISTRY)


def test_a_key_nothing_reads_is_refused(tmp_path: Path) -> None:
    # A key nothing reads is a parameter somebody believes they set.
    data = as_data("published-chain")
    data["sampling"] = 4

    with pytest.raises(ProfileError, match="sampling"):
        load(written(tmp_path, data), REGISTRY)


def test_a_key_nothing_reads_inside_a_step_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    data["steps"][0]["notes"] = "why"

    with pytest.raises(ProfileError, match="notes"):
        load(written(tmp_path, data), REGISTRY)


def test_a_profile_with_no_step_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    data["steps"] = []

    with pytest.raises(ProfileError, match="names no step"):
        load(written(tmp_path, data), REGISTRY)


def test_a_step_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    data["steps"] = ["bandpass"]

    with pytest.raises(ProfileError, match="step 0"):
        load(written(tmp_path, data), REGISTRY)


def test_a_step_missing_one_of_its_three_parts_is_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    del data["steps"][0]["sources"]

    with pytest.raises(ProfileError, match="sources"):
        load(written(tmp_path, data), REGISTRY)


def test_parameters_that_are_not_an_object_are_refused(tmp_path: Path) -> None:
    data = as_data("published-chain")
    data["steps"][0]["parameters"] = ["width"]

    with pytest.raises(ProfileError, match="parameters of step 0"):
        load(written(tmp_path, data), REGISTRY)


def test_a_file_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "published-chain.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ProfileError, match="the profile"):
        load(path, REGISTRY)


def test_a_parameter_this_loader_cannot_check_is_refused_rather_than_guessed_at(
    tmp_path: Path,
) -> None:
    # A value this cannot check is a value that reaches a surface as whatever
    # the file happened to hold, so the loader refuses the declaration rather
    # than the value. Registered into a registry of its own, so this test adds
    # nothing to the one the rest of the project reads.
    from dataclasses import dataclass

    from gutachten.surface import Surface as _Surface

    @dataclass(frozen=True)
    class OddParameters:
        cutoffs: list[float]

    class OddStep:
        identifier = "odd"
        version = "1"
        parameters_type = OddParameters
        produces = frozenset({SurfaceProperty.FILTERED})
        requires = frozenset[SurfaceProperty]()
        refuses = frozenset[SurfaceProperty]()

        def apply(self, surface: _Surface, parameters: object) -> _Surface:
            raise AssertionError("this step exists to be refused before it runs")

    registry = Registry()
    registry.register(OddStep())
    data = {
        "name": "odd",
        "version": "1",
        "description": "a step whose parameter this loader cannot check",
        "steps": [
            {
                "transform": "odd",
                "parameters": {"cutoffs": [16.0, 500.0]},
                "sources": {"cutoffs": a_source()},
            }
        ],
    }

    with pytest.raises(ProfileError, match="cannot check"):
        load(written(tmp_path, data), registry)
