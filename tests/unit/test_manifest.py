"""The manifest schema, and every record it refuses to write.

A manifest is the authority for whether two runs were the same run. A field it
records loosely is a field that cannot settle that question, so the refusals
here are the schema rather than decoration on it.
"""

from __future__ import annotations

import json

import pytest

from gutachten.manifest import (
    SCHEMA_VERSION,
    EnvironmentRecord,
    FileRecord,
    ProfileRecord,
    RunManifest,
    StepRecord,
)

A_HASH = "0" * 64
ANOTHER_HASH = "a1" + "0" * 62


def a_manifest(**overrides: object) -> RunManifest:
    arguments: dict[str, object] = {
        "inputs": (FileRecord(role="scan-a", sha256=A_HASH),),
        "profile": ProfileRecord(name="published", version="1"),
        "steps": (
            StepRecord(identifier="level", version="2", parameters=(("model", "plane"),)),
            StepRecord(
                identifier="bandpass",
                version="1",
                parameters=(("lower_um", 25.0), ("upper_um", 250.0)),
            ),
        ),
        "seed": 20260808,
        "environment": EnvironmentRecord(
            software_version="0.0.0",
            dependencies=(("numpy", "2.1.0"), ("scipy", "1.14.0")),
        ),
        "outputs": (FileRecord(role="surface", sha256=ANOTHER_HASH),),
    }
    arguments.update(overrides)
    return RunManifest(**arguments)  # type: ignore[arg-type]


def test_a_manifest_names_the_input_profile_steps_seed_environment_and_outputs() -> None:
    written = a_manifest().to_dict()

    assert written["schema_version"] == SCHEMA_VERSION
    assert written["inputs"] == [{"role": "scan-a", "sha256": A_HASH}]
    assert written["profile"] == {"name": "published", "version": "1"}
    assert [step["identifier"] for step in written["steps"]] == ["level", "bandpass"]
    assert written["steps"][1]["parameters"] == {"lower_um": 25.0, "upper_um": 250.0}
    assert written["seed"] == 20260808
    assert written["environment"]["dependencies"] == {"numpy": "2.1.0", "scipy": "1.14.0"}
    assert written["outputs"] == [{"role": "surface", "sha256": ANOTHER_HASH}]


def test_the_steps_keep_the_order_they_ran_in() -> None:
    # The steps are a sequence and not a mapping, because levelling then
    # filtering is a different procedure from filtering then levelling and a
    # sorted container would lose which one happened.
    reversed_steps = tuple(reversed(a_manifest().steps))

    assert [step["identifier"] for step in a_manifest(steps=reversed_steps).to_dict()["steps"]] == [
        "bandpass",
        "level",
    ]


def test_a_manifest_written_twice_is_written_identically() -> None:
    first = json.dumps(a_manifest().to_dict(), sort_keys=False)
    second = json.dumps(a_manifest().to_dict(), sort_keys=False)

    assert first == second


def test_a_hash_that_is_not_a_sha256_is_refused() -> None:
    with pytest.raises(ValueError, match="not 64 lowercase hex"):
        FileRecord(role="scan-a", sha256="deadbeef")
    # Upper case hex is the near miss: the same 64 characters, produced by a
    # tool that formats differently, comparing unequal to the recorded one.
    with pytest.raises(ValueError, match="not 64 lowercase hex"):
        FileRecord(role="scan-a", sha256=ANOTHER_HASH.upper())


def test_a_file_record_with_no_role_is_refused() -> None:
    with pytest.raises(ValueError, match="role must be a non-empty string"):
        FileRecord(role="  ", sha256=A_HASH)


def test_a_profile_without_a_name_or_a_version_is_refused() -> None:
    with pytest.raises(ValueError, match="profile's name"):
        ProfileRecord(name="", version="1")
    with pytest.raises(ValueError, match="profile's version"):
        ProfileRecord(name="published", version="")


def test_a_step_without_an_identifier_or_a_version_is_refused() -> None:
    with pytest.raises(ValueError, match="step's identifier"):
        StepRecord(identifier="", version="1", parameters=(("a", 1),))
    with pytest.raises(ValueError, match="step's version"):
        StepRecord(identifier="level", version="", parameters=(("a", 1),))


def test_a_step_that_records_no_parameters_is_refused() -> None:
    with pytest.raises(ValueError, match="records no parameters"):
        StepRecord(identifier="level", version="1", parameters=())


def test_a_step_recording_its_parameters_unsorted_or_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="unsorted"):
        StepRecord(identifier="level", version="1", parameters=(("b", 1), ("a", 2)))
    with pytest.raises(ValueError, match="names a parameter twice"):
        StepRecord(identifier="level", version="1", parameters=(("a", 1), ("a", 2)))


def test_an_environment_without_a_software_version_is_refused() -> None:
    with pytest.raises(ValueError, match="software version"):
        EnvironmentRecord(software_version="", dependencies=())


def test_dependencies_recorded_unsorted_or_twice_are_refused() -> None:
    with pytest.raises(ValueError, match="unsorted"):
        EnvironmentRecord(software_version="0.0.0", dependencies=(("scipy", "1"), ("numpy", "2")))
    with pytest.raises(ValueError, match="recorded twice"):
        EnvironmentRecord(software_version="0.0.0", dependencies=(("numpy", "1"), ("numpy", "2")))


def test_a_manifest_at_an_unknown_schema_version_is_refused() -> None:
    # Reading the fields that happen to be recognised would produce a re-run
    # that is not the run, which is worse than refusing to read at all.
    with pytest.raises(ValueError, match="is not the version this code writes"):
        a_manifest(schema_version=SCHEMA_VERSION + 1)


def test_a_manifest_with_no_input_or_no_step_is_refused() -> None:
    with pytest.raises(ValueError, match="names no input"):
        a_manifest(inputs=())
    with pytest.raises(ValueError, match="records no step"):
        a_manifest(steps=())


def test_a_manifest_whose_seed_is_not_an_integer_is_refused() -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        a_manifest(seed=None)
    with pytest.raises(TypeError, match="seed must be an integer"):
        a_manifest(seed=1.5)
    # True is an int in Python and would otherwise be recorded as a seed of one.
    with pytest.raises(TypeError, match="seed must be an integer"):
        a_manifest(seed=True)
