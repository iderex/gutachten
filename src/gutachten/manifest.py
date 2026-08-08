"""The run record, and the schema it is written to.

A run writes down what it did: the input, every transform with its name, version
and parameters, the resolved dependency versions, and the identity of the build
that produced the output. The manifest is what makes a result re-runnable by
somebody else rather than only by the person who ran it, and it is what the
sensitivity study varies against.

## The manifest is the run

Not a log of it. Everything that decides the numbers is in here, and a manifest
fed back in reproduces the run. That is the property the whole board rests on,
and it is why the schema is a set of frozen records with a version rather than
whatever dictionary a writer happened to assemble.

What it names: the inputs by hash, the profile by name and version, every step in
order with its version and its resolved parameters, the seed, which of the two
modes in ``gutachten.determinism`` the run was made in and what its thread count
was pinned to, the software version and the resolved versions of the
dependencies that affect a number, and the outputs by hash. A field that is not
in this list does not affect a result, and if one turns out to, the schema is
wrong and its version moves.

Inputs and outputs are named by hash rather than carried. A scan is somebody
else's file under somebody else's terms, and a hash is what lets a result be
checked by anybody who can obtain the same file while keeping the file itself out
of this repository. A hash also detects the case a path cannot: an input that
changed upstream between two runs recorded under one name.

## Versioned, and what the version is for

``SCHEMA_VERSION`` moves when the meaning of a field changes or a field that
affects a result is added or removed. A reader that meets a manifest at a version
it does not know refuses it rather than reading the fields it recognises, because
a partial read of a run record produces a re-run that is not the run.

It does not move when a transform changes; that is the transform's own version,
in its step record. The two are separate on purpose. A levelling step whose
output changed is a different procedure recorded in the same schema.

## What is here and what is not

This module holds the schema and nothing else: the records, their refusals, and a
stable serialisation. Writing a manifest at the end of a run and re-running from
one is #51. The stamp identifying the build is #29 and lands in
``EnvironmentRecord`` when it exists. The registry a step identifier resolves
against is ``gutachten.transforms.registry``.

Serialisation sorts every mapping it emits. Iteration order over an unordered
container is a run-to-run difference nobody chose, and a manifest that is the
authority for whether two runs were the same has no business differing in the
order of its own keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from gutachten.determinism import DeterminismRecord
from gutachten.surface import ParameterValue

__all__ = [
    "SCHEMA_VERSION",
    "EnvironmentRecord",
    "FileRecord",
    "ProfileRecord",
    "RunManifest",
    "StepRecord",
]

#: Moves when the meaning of a field changes, not when a transform does. Moved
#: to 2 when the determinism record was added, because whether a run pinned its
#: thread count decides whether a re-run is expected to agree with it, and a
#: version 1 manifest does not say.
SCHEMA_VERSION = 2

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def _require_text(value: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class FileRecord:
    """A file a run read or wrote, named by content rather than by path.

    ``role`` is what the run called it, such as the identifier a scan was
    fetched under. ``sha256`` is what actually arrived. Both are needed: the role
    is how a reader finds the file again, and the hash is how they know it is the
    same file.
    """

    role: str
    sha256: str

    def __post_init__(self) -> None:
        _require_text(self.role, "a file record's role")
        if not isinstance(self.sha256, str) or not _SHA256.match(self.sha256):
            raise ValueError(
                f"{self.role!r} carries {self.sha256!r} as a sha256, which is not 64 "
                "lowercase hex characters. A hash that is not checkable is a field that "
                "looks like provenance and is not."
            )

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "sha256": self.sha256}


@dataclass(frozen=True)
class ProfileRecord:
    """The named parameter set a run used, with its version."""

    name: str
    version: str

    def __post_init__(self) -> None:
        _require_text(self.name, "a profile's name")
        _require_text(self.version, "a profile's version")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class StepRecord:
    """One transform as it ran, with the parameters it resolved to.

    The parameters are the resolved ones and not the profile's, because a run
    that overrode one is a run that did something the profile does not describe.
    They are a sorted tuple of pairs so that a manifest written twice is written
    identically.
    """

    identifier: str
    version: str
    parameters: tuple[tuple[str, ParameterValue], ...]

    def __post_init__(self) -> None:
        _require_text(self.identifier, "a step's identifier")
        _require_text(self.version, "a step's version")
        keys = [key for key, _ in self.parameters]
        if sorted(keys) != keys:
            raise ValueError(f"step {self.identifier!r} records its parameters unsorted: {keys}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"step {self.identifier!r} names a parameter twice: {keys}")
        if not self.parameters:
            raise ValueError(
                f"step {self.identifier!r} records no parameters. A step with nothing "
                "recorded is a step the sensitivity study cannot vary, which is the "
                "failure this schema exists against."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "version": self.version,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class EnvironmentRecord:
    """What the numbers were produced by, beyond the code in this repository.

    ``dependencies`` is the resolved versions rather than the declared ranges. A
    result reproduced against a different SciPy is a different claim, and a range
    does not say which one ran.
    """

    software_version: str
    dependencies: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_text(self.software_version, "the software version")
        names = [name for name, _ in self.dependencies]
        if sorted(names) != names:
            raise ValueError(f"dependencies are recorded unsorted: {names}")
        if len(set(names)) != len(names):
            raise ValueError(f"a dependency is recorded twice: {names}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "software_version": self.software_version,
            "dependencies": dict(self.dependencies),
        }


@dataclass(frozen=True)
class RunManifest:
    """Everything that decided a run, in the order a reader needs it."""

    inputs: tuple[FileRecord, ...]
    profile: ProfileRecord
    steps: tuple[StepRecord, ...]
    seed: int
    determinism: DeterminismRecord
    environment: EnvironmentRecord
    outputs: tuple[FileRecord, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.determinism, DeterminismRecord):
            raise TypeError(
                f"a run manifest must record how it was made, got {self.determinism!r}. "
                "Whether the thread count was pinned decides whether a re-run is "
                "expected to agree, and a reader cannot infer it from the numbers."
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema version {self.schema_version} is not the version this "
                f"code writes, which is {SCHEMA_VERSION}. Reading the fields that happen "
                "to be recognised would produce a re-run that is not the run."
            )
        if not self.inputs:
            raise ValueError("a run manifest names no input, so nothing can be re-fetched")
        if not self.steps:
            raise ValueError(
                "a run manifest records no step. A run that transformed nothing is not "
                "the thing this schema describes, and recording it as one hides which "
                "chain produced the number."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(
                f"the seed must be an integer that was set explicitly, got {self.seed!r}. "
                "A run with no recorded seed cannot be repeated and its noise cannot be "
                "separated from a parameter effect."
            )

    def to_dict(self) -> dict[str, Any]:
        """The manifest as plain data, with every mapping in sorted key order."""
        return {
            "schema_version": self.schema_version,
            "inputs": [record.to_dict() for record in self.inputs],
            "profile": self.profile.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "seed": self.seed,
            "determinism": self.determinism.to_dict(),
            "environment": self.environment.to_dict(),
            "outputs": [record.to_dict() for record in self.outputs],
        }
