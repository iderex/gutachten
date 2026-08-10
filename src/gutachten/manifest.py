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
order with its version, its resolved parameters and whatever it measured while
it ran, the comparison stage with its own parameters where the run made one, the
seed, which of the two modes in ``gutachten.determinism`` the run was made in and
what its thread count was pinned to, the software version and the resolved
versions of the dependencies that affect a number, and the outputs by hash. A
field that is not in this list does not affect a result, and if one turns out to,
the schema is wrong and its version moves.

## What a step was told, and what it found

A step record keeps those two apart, under ``parameters`` and ``outcomes``. A
parameter is an input: a sweep varies it and a re-run has to reproduce it. An
outcome is a measurement of what the run did with that input, and the case it
exists for is a rejection threshold. A sweep reads manifests rather than
surfaces, because the surfaces a sweep of thousands of cells produces are not
kept, so a row saying a threshold moved the score with no record of how much
surface the threshold removed on the way is the less interesting half of the
measurement on its own.

They are disjoint by name and a record naming one name as both is refused. Merged
into one bag a reader could no longer tell an input a sweep varies from a
measurement of what the run did with it, and the two are read for opposite
purposes.

``resolve`` reads the parameters and never the outcomes. An outcome is not an
input, and feeding one back into a re-run would turn a reproduction into a
different experiment.

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

## The serialisation, and why it is shaped for a reader

JSON, indented, one value to a line, in the order the schema decides rather than
alphabetically. Two manifests from two runs should differ only where the runs
differ, and a reader comparing them by eye is the most common way a difference
will actually be found. One value to a line is what makes two runs differing in
one parameter produce manifests differing in one line.

Sorted where the container is unordered, and only there. Every mapping the
schema emits comes out in sorted key order, because iteration order over an
unordered container is a run-to-run difference nobody chose. The steps are a
sequence and keep the order they ran in, because levelling then filtering is a
different procedure from filtering then levelling.

## Re-running from one, and what is refused on the way

``resolve`` turns a manifest back into a chain the pipeline can run. It looks
each identifier up in the registry and compares the recorded version against the
registered one. A manifest naming a version that is no longer in the tree is
refused with both versions named, rather than run against the current step,
because silently substituting a newer step turns a reproduction into a different
experiment carrying the same label.

The parameter records are rebuilt from what the manifest recorded, not from the
profile the run used. A run that overrode one parameter is a run the profile does
not describe, and re-reading the profile would reproduce the run somebody meant
rather than the one that happened.

## What is here and what is not

The stamp identifying the build is #29 and lands in ``EnvironmentRecord`` when it
exists. The content addressed cache an input hash would be looked up in is #41,
so ``rerun`` is handed the surface rather than fetching it. The registry an
identifier resolves against is ``gutachten.transforms.registry``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gutachten.determinism import DeterminismRecord, RunMode
from gutachten.surface import ParameterValue, Surface
from gutachten.transforms.pipeline import Step, run_chain
from gutachten.transforms.registry import Registry

__all__ = [
    "SCHEMA_VERSION",
    "ComparisonRecord",
    "EnvironmentRecord",
    "FileRecord",
    "ProfileRecord",
    "RunManifest",
    "StepRecord",
    "VersionMismatch",
    "from_dict",
    "read",
    "record_run",
    "rerun",
    "resolve",
    "surface_digest",
]

#: Moves when the meaning of a field changes, not when a transform does. Moved
#: to 2 when the determinism record was added, because whether a run pinned its
#: thread count decides whether a re-run is expected to agree with it, and a
#: version 1 manifest does not say. Moved to 3 when the comparison record was
#: added, because the parameters of the stage that produces the number were not
#: recorded anywhere before it and a version 2 manifest cannot say what they
#: were. Moved to 4 when a step gained the outcomes it measured, because a
#: version 3 manifest recording no outcome and a version 4 manifest from a step
#: that found nothing are different statements, and without the version moving
#: they are the same bytes.
SCHEMA_VERSION = 4

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
    """One transform as it ran: what it was told, and what it found.

    The parameters are the resolved ones and not the profile's, because a run
    that overrode one is a run that did something the profile does not describe.
    They are a sorted tuple of pairs so that a manifest written twice is written
    identically.

    ``outcomes`` is what the step measured while it ran, and it is empty for a
    step that measured nothing. Empty rather than absent: a step that found
    nothing and a step that cannot find anything are the same shape here, and
    which of the two a reader is looking at is answered by the step, not by
    whether the key is present.
    """

    identifier: str
    version: str
    parameters: tuple[tuple[str, ParameterValue], ...]
    outcomes: tuple[tuple[str, ParameterValue], ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.identifier, "a step's identifier")
        _require_text(self.version, "a step's version")
        for plural, singular, entries in (
            ("parameters", "a parameter", self.parameters),
            ("outcomes", "an outcome", self.outcomes),
        ):
            keys = [key for key, _ in entries]
            if sorted(keys) != keys:
                raise ValueError(f"step {self.identifier!r} records its {plural} unsorted: {keys}")
            if len(set(keys)) != len(keys):
                raise ValueError(f"step {self.identifier!r} names {singular} twice: {keys}")
        if not self.parameters:
            raise ValueError(
                f"step {self.identifier!r} records no parameters. A step with nothing "
                "recorded is a step the sensitivity study cannot vary, which is the "
                "failure this schema exists against."
            )
        both = sorted(dict(self.parameters).keys() & dict(self.outcomes).keys())
        if both:
            raise ValueError(
                f"step {self.identifier!r} records {both} as both a parameter and an "
                "outcome. A sweep varies the parameters and reads the outcomes to explain "
                "what it saw, so a name carrying both leaves it varying its own "
                "explanation."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "version": self.version,
            "parameters": dict(self.parameters),
            "outcomes": dict(self.outcomes),
        }


@dataclass(frozen=True)
class ComparisonRecord:
    """The stage that turned two surfaces into numbers, and what it ran with.

    Shaped like a step record and separate from one, because a comparison takes
    two surfaces and produces neither, so it is not something the chain can
    carry. ``method`` is what the stage is called and ``version`` moves when the
    same input would produce a different number, which is the rule a transform's
    version follows applied to a stage that is not a transform.
    """

    method: str
    version: str
    parameters: tuple[tuple[str, ParameterValue], ...]

    def __post_init__(self) -> None:
        _require_text(self.method, "a comparison's method")
        _require_text(self.version, "a comparison's version")
        keys = [key for key, _ in self.parameters]
        if sorted(keys) != keys:
            raise ValueError(f"comparison {self.method!r} records its parameters unsorted: {keys}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"comparison {self.method!r} names a parameter twice: {keys}")
        if not self.parameters:
            raise ValueError(
                f"comparison {self.method!r} records no parameters. The stage that decides "
                "the number is the one a sensitivity study most needs to be able to vary, "
                "and a stage with nothing recorded cannot be varied from a manifest."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
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
    #: The comparison stage, or nothing where the run only preprocessed. Written
    #: either way rather than left out, so a manifest that made no comparison
    #: says so instead of being a manifest a reader has to guess about.
    comparison: ComparisonRecord | None = None
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
            "comparison": None if self.comparison is None else self.comparison.to_dict(),
            "seed": self.seed,
            "determinism": self.determinism.to_dict(),
            "environment": self.environment.to_dict(),
            "outputs": [record.to_dict() for record in self.outputs],
        }

    def to_text(self) -> str:
        """The manifest as the text a run writes and a reader diffs.

        ``sort_keys`` is off deliberately. The schema decides the order of its
        own keys, and sorting here would hide a schema that had stopped deciding
        it. What is sorted is what came out of an unordered container, and that
        happens in ``to_dict`` where the container is.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    def write(self, path: Path) -> None:
        """Write the manifest, with the line endings a diff across platforms needs."""
        path.write_text(self.to_text(), encoding="utf-8", newline="\n")


class VersionMismatch(Exception):
    """A manifest names a transform version the tree no longer holds."""


REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "inputs",
        "profile",
        "steps",
        "comparison",
        "seed",
        "determinism",
        "environment",
        "outputs",
    }
)


def _file_record(data: Mapping[str, Any]) -> FileRecord:
    return FileRecord(role=str(data["role"]), sha256=str(data["sha256"]))


def _step_record(data: Mapping[str, Any]) -> StepRecord:
    """One step out of a manifest this code has already agreed to read.

    The version gate stands in front of this, so a key missing here is a
    malformed manifest at the current version rather than an older one, and the
    message says so instead of sending a reader to look up which version added
    the field.
    """
    missing = sorted({"identifier", "version", "parameters", "outcomes"} - set(data))
    if missing:
        raise ValueError(
            f"a step at schema version {SCHEMA_VERSION} is missing {missing}. The version "
            "this manifest declares is one this code writes, so the field is absent rather "
            "than not yet invented."
        )
    return StepRecord(
        identifier=str(data["identifier"]),
        version=str(data["version"]),
        parameters=tuple(sorted(data["parameters"].items())),
        outcomes=tuple(sorted(data["outcomes"].items())),
    )


def from_dict(data: Mapping[str, Any]) -> RunManifest:
    """Rebuild a manifest from plain data, refusing one this code cannot read whole.

    Every field is taken from the data rather than defaulted. A field a reader
    supplies for itself is a field the manifest did not have to record, and the
    whole point of the schema is that it records all of them.

    The declared version is checked before any field inside a record is read.
    Read the other way round, a manifest from an older schema is refused for
    lacking a field that version had never heard of, which reads as a corrupt
    file and sends a reader to look for the corruption rather than to the
    version that would explain it.
    """
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        raise ValueError(
            f"a manifest is missing {missing}. Filling those in here would produce a "
            "re-run that is not the run, which is the failure the schema version exists "
            "to make visible."
        )

    declared = int(data["schema_version"])
    if declared != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema version {declared} is not the version this code writes, "
            f"which is {SCHEMA_VERSION}. Reading the fields that happen to be recognised "
            "would produce a re-run that is not the run."
        )

    determinism = data["determinism"]
    environment = data["environment"]
    comparison = data["comparison"]
    return RunManifest(
        schema_version=declared,
        inputs=tuple(_file_record(item) for item in data["inputs"]),
        profile=ProfileRecord(
            name=str(data["profile"]["name"]), version=str(data["profile"]["version"])
        ),
        steps=tuple(_step_record(step) for step in data["steps"]),
        comparison=None
        if comparison is None
        else ComparisonRecord(
            method=str(comparison["method"]),
            version=str(comparison["version"]),
            parameters=tuple(sorted(comparison["parameters"].items())),
        ),
        seed=int(data["seed"]),
        determinism=DeterminismRecord(
            mode=RunMode(determinism["mode"]), threads=determinism["threads"]
        ),
        environment=EnvironmentRecord(
            software_version=str(environment["software_version"]),
            dependencies=tuple(sorted(environment["dependencies"].items())),
        ),
        outputs=tuple(_file_record(item) for item in data["outputs"]),
    )


def read(path: Path) -> RunManifest:
    """Read a manifest, refusing one this code cannot read whole."""
    parsed: Any = json.loads(path.read_text(encoding="utf-8"))
    return from_dict(parsed)


def surface_digest(surface: Surface) -> str:
    """The sha256 of a surface's heights, in a byte form that does not drift.

    ``numpy.save`` rather than ``tobytes``, because the header carries the dtype
    and the shape. Two arrays with the same bytes and different shapes are not
    the same output, and a digest that could not tell them apart would call a
    transposed result a reproduction.
    """
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(surface.heights), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def resolve(manifest: RunManifest, registry: Registry) -> list[Step]:
    """Turn a manifest back into a chain, refusing a step the tree has moved on from.

    The parameter records are rebuilt from what the manifest recorded rather
    than from the profile it names. A run that overrode a parameter is a run the
    profile does not describe, and re-reading the profile would reproduce the
    run somebody meant instead of the one that happened.

    The outcomes are not read here and a step is not handed them. What a step
    found is a measurement of the first run, so feeding it back in would fix the
    answer the re-run is supposed to arrive at independently, and a reproduction
    that cannot disagree proves nothing.
    """
    chain: list[Step] = []
    for record in manifest.steps:
        transform = registry[record.identifier]
        if transform.version != record.version:
            raise VersionMismatch(
                f"the manifest names {record.identifier!r} at version {record.version!r} "
                f"and this tree holds version {transform.version!r}. Running the current "
                "step would make this a different experiment carrying the same label. "
                "Check out the tree that held the recorded version, or record this as a "
                "new run."
            )
        recorded = dict(record.parameters)
        declared = {field.name for field in dataclasses.fields(transform.parameters_type)}
        if declared != set(recorded):
            raise ValueError(
                f"the manifest records {sorted(recorded)} for {record.identifier!r}, and "
                f"the version {record.version!r} in this tree takes {sorted(declared)}. A "
                "step whose parameters changed without its version changing is what this "
                "compares against."
            )
        chain.append(
            Step(
                identifier=record.identifier,
                parameters=transform.parameters_type(**recorded),
            )
        )
    return chain


def record_run(
    role: str,
    surface: Surface,
    profile: ProfileRecord,
    chain: Sequence[Step],
    registry: Registry,
    seed: int,
    determinism: DeterminismRecord,
    environment: EnvironmentRecord,
    comparison: ComparisonRecord | None = None,
) -> tuple[Surface, RunManifest]:
    """Run a chain and return what it produced together with the manifest of it.

    The manifest is built from what ran rather than from what was asked for. The
    versions come off the registered transforms and the parameters off the
    records that were handed to them, so a manifest cannot describe a run that
    did not happen.

    The outcomes come off the surface that came out. A step measures while it
    runs and writes what it found into the provenance entry it appends, so what
    it found is only knowable from the result, and reading it from anywhere else
    would be recording what the step was expected to find.
    """
    already = len(surface.provenance)
    result = run_chain(chain, registry, surface)
    applied = result.provenance[already:]

    steps = []
    for step, record in zip(chain, applied, strict=True):
        if record.name != step.identifier:
            raise RuntimeError(
                f"step {step.identifier!r} left a provenance entry naming {record.name!r}. "
                "The outcomes are matched to the steps by the order they ran in, so an "
                "entry under another name would file one step's measurements under "
                "another step's parameters."
            )
        steps.append(
            StepRecord(
                identifier=step.identifier,
                version=registry[step.identifier].version,
                parameters=tuple(
                    sorted(
                        (field.name, getattr(step.parameters, field.name))
                        for field in dataclasses.fields(step.parameters)  # type: ignore[arg-type]
                    )
                ),
                outcomes=record.outcomes,
            )
        )

    manifest = RunManifest(
        inputs=(FileRecord(role=role, sha256=surface_digest(surface)),),
        profile=profile,
        steps=tuple(steps),
        seed=seed,
        determinism=determinism,
        environment=environment,
        outputs=(FileRecord(role="surface", sha256=surface_digest(result)),),
        comparison=comparison,
    )
    return result, manifest


def rerun(manifest: RunManifest, registry: Registry, surface: Surface) -> Surface:
    """Run the chain a manifest recorded, refusing a result that is not the recorded one.

    ``surface`` is handed in rather than fetched. A manifest names its inputs by
    hash and the content addressed cache those hashes are looked up in is #41.
    What can be checked today is that the surface handed in is the one the
    manifest names, and that is checked.
    """
    digest = surface_digest(surface)
    named = [record.sha256 for record in manifest.inputs]
    if digest not in named:
        raise ValueError(
            f"the surface handed in hashes to {digest} and the manifest names {named}. "
            "Re-running a recorded chain against a different input produces a number "
            "under a label saying it came from another one."
        )

    result = run_chain(resolve(manifest, registry), registry, surface)

    produced = surface_digest(result)
    expected = [record.sha256 for record in manifest.outputs]
    if produced not in expected:
        raise ValueError(
            f"the re-run produced {produced} and the manifest recorded {expected}. The "
            "chain, the parameters and the input all matched, so what moved is the code "
            "behind a step whose version did not."
        )
    return result
