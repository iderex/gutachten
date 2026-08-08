"""Named parameter sets, and where every value in one came from.

A profile fixes every parameter of every step in a chain, under a name and a
version of its own. It is the only place in this project a default value is
allowed to live: a parameter record carries none, because a default in the code
is a value nobody chose for the run that used it and is invisible to the sweep
that is supposed to be varying it. That argument is in
``gutachten.transforms.base``; this module is the artefact it points at.

## Why every value carries its provenance

A profile that only holds numbers is a profile whose numbers cannot be argued
with. The first one here reproduces a published preprocessing chain, and the
interesting thing about that chain is not the values it fixes but how few of
them can be traced to a statement anybody published. That gap is a result, so it
is recorded as data rather than as a comment: every parameter of every step
carries an ``origin`` out of a closed vocabulary, the place the value came from,
and how much weight that place carries.

``stated`` means the named source gives this value for this parameter.
``adapted`` means the source gives something this was converted from, and the
conversion is named in ``where``. ``not-sourced`` means nothing states it and
``where`` says what was used instead. A free string would collapse the three
into prose nobody can count, which is the failure this vocabulary exists
against.

## Why JSON

The step parameters in a profile and the step parameters in a run manifest are
the same mapping, and the manifest is JSON, so a profile and the manifest it
produced can be compared without converting between two formats. Three of the
five registered parameter records carry a field that is legitimately absent, and
JSON has a null to write it with, which TOML has not. Both are read by the
standard library, so neither adds a dependency; what decided it is the first
reason.

## What is refused here

A profile is text somebody edits, so every way it can be wrong is refused at
load rather than at the first surface it touches. A parameter the transform
declares and the profile does not name, a parameter the profile names and the
transform does not declare, a value of the wrong type, a parameter with no
recorded provenance, and a chain whose steps cannot run in the order given are
all refused with the profile named, because the message has to say which of
several files on disk to open.

## What is not here

Which profile an operator selects and how they select it is the operator surface
and is #122. Carrying ``profiles/`` into a built distribution is #132. Refusing
an under-specified parameter set at every route into the pipeline, rather than
at this one, is #53.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gutachten.manifest import ProfileRecord
from gutachten.surface import ParameterValue
from gutachten.transforms.base import Parameters
from gutachten.transforms.pipeline import Step, check_chain
from gutachten.transforms.registry import Registry

__all__ = [
    "ORIGINS",
    "ParameterSource",
    "Profile",
    "ProfileError",
    "ProfileStep",
    "load",
    "load_directory",
]

#: What a recorded provenance may say, and nothing else. The three are different
#: claims and a reader counting how much of a chain is actually published has to
#: be able to tell them apart.
ORIGINS = ("stated", "adapted", "not-sourced")

_TOP_LEVEL = frozenset({"name", "version", "description", "steps"})
_STEP_FIELDS = frozenset({"transform", "parameters", "sources"})
_SOURCE_FIELDS = frozenset({"origin", "where", "confidence"})


class ProfileError(Exception):
    """A profile file that cannot be turned into a chain this pipeline can run."""


@dataclass(frozen=True)
class ParameterSource:
    """Where one parameter's value came from, and how much that is worth."""

    origin: str
    where: str
    confidence: str

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ProfileError(
                f"a parameter records its origin as {self.origin!r}, which is not one of "
                f"{list(ORIGINS)}. A value outside the vocabulary cannot be counted, and "
                "counting how much of a published chain is actually published is what "
                "these records are for."
            )
        for name, value in (("where", self.where), ("confidence", self.confidence)):
            if not isinstance(value, str) or not value.strip():
                raise ProfileError(
                    f"a parameter records {value!r} as its {name}. An origin of "
                    f"{self.origin!r} with nothing behind it says less than no record at "
                    "all, because it reads as though somebody had checked."
                )


@dataclass(frozen=True)
class ProfileStep:
    """One step of a profile: the transform, its parameters, and their provenance."""

    identifier: str
    parameters: Parameters
    sources: tuple[tuple[str, ParameterSource], ...]


@dataclass(frozen=True)
class Profile:
    """A named, versioned parameter set for a whole chain."""

    name: str
    version: str
    description: str
    steps: tuple[ProfileStep, ...]

    def record(self) -> ProfileRecord:
        """What a run manifest names this profile by."""
        return ProfileRecord(name=self.name, version=self.version)

    def chain(self) -> list[Step]:
        """The chain the pipeline runs, in the order the profile gave it."""
        return [Step(identifier=step.identifier, parameters=step.parameters) for step in self.steps]


def _require_mapping(value: object, what: str, profile: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileError(
            f"{what} in profile {profile!r} is a {type(value).__name__}, not an object"
        )
    return value


def _require_text(value: object, what: str, profile: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(
            f"{what} in profile {profile!r} must be a non-empty string, got {value!r}"
        )
    return value


def _accepted_types(annotation: object) -> tuple[set[type], bool]:
    """The concrete types an annotation admits, and whether it admits nothing.

    Only the shapes a parameter record actually uses are handled: a plain type
    and a union of one with ``None``. Anything else is refused rather than
    guessed at, because a parameter this cannot check is a parameter that
    reaches a surface as whatever JSON happened to hold.
    """
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        members = set(typing.get_args(annotation))
    else:
        members = {annotation}

    optional = type(None) in members
    concrete = {member for member in members if member is not type(None)}
    if not concrete or any(not isinstance(member, type) for member in concrete):
        raise ProfileError(
            f"a parameter is declared as {annotation!r}, which this loader cannot check a "
            "profile value against. A value it cannot check is a value that reaches a "
            "surface as whatever the file happened to hold."
        )
    return typing.cast(set[type], concrete), optional


def _value_for(
    field_name: str,
    annotation: object,
    value: object,
    *,
    identifier: str,
    profile: str,
) -> ParameterValue:
    """One parameter value, checked against what its field declares.

    ``bool`` is separated from ``int`` in both directions. Python calls ``True``
    an integer, so a setting written as ``1`` and an order written as ``true``
    both pass an ``isinstance`` check that does not say otherwise, and a run
    that masked nothing because a flag arrived as a number is a run that exits
    zero.
    """
    accepted, optional = _accepted_types(annotation)
    where = f"parameter {field_name!r} of step {identifier!r} in profile {profile!r}"

    if value is None:
        if optional:
            return None
        raise ProfileError(
            f"{where} is null, and the field does not admit it. Absent and unset are "
            "different states here and the record says which one this field has."
        )

    if bool in accepted:
        if not isinstance(value, bool):
            raise ProfileError(f"{where} takes true or false, got {value!r}")
        return value
    if isinstance(value, bool):
        raise ProfileError(
            f"{where} is {value!r}, and the field is declared "
            f"{', '.join(sorted(item.__name__ for item in accepted))}. A setting written "
            "where a number belongs arrives as 0 or 1 and the run exits zero."
        )
    if int in accepted and isinstance(value, int):
        return value
    if float in accepted and isinstance(value, int | float):
        return float(value)
    if str in accepted and isinstance(value, str):
        return value

    raise ProfileError(
        f"{where} is {value!r}, and the field takes "
        f"{', '.join(sorted(item.__name__ for item in accepted))}"
    )


def _step(
    data: Mapping[str, Any],
    position: int,
    registry: Registry,
    profile: str,
) -> ProfileStep:
    unknown = sorted(set(data) - _STEP_FIELDS)
    if unknown:
        raise ProfileError(
            f"step {position} of profile {profile!r} carries {unknown}, which this schema "
            "does not read. A key nothing reads is a parameter somebody believes they set."
        )
    missing = sorted(_STEP_FIELDS - set(data))
    if missing:
        raise ProfileError(f"step {position} of profile {profile!r} is missing {missing}")

    identifier = _require_text(data["transform"], f"the transform of step {position}", profile)
    try:
        transform = registry[identifier]
    except KeyError as unknown_step:
        raise ProfileError(
            f"step {position} of profile {profile!r} names {identifier!r}, and {unknown_step}"
        ) from None

    values = _require_mapping(data["parameters"], f"the parameters of step {position}", profile)
    sources = _require_mapping(data["sources"], f"the sources of step {position}", profile)

    hints = typing.get_type_hints(transform.parameters_type)
    declared = [field.name for field in dataclasses.fields(transform.parameters_type)]

    unset = sorted(set(declared) - set(values))
    if unset:
        raise ProfileError(
            f"profile {profile!r} runs {identifier!r} without setting {unset}. Every "
            "parameter of a step is set in the profile, because the record carries no "
            "default and a value nobody wrote down is a value the sweep cannot vary. Add "
            f"{unset} to this step in the profile."
        )
    surplus = sorted(set(values) - set(declared))
    if surplus:
        raise ProfileError(
            f"profile {profile!r} sets {surplus} on {identifier!r}, which takes "
            f"{declared}. A parameter the step does not read is a setting somebody "
            "believes is in force."
        )

    undocumented = sorted(set(declared) - set(sources))
    if undocumented:
        raise ProfileError(
            f"profile {profile!r} sets {undocumented} on {identifier!r} without recording "
            "where the value came from. A number with no provenance in a profile is the "
            "hand configuration this project exists to measure, written down one layer up."
        )
    orphaned = sorted(set(sources) - set(declared))
    if orphaned:
        raise ProfileError(
            f"profile {profile!r} records provenance for {orphaned} on {identifier!r}, "
            f"which takes {declared}. A provenance entry outlives the parameter it was "
            "written for, and then it describes nothing."
        )

    parameters = transform.parameters_type(
        **{
            name: _value_for(
                name, hints[name], values[name], identifier=identifier, profile=profile
            )
            for name in declared
        }
    )
    return ProfileStep(
        identifier=identifier,
        parameters=parameters,
        sources=tuple(
            (name, _source(sources[name], name, identifier, profile)) for name in declared
        ),
    )


def _source(data: object, parameter: str, identifier: str, profile: str) -> ParameterSource:
    entry = _require_mapping(
        data, f"the provenance of {parameter!r} on step {identifier!r}", profile
    )
    wrong = sorted(set(entry) ^ _SOURCE_FIELDS)
    if wrong:
        raise ProfileError(
            f"the provenance of {parameter!r} on {identifier!r} in profile {profile!r} "
            f"differs from {sorted(_SOURCE_FIELDS)} by {wrong}"
        )
    return ParameterSource(
        origin=_require_text(entry["origin"], f"the origin of {parameter!r}", profile),
        where=entry["where"],
        confidence=entry["confidence"],
    )


def _profile(data: Mapping[str, Any], registry: Registry, called: str) -> Profile:
    unknown = sorted(set(data) - _TOP_LEVEL)
    if unknown:
        raise ProfileError(f"profile {called!r} carries {unknown}, which this schema does not read")
    missing = sorted(_TOP_LEVEL - set(data))
    if missing:
        raise ProfileError(f"profile {called!r} is missing {missing}")

    name = _require_text(data["name"], "the profile name", called)
    version = _require_text(data["version"], "the profile version", called)
    description = _require_text(data["description"], "the profile description", called)

    steps = data["steps"]
    if not isinstance(steps, Sequence) or isinstance(steps, str) or not steps:
        raise ProfileError(
            f"profile {name!r} names no step. A profile that runs nothing is not a "
            "configuration of this pipeline, and recording a run against one would hide "
            "which chain produced the number."
        )

    resolved = tuple(
        _step(_require_mapping(step, f"step {position}", name), position, registry, name)
        for position, step in enumerate(steps)
    )
    check_chain([Step(step.identifier, step.parameters) for step in resolved], registry)
    return Profile(name=name, version=version, description=description, steps=resolved)


def load(path: Path, registry: Registry) -> Profile:
    """Read one profile, refusing anything that could not run as written.

    The file name and the recorded name have to agree. A profile selected by
    name and found by file name is one lookup in two places, and the run that
    goes wrong is the one where a file was copied and its name left behind.
    """
    parsed: Any = json.loads(path.read_text(encoding="utf-8"))
    profile = _profile(_require_mapping(parsed, "the profile", path.name), registry, path.name)
    if profile.name != path.stem:
        raise ProfileError(
            f"{path.name} holds a profile called {profile.name!r}. A profile found by file "
            "name and selected by recorded name is one thing looked up two ways, and a "
            "copied file is where they part."
        )
    return profile


def load_directory(directory: Path, registry: Registry) -> tuple[Profile, ...]:
    """Every profile in ``directory``, in file name order."""
    return tuple(load(path, registry) for path in sorted(directory.glob("*.json")))
