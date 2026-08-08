"""What a preprocessing step is here: an identifier, a version and a record.

A transform takes a surface and a parameter record and returns a surface with
one more entry in its provenance chain. It declares an identifier, a semantic
version of its own, and the type of the parameter record it accepts. That is the
whole interface, and the shape of it is the decision this module exists to hold.

## Why a parameter not in the record does not exist

The published performance figures for this method sit on a preprocessing chain
that was configured by hand, and the claim this project makes is that nobody has
shown how sensitive the result is to that configuration. A sweep can only move
parameters it can see.

So a step that reads a tunable constant out of its own source, or that fills in a
default when a key is missing, is a step whose contribution to the result is
invisible to every measurement this project will make. The sensitivity report
would then understate the sensitivity while looking thorough, which is a worse
outcome than not running one at all: a number nobody produced is not quoted, and
a number produced by an apparatus with a blind spot is.

Three rules follow, and each is refused by something rather than asked for.

**A parameter record declares its fields with no defaults.** ``check_parameters``
refuses a record type carrying one. A default in the record is a value chosen
once by whoever added the field, and it then travels into every run that does not
mention the field, including the sweeps that are supposed to be varying it.

**Defaults live in named profiles.** A profile is a versioned artefact under
``profiles/``, it is named in the manifest, and it can therefore be compared
between two runs in one word. The tedium of typing every parameter is real and
the answer to it is a profile, which is recorded, rather than a default in the
code, which is not. Refusing an under-specified parameter set is #53.

**The version changes when the output changes for the same input.** It is the
transform's own version and not the software version, so a result recorded a
year ago can be identified as having come from a different levelling step rather
than being compared against a current run as though the two were one procedure.
Nothing can check that somebody bumped it; what is checked is that a version
exists and is not empty, and the rest is what the review is for.

## What is not decided here

The registry, the ordering constraints between transforms and the pipeline that
runs a chain are #49. The manifest that records a run is
``gutachten.manifest``. The individual steps are the issues that follow this
one. This module holds the interface and the rules about parameter records, and
nothing that knows what a striation is.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

from gutachten.surface import Surface, TransformRecord

__all__ = [
    "Parameters",
    "Transform",
    "check_parameters",
    "record_for",
]


@runtime_checkable
class Parameters(Protocol):
    """A transform's parameter record.

    Any frozen dataclass whose fields carry no defaults. It is a protocol rather
    than a base class so that a parameter record is a plain dataclass a reader
    can see all of, without a base class contributing fields from somewhere
    else.
    """

    __dataclass_fields__: dict[str, dataclasses.Field[object]]


@runtime_checkable
class Transform(Protocol):
    """One preprocessing step.

    ``identifier`` is what a manifest and a profile name the step by, and it is
    stable across versions: a run recorded against ``level`` is a run of the
    levelling step whichever version produced it.
    """

    @property
    def identifier(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def parameters_type(self) -> type: ...

    def apply(self, surface: Surface, parameters: Parameters) -> Surface: ...


def check_parameters(parameters_type: type) -> None:
    """Refuse a parameter record that could supply a value nobody recorded.

    Called at registration rather than at the first run, so the failure lands on
    whoever added the field instead of on whoever was running a sweep six months
    later and wondering why one parameter never moved the score.
    """
    if not dataclasses.is_dataclass(parameters_type):
        raise TypeError(
            f"{parameters_type!r} is not a dataclass; a parameter record has to be one "
            "so that its fields can be enumerated by the manifest and the sweep"
        )

    if not parameters_type.__dataclass_params__.frozen:  # type: ignore[attr-defined]
        raise TypeError(
            f"{parameters_type.__name__} is not frozen; a parameter set that can be "
            "changed after a run has recorded it makes the record a description of "
            "what was intended rather than of what ran"
        )

    with_defaults = [
        field.name
        for field in dataclasses.fields(parameters_type)
        if field.default is not dataclasses.MISSING
        or field.default_factory is not dataclasses.MISSING
    ]
    if with_defaults:
        raise TypeError(
            f"{parameters_type.__name__} gives a default to {with_defaults}. A default "
            "in a parameter record is a value nobody chose for the run that used it, and "
            "it is invisible to the sweep that is supposed to be varying it. Put the "
            "value in a named profile under profiles/ instead, where it is versioned and "
            "appears in the manifest."
        )

    if not dataclasses.fields(parameters_type):
        raise TypeError(
            f"{parameters_type.__name__} declares no parameters at all. A step with "
            "nothing to vary is not a step this pipeline can be held to; say so with a "
            "field rather than with an empty record."
        )


def record_for(transform: Transform, parameters: Parameters) -> TransformRecord:
    """The provenance entry for one application of ``transform``.

    Every field of the record goes in, so a chain read back names the numbers
    that ran rather than the ones a profile happened to hold afterwards.
    """
    if not isinstance(parameters, transform.parameters_type):
        raise TypeError(
            f"transform {transform.identifier!r} takes "
            f"{transform.parameters_type.__name__} and was given "
            f"{type(parameters).__name__}"
        )
    values = {
        field.name: getattr(parameters, field.name)
        for field in dataclasses.fields(parameters)  # type: ignore[arg-type]
    }
    return TransformRecord.of(transform.identifier, transform.version, **values)
