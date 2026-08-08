"""Running a chain of steps, and refusing a chain that cannot mean anything.

A pipeline here is a list of identifiers and parameter records. It resolves each
identifier against a registry, checks that the chain can run at all, and then
runs it. The two halves are separate on purpose: the whole chain is checked
before the first step touches a surface, so a chain that is wrong in its fifth
step fails before anything has been computed rather than after four steps of
work and a plausible looking intermediate.

## Why the order is checked rather than trusted

Masking after a bandpass filter is the case this exists for. The filter spreads
the masked region into its neighbourhood, so a mask applied afterwards removes a
region that has already leaked into the surface around it. Nothing crashes. The
run exits zero and produces a surface that is wrong in a way that looks like
data.

A profile is a text file somebody edits, and a sweep permutes what a profile
says, so an impossible order is not a thing that only arrives by
misunderstanding. It arrives by editing.

## Why constraints name properties and not transforms

A constraint saying "not after ``bandpass``" is escaped by the second filtering
step somebody adds. A constraint saying "not after anything that has filtered
the surface" is not. So each transform declares what it produces, what it
requires and what it refuses in the vocabulary of
``gutachten.transforms.base.SurfaceProperty``, and this module carries the
properties forward through the chain along with the identifier of the step that
established each one. That is what lets a refusal name both transforms, which is
the difference between a message somebody can act on and one they have to
investigate.

## What is not here

The manifest that records a run is ``gutachten.manifest``. Writing one at the
end of a run and re-running from it is #51. Refusing an under-specified
parameter set against a profile is #53. This module runs a chain it was handed
and records what it did in the surface's provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gutachten.surface import Surface
from gutachten.transforms.base import Parameters, SurfaceProperty, Transform, record_for
from gutachten.transforms.registry import Registry

__all__ = ["OrderingError", "Step", "check_chain", "run_chain"]


class OrderingError(Exception):
    """A chain whose steps cannot run in the order they were given."""


@dataclass(frozen=True)
class Step:
    """One entry in a chain: which transform, and the parameters it runs with.

    The identifier rather than the transform itself, because a chain comes from
    a profile or a manifest, both of which are text, and resolving it against
    the registry is where an unknown or renamed step is caught.
    """

    identifier: str
    parameters: Parameters


def _established(
    steps: Sequence[Step], registry: Registry
) -> tuple[list[Transform], dict[SurfaceProperty, str]]:
    """Resolve the chain and refuse it if it cannot run in this order.

    Returns the resolved transforms and, for each property the chain
    establishes, the identifier of the step that established it. Where two steps
    establish the same property, the first is kept: the question a refusal
    answers is when the surface became that way, and the second step found it
    already so.

    The parameter type is checked here rather than in the run loop, so a chain
    whose fifth step carries the wrong record fails before the first has touched
    a surface. ``record_for`` refuses the same mistake inside a step, which is
    one step too late to stop four steps of work and an intermediate somebody
    will save.
    """
    resolved: list[Transform] = []
    established: dict[SurfaceProperty, str] = {}

    for position, step in enumerate(steps):
        transform = registry[step.identifier]

        if not isinstance(step.parameters, transform.parameters_type):
            raise TypeError(
                f"step {position} runs {transform.identifier!r}, which takes "
                f"{transform.parameters_type.__name__} and was given "
                f"{type(step.parameters).__name__}"
            )

        missing = sorted(item.value for item in transform.requires - set(established))
        if missing:
            raise OrderingError(
                f"step {position} runs {transform.identifier!r}, which requires the "
                f"surface to be {', '.join(missing)} and nothing before it in this chain "
                f"does that. Established so far: "
                f"{', '.join(sorted(item.value for item in established)) or 'nothing'}."
            )

        forbidden = sorted(transform.refuses & set(established), key=lambda item: item.value)
        if forbidden:
            culprit = established[forbidden[0]]
            raise OrderingError(
                f"step {position} runs {transform.identifier!r} after "
                f"{culprit!r}, and {transform.identifier!r} refuses a surface that is "
                f"{forbidden[0].value}. {culprit!r} made it "
                f"{forbidden[0].value}, so this chain would produce a surface that is "
                "wrong in a way nothing downstream can detect."
            )

        resolved.append(transform)
        for produced in transform.produces:
            established.setdefault(produced, transform.identifier)

    return resolved, established


def check_chain(steps: Sequence[Step], registry: Registry) -> None:
    """Refuse a chain that cannot run, without running any of it."""
    _established(steps, registry)


def run_chain(steps: Sequence[Step], registry: Registry, surface: Surface) -> Surface:
    """Run the chain over ``surface``, checking the whole of it first.

    Every step appends its record to the provenance chain, so the surface that
    comes out names the steps and the parameters that produced it rather than
    arriving anonymous.
    """
    if not steps:
        raise ValueError(
            "a chain with no steps is not a run. A pipeline that transformed nothing "
            "returns its input, and recording that as a run hides which chain produced "
            "the number."
        )

    resolved, _ = _established(steps, registry)

    for transform, step in zip(resolved, steps, strict=True):
        before = len(surface.provenance)
        surface = transform.apply(surface, step.parameters)
        if len(surface.provenance) != before + 1:
            raise RuntimeError(
                f"transform {transform.identifier!r} returned a surface carrying "
                f"{len(surface.provenance) - before} new provenance entries rather than "
                "one. A step that does not record itself makes the chain that produced a "
                f"number unreadable. Build the entry with {record_for.__name__}."
            )

    return surface
