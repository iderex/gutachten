"""Finding the numbers a transform did not declare, and the steps nobody registered.

Both halves read the tree rather than asking a reviewer to. They are here
together because they fail the same way: silently, in a direction that looks
like everything is fine.

## The numbers half

A parameter that is not in the record does not exist, and the way that rule
actually gets broken is not by argument. Somebody needs a threshold, writes
``0.35`` where they need it, and every test still passes, because the number is
correct for the case in front of them. The sweep then reports that the step is
insensitive to its parameters, which is true and useless, because the number
that decides its behaviour is not one of them.

So the registered transforms are read rather than reviewed. This module parses
the source of the module each registered transform is defined in and reports
every numeric literal in it that is not structurally forced.

**What structurally forced means, and why the list is this short.** ``0``, ``1``
and ``2``, whole or as floats, are the numbers that appear in indexing, in
halving an interval, in a squared term and in a comparison against nothing. They
are not tunable: moving ``2`` in ``x ** 2`` does not adjust a threshold, it makes
it a different formula. Anything else is presumed tunable until somebody says
otherwise on the line itself.

**The escape hatch is a comment and it has to carry a reason.** A line ending in
``# structural: <reason>`` is exempt. That is deliberately visible: it appears in
the diff, it says why, and a reader can disagree with it. A switch that turned
the whole check off per module would be used once and then inherited by every
line added underneath it.

**The unit is the module, not the method.** A helper function beside ``apply``
is as much part of the step as ``apply`` is, and a constant hidden in one counts
the same. So the whole module a transform is defined in is read, and the
identifier in the report is the transform that led there. Two transforms sharing
a module are both reported for a literal in either of them, which is the right
answer for a rule about what a step may read and a slightly blunt one for
attribution.

**What this cannot do.** It reads literals, so a tunable number assembled out of
arithmetic on allowed ones, read from an environment variable, or imported from
another module passes it. It is a floor under the mistake somebody actually
makes, which is typing the number where it is needed, and it is not a proof that
every number a transform uses is declared.

## The registration half

Three separate obligations in this plan read the registry rather than the
directory: the manifest resolver, the sweep, and the constants audit above. A
step that is implemented and not registered is therefore invisible to all three
at once. It still runs, because whoever wrote it calls it directly, and the
sweep then reports on a pipeline that is missing a step.

``unregistered_transforms`` imports every module in a package, finds the classes
that satisfy the transform interface, and reports the ones the registry does not
hold. It reads the interface rather than a naming convention, so a step called
something nobody predicted is still found. What it cannot see is a step defined
outside the package it is pointed at, which is why the test points it at the
package the steps live in.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from gutachten.transforms.base import Transform
from gutachten.transforms.registry import Registry

__all__ = [
    "EXEMPTION_MARKER",
    "STRUCTURAL_VALUES",
    "UndeclaredConstant",
    "undeclared_constants",
    "unregistered_transforms",
]

#: The values that are not tunable in any reading. See the module docstring for
#: why the list stops here.
STRUCTURAL_VALUES: frozenset[float] = frozenset({0.0, 1.0, 2.0})

#: A line carrying this marker is exempt, and the text after it is the reason.
EXEMPTION_MARKER = "# structural:"


@dataclass(frozen=True)
class UndeclaredConstant:
    """One numeric literal that is not in a parameter record and not exempt."""

    identifier: str
    path: str
    line: int
    value: float

    def __str__(self) -> str:
        return (
            f"{self.identifier}: {self.path}:{self.line} uses the literal {self.value!r}, "
            f"which is neither a declared parameter nor structurally forced. Put it in "
            f"the parameter record, or mark the line '{EXEMPTION_MARKER} <reason>'."
        )


def _source_of(transform: Transform) -> tuple[Path, str]:
    try:
        path = inspect.getsourcefile(type(transform))
    except TypeError:
        # A class assembled at run time has no file for `inspect` to name, and
        # the exception it raises says "built-in class", which is a long way
        # from what actually happened.
        path = None
    if path is None or not Path(path).is_file():
        raise TypeError(
            f"transform {transform.identifier!r} has no source file, so its constants "
            "cannot be read. A transform built at run time cannot be audited and is "
            "not something this pipeline can record."
        )
    return Path(path), Path(path).read_text(encoding="utf-8")


def undeclared_constants(transforms: Iterable[Transform]) -> list[UndeclaredConstant]:
    """Every numeric literal in the given transforms that nobody declared.

    Sorted, so two runs over one registry report the same list in the same
    order.
    """
    found: list[UndeclaredConstant] = []
    for transform in transforms:
        path, source = _source_of(transform)
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, ast.Constant):
                continue
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if float(abs(value)) in STRUCTURAL_VALUES:
                continue
            line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
            if EXEMPTION_MARKER in line:
                continue
            found.append(
                UndeclaredConstant(
                    identifier=transform.identifier,
                    path=path.name,
                    line=node.lineno,
                    value=float(value),
                )
            )
    return sorted(found, key=lambda item: (item.identifier, item.path, item.line, item.value))


def unregistered_transforms(package: ModuleType, registry: Registry) -> list[str]:
    """Every transform class in ``package`` whose identifier the registry lacks.

    Returns qualified names rather than the classes, because the thing a reader
    has to act on is the file it is in, and sorted so two runs report the same
    list in the same order.
    """
    known = set(registry.identifiers())
    missing: list[str] = []

    for module in _modules_in(package):
        for name, candidate in vars(module).items():
            if not isinstance(candidate, type) or candidate.__module__ != module.__name__:
                continue
            if not _is_a_transform(candidate):
                continue
            if getattr(candidate, "identifier", "") in known:
                continue
            missing.append(f"{module.__name__}.{name}")

    return sorted(missing)


def _modules_in(package: ModuleType) -> list[ModuleType]:
    """``package`` and every module under it, imported.

    Imported rather than parsed, because the interface is a runtime protocol and
    a class that merely looks like one in source is not the thing the registry
    would have held.
    """
    modules = [package]
    # A plain module has no `__path__`, and being handed one is the ordinary
    # case rather than a mistake: `gutachten.transforms.base` is where the
    # interface lives and is worth pointing this at on its own.
    for found in pkgutil.walk_packages(getattr(package, "__path__", ()), f"{package.__name__}."):
        modules.append(importlib.import_module(found.name))
    return modules


def _is_a_transform(candidate: type) -> bool:
    """Whether ``candidate`` satisfies the transform interface.

    ``isinstance`` against a runtime protocol needs an instance, and a step's
    constructor is not something this check may call. So the members are looked
    for on the class, which is where a step declares them.
    """
    if getattr(candidate, "_is_protocol", False):
        # The interface itself declares every member it asks for, so a check
        # looking for the members finds the protocol first and reports the
        # definition of a transform as an unregistered one.
        return False
    return all(
        hasattr(candidate, member)
        for member in ("identifier", "version", "parameters_type", "produces", "refuses", "apply")
    )
