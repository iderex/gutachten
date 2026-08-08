"""Finding the numbers a transform did not declare.

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
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from gutachten.transforms.base import Transform

__all__ = ["EXEMPTION_MARKER", "STRUCTURAL_VALUES", "UndeclaredConstant", "undeclared_constants"]

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
