"""A result type with the field this project refuses, so the walk has a target.

It lives here rather than in the package, because the whole point of the walk
over ``gutachten`` is that it comes back empty. A check whose only evidence is
an empty result has not been shown to reach anything, so the same walk is
pointed at this module and has to find what is in it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    """The convenience field somebody adds beside the ratio."""

    congruent: int
    identified: bool
