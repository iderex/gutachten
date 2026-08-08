"""A transform with two numbers nobody declared. Fixture, not a step.

This is the near miss the audit is built around, written the way somebody
actually writes it: the numbers are correct for the case in front of them, every
test passes, and the sweep afterwards reports that the step is insensitive to
its parameters because the numbers that decide its behaviour are not among them.

The two sit at different depths in the syntax tree on purpose. The audit walks
the tree breadth first, so the shallower literal on the later line is reached
first, and a report that was not sorted afterwards would come out in an order no
reader could follow back to the source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gutachten.surface import Surface
from gutachten.transforms.base import SurfaceProperty, record_for


@dataclass(frozen=True)
class ClipParameters:
    """The parameter this step admits to having."""

    factor: float


class Clip:
    """Multiply by a declared factor, then clip and floor at undeclared numbers."""

    identifier = "example-clip"
    version = "1"
    parameters_type = ClipParameters
    produces = frozenset({SurfaceProperty.FILTERED})
    requires = frozenset({SurfaceProperty.LEVELLED})
    refuses = frozenset[SurfaceProperty]()

    def apply(self, surface: Surface, parameters: ClipParameters) -> Surface:
        heights = surface.heights * parameters.factor
        clipped = np.clip(heights, -0.35, 0.35)
        floor = 0.05
        clipped[np.abs(clipped) < floor] = 0.0
        return surface.with_transform(record_for(self, parameters), np.asarray(clipped))
