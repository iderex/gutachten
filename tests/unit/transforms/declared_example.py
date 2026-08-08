"""A transform that declares every number it uses. Fixture, not a step.

It exists so the audit has something it must pass, beside the module next to it
that it must fail. A check that has only ever been shown failing is a check
nobody knows the shape of.

The one literal here that is not 0, 1 or 2 carries the marker and its reason, so
the escape hatch is exercised as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gutachten.surface import Surface
from gutachten.transforms.base import record_for


@dataclass(frozen=True)
class ScaleParameters:
    """Everything this step does is decided here and nowhere else."""

    factor: float


class Scale:
    """Multiply the heights by a declared factor."""

    identifier = "example-scale"
    version = "1"
    parameters_type = ScaleParameters

    def apply(self, surface: Surface, parameters: ScaleParameters) -> Surface:
        heights = surface.heights * parameters.factor
        percent = 100.0  # structural: converting a fraction to percent is not a tunable
        _ = percent
        return surface.with_transform(record_for(self, parameters), np.asarray(heights))
