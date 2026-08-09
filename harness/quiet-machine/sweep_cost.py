"""What one cell of a sweep costs in seconds, measured rather than guessed.

A cell is a whole comparison: the preprocessing chain over both surfaces, the
registration search, and the congruent cell rule over what it found. The
registration is the expensive part and `registration_cost.py` measures it alone;
this measures what a sweep actually pays per row of its results table, which is
the number a sample size has to be divided out of.

Out of the gate for the reason this directory is named for. A wall clock reading
is a property of the machine, so the same code is slow on a loaded agent and fast
on an idle one and neither run says anything about the tree. What the suite
asserts instead is `Registration.correlations`, which is arithmetic on the
settings and is the same everywhere.

Run it with:

    uv run python harness/quiet-machine/sweep_cost.py

It prints one row per configuration and nothing else. The configurations are not
a recommendation. They are the base of the preregistered design and the corner of
the declared ranges where the search is most expensive, because the cost of a
sweep is not one number: it varies across the space the sweep moves through, and
a design costed only at its base understates what it will pay.

Paste the output into `docs/sensitivity-design.md` together with the machine, so
the sample size there follows from a number somebody observed.
"""

from __future__ import annotations

# The pin has to be applied before the numerical stack loads, because a BLAS
# reads its thread count once, when it is imported. Every other import in this
# file is therefore below it.
from gutachten.determinism import pin_threads

DETERMINISM = pin_threads()

import json  # noqa: E402
import platform  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy  # noqa: E402
import scipy  # noqa: E402

import gutachten.transforms  # noqa: E402, F401  (importing registers the steps)
from gutachten.compare.register import RegistrationParameters, angles  # noqa: E402
from gutachten.manifest import EnvironmentRecord  # noqa: E402
from gutachten.sweep.design import load, load_ranges  # noqa: E402
from gutachten.sweep.runner import run  # noqa: E402
from gutachten.transforms.registry import REGISTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: Held still so the row says what the search cost and not what the environment
#: record happened to hold. Nothing here is reported as a result.
ENVIRONMENT = EnvironmentRecord(
    software_version="not-a-release", dependencies=(("numpy", numpy.__version__),)
)

#: The base search settings of the preregistered design.
BASE = {
    "grid": 6,
    "minimum_valid": 0.5,
    "rotation_range_deg": 10.0,
    "rotation_step_deg": 2.0,
    "translation_limit": 8,
}

#: The corner of the declared ranges in `docs/ranges.json` where the search
#: evaluates the most correlations: the finest grid, the widest rotation range
#: and the finest rotation step the ranges admit.
WORST = {
    "grid": 12,
    "minimum_valid": 0.5,
    "rotation_range_deg": 30.0,
    "rotation_step_deg": 0.25,
    "translation_limit": 32,
}

#: One field size per row, in samples, per configuration. The base is measured
#: at two sizes because how the cost moves with the area is what says whether a
#: design on real scans is affordable. The worst corner is measured at the
#: smaller size only: it is nearly ninety times the base in correlations, and a
#: second size there costs hours to learn something the base rows already say.
SIZES = {"base": (192, 384), "worst": (192,)}


def a_design(directory: Path, size: int, search: dict[str, object]) -> Path:
    """A design of exactly two cells, on one matching pair, at the stated size.

    Two cells rather than one because a one-at-a-time generator produces the base
    and one arm, and dividing by the count is what makes the row a cost per cell
    rather than a cost of a run. The parameter it moves is the edge trim, which
    is the cheapest step in the chain, so what the row measures is the comparison
    and not the arm.
    """
    declared = {
        "name": "cost",
        "version": "1",
        "description": (
            "A design that exists to be timed. Not a preregistered design, not a result, "
            "and nothing here is reported."
        ),
        "generator": "one-at-a-time",
        "profile": str((ROOT / "profiles" / "published-chain.json").resolve()),
        "surface": {
            "rows": size,
            "columns": size,
            "pixel_spacing_um": 4.0,
            "striae_depth_um": 2.0,
            "striae_spacing_um": 40.0,
            "striae_angle_deg": 0.0,
            "form_depth_um": 12.0,
            "firing_pin_radius_um": 40.0,
            "firing_pin_depth_um": 25.0,
            "drag_mark_width_um": 30.0,
            "drag_mark_depth_um": 6.0,
            "drag_mark_position": 0.25,
            "noise_um": 0.2,
            "edge_dropout_fraction": 0.0,
        },
        "search": search,
        "rule": {
            "down_threshold": 6.0,
            "across_threshold": 6.0,
            "rotation_threshold_deg": 2.0,
            "correlation_threshold": 0.3,
            "consensus": "median",
            "translation_bin": None,
            "rotation_bin_deg": None,
        },
        "pairs": [{"name": "same-source", "kind": "matching", "seed": 20260809}],
        "vary": [{"parameter": "trim-edge.width", "values": [40.0, 120.0]}],
    }
    path = directory / "cost.json"
    path.write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    print(f"mode: {DETERMINISM.mode.value}, threads pinned to {DETERMINISM.threads}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"processor: {platform.processor() or 'not reported by the platform'}")
    print(f"numpy {numpy.__version__}, scipy {scipy.__version__}")
    print()
    print("search   field      correlations   cells   seconds   seconds per cell", flush=True)
    ranges = load_ranges(ROOT / "docs" / "ranges.json")
    for name, search in (("base", BASE), ("worst", WORST)):
        settings = RegistrationParameters(**search)  # type: ignore[arg-type]
        correlations = settings.grid * settings.grid * len(angles(settings))
        for size in SIZES[name]:
            with tempfile.TemporaryDirectory() as workspace:
                directory = Path(workspace)
                design = load(a_design(directory, size, search), REGISTRY, ranges)
                cells = len(design.cells())
                # One run rather than a best of several. What a sweep pays is
                # what a run takes on a machine doing this and nothing else, and
                # a minimum over repeats is a number nobody's sweep observes.
                start = time.perf_counter()
                run(design, REGISTRY, directory / "out", DETERMINISM, ENVIRONMENT, workers=1)
                elapsed = time.perf_counter() - start
            each = elapsed / cells
            print(
                f"{name:8} {size:4d}x{size:<4d} {correlations:12d} {cells:7d} "
                f"{elapsed:9.2f} {each:18.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
