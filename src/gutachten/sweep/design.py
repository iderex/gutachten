"""What a sweep is asked to do, read from a file and refused where it could not run.

A design says which parameters move, over which values, around which base, on
which pairs. It is a file rather than arguments to a function because the whole
argument for this project is that a configuration nobody recorded is a
configuration nobody can check, and that applies to the sweep at least as much as
to the chain it is sweeping.

## Why the ranges are read rather than restated

`docs/ranges.json` declares a plausible range for every parameter a sweep can
reach, with the source of each bound, and issue #81 asks that a parameter with no
declared range cannot be swept. Here is where that stops being a rule about a
document. A design naming a parameter the ranges do not declare is refused, and a
design naming a value outside the declared range is refused, so the ranges are
the axis the design is drawn over rather than a file beside it.

The path to the ranges is handed in rather than found. `docs/` is not carried
into a built distribution, and a module reaching for a path relative to itself
would work in a clone and fail for the operator this project is aimed at.

## What a cell is

One comparison: one assignment of every varied parameter, on one declared pair.
The assignment is complete rather than partial even in a one-at-a-time design, so
every row of the results table says what every varied parameter was, and the cell
sitting at the base of a one-at-a-time design is the same cell whichever
parameter's arm produced it. That is what makes the deduplication below correct
rather than convenient.

The identifier is a digest of the assignment, the pair, and the design's name and
version. It does not depend on the order the cells were generated in, so an
interrupted run recognises what it already did.

## What is not decided here

Which design is preregistered, which parameters it moves and how large the sample
is, is [#79](https://github.com/iderex/gutachten/issues/79). The low discrepancy
sampling the global analysis needs is
[#87](https://github.com/iderex/gutachten/issues/87), and neither generator here
is it: `one-at-a-time` reports a local slope at the base, which is systematically
the flattest place in the space, and `full-factorial` covers a joint space at a
cost that grows as a power of the number of parameters. Both are here because a
runner has to run something, and a report quoting either says which it was.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from gutachten.compare.cmc import CmcParameters, ConsensusRule
from gutachten.compare.register import RegistrationParameters
from gutachten.profile import Profile
from gutachten.profile import load as load_profile
from gutachten.surface import ParameterValue
from gutachten.synth import SurfaceParameters
from gutachten.transforms.registry import Registry

__all__ = [
    "COMPARISON_OWNERS",
    "GENERATORS",
    "PAIR_KINDS",
    "Bound",
    "Cell",
    "Design",
    "DesignError",
    "Pair",
    "Varied",
    "load",
    "load_ranges",
]

#: The two owners a comparison parameter is keyed under in the ranges. The same
#: strings the ranges use, rather than a second spelling of them.
COMPARISON_OWNERS = ("compare.register", "compare.cmc")

#: The generators this runner offers. Neither is the design the global analysis
#: needs; see the module docstring.
GENERATORS = ("one-at-a-time", "full-factorial")

#: What a declared pair is. The generator makes both, so a sweep needs no data
#: and no network, and the ground truth of each pair is a construction rather
#: than a label somebody attached to a file.
PAIR_KINDS = ("matching", "non-matching")


class DesignError(Exception):
    """A design that could not run as written."""


@dataclass(frozen=True)
class Bound:
    """One parameter's declared range, as the design reads it.

    ``values`` is the admissible set for a parameter that takes a word or a flag
    and is empty for one that takes a number. ``nullable`` says whether the
    declaration carries a null, which is a state a design may visit and which no
    interval contains.
    """

    parameter: str
    kind: str
    lower: float | int | None
    upper: float | int | None
    values: tuple[ParameterValue, ...]
    nullable: bool

    def admits(self, value: ParameterValue) -> bool:
        if value is None:
            return self.nullable
        if self.kind == "set":
            return value in self.values
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        assert self.lower is not None and self.upper is not None
        return self.lower <= value <= self.upper

    def spread(self, points: int) -> tuple[ParameterValue, ...]:
        """``points`` values across the range, ends included.

        A whole number parameter is stepped in whole numbers, because a design
        stepping between two fractions produces cells the parameter record
        refuses, and the refusal would land in the middle of a sweep rather than
        when the design was read.
        """
        if self.kind == "set":
            raise DesignError(
                f"{self.parameter} takes a word or a flag and a count of {points} points "
                "was asked for. A set has no interval to step along; name the values."
            )
        assert self.lower is not None and self.upper is not None
        whole = isinstance(self.lower, int) and isinstance(self.upper, int)
        step = (self.upper - self.lower) / (points - 1)
        drawn = [self.lower + step * index for index in range(points)]
        if whole:
            rounded = [round(value) for value in drawn]
            if len(set(rounded)) != len(rounded):
                raise DesignError(
                    f"{self.parameter} is a whole number between {self.lower} and "
                    f"{self.upper}, and {points} points across it repeat a value. A cell "
                    "visited twice is a cell counted twice in every summary taken over "
                    "the table."
                )
            return tuple(rounded)
        return tuple(float(value) for value in drawn)


@dataclass(frozen=True)
class Varied:
    """One parameter and the values it takes across the design."""

    parameter: str
    values: tuple[ParameterValue, ...]


@dataclass(frozen=True)
class Pair:
    """One declared pair of surfaces, and what it is by construction."""

    name: str
    kind: str
    seed: int

    @property
    def same_source(self) -> bool:
        return self.kind == "matching"


@dataclass(frozen=True)
class Cell:
    """One comparison of the design: an assignment and a pair."""

    identifier: str
    pair: Pair
    assignment: tuple[tuple[str, ParameterValue], ...]

    def value(self, parameter: str) -> ParameterValue:
        return dict(self.assignment)[parameter]


@dataclass(frozen=True)
class Design:
    """A sweep as it was written down."""

    name: str
    version: str
    description: str
    generator: str
    profile: Profile
    surface: SurfaceParameters
    search: RegistrationParameters
    rule: CmcParameters
    varied: tuple[Varied, ...]
    pairs: tuple[Pair, ...]

    def base(self, parameter: str) -> ParameterValue:
        """What ``parameter`` is held at where the design is not moving it."""
        owner, field = parameter.rsplit(".", 1)
        if owner == "compare.register":
            return _plain(getattr(self.search, field))
        if owner == "compare.cmc":
            return _plain(getattr(self.rule, field))
        for step in self.profile.steps:
            if step.identifier == owner:
                return _plain(getattr(step.parameters, field))
        raise DesignError(
            f"{parameter} names {owner!r} and the profile {self.profile.name!r} does not "
            "run it. A parameter of a step the chain never applies cannot be swept, and a "
            "design that asked would report a flat curve for a step that never ran."
        )

    def cells(self) -> tuple[Cell, ...]:
        """Every cell of the design, deduplicated, in identifier order."""
        assignments = (
            _one_at_a_time(self) if self.generator == "one-at-a-time" else _factorial(self)
        )
        found: dict[str, Cell] = {}
        for assignment in assignments:
            for pair in self.pairs:
                identifier = _identify(self, pair, assignment)
                found.setdefault(identifier, Cell(identifier, pair, assignment))
        return tuple(found[key] for key in sorted(found))


def _plain(value: object) -> ParameterValue:
    """A parameter as a manifest and a table carry it, never as an enum member."""
    if isinstance(value, ConsensusRule):
        return value.value
    if value is None or isinstance(value, str | bool | int | float):
        return value
    raise DesignError(f"{value!r} is not a value a design can record")


def _one_at_a_time(design: Design) -> list[tuple[tuple[str, ParameterValue], ...]]:
    """Each parameter moved alone, the rest held at the base.

    Every assignment names every varied parameter, so the base cell is produced
    once by each arm and deduplicated to one rather than appearing as many
    near-identical rows.
    """
    base = {varied.parameter: design.base(varied.parameter) for varied in design.varied}
    found = [tuple(sorted(base.items()))]
    for varied in design.varied:
        for value in varied.values:
            moved = dict(base)
            moved[varied.parameter] = value
            found.append(tuple(sorted(moved.items())))
    return found


def _factorial(design: Design) -> list[tuple[tuple[str, ParameterValue], ...]]:
    """Every combination of every varied parameter's values."""
    names = [varied.parameter for varied in design.varied]
    return [
        tuple(sorted(zip(names, chosen, strict=True)))
        for chosen in product(*[varied.values for varied in design.varied])
    ]


def _identify(
    design: Design, pair: Pair, assignment: tuple[tuple[str, ParameterValue], ...]
) -> str:
    """A digest of everything that decides the cell, and of nothing else.

    The design's name and version are in it so that two designs writing into one
    directory cannot claim each other's completed cells, which is the failure a
    resumable runner makes reachable.
    """
    material = json.dumps(
        {
            "design": design.name,
            "version": design.version,
            "pair": [pair.name, pair.kind, pair.seed],
            "assignment": list(assignment),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def load_ranges(path: Path) -> dict[str, Bound]:
    """The declared ranges, as the design reads them."""
    parsed: Any = json.loads(path.read_text(encoding="utf-8"))
    found: dict[str, Bound] = {}
    for parameter, entry in parsed["parameters"].items():
        kind = entry["kind"]
        found[parameter] = Bound(
            parameter=parameter,
            kind=kind,
            lower=entry["lower"]["value"] if kind == "interval" else None,
            upper=entry["upper"]["value"] if kind == "interval" else None,
            values=tuple(item["value"] for item in entry.get("values", ())),
            nullable="null" in entry,
        )
    return found


def _pairs(declared: Any) -> tuple[Pair, ...]:
    found: list[Pair] = []
    for entry in declared:
        if entry["kind"] not in PAIR_KINDS:
            raise DesignError(
                f"pair {entry['name']!r} is a {entry['kind']!r} pair and the kinds are "
                f"{list(PAIR_KINDS)}. What a pair is by construction is what the report "
                "reads its ground truth off."
            )
        found.append(Pair(name=entry["name"], kind=entry["kind"], seed=int(entry["seed"])))
    if not found:
        raise DesignError("a design declares no pair, so there is nothing to compare")
    names = [pair.name for pair in found]
    if len(set(names)) != len(names):
        raise DesignError(
            f"two pairs are declared as {sorted(names)}. A name appearing twice makes a "
            "row of the results table ambiguous about which pair it came from."
        )
    return tuple(found)


def _surface(declared: Any) -> SurfaceParameters:
    """The generator's settings, every one of them stated.

    Every field except the seed, which each declared pair carries its own. A
    seed in this block would be a number the run recorded and never used.
    """
    fields = {field.name for field in dataclasses.fields(SurfaceParameters)} - {"seed"}
    given = set(declared)
    if given != fields:
        raise DesignError(
            f"the design states {sorted(given)} for the surface and the generator takes "
            f"{sorted(fields)}, the seed excepted because each pair states its own. A "
            "surface parameter the design does not state is one nobody chose for the "
            "sweep, and it would sit at whatever the generator's own default happens to be."
        )
    return SurfaceParameters(**declared)


def _varied(declared: Any, ranges: dict[str, Bound]) -> tuple[Varied, ...]:
    found: list[Varied] = []
    for entry in declared:
        parameter = entry["parameter"]
        if parameter not in ranges:
            raise DesignError(
                f"{parameter} has no declared range. A parameter with no declared range "
                "cannot be swept: the sweep would move it over bounds chosen inside the "
                "change that produced the numbers, which is the thing docs/ranges.md "
                "exists against."
            )
        bound = ranges[parameter]
        if "values" in entry:
            values = tuple(entry["values"])
        elif "points" in entry:
            values = bound.spread(int(entry["points"]))
        elif bound.kind == "set":
            values = bound.values
        else:
            raise DesignError(
                f"{parameter} takes a number and the design says neither which values to "
                "visit nor how many points to take across its range."
            )
        outside = [value for value in values if not bound.admits(value)]
        if outside:
            raise DesignError(
                f"{parameter} is asked to visit {outside}, which its declared range does "
                "not admit. A sweep reaching outside the range it was drawn over produces "
                "an index along an axis the report does not describe."
            )
        if len(values) < 2:
            raise DesignError(
                f"{parameter} is declared as varied over {list(values)}. A parameter held "
                "at one value has not been varied, and a report naming it as swept would "
                "say the space was covered where it was not."
            )
        if len({repr(value) for value in values}) != len(values):
            raise DesignError(f"{parameter} is asked to visit a value twice: {list(values)}")
        found.append(Varied(parameter=parameter, values=values))
    if not found:
        raise DesignError(
            "a design varies nothing, so every cell is the same comparison under a different name"
        )
    names = [varied.parameter for varied in found]
    if len(set(names)) != len(names):
        raise DesignError(f"a parameter is declared varied twice: {sorted(names)}")
    return tuple(found)


def load(path: Path, registry: Registry, ranges: dict[str, Bound]) -> Design:
    """Read a design, refusing one that could not run as written.

    The profile it names is resolved relative to the design file, so a design and
    the chain it sweeps around move together rather than depending on where the
    reader's working directory happened to be.
    """
    try:
        parsed: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as broken:
        raise DesignError(f"{path.name} is not readable as JSON: {broken}") from broken

    if parsed.get("generator") not in GENERATORS:
        raise DesignError(
            f"{path.name} asks for the {parsed.get('generator')!r} generator and this "
            f"runner offers {list(GENERATORS)}."
        )

    design = Design(
        name=parsed["name"],
        version=str(parsed["version"]),
        description=parsed["description"],
        generator=parsed["generator"],
        profile=load_profile((path.parent / parsed["profile"]).resolve(), registry),
        surface=_surface(parsed["surface"]),
        search=RegistrationParameters(**parsed["search"]),
        rule=CmcParameters(
            **{
                **parsed["rule"],
                "consensus": ConsensusRule(parsed["rule"]["consensus"]),
            }
        ),
        varied=_varied(parsed["vary"], ranges),
        pairs=_pairs(parsed["pairs"]),
    )
    if design.name != path.stem:
        raise DesignError(
            f"{path.name} records the name {design.name!r}. A design is selected by its "
            "file name and found by its recorded one, and the results directory is named "
            "after the second."
        )
    # Resolving every base now rather than at the first cell, so a design naming a
    # parameter of a step the profile does not run is refused when it is read.
    for varied in design.varied:
        design.base(varied.parameter)
    return design
