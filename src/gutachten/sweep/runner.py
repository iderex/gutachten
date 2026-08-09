"""Running a design: one comparison per cell, recorded so any one can be re-run alone.

The results table is not the artefact this produces. The artefact is one manifest
per cell beside it, because a table of numbers is a claim and a manifest is what
lets somebody else check one row of it without re-running the other nine hundred.

## Resumability, and what counts as a completed cell

A full design runs for hours or days. A runner that starts over after an
interruption is a runner that gets replaced by a shell loop nobody records, so a
cell already on disk is recognised and skipped, and the run says how many it
computed and how many it reused. A resumed run is never printed as a fresh one.

A cell is complete when its manifest is on disk and reads back. The row is
written first and the manifest last, so the manifest is the completion marker and
a cell interrupted between the two writes is recomputed rather than half read. A
truncated manifest is not a completed cell either, which is why it is read rather
than stat-ed.

## The two modes, and why the parallel one is marked

The reference mode is single threaded and is what a reported number comes from.
Parallelism is available and is not it: several cells running at once share one
numerical backend whose thread count this process can no longer pin, because the
pin has to be set before the backend is imported. So a run with more than one
worker is refused unless it is declared as a fast run, and every manifest it
writes then carries the sentence saying it may not be used for anything reported.
The mark is the existing determinism record rather than a second mechanism.

Whether a parallel run would in fact agree with a reference run cell for cell is
not measured here, and the record says fast rather than claiming it does.

## What a refused cell does

It stops the run. A parameter set the step refuses is raised where it happens,
with the cell named, rather than written into the table as a missing score.
Whether a cell that cannot run should instead be skipped and counted is the open
question in [#57](https://github.com/iderex/gutachten/issues/57), and it belongs
to the design rather than to the runner, so nothing here decides it.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gutachten.compare.cmc import CmcParameters, ConsensusRule, score_pair
from gutachten.compare.register import RegistrationParameters
from gutachten.determinism import DeterminismRecord, RunMode
from gutachten.manifest import (
    EnvironmentRecord,
    FileRecord,
    RunManifest,
    read,
    record_run,
    resolve,
    surface_digest,
)
from gutachten.surface import AxisOrientation, LengthUnit, ParameterValue, Surface
from gutachten.sweep.design import Cell, Design, Pair
from gutachten.synth import SyntheticSurface, matching_pair, non_matching_pair
from gutachten.transforms.pipeline import Step, run_chain
from gutachten.transforms.registry import Registry

__all__ = [
    "CELLS",
    "RESULTS",
    "SUMMARY",
    "Report",
    "Row",
    "SweepError",
    "rerun_cell",
    "run",
    "surfaces",
]

#: Where a cell's two files live, and what the run writes beside them.
CELLS = "cells"
RESULTS = "results.csv"
SUMMARY = "summary.json"


class SweepError(Exception):
    """A sweep that could not run as asked."""


@dataclass(frozen=True)
class Row:
    """One line of the results table: which cell, on which pair, and what came out."""

    cell: str
    pair: str
    same_source: bool
    assignment: tuple[tuple[str, ParameterValue], ...]
    congruent: int
    eligible: int
    discarded: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "pair": self.pair,
            "same_source": self.same_source,
            "assignment": dict(self.assignment),
            "congruent": self.congruent,
            "eligible": self.eligible,
            "discarded": self.discarded,
        }

    @staticmethod
    def from_dict(data: Any) -> Row:
        return Row(
            cell=data["cell"],
            pair=data["pair"],
            same_source=data["same_source"],
            assignment=tuple(sorted(data["assignment"].items())),
            congruent=data["congruent"],
            eligible=data["eligible"],
            discarded=data["discarded"],
        )


@dataclass(frozen=True)
class Report:
    """What a run of the design produced, and how it was produced."""

    design: str
    version: str
    generator: str
    workers: int
    determinism: DeterminismRecord
    computed: int
    reused: int
    rows: tuple[Row, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "version": self.version,
            "generator": self.generator,
            "workers": self.workers,
            "determinism": self.determinism.to_dict(),
            "cells": {
                "computed": self.computed,
                "reused": self.reused,
                "total": self.computed + self.reused,
            },
        }

    def rendered(self) -> str:
        """What the run prints, including the mode it ran in.

        The two counts are printed whether or not either is zero, so a fresh run
        and a resumed one are told apart by reading rather than by remembering
        which was which.
        """
        return "\n".join(
            [
                f"design: {self.design} version {self.version}, {self.generator}",
                f"cells computed: {self.computed}, reused: {self.reused}",
                f"workers: {self.workers}, mode: {self.determinism.mode.value}",
                self.determinism.note,
            ]
        )


def _as_surface(generated: SyntheticSurface) -> Surface:
    return Surface(
        heights=generated.heights_um,
        spacing_y=generated.parameters.pixel_spacing_um,
        spacing_x=generated.parameters.pixel_spacing_um,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source=f"synthetic-source-{generated.source_id}",
    )


def surfaces(design: Design, pair: Pair) -> tuple[Surface, Surface]:
    """The two surfaces of a declared pair, made rather than fetched.

    The ground truth is the construction: a matching pair is two firings of one
    source and a non-matching pair is two sources. Nothing is read from disk and
    nothing is fetched, so a sweep runs under the gate's offline condition.
    """
    parameters = dataclasses.replace(design.surface, seed=pair.seed)
    make = matching_pair if pair.same_source else non_matching_pair
    first, second = make(parameters)
    return _as_surface(first), _as_surface(second)


def _chain(design: Design, cell: Cell) -> list[Step]:
    """The profile's chain with this cell's assignment applied to it."""
    assignment = dict(cell.assignment)
    steps: list[Step] = []
    for step in design.profile.steps:
        moved: dict[str, Any] = {
            parameter.rsplit(".", 1)[1]: value
            for parameter, value in assignment.items()
            if parameter.rsplit(".", 1)[0] == step.identifier
        }
        # The record is a protocol rather than a named class here, so the
        # rebuild goes through a plain value. What refuses a wrong field is the
        # dataclass itself, at the same moment it always did.
        record: Any = step.parameters
        parameters = dataclasses.replace(record, **moved) if moved else step.parameters
        steps.append(Step(identifier=step.identifier, parameters=parameters))
    return steps


def _comparison(design: Design, cell: Cell) -> tuple[RegistrationParameters, CmcParameters]:
    """The search and the rule with this cell's assignment applied to them."""
    assignment = dict(cell.assignment)
    search_moved: dict[str, Any] = {
        parameter.rsplit(".", 1)[1]: value
        for parameter, value in assignment.items()
        if parameter.startswith("compare.register.")
    }
    rule_moved: dict[str, Any] = {
        parameter.rsplit(".", 1)[1]: value
        for parameter, value in assignment.items()
        if parameter.startswith("compare.cmc.")
    }
    if "consensus" in rule_moved:
        rule_moved["consensus"] = ConsensusRule(rule_moved["consensus"])
    return (
        dataclasses.replace(design.search, **search_moved),
        dataclasses.replace(design.rule, **rule_moved),
    )


def _paths(directory: Path, identifier: str) -> tuple[Path, Path]:
    cells = directory / CELLS
    return cells / f"{identifier}.row.json", cells / f"{identifier}.manifest.json"


def _completed(directory: Path, identifier: str) -> Row | None:
    """The recorded row where the cell is complete, and nothing where it is not."""
    row_path, manifest_path = _paths(directory, identifier)
    if not (row_path.exists() and manifest_path.exists()):
        return None
    try:
        read(manifest_path)
        return Row.from_dict(json.loads(row_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # A half-written pair of files is not a completed cell. Recomputing costs
        # one cell; reading it as complete costs a row nobody can reproduce.
        return None


def _compute(
    design: Design,
    cell: Cell,
    registry: Registry,
    directory: Path,
    determinism: DeterminismRecord,
    environment: EnvironmentRecord,
) -> Row:
    """Run one cell and write its two files, the manifest last."""
    subject, reference = surfaces(design, cell.pair)
    chain = _chain(design, cell)
    search, rule = _comparison(design, cell)

    try:
        processed_subject, manifest = record_run(
            role="subject",
            surface=subject,
            profile=design.profile.record(),
            chain=chain,
            registry=registry,
            seed=cell.pair.seed,
            determinism=determinism,
            environment=environment,
        )
        processed_reference = run_chain(chain, registry, reference)
        score, comparison = score_pair(processed_subject, processed_reference, search, rule)
    except (ValueError, TypeError) as refused:
        raise SweepError(
            f"cell {cell.identifier} on pair {cell.pair.name!r} was refused: {refused}"
        ) from refused

    manifest = dataclasses.replace(
        manifest,
        inputs=(
            *manifest.inputs,
            FileRecord(role="reference", sha256=surface_digest(reference)),
        ),
        outputs=(
            FileRecord(role="subject-preprocessed", sha256=surface_digest(processed_subject)),
            FileRecord(role="reference-preprocessed", sha256=surface_digest(processed_reference)),
        ),
        comparison=comparison,
    )

    row = Row(
        cell=cell.identifier,
        pair=cell.pair.name,
        same_source=cell.pair.same_source,
        assignment=cell.assignment,
        congruent=score.congruent,
        eligible=score.eligible,
        discarded=score.discarded,
    )
    row_path, manifest_path = _paths(directory, cell.identifier)
    row_path.write_text(json.dumps(row.to_dict(), indent=2, sort_keys=True) + "\n", "utf-8")
    manifest.write(manifest_path)
    return row


def _write_table(directory: Path, design: Design, rows: tuple[Row, ...]) -> None:
    """The results table, one line per cell, in cell order.

    Every value is written as JSON so a null, a flag and a word come back as what
    they were rather than as three spellings of a string. The columns are the
    varied parameters in name order, so two runs of one design write the same
    header whatever order the cells finished in.
    """
    columns = [varied.parameter for varied in sorted(design.varied, key=lambda v: v.parameter)]
    with (directory / RESULTS).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["cell", "pair", "same_source", *columns, "congruent", "eligible", "discarded"]
        )
        for row in rows:
            values = dict(row.assignment)
            writer.writerow(
                [
                    row.cell,
                    row.pair,
                    json.dumps(row.same_source),
                    *[json.dumps(values[column]) for column in columns],
                    row.congruent,
                    row.eligible,
                    row.discarded,
                ]
            )


def run(
    design: Design,
    registry: Registry,
    directory: Path,
    determinism: DeterminismRecord,
    environment: EnvironmentRecord,
    workers: int,
) -> Report:
    """Run every cell of ``design`` into ``directory``, reusing what is already there."""
    if workers < 1:
        raise SweepError(f"a sweep needs at least one worker, got {workers}")
    if workers > 1 and determinism.mode is RunMode.REFERENCE:
        raise SweepError(
            f"{workers} workers were asked for in reference mode. Several cells at once "
            "share a numerical backend this process can no longer pin, because the pin "
            "has to be set before the backend is imported. Declare the run as fast, so "
            "every manifest it writes carries the sentence saying it may not be used for "
            "anything reported."
        )

    (directory / CELLS).mkdir(parents=True, exist_ok=True)
    cells = design.cells()
    reused = {cell.identifier: _completed(directory, cell.identifier) for cell in cells}
    outstanding = [cell for cell in cells if reused[cell.identifier] is None]

    def one(cell: Cell) -> Row:
        return _compute(design, cell, registry, directory, determinism, environment)

    if workers == 1:
        computed = {cell.identifier: one(cell) for cell in outstanding}
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            computed = {
                cell.identifier: row
                for cell, row in zip(outstanding, pool.map(one, outstanding), strict=True)
            }

    def recorded(cell: Cell) -> Row:
        found = computed.get(cell.identifier)
        return found if found is not None else _required(reused[cell.identifier], cell.identifier)

    rows = tuple(recorded(cell) for cell in cells)
    report = Report(
        design=design.name,
        version=design.version,
        generator=design.generator,
        workers=workers,
        determinism=determinism,
        computed=len(computed),
        reused=len(cells) - len(computed),
        rows=rows,
    )
    _write_table(directory, design, rows)
    (directory / SUMMARY).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _required(row: Row | None, identifier: str) -> Row:
    if row is None:  # pragma: no cover - the partition above admits no other case
        raise SweepError(f"cell {identifier} was neither computed nor reused")
    return row


def rerun_cell(directory: Path, identifier: str, registry: Registry, *surfaces: Surface) -> Row:
    """Re-run one cell from its own manifest and refuse a result that is not its row.

    The computation reads the manifest and nothing else: the chain, every
    parameter and the comparison settings come off the record of what ran, not off
    the design that generated it. The row is read only to be compared against, and
    the recorded row comes back so a caller has the thing that was checked.

    The surfaces are handed in rather than fetched, as ``manifest.rerun`` does.
    A manifest names its inputs by hash and the content addressed cache those
    hashes are looked up in is #41.
    """
    row_path, manifest_path = _paths(directory, identifier)
    manifest = read(manifest_path)
    recorded = Row.from_dict(json.loads(row_path.read_text(encoding="utf-8")))

    named = {record.sha256 for record in manifest.inputs}
    given = {surface_digest(surface) for surface in surfaces}
    if given != named:
        raise SweepError(
            f"cell {identifier} names inputs {sorted(named)} and was handed {sorted(given)}. "
            "Re-running a recorded cell against another input produces a number under a "
            "label saying it came from this one."
        )

    chain = resolve(manifest, registry)
    processed = {
        surface_digest(surface): run_chain(chain, registry, surface) for surface in surfaces
    }
    subject = processed[_role(manifest, "subject")]
    reference = processed[_role(manifest, "reference")]
    search, rule = _settings(manifest)
    score, _ = score_pair(subject, reference, search, rule)

    produced = (score.congruent, score.eligible, score.discarded)
    expected = (recorded.congruent, recorded.eligible, recorded.discarded)
    if produced != expected:
        raise SweepError(
            f"cell {identifier} re-runs to {produced} and its row records {expected}. The "
            "chain, the parameters and the inputs all matched, so what moved is the code "
            "behind a step or a rule whose version did not."
        )
    return recorded


def _role(manifest: RunManifest, role: str) -> str:
    for record in manifest.inputs:
        if record.role == role:
            return record.sha256
    raise SweepError(f"the manifest names no input in the role {role!r}")


def _settings(manifest: RunManifest) -> tuple[RegistrationParameters, CmcParameters]:
    """The search and the rule, split out of the one record that carries both.

    Split by the fields each record declares rather than by a list written here,
    so a parameter added to either is carried without an edit in this file.
    """
    if manifest.comparison is None:
        raise SweepError("the manifest records no comparison, so there is no score to re-run")
    recorded = dict(manifest.comparison.parameters)
    search_fields = {field.name for field in dataclasses.fields(RegistrationParameters)}
    rule_fields = {field.name for field in dataclasses.fields(CmcParameters)}
    missing = (search_fields | rule_fields) - set(recorded)
    if missing:
        raise SweepError(
            f"the manifest's comparison record leaves {sorted(missing)} unrecorded, so the "
            "settings that produced the number cannot be rebuilt from it"
        )
    rule: dict[str, Any] = {name: recorded[name] for name in rule_fields}
    rule["consensus"] = ConsensusRule(rule["consensus"])
    search: dict[str, Any] = {name: recorded[name] for name in search_fields}
    return RegistrationParameters(**search), CmcParameters(**rule)
