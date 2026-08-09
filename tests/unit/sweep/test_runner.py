"""The runner runs a design, records every cell, and survives being interrupted.

The design under test is `small.json`, which is the reproduction chain on a
generated pair with two parameters moving. Both of them move the score on this
surface, deliberately: a runner that built the chain and then never applied the
assignment would produce a table of identical rows, and a test over a design
whose parameters changed nothing could not tell the two apart.

Everything here runs offline and headless, because the pairs are generated rather
than fetched. That is a property of the design shape rather than of this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gutachten.transforms  # noqa: F401  (importing registers the steps)
from gutachten.determinism import FAST_NOTE, REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import EnvironmentRecord, read
from gutachten.sweep.design import Design, load, load_ranges
from gutachten.sweep.runner import (
    CELLS,
    RESULTS,
    SUMMARY,
    Report,
    Row,
    SweepError,
    rerun_cell,
    run,
    surfaces,
)
from gutachten.transforms.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[3]
SMALL = Path(__file__).resolve().parent / "small.json"
PROFILE = ROOT / "profiles" / "published-chain.json"

REFERENCE = DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS)
FAST = DeterminismRecord(mode=RunMode.FAST, threads=None)

#: Held still, as the determinism fixture holds it still and for the same reason.
ENVIRONMENT = EnvironmentRecord(
    software_version="0.0.0", dependencies=(("numpy", "recorded-elsewhere"),)
)


@pytest.fixture(scope="module")
def design() -> Design:
    return load(SMALL, REGISTRY, load_ranges(ROOT / "docs" / "ranges.json"))


@pytest.fixture(scope="module")
def swept(design: Design, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Report]:
    """One run of the design, shared by the tests that only read what it wrote."""
    directory = tmp_path_factory.mktemp("swept")
    return directory, run(design, REGISTRY, directory, REFERENCE, ENVIRONMENT, workers=1)


def test_a_small_design_runs_end_to_end(design: Design, swept: tuple[Path, Report]) -> None:
    directory, report = swept
    assert report.computed == len(design.cells())
    assert report.reused == 0
    assert len(report.rows) == len(design.cells())
    assert (directory / RESULTS).exists()


def test_every_cell_leaves_a_manifest(design: Design, swept: tuple[Path, Report]) -> None:
    """One manifest per cell is the artefact, and the table is the summary of it.

    A table without them is a set of numbers nobody can check a single row of
    without re-running the whole design.
    """
    directory, _ = swept
    for cell in design.cells():
        manifest = read(directory / CELLS / f"{cell.identifier}.manifest.json")
        assert manifest.comparison is not None
        assert {record.role for record in manifest.inputs} == {"subject", "reference"}
        assert dict(manifest.comparison.parameters)["grid"] == cell.value("compare.register.grid")


def test_the_results_table_holds_one_line_per_cell(
    design: Design, swept: tuple[Path, Report]
) -> None:
    directory, report = swept
    lines = (directory / RESULTS).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(report.rows) + 1
    header = lines[0].split(",")
    assert header[:3] == ["cell", "pair", "same_source"]
    assert set(header) >= {varied.parameter for varied in design.varied}
    assert header[-3:] == ["congruent", "eligible", "discarded"]


def test_moving_either_parameter_alone_moves_the_score(
    design: Design, swept: tuple[Path, Report]
) -> None:
    """An assignment that never reached the pipeline is silent, so it is asserted.

    A runner that assembled the chain out of the profile and forgot the cell's
    own values would still run every cell and write every manifest, and the
    sensitivity report it fed would say the pipeline is flat. Each arm is checked
    separately: one parameter belongs to the chain and the other to the search,
    and they reach the run by two different routes.
    """
    _, report = swept
    scored = {
        row.assignment: (row.congruent, row.eligible)
        for row in report.rows
        if row.pair == "same-source"
    }
    base = tuple(
        sorted((varied.parameter, design.base(varied.parameter)) for varied in design.varied)
    )
    moved = [assignment for assignment in scored if assignment != base]
    assert len(moved) == len(design.varied)
    for assignment in moved:
        assert scored[assignment] != scored[base], (assignment, scored)


def test_a_resumed_run_reuses_what_is_there_and_prints_both_counts(
    design: Design, tmp_path: Path
) -> None:
    """Interrupting and restarting completes the remainder and reuses the rest.

    The interruption is modelled by deleting one cell's manifest, which is what
    an interrupted run leaves behind: the runner writes the row first and the
    manifest last, so the manifest is the completion marker.
    """
    first = run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=1)
    assert first.reused == 0

    interrupted = design.cells()[0].identifier
    (tmp_path / CELLS / f"{interrupted}.manifest.json").unlink()

    second = run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=1)
    assert (second.computed, second.reused) == (1, len(design.cells()) - 1)
    assert "cells computed: 1, reused:" in second.rendered()
    assert {row.cell for row in second.rows} == {row.cell for row in first.rows}


def test_a_truncated_manifest_is_not_a_completed_cell(design: Design, tmp_path: Path) -> None:
    """Recognition reads the manifest rather than looking for the file.

    A cell whose manifest was cut off mid-write parses as nothing, and reading it
    as complete would leave a row in the table that no manifest describes.
    """
    run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=1)
    truncated = design.cells()[1].identifier
    (tmp_path / CELLS / f"{truncated}.manifest.json").write_text("{ half a mani", "utf-8")

    resumed = run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=1)
    assert resumed.computed == 1


def test_a_cell_re_run_alone_from_its_manifest_reproduces_its_row(
    design: Design, swept: tuple[Path, Report]
) -> None:
    """The claim the whole recording exists to support.

    Nothing of the design reaches the re-run: the chain, every parameter and both
    comparison settings come off the manifest. The surfaces are handed in because
    a manifest names its inputs by hash and the cache those are looked up in is
    #41.
    """
    directory, report = swept
    cell = design.cells()[0]
    subject, reference = surfaces(design, cell.pair)
    row = rerun_cell(directory, cell.identifier, REGISTRY, subject, reference)
    assert row == next(found for found in report.rows if found.cell == cell.identifier)


def test_a_cell_re_run_against_another_input_is_refused(
    design: Design, swept: tuple[Path, Report]
) -> None:
    directory, _ = swept
    cell = design.cells()[0]
    subject, _ = surfaces(design, cell.pair)
    other, _ = surfaces(design, design.pairs[1])
    with pytest.raises(SweepError, match="names inputs"):
        rerun_cell(directory, cell.identifier, REGISTRY, subject, other)


def test_a_cell_whose_row_no_longer_matches_its_manifest_is_refused(
    design: Design, tmp_path: Path
) -> None:
    """The refusal that makes the reproduction claim worth anything.

    Without it the re-run would recompute a number and hand back a row nobody
    compared it against, which is the shape of a check that always passes.
    """
    run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=1)
    cell = design.cells()[0]
    path = tmp_path / CELLS / f"{cell.identifier}.row.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    recorded["congruent"] = recorded["congruent"] + 1
    path.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subject, reference = surfaces(design, cell.pair)
    with pytest.raises(SweepError, match="re-runs to"):
        rerun_cell(tmp_path, cell.identifier, REGISTRY, subject, reference)


def test_the_parallel_mode_marks_its_own_output(design: Design, tmp_path: Path) -> None:
    """Every cell of a parallel run carries the sentence, not only the summary.

    A manifest is read on its own, long after the run directory it arrived in has
    been copied somewhere else, so the mark has to be on the record rather than
    beside it.
    """
    report = run(design, REGISTRY, tmp_path, FAST, ENVIRONMENT, workers=3)
    assert report.workers == 3
    assert FAST_NOTE in report.rendered()

    summary = json.loads((tmp_path / SUMMARY).read_text(encoding="utf-8"))
    assert summary["workers"] == 3
    assert summary["determinism"]["reportable"] is False

    for cell in design.cells():
        manifest = read(tmp_path / CELLS / f"{cell.identifier}.manifest.json")
        assert manifest.determinism.mode is RunMode.FAST
        assert manifest.determinism.note == FAST_NOTE


def test_a_parallel_run_reaches_the_same_rows_as_a_reference_run(
    design: Design, swept: tuple[Path, Report], tmp_path: Path
) -> None:
    """What agrees is the rows, and the record still says fast.

    Whether every cell of a parallel run is bit identical to its reference twin
    on every machine is not measured anywhere here, so the mode stays fast and
    this asserts the counts rather than the arithmetic behind them.
    """
    _, reference = swept
    parallel = run(design, REGISTRY, tmp_path, FAST, ENVIRONMENT, workers=3)
    numbers = {(row.cell, row.congruent, row.eligible) for row in parallel.rows}
    assert numbers == {(row.cell, row.congruent, row.eligible) for row in reference.rows}


def test_more_than_one_worker_in_reference_mode_is_refused(design: Design, tmp_path: Path) -> None:
    """The mode is a claim about the arithmetic and parallelism unmakes it.

    Letting it through would put the reference note on a run nobody pinned, which
    is a negative disclosure turned into a positive one.
    """
    with pytest.raises(SweepError, match="reference mode"):
        run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=2)


def test_a_run_with_no_worker_is_refused(design: Design, tmp_path: Path) -> None:
    with pytest.raises(SweepError, match="at least one worker"):
        run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=0)


def test_a_cell_the_chain_refuses_stops_the_run_naming_the_cell(
    tmp_path: Path,
) -> None:
    """A parameter set no step will run is raised, not written in as a gap.

    Whether such a cell should instead be skipped and counted is the open
    question in #57, and it belongs to the design. What the runner may not do is
    decide it quietly by leaving a hole in the table.
    """
    declared = json.loads(SMALL.read_text(encoding="utf-8"))
    declared["profile"] = str(PROFILE)
    declared["name"] = "impossible"
    declared["vary"] = [
        {"parameter": "trim-edge.width", "values": [40.0, 400.0]},
        {"parameter": "compare.register.grid", "values": [3, 5]},
    ]
    path = tmp_path / "impossible.json"
    path.write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    design = load(path, REGISTRY, load_ranges(ROOT / "docs" / "ranges.json"))

    with pytest.raises(SweepError, match="was refused"):
        run(design, REGISTRY, tmp_path / "out", REFERENCE, ENVIRONMENT, workers=1)


def test_a_row_survives_being_written_and_read(swept: tuple[Path, Report]) -> None:
    """The row on disk is the row in memory, nulls and flags included."""
    directory, report = swept
    row = report.rows[0]
    written = json.loads((directory / CELLS / f"{row.cell}.row.json").read_text(encoding="utf-8"))
    assert Row.from_dict(written) == row


def test_a_consensus_swap_that_leaves_the_bins_behind_is_refused(tmp_path: Path) -> None:
    """A coupling between two parameters the design has to respect, met at the cell.

    The histogram rule needs both bin widths and the median rule refuses them
    stated, so a design moving the consensus alone reaches a cell no rule will
    run. The runner names the cell rather than deciding for the design what such
    a cell means, which is #57's open question.
    """
    declared = json.loads(SMALL.read_text(encoding="utf-8"))
    declared["profile"] = str(PROFILE)
    declared["name"] = "swapped"
    declared["rule"] |= {
        "consensus": "histogram-mode",
        "translation_bin": 2.0,
        "rotation_bin_deg": 1.0,
    }
    declared["vary"] = [
        {"parameter": "compare.cmc.consensus"},
        {"parameter": "compare.cmc.correlation_threshold", "values": [0.0, 0.3]},
    ]
    path = tmp_path / "swapped.json"
    path.write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    design = load(path, REGISTRY, load_ranges(ROOT / "docs" / "ranges.json"))

    assert design.base("compare.cmc.consensus") == "histogram-mode"
    assert design.base("compare.cmc.correlation_threshold") == 0.3
    with pytest.raises(SweepError, match="was refused"):
        run(design, REGISTRY, tmp_path / "out", REFERENCE, ENVIRONMENT, workers=1)


def test_a_manifest_recording_no_comparison_cannot_be_re_run(
    design: Design, tmp_path: Path
) -> None:
    """The settings that produced the number are what a re-run rebuilds from.

    A manifest without them describes a preprocessing run, and re-running it
    would produce a score under settings nobody recorded.
    """
    run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=1)
    cell = design.cells()[0]
    path = tmp_path / CELLS / f"{cell.identifier}.manifest.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    recorded["comparison"] = None
    path.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")

    subject, reference = surfaces(design, cell.pair)
    with pytest.raises(SweepError, match="records no comparison"):
        rerun_cell(tmp_path, cell.identifier, REGISTRY, subject, reference)


def test_a_manifest_naming_no_subject_cannot_be_re_run(design: Design, tmp_path: Path) -> None:
    """Which of the two surfaces was the subject is not recoverable from a hash.

    The registration divides the subject into cells and turns the reference, so
    the two are not interchangeable and a manifest that lost the roles cannot say
    which was which.
    """
    run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=1)
    cell = design.cells()[0]
    path = tmp_path / CELLS / f"{cell.identifier}.manifest.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    recorded["inputs"][0]["role"] = "scan"
    path.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")

    subject, reference = surfaces(design, cell.pair)
    with pytest.raises(SweepError, match="no input in the role"):
        rerun_cell(tmp_path, cell.identifier, REGISTRY, subject, reference)


EXCLUSION = Path(__file__).resolve().parent / "exclusion.json"


def test_the_drag_mark_exclusion_is_enumerated_by_the_sweep(tmp_path: Path) -> None:
    """The clause #57 stayed open on, over the chain that actually masks.

    Excluding the drag mark is what the published chains do and it is a choice
    rather than a fact, so the whole point of the step is that not excluding it
    is a configuration a sweep can visit rather than a code change. Nothing
    enumerated anything until now, and a setting nothing enumerates is a setting
    the sensitivity report cannot say what costs.

    That the two configurations reach different scores is asserted because a
    runner that recorded the setting and never applied it would pass every other
    check here. Which of them scores higher is not a result: it is one generated
    pair under one configuration, and the design is a fixture.
    """
    design = load(EXCLUSION, REGISTRY, load_ranges(ROOT / "docs" / "ranges.json"))
    assert [varied.parameter for varied in design.varied] == ["mask-marks.exclude_drag"]

    report = run(design, REGISTRY, tmp_path, REFERENCE, ENVIRONMENT, workers=1)
    visited = {row.assignment[0][1]: row for row in report.rows}
    assert set(visited) == {True, False}

    for value, row in visited.items():
        manifest = read(tmp_path / CELLS / f"{row.cell}.manifest.json")
        recorded = {step.identifier: dict(step.parameters) for step in manifest.steps}
        assert recorded["mask-marks"]["exclude_drag"] is value

    assert visited[True].congruent != visited[False].congruent
