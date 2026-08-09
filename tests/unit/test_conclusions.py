"""The refusal to state a match, an identification or an exclusion, proven.

Each half of the check is shown to bite for the reason it names, by a pair that
differs in one thing: a template with a conclusion sentence and the same
template without it, a quotation with its source named and the same quotation
without, a record with a verdict boolean and the same record without.

The near misses are the point. A method name one word short of the declared
name, a mark with nothing after the colon, a count over a labelled set written
without saying which set. Each of those is a plausible thing to write and each
of them lands on the wrong side of a check that was built carelessly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest

from gutachten.compare.cmc import METHOD, METHOD_HIGH
from gutachten.conclusions import (
    NAMES,
    PLAIN_WORDS,
    REFUSED_STEMS,
    ConclusionField,
    conclusion_fields,
    conclusion_fields_in,
    conclusion_words,
)
from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import (
    ComparisonRecord,
    EnvironmentRecord,
    FileRecord,
    ProfileRecord,
    RunManifest,
    StepRecord,
)
from gutachten.transforms.registry import REGISTRY

# The template a run would render, with one slot. The pair either side of the
# slot is what shows the check is doing the work rather than the surrounding
# prose happening to be clean.
A_REPORT_TEMPLATE = """\
Comparison report

Score: {score} congruent cells of {eligible} eligible.
Propositions: the two cases were fired by the same firearm, against the two
cases were fired by different firearms from the relevant population.
Log likelihood ratio: {log_lr} (interval {low} to {high}).
{sentence}
"""

A_CONCLUSION_SENTENCE = "The two cases are a match."


def rendered(sentence: str = "") -> str:
    """The template with its slots filled, which is what an operator would read."""
    return A_REPORT_TEMPLATE.format(
        score="18", eligible="42", log_lr="3.1", low="2.4", high="3.8", sentence=sentence
    )


def test_a_template_that_states_no_conclusion_passes() -> None:
    assert conclusion_words(rendered(), source="report") == []


def test_a_conclusion_sentence_added_to_the_template_is_refused() -> None:
    found = conclusion_words(rendered(A_CONCLUSION_SENTENCE), source="report")

    assert [item.word for item in found] == ["match"]
    assert found[0].source == "report"
    assert "#101" in str(found[0])


def test_every_stem_is_reached_by_a_word_somebody_would_actually_write() -> None:
    # One sentence per stem, so a stem removed from the vocabulary reds this
    # rather than passing on the strength of the other five.
    written = {
        "conclu": "We conclude the two cases share a source.",
        "exclud": "The second firearm is excluded as a source.",
        "exclus": "This is an exclusion.",
        "identif": "The comparison supports an identification.",
        "inconclu": "The comparison is inconclusive.",
        "match": "The two cases are a match.",
    }

    assert sorted(written) == sorted(REFUSED_STEMS)
    for stem, sentence in written.items():
        found = conclusion_words(sentence)
        assert [item.word.lower().startswith(stem) for item in found] == [True], stem


def test_a_word_that_merely_starts_the_same_way_is_not_refused() -> None:
    # 'concentration' and 'exclave' start inside the same letters and neither is
    # a conclusion. A vocabulary written as bare substrings would take both.
    assert conclusion_words("The concentration of an exclave in the matrix.") == []


def test_the_findings_are_in_reading_order() -> None:
    text = "An identification and a match.\nAn exclusion.\n"

    found = conclusion_words(text)

    assert [(item.line, item.column) for item in found] == [(1, 4), (1, 25), (2, 4)]


def test_a_quotation_naming_its_source_is_not_refused() -> None:
    quoted = (
        "The algorithm classified all 433 matching and 4812 non-matching pairs "
        "correctly [cited: Song et al 2018]."
    )

    assert conclusion_words(quoted) == []


def test_the_same_quotation_without_its_source_is_refused() -> None:
    # The pair with the line above. One mark is the whole difference between
    # them, which is what says the mark is what exempts and not the wording.
    quoted = "The algorithm classified all 433 matching and 4812 non-matching pairs correctly."

    assert [item.word for item in conclusion_words(quoted)] == ["matching", "matching"]


def test_a_mark_naming_nothing_does_not_exempt() -> None:
    # The near miss for the exemption itself. '[cited:]' is what somebody writes
    # when the mark is treated as a switch, and a check that accepted it would
    # have built the switch this design is against.
    for empty in ("[cited:]", "[cited: ]", "[measured over]", "[measured over  ]"):
        found = conclusion_words(f"The two cases are a match. {empty}")
        assert [item.word for item in found] == ["match"], empty


def test_a_count_over_a_named_ground_truth_set_is_not_refused() -> None:
    # The distinction the issue asks to be documented, as a test. This sentence
    # is a property of a threshold applied to a labelled set, and it is what the
    # sensitivity milestone reports.
    measured = (
        "At threshold 6, 431 of 433 matching pairs and 0 of 4812 non-matching pairs "
        "fell above it [measured over the reference set of 2026-08-09]."
    )

    assert conclusion_words(measured) == []


def test_the_same_count_without_the_set_it_was_measured_over_is_refused() -> None:
    # A count that does not say what it was counted over is already a defect on
    # this board, so the mark asks for something the report owes anyway.
    measured = (
        "At threshold 6, 431 of 433 matching pairs and 0 of 4812 non-matching pairs fell above it."
    )

    assert [item.word for item in conclusion_words(measured)] == ["matching", "matching"]


def test_a_mark_exempts_its_own_line_and_not_the_one_after_it() -> None:
    # The bound on the mark, asserted rather than described. A mark that opened
    # a region would sit at the top of a document and cover everything anybody
    # added underneath it.
    text = "Quoted here [cited: a paper].\nThe two cases are a match.\n"

    found = conclusion_words(text)

    assert [(item.line, item.word) for item in found] == [(2, "match")]


def test_the_method_name_is_a_name_and_is_not_refused() -> None:
    for name in (METHOD, METHOD_HIGH, "congruent matching cells", "high congruent matching cells"):
        assert conclusion_words(f"Method: {name}.") == [], name


def test_the_method_name_one_word_short_is_refused() -> None:
    # The near miss for the name list. 'matching cells' is what somebody writes
    # when they shorten the method name in a sentence, and it is no longer the
    # declared name of anything.
    found = conclusion_words("Method: matching cells.")

    assert [item.word for item in found] == ["matching"]


def test_the_longer_name_is_removed_whole_rather_than_leaving_its_tail() -> None:
    # 'high congruent matching cells' contains 'congruent matching cells'. If the
    # shorter entry were tried first the longer one would still be covered, so
    # what this pins is the ordering that keeps the reported column honest.
    assert next(name for name, _reason in NAMES) == "high congruent matching cells"
    assert conclusion_words("high congruent matching cells and nothing else") == []


def test_every_declared_name_carries_a_reason() -> None:
    # The name list is the part of this check that can be widened until nothing
    # is refused, so an entry without an argument for it is refused here.
    assert [name for name, reason in NAMES if not reason.strip()] == []
    assert [word for word, reason in PLAIN_WORDS if not reason.strip()] == []


def test_the_key_a_step_is_recorded_under_is_not_a_conclusion() -> None:
    # 'identifier' shares nine letters with 'identified' and is on almost every
    # line of a manifest. The pair is what shows the exemption is the word and
    # not the stem: the near miss beside it is still refused.
    assert conclusion_words('"identifier": "level"') == []
    assert conclusion_words("Two identifiers were recorded.") == []

    found = conclusion_words("The comparison supports an identification.")

    assert [item.word for item in found] == ["identification"]


def a_reference_run() -> RunManifest:
    """A manifest of a run that made a comparison, which is rendered output."""
    return RunManifest(
        inputs=(FileRecord(role="scan-a", sha256="a" * 64),),
        profile=ProfileRecord(name="published", version="1"),
        steps=(StepRecord(identifier="level", version="2", parameters=(("model", "plane"),)),),
        comparison=ComparisonRecord(
            method=METHOD,
            version="1",
            parameters=(("correlation_threshold", 0.5), ("down_threshold", 20.0)),
        ),
        seed=20260809,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(software_version="0.0.0", dependencies=(("numpy", "2.1.0"),)),
        outputs=(FileRecord(role="surface", sha256="b" * 64),),
    )


def test_the_manifest_a_run_renders_states_no_conclusion() -> None:
    # The one artefact this project renders today. It carries the method name,
    # so it is also the case the name list exists for.
    text = a_reference_run().to_text()

    assert METHOD in text
    assert conclusion_words(text, source="manifest") == []


def test_without_the_name_list_that_same_manifest_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The name list deleted, and the suite goes red. Without this the manifest
    # test above would pass equally well if the method were called something
    # with no conclusion word in it at all.
    monkeypatch.setattr("gutachten.conclusions.NAMES", ())

    found = conclusion_words(a_reference_run().to_text(), source="manifest")

    assert [item.word for item in found] == ["matching"]


@dataclass(frozen=True)
class AScoreWithAVerdict:
    """What a convenience field looks like when somebody adds one."""

    congruent: int
    eligible: int
    is_match: bool


@dataclass(frozen=True)
class AScoreWithoutOne:
    """The same record with the field removed and nothing else changed."""

    congruent: int
    eligible: int


def test_a_verdict_boolean_on_a_result_type_is_found_and_named() -> None:
    found = conclusion_fields(AScoreWithAVerdict)

    assert found == [ConclusionField(record="AScoreWithAVerdict", field="is_match")]
    assert "#101" in str(found[0])


def test_the_same_record_without_the_field_passes() -> None:
    assert conclusion_fields(AScoreWithoutOne) == []


@dataclass(frozen=True)
class AScoreWithACount:
    """The score and its denominator, which are the numbers this project reports."""

    congruent: int
    eligible: int
    matched_cells: int


def test_a_count_is_not_a_verdict_even_where_its_name_carries_the_word() -> None:
    # 'matched_cells' is an integer. Refusing it would refuse the score itself,
    # and a check that refuses the thing the project reports gets switched off.
    assert conclusion_fields(AScoreWithACount) == []


@dataclass(frozen=True)
class AScoreWithAWordedVerdict:
    """The bound, stated rather than asserted away."""

    congruent: int
    conclusion: str


def test_a_verdict_carried_as_a_string_is_not_caught_and_that_is_the_bound() -> None:
    # Booleans are the shape a conclusion takes in a record. A string field is
    # not read here, and the text half is what meets it once it is rendered.
    assert conclusion_fields(AScoreWithAWordedVerdict) == []

    rendered_by_hand = "Conclusion: identification."
    assert [item.word for item in conclusion_words(rendered_by_hand)] == [
        "Conclusion",
        "identification",
    ]


def test_a_record_whose_annotations_are_types_rather_than_strings_is_read_too() -> None:
    # Every module in this package defers its annotations, so a field's type
    # arrives as the string 'bool'. A record built anywhere else hands over the
    # type itself, and a check that only read the string would pass it silently.
    built = dataclasses.make_dataclass("BuiltElsewhere", [("congruent", int), ("is_match", bool)])

    assert conclusion_fields(built) == [ConclusionField(record="BuiltElsewhere", field="is_match")]


def test_a_type_that_is_not_a_record_cannot_be_read_and_says_so() -> None:
    class NotARecord:
        pass

    with pytest.raises(TypeError, match="not a dataclass"):
        conclusion_fields(NotARecord)


def settings_records() -> tuple[type, ...]:
    """The records that are parameters rather than results.

    The transform half comes from the registry, which is the authority for what
    a step is, so a step added tomorrow brings its parameter record with it. The
    comparison half is named here because the comparison stage is not a
    transform and registers nothing, which is the same gap #81 records from the
    other side. A record added there and not added here reds the walk below, and
    the repair is one line with an argument attached.
    """
    from gutachten.compare.cmc import CmcParameters, HighCmcParameters
    from gutachten.compare.register import RegistrationParameters

    registered = tuple(
        REGISTRY[identifier].parameters_type for identifier in REGISTRY.identifiers()
    )
    return (*registered, CmcParameters, HighCmcParameters, RegistrationParameters)


def test_no_result_type_this_project_ships_carries_a_verdict() -> None:
    # The whole package rather than a list of types somebody remembered. This is
    # what starts failing the day a record is added with a convenience field on
    # it.
    import gutachten

    assert conclusion_fields_in(gutachten, settings=settings_records()) == []


def test_a_setting_that_asks_for_a_region_to_be_left_out_is_not_a_verdict() -> None:
    # 'exclude_drag' is a masking setting and it carries a refused stem. Without
    # the settings list the walk above refuses the parameter #57 exists to make
    # movable, which is the sweep's own vocabulary.
    import gutachten
    from gutachten.transforms.marks import MarkParameters

    unfiltered = conclusion_fields_in(gutachten)

    assert MarkParameters in settings_records()
    assert [item.field for item in unfiltered if item.record == "MarkParameters"] == [
        "exclude_drag",
        "exclude_extractor",
    ]


def test_the_package_walk_finds_a_verdict_where_one_exists() -> None:
    # The walk itself proven, so the empty result above is a measurement rather
    # than a check that never reached anything.
    import tests.unit.conclusion_example

    found = conclusion_fields_in(tests.unit.conclusion_example)

    assert found == [ConclusionField(record="Verdict", field="identified")]
