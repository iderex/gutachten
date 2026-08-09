"""The congruent matching cells rule, against pairs whose answer is known.

The ground truth is the construction: a matching pair is two surfaces built from
one pair of striation sources with a rotation and a translation applied to one of
them, and a non-matching pair is two surfaces built from different sources. Which
is which is decided when they are made, not by looking at the score.

The surfaces carry measurement noise here, unlike the ones the registration is
tested against. A rule about how many cells agree is a rule about a spread, and a
spread of exactly zero is not the case it has to survive.

Every refusal here was removed in turn and the suite watched go red.
"""

from __future__ import annotations

import numpy as np
import pytest

from gutachten.compare.cmc import (
    METHOD,
    METHOD_HIGH,
    VERSION,
    CmcParameters,
    ConsensusRule,
    HighCmcParameters,
    Score,
    disagreements,
    record,
    record_high,
    score,
    score_high,
    score_pair,
    score_pair_high,
)
from gutachten.compare.register import (
    CellRegistration,
    Registration,
    RegistrationParameters,
    register,
)
from gutachten.determinism import REFERENCE_THREADS, DeterminismRecord, RunMode
from gutachten.manifest import EnvironmentRecord, ProfileRecord, record_run
from gutachten.surface import AxisOrientation, LengthUnit, Surface
from gutachten.synth import SurfaceParameters, generate
from gutachten.transforms.bandpass import BandpassParameters, RobustGaussianBandpass
from gutachten.transforms.pipeline import Step
from gutachten.transforms.registry import Registry
from tests.support.tolerance import assert_close

SIZE = 192
SPACING_UM = 4.0
NOISE_UM = 0.2

#: The two striation fields a surface here is built from, crossed at a right
#: angle so that both components of a displacement are determined.
ALONG_SPACING_UM = 40.0
ACROSS_SPACING_UM = 56.0

#: The sources the reference is built from, and the different ones a
#: non-matching partner is built from.
SOURCES = (11, 22)
OTHER_SOURCES = (31, 42)

ROTATION_DEG = 6.0
TRANSLATION_PX = (2.0, -3.0)
TRUE_DOWN = -int(TRANSLATION_PX[0])
TRUE_ACROSS = -int(TRANSLATION_PX[1])

SEARCH = RegistrationParameters(
    grid=4,
    minimum_valid=0.5,
    rotation_range_deg=10.0,
    rotation_step_deg=2.0,
    translation_limit=6,
)
CELLS = SEARCH.grid * SEARCH.grid


def field(*, striae_angle_deg: float, striae_spacing_um: float, seed: int) -> SurfaceParameters:
    return SurfaceParameters(
        rows=SIZE,
        columns=SIZE,
        pixel_spacing_um=SPACING_UM,
        striae_angle_deg=striae_angle_deg,
        striae_spacing_um=striae_spacing_um,
        form_depth_um=0.0,
        firing_pin_depth_um=0.0,
        drag_mark_depth_um=0.0,
        noise_um=NOISE_UM,
        seed=seed,
    )


def crossed(
    sources: tuple[int, int],
    *,
    seed: int,
    rotation_deg: float = 0.0,
    translation_px: tuple[float, float] = (0.0, 0.0),
    name: str,
) -> Surface:
    along = generate(
        field(striae_angle_deg=0.0, striae_spacing_um=ALONG_SPACING_UM, seed=seed),
        source_id=sources[0],
        rotation_deg=rotation_deg,
        translation_px=translation_px,
    )
    across = generate(
        field(striae_angle_deg=90.0, striae_spacing_um=ACROSS_SPACING_UM, seed=seed + 1),
        source_id=sources[1],
        rotation_deg=rotation_deg,
        translation_px=translation_px,
    )
    return Surface(
        heights=np.asarray(along.heights_um) + np.asarray(across.heights_um),
        spacing_y=SPACING_UM,
        spacing_x=SPACING_UM,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source=name,
    )


def a_reference() -> Surface:
    return crossed(SOURCES, seed=1, name="reference")


def a_matching_subject() -> Surface:
    # A different seed, so the noise is that of a second measurement rather than
    # the same draw twice, and the same sources, so the striae are the same face.
    return crossed(
        SOURCES,
        seed=5,
        rotation_deg=ROTATION_DEG,
        translation_px=TRANSLATION_PX,
        name="matching-subject",
    )


def a_non_matching_subject() -> Surface:
    return crossed(
        OTHER_SOURCES,
        seed=5,
        rotation_deg=ROTATION_DEG,
        translation_px=TRANSLATION_PX,
        name="non-matching-subject",
    )


def rule(**overrides: object) -> CmcParameters:
    values: dict[str, object] = {
        "down_threshold": 1.0,
        "across_threshold": 1.0,
        "rotation_threshold_deg": 2.0,
        # Deliberately low. The point of the measurement below is that agreement
        # separates these pairs and a correlation threshold does not, so the
        # threshold is set where it rejects almost nothing.
        "correlation_threshold": 0.5,
        "consensus": ConsensusRule.MEDIAN,
    }
    values.update(overrides)
    return CmcParameters(**values)  # type: ignore[arg-type]


def scored(subject: Surface, **overrides: object) -> Score:
    return score(register(subject, a_reference(), SEARCH), rule(**overrides))


def test_the_score_separates_a_matching_pair_from_a_non_matching_one() -> None:
    # The clause asking for separation on the synthetic set with known ground
    # truth. Both pairs are the same size, run through the same search with the
    # same settings, and differ only in whether the striae came from one pair of
    # sources or two.
    matching = scored(a_matching_subject())
    non_matching = scored(a_non_matching_subject())

    assert matching.congruent == 16
    assert non_matching.congruent == 4
    assert matching.eligible == non_matching.eligible == CELLS


def test_the_same_separation_holds_under_the_histogram_rule() -> None:
    # A different consensus rule is a different measurement, and the separation
    # is a property of the pairs rather than of the summary, so it survives.
    settings: dict[str, object] = {
        "consensus": ConsensusRule.HISTOGRAM_MODE,
        "translation_bin": 2.0,
        "rotation_bin_deg": 2.0,
    }
    matching = scored(a_matching_subject(), **settings)
    non_matching = scored(a_non_matching_subject(), **settings)

    assert matching.congruent == 16
    assert non_matching.congruent == 6


def test_a_correlation_threshold_on_its_own_separates_nothing() -> None:
    # Why the method counts cells that agree instead of thresholding a
    # correlation. With the three agreement thresholds opened wide enough to
    # reject nothing, only the correlation condition is left, and at a threshold
    # that keeps the matching pair whole it keeps the non-matching pair whole
    # too. A reader meeting 0.8 in an output would otherwise take it for a match.
    wide: dict[str, object] = {
        "down_threshold": 1000.0,
        "across_threshold": 1000.0,
        "rotation_threshold_deg": 1000.0,
        "correlation_threshold": 0.4,
    }
    matching = scored(a_matching_subject(), **wide)
    non_matching = scored(a_non_matching_subject(), **wide)

    assert matching.congruent == CELLS
    assert non_matching.congruent == CELLS


def test_the_consensus_is_the_applied_transform_on_a_matching_pair() -> None:
    for settings in (
        {},
        {
            "consensus": ConsensusRule.HISTOGRAM_MODE,
            "translation_bin": 2.0,
            "rotation_bin_deg": 2.0,
        },
    ):
        found = scored(a_matching_subject(), **settings)
        assert_close(
            [
                found.consensus.down,
                found.consensus.across,
                found.consensus.rotation_deg,
            ],
            [float(TRUE_DOWN), float(TRUE_ACROSS), ROTATION_DEG],
            what=f"the consensus under {settings.get('consensus', ConsensusRule.MEDIAN)}",
            atol=0.0,
        )


def test_the_histogram_consensus_is_not_shifted_to_the_centre_of_its_bin() -> None:
    # The near miss the median of the bin's members exists for. Every cell of a
    # matching pair reports the same three values, so the cells agree exactly,
    # and a consensus reported as the centre of the box they fell in would be up
    # to half a bin from anything any cell said. Every cell would then be judged
    # against a number none of them produced.
    found = scored(
        a_matching_subject(),
        consensus=ConsensusRule.HISTOGRAM_MODE,
        translation_bin=2.0,
        rotation_bin_deg=2.0,
    )

    # The bin holding a rotation of 6.0 at a width of 2.0 runs from 6.0 to 8.0
    # and its centre is 7.0, which is what this asserts is not reported.
    assert_close(
        found.consensus.rotation_deg,
        ROTATION_DEG,
        what="the histogram consensus against the value every cell reported",
        atol=0.0,
    )


def test_the_eligible_and_discarded_counts_come_with_every_score() -> None:
    # The clause asking for the denominators. A score of four means one thing
    # out of sixteen cells and another out of five.
    found = scored(a_non_matching_subject())

    assert found.eligible == CELLS
    assert found.congruent + found.discarded == found.eligible
    assert found.discarded == 12


def test_the_rejection_counts_say_which_condition_did_the_work() -> None:
    found = scored(a_non_matching_subject())

    # They overlap by construction: a cell failing three conditions is counted
    # in three of them, so they do not sum to the number discarded.
    overlapping = (
        found.failed_correlation + found.failed_down + found.failed_across + found.failed_rotation
    )
    assert overlapping > found.discarded
    # On this pair it is the displacement across the field that scatters, which
    # is the quantity the striae of two different sources disagree most about.
    assert found.failed_across == 12
    assert found.failed_correlation == 5


def test_each_of_the_four_thresholds_moves_the_score() -> None:
    # A threshold wired to nothing is a parameter a sweep would move for no
    # effect and a report would then call insensitive. Each one is tightened on
    # its own against the same pair and the score has to fall.
    subject = a_non_matching_subject()
    loose: dict[str, object] = {
        "down_threshold": 1000.0,
        "across_threshold": 1000.0,
        "rotation_threshold_deg": 1000.0,
        "correlation_threshold": 0.0,
    }
    baseline = scored(subject, **loose).congruent
    assert baseline == CELLS

    # The measured fall in each direction, so the test says how much each
    # threshold is worth on this pair rather than only that it is worth
    # something.
    for name, tightened, expected in (
        ("down_threshold", 0.0, 6),
        ("across_threshold", 0.0, 0),
        ("rotation_threshold_deg", 0.0, 7),
        ("correlation_threshold", 0.8, 1),
    ):
        moved = scored(subject, **{**loose, name: tightened}).congruent
        assert moved == expected, f"tightening {name} gave {moved} rather than {expected}"


def test_the_bin_width_changes_which_cells_count() -> None:
    # The parameter easiest to overlook in the whole method. It is not a display
    # setting: it decides which cells fall into one bin and therefore what the
    # consensus is and which cells are judged to agree with it.
    subject = a_non_matching_subject()
    settings: dict[str, object] = {
        "consensus": ConsensusRule.HISTOGRAM_MODE,
        "rotation_bin_deg": 2.0,
    }

    narrow = scored(subject, translation_bin=2.0, **settings)
    wide = scored(subject, translation_bin=4.0, **settings)

    assert (narrow.consensus.down, narrow.consensus.across) == (5.0, -4.0)
    assert (wide.consensus.down, wide.consensus.across) == (5.0, -3.0)
    assert (narrow.congruent, wide.congruent) == (6, 7)


def a_registration(values: tuple[tuple[int, int, float], ...]) -> Registration:
    """A registration assembled by hand, so a rule can be put in a stated corner."""
    matches = tuple(
        CellRegistration(
            row=index,
            column=0,
            down=down,
            across=across,
            rotation_deg=rotation,
            correlation=1.0,
            overlap=100,
        )
        for index, (down, across, rotation) in enumerate(values)
    )
    angles = tuple(sorted({match.rotation_deg for match in matches}))
    return Registration(
        matches=matches,
        by_angle=tuple(
            (angle, tuple(m for m in matches if m.rotation_deg == angle)) for angle in angles
        ),
        angles_deg=angles,
        correlations=len(values) * len(angles),
    )


def test_two_equally_full_bins_are_decided_by_the_lower_one() -> None:
    # Something has to decide a tie, and a rule written down is better than
    # whichever bin the iteration reached first. Four cells split two and two
    # between the bin from zero to two and the bin from two to four.
    registration = a_registration(((0, 0, 0.0), (0, 0, 0.0), (2, 2, 2.0), (2, 2, 2.0)))

    found = score(
        registration,
        rule(
            consensus=ConsensusRule.HISTOGRAM_MODE,
            translation_bin=2.0,
            rotation_bin_deg=2.0,
        ),
    )

    assert_close(
        [found.consensus.down, found.consensus.across, found.consensus.rotation_deg],
        [0.0, 0.0, 0.0],
        what="the consensus where two bins hold the same number of cells",
        atol=0.0,
    )


def test_the_score_is_the_same_on_two_runs() -> None:
    subject = a_matching_subject()

    first = scored(subject)
    second = scored(subject)

    assert first == second


def test_every_setting_reaches_the_manifest() -> None:
    # The clause asking for the four thresholds and the consensus rule with its
    # parameters in the manifest. Asserted against the text a reader will have
    # rather than against the record object. The search settings are in the same
    # record because the same four thresholds over a coarser rotation step count
    # different cells.
    registry = Registry()
    registry.register(RobustGaussianBandpass())

    _, manifest = record_run(
        role="input",
        surface=a_reference(),
        profile=ProfileRecord(name="a-profile", version="1"),
        chain=[
            Step(
                identifier="bandpass",
                parameters=BandpassParameters(
                    short_cutoff=20.0, long_cutoff=120.0, robust_tuning=None, robust_passes=None
                ),
            )
        ],
        registry=registry,
        seed=0,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
        comparison=record(
            SEARCH,
            rule(
                consensus=ConsensusRule.HISTOGRAM_MODE,
                translation_bin=2.0,
                rotation_bin_deg=1.5,
            ),
        ),
    )

    text = manifest.to_text()
    for fragment in (
        f'"method": "{METHOD}"',
        f'"version": "{VERSION}"',
        '"down_threshold": 1.0',
        '"across_threshold": 1.0',
        '"rotation_threshold_deg": 2.0',
        '"correlation_threshold": 0.5',
        '"consensus": "histogram-mode"',
        '"translation_bin": 2.0',
        '"rotation_bin_deg": 1.5',
        '"grid": 4',
        '"minimum_valid": 0.5',
        '"rotation_range_deg": 10.0',
        '"rotation_step_deg": 2.0',
        '"translation_limit": 6',
    ):
        assert fragment in text, f"{fragment} is not in the manifest"


def test_a_median_rule_records_no_bin_width_rather_than_an_unread_one() -> None:
    written = record(SEARCH, rule()).to_dict()["parameters"]

    assert written["consensus"] == "median"  # type: ignore[index]
    assert written["translation_bin"] is None  # type: ignore[index]
    assert written["rotation_bin_deg"] is None  # type: ignore[index]


def test_the_record_and_the_score_come_out_of_one_call() -> None:
    # So that a count cannot reach a manifest under settings that are not the
    # ones it was produced with.
    found, written = score_pair(a_matching_subject(), a_reference(), SEARCH, rule())

    assert found.congruent == 16
    assert written.method == METHOD
    assert dict(written.parameters)["correlation_threshold"] == 0.5


def test_a_registration_with_no_cell_is_refused_rather_than_scored_at_zero() -> None:
    empty = register(a_matching_subject(), a_reference(), SEARCH)
    stripped = Registration(matches=(), by_angle=(), angles_deg=empty.angles_deg, correlations=0)

    with pytest.raises(ValueError, match="nothing to take a consensus over"):
        score(stripped, rule())


def test_a_threshold_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(TypeError, match="down_threshold must be a number"):
        record(SEARCH, rule(down_threshold="one"))


def test_a_threshold_that_is_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="across_threshold must be finite"):
        record(SEARCH, rule(across_threshold=float("nan")))


def test_a_negative_agreement_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        record(SEARCH, rule(rotation_threshold_deg=-1.0))


def test_a_correlation_threshold_outside_the_range_a_correlation_lives_in_is_refused() -> None:
    with pytest.raises(ValueError, match="lies between -1 and 1"):
        record(SEARCH, rule(correlation_threshold=1.5))


def test_a_consensus_that_is_not_one_of_the_rules_is_refused() -> None:
    with pytest.raises(TypeError, match="must be a ConsensusRule"):
        record(SEARCH, rule(consensus="median"))


def test_the_histogram_rule_without_its_bin_widths_is_refused() -> None:
    with pytest.raises(ValueError, match="needs \\['translation_bin', 'rotation_bin_deg'\\]"):
        record(SEARCH, rule(consensus=ConsensusRule.HISTOGRAM_MODE))


def test_a_bin_width_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="positive finite width"):
        record(
            SEARCH,
            rule(
                consensus=ConsensusRule.HISTOGRAM_MODE,
                translation_bin=0.0,
                rotation_bin_deg=2.0,
            ),
        )


def test_a_bin_width_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(TypeError, match="rotation_bin_deg must be a number"):
        record(
            SEARCH,
            rule(
                consensus=ConsensusRule.HISTOGRAM_MODE,
                translation_bin=2.0,
                rotation_bin_deg="two",
            ),
        )


def test_a_median_rule_stating_a_bin_width_is_refused() -> None:
    with pytest.raises(ValueError, match="does not read"):
        record(SEARCH, rule(translation_bin=2.0))


def high(**overrides: object) -> HighCmcParameters:
    values: dict[str, object] = {
        "down_threshold": 1.0,
        "across_threshold": 1.0,
        "correlation_threshold": 0.5,
        "consensus": ConsensusRule.MEDIAN,
        "high_tolerance": 1,
    }
    values.update(overrides)
    return HighCmcParameters(**values)  # type: ignore[arg-type]


def test_both_rules_run_from_one_registration() -> None:
    # The clause asking for that. One search, two decision layers, so a
    # difference between the two scores is a difference between the rules and
    # not between two pipelines.
    registration = register(a_matching_subject(), a_reference(), SEARCH)

    original = score(registration, rule())
    variant = score_high(registration, high())

    assert original.eligible == variant.eligible == CELLS
    assert original.congruent == 16
    assert variant.congruent == 16
    assert variant.peak == 16
    assert variant.high_angles_deg == (ROTATION_DEG,)


def test_the_count_by_orientation_is_reported_and_not_only_its_peak() -> None:
    # The shape is the thing the variant is about. A matching pair puts a spike
    # on one orientation, and a reader given only the peak cannot tell that from
    # a count spread over the whole range.
    variant = score_high(register(a_matching_subject(), a_reference(), SEARCH), high())

    assert variant.per_angle == (
        (-10.0, 0),
        (-8.0, 0),
        (-6.0, 0),
        (-4.0, 0),
        (-2.0, 0),
        (0.0, 0),
        (2.0, 2),
        (4.0, 4),
        (6.0, 16),
        (8.0, 4),
        (10.0, 0),
    )


def test_the_two_rules_disagree_by_naming_cells_and_not_by_a_difference_of_totals() -> None:
    # The clause asking for the disagreements to be enumerated. On the
    # non-matching pair the variant recovers three cells the original rule
    # discards, and it recovers them because they agreed with each other at an
    # orientation the single consensus did not land on.
    registration = register(a_non_matching_subject(), a_reference(), SEARCH)

    original = score(registration, rule())
    variant = score_high(registration, high())
    only_original, only_high = disagreements(original, variant)

    assert (original.congruent, variant.congruent) == (4, 7)
    assert only_original == ()
    assert only_high == ((0, 2), (1, 2), (3, 3))


def test_the_variant_raises_the_non_matching_score_as_well_as_the_matching_one() -> None:
    # A result that makes the method look worse, reported in the same shape as a
    # flattering one. Recovering cells the original rule misses is what the
    # variant is for, and on this construction it recovers them for a
    # non-matching pair too, so the gap between the two pairs narrows from
    # 16 against 4 to 16 against 7. Whether that holds on real data is #77.
    matching = score_high(register(a_matching_subject(), a_reference(), SEARCH), high())
    non_matching = score_high(register(a_non_matching_subject(), a_reference(), SEARCH), high())

    assert (matching.congruent, non_matching.congruent) == (16, 7)
    assert matching.congruent > non_matching.congruent


def two_peaks() -> Registration:
    """A registration with two orientations that nearly tie, built by hand.

    The case the variant exists for does not arise on the generated pairs above,
    where one orientation wins outright, so it is constructed. Six cells, four
    agreeing at one orientation and three at another, with every cell present at
    both.
    """
    layout: dict[float, tuple[tuple[int, int, int], ...]] = {
        # angle: (cell, down, across) for each of the six cells
        0.0: ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 9, 9), (5, 9, 9)),
        2.0: ((0, 20, 20), (1, 20, 20), (2, -5, -5), (3, -5, -5), (4, -5, -5), (5, 20, 20)),
    }
    by_angle = tuple(
        (
            angle,
            tuple(
                CellRegistration(
                    row=cell,
                    column=0,
                    down=down,
                    across=across,
                    rotation_deg=angle,
                    correlation=0.9,
                    overlap=100,
                )
                for cell, down, across in cells
            ),
        )
        for angle, cells in layout.items()
    )
    # Every cell correlates equally at both orientations, so the best-over-angles
    # answer is the first orientation searched, which is what the original rule
    # sees and the variant does not have to.
    return Registration(
        matches=by_angle[0][1],
        by_angle=by_angle,
        angles_deg=tuple(layout),
        correlations=12,
    )


def test_the_tolerance_decides_how_many_orientations_count_as_the_peak() -> None:
    registration = two_peaks()
    settings: dict[str, object] = {
        "consensus": ConsensusRule.HISTOGRAM_MODE,
        "translation_bin": 2.0,
    }

    tight = score_high(registration, high(high_tolerance=0, **settings))
    loose = score_high(registration, high(high_tolerance=1, **settings))

    assert tight.peak == loose.peak == 4
    assert tight.high_angles_deg == (0.0,)
    assert loose.high_angles_deg == (0.0, 2.0)
    assert tight.cells == ((0, 0), (1, 0), (2, 0), (3, 0))
    assert loose.cells == ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))


def test_the_variant_recovers_a_cell_the_single_consensus_discards() -> None:
    # Why the variant exists, on the construction that shows it. Cell four agrees
    # with cells two and three at the second orientation and with nobody at the
    # first, so the rule that keeps one orientation throws it away.
    registration = two_peaks()
    settings: dict[str, object] = {
        "consensus": ConsensusRule.HISTOGRAM_MODE,
        "translation_bin": 2.0,
    }

    original = score(
        registration, rule(rotation_threshold_deg=0.0, rotation_bin_deg=2.0, **settings)
    )
    variant = score_high(registration, high(high_tolerance=1, **settings))
    only_original, only_high = disagreements(original, variant)

    assert original.congruent == 4
    assert variant.congruent == 5
    assert only_original == ()
    assert only_high == ((4, 0),)


def test_a_pair_where_no_orientation_holds_two_agreeing_cells_scores_nothing() -> None:
    # Every orientation counts zero, so there is no peak to be near. Naming every
    # orientation as within tolerance of a peak of nothing would report the whole
    # search as the region of agreement.
    registration = two_peaks()

    variant = score_high(registration, high(correlation_threshold=0.99))

    assert variant.peak == 0
    assert variant.congruent == 0
    assert variant.high_angles_deg == ()
    assert variant.discarded == variant.eligible


def test_each_rule_says_which_rule_it_is() -> None:
    # The clause asking that a report can state which rule produced a score.
    registration = register(a_matching_subject(), a_reference(), SEARCH)

    assert score(registration, rule()).rule == METHOD
    assert score_high(registration, high()).rule == METHOD_HIGH


def test_the_two_rules_are_recorded_separately() -> None:
    original = dict(record(SEARCH, rule()).parameters)
    variant = dict(record_high(SEARCH, high()).parameters)

    assert record(SEARCH, rule()).method == METHOD
    assert record_high(SEARCH, high()).method == METHOD_HIGH
    assert variant["high_tolerance"] == 1
    # The variant reads no rotation threshold, so its record states none. A
    # record carrying one would say a threshold decided a count that never
    # looked at it.
    assert "rotation_threshold_deg" not in variant
    assert "rotation_bin_deg" not in variant
    assert "rotation_threshold_deg" in original


def test_the_variant_settings_reach_the_manifest() -> None:
    registry = Registry()
    registry.register(RobustGaussianBandpass())

    _, manifest = record_run(
        role="input",
        surface=a_reference(),
        profile=ProfileRecord(name="a-profile", version="1"),
        chain=[
            Step(
                identifier="bandpass",
                parameters=BandpassParameters(
                    short_cutoff=20.0, long_cutoff=120.0, robust_tuning=None, robust_passes=None
                ),
            )
        ],
        registry=registry,
        seed=0,
        determinism=DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS),
        environment=EnvironmentRecord(software_version="0.0.0", dependencies=()),
        comparison=record_high(
            SEARCH,
            high(consensus=ConsensusRule.HISTOGRAM_MODE, translation_bin=2.0, high_tolerance=3),
        ),
    )

    text = manifest.to_text()
    for fragment in (
        f'"method": "{METHOD_HIGH}"',
        '"high_tolerance": 3',
        '"translation_bin": 2.0',
        '"consensus": "histogram-mode"',
        '"rotation_step_deg": 2.0',
    ):
        assert fragment in text, f"{fragment} is not in the manifest"


def test_the_record_and_the_variant_score_come_out_of_one_call() -> None:
    found, written = score_pair_high(a_matching_subject(), a_reference(), SEARCH, high())

    assert found.congruent == 16
    assert written.method == METHOD_HIGH


def test_a_registration_with_no_cell_is_refused_by_the_variant_too() -> None:
    stripped = Registration(matches=(), by_angle=(), angles_deg=(0.0,), correlations=0)

    with pytest.raises(ValueError, match="nothing to take a consensus over"):
        score_high(stripped, high())


def test_a_tolerance_that_is_not_a_whole_number_of_cells_is_refused() -> None:
    with pytest.raises(TypeError, match="whole one"):
        record_high(SEARCH, high(high_tolerance=1.5))


def test_a_tolerance_given_as_a_boolean_is_refused() -> None:
    with pytest.raises(TypeError, match="whole one"):
        record_high(SEARCH, high(high_tolerance=True))


def test_a_negative_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        record_high(SEARCH, high(high_tolerance=-1))


def test_the_variant_refuses_a_histogram_rule_with_no_bin_width() -> None:
    with pytest.raises(ValueError, match="needs"):
        record_high(SEARCH, high(consensus=ConsensusRule.HISTOGRAM_MODE))


def test_the_variant_refuses_a_median_rule_that_states_a_bin_width() -> None:
    with pytest.raises(ValueError, match="does not read"):
        record_high(SEARCH, high(translation_bin=2.0))


def from_layout(layout: dict[float, tuple[tuple[int, int, int, float], ...]]) -> Registration:
    """A registration built from what each cell did at each orientation.

    ``matches`` is derived here rather than stated, as the search derives it: the
    orientation where a cell correlated best. A fixture that stated the two
    independently could describe a search that never happened.
    """
    by_angle = tuple(
        (
            angle,
            tuple(
                CellRegistration(
                    row=cell,
                    column=0,
                    down=down,
                    across=across,
                    rotation_deg=angle,
                    correlation=correlation,
                    overlap=100,
                )
                for cell, down, across, correlation in cells
            ),
        )
        for angle, cells in layout.items()
    )
    best: dict[int, CellRegistration] = {}
    for _, matches in by_angle:
        for match in matches:
            standing = best.get(match.row)
            if standing is None or match.correlation > standing.correlation:
                best[match.row] = match
    return Registration(
        matches=tuple(best[row] for row in sorted(best)),
        by_angle=by_angle,
        angles_deg=tuple(layout),
        correlations=sum(len(matches) for _, matches in by_angle),
    )


def a_cell_whose_best_orientation_is_nowhere_near_the_peak() -> Registration:
    """Six cells, four agreeing at one orientation and one stray agreeing elsewhere.

    Cell five correlates best at an orientation where no other cell agrees with
    it. Best over orientations it lands on the same displacement as the four, so
    the single-consensus rule counts it. The variant does not, because the
    orientation it came from is not near the busiest one.
    """
    return from_layout(
        {
            0.0: (
                (0, 30, 30, 0.30),
                (1, 30, 30, 0.30),
                (2, 30, 30, 0.30),
                (3, 30, 30, 0.30),
                (4, 30, 30, 0.30),
                (5, 0, 0, 0.95),
            ),
            2.0: (
                (0, 30, 30, 0.30),
                (1, 30, 30, 0.30),
                (2, 30, 30, 0.30),
                (3, 30, 30, 0.30),
                (4, 30, 30, 0.30),
                (5, 30, 30, 0.30),
            ),
            4.0: (
                (0, 0, 0, 0.90),
                (1, 0, 0, 0.90),
                (2, 0, 0, 0.90),
                (3, 0, 0, 0.90),
                (4, 30, 30, 0.90),
                (5, 30, 30, 0.90),
            ),
        }
    )


def test_the_single_consensus_rule_counts_a_cell_the_variant_does_not() -> None:
    # The other direction of the disagreement, and the cost of the variant
    # rather than its benefit. A subtraction of the two totals would report this
    # as the variant losing one cell; naming them says which cell and why.
    registration = a_cell_whose_best_orientation_is_nowhere_near_the_peak()
    settings: dict[str, object] = {
        "consensus": ConsensusRule.HISTOGRAM_MODE,
        "translation_bin": 2.0,
    }

    original = score(
        registration,
        rule(rotation_threshold_deg=5.0, rotation_bin_deg=2.0, **settings),
    )
    variant = score_high(registration, high(high_tolerance=1, **settings))
    only_original, only_high = disagreements(original, variant)

    assert (original.congruent, variant.congruent) == (5, 4)
    assert only_original == ((5, 0),)
    assert only_high == ()
