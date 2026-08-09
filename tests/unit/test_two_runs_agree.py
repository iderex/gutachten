"""Two runs of one chain agree byte for byte, and an unseeded draw is refused.

The sensitivity study is what this is for. It measures how far a score moves
when a parameter moves, so an implementation whose own noise floor is unknown
makes every number it produces worth less than it looks.

The near miss the issue asks effort to be spent on is an unseeded generator.
Somebody adds a step that reaches for a default random source, every test still
passes, and two runs quietly differ in their last digits. Two things catch that
here and neither replaces the other. The byte comparison catches it wherever the
fixture reaches, which is why the fixture has to reach every registered step and
why a step added without reaching it reds the build. The refusal catches it at
the call, with the function named, which is the difference between a diagnosis
and a search.
"""

from __future__ import annotations

import numpy as np
import pytest

from gutachten.determinism import GLOBAL_DRAWS, UnseededDraw, refuse_unseeded_draws
from gutachten.manifest import surface_digest
from gutachten.transforms.registry import REGISTRY
from tests.golden.determinism_fixture import one_run, the_chain


def test_two_runs_of_the_same_chain_produce_the_same_bytes() -> None:
    first, first_manifest = one_run()
    second, second_manifest = one_run()

    assert first.heights.tobytes() == second.heights.tobytes()
    assert surface_digest(first) == surface_digest(second)
    assert first_manifest.to_text() == second_manifest.to_text()


def test_the_manifest_is_compared_whole_because_no_field_of_it_is_expected_to_differ() -> None:
    # The issue allows for fields excluded by name, such as a wall clock time.
    # This schema carries none, so the comparison above is over the whole text
    # and there is no exclusion list to go stale. What keeps that true is this:
    # a field added that differs between two runs reds the test above, and a
    # reader who then reaches for an exclusion has to argue for it here.
    first, _ = one_run()
    second, _ = one_run()
    keys = set(one_run()[1].to_dict())

    assert surface_digest(first) == surface_digest(second)
    assert "recorded_at" not in keys and "timestamp" not in keys and "duration" not in keys


def test_the_fixture_runs_every_step_the_registry_holds() -> None:
    # This is the clause that makes the comparison above worth anything. A step
    # added to the tree and not reached by the fixture is a step the two-run
    # comparison never executes, so an unseeded draw inside it would pass.
    assert sorted(the_chain()) == sorted(REGISTRY.identifiers())


def test_a_step_added_without_reaching_the_fixture_reds_the_build() -> None:
    # The assertion above deleted in effect: a registry holding one more step
    # than the chain runs is what the tree looks like the moment somebody adds a
    # transform and stops there.
    registry_after_an_addition = (*REGISTRY.identifiers(), "a-step-nobody-added-to-a-profile")

    assert sorted(the_chain()) != sorted(registry_after_an_addition)


@pytest.mark.parametrize("name", GLOBAL_DRAWS)
def test_a_draw_from_the_global_generator_is_refused_and_names_itself(name: str) -> None:
    with refuse_unseeded_draws(np.random), pytest.raises(UnseededDraw, match=name):
        getattr(np.random, name)()


def test_a_generator_built_with_no_seed_is_refused() -> None:
    with refuse_unseeded_draws(np.random), pytest.raises(UnseededDraw, match="no seed"):
        np.random.default_rng()


def test_a_generator_built_from_the_run_seed_is_not_refused() -> None:
    # The pair with the line above. The refusal is about the missing seed and
    # not about the call, so a seeded generator has to come through unchanged.
    with refuse_unseeded_draws(np.random):
        drawn = np.random.default_rng(20260809).normal(size=3)

    assert drawn.tolist() == np.random.default_rng(20260809).normal(size=3).tolist()


def test_the_chain_runs_without_drawing_from_anything_unseeded() -> None:
    # The refusal pointed at the run itself rather than at a fixture. Every
    # registered step executes inside this window, so a step reaching for a
    # default random source fails here with its own call named.
    with refuse_unseeded_draws(np.random):
        surface, _ = one_run()

    assert np.isfinite(surface.heights).any()


def test_the_window_puts_every_entry_point_back() -> None:
    # A guard that left numpy patched would turn one failure into a suite full
    # of them, in modules that have nothing to do with this file.
    before = {name: getattr(np.random, name) for name in (*GLOBAL_DRAWS, "default_rng")}

    with refuse_unseeded_draws(np.random):
        pass

    assert {name: getattr(np.random, name) for name in before} == before


def test_the_window_puts_them_back_even_where_the_body_raised() -> None:
    # `default_rng()` rather than a legacy call, because the linter refuses a
    # literal `np.random.normal(...)` in this tree by its own rule and would red
    # the file before this guard was reached. That the two overlap is the point:
    # the linter reads what is written and this reads what is called.
    with pytest.raises(UnseededDraw), refuse_unseeded_draws(np.random):
        np.random.default_rng()

    assert np.random.default_rng(1).normal(size=1).size == 1
