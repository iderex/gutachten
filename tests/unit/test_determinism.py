"""The two modes, what each records, and what neither may claim.

A mode that is only a name in a docstring is a mode nobody ran in. Every
refusal here was deleted in turn and the suite watched go red.
"""

from __future__ import annotations

import pytest

from gutachten.determinism import (
    FAST_NOTE,
    NUMERICAL_MODULES,
    REFERENCE_THREADS,
    THREAD_VARIABLES,
    DeterminismRecord,
    RunMode,
    fast_mode,
    pin_threads,
    refuse_a_late_pin,
)

# A process that has imported nothing of the numerical stack yet, which is the
# only moment a thread pin can still take effect.
NOTHING_LOADED: tuple[str, ...] = ("builtins", "os", "sys")


def test_pinning_sets_every_variable_a_backend_in_this_graph_reads() -> None:
    # Every one of them, not the one belonging to whichever BLAS happens to be
    # installed here, because which BLAS a wheel carries is decided at install
    # time and is not visible from this side.
    environment: dict[str, str] = {}

    record = pin_threads(environment=environment, loaded=NOTHING_LOADED)

    assert set(environment) == set(THREAD_VARIABLES)
    assert set(environment.values()) == {str(REFERENCE_THREADS)}
    assert record == DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS)


@pytest.mark.parametrize("module", NUMERICAL_MODULES)
def test_a_pin_arriving_after_the_backend_loaded_is_refused_and_names_it(module: str) -> None:
    # The near miss the whole module is shaped around. Setting the variables
    # here would succeed, change nothing, and leave a run labelled reference
    # that was threaded, which is worse than an honest fast run.
    with pytest.raises(RuntimeError, match=f"{module} is already imported"):
        pin_threads(environment={}, loaded=(*NOTHING_LOADED, module))


def test_a_refused_pin_leaves_the_environment_alone() -> None:
    # Half a pin is not a state this project has a word for.
    environment: dict[str, str] = {}

    with pytest.raises(RuntimeError, match="already imported"):
        pin_threads(environment=environment, loaded=("numpy",))

    assert environment == {}


def test_the_suite_itself_is_past_the_moment_a_pin_would_work() -> None:
    # Stated rather than worked around. This suite imports numpy, so the
    # refusal above is what a real call from here would meet, and the tests
    # that show a successful pin do it against a module set passed in.
    import sys

    assert set(sys.modules) & set(NUMERICAL_MODULES)
    with pytest.raises(RuntimeError, match="already imported"):
        refuse_a_late_pin(sys.modules)


def test_a_reference_run_records_the_pin_and_says_it_may_be_reported() -> None:
    record = DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS)

    assert record.reportable
    assert record.to_dict() == {
        "mode": "reference",
        "note": record.note,
        "reportable": True,
        "threads": REFERENCE_THREADS,
    }
    assert "byte for byte" in record.note


def test_a_fast_run_marks_its_own_output_as_not_for_reporting() -> None:
    # The mark is written out rather than left to be derived from the mode
    # name, because a manifest is read by somebody who did not run it and a
    # mode name only warns a reader who already knows what it costs.
    record = fast_mode()

    assert not record.reportable
    assert record.to_dict() == {
        "mode": "fast",
        "note": FAST_NOTE,
        "reportable": False,
        "threads": None,
    }
    assert "may not be used for anything reported" in record.note


def test_a_reference_run_pinned_to_anything_but_one_is_refused() -> None:
    # Four threads is reproducible on a four core machine and on no other,
    # which is the property the pin exists to remove.
    with pytest.raises(ValueError, match="reference mode is 1"):
        DeterminismRecord(mode=RunMode.REFERENCE, threads=4)
    with pytest.raises(ValueError, match="reference mode is 1"):
        DeterminismRecord(mode=RunMode.REFERENCE, threads=None)


def test_a_fast_run_recording_a_thread_count_is_refused() -> None:
    with pytest.raises(ValueError, match="Fast mode pins nothing"):
        DeterminismRecord(mode=RunMode.FAST, threads=REFERENCE_THREADS)


def test_a_mode_that_is_not_one_of_the_two_is_refused() -> None:
    # The near miss is the string, because "reference" reads correctly
    # everywhere it appears and compares equal to nothing.
    with pytest.raises(TypeError, match="must be a RunMode"):
        DeterminismRecord(mode="reference", threads=1)  # type: ignore[arg-type]
