"""The fuzz session against the container reader, out of the gate by configuration.

The reader is the one component that consumes bytes from outside this project,
and it stacks three parsers: a zip holding an XML document holding a binary
array. Every refusal it makes is a place where a crafted file could instead
produce a crash, an unbounded allocation, or a surface that is structurally
wrong. A session that damages a valid container a few million times finds those
faster than anybody writing fixtures by hand.

## The property, and the distinction it exists to hold

For any bytes at all, reading either produces a surface this project's own type
admits, or refuses with a reason the reader declares. A third outcome, any other
exception, is a defect: it means an input reached a branch nobody wrote a
refusal for, and whoever meets it in a laboratory gets a traceback out of a
library instead of a sentence saying what was wrong with the file.

The target asserts that distinction rather than the author remembering it, which
is the whole difference between a fuzz run and a loop that catches everything.

## Why this is not in the gate

It is a session with a time budget rather than a test with an outcome. Running
it on every pull request would add minutes to the merge path and would still be
a different set of inputs each time, so a green run would say less than it looks
like it says. It runs on a schedule instead, and what runs in the gate is the
fixtures it produces: a finding is added to the conformance corpus in
`tests/unit/x3p/containers.py`, where every pull request runs it, so a fix stays
fixed even though the session itself is not on the merge path.

The exclusion is configuration rather than a skip. `-m "not fuzz"` sits in the
`addopts` of `pyproject.toml`, and `tests/unit/test_the_fuzz_session_is_out_of_
the_gate.py` refuses its removal. A default run prints the session as deselected
rather than as passed.

## What this is not

It is a mutation session without coverage feedback. It does not know which
branches an input reached, so it cannot steer towards the ones nobody has
reached yet, and a coverage guided fuzzer would find in an hour what this finds
in a day or never. What it has instead is a corpus of containers that are valid
by construction and a set of damage operators drawn from how files actually
arrive broken. That is a real limitation and not a detail: a clean session here
is weak evidence, and the session prints what it tried so that nobody reads it
as more.
"""

from __future__ import annotations

import base64
import os
import random
import time
from collections import Counter

import numpy as np
import pytest

from gutachten.surface import Surface
from gutachten.x3p.reader import X3PError, read_bytes
from tests.fuzz.mutate import mutate, shrink
from tests.unit.x3p.containers import FIXTURES

#: How long one session runs, in seconds. Read from the environment so the
#: scheduled run and a run somebody starts by hand are the same code with
#: different budgets, and defaulted low so that starting it by accident costs a
#: minute rather than an afternoon.
BUDGET_SECONDS = float(os.environ.get("GUTACHTEN_FUZZ_SECONDS", "60"))

#: The seed the session starts from. Fixed by default so two runs of the same
#: budget try the same inputs and a finding is reproducible from the number
#: printed beside it. The scheduled run sets its own, because a session that
#: tries the same inputs every night stops finding anything after the first.
SEED = int(os.environ.get("GUTACHTEN_FUZZ_SEED", "20260809"))

pytestmark = pytest.mark.fuzz


def corpus() -> tuple[bytes, ...]:
    """The seeds, which are the conformance containers built by the writer.

    Both halves of that corpus, the ones that read and the ones that are
    refused. A refused container is a better seed than a valid one for the
    branches near a refusal, which is where a missing refusal would sit.
    """
    return tuple(fixture.build() for fixture in FIXTURES)


def read_or_refuse(data: bytes) -> str:
    """The reason the reader gave, or ``read``, raising anything it did not give.

    The one place the property lives. An ``X3PError`` is the reader answering.
    Anything else leaves this function as itself, and the session records it as a
    finding.
    """
    try:
        surface = read_bytes(data, source="fuzz")
    except X3PError as refused:
        return refused.reason
    if not isinstance(surface, Surface):  # pragma: no cover - the type says otherwise
        raise TypeError(f"the reader returned {type(surface).__name__} rather than a Surface")
    rows, columns = surface.shape
    if rows < 1 or columns < 1:
        raise ValueError(f"the reader accepted a container and produced a {surface.shape} surface")
    if surface.heights.dtype != np.float64:
        raise TypeError(f"the reader produced heights of {surface.heights.dtype}")
    if np.isinf(surface.heights).any():
        raise ValueError("the reader produced a surface carrying an infinite height")
    if surface.heights.flags.writeable:
        raise ValueError("the reader produced a surface whose heights can still be written")
    return "read"


def fails(data: bytes) -> bool:
    """Whether this input is a finding, for the reduction to steer by."""
    try:
        read_or_refuse(data)
    except (X3PError, AssertionError):  # pragma: no cover - X3PError is caught above
        return False
    except Exception:
        return True
    return False


def test_the_reader_either_reads_or_refuses_with_a_reason_it_declares() -> None:
    """The session. It ends when its budget does, and says what it tried."""
    seeds = corpus()
    rng = random.Random(SEED)
    outcomes: Counter[str] = Counter()
    tried = 0
    deadline = time.monotonic() + BUDGET_SECONDS
    while time.monotonic() < deadline:
        data = mutate(rng.choice(seeds), rng)
        tried += 1
        try:
            outcomes[read_or_refuse(data)] += 1
        except Exception as found:
            smallest = shrink(data, fails)
            raise AssertionError(
                f"an input reached a branch this reader declares no refusal for, after "
                f"{tried} inputs from seed {SEED}. The reduction got it to "
                f"{len(smallest)} bytes, and its base64 is below so it can go straight "
                f"into tests/unit/x3p/containers.py as a fixture.\n"
                f"{type(found).__name__}: {found}\n"
                f"{base64.b64encode(smallest).decode('ascii')}"
            ) from found

    print(
        f"\nfuzz: seed {SEED}, {BUDGET_SECONDS:.0f}s budget, {tried} inputs, "
        f"{sum(outcomes.values())} answered"
    )
    for reason, count in sorted(outcomes.items()):
        print(f"  {reason:32} {count}")
    assert tried > 0, (
        "the session tried nothing, so its budget was zero or its clock did not move. A "
        "session that ran no input reports the same green as one that ran millions."
    )
