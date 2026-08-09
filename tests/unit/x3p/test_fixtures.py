"""The conformance corpus, checked as a corpus rather than one fixture at a time.

Three things are asserted here and each is a different failure.

That every refusal the reader declares has a container reaching it. A refusal
nobody can reach is a refusal nobody has tested, and the way it arrives is a
reader growing a new branch while the fixtures stay where they were. The check
reads the reader's own vocabulary and the corpus, so neither can move without
the other.

That each refused container is refused for the reason it was built to reach. A
fixture can trip a different refusal on the way to the one it was written for,
and a test asserting only that something was raised would call that a pass.

That each accepted container reads. A corpus of things that must not be read,
with nothing in it that must, would go green on a reader that refused everything.
"""

from __future__ import annotations

import pytest

from gutachten.x3p.reader import REASONS, X3PError, read_bytes
from tests.unit.x3p.containers import FIXTURES

TREE = __file__

REFUSED = [fixture for fixture in FIXTURES if fixture.reason is not None]
ACCEPTED = [fixture for fixture in FIXTURES if fixture.reason is None]


def test_every_refusal_the_reader_declares_has_a_container_that_reaches_it() -> None:
    """The completeness check, in the direction that catches a new refusal.

    A branch added to the reader with no fixture is a branch whose message,
    whose reason and whose condition nobody has run. This is what makes the
    corpus a corpus rather than a pile of examples somebody added to.
    """
    reached = {fixture.reason for fixture in REFUSED}
    missing = sorted(set(REASONS) - reached)
    assert not missing, (
        f"{missing} are refusals the reader declares and no container in the corpus "
        "reaches them. Add one to FIXTURES, or take the refusal out of the reader."
    )


def test_no_container_claims_a_refusal_the_reader_does_not_declare() -> None:
    stray = sorted({fixture.reason for fixture in REFUSED} - set(REASONS))
    assert not stray, f"{stray} are claimed by the corpus and the reader declares no such reason"


def test_every_fixture_is_named_once() -> None:
    names = [fixture.name for fixture in FIXTURES]
    assert len(set(names)) == len(names), f"a fixture name is used twice: {sorted(names)}"


@pytest.mark.parametrize("fixture", REFUSED, ids=lambda fixture: fixture.name)
def test_each_refused_container_is_refused_for_its_stated_reason(fixture) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(X3PError) as raised:
        read_bytes(fixture.build(), source=fixture.name)
    assert raised.value.reason == fixture.reason, (
        f"{fixture.name} was built to reach {fixture.reason!r} and was refused as "
        f"{raised.value.reason!r}. A fixture that trips a different refusal on the way "
        "proves that one and not the one it is filed under."
    )


@pytest.mark.parametrize("fixture", ACCEPTED, ids=lambda fixture: fixture.name)
def test_each_accepted_container_reads(fixture) -> None:  # type: ignore[no-untyped-def]
    surface = read_bytes(fixture.build(), source=fixture.name)
    assert surface.shape == (2, 3)
    assert surface.source == fixture.name


def test_the_corpus_is_built_rather_than_stored() -> None:
    """Nothing beside these tests is a container somebody put there.

    The fixtures are built in memory from the writer, so no clone carries a scan
    and nothing is downloaded. A container dropped into this directory would be a
    file the suite reads and nobody can regenerate, and this is what reddens when
    one appears.
    """
    from pathlib import Path

    directory = Path(TREE).resolve().parent
    stray = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix not in (".py", ".md")
    )
    assert not stray, (
        f"{stray} sit beside the conformance tests and are neither code nor prose. The "
        "corpus is generated in this repository rather than downloaded, and a stored "
        "container is one nobody can rebuild."
    )
