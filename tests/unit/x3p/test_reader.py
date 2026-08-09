"""What the reader does beyond accepting or refusing a container.

Which containers are refused, and for which reason, is `test_fixtures.py`: the
corpus is checked as a corpus there, so a refusal added to the reader without a
container reaching it reddens rather than being remembered. What is here is the
behaviour a reason alone says nothing about, and each of these would pass a
reader that got every refusal right and the reading wrong.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gutachten.surface import AxisOrientation, LengthUnit
from gutachten.x3p.reader import MAX_HEIGHT_RANGE_MICROMETRES, X3PError, read, read_bytes
from gutachten.x3p.writer import DOCUMENT, PAYLOAD, to_bytes
from tests.support.tolerance import assert_close
from tests.unit.x3p.containers import FIXTURES, a_surface, digest_of, members, repack

#: What a value converted to metres and back is compared within, stated here
#: because the call sites below are the same comparison. Two roundings of a
#: double give a few times 1e-16; 1e-12 is four orders above that and far below
#: the resolution of any instrument this project will read.
RTOL = 1e-12
ATOL = 1e-9


def built(name: str) -> bytes:
    """One container of the corpus, by the name it is filed under."""
    return next(fixture.build for fixture in FIXTURES if fixture.name == name)()


def test_a_container_this_project_wrote_reads_back() -> None:
    surface = a_surface()
    back = read_bytes(to_bytes(surface), source="a-fixture")
    assert back.shape == surface.shape
    assert back.unit is LengthUnit.MICROMETRE
    assert (back.missing == surface.missing).all()
    assert_close(
        back.observed, surface.observed, what="heights through a container", atol=ATOL, rtol=RTOL
    )


def test_the_source_is_the_callers_and_never_the_files() -> None:
    """A container cannot name itself into a surface's identity.

    What a file says about where it came from is a claim by whoever wrote it, and
    the identity a result is traced back by is the one the caller resolved.
    """
    assert read_bytes(to_bytes(a_surface()), source="the-caller-said-so").source == (
        "the-caller-said-so"
    )


def test_reading_from_disk_takes_its_identity_from_the_file_name(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scan-0001.x3p"
    path.write_bytes(to_bytes(a_surface()))
    assert read(path).source == "scan-0001.x3p"


def test_an_empty_source_is_refused() -> None:
    with pytest.raises(ValueError, match="source identity"):
        read_bytes(to_bytes(a_surface()), source="")


def test_an_incremental_height_axis_applies_its_increment() -> None:
    """The half of the axis rule a refusal cannot demonstrate.

    Applying an increment to an absolute axis produced a surface wrong by seven
    orders of magnitude on the one real container measured. Both kinds read, so
    only the heights say whether the reader told them apart.
    """
    back = read_bytes(built("incremental-height-axis"), source="incremental")
    assert_close(
        back.observed,
        a_surface().observed,
        what="heights off an incremental axis",
        atol=ATOL,
        rtol=RTOL,
    )


def test_the_two_spacings_are_read_off_their_own_axes() -> None:
    """A reader that swapped them would pass every square container."""
    surface = a_surface()
    back = read_bytes(to_bytes(surface), source="asymmetric")
    assert_close(
        [back.spacing_y, back.spacing_x],
        [surface.spacing_y, surface.spacing_x],
        what="the two spacings",
        atol=ATOL,
        rtol=RTOL,
    )


def test_a_tampered_payload_is_refused_with_both_checksums_named() -> None:
    """The refusal the whole reader exists around, and what its message carries.

    A silently corrupted evidence file that produces a plausible score is the
    worst outcome available here. A reader saying only that a checksum did not
    match leaves whoever meets it with nothing to compare, so both values are in
    the message and this is what asserts it.
    """
    parts = members()
    payload = bytearray(parts[PAYLOAD])
    payload[0] ^= 0x01
    parts[PAYLOAD] = bytes(payload)
    with pytest.raises(X3PError) as raised:
        read_bytes(repack(parts), source="a-fixture")
    assert digest_of(parts[DOCUMENT]).decode("ascii") in str(raised.value)


def test_a_reason_outside_the_vocabulary_cannot_be_raised() -> None:
    """The vocabulary is what the corpus check reads, so it is closed.

    A refusal introduced under a name nobody declared would pass that check by
    never being counted against it.
    """
    with pytest.raises(ValueError, match="not one of this reader's reasons"):
        raise X3PError("felt wrong", "a reason nobody declared")


def test_the_orientation_of_a_foreign_container_is_the_documented_assumption() -> None:
    """The format declares none, and the reader says which way it guessed.

    A comment written by another tool is prose. Reading half of it would put an
    orientation nobody wrote into a surface, which is worse than the assumption
    the module documents.
    """
    back = read_bytes(built("a-comment-written-by-another-tool"), source="foreign")
    assert back.orientation is AxisOrientation.Y_DOWN
    assert back.provenance == ()


def test_a_container_carrying_a_broken_metadata_object_falls_back() -> None:
    """Half of this project's own metadata is not half an answer."""
    parts = members()
    parts[DOCUMENT] = parts[DOCUMENT].replace(b"<p:Comment>{", b"<p:Comment>{not json")
    back = read_bytes(repack(parts), source="broken-comment")
    assert back.orientation is AxisOrientation.Y_DOWN
    assert back.provenance == ()


def test_a_container_whose_entries_sit_under_a_directory_reads_the_same_surface() -> None:
    """The layout a widely used writer actually emits, measured on #33.

    A reader that opens `main.xml` by name fails on every file that tool wrote,
    and one that takes the first XML member it finds reads a different file than
    the one the payload belongs to.
    """
    back = read_bytes(built("entries-under-a-directory"), source="under-a-directory")
    assert_close(
        back.observed,
        a_surface().observed,
        what="heights out of a container under a directory",
        atol=ATOL,
        rtol=RTOL,
    )


def test_the_archive_is_read_from_memory_and_nothing_is_extracted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nothing this reader does puts a member on disk.

    The traversal refusals in the corpus are about names in an archive rather
    than about files that appeared, and this is the assertion that says so.
    """
    before = set(tmp_path.iterdir())
    read_bytes(to_bytes(a_surface()), source="in-memory")
    assert set(tmp_path.iterdir()) == before


def test_an_absent_measurement_survives_as_an_absence() -> None:
    """A reader that filled it in would produce a plausible surface and a wrong one."""
    back = read_bytes(to_bytes(a_surface()), source="with-a-gap")
    assert back.missing.sum() == 1
    assert np.isnan(back.heights[1, 0])


def test_a_truncated_archive_is_refused() -> None:
    """Half a file arrives from a fetch that stopped, and it is not a container."""
    written = to_bytes(a_surface())
    with pytest.raises(X3PError) as raised:
        read_bytes(written[: len(written) // 2], source="half-a-file")
    assert raised.value.reason == "not-a-container"


def test_a_surface_mis_scaled_by_a_thousand_is_refused_with_both_numbers_named() -> None:
    """The factor of a thousand, and what whoever meets it is given to work with.

    The container is well formed in every other way, so a reader without the
    range check accepts it and every step after the reader processes a surface
    wrong by three orders of magnitude without complaint. The message carries
    the span that arrived and the bound it was judged against, because a refusal
    naming neither cannot tell a file that is slightly over from one that is
    absurd.
    """
    with pytest.raises(X3PError) as raised:
        read_bytes(built("a-surface-mis-scaled-by-a-thousand"), source="in-millimetres")
    assert raised.value.reason == "height-range-implausible"
    message = str(raised.value)
    assert str(MAX_HEIGHT_RANGE_MICROMETRES) in message
    named = float(message.split(" span ")[1].split(" ")[0])
    observed = a_surface().observed
    assert_close(
        named,
        (float(observed.max()) - float(observed.min())) * 1000.0,
        what="the span the refusal names",
        atol=ATOL,
        rtol=RTOL,
    )


def test_the_same_surface_at_the_scale_the_format_fixes_reads() -> None:
    """The other side of the guard, which is what stops it being a refusal of everything.

    The two containers differ in one factor and in nothing else, so a guard that
    refused both would pass the test above and be useless, and this is what
    catches a bound set below what a cartridge case is.
    """
    back = read_bytes(built("written-by-this-project"), source="at-the-right-scale")
    assert_close(
        back.observed,
        a_surface().observed,
        what="heights at the scale the format fixes",
        atol=ATOL,
        rtol=RTOL,
    )
    assert float(back.observed.max()) - float(back.observed.min()) < MAX_HEIGHT_RANGE_MICROMETRES


def test_the_range_is_judged_on_the_measured_samples_and_not_on_the_absences() -> None:
    """A maximum over an array carrying not-a-number is not-a-number.

    That spelling of the guard compares against not-a-number, which is false for
    every container, so it passes the whole corpus while refusing nothing. Every
    real scan loses samples at its edge, so it would have been invisible.
    """
    mis_scaled = built("a-surface-mis-scaled-by-a-thousand")
    assert np.isnan(np.frombuffer(members()[PAYLOAD], dtype="<f8")).any()
    with pytest.raises(X3PError) as raised:
        read_bytes(mis_scaled, source="with-a-gap-and-mis-scaled")
    assert raised.value.reason == "height-range-implausible"


def test_a_container_in_which_nothing_was_measured_reads() -> None:
    """No span, so nothing about the scale is in question and nothing is refused.

    Whether an entirely absent surface is worth reading at all is a separate
    question, and answering it inside this refusal would make the reason on the
    error a statement the container does not support.
    """
    back = read_bytes(built("every-sample-absent"), source="nothing-measured")
    assert back.missing.all()
    assert back.observed.size == 0


def test_an_overlapping_entry_is_refused_even_where_the_caller_silences_warnings() -> None:
    """The half of that refusal the rest of the suite cannot see.

    The archive layer only warns about this shape, and the suite runs with
    warnings as errors, so every other test here would pass on a reader that did
    nothing about it and let the caller's filter decide. An operator's default
    filter prints a line and carries on reading entries that share their bytes,
    which is the zip bomb this is about.

    So the filter is set to ignore for the length of this call. What is asserted
    is that the refusal is the reader's own behaviour and not the runner's.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(X3PError) as raised:
            read_bytes(built("entries-that-share-their-bytes"), source="overlapping")
    assert raised.value.reason == "overlapping-entries"
