"""What the reader accepts, and what it refuses for the reason it names.

Every refusal is asserted on the reason rather than on the message, because a
fixture built to reach one refusal can trip a different one on the way there and
a test matching prose would not notice. The reasons are a closed vocabulary in
the reader, so a refusal nobody named cannot be introduced quietly.

The fixtures are containers this project's writer produced and then changed in
exactly one way. `containers.py` beside this file says why.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from gutachten.surface import AxisOrientation, LengthUnit
from gutachten.x3p import reader
from gutachten.x3p.reader import REASONS, X3PError, read, read_bytes
from gutachten.x3p.writer import DOCUMENT, PAYLOAD, to_bytes
from tests.support.tolerance import assert_close
from tests.unit.x3p.containers import a_surface, changed, members, repack, with_payload

#: The tolerance a value converted to metres and back is compared within, and it
#: is stated at every call site rather than hidden here. Micrometres to metres is
#: a multiplication by a number that is not a power of two, so the round trip is
#: two roundings; a relative tolerance of 1e-12 is four orders above what two
#: roundings of a double can produce and far below any instrument resolution.
RTOL = 1e-12
ATOL = 1e-9


def test_a_container_this_project_wrote_reads_back() -> None:
    surface = a_surface()
    back = read_bytes(to_bytes(surface), source="a-fixture")
    assert back.shape == surface.shape
    assert back.unit is LengthUnit.MICROMETRE
    assert_close(
        back.observed, surface.observed, what="heights through a container", atol=ATOL, rtol=RTOL
    )
    assert (back.missing == surface.missing).all()


def test_the_source_is_the_callers_and_never_the_files() -> None:
    """A container cannot name itself into a surface's provenance.

    What a file says about where it came from is a claim by whoever wrote it, and
    the identity a result is traced back by is the one the caller resolved.
    """
    back = read_bytes(to_bytes(a_surface()), source="the-caller-said-so")
    assert back.source == "the-caller-said-so"


def test_reading_from_disk_takes_its_identity_from_the_file_name(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scan-0001.x3p"
    path.write_bytes(to_bytes(a_surface()))
    assert read(path).source == "scan-0001.x3p"


def test_an_empty_source_is_refused() -> None:
    with pytest.raises(ValueError, match="source identity"):
        read_bytes(to_bytes(a_surface()), source="")


def refused(data: bytes) -> str:
    with pytest.raises(X3PError) as raised:
        read_bytes(data, source="a-fixture")
    assert raised.value.reason in REASONS
    return raised.value.reason


def test_a_file_that_is_not_a_zip_is_refused() -> None:
    assert refused(b"ISO5436_2 but not in an archive") == "not-a-container"


def test_an_entry_that_climbs_out_of_the_archive_is_refused() -> None:
    """The standard zip attack, refused on the name rather than on extraction.

    This reader never writes a member to disk, so nothing here would be
    overwritten. The name is refused anyway: a container carrying one was not
    written to be read, and the fixtures for the fuzzer in #115 start from that.
    """
    parts = members()
    parts["../escaped.txt"] = b"outside"
    assert refused(repack(parts)) == "entry-outside-the-archive"


def test_an_entry_naming_an_absolute_location_is_refused() -> None:
    parts = members()
    parts["/etc/passwd"] = b"outside"
    assert refused(repack(parts)) == "entry-outside-the-archive"


def test_a_point_data_link_that_climbs_out_of_the_archive_is_refused() -> None:
    """The same attack one level in, through the document rather than the archive.

    A reader that hardened the member names and then followed the link the
    document gives has moved the hole rather than closed it.
    """
    assert (
        refused(changed(b"<p:PointDataLink>bindata/data.bin", b"<p:PointDataLink>../../etc/passwd"))
        == "entry-outside-the-archive"
    )


def test_a_member_larger_than_this_reader_admits_is_refused(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The declared size is read before anything is decompressed.

    The cap is lowered here rather than a half gigabyte fixture being built,
    because what is under test is that the declared size is consulted at all. A
    container of a few hundred bytes can declare a member of a terabyte, and a
    reader that finds out by allocating it has already lost.
    """
    monkeypatch.setattr(reader, "MAX_ENTRY_BYTES", 16)
    assert refused(to_bytes(a_surface())) == "entry-too-large"


def test_an_archive_with_no_metadata_document_is_refused() -> None:
    parts = members()
    del parts[DOCUMENT]
    assert refused(repack(parts)) == "no-metadata-document"


def test_an_archive_with_two_metadata_documents_is_refused() -> None:
    """Choosing one would make the surface depend on the order of the entries."""
    parts = members()
    parts["second/main.xml"] = parts[DOCUMENT]
    assert refused(repack(parts)) == "several-metadata-documents"


def test_a_container_whose_entries_sit_under_a_directory_reads() -> None:
    """The layout a widely used writer actually emits, measured on #33.

    A reader that opens `main.xml` by name fails on every file that tool wrote.
    """
    parts = {f"a-scan/{name}": content for name, content in members().items()}
    parts["a-scan/main.xml"] = parts["a-scan/main.xml"]
    back = read_bytes(repack(parts), source="under-a-directory")
    assert back.shape == a_surface().shape


def test_a_metadata_document_that_is_not_xml_is_refused() -> None:
    parts = members()
    parts[DOCUMENT] = b"<<< not a document"
    assert refused(repack(parts)) == "not-xml"


def test_an_empty_metadata_document_is_refused() -> None:
    parts = members()
    parts[DOCUMENT] = b"<?xml version='1.0'?><!-- nothing -->"
    assert refused(repack(parts)) == "not-xml"


def test_a_document_type_declaration_is_refused_before_it_expands() -> None:
    """The billion laughs shape, refused at the declaration.

    The refusal is on the declaration rather than on the reference, which is
    stronger: an expansion cannot be counted and stopped part way if it never
    starts.
    """
    parts = members()
    parts[DOCUMENT] = (
        b"<?xml version='1.0'?><!DOCTYPE ISO5436_2 [<!ENTITY a 'aaaaaaaaaa'>"
        b"<!ENTITY b '&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;'>]><ISO5436_2>&b;</ISO5436_2>"
    )
    assert refused(repack(parts)) == "document-type-declaration"


def test_an_external_entity_is_refused_without_being_resolved() -> None:
    """The reader does not reach a file or a host named by an evidence file.

    The refusal names the declaration rather than the external reference, because
    the declaration is where it is stopped. That is a disclosure about which
    check fired, not a weaker guarantee: nothing outside the archive is opened
    either way.
    """
    parts = members()
    parts[DOCUMENT] = (
        b"<?xml version='1.0'?><!DOCTYPE ISO5436_2 ["
        b"<!ENTITY out SYSTEM 'file:///etc/passwd'>]><ISO5436_2>&out;</ISO5436_2>"
    )
    assert refused(repack(parts)) == "document-type-declaration"


def test_point_data_carried_inline_is_refused_as_recognised_and_unsupported() -> None:
    """Understood and declined, which is a different statement from not understood.

    A wrong answer out of a partially understood file is worse than a refusal, so
    the message says the construct was recognised.
    """
    assert (
        refused(
            changed(b"<p:DataLink>", b"<p:DataList><p:Datum>1</p:Datum></p:DataList><p:DataLink>")
        )
        == "recognised-and-unsupported"
    )


def test_a_stack_of_layers_is_refused() -> None:
    assert refused(changed(b"<p:SizeZ>1</p:SizeZ>", b"<p:SizeZ>3</p:SizeZ>")) == (
        "recognised-and-unsupported"
    )


def test_a_payload_of_another_data_type_is_refused() -> None:
    assert (
        refused(
            changed(
                b"<p:CZ><p:AxisType>A</p:AxisType><p:DataType>D</p:DataType>",
                b"<p:CZ><p:AxisType>A</p:AxisType><p:DataType>I16</p:DataType>",
            )
        )
        == "recognised-and-unsupported"
    )


def test_a_height_axis_of_an_unknown_type_is_refused() -> None:
    assert (
        refused(changed(b"<p:CZ><p:AxisType>A</p:AxisType>", b"<p:CZ><p:AxisType>Q</p:AxisType>"))
        == "recognised-and-unsupported"
    )


def test_an_incremental_height_axis_applies_its_increment() -> None:
    """The other half of the axis rule, and the reason it is read at all.

    Applying an increment to an absolute axis produced a surface wrong by seven
    orders of magnitude on the one real container measured. The two cases are
    handled apart, so both are exercised.
    """
    surface = a_surface()
    document = members()[DOCUMENT]
    scaled = np.ascontiguousarray(surface.heights * 1e-6 / 4.0, dtype="<f8").tobytes()
    parts = members()
    parts[DOCUMENT] = document.replace(
        b"<p:CZ><p:AxisType>A</p:AxisType><p:DataType>D</p:DataType><p:Increment>1.0</p:Increment>",
        b"<p:CZ><p:AxisType>I</p:AxisType><p:DataType>D</p:DataType><p:Increment>4.0</p:Increment>",
    )
    import hashlib

    parts[PAYLOAD] = scaled
    parts[DOCUMENT] = parts[DOCUMENT].replace(
        _digest(document), hashlib.md5(scaled, usedforsecurity=False).hexdigest().encode("ascii")
    )
    back = read_bytes(repack(parts), source="incremental")
    assert_close(
        back.observed,
        surface.observed,
        what="heights off an incremental axis",
        atol=ATOL,
        rtol=RTOL,
    )


def _digest(document: bytes) -> bytes:
    start = document.index(b"<p:MD5ChecksumPointData>") + len(b"<p:MD5ChecksumPointData>")
    return document[start : start + 32]


def test_an_absent_axis_increment_is_refused() -> None:
    """The refusal #45 asks for, aimed at what the format actually carries.

    No X3P file declares a unit; the format fixes lengths as metres, so a check
    for a missing unit element would never fire. What can be absent is the
    increment, and an axis with none says nothing about how far apart two samples
    are.
    """
    assert refused(changed(b"<p:Increment>4e-06</p:Increment>", b"")) == "element-missing"


def test_an_axis_increment_that_is_not_a_number_is_refused() -> None:
    assert refused(
        changed(b"<p:Increment>4e-06</p:Increment>", b"<p:Increment>wide</p:Increment>")
    ) == ("not-a-number")


def test_an_axis_increment_of_nothing_is_refused() -> None:
    assert refused(
        changed(b"<p:Increment>4e-06</p:Increment>", b"<p:Increment>0.0</p:Increment>")
    ) == ("increment-not-positive")


def test_a_declared_size_that_disagrees_with_the_payload_is_refused() -> None:
    """The near miss is one row, not a wild number.

    Reading the shorter of the two produces a surface of the declared shape
    carrying somebody else's numbers at the end of it, which is a plausible
    surface and a wrong one.
    """
    assert refused(changed(b"<p:SizeY>2</p:SizeY>", b"<p:SizeY>3</p:SizeY>")) == (
        "size-disagrees-with-the-payload"
    )


def test_a_matrix_dimension_that_is_not_a_count_is_refused() -> None:
    assert refused(changed(b"<p:SizeX>3</p:SizeX>", b"<p:SizeX>three</p:SizeX>")) == "not-a-number"


def test_a_matrix_dimension_of_zero_is_refused() -> None:
    assert refused(changed(b"<p:SizeX>3</p:SizeX>", b"<p:SizeX>0</p:SizeX>")) == "not-a-number"


def test_a_point_data_link_naming_nothing_in_the_archive_is_refused() -> None:
    assert refused(changed(b"bindata/data.bin", b"bindata/absent.bin")) == "element-missing"


def test_a_tampered_payload_is_refused_with_both_checksums_named() -> None:
    """The refusal the whole reader exists around.

    A silently corrupted evidence file that produces a plausible score is the
    worst outcome available here, so the one byte changed below is one byte, not
    a mangled file.
    """
    parts = members()
    payload = bytearray(parts[PAYLOAD])
    payload[0] ^= 0x01
    with pytest.raises(X3PError) as raised:
        read_bytes(with_payload(bytes(payload)), source="a-fixture")
    assert raised.value.reason == "point-data-checksum-mismatch"
    assert _digest(parts[DOCUMENT]).decode("ascii") in str(raised.value)


def test_an_infinite_height_is_refused() -> None:
    """An absence is not-a-number and an infinity is neither that nor a measurement."""
    surface = a_surface()
    heights = np.array(surface.heights, dtype="<f8")
    heights[0, 0] = np.inf
    payload = np.ascontiguousarray(heights * 1e-6, dtype="<f8").tobytes()
    import hashlib

    parts = members()
    parts[PAYLOAD] = payload
    parts[DOCUMENT] = parts[DOCUMENT].replace(
        _digest(parts[DOCUMENT]),
        hashlib.md5(payload, usedforsecurity=False).hexdigest().encode("ascii"),
    )
    assert refused(repack(parts)) == "height-not-finite"


def test_a_truncated_archive_is_refused() -> None:
    written = to_bytes(a_surface())
    with pytest.raises(X3PError) as raised:
        read_bytes(written[: len(written) // 2], source="a-fixture")
    assert raised.value.reason in ("not-a-container",)


def test_a_reason_outside_the_vocabulary_cannot_be_raised() -> None:
    """The vocabulary is what a test asserts on, so it is closed.

    A refusal introduced under a name nobody declared would pass every test here
    by never being asserted against.
    """
    with pytest.raises(ValueError, match="not one of this reader's reasons"):
        raise X3PError("felt wrong", "a reason nobody declared")


def test_the_orientation_of_a_foreign_container_is_the_documented_assumption() -> None:
    """The format declares none, and the reader says which way it guessed.

    A comment written by another tool is prose. Reading half of it would put an
    orientation nobody wrote into a surface, which is worse than the assumption
    the module documents.
    """
    assert (
        read_bytes(
            changed(b"<p:Comment>", b"<p:Comment>written by another tool"), "foreign"
        ).orientation
        is AxisOrientation.Y_DOWN
    )


def test_a_container_carrying_a_broken_metadata_object_falls_back() -> None:
    parts = members()
    parts[DOCUMENT] = parts[DOCUMENT].replace(b"<p:Comment>{", b"<p:Comment>{not json")
    back = read_bytes(repack(parts), source="broken-comment")
    assert back.orientation is AxisOrientation.Y_DOWN
    assert back.provenance == ()


def test_the_archive_is_read_from_memory_and_nothing_is_extracted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nothing this reader does puts a member on disk.

    The traversal refusals above are about names in an archive rather than about
    files that appeared, and this is the assertion that says so.
    """
    before = set(tmp_path.iterdir())
    read_bytes(to_bytes(a_surface()), source="in-memory")
    assert set(tmp_path.iterdir()) == before


def test_the_reader_reads_a_zip_written_without_compression() -> None:
    """A container stored rather than deflated is still a container."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, content in members().items():
            archive.writestr(name, content)
    assert read_bytes(buffer.getvalue(), source="stored").shape == a_surface().shape


def test_a_document_missing_a_whole_branch_is_refused() -> None:
    """The path is walked one element at a time and stops at the first absence.

    Renaming the link element rather than deleting its text is the near miss: the
    document stays well formed, so a reader that only guarded against broken XML
    would walk into a None.
    """
    assert refused(changed(b"p:DataLink>", b"p:DataLinkage>")) == "element-missing"


def test_an_axis_increment_that_is_not_finite_is_refused() -> None:
    """An infinity parses as a float and is not a spacing.

    Refusing only what fails to parse would let this through, and every distance
    computed from it afterwards is an infinity that no later step reports.
    """
    assert refused(
        changed(b"<p:Increment>4e-06</p:Increment>", b"<p:Increment>inf</p:Increment>")
    ) == ("not-a-number")


def test_a_member_whose_bytes_were_altered_after_packing_is_refused() -> None:
    """The archive's own checksum, met before this reader's.

    A container stored uncompressed and then edited in place fails the zip
    checksum on read. That arrives here as a container that could not be read,
    which is a different reason from a payload whose declared point data checksum
    disagrees, and both are refusals rather than one of them being a crash.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, content in members().items():
            archive.writestr(name, content)
    written = bytearray(buffer.getvalue())
    payload = members()[PAYLOAD]
    at = written.index(payload)
    written[at] ^= 0xFF
    assert refused(bytes(written)) == "not-a-container"
