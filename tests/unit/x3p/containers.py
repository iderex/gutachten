"""The conformance corpus, built here, so the reader is tested against no download.

Every fixture starts as a container this project's own writer produced and is
then changed in exactly one way. That shape is deliberate. A malformed container
assembled from scratch tends to be malformed in several ways at once, and a test
over one of those cannot say which of them the refusal was about, which is the
difference between proving a refusal and proving that the reader raised
something.

The accepted containers come from the writer rather than from a stored file, so
the two sides of the round trip are the two sides that ship, and the corpus moves
with the format code instead of drifting behind it.

## Why nothing here is a file in the tree

The refusable containers do not exist in any public corpus, and a corpus of real
scans would put tens of megabytes into every clone. These are built in memory
when a test asks for one, so nothing is downloaded, nothing is stored, and the
rule about holding a byte exact fixture as base64 does not reach them: that rule
exists because a raw file is normalised on its way into git, and none of these
ever goes near git. `test_fixtures.py` refuses a stored one appearing beside
them.

## The catalogue is what the completeness check reads

`FIXTURES` is the corpus. Each entry names itself, names the refusal it is built
to reach or names none where it is meant to read, and carries the code that
builds it. The check that every refusal the reader declares has a fixture reading
it reads this list, so a refusal added to the reader without one reddens the
suite rather than being remembered.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gutachten.surface import AxisOrientation, LengthUnit, Surface, TransformRecord
from gutachten.x3p.writer import DOCUMENT, PAYLOAD, to_bytes


def a_surface() -> Surface:
    """The surface every fixture is built from.

    Small, not square, and with one absent measurement, because a square surface
    hides a transposed reader and absence is the thing a container writer is
    most likely to lose.
    """
    heights = np.array([[1.5, -2.25, 3.0], [np.nan, 5.5, 6.125]], dtype=np.float64)
    return Surface(
        heights=heights,
        spacing_y=4.0,
        spacing_x=2.5,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="generated-for-a-fixture",
        provenance=(TransformRecord.of("level", "1", model="plane", order=None),),
    )


def members(surface: Surface | None = None) -> dict[str, bytes]:
    """The members of a container the writer produced, ready to be changed."""
    written = to_bytes(surface if surface is not None else a_surface())
    with zipfile.ZipFile(io.BytesIO(written)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def repack(parts: dict[str, bytes], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    """A zip holding exactly these members, in the order given."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in parts.items():
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            archive.writestr(info, content)
    return buffer.getvalue()


def changed(before: bytes, after: bytes, surface: Surface | None = None) -> bytes:
    """A container whose metadata document has ``before`` replaced by ``after``.

    The replacement is asserted to have changed something. A fixture built on a
    string the document does not carry is a fixture that reads as valid and
    proves the reader accepts a file, which is the opposite of what it exists
    for, and it is the failure a mutation that missed produces.
    """
    parts = members(surface)
    document = parts[DOCUMENT]
    if before not in document:
        raise AssertionError(f"{before!r} is not in the metadata document, so nothing changed")
    parts[DOCUMENT] = document.replace(before, after)
    return repack(parts)


def digest_of(document: bytes) -> bytes:
    """The point data checksum a document declares, as it is written there."""
    start = document.index(b"<p:MD5ChecksumPointData>") + len(b"<p:MD5ChecksumPointData>")
    return document[start : start + 32]


def with_payload(payload: bytes, *, restate_checksum: bool) -> bytes:
    """A container carrying ``payload`` instead of the one the writer produced.

    ``restate_checksum`` decides which of two different fixtures this is. Left
    alone, the declared checksum no longer matches and the container is one whose
    point data was tampered with. Restated, the checksum agrees again and the
    container is a valid one carrying whatever the payload says, which is how the
    fixtures for the axis and the height refusals are built.
    """
    parts = members()
    parts[PAYLOAD] = payload
    if restate_checksum:
        parts[DOCUMENT] = parts[DOCUMENT].replace(
            digest_of(parts[DOCUMENT]),
            hashlib.md5(payload, usedforsecurity=False).hexdigest().encode("ascii"),
        )
    return repack(parts)


def _written() -> bytes:
    return to_bytes(a_surface())


def _under_a_directory() -> bytes:
    return repack({f"a-scan/{name}": content for name, content in members().items()})


def _stored_without_compression() -> bytes:
    return repack(members(), compression=zipfile.ZIP_STORED)


def _incremental_height_axis() -> bytes:
    """A container whose heights are stored against an increment rather than absolutely.

    The other half of the axis rule. Applying an increment to an absolute axis is
    the seven orders of magnitude defect measured on the one real container, so
    both cases have a fixture and neither is only argued about.
    """
    parts = members()
    parts[DOCUMENT] = parts[DOCUMENT].replace(
        b"<p:CZ><p:AxisType>A</p:AxisType><p:DataType>D</p:DataType><p:Increment>1.0</p:Increment>",
        b"<p:CZ><p:AxisType>I</p:AxisType><p:DataType>D</p:DataType><p:Increment>4.0</p:Increment>",
    )
    scaled = np.ascontiguousarray(a_surface().heights * 1e-6 / 4.0, dtype="<f8").tobytes()
    parts[PAYLOAD] = scaled
    parts[DOCUMENT] = parts[DOCUMENT].replace(
        digest_of(parts[DOCUMENT]),
        hashlib.md5(scaled, usedforsecurity=False).hexdigest().encode("ascii"),
    )
    return repack(parts)


def _a_foreign_comment() -> bytes:
    return changed(b"<p:Comment>", b"<p:Comment>written by another tool, in prose")


def _not_a_container() -> bytes:
    return b"ISO5436_2 but not in an archive"


def _altered_after_packing() -> bytes:
    """A container stored uncompressed and then edited in place.

    The archive's own checksum catches it before this project's does, which is a
    different refusal from a payload whose declared point data checksum
    disagrees, and both are refusals rather than one of them being a crash.
    """
    written = bytearray(_stored_without_compression())
    payload = members()[PAYLOAD]
    written[written.index(payload)] ^= 0xFF
    return bytes(written)


def _declares_a_huge_member() -> bytes:
    """A small archive whose header claims a member no scan has.

    The declared size is patched in the central directory, which is where a
    reader learns how much it is about to allocate. The payload is untouched, so
    this fixture reaches the size refusal and nothing else: a reader that read
    first and checked afterwards would have allocated before it noticed.

    Written uncompressed so the entry offsets are the ones the writer recorded
    and the patch lands where the format says it does.
    """
    written = bytearray(_stored_without_compression())
    marker = b"PK"
    at = written.index(marker)
    while at != -1:
        length = int.from_bytes(written[at + 28 : at + 30], "little")
        if bytes(written[at + 46 : at + 46 + length]) == DOCUMENT.encode("ascii"):
            written[at + 24 : at + 28] = (0xF0000000).to_bytes(4, "little")
            break
        at = written.find(marker, at + 1)
    else:  # pragma: no cover - the writer always records the document
        raise AssertionError("no central directory entry for the metadata document")
    return bytes(written)


def _escaping_entry() -> bytes:
    parts = members()
    parts["../escaped.txt"] = b"outside"
    return repack(parts)


def _absolute_entry() -> bytes:
    parts = members()
    parts["/etc/passwd"] = b"outside"
    return repack(parts)


def _escaping_point_data_link() -> bytes:
    return changed(b"<p:PointDataLink>bindata/data.bin", b"<p:PointDataLink>../../etc/passwd")


def _no_metadata_document() -> bytes:
    parts = members()
    del parts[DOCUMENT]
    return repack(parts)


def _two_metadata_documents() -> bytes:
    parts = members()
    parts["second/main.xml"] = parts[DOCUMENT]
    return repack(parts)


def _not_xml() -> bytes:
    parts = members()
    parts[DOCUMENT] = b"<<< not a document"
    return repack(parts)


def _entity_expansion() -> bytes:
    """The billion laughs shape, refused at the declaration rather than at the use."""
    parts = members()
    parts[DOCUMENT] = (
        b"<?xml version='1.0'?><!DOCTYPE ISO5436_2 [<!ENTITY a 'aaaaaaaaaa'>"
        b"<!ENTITY b '&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;'>]><ISO5436_2>&b;</ISO5436_2>"
    )
    return repack(parts)


def _external_entity() -> bytes:
    parts = members()
    parts[DOCUMENT] = (
        b"<?xml version='1.0'?><!DOCTYPE ISO5436_2 ["
        b"<!ENTITY out SYSTEM 'file:///etc/passwd'>]><ISO5436_2>&out;</ISO5436_2>"
    )
    return repack(parts)


def _inline_point_data() -> bytes:
    return changed(b"<p:DataLink>", b"<p:DataList><p:Datum>1</p:Datum></p:DataList><p:DataLink>")


def _stacked_layers() -> bytes:
    return changed(b"<p:SizeZ>1</p:SizeZ>", b"<p:SizeZ>3</p:SizeZ>")


def _another_payload_type() -> bytes:
    return changed(
        b"<p:CZ><p:AxisType>A</p:AxisType><p:DataType>D</p:DataType>",
        b"<p:CZ><p:AxisType>A</p:AxisType><p:DataType>I16</p:DataType>",
    )


def _unknown_axis_type() -> bytes:
    return changed(b"<p:CZ><p:AxisType>A</p:AxisType>", b"<p:CZ><p:AxisType>Q</p:AxisType>")


def _absent_increment() -> bytes:
    return changed(b"<p:Increment>4e-06</p:Increment>", b"")


def _absent_data_link() -> bytes:
    return changed(b"p:DataLink>", b"p:DataLinkage>")


def _increment_not_a_number() -> bytes:
    return changed(b"<p:Increment>4e-06</p:Increment>", b"<p:Increment>wide</p:Increment>")


def _increment_not_finite() -> bytes:
    return changed(b"<p:Increment>4e-06</p:Increment>", b"<p:Increment>inf</p:Increment>")


def _increment_of_nothing() -> bytes:
    return changed(b"<p:Increment>4e-06</p:Increment>", b"<p:Increment>0.0</p:Increment>")


def _shape_disagrees_with_the_payload() -> bytes:
    return changed(b"<p:SizeY>2</p:SizeY>", b"<p:SizeY>3</p:SizeY>")


def _dimension_not_a_count() -> bytes:
    return changed(b"<p:SizeX>3</p:SizeX>", b"<p:SizeX>three</p:SizeX>")


def _dimension_of_zero() -> bytes:
    """A shape that parses as a number and is not a shape.

    Zero columns multiplies out to a payload of no bytes, so a reader checking
    only that the declared size matched the payload would accept it and hand on
    an array with nothing in it.
    """
    return changed(b"<p:SizeX>3</p:SizeX>", b"<p:SizeX>0</p:SizeX>")


def _point_data_link_names_nothing() -> bytes:
    return changed(b"bindata/data.bin", b"bindata/absent.bin")


def _tampered_payload() -> bytes:
    payload = bytearray(members()[PAYLOAD])
    payload[0] ^= 0x01
    return with_payload(bytes(payload), restate_checksum=False)


def _infinite_height() -> bytes:
    heights = np.array(a_surface().heights, dtype="<f8")
    heights[0, 0] = np.inf
    payload = np.ascontiguousarray(heights * 1e-6, dtype="<f8").tobytes()
    return with_payload(payload, restate_checksum=True)


def _mis_scaled_by_a_thousand() -> bytes:
    """The same surface with millimetres written where the format fixes metres.

    The factor of a thousand. Nothing in the container is malformed: the
    checksum is restated so it agrees, the shape agrees, the axis types are the
    ones the reader supports, and every number parses. Only the size of the
    numbers says anything is wrong, which is why the range check is the only
    place this can be caught and why a reader without one hands it on.

    Built from the fixture surface rather than from a flat one, so it carries the
    absent sample the fixture carries. A range check taking a plain maximum over
    an array holding a not-a-number compares against not-a-number and passes, so
    this container is also what refuses that spelling of the guard.
    """
    heights = np.asarray(a_surface().heights, dtype="<f8")
    payload = np.ascontiguousarray(heights * 1e-6 * 1000.0, dtype="<f8").tobytes()
    return with_payload(payload, restate_checksum=True)


def _every_sample_absent() -> bytes:
    """A container in which nothing was measured at all.

    It reads. An entirely absent surface says nothing about whether the scale is
    right, so the range check has no span to judge and lets it through rather
    than refusing it under a reason about the scale. Filed here so that the
    decision is a container somebody can run and not a sentence in a docstring.
    """
    nothing = np.full((2, 3), np.nan, dtype=np.float64)
    absent = Surface(
        heights=nothing,
        spacing_y=4.0,
        spacing_x=2.5,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="generated-for-a-fixture",
    )
    return to_bytes(absent)


@dataclass(frozen=True)
class Fixture:
    """One container of the corpus, and what it is built to establish."""

    name: str
    #: The refusal this container is built to reach, or nothing where it reads.
    reason: str | None
    build: Callable[[], bytes]


#: The corpus. Every refusal the reader declares appears here at least once, and
#: `test_fixtures.py` refuses a reader that grows one this list does not reach.
FIXTURES: tuple[Fixture, ...] = (
    Fixture("written-by-this-project", None, _written),
    Fixture("entries-under-a-directory", None, _under_a_directory),
    Fixture("stored-without-compression", None, _stored_without_compression),
    Fixture("incremental-height-axis", None, _incremental_height_axis),
    Fixture("a-comment-written-by-another-tool", None, _a_foreign_comment),
    Fixture("every-sample-absent", None, _every_sample_absent),
    Fixture("not-an-archive-at-all", "not-a-container", _not_a_container),
    Fixture("altered-after-packing", "not-a-container", _altered_after_packing),
    Fixture("a-header-declaring-a-huge-member", "entry-too-large", _declares_a_huge_member),
    Fixture("an-entry-climbing-out", "entry-outside-the-archive", _escaping_entry),
    Fixture("an-entry-naming-a-root", "entry-outside-the-archive", _absolute_entry),
    Fixture(
        "a-point-data-link-climbing-out",
        "entry-outside-the-archive",
        _escaping_point_data_link,
    ),
    Fixture("no-main-xml", "no-metadata-document", _no_metadata_document),
    Fixture("two-main-xml", "several-metadata-documents", _two_metadata_documents),
    Fixture("a-document-that-is-not-xml", "not-xml", _not_xml),
    Fixture("an-entity-expansion", "document-type-declaration", _entity_expansion),
    Fixture("an-external-entity", "document-type-declaration", _external_entity),
    Fixture("point-data-carried-inline", "recognised-and-unsupported", _inline_point_data),
    Fixture("a-stack-of-layers", "recognised-and-unsupported", _stacked_layers),
    Fixture("another-payload-type", "recognised-and-unsupported", _another_payload_type),
    Fixture("an-unknown-axis-type", "recognised-and-unsupported", _unknown_axis_type),
    Fixture("an-axis-with-no-increment", "element-missing", _absent_increment),
    Fixture("no-data-link-at-all", "element-missing", _absent_data_link),
    Fixture("a-link-naming-nothing", "element-missing", _point_data_link_names_nothing),
    Fixture("an-increment-that-is-a-word", "not-a-number", _increment_not_a_number),
    Fixture("an-increment-that-is-infinite", "not-a-number", _increment_not_finite),
    Fixture("a-dimension-that-is-a-word", "not-a-number", _dimension_not_a_count),
    Fixture("a-dimension-of-zero", "not-a-number", _dimension_of_zero),
    Fixture("an-increment-of-nothing", "increment-not-positive", _increment_of_nothing),
    Fixture(
        "a-declared-shape-one-row-too-tall",
        "size-disagrees-with-the-payload",
        _shape_disagrees_with_the_payload,
    ),
    Fixture("one-flipped-payload-bit", "point-data-checksum-mismatch", _tampered_payload),
    Fixture("an-infinite-height", "height-not-finite", _infinite_height),
    Fixture(
        "a-surface-mis-scaled-by-a-thousand",
        "height-range-implausible",
        _mis_scaled_by_a_thousand,
    ),
)
