"""Reading an X3P container into a surface, and refusing what should not be read.

This is where bytes from outside the project enter it. A zip holding an XML
document holding a binary array is three parsers stacked, and every one of them
has a standard way of being attacked, so the refusals here are not defensive
programming in general: each names a construct and says why it is refused.

## What the format actually does, measured rather than read off the standard

One real container was read to establish this, `csafe-logo.x3p` from the
`x3ptools` R package. It is not from the public database and is not a cartridge
case; it is a container an independent and widely used tool wrote, so it says
what a real writer emits. The evidence is on
[#33](https://github.com/iderex/gutachten/issues/33) with the commands that
produced it. Four things came out of it and all four decide code here.

**The entries may sit under a directory rather than at the archive root.** So the
metadata document is found by looking for a member whose name ends in `main.xml`
at any depth, and the payload is resolved relative to that member rather than to
the root. Two such members are refused rather than one of them being chosen.

**The namespace is declared with a prefix nothing uses.** In that file every
element is in no namespace at all, while a correctly written file would put them
in the ISO 5436-2 namespace. A reader matching on the namespaced name finds
nothing in the first and a reader ignoring namespaces reads both, so this one
ignores them: expat is created with namespace processing off and any prefix is
stripped from the element name.

**Lengths are metres and no file declares a unit.** The format fixes it. So a
refusal aimed at a missing unit element would never fire, and what is refused
here instead is an increment that is absent, not a number, or not positive. That
the unit is not declared anywhere in the format is the finding behind
[#45](https://github.com/iderex/gutachten/issues/45), and the implausible height
range check that issue asks for is not in this module yet.

**An absolute height axis carries the quantity and its increment is not a scale
factor.** Multiplying by the increment as well produced a surface that was
smooth, correctly shaped and wrong by seven orders of magnitude on that file,
which is exactly the defect class this project exists to catch. So the axis type
is read and the two cases are handled apart.

## The two checksums are not the same check

`Record3/DataLink/MD5ChecksumPointData` is the md5 of the payload member, it
verified exactly on the file measured, and it is the one that guards the numbers.
A mismatch is refused here with both values in the message.

The container level `md5checksum.hex` did **not** verify against `main.xml` on
that file, and what it is the md5 of was not established. One file is not enough
to say whether that writer is wrong or the reading is, so this reader does not
refuse on it and does not claim to have checked it. Refusing would reject a
container written by the most widely used tool in the field on the strength of a
single unexplained observation.

## What comes out

A `gutachten.surface.Surface` in micrometres. The heights are converted from the
metres the format stores, and the conversion is where a factor of a million lives
exactly once.

The format declares no axis orientation, and `Surface` requires one. This reader
records `Y_DOWN`, which is what an instrument writing an array in image order
means, and that is an assumption rather than something read out of the file.
Recording it is the whole point: two surfaces that disagree about it are then a
refusable condition rather than a low score. #45 is where the orientation
question is argued.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final
from xml.parsers import expat

import numpy as np

from gutachten.surface import AxisOrientation, LengthUnit, Surface, TransformRecord
from gutachten.x3p.writer import METADATA_KEY

__all__ = ["MAX_ENTRY_BYTES", "REASONS", "X3PError", "read", "read_bytes"]

#: Every reason this reader refuses a container. A closed vocabulary rather than
#: prose, so a test asserts that a fixture was refused for the reason it was
#: built to reach rather than for some other error on the way there.
REASONS: Final[tuple[str, ...]] = (
    "not-a-container",
    "entry-outside-the-archive",
    "entry-too-large",
    "no-metadata-document",
    "several-metadata-documents",
    "not-xml",
    "document-type-declaration",
    "recognised-and-unsupported",
    "element-missing",
    "not-a-number",
    "increment-not-positive",
    "size-disagrees-with-the-payload",
    "point-data-checksum-mismatch",
    "height-not-finite",
)

#: The largest member this reader will decompress. A scan of 4000 by 4000
#: samples at eight bytes each is 128 megabytes, so this admits one four times
#: that area and refuses the declared size of an archive that claims to hold a
#: terabyte. It is a bound on what an attacker can make this process allocate,
#: not a statement about what a scan is.
MAX_ENTRY_BYTES: Final[int] = 512 * 1024 * 1024

#: What the format stores lengths in. Written here once, because the conversion
#: to the internal unit is the most expensive arithmetic error available in this
#: project and it should exist in exactly one place.
_FILE_UNIT: Final[LengthUnit] = LengthUnit.METRE


class X3PError(Exception):
    """A container this reader will not read, with the reason it will not.

    ``reason`` comes from ``REASONS`` and ``detail`` is the sentence a person
    reads. A test asserts on the reason so that a fixture built to reach one
    refusal cannot pass by tripping another one earlier.
    """

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in REASONS:
            raise ValueError(f"{reason!r} is not one of this reader's reasons: {list(REASONS)}")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class _Element:
    """One element of the metadata document, with its prefix already stripped."""

    tag: str
    text: str
    children: tuple[_Element, ...]

    def child(self, *path: str) -> _Element | None:
        found: _Element | None = self
        for name in path:
            if found is None:
                return None
            matching = [item for item in found.children if item.tag == name]
            found = matching[0] if len(matching) == 1 else None
        return found

    def required(self, *path: str) -> _Element:
        found = self.child(*path)
        if found is None:
            raise X3PError(
                "element-missing",
                f"the metadata document has no single {'/'.join(path)} under {self.tag!r}. "
                "A container missing it describes a surface this reader cannot assemble, "
                "and assembling one anyway would put a guess where a measurement goes.",
            )
        return found


class _Builder:
    """Collects the document expat walks, refusing the constructs it must not read."""

    def __init__(self) -> None:
        self.root: _Element | None = None
        self._stack: list[tuple[str, list[str], list[_Element]]] = []

    def start(self, name: str, _attributes: dict[str, str]) -> None:
        self._stack.append((name.rsplit(":", 1)[-1], [], []))

    def characters(self, data: str) -> None:
        if self._stack:
            self._stack[-1][1].append(data)

    def end(self, _name: str) -> None:
        tag, text, children = self._stack.pop()
        element = _Element(tag=tag, text="".join(text).strip(), children=tuple(children))
        if self._stack:
            self._stack[-1][2].append(element)
        else:
            self.root = element


def _refuse_doctype(*_arguments: object) -> None:
    raise X3PError(
        "document-type-declaration",
        "the metadata document carries a document type declaration. That is where an "
        "entity expansion or an external entity reference is introduced, both of which "
        "turn reading an evidence file into fetching whatever the file names, and no "
        "container this project accepts needs one.",
    )


def _parse(document: bytes) -> _Element:
    """Parse the metadata document with entities and external references refused.

    Namespace processing is off, so an element written under a prefix arrives
    with the prefix attached and one written under none arrives bare. The prefix
    is stripped in the builder. Both real spellings of this format's documents
    therefore read the same way.
    """
    builder = _Builder()
    parser = expat.ParserCreate()
    # The handler slots take callables of three different shapes and all three
    # are being replaced by one refusal, so they are set through a plain value.
    # What matters is that they are set before the first byte is fed.
    slots: Any = parser
    slots.StartDoctypeDeclHandler = _refuse_doctype
    slots.EntityDeclHandler = _refuse_doctype
    slots.ExternalEntityRefHandler = _refuse_doctype
    parser.StartElementHandler = builder.start
    parser.EndElementHandler = builder.end
    parser.CharacterDataHandler = builder.characters
    try:
        parser.Parse(document, True)
    except expat.ExpatError as broken:
        raise X3PError(
            "not-xml", f"the metadata document is not well formed XML: {broken}"
        ) from broken
    if builder.root is None:  # pragma: no cover - expat refuses a document with no root
        # Kept rather than asserted. An assertion is removed under -O and this
        # is the branch that would then hand a None to the caller.
        raise X3PError("not-xml", "the metadata document holds no element at all")
    return builder.root


def _safe_name(name: str) -> str:
    """The archive member name, refused where it does not stay inside the archive."""
    if name.startswith("/") or name.startswith("\\") or ":" in name:
        raise X3PError(
            "entry-outside-the-archive",
            f"the archive holds an entry named {name!r}, which names an absolute location "
            "rather than a place inside the archive. An extractor following it writes "
            "outside the directory it was pointed at.",
        )
    parts = name.replace("\\", "/").split("/")
    if ".." in parts:
        raise X3PError(
            "entry-outside-the-archive",
            f"the archive holds an entry named {name!r}, which climbs out of the archive. "
            "This reader never writes an entry to disk, and refuses the name anyway: a "
            "container carrying one was not written to be read.",
        )
    return name.replace("\\", "/")


def _member(archive: zipfile.ZipFile, name: str) -> bytes:
    """One member's bytes, refusing a declared size no scan has.

    The declared size is read before anything is decompressed, which is the whole
    point: an archive of a few kilobytes can declare a member of a terabyte, and a
    reader that finds out by allocating it has already lost.

    A header that lies the other way, declaring less than the member holds, is
    caught by the archive's own checksum when the member is read, and that
    arrives here as a container that could not be read.
    """
    info = archive.getinfo(name)
    if info.file_size > MAX_ENTRY_BYTES:
        raise X3PError(
            "entry-too-large",
            f"the entry {name!r} declares {info.file_size} bytes and this reader admits "
            f"{MAX_ENTRY_BYTES}. A declared size beyond any scan is how a small archive "
            "is made to allocate a large amount of memory.",
        )
    try:
        with archive.open(info) as handle:
            return handle.read()
    except (zipfile.BadZipFile, EOFError) as broken:
        raise X3PError(
            "not-a-container",
            f"the entry {name!r} could not be read out of the archive: {broken}",
        ) from broken


def _number(element: _Element, what: str) -> float:
    try:
        value = float(element.text)
    except ValueError as broken:
        raise X3PError(
            "not-a-number", f"{what} is {element.text!r}, which is not a number"
        ) from broken
    if not np.isfinite(value):
        raise X3PError("not-a-number", f"{what} is {element.text!r}, which is not finite")
    return value


def _increment(axes: _Element, axis: str) -> float:
    """One axis increment, in metres, refused where it could not be a spacing."""
    value = _number(axes.required(axis, "Increment"), f"the {axis} increment")
    if value <= 0.0:
        raise X3PError(
            "increment-not-positive",
            f"the {axis} increment is {value!r}. It is the distance between neighbouring "
            "samples, and a spacing of nothing or less is not one. The format states no "
            "unit anywhere, so this is the only thing about the axis a reader can check.",
        )
    return value


def _size(dimension: _Element, axis: str) -> int:
    text = dimension.required(axis).text
    try:
        value = int(text)
    except ValueError as broken:
        raise X3PError(
            "not-a-number", f"the matrix dimension {axis} is {text!r}, which is not a count"
        ) from broken
    if value < 1:
        raise X3PError("not-a-number", f"the matrix dimension {axis} is {value}")
    return value


def _heights(payload: bytes, record: _Element, axes: _Element) -> np.ndarray:
    """The height array, in the metres the format stores, refusing the shapes it is not."""
    dimension = record.required("MatrixDimension")
    size_x, size_y, size_z = (_size(dimension, name) for name in ("SizeX", "SizeY", "SizeZ"))
    if size_z != 1:
        raise X3PError(
            "recognised-and-unsupported",
            f"the container declares SizeZ={size_z}. A stack of layers is a construct this "
            "format allows and this reader does not support; a surface here is one height "
            "per point.",
        )

    kind = axes.required("CZ", "DataType").text
    if kind != "D":
        raise X3PError(
            "recognised-and-unsupported",
            f"the height axis declares the data type {kind!r}. This reader supports {'D'!r}, "
            "which is the double precision payload the public database and the common "
            "instruments emit. The construct is recognised and is not supported, which is a "
            "different statement from not having understood the file.",
        )

    expected = size_x * size_y * 8
    if len(payload) != expected:
        raise X3PError(
            "size-disagrees-with-the-payload",
            f"the container declares {size_y} by {size_x} points, which is {expected} bytes "
            f"of double precision payload, and the payload is {len(payload)} bytes. Reading "
            "the shorter of the two would produce a surface of the declared shape carrying "
            "somebody else's numbers at the end of it.",
        )

    stored = np.frombuffer(payload, dtype="<f8").reshape(size_y, size_x).astype(np.float64)
    axis_type = axes.required("CZ", "AxisType").text
    if axis_type == "A":
        # Absolute: the stored value is the quantity. Its increment is not a
        # scale factor and applying one is the seven-orders-of-magnitude defect
        # measured on the file this reader was established against.
        heights = stored
    elif axis_type == "I":
        heights = stored * _increment(axes, "CZ")
    else:
        raise X3PError(
            "recognised-and-unsupported",
            f"the height axis declares the type {axis_type!r}. This reader supports "
            "an absolute axis and an incremental one, and nothing else.",
        )

    if np.isinf(heights).any():
        raise X3PError(
            "height-not-finite",
            "the payload carries an infinite height. An absent measurement is not-a-number "
            "and an infinity is neither a measurement nor an absence, so there is nothing "
            "this reader could record it as.",
        )
    return heights


#: The default orientation, and it is an assumption rather than a reading. The
#: format has no element for it, so a container written by anything but this
#: project says nothing about which way its row axis increases, and `Surface`
#: requires an answer.
_ASSUMED_ORIENTATION: Final[AxisOrientation] = AxisOrientation.Y_DOWN


def _declared(root: _Element) -> tuple[AxisOrientation, tuple[TransformRecord, ...]]:
    """The orientation and the provenance chain a container of ours carries.

    ``Record2/Comment`` is a free string in this format, so a comment written by
    anything else is prose and is left alone. This project's own containers put a
    JSON object there under one key, and only that shape is read.

    What a container says about its own provenance is a claim by whoever wrote
    it. It is recorded as read and never as verified, and nothing in this project
    treats a chain out of a file as evidence that those steps ran.
    """
    comment = root.child("Record2", "Comment")
    if comment is None or not comment.text.startswith("{"):
        return _ASSUMED_ORIENTATION, ()
    try:
        parsed = json.loads(comment.text)
        declared = parsed[METADATA_KEY]
        orientation = AxisOrientation(declared["orientation"])
        chain = tuple(
            TransformRecord(
                name=step["name"],
                version=step["version"],
                parameters=tuple(sorted(step["parameters"].items())),
                outcomes=tuple(sorted(step["outcomes"].items())),
            )
            for step in declared["provenance"]
        )
    except (json.JSONDecodeError, TypeError, KeyError, ValueError, AttributeError):
        # A comment that starts like this project's own and is not it. Reading
        # half of it would put an orientation nobody wrote into a surface, which
        # is worse than the documented assumption.
        return _ASSUMED_ORIENTATION, ()
    return orientation, chain


def read_bytes(data: bytes, source: str) -> Surface:
    """Read a container held in memory, under the identity ``source``.

    The identity is handed in rather than derived. A container arriving over a
    fetch has a name in the database it came from and none on this machine, and a
    surface with no traceable source is one nobody can check.
    """
    if not source:
        raise ValueError("a surface read from a container needs a source identity")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as broken:
        raise X3PError(
            "not-a-container", f"the file is not a readable zip archive: {broken}"
        ) from broken

    names = [_safe_name(info.filename) for info in archive.infolist()]
    documents = [name for name in names if PurePosixPath(name).name.lower() == "main.xml"]
    if not documents:
        raise X3PError(
            "no-metadata-document",
            "the archive holds no main.xml at any depth. Every X3P container carries its "
            "axis definitions there, and there is nothing else in the archive that says "
            "what the payload means.",
        )
    if len(documents) > 1:
        raise X3PError(
            "several-metadata-documents",
            f"the archive holds {sorted(documents)}. Choosing one of them would make the "
            "surface depend on the order the archive happens to list its entries in.",
        )

    document = documents[0]
    root = _parse(_member(archive, document))
    axes = root.required("Record1", "Axes")
    record3 = root.required("Record3")
    if record3.child("DataList") is not None:
        raise X3PError(
            "recognised-and-unsupported",
            "the container carries its point data inline as a DataList. That is a form the "
            "format allows and this reader does not support; it reads the binary payload a "
            "DataLink names.",
        )

    link = record3.required("DataLink", "PointDataLink").text
    payload_name = _safe_name(posixpath.join(posixpath.dirname(document), link))
    if payload_name not in names:
        raise X3PError(
            "element-missing",
            f"the container names {link!r} as its point data and the archive holds no such entry.",
        )
    payload = _member(archive, payload_name)

    declared = record3.required("DataLink", "MD5ChecksumPointData").text.strip().lower()
    # usedforsecurity=False because this is an integrity check the format
    # specifies, not a security primitive, and saying so keeps the reader working
    # on a build where the weak digests are restricted.
    found = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    if declared != found:
        raise X3PError(
            "point-data-checksum-mismatch",
            f"the container declares the point data checksum {declared!r} and the payload "
            f"hashes to {found!r}. A silently corrupted evidence file that produces a "
            "plausible score is the worst outcome available here.",
        )

    heights = _heights(payload, record3, axes)
    factor = _FILE_UNIT.micrometres
    orientation, provenance = _declared(root)
    return Surface(
        heights=heights * factor,
        spacing_y=_increment(axes, "CY") * factor,
        spacing_x=_increment(axes, "CX") * factor,
        unit=LengthUnit.MICROMETRE,
        orientation=orientation,
        source=source,
        provenance=provenance,
    )


def read(path: Path) -> Surface:
    """Read a container from disk, under the identity of its file name."""
    return read_bytes(path.read_bytes(), source=path.name)
