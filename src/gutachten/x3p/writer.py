"""Writing a surface back out as an X3P container.

A project that argues for openness and can only consume the open format would be
an odd shape. Three things need this: emitting a preprocessed surface so somebody
can open it and see what the preprocessing actually did, producing the
conformance fixtures the reader is tested against, and letting a result be
re-opened in the tools other people already run.

## What is written, and why in this shape

`main.xml` at the archive root, `bindata/data.bin` beside it, and
`md5checksum.hex`. The root layout is what the format's own description puts
there. The reader in this package also reads a container whose entries sit under
a directory, because a widely used writer emits that, but there is no reason to
produce one.

The elements are written in the ISO 5436-2 namespace under a prefix that is
actually used. The one real container measured for the reader declares the
namespace and then uses no prefix at all, so every element in it is in no
namespace; a reader matching on the namespaced name finds nothing in that file.
Writing the correct form and reading both is the only combination that does not
propagate somebody else's defect.

The height axis is written as absolute, so the stored value is the height itself
and no increment is applied to it. That is the case the reader was established
against and it is the one where an increment applied twice cannot happen.

An absent measurement is written as not-a-number in the payload. The one
container examined carried no invalid point, so how that writer encodes absence
was not established, and any other encoding here would be invented rather than
observed. This is a disclosure and not a claim of conformance.

## What this project's own containers carry that the format has no room for

`Record2/Comment` holds a JSON object under one key. It carries the provenance
chain, so a preprocessed surface names the transforms that produced it rather
than arriving anonymous, and the axis orientation, because the format declares
none and `Surface` requires one.

A container from anywhere else has no such object, and the reader then records
the orientation it documents as an assumption. What a container says about its
own provenance is a claim by whoever wrote it. The reader records it as read
rather than as verified, and no check in this project treats a chain out of a
file as evidence that those steps ran.

## The container is byte identical when written twice

Every member is stored with a fixed timestamp and a fixed compression setting,
because a zip otherwise carries the clock of the machine that made it. Two runs
of one pipeline would then produce two different files holding one surface, which
breaks a content addressed cache and a golden comparison at once, for a reason
that has nothing to do with the surface.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import escape

import numpy as np

from gutachten.surface import LengthUnit, Surface

__all__ = [
    "DOCUMENT",
    "METADATA_KEY",
    "NAMESPACE",
    "PAYLOAD",
    "to_bytes",
    "write",
]

#: The ISO 5436-2 namespace, bound to a prefix this writer actually uses.
NAMESPACE: Final[str] = "http://www.opengps.eu/2008/ISO5436_2"

#: Where the two members live in an archive this writer produces.
DOCUMENT: Final[str] = "main.xml"
PAYLOAD: Final[str] = "bindata/data.bin"
_CHECKSUM: Final[str] = "md5checksum.hex"

#: The one key in `Record2/Comment` under which this project's own metadata sits.
#: One key rather than a bare object, so a comment written by anything else is
#: recognisably not this and is left alone.
METADATA_KEY: Final[str] = "gutachten"

#: Fixed so a container written twice is the same bytes. The earliest timestamp
#: a zip can carry, which is a value nobody will mistake for a real one.
_TIMESTAMP: Final[tuple[int, int, int, int, int, int]] = (1980, 1, 1, 0, 0, 0)

_FILE_UNIT: Final[LengthUnit] = LengthUnit.METRE


def _element(tag: str, text: str) -> str:
    return f"<p:{tag}>{escape(text)}</p:{tag}>"


def _metadata(surface: Surface) -> str:
    """This project's own record, as the JSON that goes into the comment."""
    chain: list[dict[str, Any]] = [
        {
            "name": record.name,
            "version": record.version,
            "parameters": dict(record.parameters),
            "outcomes": dict(record.outcomes),
        }
        for record in surface.provenance
    ]
    return json.dumps(
        {METADATA_KEY: {"orientation": surface.orientation.value, "provenance": chain}},
        sort_keys=True,
    )


def _document(surface: Surface, digest: str) -> bytes:
    """The metadata document, in metres, describing the payload beside it."""
    rows, columns = surface.shape
    factor = surface.unit.micrometres / _FILE_UNIT.micrometres
    axes = "".join(
        [
            "<p:CX>"
            + _element("AxisType", "I")
            + _element("DataType", "D")
            + _element("Increment", repr(surface.spacing_x * factor))
            + _element("Offset", repr(0.0))
            + "</p:CX>",
            "<p:CY>"
            + _element("AxisType", "I")
            + _element("DataType", "D")
            + _element("Increment", repr(surface.spacing_y * factor))
            + _element("Offset", repr(0.0))
            + "</p:CY>",
            # Absolute, so the stored value is the height. The increment is
            # written because the format's schema carries one and it is not a
            # scale factor on this axis; the reader ignores it here for exactly
            # that reason.
            "<p:CZ>"
            + _element("AxisType", "A")
            + _element("DataType", "D")
            + _element("Increment", repr(1.0))
            + _element("Offset", repr(0.0))
            + "</p:CZ>",
        ]
    )
    body = "".join(
        [
            f'<?xml version="1.0" encoding="UTF-8"?><p:ISO5436_2 xmlns:p="{NAMESPACE}">',
            f"<p:Record1><p:Revision>ISO5436 - 2000</p:Revision>"
            f"<p:FeatureType>SUR</p:FeatureType><p:Axes>{axes}</p:Axes></p:Record1>",
            "<p:Record2>"
            + _element("Creator", "gutachten")
            + _element("Comment", _metadata(surface))
            + "</p:Record2>",
            "<p:Record3><p:MatrixDimension>"
            + _element("SizeX", str(columns))
            + _element("SizeY", str(rows))
            + _element("SizeZ", "1")
            + "</p:MatrixDimension><p:DataLink>"
            + _element("PointDataLink", PAYLOAD)
            + _element("MD5ChecksumPointData", digest)
            + "</p:DataLink></p:Record3>",
            "<p:Record4>" + _element("ChecksumFile", _CHECKSUM) + "</p:Record4>",
            "</p:ISO5436_2>",
        ]
    )
    return body.encode("utf-8")


def to_bytes(surface: Surface) -> bytes:
    """The container holding ``surface``, as bytes.

    The heights are converted to the metres the format stores. That conversion is
    the reason a round trip is equal within a tolerance rather than exactly: the
    factor is not a power of two and the arithmetic is binary.
    """
    factor = surface.unit.micrometres / _FILE_UNIT.micrometres
    payload = np.ascontiguousarray(surface.heights * factor, dtype="<f8").tobytes()
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    document = _document(surface, digest)
    checksum = hashlib.md5(document, usedforsecurity=False).hexdigest() + "\n"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in (
            (DOCUMENT, document),
            (PAYLOAD, payload),
            (_CHECKSUM, checksum.encode("ascii")),
        ):
            info = zipfile.ZipInfo(filename=name, date_time=_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return buffer.getvalue()


def write(surface: Surface, path: Path) -> None:
    """Write ``surface`` as a container at ``path``."""
    path.write_bytes(to_bytes(surface))
