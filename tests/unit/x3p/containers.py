"""Containers built here, so the reader is tested against files nobody downloaded.

Every fixture below starts as a container this project's own writer produced and
is then changed in exactly one way. That shape is deliberate. A malformed
container assembled from scratch tends to be malformed in several ways at once,
and a test over one cannot say which of them the refusal was about, which is the
difference between a fixture that proves a refusal and one that proves the reader
raised something.

The accepted container comes from the writer rather than from a stored file, so
the two sides of the round trip are the two sides that ship. Nothing here reaches
a network and nothing is stored in the tree.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np

from gutachten.surface import AxisOrientation, LengthUnit, Surface, TransformRecord
from gutachten.x3p.writer import DOCUMENT, PAYLOAD, to_bytes


#: The surface every fixture is built from. Small, with one absent measurement,
#: because absence is the thing a reader is most likely to lose.
def a_surface() -> Surface:
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


def repack(members: dict[str, bytes]) -> bytes:
    """A zip holding exactly these members, in the order given."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return buffer.getvalue()


def changed(before: bytes, after: bytes, surface: Surface | None = None) -> bytes:
    """A container whose metadata document has ``before`` replaced by ``after``.

    The replacement is asserted to have changed something. A fixture built on a
    string the document does not carry is a fixture that reads as valid and
    proves the reader accepts a file, which is the opposite of what it was
    written for.
    """
    parts = members(surface)
    document = parts[DOCUMENT]
    if before not in document:
        raise AssertionError(f"{before!r} is not in the metadata document, so nothing changed")
    parts[DOCUMENT] = document.replace(before, after)
    return repack(parts)


def with_payload(payload: bytes) -> bytes:
    """A container whose point data has been replaced, leaving the metadata alone."""
    parts = members()
    parts[PAYLOAD] = payload
    return repack(parts)
