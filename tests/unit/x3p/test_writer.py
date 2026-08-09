"""Write then read returns the surface, over generated cases rather than examples.

The round trip is the property that matters, and it is checked over generated
surfaces because the cases that break a container writer are the ones nobody
thinks to write down: a single row, a single column, an array with no measurement
in it at all, an aspect ratio no scan has. A handful of hand written examples
covers the shapes whoever wrote them already had in mind.

The equality is within a declared tolerance rather than exact, and the reason is
one multiplication. The format stores lengths in metres and this project works in
micrometres, so a height goes out through a factor of a millionth and comes back
through a factor of a million. Neither is a power of two.
"""

from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from gutachten.surface import AxisOrientation, LengthUnit, Surface, TransformRecord
from gutachten.x3p.reader import MAX_HEIGHT_RANGE_MICROMETRES, X3PError, read_bytes
from gutachten.x3p.writer import DOCUMENT, PAYLOAD, to_bytes, write
from tests.support.tolerance import assert_close
from tests.unit.x3p.containers import a_surface

#: What a round trip is compared within, stated at every call site below. Two
#: roundings of a double give a relative error of a few times 1e-16; 1e-12 is
#: four orders above that and far below the resolution of any instrument whose
#: output this project will read. The absolute floor is a femtometre, which is
#: there so a height of nearly zero is compared meaningfully rather than against
#: a relative tolerance of nothing.
RTOL = 1e-12
ATOL = 1e-9

#: Half of the widest surface the reader will read, less a micrometre, so that
#: no drawn pair of heights spans more than the reader admits even after the
#: conversion out to metres and back has moved the last bits.
WIDEST = MAX_HEIGHT_RANGE_MICROMETRES / 2.0 - 1.0

#: A bounded height, or an absent measurement, drawn as two strategies because
#: hypothesis refuses to conflate a bounded float with a not-a-number. The bound
#: is there because the tolerance above is a statement about doubles rather than
#: about the whole float range: a height of 1e300 micrometres is not a surface,
#: and generating one would test the arithmetic of the conversion rather than the
#: container.
#:
#: It is now also what the reader admits. Since the range check landed, a
#: container whose measured samples span more than
#: `MAX_HEIGHT_RANGE_MICROMETRES` is refused, so a surface drawn wider than that
#: is one the writer produces and the reader declines. The property under test is
#: the round trip, so the strategy stays inside the readable range and the
#: asymmetry itself is asserted separately below rather than left as the reason a
#: generated case fails.
HEIGHTS = st.one_of(
    st.floats(
        min_value=-WIDEST,
        max_value=WIDEST,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ),
    st.just(float("nan")),
)
SPACINGS = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def surfaces(draw: st.DrawFn) -> Surface:
    """A surface of any small shape, with any pattern of absent measurements."""
    rows = draw(st.integers(min_value=1, max_value=12))
    columns = draw(st.integers(min_value=1, max_value=12))
    heights = draw(npst.arrays(dtype=np.float64, shape=(rows, columns), elements=HEIGHTS))
    return Surface(
        heights=heights,
        spacing_y=draw(SPACINGS),
        spacing_x=draw(SPACINGS),
        unit=LengthUnit.MICROMETRE,
        orientation=draw(st.sampled_from(list(AxisOrientation))),
        source="generated",
    )


@given(surface=surfaces())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_write_then_read_returns_the_surface(surface: Surface) -> None:
    back = read_bytes(to_bytes(surface), source=surface.source)

    assert back.shape == surface.shape
    assert back.orientation is surface.orientation
    assert (back.missing == surface.missing).all()
    assert_close(
        back.observed, surface.observed, what="heights through a round trip", atol=ATOL, rtol=RTOL
    )
    assert_close(
        [back.spacing_y, back.spacing_x],
        [surface.spacing_y, surface.spacing_x],
        what="spacings through a round trip",
        atol=ATOL,
        rtol=RTOL,
    )


def test_a_surface_with_no_measurement_at_all_round_trips() -> None:
    """The case a writer loses by treating absence as a value it can skip."""
    surface = Surface(
        heights=np.full((3, 4), np.nan),
        spacing_y=1.0,
        spacing_x=1.0,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="all-missing",
    )
    back = read_bytes(to_bytes(surface), source="all-missing")
    assert back.missing.all()
    assert back.shape == (3, 4)


def test_a_single_row_and_a_single_column_round_trip() -> None:
    """The aspect ratios where a row major layout is read the wrong way round.

    A reader that transposed would pass every square fixture and fail here, which
    is why the two are separate rather than one generated case that might not be
    drawn.
    """
    for shape in ((1, 9), (9, 1)):
        heights = np.arange(9, dtype=np.float64).reshape(shape)
        surface = Surface(
            heights=heights,
            spacing_y=2.0,
            spacing_x=3.0,
            unit=LengthUnit.MICROMETRE,
            orientation=AxisOrientation.Y_DOWN,
            source="thin",
        )
        back = read_bytes(to_bytes(surface), source="thin")
        assert back.shape == shape
        assert_close(back.heights, heights, what=f"a {shape} surface", atol=ATOL, rtol=RTOL)


def test_the_two_spacings_are_not_interchangeable() -> None:
    """A writer that swapped the axes would round trip a square surface unmoved."""
    surface = a_surface()
    back = read_bytes(to_bytes(surface), source="asymmetric")
    assert back.spacing_y != back.spacing_x
    assert_close(
        [back.spacing_y], [surface.spacing_y], what="the row spacing", atol=ATOL, rtol=RTOL
    )


def test_the_provenance_chain_survives_the_round_trip() -> None:
    """A preprocessed surface names the transforms that produced it.

    The format has no element for a chain, so this project's own containers carry
    it in the comment. What comes back is a claim by whoever wrote the file and
    is recorded as read rather than as verified, which is the reader's docstring
    and not something this test can check.
    """
    chain = (
        TransformRecord.of("trim-edge", "1", width=40.0, criterion="frame"),
        TransformRecord.of("level", "1", model="plane", order=None).with_outcomes(removed=3),
    )
    surface = Surface(
        heights=np.array([[1.0, 2.0], [3.0, 4.0]]),
        spacing_y=1.0,
        spacing_x=1.0,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_UP,
        source="preprocessed",
        provenance=chain,
    )
    assert read_bytes(to_bytes(surface), source="preprocessed").provenance == chain


def test_the_written_container_declares_the_checksum_of_its_own_payload() -> None:
    """The check the reader refuses on, asserted from the other side.

    A writer whose declared checksum did not match what it wrote would make every
    container it produced unreadable by this project's own reader, and the round
    trip test above would say so without saying why.
    """
    with zipfile.ZipFile(io.BytesIO(to_bytes(a_surface()))) as archive:
        document = archive.read(DOCUMENT).decode("utf-8")
        payload = archive.read(PAYLOAD)
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    assert f"<p:MD5ChecksumPointData>{digest}</p:MD5ChecksumPointData>" in document


def test_the_container_carries_the_checksum_file_the_format_names() -> None:
    """Written, and deliberately not enforced when read.

    The one real container measured for the reader does not verify under the
    reading its own Record4 gives, and what its checksum file is the md5 of was
    not established. So this writes the reading the format states and the reader
    refuses nothing on it, and both halves are disclosed where they are.
    """
    with zipfile.ZipFile(io.BytesIO(to_bytes(a_surface()))) as archive:
        names = archive.namelist()
        stated = archive.read("md5checksum.hex").decode("ascii").strip()
        document = archive.read(DOCUMENT)
    assert names == [DOCUMENT, PAYLOAD, "md5checksum.hex"]
    assert stated == hashlib.md5(document, usedforsecurity=False).hexdigest()


def test_a_container_written_twice_is_the_same_bytes() -> None:
    """A zip carries the clock of the machine that made it unless it is told not to.

    Two runs of one pipeline would otherwise produce two files holding one
    surface, which breaks a content addressed cache and a golden comparison at
    once, for a reason that has nothing to do with the surface.
    """
    surface = a_surface()
    assert to_bytes(surface) == to_bytes(surface)


def test_writing_to_a_path_writes_the_same_bytes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    surface = a_surface()
    path = tmp_path / "written.x3p"
    write(surface, path)
    assert path.read_bytes() == to_bytes(surface)
    assert read_bytes(path.read_bytes(), source="from-disk").shape == surface.shape


def test_a_surface_in_another_unit_round_trips_to_the_same_measurement() -> None:
    """The reader converts to one internal unit, so the unit field does not survive.

    What survives is the measurement. A surface declared in millimetres comes
    back in micrometres holding the same lengths, which is the conversion
    happening once at the reader rather than in every function downstream.
    """
    surface = Surface(
        heights=np.array([[0.001, 0.002], [0.003, 0.004]]),
        spacing_y=0.004,
        spacing_x=0.0025,
        unit=LengthUnit.MILLIMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="in-millimetres",
    )
    back = read_bytes(to_bytes(surface), source="in-millimetres")
    assert back.unit is LengthUnit.MICROMETRE
    assert_close(
        back.observed,
        surface.observed * LengthUnit.MILLIMETRE.micrometres,
        what="millimetres read back as micrometres",
        atol=ATOL,
        rtol=RTOL,
    )
    assert_close(
        [back.spacing_x],
        [surface.spacing_x * LengthUnit.MILLIMETRE.micrometres],
        what="a millimetre spacing read back as micrometres",
        atol=ATOL,
        rtol=RTOL,
    )


def test_one_measurement_declared_in_two_units_reads_to_one_surface() -> None:
    """The same lengths written from millimetres and from micrometres agree.

    The format carries no unit element at all, so the only place a unit exists
    is on the surface handed to the writer, and this is the strongest form the
    clause in [#45] can take here.

    The two containers are not byte for byte identical and are not asserted to
    be. A millimetre converted to metres and a micrometre converted to metres
    reach the same number by two different multiplications, neither factor is a
    power of two, and the results differ in the last bits. What is asserted is
    the measurement, within the tolerance stated at the call site.

    [#45]: https://github.com/iderex/gutachten/issues/45
    """
    lengths = np.array([[1.0, 2.0], [3.0, 4.0]])
    in_micrometres = Surface(
        heights=lengths,
        spacing_y=4.0,
        spacing_x=2.5,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="one-measurement",
    )
    in_millimetres = Surface(
        heights=lengths / LengthUnit.MILLIMETRE.micrometres,
        spacing_y=4.0 / LengthUnit.MILLIMETRE.micrometres,
        spacing_x=2.5 / LengthUnit.MILLIMETRE.micrometres,
        unit=LengthUnit.MILLIMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="one-measurement",
    )
    from_millimetres = read_bytes(to_bytes(in_millimetres), source="one-measurement")
    from_micrometres = read_bytes(to_bytes(in_micrometres), source="one-measurement")
    assert from_millimetres.unit is from_micrometres.unit is LengthUnit.MICROMETRE
    assert_close(
        from_millimetres.observed,
        from_micrometres.observed,
        what="one measurement declared in two units",
        atol=ATOL,
        rtol=RTOL,
    )
    assert_close(
        [from_millimetres.spacing_y, from_millimetres.spacing_x],
        [from_micrometres.spacing_y, from_micrometres.spacing_x],
        what="one spacing declared in two units",
        atol=ATOL,
        rtol=RTOL,
    )


def test_the_writer_will_write_a_surface_this_reader_will_not_read() -> None:
    """The asymmetry, asserted rather than left for a generated case to discover.

    The writer writes whatever surface it is given. The range check is aimed at
    bytes arriving from outside, where no unit is declared and nothing in the
    container contradicts a writer that used the wrong one, so it refuses on the
    reading side only. A surface with a millimetre of relief is therefore
    writable here and not readable back, and that is the state of the tree rather
    than an oversight.
    """
    wide = Surface(
        heights=np.array([[0.0, MAX_HEIGHT_RANGE_MICROMETRES * 2.0]]),
        spacing_y=1.0,
        spacing_x=1.0,
        unit=LengthUnit.MICROMETRE,
        orientation=AxisOrientation.Y_DOWN,
        source="wider-than-the-reader-admits",
    )
    written = to_bytes(wide)
    with pytest.raises(X3PError) as raised:
        read_bytes(written, source="wider-than-the-reader-admits")
    assert raised.value.reason == "height-range-implausible"
