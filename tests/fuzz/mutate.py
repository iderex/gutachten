"""Turning a valid container into a great many nearly valid ones.

A container this reader accepts is a zip holding an XML document holding a
binary array, and the inputs worth trying are the ones that still look like
that. Bytes drawn uniformly at random are refused at the first branch of the
zip reader and say nothing about the two parsers behind it, so every input here
starts as a container the writer produced and is damaged from there.

Every operator below is one of the ways a file arrives damaged in practice: a
byte flipped in transit, a run of bytes truncated by a fetch that stopped, a
length field overwritten, a chunk moved. They are applied a few at a time so
that an input is usually still a zip and often still holds a parseable document,
which is where the interesting failures are.

Deterministic given its seed. A fuzz finding nobody can reproduce is a bug
report with no attachment, so the seed that produced an input is what the
session prints and what a regression fixture records.
"""

from __future__ import annotations

import random
from collections.abc import Callable

#: The most damage one input takes. Above a handful the input stops resembling a
#: container at all and the run spends its budget re-proving that the zip reader
#: refuses noise.
MAX_OPERATIONS = 4

#: The longest run a single operator will cut, insert or move, in bytes. Large
#: enough to remove a whole member header, small enough that the archive usually
#: survives.
MAX_RUN = 64


def _flip_a_bit(data: bytearray, rng: random.Random) -> None:
    at = rng.randrange(len(data))
    data[at] ^= 1 << rng.randrange(8)


def _set_a_byte(data: bytearray, rng: random.Random) -> None:
    """A byte replaced by an interesting one rather than by any one.

    Zero, the two all-ones bytes and the ASCII digits are what length fields,
    counts and sizes are made of, and a random byte reaches them one time in
    twenty five.
    """
    data[rng.randrange(len(data))] = rng.choice([0x00, 0x01, 0x7F, 0x80, 0xFF, 0x30, 0x39])


def _cut_a_run(data: bytearray, rng: random.Random) -> None:
    at = rng.randrange(len(data))
    del data[at : at + rng.randint(1, MAX_RUN)]


def _duplicate_a_run(data: bytearray, rng: random.Random) -> None:
    at = rng.randrange(len(data))
    length = rng.randint(1, MAX_RUN)
    data[at:at] = data[at : at + length]


def _truncate(data: bytearray, rng: random.Random) -> None:
    """A fetch that stopped, which is the damage a reader meets most often.

    At least one byte is kept, so the operator never produces the empty input,
    and it does nothing at all to an input already one byte long. Without that
    the draw below has an empty range to choose from, which is a defect in this
    file rather than a finding about the reader, and it took one to notice.
    """
    if len(data) > 1:
        del data[rng.randrange(1, len(data)) :]


def _overwrite_a_number(data: bytearray, rng: random.Random) -> None:
    """Four bytes replaced by a little endian value a size field would notice.

    The zip format carries its sizes and offsets as little endian integers, and
    the reader's own bound on what it will decompress is read off one of them.
    """
    at = rng.randrange(max(1, len(data) - 4))
    value = rng.choice([0, 1, 0xFFFF, 0x7FFFFFFF, 0xFFFFFFFF])
    data[at : at + 4] = value.to_bytes(4, "little")


OPERATORS: tuple[Callable[[bytearray, random.Random], None], ...] = (
    _flip_a_bit,
    _set_a_byte,
    _cut_a_run,
    _duplicate_a_run,
    _truncate,
    _overwrite_a_number,
)


def mutate(source: bytes, rng: random.Random) -> bytes:
    """One damaged container, from ``source`` and the state of ``rng``."""
    data = bytearray(source)
    for _ in range(rng.randint(1, MAX_OPERATIONS)):
        if not data:
            break
        rng.choice(OPERATORS)(data, rng)
    return bytes(data)


def shrink(data: bytes, still_fails: Callable[[bytes], bool]) -> bytes:
    """The smallest input this reduction can reach that still fails.

    A finding becomes a regression fixture, and a fixture the size of a whole
    container is one nobody can read. This removes a run at a time, halving the
    run length whenever a pass removes nothing, which is the cheap reduction
    rather than a good one: it will not find a smaller input that needs two
    distant edits at once, and it does not claim to.
    """
    smallest = data
    length = max(1, len(smallest) // 2)
    while length >= 1:
        at = 0
        while at < len(smallest):
            candidate = smallest[:at] + smallest[at + length :]
            if candidate and still_fails(candidate):
                smallest = candidate
            else:
                at += length
        length //= 2
    return smallest
