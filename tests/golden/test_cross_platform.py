"""The three platforms agree to a tolerance that was measured rather than hoped.

`docs/determinism.md` promised agreement within a declared tolerance and said in
those words that the tolerance had not been measured. This is where it was. The
same fixture ran on the ubuntu, macos and windows legs of one workflow run and
each reported its five numbers, and `cross_platform.json` holds all three
readings, the largest spread between them, the declared tolerance and the
argument for it.

The argument matters more than the number. A tolerance set at the largest
deviation seen once is a tolerance that reds on the next runner image for a
reason that is not a defect, and one set wide enough to be comfortable is a
tolerance that absorbs the failure it exists to catch. This one is bounded from
below by the reordering of floating point addition that a differently built
numerical backend does, and from above by the finest height an instrument in
this field resolves. Both bounds are in the recording.

`measured` is compared at zero tolerance and the others are not. It is a count
of samples, so a platform that kept one sample more than another disagreed about
the shape of a mask rather than about arithmetic, and no tolerance on a height
covers that.

**What this does not establish.** Three runner images, one numerical build each,
one processor family each, on the day the recording was taken. It is a
measurement over what the gate runs and not a guarantee about every machine, and
`docs/determinism.md` keeps the paragraph saying bit identical results across
different processors and library builds are not promised at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.golden.determinism_fixture import one_run, summary
from tests.support.tolerance import assert_close

RECORDING = Path(__file__).with_name("cross_platform.json")
RECORDED: dict[str, Any] = json.loads(RECORDING.read_text(encoding="utf-8"))

#: The declared cross-platform tolerance, in micrometres. Read from the
#: recording rather than repeated here, so the number and the argument for it
#: cannot drift into two places.
TOLERANCE_UM: float = RECORDED["tolerance_um"]

#: A count rather than a height, so nothing is allowed to differ.
EXACT = ("measured",)


@pytest.mark.parametrize("quantity", sorted(RECORDED["summary"]))
def test_this_platform_agrees_with_the_recording(quantity: str) -> None:
    observed = summary(one_run()[0])
    tolerance = 0.0 if quantity in EXACT else TOLERANCE_UM

    assert_close(
        observed[quantity],
        RECORDED["summary"][quantity],
        what=f"{quantity} against the recording made on {RECORDED['recorded_on']}",
        atol=tolerance,
    )


def test_the_declared_tolerance_sits_above_what_the_three_platforms_did() -> None:
    # The lower bound of the argument, asserted rather than described. A
    # tolerance under the observed spread is one that reds on a platform that
    # did nothing wrong.
    assert RECORDED["observed_spread_um"] < TOLERANCE_UM


def test_the_declared_tolerance_sits_below_anything_an_instrument_resolves() -> None:
    # The upper bound. One nanometre is the finest height an instrument in this
    # field resolves, so a tolerance at or above it could absorb a difference
    # somebody could measure, which is the failure the number exists against.
    one_nanometre_in_um = 1e-3

    assert one_nanometre_in_um > TOLERANCE_UM


def test_the_recorded_spread_is_the_spread_of_the_recorded_readings() -> None:
    # The recording carries three readings and one spread, and a spread copied
    # in by hand is a number nothing checks. This recomputes it.
    readings = RECORDED["readings"]
    widest = max(
        max(reading[quantity] for reading in readings.values())
        - min(reading[quantity] for reading in readings.values())
        for quantity in RECORDED["summary"]
    )

    assert_close(
        widest,
        RECORDED["observed_spread_um"],
        what="the spread recomputed from the three readings",
        atol=0.0,
    )


def test_the_recording_is_one_of_the_readings_rather_than_an_average_of_them() -> None:
    # An average of three platforms is a run nobody made, and comparing against
    # it would mean no platform ever reproduces the recording exactly.
    assert RECORDED["summary"] in list(RECORDED["readings"].values())
