"""The one place a numerical comparison is made.

Every numerical assertion in the suite goes through ``assert_close``, and the
tolerance is an argument with no default. Two failures are prevented by that.

A tolerance buried as a constant inside a test is a tolerance nobody revisits.
It is chosen once to make a test pass, and from then on it silently absorbs
drift: a change that moves a score by more than the physics allows still passes,
because the number it has to beat was picked by whoever was closing the ticket.

A tolerance that is not in the failure message is a failure nobody can read. A
report saying two arrays differ tells you nothing about whether they differ by a
rounding error or by a factor of a thousand, and in a pipeline whose defects are
mostly numbers that quietly changed, that difference is the whole diagnosis.

So the tolerance is passed in at every call site, it is stated in the message,
and the largest observed deviation is stated next to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["assert_close", "ToleranceError"]


class ToleranceError(AssertionError):
    """Raised when a numerical comparison exceeds the tolerance it was given."""


def assert_close(
    actual: Any,
    expected: Any,
    *,
    what: str,
    atol: float,
    rtol: float = 0.0,
) -> None:
    """Assert ``actual`` equals ``expected`` within an explicitly stated tolerance.

    ``what`` names the quantity being compared and appears in the failure
    message, because "arrays differ" does not tell a reader which number moved.
    ``atol`` and ``rtol`` are keyword-only and ``atol`` has no default, so a
    tolerance is always a decision somebody made at the call site rather than
    one inherited from a library.

    The comparison is ``|actual - expected| <= atol + rtol * |expected|``, which
    is the numpy convention, and it is asymmetric in exactly the way that
    convention is.
    """
    if atol < 0.0 or rtol < 0.0:
        raise ValueError(f"tolerances must not be negative, got atol={atol!r} rtol={rtol!r}")

    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)

    if a.shape != e.shape:
        raise ToleranceError(
            f"{what}: shapes differ, actual {a.shape} against expected {e.shape}. "
            f"No tolerance can reconcile a shape mismatch."
        )

    allowed = atol + rtol * np.abs(e)
    finite = np.isfinite(a) & np.isfinite(e)

    # A tolerance is a statement about how far two numbers may be apart, and a
    # missing measurement is not a number that is far away. Comparing them by
    # subtraction would let any tolerance wide enough swallow the difference
    # between a result and no result at all, so the two cases are counted
    # separately and reported separately.
    deviation = np.where(finite, np.abs(a - e), 0.0)
    over_tolerance = finite & (deviation > allowed)
    non_finite_mismatch = ~finite & (a != e)

    if not np.any(over_tolerance) and not np.any(non_finite_mismatch):
        return

    total = int(a.size)
    parts = [f"{what}:"]
    if np.any(over_tolerance):
        worst = float(np.max(np.where(over_tolerance, deviation, 0.0)))
        parts.append(
            f"{int(np.count_nonzero(over_tolerance))} of {total} value(s) outside the "
            f"stated tolerance atol={atol!r} rtol={rtol!r}, largest deviation {worst!r}."
        )
    if np.any(non_finite_mismatch):
        parts.append(
            f"{int(np.count_nonzero(non_finite_mismatch))} of {total} value(s) differ in a "
            f"way no tolerance covers, because at least one side is not finite. "
            f"A missing measurement is not a number that is nearly right."
        )
    parts.append(
        "If the tolerance is wrong, change it at the call site and say why, rather "
        "than widening it until the test passes."
    )
    raise ToleranceError(" ".join(parts))
