"""Property tests over generated inputs.

A unit test asks whether the helper is right about the cases somebody thought
of. These ask whether it is right about the cases nobody did, which is the class
of input a numerical pipeline actually meets.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from tests.support.tolerance import ToleranceError, assert_close

finite_floats = st.floats(
    min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False, width=64
)


@given(value=finite_floats)
def test_any_finite_value_equals_itself_at_zero_tolerance(value: float) -> None:
    assert_close(value, value, what="a value against itself", atol=0.0)


@given(
    array=npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=2, max_side=8),
        elements=finite_floats,
    )
)
def test_any_finite_array_equals_itself_at_zero_tolerance(array: np.ndarray) -> None:
    assert_close(array, array, what="an array against itself", atol=0.0)


@given(expected=finite_floats, offset=finite_floats, atol=st.floats(0.0, 1e6))
@settings(max_examples=200)
def test_the_verdict_matches_the_stated_rule(expected: float, offset: float, atol: float) -> None:
    # The helper is allowed to be strict or lenient, but it is not allowed to be
    # something other than what its docstring says. This pins the verdict to the
    # rule rather than to the implementation.
    actual = expected + offset
    if not np.isfinite(actual):
        return
    should_pass = abs(actual - expected) <= atol
    try:
        assert_close(actual, expected, what="a generated pair", atol=atol)
    except ToleranceError:
        assert not should_pass
    else:
        assert should_pass


@given(value=finite_floats, atol=st.floats(0.0, 1e12))
def test_a_missing_measurement_never_passes_however_wide_the_tolerance(
    value: float, atol: float
) -> None:
    try:
        assert_close(np.nan, value, what="a missing measurement", atol=atol)
    except ToleranceError:
        return
    raise AssertionError("a missing measurement was accepted as a value")
