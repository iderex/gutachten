"""Unit tests over the one function every numerical comparison goes through.

If this helper is wrong, every numerical test in the tree is wrong in the same
direction and none of them says so, which is why it is tested harder than what
it compares.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.support.tolerance import ToleranceError, assert_close


def test_equal_values_pass() -> None:
    assert_close(1.0, 1.0, what="a scalar against itself", atol=0.0)


def test_a_difference_inside_the_tolerance_passes() -> None:
    assert_close(1.0 + 1e-9, 1.0, what="a scalar off by a nanounit", atol=1e-8)


def test_a_difference_outside_the_tolerance_fails() -> None:
    with pytest.raises(ToleranceError) as raised:
        assert_close(1.0 + 1e-6, 1.0, what="a scalar off by a microunit", atol=1e-8)
    assert "a scalar off by a microunit" in str(raised.value)


def test_the_failure_message_states_the_tolerance_and_the_deviation() -> None:
    with pytest.raises(ToleranceError) as raised:
        assert_close([0.0, 5.0], [0.0, 0.0], what="an array with one bad cell", atol=1.0)
    message = str(raised.value)
    assert "atol=1.0" in message
    assert "5.0" in message
    assert "1 of 2" in message


def test_a_shape_mismatch_is_not_a_tolerance_question() -> None:
    with pytest.raises(ToleranceError) as raised:
        assert_close([1.0, 2.0], [1.0], what="arrays of different length", atol=1e6)
    assert "shapes differ" in str(raised.value)


def test_a_relative_tolerance_scales_with_the_expected_value() -> None:
    assert_close(1000.1, 1000.0, what="a large value", atol=0.0, rtol=1e-3)
    with pytest.raises(ToleranceError):
        assert_close(1.1, 1.0, what="a small value at the same relative tolerance", atol=0.0, rtol=1e-3)


def test_a_negative_tolerance_is_refused_rather_than_treated_as_zero() -> None:
    with pytest.raises(ValueError):
        assert_close(1.0, 1.0, what="a scalar", atol=-1.0)


def test_nan_never_compares_equal_however_wide_the_tolerance() -> None:
    with pytest.raises(ToleranceError):
        assert_close(np.nan, 0.0, what="a missing measurement against zero", atol=1e12)


def test_a_missing_measurement_on_both_sides_still_fails() -> None:
    # Two arrays that are both entirely absent are not two arrays that agree.
    # Reporting them as equal would let a pipeline that produced nothing pass a
    # golden test against a recording of nothing.
    with pytest.raises(ToleranceError):
        assert_close(np.nan, np.nan, what="absent against absent", atol=1e12)
