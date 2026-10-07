"""
Testy jednostkowe dla mcdm/numeric_utils.py -- wspolnego zabezpieczenia
przed dzieleniem przez zero, uzywanego przez TOPSIS i ELECTRE.
"""

import numpy as np
import pytest

from mcdm.numeric_utils import DEFAULT_EPSILON, safe_denominator, safe_divide


# ----------------------------------------------------------------------
# safe_denominator
# ----------------------------------------------------------------------


def test_safe_denominator_replaces_zeros_with_epsilon():
    result = safe_denominator(np.array([2.0, 0.0, -3.0, 0.0]))
    np.testing.assert_array_equal(result, [2.0, DEFAULT_EPSILON, -3.0, DEFAULT_EPSILON])


def test_safe_denominator_leaves_nonzero_values_untouched():
    values = np.array([1.5, -2.5, 100.0])
    result = safe_denominator(values)
    np.testing.assert_array_equal(result, values)


def test_safe_denominator_does_not_mutate_input_array():
    values = np.array([0.0, 1.0])
    original = values.copy()
    safe_denominator(values)
    np.testing.assert_array_equal(values, original)


def test_safe_denominator_accepts_custom_epsilon():
    result = safe_denominator(np.array([0.0]), eps=0.5)
    assert result[0] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# safe_divide
# ----------------------------------------------------------------------


def test_safe_divide_matches_plain_division_when_no_zeros():
    numerator = np.array([4.0, 9.0])
    denominator = np.array([2.0, 3.0])
    np.testing.assert_allclose(safe_divide(numerator, denominator), [2.0, 3.0])


def test_safe_divide_avoids_inf_and_nan_on_zero_denominator():
    numerator = np.array([1.0, 0.0])
    denominator = np.array([0.0, 0.0])
    result = safe_divide(numerator, denominator)
    assert not np.any(np.isinf(result))
    assert not np.any(np.isnan(result))


def test_safe_divide_zero_over_zero_is_treated_as_zero():
    """Przypadek D+ = D- = 0 w TOPSIS (wszystkie alternatywy identyczne) --
    0/eps musi dac 0, a nie NaN, zeby wskaznik bliskosci pozostal
    dobrze zdefiniowany."""
    result = safe_divide(np.array([0.0]), np.array([0.0]))
    assert result[0] == pytest.approx(0.0)


def test_safe_divide_does_not_mutate_inputs():
    numerator = np.array([1.0, 2.0])
    denominator = np.array([0.0, 2.0])
    num_copy, denom_copy = numerator.copy(), denominator.copy()
    safe_divide(numerator, denominator)
    np.testing.assert_array_equal(numerator, num_copy)
    np.testing.assert_array_equal(denominator, denom_copy)
