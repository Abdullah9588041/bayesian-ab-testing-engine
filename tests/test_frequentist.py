"""Tests for frequentist procedures."""

import numpy as np
import pytest
from scipy import stats

from bayes_ab.frequentist import (
    chi_square_test,
    confidence_interval_diff,
    two_proportion_ztest,
)


def test_ztest_reference_value():
    """Cross-checked against the normal approximation computed by hand:
    s_a=100/10000, s_b=120/10000 -> z ~ 1.3559, p ~ 0.1752 (two-sided)."""
    r = two_proportion_ztest(100, 10_000, 120, 10_000)
    assert r["z_statistic"] == pytest.approx(1.3559, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.1752, abs=1e-3)
    assert r["diff_b_minus_a"] == pytest.approx(0.002)


def test_ztest_identical_groups():
    r = two_proportion_ztest(50, 1_000, 50, 1_000)
    assert r["z_statistic"] == pytest.approx(0.0)
    assert r["p_value"] == pytest.approx(1.0)
    lo, hi = r["ci_lower"], r["ci_upper"]
    assert lo <= 0 <= hi


def test_ztest_antisymmetric():
    r1 = two_proportion_ztest(100, 10_000, 120, 10_000)
    r2 = two_proportion_ztest(120, 10_000, 100, 10_000)
    assert r1["z_statistic"] == pytest.approx(-r2["z_statistic"])
    assert r1["diff_b_minus_a"] == pytest.approx(-r2["diff_b_minus_a"])


def test_chi_square_matches_scipy():
    s_a, n_a, s_b, n_b = 48, 1_000, 72, 1_000
    table = np.array([[s_a, n_a - s_a], [s_b, n_b - s_b]])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    r = chi_square_test(s_a, n_a, s_b, n_b)
    assert r["chi2_statistic"] == pytest.approx(chi2)
    assert r["p_value"] == pytest.approx(p)
    assert r["degrees_of_freedom"] == dof


def test_chi_square_strong_effect_rejects():
    r = chi_square_test(50, 10_000, 90, 10_000)
    assert r["p_value"] < 0.001


def test_ci_covers_true_difference():
    # CI from data simulated with a known 1pp lift should cover 0.01 often;
    # here we just check the interval is well-formed and contains the estimate.
    lo, hi = confidence_interval_diff(100, 10_000, 200, 10_000)
    assert lo < 0.01 < hi
    assert hi - lo > 0


def test_invalid_alternative():
    with pytest.raises(ValueError):
        two_proportion_ztest(1, 10, 1, 10, alternative="bogus")
