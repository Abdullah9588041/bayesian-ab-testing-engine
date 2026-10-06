"""Frequentist counterparts for honest comparison with the Bayesian analysis."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def two_proportion_ztest(
    s_a: int, n_a: int, s_b: int, n_b: int, alternative: str = "two-sided"
) -> Dict[str, float]:
    """Two-proportion z-test (pooled variance, normal approximation).

    H0: p_A == p_B. Returns the z-statistic, p-value, observed difference
    (p_B - p_A) and a Wald confidence interval for the difference.
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'.")
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Sample sizes must be positive.")
    p_a, p_b = s_a / n_a, s_b / n_b
    p_pool = (s_a + s_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = 0.0 if se == 0 else (p_b - p_a) / se
    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":  # H1: p_B > p_A
        p_value = stats.norm.sf(z)
    else:
        p_value = stats.norm.cdf(z)
    ci = confidence_interval_diff(s_a, n_a, s_b, n_b, alpha=0.05)
    return {
        "z_statistic": float(z),
        "p_value": float(p_value),
        "diff_b_minus_a": float(p_b - p_a),
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "alternative": alternative,
    }


def chi_square_test(s_a: int, n_a: int, s_b: int, n_b: int) -> Dict[str, float]:
    """Chi-square test of independence on the 2x2 success/failure table."""
    table = np.array([[s_a, n_a - s_a], [s_b, n_b - s_b]])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
    }


def confidence_interval_diff(
    s_a: int, n_a: int, s_b: int, n_b: int, alpha: float = 0.05
) -> tuple:
    """Wald (normal approximation) confidence interval for p_B - p_A."""
    p_a, p_b = s_a / n_a, s_b / n_b
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return (float(diff - z * se), float(diff + z * se))
