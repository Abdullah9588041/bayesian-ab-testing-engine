"""Tests for sequential monitoring, the peeking simulation, and sample sizing."""

import numpy as np
import pandas as pd
import pytest

from bayes_ab.sample_size import (
    bayesian_assurance,
    find_sample_size_for_assurance,
    frequentist_sample_size,
)
from bayes_ab.sequential import run_sequential, simulate_peeking_bias


def _make_daily(days=10, p_a=0.05, p_b=0.08, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "day": [f"2026-01-{d+1:02d}" for d in range(days)],
            "trials_a": [n] * days,
            "successes_a": rng.binomial(n, p_a, days),
            "trials_b": [n] * days,
            "successes_b": rng.binomial(n, p_b, days),
        }
    )


def test_sequential_stops_on_strong_effect():
    traj = run_sequential(_make_daily())
    assert len(traj) < 10  # stopped before the end
    last = traj.iloc[-1]
    assert last["decision"].startswith("stop")
    assert last["prob_b_beats_a"] > 0.95
    expected = {"day", "cum_trials_a", "prob_b_beats_a", "expected_loss_a",
                "expected_loss_b", "decision"}
    assert expected.issubset(traj.columns)


def test_sequential_keeps_going_without_evidence():
    # Small daily n keeps posterior uncertainty (and hence expected loss)
    # above the stopping threshold, so a null-like stream never stops.
    daily = _make_daily(p_a=0.05, p_b=0.05, n=500, seed=11)
    traj = run_sequential(daily, loss_threshold=1e-6)
    assert len(traj) == 10
    assert (traj["decision"] == "continue").all()


def test_sequential_min_days_respected():
    traj = run_sequential(_make_daily(p_a=0.01, p_b=0.30, days=6), min_days=3)
    # Never stops before min_days even with an overwhelming effect.
    stopped = traj[traj["decision"].str.startswith("stop")]
    assert not stopped.empty
    assert stopped.index[0] >= 2  # 0-based: first stop at day 3 or later


def test_peeking_bias_demonstration():
    """Naive repeated p-value checks inflate the false-positive rate above
    the valid fixed-sample rate on the same simulated data (H0).

    The Bayesian sequential rule's observed decision rates are reported
    honestly as well: a 0.95 posterior threshold is a posterior statement,
    not a frequentist error guarantee, so its H0 decision rate is *not*
    expected to sit at 0.05 (see docs/math_notes.md section 4).
    """
    res = simulate_peeking_bias(
        p_a=0.005, p_b=0.005, n_per_peek=4000, n_peeks=5, n_sims=500, seed=123
    )
    assert res["fixed_sample_frequentist_reject_rate"] == pytest.approx(0.05, abs=0.03)
    assert (
        res["naive_peeking_frequentist_reject_rate"]
        > res["fixed_sample_frequentist_reject_rate"]
    )
    for key in ("bayesian_sequential_decide_b_rate", "bayesian_sequential_decide_a_rate"):
        assert 0.0 <= res[key] <= 1.0


def test_frequentist_sample_size_textbook_value():
    """50% vs 55%, alpha=0.05, power=0.8 -> 1565 per group (standard result)."""
    assert frequentist_sample_size(0.5, 0.55, alpha=0.05, power=0.8) == 1565


def test_sample_size_shrinks_with_larger_effect():
    n_small = frequentist_sample_size(0.5, 0.55)
    n_large = frequentist_sample_size(0.5, 0.65)
    assert n_large < n_small


def test_bayesian_assurance_monotone_in_n():
    a1 = bayesian_assurance(0.05, 0.06, n=5_000, n_sims=400, seed=5)
    a2 = bayesian_assurance(0.05, 0.06, n=50_000, n_sims=400, seed=5)
    assert a2["assurance"] >= a1["assurance"]
    assert 0 <= a1["assurance"] <= 1


def test_find_sample_size_for_assurance():
    res = find_sample_size_for_assurance(
        0.05, 0.06, target_assurance=0.8,
        candidate_ns=[1_000, 5_000, 20_000], seed=5,
    )
    assert res["recommended_n_per_variant"] in {1_000, 5_000, 20_000, None}
    assert len(res["grid"]) == 3
