"""Tests for the Beta-Binomial model against known analytical values."""

import numpy as np
import pytest
from scipy import stats

from bayes_ab.models import (
    BetaBinomialModel,
    decision_summary,
    expected_loss,
    prob_b_beats_a,
)


def test_posterior_matches_conjugate_update():
    """Uniform prior + 3 successes / 10 trials -> Beta(4, 8) exactly."""
    m = BetaBinomialModel.fit(successes=3, trials=10, prior_alpha=1.0, prior_beta=1.0)
    assert m.alpha == 4.0
    assert m.beta == 8.0
    assert m.successes == 3
    assert m.trials == 10


def test_posterior_mean_and_variance():
    m = BetaBinomialModel.fit(3, 10)
    assert m.mean() == pytest.approx(4 / 12)
    # Beta(a,b) variance = ab / ((a+b)^2 (a+b+1))
    assert m.variance() == pytest.approx(4 * 8 / (12**2 * 13))


def test_credible_interval_matches_scipy():
    m = BetaBinomialModel.fit(7, 20, prior_alpha=2.0, prior_beta=5.0)
    lo, hi = m.credible_interval(0.95)
    assert lo == pytest.approx(stats.beta.ppf(0.025, 9, 18))
    assert hi == pytest.approx(stats.beta.ppf(0.975, 9, 18))


def test_prob_b_beats_a_symmetry():
    """Identical posteriors must give P(B > A) = 0.5 (up to quadrature error)."""
    a = BetaBinomialModel.fit(5, 100)
    b = BetaBinomialModel.fit(5, 100)
    assert prob_b_beats_a(a, b) == pytest.approx(0.5, abs=1e-6)
    assert prob_b_beats_a(a, b) + prob_b_beats_a(b, a) == pytest.approx(1.0, abs=1e-6)


def test_prob_b_beats_a_against_monte_carlo():
    a = BetaBinomialModel.fit(48, 1000)
    b = BetaBinomialModel.fit(72, 1000)
    rng = np.random.default_rng(0)
    mc = float(np.mean(rng.beta(b.alpha, b.beta, 200_000) > rng.beta(a.alpha, a.beta, 200_000)))
    assert prob_b_beats_a(a, b) == pytest.approx(mc, abs=0.01)


def test_expected_loss_against_monte_carlo():
    a = BetaBinomialModel.fit(48, 1000)
    b = BetaBinomialModel.fit(72, 1000)
    rng = np.random.default_rng(1)
    sa = rng.beta(a.alpha, a.beta, 300_000)
    sb = rng.beta(b.alpha, b.beta, 300_000)
    mc_loss_b = float(np.mean(np.maximum(sa - sb, 0.0)))
    mc_loss_a = float(np.mean(np.maximum(sb - sa, 0.0)))
    assert expected_loss(a, b, choose="B") == pytest.approx(mc_loss_b, abs=2e-4)
    assert expected_loss(a, b, choose="A") == pytest.approx(mc_loss_a, abs=2e-4)


def test_expected_loss_bounded():
    """Loss can never exceed the maximum possible conversion-rate gap."""
    a = BetaBinomialModel.fit(0, 10)
    b = BetaBinomialModel.fit(10, 10)
    assert 0 <= expected_loss(a, b, choose="B") <= 1
    assert 0 <= expected_loss(a, b, choose="A") <= 1
    # Choosing the clearly better variant (B) has ~zero loss.
    assert expected_loss(a, b, choose="B") < 1e-3


def test_decision_summary_strong_effect():
    a = BetaBinomialModel.fit(50, 10_000)
    b = BetaBinomialModel.fit(90, 10_000)
    s = decision_summary(a, b)
    assert s["prob_b_beats_a"] > 0.99
    assert s["decision"] == "Choose B (test)"
    assert s["expected_loss_choose_b"] < s["expected_loss_choose_a"]


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        BetaBinomialModel.fit(successes=11, trials=10)
    with pytest.raises(ValueError):
        BetaBinomialModel.fit(successes=1, trials=0)
    with pytest.raises(ValueError):
        BetaBinomialModel.fit(1, 10, prior_alpha=0)
