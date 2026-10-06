"""Sample-size planning: frequentist power analysis and Bayesian assurance."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats

from .models import BetaBinomialModel, prob_b_beats_a


def frequentist_sample_size(
    p_a: float,
    p_b: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-variant sample size for a two-sided two-proportion z-test.

    Standard normal-approximation formula:
        n = (z_{1-alpha/2} * sqrt(2 p_bar q_bar)
             + z_{power} * sqrt(p_a q_a + p_b q_b))^2 / (p_b - p_a)^2
    Returns the ceiling, i.e. the smallest integer n achieving the target.
    """
    if not 0 < p_a < 1 and not 0 < p_b < 1:
        raise ValueError("Rates must be probabilities.")
    if p_a == p_b:
        raise ValueError("p_a and p_b must differ for sample size planning.")
    p_bar = (p_a + p_b) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))
    )
    n = (numerator / abs(p_b - p_a)) ** 2
    return int(np.ceil(n))


def bayesian_assurance(
    p_a: float,
    p_b: float,
    n: int,
    prob_threshold: float = 0.95,
    n_sims: int = 2000,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Bayesian assurance: P(a future experiment yields a decisive answer).

    Simulates n_sims experiments of size n from the *assumed true* rates,
    then measures how often the posterior rule P(B > A) >= prob_threshold
    fires (correctly if p_b > p_a). Unlike frequentist power, assurance
    averages over sampling variability of the data-generating process and
    speaks directly about the decision rule that will be used.
    """
    rng = np.random.default_rng(seed)
    decisive = 0
    probs = np.empty(n_sims)
    for i in range(n_sims):
        s_a = int(rng.binomial(n, p_a))
        s_b = int(rng.binomial(n, p_b))
        prob = prob_b_beats_a(
            BetaBinomialModel.fit(s_a, n, prior_alpha, prior_beta),
            BetaBinomialModel.fit(s_b, n, prior_alpha, prior_beta),
        )
        probs[i] = prob
        if prob >= prob_threshold:
            decisive += 1
    return {
        "n_per_variant": n,
        "assurance": float(decisive / n_sims),
        "median_prob_b_beats_a": float(np.median(probs)),
        "n_sims": n_sims,
        "prob_threshold": prob_threshold,
    }


def find_sample_size_for_assurance(
    p_a: float,
    p_b: float,
    target_assurance: float = 0.8,
    candidate_ns: List[int] | None = None,
    seed: int = 42,
    **kwargs,
) -> Dict:
    """Grid search for the smallest n whose Bayesian assurance hits the target."""
    if candidate_ns is None:
        candidate_ns = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000,
                        250_000, 500_000]
    results = []
    chosen = None
    for n in candidate_ns:
        res = bayesian_assurance(p_a, p_b, n, seed=seed, **kwargs)
        results.append(res)
        if chosen is None and res["assurance"] >= target_assurance:
            chosen = n
    return {
        "target_assurance": target_assurance,
        "recommended_n_per_variant": chosen,
        "grid": results,
    }
