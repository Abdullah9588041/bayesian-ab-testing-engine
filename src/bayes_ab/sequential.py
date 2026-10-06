"""Sequential monitoring and early stopping for Bayesian A/B tests.

A Bayesian test may be monitored continuously: because posterior
probabilities and expected loss are statements about *parameters given the
data observed so far*, the stopping rule does not distort their meaning the
way it distorts frequentist p-values ("peeking"). The module also includes a
simulation that quantifies the peeking bias of naive repeated p-value checks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .frequentist import two_proportion_ztest
from .models import BetaBinomialModel, expected_loss, prob_b_beats_a


def run_sequential(
    daily: pd.DataFrame,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    prob_threshold: float = 0.95,
    loss_threshold: float = 0.0005,
    min_days: int = 3,
) -> pd.DataFrame:
    """Monitor an A/B test day-by-day and apply the early-stopping rule.

    Parameters
    ----------
    daily:
        DataFrame with columns ['day', 'trials_a', 'successes_a',
        'trials_b', 'successes_b'] in chronological order.
    prob_threshold:
        Stop and declare B the winner when P(B > A) >= prob_threshold
        (or A when P(B > A) <= 1 - prob_threshold).
    loss_threshold:
        Stop when the expected loss of the currently leading variant
        drops below this value (expected foregone conversion rate).
    min_days:
        Do not stop before this many days of data, guarding against
        extreme early noise.

    Returns
    -------
    DataFrame with one row per day: cumulative counts, P(B > A),
    expected losses, and the decision at each look.
    """
    required = {"day", "trials_a", "successes_a", "trials_b", "successes_b"}
    if not required.issubset(daily.columns):
        raise ValueError(f"daily must contain columns {sorted(required)}")

    rows: List[Dict] = []
    stopped = False
    for i, row in enumerate(daily.itertuples(), start=1):
        prev = rows[-1] if rows else None
        n_a = (prev["cum_trials_a"] if prev else 0) + row.trials_a
        s_a = (prev["cum_successes_a"] if prev else 0) + row.successes_a
        n_b = (prev["cum_trials_b"] if prev else 0) + row.trials_b
        s_b = (prev["cum_successes_b"] if prev else 0) + row.successes_b

        a = BetaBinomialModel.fit(s_a, n_a, prior_alpha, prior_beta)
        b = BetaBinomialModel.fit(s_b, n_b, prior_alpha, prior_beta)
        prob = prob_b_beats_a(a, b)
        loss_a = expected_loss(a, b, choose="A")
        loss_b = expected_loss(a, b, choose="B")

        decision = "continue"
        if i >= min_days:
            leader_loss = loss_b if prob >= 0.5 else loss_a
            if prob >= prob_threshold or (1 - prob) >= prob_threshold:
                decision = "stop: choose B" if prob >= 0.5 else "stop: choose A"
                stopped = True
            elif leader_loss <= loss_threshold:
                decision = "stop: negligible loss"
                stopped = True

        rows.append(
            {
                "day": row.day,
                "cum_trials_a": n_a,
                "cum_successes_a": s_a,
                "cum_trials_b": n_b,
                "cum_successes_b": s_b,
                "rate_a": s_a / n_a,
                "rate_b": s_b / n_b,
                "prob_b_beats_a": prob,
                "expected_loss_a": loss_a,
                "expected_loss_b": loss_b,
                "decision": decision,
            }
        )
        if stopped:
            break
    return pd.DataFrame(rows)


def simulate_peeking_bias(
    p_a: float,
    p_b: float,
    n_per_peek: int,
    n_peeks: int,
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Compare three testing strategies under repeated peeking.

    Strategies evaluated on the same simulated experiments:
      1. fixed-sample frequentist: single z-test at the final sample size
         (valid control of the Type I error rate).
      2. naive peeking: two-sided z-test at every peek, reject if any
         p-value < alpha (the classic "peeking" mistake).
      3. bayesian sequential: at every peek, declare B the winner if
         P(B > A | data so far) > 0.95 (uniform prior).

    Returns observed decision rates (not theoretical guarantees): under the
    null (p_a == p_b) these are empirical false-positive rates; under an
    alternative they are empirical power. The Bayesian rule is *reported*
    honestly as an observed rate — Bayesian posteriors do not promise
    frequentist Type I error control, which is itself an important finding.
    """
    rng = np.random.default_rng(seed)
    fixed_reject = 0
    peek_reject = 0
    bayes_decide_b = 0
    bayes_decide_a = 0
    total_n = n_per_peek * n_peeks

    for _ in range(n_sims):
        # Binomial outcomes per peek-batch for both variants.
        batch_a = rng.binomial(n_per_peek, p_a, size=n_peeks)
        batch_b = rng.binomial(n_per_peek, p_b, size=n_peeks)
        cum_a = np.cumsum(batch_a)
        cum_b = np.cumsum(batch_b)

        # 1. Fixed-sample frequentist: a single z-test at the planned
        #    final sample size, evaluated independently of any peeking.
        s_a_full, s_b_full = int(cum_a[-1]), int(cum_b[-1])
        if two_proportion_ztest(s_a_full, total_n, s_b_full, total_n)["p_value"] < alpha:
            fixed_reject += 1

        # 2 & 3. Peek loop: naive repeated testing + Bayesian sequential rule.
        peeked = False
        for k in range(1, n_peeks + 1):
            n = k * n_per_peek
            s_a, s_b = int(cum_a[k - 1]), int(cum_b[k - 1])
            z = two_proportion_ztest(s_a, n, s_b, n)
            if z["p_value"] < alpha:
                peeked = True
            # Bayesian look with the same data.
            a = BetaBinomialModel.fit(s_a, n)
            b = BetaBinomialModel.fit(s_b, n)
            prob = prob_b_beats_a(a, b)
            if prob > 0.95:
                bayes_decide_b += 1
                break
            if prob < 0.05:
                bayes_decide_a += 1
                break
        if peeked:
            peek_reject += 1

    return {
        "n_sims": n_sims,
        "p_a": p_a,
        "p_b": p_b,
        "fixed_sample_frequentist_reject_rate": fixed_reject / n_sims,
        "naive_peeking_frequentist_reject_rate": peek_reject / n_sims,
        "bayesian_sequential_decide_b_rate": bayes_decide_b / n_sims,
        "bayesian_sequential_decide_a_rate": bayes_decide_a / n_sims,
        "nominal_alpha": alpha,
    }
