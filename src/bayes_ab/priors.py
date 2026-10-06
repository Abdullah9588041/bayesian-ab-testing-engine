"""Prior sensitivity analysis: how much do conclusions depend on the prior?"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .models import BetaBinomialModel, expected_loss, prob_b_beats_a

#: (name -> (alpha0, beta0, description)). Priors are applied to both variants.
STANDARD_PRIORS: Dict[str, Tuple[float, float, str]] = {
    "uniform": (
        1.0,
        1.0,
        "Uniform Beta(1,1): every conversion rate equally likely a priori; "
        "adds 1 pseudo-success and 1 pseudo-failure.",
    ),
    "jeffreys": (
        0.5,
        0.5,
        "Jeffreys Beta(1/2,1/2): objective, reparameterization-invariant "
        "prior for the Binomial model.",
    ),
    "historical": (
        5.0,
        995.0,
        "Informative Beta(5,995): encodes past campaigns averaging ~0.5% "
        "conversion (prior mean 0.005, strength of 1000 observations).",
    ),
}


def prior_sensitivity(
    s_a: int,
    n_a: int,
    s_b: int,
    n_b: int,
    priors: Dict[str, Tuple[float, float, str]] | None = None,
) -> pd.DataFrame:
    """Recompute P(B > A) and expected losses under several priors.

    With millions of observations the likelihood dominates and results
    should barely move — that robustness is itself a finding worth
    reporting. With small samples the table reveals how much the prior
    matters.
    """
    priors = priors or STANDARD_PRIORS
    rows = []
    for name, (a0, b0, desc) in priors.items():
        a = BetaBinomialModel.fit(s_a, n_a, a0, b0)
        b = BetaBinomialModel.fit(s_b, n_b, a0, b0)
        rows.append(
            {
                "prior": name,
                "prior_alpha": a0,
                "prior_beta": b0,
                "prior_mean": a0 / (a0 + b0),
                "prob_b_beats_a": prob_b_beats_a(a, b),
                "expected_loss_choose_a": expected_loss(a, b, choose="A"),
                "expected_loss_choose_b": expected_loss(a, b, choose="B"),
                "posterior_mean_a": a.mean(),
                "posterior_mean_b": b.mean(),
                "description": desc,
            }
        )
    return pd.DataFrame(rows)
