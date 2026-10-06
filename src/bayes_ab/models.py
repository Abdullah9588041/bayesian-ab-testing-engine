"""Beta-Binomial conjugate Bayesian model for A/B testing.

Derivation (see docs/math_notes.md):
    Prior:      p ~ Beta(alpha0, beta0)
    Likelihood: s | p ~ Binomial(n, p)
    Posterior:  p | s ~ Beta(alpha0 + s, beta0 + n - s)

From the two independent posteriors we compute:
    - P(p_B > p_A): posterior probability variant B beats A
    - Expected loss of choosing each variant (decision rule)
    - Equal-tailed credible intervals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import integrate, stats
from scipy.special import betainc, betaincinv


@dataclass(frozen=True)
class BetaBinomialModel:
    """Conjugate Beta-Binomial model for one variant's conversion rate."""

    alpha: float  # posterior alpha
    beta: float  # posterior beta
    successes: int  # observed successes s
    trials: int  # observed trials n
    prior_alpha: float  # prior alpha0
    prior_beta: float  # prior beta0

    @classmethod
    def fit(
        cls,
        successes: int,
        trials: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> "BetaBinomialModel":
        """Fit the model: posterior = Beta(prior_alpha + s, prior_beta + n - s)."""
        if successes < 0 or trials <= 0 or successes > trials:
            raise ValueError("Require 0 <= successes <= trials and trials > 0.")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Prior parameters must be positive.")
        return cls(
            alpha=prior_alpha + successes,
            beta=prior_beta + trials - successes,
            successes=successes,
            trials=trials,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )

    def mean(self) -> float:
        """Posterior mean E[p | data] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Posterior variance of p."""
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Equal-tailed posterior credible interval for p."""
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1).")
        tail = (1 - level) / 2
        return (
            float(stats.beta.ppf(tail, self.alpha, self.beta)),
            float(stats.beta.ppf(1 - tail, self.alpha, self.beta)),
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Posterior density evaluated at x."""
        return stats.beta.pdf(x, self.alpha, self.beta)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Posterior CDF evaluated at x."""
        return stats.beta.cdf(x, self.alpha, self.beta)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n posterior samples of p using the provided RNG."""
        return rng.beta(self.alpha, self.beta, size=n)


def prob_b_beats_a(a: BetaBinomialModel, b: BetaBinomialModel) -> float:
    """Posterior probability P(p_B > p_A) via numerical quadrature.

    P(B > A) = integral_0^1 f_A(x) * (1 - F_B(x)) dx
    The integral is evaluated in *quantile space* via the substitution
    u = F_A(x), which removes the sharp density spikes that defeat
    adaptive quadrature when posteriors are tight (large n):

        P(B > A) = integral_0^1 [1 - F_B(F_A^{-1}(u))] du.

    The integrand is smooth and bounded in [0, 1].
    """
    a_alpha, a_beta = a.alpha, a.beta
    b_alpha, b_beta = b.alpha, b.beta

    def integrand(u: float) -> float:
        x = betaincinv(a_alpha, a_beta, u)  # F_A^{-1}(u)
        return 1.0 - betainc(b_alpha, b_beta, x)  # 1 - F_B(x)

    value, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-9)
    return float(np.clip(value, 0.0, 1.0))


def _expected_shortfall(a: BetaBinomialModel, b: BetaBinomialModel) -> float:
    """E[max(p_A - p_B, 0)]: expected loss incurred if we choose B but A is better.

    Uses the identity, for X ~ Beta(alpha, beta):

        E[(X - c)+] = alpha/(alpha+beta) * (1 - F_{Beta(alpha+1, beta)}(c))
                       - c * (1 - F_X(c)),

    and integrates over the *quantiles* of the chosen variant's posterior
    (substitution v = F_B(c)), again to keep quadrature stable for tight
    posteriors:

        E[max(p_A - p_B, 0)] = integral_0^1 E[(p_A - F_B^{-1}(v))_+] dv.
    """
    alpha, beta = a.alpha, a.beta
    b_alpha, b_beta = b.alpha, b.beta
    mean_a = alpha / (alpha + beta)

    def inner(c: float) -> float:
        sf_x = 1.0 - betainc(alpha, beta, c)
        sf_x1 = 1.0 - betainc(alpha + 1.0, beta, c)
        return mean_a * sf_x1 - c * sf_x

    def integrand(v: float) -> float:
        c = betaincinv(b_alpha, b_beta, v)  # F_B^{-1}(v)
        return max(inner(c), 0.0)

    value, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-9)
    return float(max(value, 0.0))


def expected_loss(
    a: BetaBinomialModel, b: BetaBinomialModel, choose: str = "B"
) -> float:
    """Expected loss (expected foregone conversion rate) of deploying a variant.

    Choosing "B": loss = E[max(p_A - p_B, 0)] — how much rate we lose on
    average if B turns out worse than A. Choosing "A" is symmetric.
    This is the decision-theoretic quantity behind the stopping rule:
    stop when the expected loss of the leading variant is negligible.
    """
    if choose == "B":
        return _expected_shortfall(a, b)
    if choose == "A":
        return _expected_shortfall(b, a)
    raise ValueError("choose must be 'A' or 'B'.")


def decision_summary(
    a: BetaBinomialModel,
    b: BetaBinomialModel,
    prob_threshold: float = 0.95,
) -> dict:
    """One-line decision summary comparing the two variants."""
    prob = prob_b_beats_a(a, b)
    loss_a = expected_loss(a, b, choose="A")
    loss_b = expected_loss(a, b, choose="B")
    if prob >= prob_threshold:
        decision = "Choose B (test)"
    elif 1 - prob >= prob_threshold:
        decision = "Choose A (control)"
    else:
        decision = "Continue collecting data"
    return {
        "prob_b_beats_a": prob,
        "expected_loss_choose_a": loss_a,
        "expected_loss_choose_b": loss_b,
        "posterior_mean_a": a.mean(),
        "posterior_mean_b": b.mean(),
        "credible_interval_a": a.credible_interval(),
        "credible_interval_b": b.credible_interval(),
        "decision": decision,
    }
