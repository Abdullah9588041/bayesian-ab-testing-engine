"""bayes_ab: Bayesian A/B testing engine.

Beta-Binomial conjugate models, sequential early-stopping rules,
frequentist comparisons, sample-size planning, prior sensitivity
analysis, and visualization.
"""

from .frequentist import chi_square_test, confidence_interval_diff, two_proportion_ztest
from .models import (
    BetaBinomialModel,
    decision_summary,
    expected_loss,
    prob_b_beats_a,
)
from .priors import STANDARD_PRIORS, prior_sensitivity
from .sample_size import (
    bayesian_assurance,
    find_sample_size_for_assurance,
    frequentist_sample_size,
)
from .sequential import run_sequential, simulate_peeking_bias

__all__ = [
    "BetaBinomialModel",
    "decision_summary",
    "expected_loss",
    "prob_b_beats_a",
    "STANDARD_PRIORS",
    "prior_sensitivity",
    "bayesian_assurance",
    "find_sample_size_for_assurance",
    "frequentist_sample_size",
    "run_sequential",
    "simulate_peeking_bias",
    "two_proportion_ztest",
    "chi_square_test",
    "confidence_interval_diff",
]

__version__ = "1.0.0"
