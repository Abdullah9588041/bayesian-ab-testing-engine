"""Tests for prior sensitivity analysis."""

import pandas as pd

from bayes_ab.priors import STANDARD_PRIORS, prior_sensitivity


def test_standard_priors_well_formed():
    assert set(STANDARD_PRIORS) == {"uniform", "jeffreys", "historical"}
    for name, (a0, b0, desc) in STANDARD_PRIORS.items():
        assert a0 > 0 and b0 > 0 and isinstance(desc, str)
    assert STANDARD_PRIORS["historical"][0] / sum(STANDARD_PRIORS["historical"][:2]) == (
        5 / 1000
    )


def test_prior_sensitivity_frame():
    sens = prior_sensitivity(48, 1_000, 72, 1_000)
    assert isinstance(sens, pd.DataFrame)
    assert list(sens["prior"]) == ["uniform", "jeffreys", "historical"]
    assert ((sens["prob_b_beats_a"] >= 0) & (sens["prob_b_beats_a"] <= 1)).all()
    assert (sens["expected_loss_choose_b"] >= 0).all()


def test_priors_converge_with_big_data():
    """With 5M observations the prior is irrelevant: probabilities agree."""
    sens = prior_sensitivity(24_000, 5_000_000, 35_000, 5_000_000)
    spread = sens["prob_b_beats_a"].max() - sens["prob_b_beats_a"].min()
    assert spread < 1e-6


def test_prior_matters_with_tiny_data():
    """With almost no data the informative prior pulls the conclusion."""
    sens = prior_sensitivity(1, 10, 2, 10)
    spread = sens["prob_b_beats_a"].max() - sens["prob_b_beats_a"].min()
    assert spread > 0.01
