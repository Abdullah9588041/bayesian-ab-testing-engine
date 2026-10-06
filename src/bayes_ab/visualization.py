"""Publication-style figures for the Bayesian A/B testing analysis."""

from __future__ import annotations

from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import BetaBinomialModel

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)

COLOR_A = "#3b82f6"  # blue: control
COLOR_B = "#ef4444"  # red: test


def plot_posteriors(
    a: BetaBinomialModel,
    b: BetaBinomialModel,
    prob_b_beats_a: float,
    path: str,
    label_a: str = "A (control)",
    label_b: str = "B (test)",
) -> None:
    """Overlapping posterior densities with means, credible intervals, P(B>A)."""
    lo = min(a.credible_interval()[0], b.credible_interval()[0])
    hi = max(a.credible_interval()[1], b.credible_interval()[1])
    pad = 0.25 * (hi - lo)
    x = np.linspace(max(0.0, lo - pad), hi + pad, 600)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, a.pdf(x), color=COLOR_A, lw=2, label=f"{label_a} posterior")
    ax.plot(x, b.pdf(x), color=COLOR_B, lw=2, label=f"{label_b} posterior")
    ax.fill_between(x, a.pdf(x), alpha=0.15, color=COLOR_A)
    ax.fill_between(x, b.pdf(x), alpha=0.15, color=COLOR_B)
    for model, color in ((a, COLOR_A), (b, COLOR_B)):
        lo_i, hi_i = model.credible_interval()
        ax.axvline(model.mean(), color=color, ls="--", lw=1)
        ax.hlines(0, lo_i, hi_i, color=color, lw=4, alpha=0.6)
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Posterior density")
    ax.set_title(f"Posterior conversion rates — P(B > A) = {prob_b_beats_a:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sequential_trajectory(traj: pd.DataFrame, path: str) -> None:
    """P(B > A) and expected loss of the leader as data accumulates."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    days = traj["day"]

    ax1.plot(days, traj["prob_b_beats_a"], color="#7c3aed", lw=2)
    ax1.axhline(0.95, color="k", ls="--", lw=1, label="stop threshold (0.95)")
    ax1.axhline(0.05, color="k", ls="--", lw=1)
    ax1.axhline(0.5, color="gray", ls=":", lw=1)
    ax1.fill_between(days, 0.95, 1.0, color="green", alpha=0.08)
    ax1.fill_between(days, 0.0, 0.05, color="green", alpha=0.08)
    ax1.set_ylabel("P(B > A | data)")
    ax1.set_title("Sequential monitoring: posterior probability over time")
    ax1.legend()
    ax1.set_ylim(0, 1)

    leader_loss = np.minimum(traj["expected_loss_a"], traj["expected_loss_b"])
    ax2.plot(days, leader_loss * 100, color="#0d9488", lw=2)
    ax2.set_xlabel("Day of experiment")
    ax2.set_ylabel("Expected loss of leader (pp)")
    ax2.set_title("Expected loss of the leading variant over time")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_decision_dashboard(
    summary: Dict, traj: pd.DataFrame, path: str
) -> None:
    """Single-figure dashboard: verdict, key numbers, trajectory, losses."""
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    verdict = summary["decision"]
    ax.text(
        0.5,
        0.55,
        verdict.upper(),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        color="#15803d" if "B" in verdict else "#b45309",
    )
    ax.text(
        0.5,
        0.25,
        f"P(B > A) = {summary['prob_b_beats_a']:.4f}   |   "
        f"E[loss | choose B] = {summary['expected_loss_choose_b']:.6f}   |   "
        f"E[loss | choose A] = {summary['expected_loss_choose_a']:.6f}",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_title("Bayesian A/B test — decision dashboard", fontsize=14, weight="bold")

    ax = fig.add_subplot(gs[1, 0])
    days = traj["day"]
    ax.plot(days, traj["prob_b_beats_a"], color="#7c3aed", lw=2)
    ax.axhline(0.95, color="k", ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Day")
    ax.set_ylabel("P(B > A)")

    ax = fig.add_subplot(gs[1, 1])
    labels = ["Choose A", "Choose B"]
    losses = [summary["expected_loss_choose_a"], summary["expected_loss_choose_b"]]
    ax.bar(labels, losses, color=[COLOR_A, COLOR_B], alpha=0.7)
    ax.set_ylabel("Expected loss (conversion rate)")
    ax.set_title("Expected loss per decision")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_peeking_comparison(results: Dict[str, Dict[str, float]], path: str) -> None:
    """False-positive / decision rates under H0 vs H1 for the three strategies."""
    labels = ["Fixed-sample\nfrequentist", "Naive peeking\nfrequentist", "Bayesian\nsequential"]
    h0 = [results["h0"]["fixed_sample_frequentist_reject_rate"],
          results["h0"]["naive_peeking_frequentist_reject_rate"],
          results["h0"]["bayesian_sequential_decide_b_rate"]]
    h1 = [results["h1"]["fixed_sample_frequentist_reject_rate"],
          results["h1"]["naive_peeking_frequentist_reject_rate"],
          results["h1"]["bayesian_sequential_decide_b_rate"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, h0, 0.4, label="H0: no difference (false-positive rate)", color="#f59e0b")
    ax.bar(x + 0.2, h1, 0.4, label="H1: real difference (power)", color="#10b981")
    ax.axhline(0.05, color="k", ls="--", lw=1, label="nominal α = 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Observed rate over simulated experiments")
    ax.set_title("Peeking bias: naive repeated testing vs Bayesian sequential rule")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_prior_sensitivity(sens: pd.DataFrame, path: str) -> None:
    """P(B > A) under each prior — visual robustness check."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(sens["prior"], sens["prob_b_beats_a"], color="#6366f1", alpha=0.8)
    ax.axvline(0.95, color="k", ls="--", lw=1, label="decision threshold")
    ax.set_xlabel("P(B > A)")
    ax.set_title("Prior sensitivity: decision probability across priors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
