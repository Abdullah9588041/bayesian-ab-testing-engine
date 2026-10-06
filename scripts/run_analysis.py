"""End-to-end Bayesian A/B testing analysis on real campaign data.

Pipeline:
  1. Load + clean the raw daily campaign CSVs -> data/clean/ab_daily.csv
  2. Fit Beta-Binomial posteriors (uniform prior) and compute the
     decision summary: P(B > A), expected losses, credible intervals
  3. Sequential monitoring over the 30 campaign days (early-stopping rule)
  4. Frequentist comparison (z-test, chi-square, CI)
  5. Prior sensitivity analysis
  6. Sample-size planning for a future experiment
  7. Peeking-bias simulation (H0 and H1)
  8. Figures -> results/figures/ ; all numbers -> results/metrics.json

Reproducibility: every stochastic step uses a fixed seed (SEED = 42).
Run:  python scripts/run_analysis.py   (from the repository root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayes_ab import (  # noqa: E402
    BetaBinomialModel,
    bayesian_assurance,
    chi_square_test,
    decision_summary,
    expected_loss,
    find_sample_size_for_assurance,
    frequentist_sample_size,
    prior_sensitivity,
    prob_b_beats_a,
    run_sequential,
    simulate_peeking_bias,
    two_proportion_ztest,
)
from bayes_ab.visualization import (  # noqa: E402
    plot_decision_dashboard,
    plot_peeking_comparison,
    plot_posteriors,
    plot_prior_sensitivity,
    plot_sequential_trajectory,
)

SEED = 42
FIG_DIR = ROOT / "results" / "figures"
CLEAN_DIR = ROOT / "data" / "clean"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean() -> pd.DataFrame:
    """Read the raw campaign files and build the tidy daily A/B table."""
    frames = []
    for fname, variant in (("control_group.csv", "A"), ("test_group.csv", "B")):
        df = pd.read_csv(ROOT / "data" / fname, sep=";")
        df = df.rename(
            columns={
                "Date": "date",
                "# of Impressions": "impressions",
                "# of Purchase": "purchases",
            }
        )
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        frames.append(
            df[["date", "impressions", "purchases"]].assign(variant=variant)
        )
    wide = (
        pd.concat(frames)
        .pivot_table(
            index="date",
            columns="variant",
            values=["impressions", "purchases"],
            aggfunc="sum",
        )
        .sort_index()
    )
    daily = pd.DataFrame(
        {
            "day": wide.index.strftime("%Y-%m-%d"),
            "trials_a": wide[("impressions", "A")].astype(int).to_numpy(),
            "successes_a": wide[("purchases", "A")].astype(int).to_numpy(),
            "trials_b": wide[("impressions", "B")].astype(int).to_numpy(),
            "successes_b": wide[("purchases", "B")].astype(int).to_numpy(),
        }
    )
    daily.to_csv(CLEAN_DIR / "ab_daily.csv", index=False)
    return daily


def main() -> None:
    metrics: dict = {"seed": SEED}

    # 1. Data ----------------------------------------------------------------
    daily = load_and_clean()
    n_a = int(daily["trials_a"].sum())
    s_a = int(daily["successes_a"].sum())
    n_b = int(daily["trials_b"].sum())
    s_b = int(daily["successes_b"].sum())
    metrics["data"] = {
        "n_days": len(daily),
        "variant_a": {"trials": n_a, "successes": s_a, "rate": s_a / n_a},
        "variant_b": {"trials": n_b, "successes": s_b, "rate": s_b / n_b},
    }
    print(f"A (control): {s_a:,}/{n_a:,} = {s_a/n_a:.4%}")
    print(f"B (test):    {s_b:,}/{n_b:,} = {s_b/n_b:.4%}")

    # 2. Bayesian model (uniform prior) --------------------------------------
    a = BetaBinomialModel.fit(s_a, n_a, 1.0, 1.0)
    b = BetaBinomialModel.fit(s_b, n_b, 1.0, 1.0)
    prob = prob_b_beats_a(a, b)
    summary = decision_summary(a, b, prob_threshold=0.95)
    summary["credible_interval_a"] = list(summary["credible_interval_a"])
    summary["credible_interval_b"] = list(summary["credible_interval_b"])
    metrics["bayesian_uniform_prior"] = summary
    print(f"\nP(B > A) = {prob:.6f}")
    print(f"Decision: {summary['decision']}")
    print(f"E[loss|choose B] = {summary['expected_loss_choose_b']:.8f}")
    plot_posteriors(a, b, prob, str(FIG_DIR / "posteriors.png"))

    # 3. Sequential monitoring over the 30 campaign days ---------------------
    traj = run_sequential(daily, prob_threshold=0.95, loss_threshold=0.0005,
                          min_days=3)
    traj.to_csv(ROOT / "results" / "sequential_trajectory.csv", index=False)
    stop_row = traj.iloc[-1]
    metrics["sequential"] = {
        "days_until_decision": int(len(traj)),
        "final_decision": stop_row["decision"],
        "final_prob_b_beats_a": float(stop_row["prob_b_beats_a"]),
        "final_expected_loss_b": float(stop_row["expected_loss_b"]),
        "data_used_fraction": float(
            (stop_row["cum_trials_a"] + stop_row["cum_trials_b"]) / (n_a + n_b)
        ),
    }
    print(f"\nSequential: stopped on day {len(traj)} -> {stop_row['decision']}")
    plot_sequential_trajectory(traj, str(FIG_DIR / "sequential_trajectory.png"))
    plot_decision_dashboard(summary, traj, str(FIG_DIR / "decision_dashboard.png"))

    # 4. Frequentist comparison ----------------------------------------------
    z = two_proportion_ztest(s_a, n_a, s_b, n_b)
    chi2 = chi_square_test(s_a, n_a, s_b, n_b)
    metrics["frequentist"] = {"two_proportion_ztest": z, "chi_square": chi2}
    print(f"\nFrequentist z-test: z = {z['z_statistic']:.3f}, "
          f"p = {z['p_value']:.3e}")
    print(f"Chi-square: chi2 = {chi2['chi2_statistic']:.2f}, "
          f"p = {chi2['p_value']:.3e}")

    # 5. Prior sensitivity ----------------------------------------------------
    sens = prior_sensitivity(s_a, n_a, s_b, n_b)
    sens.to_csv(ROOT / "results" / "prior_sensitivity.csv", index=False)
    metrics["prior_sensitivity"] = sens.drop(columns=["description"]).to_dict(
        orient="records"
    )
    print("\nPrior sensitivity:")
    print(sens[["prior", "prob_b_beats_a", "expected_loss_choose_b"]]
          .to_string(index=False))
    plot_prior_sensitivity(sens, str(FIG_DIR / "prior_sensitivity.png"))

    # 6. Sample-size planning for a future experiment -------------------------
    # Detect a 0.2pp lift over a 0.5% baseline (business-relevant scale).
    n_freq = frequentist_sample_size(0.005, 0.007, alpha=0.05, power=0.8)
    assurance = find_sample_size_for_assurance(
        0.005, 0.007, target_assurance=0.8,
        candidate_ns=[5_000, 10_000, 25_000, 50_000, 100_000], seed=SEED,
    )
    metrics["sample_size_planning"] = {
        "scenario": "detect 0.5% -> 0.7% lift, alpha=0.05",
        "frequentist_power_80pct_n_per_variant": n_freq,
        "bayesian_assurance_target_0.8": {
            "recommended_n_per_variant": assurance["recommended_n_per_variant"],
            "grid": [
                {"n": g["n_per_variant"], "assurance": g["assurance"]}
                for g in assurance["grid"]
            ],
        },
    }
    print(f"\nSample size (frequentist, 80% power): {n_freq:,}/variant")
    print("Bayesian assurance grid:",
          [(g["n_per_variant"], round(g["assurance"], 3))
           for g in assurance["grid"]])

    # 7. Peeking-bias simulation ----------------------------------------------
    # 10 peeks: enough repeated looks for the naive-p-value inflation to
    # show clearly, at the same per-variant scale as the real experiment.
    peek_h0 = simulate_peeking_bias(
        p_a=0.0057, p_b=0.0057, n_per_peek=50_000, n_peeks=10,
        n_sims=2000, seed=SEED,
    )
    peek_h1 = simulate_peeking_bias(
        p_a=0.0048, p_b=0.0069, n_per_peek=50_000, n_peeks=10,
        n_sims=2000, seed=SEED,
    )
    metrics["peeking_simulation"] = {"h0": peek_h0, "h1": peek_h1}
    print("\nPeeking simulation (H0, no true difference):")
    for k, v in peek_h0.items():
        print(f"  {k}: {v}")
    print("Peeking simulation (H1, true lift 0.48% -> 0.69%):")
    for k, v in peek_h1.items():
        print(f"  {k}: {v}")
    plot_peeking_comparison({"h0": peek_h0, "h1": peek_h1},
                            str(FIG_DIR / "peeking_comparison.png"))

    # 8. Persist metrics -------------------------------------------------------
    with open(ROOT / "results" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nWrote results/metrics.json and figures to", FIG_DIR)


if __name__ == "__main__":
    main()
