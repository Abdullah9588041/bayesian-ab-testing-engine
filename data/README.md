# Data

## Source (REAL data — no synthetic generation for the main analysis)

`control_group.csv` and `test_group.csv` are the original daily Facebook
marketing campaign files from the public repository
[Arvi55/A-B-Testing-Analysis-using-Python](https://github.com/Arvi55/A-B-Testing-Analysis-using-Python)
(August 2019, 30 days per campaign). They are used **unmodified**.

| File | Campaign | Rows | Impressions | Purchases | Conversion rate |
|---|---|---|---|---|---|
| `control_group.csv` | Control Campaign (variant A) | 30 | 3,177,233 | 15,161 | 0.4772% |
| `test_group.csv` | Test Campaign (variant B) | 30 | 2,237,544 | 15,637 | 0.6988% |

Columns (semicolon-separated in the originals): `Campaign Name`, `Date`
(`D.MM.YYYY`), `Spend [USD]`, `# of Impressions`, `Reach`,
`# of Website Clicks`, `# of Searches`, `# of View Content`,
`# of Add to Cart`, `# of Purchase`.

### How the data is used

Each row (one campaign-day) is treated as a batch of Bernoulli trials:

- **trials** = `# of Impressions` (number of ad exposures that day)
- **successes** = `# of Purchase` (number of purchases that day)

The 30 days, in chronological order, provide a natural sequence of batches
for the sequential-monitoring analysis (`run_sequential`): each "peek"
adds one more day of real data, exactly as a live experiment dashboard
would. `scripts/run_analysis.py` performs this cleaning and writes the
tidy file `data/clean/ab_daily.csv` with columns
`day, trials_a, successes_a, trials_b, successes_b`.

### Limitations of the dataset (stated honestly)

- The data are **daily aggregates**, not user-level randomization records;
  we cannot verify the randomization protocol or check for SRM
  (sample-ratio mismatch) at the user level.
- Campaigns may have differed in targeting/budget, not only creative —
  see `Spend [USD]` and `Reach` columns — so "B beats A" is an association
  in observational campaign data, used here to demonstrate the
  *statistical machinery*, not to claim a causal creative effect.
- Purchases are attributed per day with no attribution-window detail.
