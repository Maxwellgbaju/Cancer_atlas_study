import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, norm
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("results/step18_external_validation_zscores.csv")

# Convert each section-marker z-score to a two-sided p-value via normal
# approximation (squidpy's permutation-based nhood_enrichment returns a
# z-score directly; this recovers a p-value for FDR correction across
# all section-marker tests, matching this study's standard practice of
# correcting within each family of tests).
df["pval"] = 2 * (1 - norm.cdf(np.abs(df["zscore"])))
df["FDR"] = multipletests(df["pval"], method="fdr_bh")[1]
df = df.sort_values("FDR")
df.to_csv("results/step19_external_validation_FDR.csv", index=False)

print("=== All section-marker tests, FDR-corrected ===")
print(df.to_string(index=False))

print("\n=== Per-marker summary across sections (direction test) ===")
summary_rows = []
for marker in df["marker"].unique():
    sub = df[df["marker"] == marker]
    if len(sub) >= 3:
        stat, p = wilcoxon(sub["zscore"])
        n_negative = (sub["zscore"] < 0).sum()
        summary_rows.append({"marker": marker, "n_sections": len(sub), "n_negative_direction": n_negative,
                              "mean_zscore": sub["zscore"].mean(), "wilcoxon_p": p})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results/step19_marker_summary.csv", index=False)
print(summary_df.to_string(index=False))
