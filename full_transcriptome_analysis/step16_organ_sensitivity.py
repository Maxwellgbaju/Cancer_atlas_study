import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

pseudobulk = pd.read_csv("results/step13_pseudobulk_expression.csv")
meta_cols = ["section_id", "organ", "niche_type"]
gene_cols = [c for c in pseudobulk.columns if c not in meta_cols]

WELL_REPRESENTED = ["breast", "pancreas", "ovary"]
sub = pseudobulk[pseudobulk["organ"].isin(WELL_REPRESENTED)]

cold = sub[sub["niche_type"] == "immune_cold"].set_index("section_id")
hot = sub[sub["niche_type"] == "immune_hot"].set_index("section_id")
common = sorted(set(cold.index) & set(hot.index))
cold, hot = cold.loc[common, gene_cols], hot.loc[common, gene_cols]
print(f"Restricted to breast/pancreas/ovary only: {len(common)} paired sections")

log_cold, log_hot = np.log1p(cold), np.log1p(hot)
diffs = log_cold.values - log_hot.values
valid = diffs.std(axis=0) > 0
t_stats, p_vals = stats.ttest_rel(log_cold.values[:, valid], log_hot.values[:, valid], axis=0)

restricted_genes = np.array(gene_cols)[valid]
for g in ["MIF", "VEGFA", "SLC2A1", "HK2", "CD74"]:
    if g in restricted_genes:
        idx = list(restricted_genes).index(g)
        print(f"{g}: t={t_stats[idx]:.3f}, p={p_vals[idx]:.2e}")
    else:
        print(f"{g}: not testable in this subset")
