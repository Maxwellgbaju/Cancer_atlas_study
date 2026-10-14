import pandas as pd
import numpy as np
from scipy import stats
import gseapy as gp

pseudobulk = pd.read_csv("results/step13_pseudobulk_expression.csv")
meta_cols = ["section_id", "organ", "niche_type"]
gene_cols = [c for c in pseudobulk.columns if c not in meta_cols]

cold = pseudobulk[pseudobulk["niche_type"] == "immune_cold"].set_index("section_id")
hot = pseudobulk[pseudobulk["niche_type"] == "immune_hot"].set_index("section_id")
common = sorted(set(cold.index) & set(hot.index))
cold, hot = cold.loc[common, gene_cols], hot.loc[common, gene_cols]

# Paired t-statistic on log1p-transformed pseudobulk: a continuous statistic
# (unlike Wilcoxon's bounded integer rank-sum), standard for GSEA preranking
# (this is the same logic behind DESeq2's 'stat' column commonly fed to fgsea).
log_cold = np.log1p(cold)
log_hot = np.log1p(hot)
diffs = log_cold.values - log_hot.values
valid = diffs.std(axis=0) > 0

t_stats, _ = stats.ttest_rel(log_cold.values[:, valid], log_hot.values[:, valid], axis=0)
rank_df = pd.DataFrame({"gene": np.array(gene_cols)[valid], "rank_score": t_stats}).sort_values("rank_score", ascending=False)

n_unique = rank_df["rank_score"].nunique()
print(f"Genes ranked: {len(rank_df)} | Unique rank values: {n_unique} ({100*n_unique/len(rank_df):.1f}%)")

IMMUNE_IDENTITY_GENES = {
    "CD2","CD3D","CD3E","CD3G","CD4","CD8A","CD8B","IL7R","CD28","CD40LG",
    "LCK","CCL5","GZMA","GZMB","GZMK","PRF1","CD19","CD79A","CD79B","MS4A1",
    "CD68","CD14","ITGAM","PTPRC","FOXP3","PDCD1","CTLA4","TNFRSF4","SASH3",
    "APBB1IP","TRAF3IP3","AKNA","GPR174","TAGAP","ARHGAP4","DENND1C","MYO1F",
    "FAM78A","SH2D3C"
}
filtered_rank = rank_df[~rank_df["gene"].isin(IMMUNE_IDENTITY_GENES)]

def run_prerank(rank_series, label):
    res = gp.prerank(rnk=rank_series, gene_sets="MSigDB_Hallmark_2020",
                      outdir=f"results/gsea_{label}_v2", seed=42, permutation_num=1000,
                      min_size=10, max_size=500)
    out = res.res2d.sort_values("FDR q-val")
    out.to_csv(f"results/step15b_gsea_{label}.csv", index=False)
    return out

full_res = run_prerank(rank_df, "full")
filt_res = run_prerank(filtered_rank, "immune_filtered")

full_sig = set(full_res[full_res["FDR q-val"] < 0.05]["Term"])
filt_sig = set(filt_res[filt_res["FDR q-val"] < 0.05]["Term"])

print(f"\n--- FULL RUN (top 25 by FDR) ---")
print(full_res[["Term", "NES", "FDR q-val"]].head(25).to_string(index=False))
print(f"\n--- IMMUNE-FILTERED RUN (top 25 by FDR) ---")
print(filt_res[["Term", "NES", "FDR q-val"]].head(25).to_string(index=False))

print("\n=== COMPARISON ===")
print(f"\nRobust (both runs):")
for t in sorted(full_sig & filt_sig): print(f"  {t}")
print(f"\nComposition-driven (full run only):")
for t in sorted(full_sig - filt_sig): print(f"  {t}")
print(f"\nUnmasked by filtering (filtered run only -- closest attention):")
for t in sorted(filt_sig - full_sig): print(f"  {t}")
