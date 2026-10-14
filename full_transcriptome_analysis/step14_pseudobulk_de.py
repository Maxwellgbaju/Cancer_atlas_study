import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

pseudobulk = pd.read_csv("results/step13_pseudobulk_expression.csv")
meta_cols = ["section_id", "organ", "niche_type"]
gene_cols = [c for c in pseudobulk.columns if c not in meta_cols]

cold = pseudobulk[pseudobulk["niche_type"] == "immune_cold"].set_index("section_id")
hot = pseudobulk[pseudobulk["niche_type"] == "immune_hot"].set_index("section_id")

common_sections = sorted(set(cold.index) & set(hot.index))
cold = cold.loc[common_sections, gene_cols]
hot = hot.loc[common_sections, gene_cols]
print(f"Paired sections for DE test: {len(common_sections)}")

results = []
for gene in gene_cols:
    cold_vals = cold[gene].values
    hot_vals = hot[gene].values
    if np.std(cold_vals - hot_vals) == 0:
        continue
    try:
        stat, p = stats.wilcoxon(cold_vals, hot_vals)
    except ValueError:
        continue
    log2fc = np.log2(cold_vals.mean() + 1e-9) - np.log2(hot_vals.mean() + 1e-9)
    results.append({"gene": gene, "mean_cold": cold_vals.mean(), "mean_hot": hot_vals.mean(),
                     "log2FC_cold_vs_hot": log2fc, "wilcoxon_stat": stat, "p": p})

de_df = pd.DataFrame(results)
de_df["FDR"] = multipletests(de_df["p"], method="fdr_bh")[1]
de_df = de_df.sort_values("FDR")
de_df.to_csv("results/step14_full_transcriptome_DE.csv", index=False)

sig = de_df[de_df["FDR"] < 0.05]
print(f"\nGenes tested: {len(de_df)}")
print(f"FDR < 0.05: {len(sig)}")
print(f"\nTop 20 by FDR:")
print(de_df.head(20)[["gene", "log2FC_cold_vs_hot", "FDR"]].to_string(index=False))

print(f"\n--- Genes of specific prior interest ---")
for g in ["MIF", "CD74", "CD274", "IDO1", "TGFB1", "VEGFA"]:
    if g in de_df["gene"].values:
        row = de_df[de_df["gene"] == g].iloc[0]
        print(f"{g}: log2FC={row['log2FC_cold_vs_hot']:.3f}, FDR={row['FDR']:.3e}")
    else:
        print(f"{g}: not found in tested gene set")
