import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import gseapy as gp

h5ad_dir = Path("data/external_test/h5ad_only")
MIN_SPOTS = 10
pseudobulk_rows = []

for h5ad_path in sorted(h5ad_dir.glob("*.h5ad")):
    section_id = h5ad_path.stem
    adata = sc.read_h5ad(h5ad_path)

    tumor_mask = adata.obs["ground_truth"] == "TUM"
    if tumor_mask.sum() < 2 * MIN_SPOTS:
        print(f"{section_id}: too few tumor spots ({tumor_mask.sum()}), skipped")
        continue
    adata_tumor = adata[tumor_mask].copy()

    mif_expr = adata_tumor[:, "MIF"].X.toarray().flatten()
    mif_median = np.median(mif_expr)
    high_mask = mif_expr > mif_median
    low_mask = ~high_mask

    if high_mask.sum() < MIN_SPOTS or low_mask.sum() < MIN_SPOTS:
        print(f"{section_id}: insufficient spots per group after median split, skipped")
        continue

    full_expr = adata_tumor.X.toarray() if not isinstance(adata_tumor.X, np.ndarray) else adata_tumor.X
    high_pseudobulk = full_expr[high_mask].mean(axis=0)
    low_pseudobulk = full_expr[low_mask].mean(axis=0)

    pseudobulk_rows.append({"section_id": section_id, "group": "mif_high",
                             **dict(zip(adata_tumor.var_names, high_pseudobulk))})
    pseudobulk_rows.append({"section_id": section_id, "group": "mif_low",
                             **dict(zip(adata_tumor.var_names, low_pseudobulk))})
    print(f"{section_id}: tumor spots={tumor_mask.sum()}, MIF-high={high_mask.sum()}, MIF-low={low_mask.sum()}")

pseudobulk_df = pd.DataFrame(pseudobulk_rows)
pseudobulk_df.to_csv("results/step20_external_pseudobulk.csv", index=False)
print(f"\nSections with valid paired profiles: {pseudobulk_df['section_id'].nunique()}")

# --- Differential expression: paired t-test, log1p pseudobulk ---
meta_cols = ["section_id", "group"]
gene_cols = [c for c in pseudobulk_df.columns if c not in meta_cols]
high = pseudobulk_df[pseudobulk_df["group"] == "mif_high"].set_index("section_id")
low = pseudobulk_df[pseudobulk_df["group"] == "mif_low"].set_index("section_id")
common = sorted(set(high.index) & set(low.index))
high, low = high.loc[common, gene_cols], low.loc[common, gene_cols]
print(f"Paired sections for DE test: {len(common)}")

log_high, log_low = np.log1p(high), np.log1p(low)
diffs = log_high.values - log_low.values
valid = diffs.std(axis=0) > 0
t_stats, p_vals = stats.ttest_rel(log_high.values[:, valid], log_low.values[:, valid], axis=0)

de_df = pd.DataFrame({"gene": np.array(gene_cols)[valid], "t_stat": t_stats, "p": p_vals})
de_df["FDR"] = multipletests(de_df["p"], method="fdr_bh")[1]
de_df = de_df.sort_values("FDR")
de_df.to_csv("results/step20_external_DE.csv", index=False)

sig = de_df[de_df["FDR"] < 0.05]
print(f"\nGenes tested: {len(de_df)} | FDR<0.05: {len(sig)}")
print(de_df.head(15).to_string(index=False))

for g in ["MIF", "VEGFA", "SLC2A1", "HK2", "CD74", "TGFB1"]:
    if g in de_df["gene"].values:
        row = de_df[de_df["gene"] == g].iloc[0]
        print(f"{g}: t={row['t_stat']:.3f}, FDR={row['FDR']:.3e}")
    else:
        print(f"{g}: not testable (insufficient variance or not present)")

# --- GSEA preranked, Hallmark ---
rank_df = de_df.sort_values("t_stat", ascending=False)[["gene", "t_stat"]]
rank_df.columns = ["gene_name", "rank_score"]
n_unique = rank_df["rank_score"].nunique()
print(f"\nRanked genes: {len(rank_df)} | Unique values: {n_unique} ({100*n_unique/len(rank_df):.1f}%)")

res = gp.prerank(rnk=rank_df, gene_sets="MSigDB_Hallmark_2020", outdir="results/gsea_external",
                  seed=42, permutation_num=1000, min_size=10, max_size=500)
out = res.res2d.sort_values("FDR q-val")
out.to_csv("results/step20_external_GSEA.csv", index=False)

sig_gsea = out[out["FDR q-val"] < 0.05]
print(f"\n=== External cohort GSEA: pathways at FDR<0.05 ===")
print(sig_gsea[["Term", "NES", "FDR q-val"]].to_string(index=False))

print("\n=== Specific check: does Hypoxia replicate, and is MIF in its leading edge? ===")
hyp = out[out["Term"] == "Hypoxia"]
if len(hyp):
    print(hyp[["Term", "NES", "FDR q-val"]].to_string(index=False))
    leading = hyp["Lead_genes"].values[0]
    print(f"MIF in leading edge: {'YES' if 'MIF' in leading.split(';') else 'NO'}")
    print(f"Leading edge genes: {leading}")
else:
    print("Hypoxia not found in results")

for pathway in ["TGF-beta Signaling", "Epithelial Mesenchymal Transition"]:
    row = out[out["Term"] == pathway]
    if len(row):
        print(f"\n{pathway}: NES={row['NES'].values[0]:.3f}, FDR={row['FDR q-val'].values[0]:.4f}")
