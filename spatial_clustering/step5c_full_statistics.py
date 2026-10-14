import pandas as pd
import numpy as np
from scipy.stats import kruskal, wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests

morans = pd.read_csv("results/step5_morans_i_all_organs.csv")
binary = pd.read_csv("results/step5b_morans_i_binary.csv")

genes = ["MIF", "PDCD1", "CD8A", "CD8B", "FOXP3", "CD163"]
organs = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]
dfs = [pd.read_csv(f"results/step4_panel_expression_{o}.csv") for o in organs]
combined = pd.concat(dfs)
detection_rates = {g: (combined[g] > 0).mean() * 100 for g in genes}

def safe_wilcoxon(a, b, label):
    """Paired Wilcoxon, explicitly dropping pairs where either value is NaN
    (undefined Moran's I for that section, typically from near-zero variance
    in an extremely sparse gene), reporting how many pairs were excluded."""
    a, b = np.array(a), np.array(b)
    valid = ~(np.isnan(a) | np.isnan(b))
    n_excluded = int((~valid).sum())
    a_valid, b_valid = a[valid], b[valid]
    diffs = a_valid - b_valid
    if n_excluded > 0:
        print(f"  NOTE: {label} -- excluded {n_excluded} section(s) with undefined Moran's I before testing")
    if (diffs != 0).sum() == 0:
        print(f"  NOTE: {label} -- no non-zero differences remain; cannot compute test")
        return np.nan, len(a_valid), n_excluded
    stat, p = wilcoxon(a_valid, b_valid)
    return p, len(a_valid), n_excluded

print("=" * 70)
print("PANEL A: Kruskal-Wallis test -- MIF clustering by organ")
print("=" * 70)
mif_by_organ = morans[morans["gene"] == "MIF"]
groups = [mif_by_organ[mif_by_organ["organ"] == o]["morans_i"].dropna().values for o in organs]
stat, p = kruskal(*groups)
print(f"MIF: Kruskal-Wallis H={stat:.3f}, p={p:.4f}")

print()
print("=" * 70)
print("NEW: Kruskal-Wallis test -- is clustering organ-dependent, for EVERY gene?")
print("=" * 70)
kw_results = []
for gene in genes:
    gene_data = morans[morans["gene"] == gene]
    groups = [gene_data[gene_data["organ"] == o]["morans_i"].dropna().values for o in organs]
    groups = [g for g in groups if len(g) > 0]
    stat, p = kruskal(*groups)
    kw_results.append({"gene": gene, "kruskal_H": stat, "pval": p})
kw_df = pd.DataFrame(kw_results)
kw_df["FDR"] = multipletests(kw_df["pval"], method="fdr_bh")[1]
print(kw_df.to_string(index=False))
kw_df.to_csv("results/step5c_organ_dependence_all_genes.csv", index=False)

print()
print("=" * 70)
print("PANEL B: Paired Wilcoxon tests -- MIF vs. each immune marker")
print("=" * 70)
mif_vals = morans[morans["gene"] == "MIF"][["section_id", "morans_i"]].rename(columns={"morans_i": "mif"})
panel_b_results = []
for gene in ["CD163", "CD8A", "CD8B", "PDCD1", "FOXP3"]:
    gene_vals = morans[morans["gene"] == gene][["section_id", "morans_i"]].rename(columns={"morans_i": gene})
    merged = mif_vals.merge(gene_vals, on="section_id")
    p, n_used, n_excl = safe_wilcoxon(merged["mif"], merged[gene], f"MIF vs {gene}")
    panel_b_results.append({"comparison": f"MIF vs {gene}", "n_used": n_used, "n_excluded": n_excl, "pval": p})
panel_b_df = pd.DataFrame(panel_b_results)
valid_mask = panel_b_df["pval"].notna()
panel_b_df["FDR"] = np.nan
panel_b_df.loc[valid_mask, "FDR"] = multipletests(panel_b_df.loc[valid_mask, "pval"], method="fdr_bh")[1]
print(panel_b_df.to_string(index=False))
panel_b_df.to_csv("results/step5c_panelB_stats.csv", index=False)

print()
print("=" * 70)
print("PANEL C: Spearman correlation -- detection rate vs. mean Moran's I")
print("=" * 70)
mean_moran = morans.groupby("gene")["morans_i"].mean()
rho, p = spearmanr([detection_rates[g] for g in genes], [mean_moran[g] for g in genes])
print(f"Spearman rho={rho:.3f}, p={p:.4f} (n=6 genes -- interpret with caution given small n)")

print()
print("=" * 70)
print("PANEL D: Paired Wilcoxon -- continuous vs. binary Moran's I, per sparse gene")
print("=" * 70)
sparse_genes = ["PDCD1", "CD8A", "CD8B", "FOXP3"]
panel_d_results = []
for gene in sparse_genes:
    cont = morans[morans["gene"] == gene][["section_id", "morans_i"]].rename(columns={"morans_i": "continuous"})
    binr = binary[binary["gene"] == gene][["section_id", "morans_i_binary"]].rename(columns={"morans_i_binary": "binary"})
    merged = cont.merge(binr, on="section_id")
    p, n_used, n_excl = safe_wilcoxon(merged["continuous"], merged["binary"], f"{gene} cont-vs-binary")
    panel_d_results.append({"gene": gene, "n_used": n_used, "n_excluded": n_excl,
                             "mean_continuous": merged["continuous"].mean(), "mean_binary": merged["binary"].mean(), "pval": p})
panel_d_df = pd.DataFrame(panel_d_results)
valid_mask_d = panel_d_df["pval"].notna()
panel_d_df["FDR"] = np.nan
panel_d_df.loc[valid_mask_d, "FDR"] = multipletests(panel_d_df.loc[valid_mask_d, "pval"], method="fdr_bh")[1]
print(panel_d_df.to_string(index=False))
panel_d_df.to_csv("results/step5c_panelD_stats.csv", index=False)
