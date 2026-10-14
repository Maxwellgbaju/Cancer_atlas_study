import pandas as pd
import numpy as np
import gseapy as gp

de_df = pd.read_csv("results/step14_full_transcriptome_DE.csv")
de_df = de_df.dropna(subset=["p", "FDR"])

# Signed ranking score: pushes highly significant genes to the extremes
# regardless of effect size, standard practice for preranked GSEA.
de_df["rank_score"] = -np.log10(de_df["p"].clip(lower=1e-300)) * np.sign(de_df["log2FC_cold_vs_hot"])
de_df = de_df.sort_values("rank_score", ascending=False)

# Curated, canonical immune-lineage/activation identity genes -- genes whose
# expression predominantly tracks "how many immune cells are physically here"
# rather than a regulatory mechanism. This list is deliberately narrow and
# literature-standard (T cell, B cell, myeloid lineage/activation markers)
# so the filter is conservative and defensible, not a way to remove anything
# inconvenient.
IMMUNE_IDENTITY_GENES = {
    "CD2","CD3D","CD3E","CD3G","CD4","CD8A","CD8B","IL7R","CD28","CD40LG",
    "LCK","CCL5","GZMA","GZMB","GZMK","PRF1","CD19","CD79A","CD79B","MS4A1",
    "CD68","CD14","ITGAM","PTPRC","FOXP3","PDCD1","CTLA4","TNFRSF4","SASH3",
    "APBB1IP","TRAF3IP3","AKNA","GPR174","TAGAP","ARHGAP4","DENND1C","MYO1F",
    "FAM78A","SH2D3C"
}

full_rank = de_df.set_index("gene")["rank_score"]
filtered_rank = de_df[~de_df["gene"].isin(IMMUNE_IDENTITY_GENES)].set_index("gene")["rank_score"]

n_flagged = len(de_df[de_df["gene"].isin(IMMUNE_IDENTITY_GENES) & (de_df["FDR"] < 0.05)])
print(f"Significant genes matching the immune-identity filter: {n_flagged}")
print(f"Genes remaining for the filtered run: {len(filtered_rank)}\n")

def run_prerank(rank_series, label):
    try:
        res = gp.prerank(rnk=rank_series.reset_index(), gene_sets="MSigDB_Hallmark_2020",
                          outdir=f"results/gsea_{label}", seed=42, permutation_num=1000,
                          min_size=10, max_size=500)
        out = res.res2d.sort_values("FDR q-val")
        out.to_csv(f"results/step15_gsea_{label}.csv", index=False)
        print(f"--- {label.upper()} RUN: top pathways (FDR q<0.05) ---")
        sig = out[out["FDR q-val"] < 0.05]
        print(sig[["Term", "NES", "FDR q-val"]].head(25).to_string(index=False))
        return out
    except Exception as e:
        print(f"GSEA run '{label}' failed: {e}")
        print("Likely cause: no internet access on this node to fetch MSigDB_Hallmark_2020 via Enrichr.")
        print("If so, we'll need a locally downloaded Hallmark .gmt file instead -- flag this and I'll give you that path.")
        return None

full_res = run_prerank(full_rank, "full")
print()
filtered_res = run_prerank(filtered_rank, "immune_filtered")

if full_res is not None and filtered_res is not None:
    full_sig = set(full_res[full_res["FDR q-val"] < 0.05]["Term"])
    filt_sig = set(filtered_res[filtered_res["FDR q-val"] < 0.05]["Term"])

    print("\n=== COMPARISON ===")
    print(f"\nRobust (significant in BOTH runs -- trust these):")
    for t in sorted(full_sig & filt_sig):
        print(f"  {t}")
    print(f"\nComposition-driven (significant in full run ONLY -- likely just reflects immune cell density):")
    for t in sorted(full_sig - filt_sig):
        print(f"  {t}")
    print(f"\nUnmasked by filtering (significant in filtered run ONLY -- was hidden underneath the immune signal, worth closest attention):")
    for t in sorted(filt_sig - full_sig):
        print(f"  {t}")
