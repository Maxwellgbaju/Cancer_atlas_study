import pandas as pd
import numpy as np

pseudobulk = pd.read_csv("results/step20_external_pseudobulk.csv")
de_df = pd.read_csv("results/step20_external_DE.csv")

# --- Check 1: organ-split direction (kidney vs lung separately) ---
pseudobulk["organ"] = pseudobulk["section_id"].str[:2].map({"KC": "kidney", "LC": "lung"})
print("=== Sections per organ ===")
print(pseudobulk.drop_duplicates("section_id")["organ"].value_counts())

for gene in ["MIF", "VEGFA", "SLC2A1"]:
    print(f"\n--- {gene}: mean expression by organ and group ---")
    sub = pseudobulk[["section_id", "organ", "group", gene]]
    print(sub.pivot_table(index=["organ", "section_id"], columns="group", values=gene))

# --- Check 2: immune-gene-filtered GSEA re-run ---
IMMUNE_IDENTITY_GENES = {
    "CD2","CD3D","CD3E","CD3G","CD4","CD8A","CD8B","IL7R","CD28","CD40LG",
    "LCK","CCL5","GZMA","GZMB","GZMK","PRF1","CD19","CD79A","CD79B","MS4A1",
    "CD68","CD14","ITGAM","PTPRC","FOXP3","PDCD1","CTLA4","TNFRSF4","SASH3",
    "APBB1IP","TRAF3IP3","AKNA","GPR174","TAGAP","ARHGAP4","DENND1C","MYO1F",
    "FAM78A","SH2D3C"
}
n_flagged = de_df[de_df["gene"].isin(IMMUNE_IDENTITY_GENES)]
print(f"\nImmune-identity genes present in this ranked list: {len(n_flagged)}")
filtered_rank = de_df[~de_df["gene"].isin(IMMUNE_IDENTITY_GENES)].sort_values("t_stat", ascending=False)[["gene", "t_stat"]]
filtered_rank.columns = ["gene_name", "rank_score"]
filtered_rank.to_csv("results/step21_external_filtered_rank.csv", index=False)
print(f"Filtered ranking saved: {len(filtered_rank)} genes")
