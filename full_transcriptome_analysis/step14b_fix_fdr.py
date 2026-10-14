import pandas as pd
from statsmodels.stats.multitest import multipletests

de_df = pd.read_csv("results/step14_full_transcriptome_DE.csv")

valid = de_df["p"].notna()
print(f"Genes with a valid p-value: {valid.sum()} / {len(de_df)}")
print(f"Genes dropped (degenerate/all-tied, untestable): {(~valid).sum()}")

de_df.loc[valid, "FDR"] = multipletests(de_df.loc[valid, "p"], method="fdr_bh")[1]
de_df.loc[~valid, "FDR"] = None  # explicitly "not tested", not "not significant"

de_df = de_df.sort_values("FDR", na_position="last")
de_df.to_csv("results/step14_full_transcriptome_DE.csv", index=False)

sig = de_df[de_df["FDR"] < 0.05]
print(f"\nFDR < 0.05: {len(sig)}")
print(f"\nTop 20 by FDR:")
print(de_df.head(20)[["gene", "log2FC_cold_vs_hot", "FDR"]].to_string(index=False))

print(f"\n--- Genes of specific prior interest ---")
for g in ["MIF", "CD74", "CD274", "IDO1", "TGFB1", "VEGFA"]:
    if g in de_df["gene"].values:
        row = de_df[de_df["gene"] == g].iloc[0]
        print(f"{g}: log2FC={row['log2FC_cold_vs_hot']:.3f}, FDR={row['FDR']}")
