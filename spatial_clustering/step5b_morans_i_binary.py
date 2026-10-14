import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from pathlib import Path

SPARSE_GENES = ["PDCD1", "CD8A", "CD8B", "FOXP3"]  # CD163 detection rate (33%) is reasonable enough to leave continuous
ORGANS = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]

results = []

for organ in ORGANS:
    organ_dir = Path("data") / "processed" / organ
    h5ad_files = sorted(organ_dir.glob("*.h5ad"))

    for h5ad_path in h5ad_files:
        section_id = h5ad_path.stem
        adata = sc.read_h5ad(h5ad_path)
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

        for gene in SPARSE_GENES:
            if gene not in adata.var_names:
                continue
            # Create a binary (detected vs not) version of this gene as a new .obs column
            expr = adata[:, gene].X.toarray().flatten()
            adata.obs[f"{gene}_detected"] = (expr > 0).astype(float)

        genes_present = [g for g in SPARSE_GENES if f"{g}_detected" in adata.obs.columns]
        if not genes_present:
            continue

        # squidpy's spatial_autocorr can run on .obs columns directly via the `attr` argument
        sq.gr.spatial_autocorr(adata, mode="moran", genes=[f"{g}_detected" for g in genes_present],
                                attr="obs", n_perms=100, n_jobs=1)
        moran_df = adata.uns["moranI"]

        for gene in genes_present:
            key = f"{gene}_detected"
            if key in moran_df.index:
                results.append({
                    "organ": organ, "section_id": section_id, "gene": gene,
                    "morans_i_binary": moran_df.loc[key, "I"],
                    "pval": moran_df.loc[key, "pval_norm"],
                })

results_df = pd.DataFrame(results)
results_df.to_csv("results/step5b_morans_i_binary.csv", index=False)

print("=== Binary (detected/not-detected) Moran's I, by gene ===")
print(results_df.groupby("gene")["morans_i_binary"].agg(["mean", "std", "count"]).to_string())
