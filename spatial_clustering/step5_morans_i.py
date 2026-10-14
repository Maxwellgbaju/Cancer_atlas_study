import scanpy as sc
import squidpy as sq
import pandas as pd
from pathlib import Path

GENE_PANEL = ["MIF", "PDCD1", "CD8A", "CD8B", "FOXP3", "CD163"]
ORGANS = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]

results = []

for organ in ORGANS:
    organ_dir = Path("data") / "processed" / organ
    h5ad_files = sorted(organ_dir.glob("*.h5ad"))
    print(f"\n=== {organ}: {len(h5ad_files)} sections ===")

    for h5ad_path in h5ad_files:
        section_id = h5ad_path.stem
        adata = sc.read_h5ad(h5ad_path)

        # Build a spatial (physical-coordinate) neighbor graph, not an
        # expression-similarity graph.
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

        genes_present = [g for g in GENE_PANEL if g in adata.var_names]
        if not genes_present:
            print(f"  {section_id}: no panel genes found, skipping")
            continue

        sq.gr.spatial_autocorr(adata, mode="moran", genes=genes_present, n_perms=100, n_jobs=1)
        moran_df = adata.uns["moranI"]

        for gene in genes_present:
            if gene in moran_df.index:
                results.append({
                    "organ": organ,
                    "section_id": section_id,
                    "gene": gene,
                    "morans_i": moran_df.loc[gene, "I"],
                    "pval": moran_df.loc[gene, "pval_norm"],
                    "n_spots": adata.n_obs,
                })
        print(f"  {section_id}: n={adata.n_obs}, MIF Moran's I = "
              f"{moran_df.loc['MIF', 'I']:.4f} (p={moran_df.loc['MIF', 'pval_norm']:.4g})" if 'MIF' in moran_df.index else f"  {section_id}: MIF not in panel genes found")

results_df = pd.DataFrame(results)
results_df.to_csv("results/step5_morans_i_all_organs.csv", index=False)
print(f"\n\nSaved {len(results_df)} gene-section results to results/step5_morans_i_all_organs.csv")

print("\n=== MIF Moran's I summary by organ ===")
mif_summary = results_df[results_df["gene"] == "MIF"].groupby("organ")["morans_i"].agg(["mean", "std", "min", "max", "count"])
print(mif_summary.to_string())
