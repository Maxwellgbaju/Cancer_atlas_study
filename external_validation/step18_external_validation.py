import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

IMMUNE_MARKERS = ["PDCD1", "CD8A", "CD8B", "FOXP3", "CD163"]
MIN_MARKER_POS_SPOTS = 20
h5ad_dir = Path("data/external_test/h5ad_only")

results = []
for h5ad_path in sorted(h5ad_dir.glob("*.h5ad")):
    section_id = h5ad_path.stem
    adata = sc.read_h5ad(h5ad_path)

    # Tumor-restricted compartment, using real pathologist ground truth
    # instead of an EPCAM-derived median split, which this dataset
    # uniquely allows us to do.
    tumor_mask = adata.obs["ground_truth"] == "TUM"
    if tumor_mask.sum() < 20:
        print(f"{section_id}: too few ground-truth tumor spots ({tumor_mask.sum()}), skipped")
        continue
    adata_tumor = adata[tumor_mask].copy()

    sq.gr.spatial_neighbors(adata_tumor, coord_type="generic", n_neighs=6)

    mif_expr = adata_tumor[:, "MIF"].X.toarray().flatten()
    mif_median = np.median(mif_expr)
    mif_high = mif_expr > mif_median

    for marker in IMMUNE_MARKERS:
        if marker not in adata_tumor.var_names:
            continue
        marker_expr = adata_tumor[:, marker].X.toarray().flatten()
        marker_pos = marker_expr > 0

        if marker_pos.sum() < MIN_MARKER_POS_SPOTS:
            continue

        category = np.where(mif_high & marker_pos, "both",
                    np.where(mif_high, "MIF_high_only",
                    np.where(marker_pos, "marker_pos_only", "neither")))
        adata_tumor.obs["category"] = pd.Categorical(category)

        try:
            sq.gr.nhood_enrichment(adata_tumor, cluster_key="category", n_perms=100, seed=0, show_progress_bar=False)
            cats = adata_tumor.obs["category"].cat.categories.tolist()
            if "MIF_high_only" in cats and "marker_pos_only" in cats:
                i, j = cats.index("MIF_high_only"), cats.index("marker_pos_only")
                z = adata_tumor.uns["category_nhood_enrichment"]["zscore"][i, j]
                results.append({"section_id": section_id, "cancer_type": "kidney" if section_id.startswith("KC") else "lung",
                                 "marker": marker, "n_marker_pos": int(marker_pos.sum()),
                                 "n_mif_high": int(mif_high.sum()), "zscore": z})
        except Exception as e:
            print(f"{section_id}/{marker}: enrichment failed ({e})")

    print(f"{section_id}: done")

results_df = pd.DataFrame(results)
results_df.to_csv("results/step18_external_validation_zscores.csv", index=False)
print(f"\n{results_df.to_string(index=False)}")

print("\n=== Per-marker summary across sections ===")
for marker in IMMUNE_MARKERS:
    sub = results_df[results_df["marker"] == marker]
    if len(sub) >= 3:
        stat, p = wilcoxon(sub["zscore"])
        print(f"{marker}: n_sections={len(sub)}, mean_z={sub['zscore'].mean():.2f}, Wilcoxon p={p:.4f}")
    else:
        print(f"{marker}: n_sections={len(sub)} (too few for a test)")
