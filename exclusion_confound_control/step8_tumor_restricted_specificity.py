import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from pathlib import Path

MARKERS = ["CD163", "CD8A", "CD8B", "PDCD1", "FOXP3"]
MIN_POSITIVE_SPOTS = 20
TUMOR_QUANTILE = 0.5  # EPCAM-high defines "tumor compartment"; MIF split happens WITHIN this subset
ORGANS = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]

results = []
for organ in ORGANS:
    organ_dir = Path("data") / "processed" / organ
    for h5ad_path in sorted(organ_dir.glob("*.h5ad")):
        section_id = h5ad_path.stem
        adata_full = sc.read_h5ad(h5ad_path)
        if "EPCAM" not in adata_full.var_names or "MIF" not in adata_full.var_names:
            continue

        epcam_expr = adata_full[:, "EPCAM"].X.toarray().flatten()
        is_tumor = epcam_expr > np.quantile(epcam_expr, TUMOR_QUANTILE)

        # Restrict to tumor-compartment spots only, keeping their real spatial coordinates
        adata = adata_full[is_tumor].copy()
        if adata.n_obs < 50:
            continue

        mif_expr_tumor = adata[:, "MIF"].X.toarray().flatten()
        mif_high_within_tumor = mif_expr_tumor > np.quantile(mif_expr_tumor, 0.5)

        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

        for marker in MARKERS:
            if marker not in adata.var_names:
                continue
            marker_expr = adata[:, marker].X.toarray().flatten()
            marker_pos = marker_expr > 0
            n_positive = int(marker_pos.sum())
            if n_positive < MIN_POSITIVE_SPOTS:
                continue

            category = np.full(adata.n_obs, "Neither", dtype=object)
            category[mif_high_within_tumor & ~marker_pos] = "MIF_high_only"
            category[~mif_high_within_tumor & marker_pos] = "Marker_pos_only"
            category[mif_high_within_tumor & marker_pos] = "Both"
            adata.obs["_category"] = pd.Categorical(category, categories=["MIF_high_only", "Marker_pos_only", "Both", "Neither"])
            try:
                sq.gr.nhood_enrichment(adata, cluster_key="_category", n_perms=100, seed=0, show_progress_bar=False)
                cats = list(adata.obs["_category"].cat.categories)
                zmat = adata.uns["_category_nhood_enrichment"]["zscore"]
                zscore = zmat[cats.index("MIF_high_only"), cats.index("Marker_pos_only")]
                status = "ok"
            except Exception as e:
                zscore = np.nan
                status = f"failed: {e}"

            results.append({"organ": organ, "section_id": section_id, "marker": marker,
                             "n_tumor_spots": adata.n_obs, "n_positive": n_positive, "zscore": zscore, "status": status})
    print(f"{organ}: done")

results_df = pd.DataFrame(results)
results_df.to_csv("results/step8_tumor_restricted_zscores.csv", index=False)

print("\n=== MIF exclusion z-score, WITHIN tumor compartment only (EPCAM-high spots), by marker ===")
ok = results_df[results_df["status"] == "ok"]
print(ok.groupby("marker")["zscore"].agg(["mean", "std", "count"]).to_string())

print("\n=== Comparison: original (whole-tissue) MIF result vs. tumor-restricted MIF result ===")
mif_results = pd.read_csv("results/step6_exclusion_zscores.csv")
mif_ok = mif_results[mif_results["status"] == "ok"]
for m in MARKERS:
    orig_mean = mif_ok[mif_ok["marker"] == m]["zscore"].mean()
    restricted_mean = ok[ok["marker"] == m]["zscore"].mean()
    print(f"{m:8s}: whole-tissue MIF={orig_mean:.2f}  tumor-restricted MIF={restricted_mean:.2f}")
