import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from pathlib import Path

MARKERS = ["CD163", "CD8A", "CD8B", "PDCD1", "FOXP3"]
# Confidence tier reflects each marker's detection rate (established in Step 5):
# CD163/CD8A are well-detected; the rest are sparse and should be interpreted cautiously.
HIGH_CONFIDENCE = {"CD163", "CD8A"}
MIN_POSITIVE_SPOTS = 20  # minimum marker-positive spots required to trust a result
MIF_HIGH_QUANTILE = 0.75  # top 25% of MIF expression within each section = "MIF-high"

ORGANS = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]

results = []

for organ in ORGANS:
    organ_dir = Path("data") / "processed" / organ
    h5ad_files = sorted(organ_dir.glob("*.h5ad"))
    print(f"\n=== {organ}: {len(h5ad_files)} sections ===")

    for h5ad_path in h5ad_files:
        section_id = h5ad_path.stem
        adata = sc.read_h5ad(h5ad_path)

        if "MIF" not in adata.var_names:
            continue
        mif_expr = adata[:, "MIF"].X.toarray().flatten()
        mif_threshold = np.quantile(mif_expr, MIF_HIGH_QUANTILE)
        mif_high = mif_expr > mif_threshold

        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

        for marker in MARKERS:
            if marker not in adata.var_names:
                continue
            marker_expr = adata[:, marker].X.toarray().flatten()
            marker_pos = marker_expr > 0
            n_positive = int(marker_pos.sum())

            if n_positive < MIN_POSITIVE_SPOTS:
                results.append({
                    "organ": organ, "section_id": section_id, "marker": marker,
                    "confidence": "high" if marker in HIGH_CONFIDENCE else "low",
                    "n_positive": n_positive, "n_spots": adata.n_obs,
                    "zscore": np.nan, "status": "insufficient_positive_spots",
                })
                continue

            # Four mutually exclusive categories, matching the earlier design mockup
            category = np.full(adata.n_obs, "Neither", dtype=object)
            category[mif_high & ~marker_pos] = "MIF_high_only"
            category[~mif_high & marker_pos] = "Marker_pos_only"
            category[mif_high & marker_pos] = "Both"
            adata.obs["_category"] = pd.Categorical(category, categories=["MIF_high_only", "Marker_pos_only", "Both", "Neither"])

            try:
                sq.gr.nhood_enrichment(adata, cluster_key="_category", n_perms=100, seed=0, show_progress_bar=False)
                cats = list(adata.obs["_category"].cat.categories)
                zmat = adata.uns["_category_nhood_enrichment"]["zscore"]
                i, j = cats.index("MIF_high_only"), cats.index("Marker_pos_only")
                zscore = zmat[i, j]
                status = "ok"
            except Exception as e:
                zscore = np.nan
                status = f"failed: {e}"

            results.append({
                "organ": organ, "section_id": section_id, "marker": marker,
                "confidence": "high" if marker in HIGH_CONFIDENCE else "low",
                "n_positive": n_positive, "n_spots": adata.n_obs,
                "zscore": zscore, "status": status,
            })

        print(f"  {section_id}: done")

results_df = pd.DataFrame(results)
results_df.to_csv("results/step6_exclusion_zscores.csv", index=False)
print(f"\n\nSaved {len(results_df)} section-marker results to results/step6_exclusion_zscores.csv")

print("\n=== Summary: mean exclusion z-score by marker (negative = exclusion, positive = co-localization) ===")
valid = results_df[results_df["status"] == "ok"]
summary = valid.groupby(["marker", "confidence"])["zscore"].agg(["mean", "std", "count"])
print(summary.to_string())

print(f"\nSections excluded for insufficient positive spots (< {MIN_POSITIVE_SPOTS}), by marker:")
print(results_df[results_df["status"] == "insufficient_positive_spots"].groupby("marker").size().to_string())
