import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from pathlib import Path

CONTROL_GENE = "EPCAM"  # generic epithelial/tumor marker, no known immune-suppressive role
MARKERS = ["CD163", "CD8A", "CD8B", "PDCD1", "FOXP3"]
MIN_POSITIVE_SPOTS = 20
HIGH_QUANTILE = 0.75
ORGANS = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]

results = []
for organ in ORGANS:
    organ_dir = Path("data") / "processed" / organ
    for h5ad_path in sorted(organ_dir.glob("*.h5ad")):
        section_id = h5ad_path.stem
        adata = sc.read_h5ad(h5ad_path)
        if CONTROL_GENE not in adata.var_names:
            continue
        ctrl_expr = adata[:, CONTROL_GENE].X.toarray().flatten()
        ctrl_high = ctrl_expr > np.quantile(ctrl_expr, HIGH_QUANTILE)
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
            category[ctrl_high & ~marker_pos] = "Ctrl_high_only"
            category[~ctrl_high & marker_pos] = "Marker_pos_only"
            category[ctrl_high & marker_pos] = "Both"
            adata.obs["_category"] = pd.Categorical(category, categories=["Ctrl_high_only", "Marker_pos_only", "Both", "Neither"])
            try:
                sq.gr.nhood_enrichment(adata, cluster_key="_category", n_perms=100, seed=0, show_progress_bar=False)
                cats = list(adata.obs["_category"].cat.categories)
                zmat = adata.uns["_category_nhood_enrichment"]["zscore"]
                zscore = zmat[cats.index("Ctrl_high_only"), cats.index("Marker_pos_only")]
                status = "ok"
            except Exception as e:
                zscore = np.nan
                status = f"failed: {e}"

            results.append({"organ": organ, "section_id": section_id, "marker": marker,
                             "n_positive": n_positive, "zscore": zscore, "status": status})
    print(f"{organ}: done")

results_df = pd.DataFrame(results)
results_df.to_csv("results/step7_epcam_control_zscores.csv", index=False)

print("\n=== EPCAM (control) exclusion z-score, by marker ===")
ok = results_df[results_df["status"] == "ok"]
print(ok.groupby("marker")["zscore"].agg(["mean", "std", "count"]).to_string())

print("\n=== Direct comparison: MIF vs. EPCAM exclusion strength, same markers ===")
mif_results = pd.read_csv("results/step6_exclusion_zscores.csv")
mif_ok = mif_results[mif_results["status"] == "ok"]
for m in MARKERS:
    mif_mean = mif_ok[mif_ok["marker"] == m]["zscore"].mean()
    epcam_mean = ok[ok["marker"] == m]["zscore"].mean()
    print(f"{m:8s}: MIF={mif_mean:.2f}  EPCAM={epcam_mean:.2f}")
