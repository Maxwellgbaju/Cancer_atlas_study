import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

niches = pd.read_csv("results/step11_niche_assignments.csv")
IMMUNE_COLD_NICHE = 5
IMMUNE_HOT_NICHE = 1
MIN_SPOTS = 10

organs = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]
pseudobulk_rows = []
validation_log = []

for organ in organs:
    organ_dir = Path("data") / "processed" / organ
    for h5ad_path in sorted(organ_dir.glob("*.h5ad")):
        section_id = h5ad_path.stem
        section_niches = niches[niches["section_id"] == section_id].reset_index(drop=True)
        if len(section_niches) == 0:
            continue

        adata = sc.read_h5ad(h5ad_path)

        # Alignment validation: row order must match between the saved niche
        # assignment and a fresh reload of this section's h5ad, since no
        # explicit barcode was saved in the original niche-assignment run.
        if adata.n_obs != len(section_niches):
            validation_log.append(f"{section_id}: SPOT COUNT MISMATCH ({adata.n_obs} vs {len(section_niches)}) -- SKIPPED")
            continue
        if "MIF" in adata.var_names:
            fresh_mif = adata[:, "MIF"].X.toarray().flatten()
            # Reconstruct the same neighbor-smoothing used in step11 for a spot check
            import squidpy as sq
            sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
            conn = adata.obsp["spatial_connectivities"]
            neighbor_sum = conn.dot(fresh_mif.reshape(-1, 1)).flatten()
            neighbor_count = np.asarray(conn.sum(axis=1)).flatten()
            neighbor_count[neighbor_count == 0] = 1
            fresh_smoothed_mif = (fresh_mif + neighbor_sum) / (1 + neighbor_count)
            corr = np.corrcoef(fresh_smoothed_mif, section_niches["MIF"].values)[0, 1]
            if corr < 0.99:
                validation_log.append(f"{section_id}: ALIGNMENT CHECK FAILED (corr={corr:.3f}) -- SKIPPED")
                continue

        cold_mask = (section_niches["niche"] == IMMUNE_COLD_NICHE).values
        hot_mask = (section_niches["niche"] == IMMUNE_HOT_NICHE).values

        if cold_mask.sum() < MIN_SPOTS or hot_mask.sum() < MIN_SPOTS:
            continue

        full_expr = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        cold_pseudobulk = full_expr[cold_mask].mean(axis=0)
        hot_pseudobulk = full_expr[hot_mask].mean(axis=0)

        pseudobulk_rows.append({"section_id": section_id, "organ": organ, "niche_type": "immune_cold",
                                 **dict(zip(adata.var_names, cold_pseudobulk))})
        pseudobulk_rows.append({"section_id": section_id, "organ": organ, "niche_type": "immune_hot",
                                 **dict(zip(adata.var_names, hot_pseudobulk))})
    print(f"{organ}: done")

pseudobulk_df = pd.DataFrame(pseudobulk_rows)
pseudobulk_df.to_csv("results/step13_pseudobulk_expression.csv", index=False)

print(f"\nSections with valid paired pseudobulk profiles: {pseudobulk_df['section_id'].nunique()}")
print(f"Total genes captured: {pseudobulk_df.shape[1] - 3}")
if validation_log:
    print(f"\n{len(validation_log)} sections skipped (alignment/count issues):")
    for line in validation_log:
        print(f"  {line}")
else:
    print("\nAll sections passed alignment validation cleanly.")
