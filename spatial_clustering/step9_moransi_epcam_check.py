import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from pathlib import Path

ORGANS = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]
results = []

for organ in ORGANS:
    organ_dir = Path("data") / "processed" / organ
    for h5ad_path in sorted(organ_dir.glob("*.h5ad")):
        section_id = h5ad_path.stem
        adata = sc.read_h5ad(h5ad_path)
        if "EPCAM" not in adata.var_names:
            continue
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
        try:
            sq.gr.spatial_autocorr(adata, mode="moran", genes=["EPCAM"], n_perms=100, n_jobs=1)
            moran_i = adata.uns["moranI"].loc["EPCAM", "I"]
            results.append({"organ": organ, "section_id": section_id, "morans_i": moran_i})
        except Exception:
            continue
    print(f"{organ}: done")

df = pd.DataFrame(results)
df.to_csv("results/step9_epcam_morans_i.csv", index=False)

print(f"\n=== EPCAM Moran's I (control gene): mean={df['morans_i'].mean():.3f}, sd={df['morans_i'].std():.3f}, n={len(df)} ===")

mif_morans = pd.read_csv("results/step5_morans_i_all_organs.csv")
mif_only = mif_morans[mif_morans["gene"] == "MIF"]
print(f"For direct comparison, MIF Moran's I: mean={mif_only['morans_i'].mean():.3f}, sd={mif_only['morans_i'].std():.3f}, n={len(mif_only)}")

# EPCAM detection rate, for context (matches the same logic used for MIF/immune markers)
dfs = [pd.read_csv(f"results/step4_panel_expression_{o}.csv") for o in ORGANS]
combined = pd.concat(dfs)
if "EPCAM" in combined.columns:
    print(f"EPCAM detection rate: {(combined['EPCAM'] > 0).mean()*100:.1f}%")
else:
    print("NOTE: EPCAM not in the original 6-gene panel CSVs -- detection rate not directly available from this file; already confirmed present in h5ad files above.")
