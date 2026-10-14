import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PANEL = ["MIF", "PDCD1", "CD8A", "CD8B", "FOXP3", "CD163", "EPCAM"]
ORGANS = ["breast", "lung", "kidney", "pancreas", "ovary", "colorectal"]
N_NICHES = 6  # reasonable starting number of unsupervised niche types

all_rows = []
for organ in ORGANS:
    organ_dir = Path("data") / "processed" / organ
    for h5ad_path in sorted(organ_dir.glob("*.h5ad")):
        section_id = h5ad_path.stem
        adata = sc.read_h5ad(h5ad_path)
        if not all(g in adata.var_names for g in PANEL):
            continue

        # Neighborhood-smoothed expression: each spot's own value averaged
        # with its physical neighbors, so niches reflect coherent tissue
        # regions rather than single noisy spots.
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
        conn = adata.obsp["spatial_connectivities"]
        raw_expr = adata[:, PANEL].X.toarray()
        neighbor_sum = conn.dot(raw_expr)
        neighbor_count = np.asarray(conn.sum(axis=1)).flatten()
        neighbor_count[neighbor_count == 0] = 1
        smoothed = (raw_expr + neighbor_sum) / (1 + neighbor_count[:, None])

        df = pd.DataFrame(smoothed, columns=PANEL)
        df["organ"] = organ
        df["section_id"] = section_id
        all_rows.append(df)
    print(f"{organ}: done")

combined = pd.concat(all_rows, ignore_index=True)
print(f"\nTotal spots across all sections: {len(combined)}")

# Cluster on standardized, smoothed expression, pooled across all organs
X = StandardScaler().fit_transform(combined[PANEL].values)
kmeans = KMeans(n_clusters=N_NICHES, random_state=0, n_init=10)
combined["niche"] = kmeans.fit_predict(X)

combined.to_csv("results/step11_niche_assignments.csv", index=False)

print("\n=== Niche characterization: mean (smoothed) expression per niche ===")
niche_profiles = combined.groupby("niche")[PANEL].mean()
niche_profiles["n_spots"] = combined.groupby("niche").size()
print(niche_profiles.to_string())

print("\n=== Which niche(s) does MIF peak in, and what is that niche's immune profile? ===")
mif_ranked_niches = niche_profiles.sort_values("MIF", ascending=False)
print(mif_ranked_niches[["MIF", "EPCAM", "CD163", "CD8A", "CD8B", "PDCD1", "FOXP3", "n_spots"]].to_string())
