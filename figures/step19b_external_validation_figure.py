import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42
})

df = pd.read_csv("results/step19_external_validation_FDR.csv")
summary_df = pd.read_csv("results/step19_marker_summary.csv")

fig = plt.figure(figsize=(12, 5.5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

markers_order = ["CD163", "CD8A", "CD8B", "FOXP3", "PDCD1"]
y_jitter = {m: i for i, m in enumerate(markers_order)}
# Fixed: compute the numeric y-position as a plain integer array, not a
# pandas Categorical, before adding jitter.
df["y_base"] = df["marker"].map(y_jitter).astype(float)
df["y_pos"] = df["y_base"] + np.random.uniform(-0.15, 0.15, len(df))

ax1 = fig.add_subplot(gs[0])
colors = df["cancer_type"].map({"kidney": "#8B4513", "lung": "#4472C4"})
ax1.scatter(df["zscore"], df["y_pos"], c=colors, s=50, alpha=0.8, edgecolor="black", linewidth=0.5)
ax1.axvline(0, color="black", linewidth=0.8)
ax1.set_yticks(range(len(markers_order)))
ax1.set_yticklabels(markers_order)
ax1.set_xlabel("MIF-high vs. marker-positive\nneighborhood enrichment z-score")
ax1.set_title("A. External validation cohort\n(independent dataset, tumor-restricted)", fontsize=9.5, fontweight="bold", loc="left")
ax1.legend(handles=[Patch(color="#8B4513", label="Kidney (n=3)"), Patch(color="#4472C4", label="Lung (n=5)")],
           loc="lower right", fontsize=7.5, frameon=False)

ax2 = fig.add_subplot(gs[1])
summary_df = summary_df.set_index("marker").reindex(markers_order).reset_index()
colors_b = ["#C00000" if (not pd.isna(p) and p < 0.1) else "#888888" for p in summary_df["wilcoxon_p"]]
ax2.barh(summary_df["marker"], summary_df["mean_zscore"].fillna(0), color=colors_b, alpha=0.85)
ax2.axvline(0, color="black", linewidth=0.8)
for i, row in summary_df.iterrows():
    if pd.isna(row["mean_zscore"]):
        continue
    label = f"p={row['wilcoxon_p']:.3f} (n={int(row['n_sections'])} sections)" if not pd.isna(row["wilcoxon_p"]) else f"n={int(row['n_sections'])} (insufficient)"
    ax2.text(row["mean_zscore"] + (0.15 if row["mean_zscore"] >= 0 else -0.15), i, label,
              va="center", ha="left" if row["mean_zscore"] >= 0 else "right", fontsize=7.5)
ax2.set_xlabel("Mean z-score across sections")
ax2.set_title("B. Per-marker summary\n(red: p<0.10, grey: n.s. or underpowered)", fontsize=9.5, fontweight="bold", loc="left")

fig.suptitle("Independent validation of MIF-associated immune exclusion in an external kidney and lung cohort",
             fontsize=11.5, fontweight="bold", y=1.03)

fig.savefig("results/Figure5_external_validation.png", dpi=600, bbox_inches="tight")
fig.savefig("results/Figure5_external_validation.eps", format="eps", bbox_inches="tight")
print("Saved results/Figure5_external_validation.png and .eps")
