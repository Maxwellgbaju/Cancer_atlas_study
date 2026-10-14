import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

PANEL = ["MIF", "PDCD1", "CD8A", "CD8B", "FOXP3", "CD163", "EPCAM"]
niches = pd.read_csv("results/step11_niche_assignments.csv")
stats_df = pd.read_csv("results/step12_niche_comparison_stats.csv")

NICHE_NAMES = {
    5: "MIF-high,\nimmune-excluded",
    3: "Tumor-dense",
    1: "Immune-\ninfiltrated",
    4: "Intermediate/\nmixed",
    0: "Macrophage-\nenriched",
    2: "Low-signal/\nbackground",
}
IMMUNE_COLD_NICHE = 5
IMMUNE_HOT_NICHE = 1
PURE_TUMOR_NICHE = 3

niche_profiles = niches.groupby("niche")[PANEL].mean()
niche_sizes = niches.groupby("niche").size()
niche_order = niche_profiles.sort_values("MIF", ascending=False).index.tolist()

fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: DOT PLOT instead of heatmap -- field-standard for expression-across-clusters
ax = fig.add_subplot(gs[0, 0])
norm_profiles = niche_profiles.loc[niche_order] / niche_profiles.loc[niche_order].max()
for yi, n in enumerate(niche_order):
    for xi, gene in enumerate(PANEL):
        val = norm_profiles.loc[n, gene]
        ax.scatter(xi, yi, s=val * 600 + 15, color="#C00000", alpha=0.75, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(PANEL))); ax.set_xticklabels(PANEL, rotation=45, ha="right")
ax.set_yticks(range(len(niche_order)))
ax.set_yticklabels([f"{NICHE_NAMES[n]}\n(n={niche_sizes[n]:,})" for n in niche_order], fontsize=8)
ax.set_xlim(-0.5, len(PANEL)-0.5); ax.set_ylim(-0.5, len(niche_order)-0.5)
ax.grid(alpha=0.2, linestyle="--")
ax.set_title("A. Unsupervised niche profiles\n(dot size = normalized expression per gene)", fontsize=10.5, fontweight="bold")

# Panel B: unchanged, immune-cold vs immune-hot with significance
ax = fig.add_subplot(gs[0, 1])
immune_markers = ["CD163", "CD8A", "CD8B", "PDCD1", "FOXP3"]
x = np.arange(len(immune_markers)); width = 0.35
cold_vals = stats_df["niche5_immune_cold_mean"].values
hot_vals = stats_df["niche1_immune_hot_mean"].values
ax.bar(x - width/2, cold_vals, width, label=f"{NICHE_NAMES[IMMUNE_COLD_NICHE].replace(chr(10),' ')}", color="#C00000", alpha=0.8)
ax.bar(x + width/2, hot_vals, width, label=f"{NICHE_NAMES[IMMUNE_HOT_NICHE].replace(chr(10),' ')}", color="#2E5FA3", alpha=0.8)
for i, row in stats_df.iterrows():
    stars = "***" if row["FDR"] < 0.001 else ("**" if row["FDR"] < 0.01 else ("*" if row["FDR"] < 0.05 else "ns"))
    ax.text(i, max(cold_vals[i], hot_vals[i]) * 1.05, stars, ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(immune_markers)
ax.set_ylabel("Mean expression (smoothed)")
ax.set_title(f"B. MIF-high/immune-excluded vs. immune-infiltrated niche\n(paired by section, n={int(stats_df['n_sections'].iloc[0])} sections)", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5)

# Panel C: niche sizes, with names
ax = fig.add_subplot(gs[1, 0])
colors_c = ["#C00000" if n == IMMUNE_COLD_NICHE else ("#2E5FA3" if n == IMMUNE_HOT_NICHE else "#BFBFBF") for n in niche_order]
ax.bar([NICHE_NAMES[n].replace("\n"," ") for n in niche_order], [niche_sizes[n] for n in niche_order], color=colors_c, alpha=0.8)
ax.set_ylabel("Number of spots")
ax.set_title("C. Niche sizes across full dataset\n(451,290 spots, all sections pooled)", fontsize=10.5, fontweight="bold")
ax.tick_params(axis='x', rotation=30, labelsize=8)

# Panel D: unchanged logic, with names
ax = fig.add_subplot(gs[1, 1])
for n in niche_order:
    color = "#C00000" if n == IMMUNE_COLD_NICHE else ("#7B3F9E" if n == PURE_TUMOR_NICHE else "#4472C4")
    ax.scatter(niche_profiles.loc[n, "EPCAM"], niche_profiles.loc[n, "MIF"], s=niche_sizes[n]/500, color=color, alpha=0.7, edgecolor="black")
    ax.annotate(NICHE_NAMES[n].replace("\n"," "), (niche_profiles.loc[n, "EPCAM"], niche_profiles.loc[n, "MIF"]), xytext=(5,5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Mean EPCAM (tumor density)")
ax.set_ylabel("Mean MIF")
ax.set_title("D. MIF's peak niche is not the tumor-dense niche\n(bubble size = niche size)", fontsize=10.5, fontweight="bold")
ax.grid(alpha=0.3)

plt.suptitle("Unsupervised spatial niche analysis identifies a distinct MIF-high, immune-excluded tissue neighborhood",
             fontsize=13, fontweight="bold", y=1.02)
plt.savefig("results/Figure_niche_analysis_v2.png", dpi=300, bbox_inches="tight")
plt.savefig("results/Figure_niche_analysis_v2.eps", format="eps", bbox_inches="tight")
print("Saved Figure_niche_analysis_v2.png")
