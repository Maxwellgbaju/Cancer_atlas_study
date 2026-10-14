import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

PANEL = ["MIF", "PDCD1", "CD8A", "CD8B", "FOXP3", "CD163", "EPCAM"]
niches = pd.read_csv("results/step11_niche_assignments.csv")

IMMUNE_COLD_NICHE = 5   # highest MIF, lowest immune markers
IMMUNE_HOT_NICHE = 1    # highest immune markers
PURE_TUMOR_NICHE = 3    # highest EPCAM

# --- Per-section paired test: Niche 5 vs Niche 1, for each immune marker ---
MIN_SPOTS_PER_NICHE = 10
immune_markers = ["CD163", "CD8A", "CD8B", "PDCD1", "FOXP3"]
results = []
for marker in immune_markers:
    per_section = []
    for section_id, grp in niches.groupby("section_id"):
        cold = grp[grp["niche"] == IMMUNE_COLD_NICHE]
        hot = grp[grp["niche"] == IMMUNE_HOT_NICHE]
        if len(cold) >= MIN_SPOTS_PER_NICHE and len(hot) >= MIN_SPOTS_PER_NICHE:
            per_section.append({"section_id": section_id, "cold_mean": cold[marker].mean(), "hot_mean": hot[marker].mean()})
    df = pd.DataFrame(per_section)
    if len(df) >= 5:
        stat, p = wilcoxon(df["cold_mean"], df["hot_mean"])
    else:
        p = np.nan
    results.append({"marker": marker, "n_sections": len(df),
                     "niche5_immune_cold_mean": df["cold_mean"].mean() if len(df) else np.nan,
                     "niche1_immune_hot_mean": df["hot_mean"].mean() if len(df) else np.nan,
                     "pval": p})

stats_df = pd.DataFrame(results)
from statsmodels.stats.multitest import multipletests
stats_df["FDR"] = multipletests(stats_df["pval"], method="fdr_bh")[1]
stats_df.to_csv("results/step12_niche_comparison_stats.csv", index=False)
print(stats_df.to_string(index=False))

# --- Figure ---
fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

niche_profiles = niches.groupby("niche")[PANEL].mean()
niche_sizes = niches.groupby("niche").size()

# Panel A: niche profile heatmap
ax = fig.add_subplot(gs[0, 0])
niche_order = niche_profiles.sort_values("MIF", ascending=False).index.tolist()
norm_profiles = niche_profiles.loc[niche_order] / niche_profiles.loc[niche_order].max()
im = ax.imshow(norm_profiles[PANEL].values, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(PANEL))); ax.set_xticklabels(PANEL, rotation=45, ha="right")
ax.set_yticks(range(len(niche_order))); ax.set_yticklabels([f"Niche {n}\n(n={niche_sizes[n]:,})" for n in niche_order], fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Expression\n(normalized per gene)")
ax.set_title("A. Unsupervised niche profiles\n(rows ranked by MIF)", fontsize=10.5, fontweight="bold")

# Panel B: Niche 5 (immune-cold) vs Niche 1 (immune-hot), with significance
ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(immune_markers)); width = 0.35
cold_vals = stats_df["niche5_immune_cold_mean"].values
hot_vals = stats_df["niche1_immune_hot_mean"].values
ax.bar(x - width/2, cold_vals, width, label=f"Niche {IMMUNE_COLD_NICHE} (MIF-high, immune-cold)", color="#C00000", alpha=0.8)
ax.bar(x + width/2, hot_vals, width, label=f"Niche {IMMUNE_HOT_NICHE} (immune-hot)", color="#2E5FA3", alpha=0.8)
for i, row in stats_df.iterrows():
    stars = "***" if row["FDR"] < 0.001 else ("**" if row["FDR"] < 0.01 else ("*" if row["FDR"] < 0.05 else "ns"))
    ax.text(i, max(cold_vals[i], hot_vals[i]) * 1.05, stars, ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(immune_markers)
ax.set_ylabel("Mean expression (smoothed)")
ax.set_title(f"B. Immune-cold vs. immune-hot niche\n(paired by section, n={stats_df['n_sections'].iloc[0]} sections)", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=8)

# Panel C: niche sizes
ax = fig.add_subplot(gs[1, 0])
colors_c = ["#C00000" if n == IMMUNE_COLD_NICHE else ("#2E5FA3" if n == IMMUNE_HOT_NICHE else "#BFBFBF") for n in niche_order]
ax.bar([f"Niche {n}" for n in niche_order], [niche_sizes[n] for n in niche_order], color=colors_c, alpha=0.8)
ax.set_ylabel("Number of spots")
ax.set_title("C. Niche sizes across full dataset\n(451,290 spots, all sections pooled)", fontsize=10.5, fontweight="bold")
ax.tick_params(axis='x', rotation=30)

# Panel D: MIF vs EPCAM per niche -- shows MIF-peak niche is NOT the purest-tumor niche
ax = fig.add_subplot(gs[1, 1])
for n in niche_order:
    color = "#C00000" if n == IMMUNE_COLD_NICHE else ("#7B3F9E" if n == PURE_TUMOR_NICHE else "#4472C4")
    ax.scatter(niche_profiles.loc[n, "EPCAM"], niche_profiles.loc[n, "MIF"], s=niche_sizes[n]/500, color=color, alpha=0.7, edgecolor="black")
    ax.annotate(f"Niche {n}", (niche_profiles.loc[n, "EPCAM"], niche_profiles.loc[n, "MIF"]), xytext=(5,5), textcoords="offset points", fontsize=9)
ax.set_xlabel("Mean EPCAM (tumor density)")
ax.set_ylabel("Mean MIF")
ax.set_title("D. MIF's peak niche is not the purest-tumor niche\n(bubble size = niche size)", fontsize=10.5, fontweight="bold")
ax.grid(alpha=0.3)

plt.suptitle("Unsupervised spatial niche analysis identifies a distinct MIF-high, immune-cold tissue neighborhood",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/Figure_niche_analysis.png", dpi=300, bbox_inches="tight")
print("\nSaved Figure_niche_analysis.png")
