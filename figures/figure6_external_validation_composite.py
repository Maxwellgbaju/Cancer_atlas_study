import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from adjustText import adjust_text

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42
})

exclusion_df = pd.read_csv("results/step19_external_validation_FDR.csv")
de_df = pd.read_csv("results/step20_external_DE.csv")
gsea_df = pd.read_csv("results/step21_external_filtered_GSEA.csv")
pseudobulk = pd.read_csv("results/step20_external_pseudobulk.csv")

# Original cohort per-organ, per-marker means (from step6, already computed)
original_means = pd.DataFrame({
    "marker": ["CD163", "CD8A", "FOXP3"],
    "kidney": [-9.622, -6.300, -2.869],
    "lung": [-8.115, -4.520, -1.859],
})
external_means = exclusion_df.groupby("marker")["zscore"].mean()

fig = plt.figure(figsize=(14, 11))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

# --- Panel A: exclusion z-score comparison, original vs external ---
ax1 = fig.add_subplot(gs[0, 0])
markers = ["CD163", "CD8A", "FOXP3"]
x = np.arange(len(markers))
width = 0.25
kidney_vals = original_means.set_index("marker").loc[markers, "kidney"].values
lung_vals = original_means.set_index("marker").loc[markers, "lung"].values
ext_vals = [external_means.get(m, np.nan) for m in markers]

ax1.bar(x - width, kidney_vals, width, label="Original kidney\n(whole-tissue)", color="#8B4513", alpha=0.85)
ax1.bar(x, lung_vals, width, label="Original lung\n(whole-tissue)", color="#4472C4", alpha=0.85)
ax1.bar(x + width, ext_vals, width, label="External cohort\n(tumor-restricted)", color="#27ae60", alpha=0.85)
ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(markers)
ax1.set_ylabel("Mean neighborhood enrichment z-score")
ax1.set_title("A. Exclusion z-score replication\n(same rank order, smaller magnitude in tumor-restricted estimate)", fontsize=9, loc="left", fontweight="bold")
ax1.legend(fontsize=7, frameon=False, loc="lower right")

# --- Panel B: external DE, genes of interest highlighted ---
ax2 = fig.add_subplot(gs[0, 1])
de_df["neglog10FDR"] = -np.log10(de_df["FDR"].clip(lower=1e-300))
nonsig = de_df[de_df["FDR"] >= 0.05]
ax2.scatter(nonsig["t_stat"], nonsig["neglog10FDR"], s=5, c="#cccccc", alpha=0.5, linewidths=0, rasterized=True)

highlight_genes = ["MIF", "VEGFA", "SLC2A1", "HK2", "CD74", "TGFB1"]
texts = []
for gene in highlight_genes:
    row = de_df[de_df["gene"] == gene]
    if len(row) == 0:
        continue
    xg, yg = row["t_stat"].values[0], row["neglog10FDR"].values[0]
    ax2.scatter([xg], [yg], s=45, c="#e74c3c" if gene == "MIF" else "black", zorder=5, edgecolors="white", linewidths=0.8)
    texts.append(ax2.text(xg, yg, gene, fontsize=8, fontweight="bold", zorder=10))
adjust_text(texts, ax=ax2, arrowprops=dict(arrowstyle="-", color="black", lw=0.5, shrinkA=5, shrinkB=5),
            expand_points=(2.0, 2.0), expand_text=(1.3, 1.3), force_text=(0.8, 0.9))
ax2.axvline(0, color="grey", linewidth=0.5)
ax2.set_xlabel("Paired t-statistic (MIF-high vs MIF-low tumor spots)")
ax2.set_ylabel("-log$_{10}$(FDR)")
ax2.set_title("B. External cohort differential expression\n(n=8 paired sections; no gene survives FDR alone, as expected at this n)", fontsize=8.5, loc="left", fontweight="bold")

# --- Panel C: GSEA pathways, immune-filtered, external cohort ---
ax3 = fig.add_subplot(gs[1, 0])
IMMUNE_PATHWAYS = {"Interferon Alpha Response", "Interferon Gamma Response", "TNF-alpha Signaling via NF-kB",
                    "IL-6/JAK/STAT3 Signaling", "IL-2/STAT5 Signaling"}
sig_gsea = gsea_df[gsea_df["FDR q-val"] < 0.01].copy()
sig_gsea["category"] = sig_gsea["Term"].apply(lambda t: "Immune activity" if t in IMMUNE_PATHWAYS else "Metabolic / hypoxia / stromal")
sig_gsea = sig_gsea.sort_values("NES", ascending=True).tail(18)
colors = sig_gsea["category"].map({"Immune activity": "#7f8c8d", "Metabolic / hypoxia / stromal": "#27ae60"})
ax3.barh(sig_gsea["Term"], sig_gsea["NES"], color=colors, height=0.65)
for pathway in ["Hypoxia", "TGF-beta Signaling"]:
    if pathway in sig_gsea["Term"].values:
        idx = list(sig_gsea["Term"]).index(pathway)
        ax3.get_yticklabels()[idx].set_fontweight("bold")
ax3.axvline(0, color="black", linewidth=0.6)
ax3.set_xlabel("Normalized Enrichment Score (NES)")
ax3.set_title("C. Hallmark pathway enrichment (external cohort)\n(immune-gene-filtered; top terms by NES, FDR<0.01)", fontsize=9, loc="left", fontweight="bold")
ax3.legend(handles=[Patch(color="#7f8c8d", label="Immune activity"), Patch(color="#27ae60", label="Metabolic/hypoxia/stromal")],
           loc="lower right", fontsize=7, frameon=False)
ax3.tick_params(axis="y", labelsize=7)

# --- Panel D: per-section MIF consistency, paired lines, by organ ---
ax4 = fig.add_subplot(gs[1, 1])
pseudobulk["organ"] = pseudobulk["section_id"].str[:2].map({"KC": "kidney", "LC": "lung"})
high = pseudobulk[pseudobulk["group"] == "mif_high"].set_index("section_id")
low = pseudobulk[pseudobulk["group"] == "mif_low"].set_index("section_id")
sections = sorted(set(high.index) & set(low.index))
for i, s in enumerate(sections):
    organ = high.loc[s, "organ"]
    color = "#8B4513" if organ == "kidney" else "#4472C4"
    y_high, y_low = np.log1p(high.loc[s, "MIF"]), np.log1p(low.loc[s, "MIF"])
    ax4.plot([0, 1], [y_low, y_high], color=color, alpha=0.7, linewidth=1.3, marker="o", markersize=4)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(["MIF-low\ntumor spots", "MIF-high\ntumor spots"])
ax4.set_ylabel("log$_1$(MIF pseudobulk expression)")
ax4.set_title("D. Per-section MIF consistency\n(8/8 sections same direction; LC2 shows largest dynamic range)", fontsize=9, loc="left", fontweight="bold")
ax4.legend(handles=[Patch(color="#8B4513", label="Kidney (n=3)"), Patch(color="#4472C4", label="Lung (n=5)")],
           loc="upper left", fontsize=7.5, frameon=False)

fig.suptitle("Independent external validation: spatial exclusion and mechanistic replication in kidney and lung cancer",
             fontsize=12, fontweight="bold", y=1.0)

fig.savefig("results/Figure5_external_validation_composite.png", dpi=600, bbox_inches="tight")
fig.savefig("results/Figure5_external_validation_composite.eps", format="eps", bbox_inches="tight")
print("Saved Figure5_external_validation_composite.png and .eps")

# --- Supplementary table ---
exclusion_df.to_csv("results/SuppTable_external_exclusion_zscores.csv", index=False)
de_df.sort_values("FDR").to_csv("results/SuppTable_external_DE.csv", index=False)
gsea_df.sort_values("FDR q-val").to_csv("results/SuppTable_external_GSEA.csv", index=False)
print("Saved SuppTable_external_exclusion_zscores.csv, SuppTable_external_DE.csv, SuppTable_external_GSEA.csv")
