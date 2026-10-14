import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from adjustText import adjust_text

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42
})

de_df = pd.read_csv("results/step14_full_transcriptome_DE.csv")
gsea_df = pd.read_csv("results/step15b_gsea_full.csv")
pseudobulk = pd.read_csv("results/step13_pseudobulk_expression.csv")

fig = plt.figure(figsize=(14, 11))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# --- Panel A: Volcano plot ---
ax1 = fig.add_subplot(gs[0, 0])
de_df["neglog10FDR"] = -np.log10(de_df["FDR"].clip(lower=1e-300))

# Flag near-silent-gene fold-change artifacts; excluded from the VISIBLE axis
# range only -- full data remains in the saved supplementary table untouched.
pseudo_expr_floor = 0.5
cold_cols = [c for c in pseudobulk.columns if c not in ["section_id", "organ", "niche_type"]]
mean_expr = pseudobulk.groupby("niche_type")[cold_cols].mean().T
mean_expr.columns = ["mean_cold", "mean_hot"]
low_expr_genes = set(mean_expr[(mean_expr["mean_cold"] < pseudo_expr_floor) & (mean_expr["mean_hot"] < pseudo_expr_floor)].index)
de_df["low_confidence_fc"] = de_df["gene"].isin(low_expr_genes)
print(f"Genes flagged as low-expression FC artifacts (excluded from plot range only): {de_df['low_confidence_fc'].sum()}")

plot_df = de_df[~de_df["low_confidence_fc"]]
sig_up = plot_df[(plot_df["FDR"] < 0.05) & (plot_df["log2FC_cold_vs_hot"] > 0)]
sig_down = plot_df[(plot_df["FDR"] < 0.05) & (plot_df["log2FC_cold_vs_hot"] < 0)]
nonsig = plot_df[plot_df["FDR"] >= 0.05]

ax1.scatter(nonsig["log2FC_cold_vs_hot"], nonsig["neglog10FDR"], s=4, c="#cccccc", alpha=0.4, linewidths=0, rasterized=True)
ax1.scatter(sig_up["log2FC_cold_vs_hot"], sig_up["neglog10FDR"], s=5, c="#c0392b", alpha=0.4, linewidths=0, rasterized=True)
ax1.scatter(sig_down["log2FC_cold_vs_hot"], sig_down["neglog10FDR"], s=5, c="#2980b9", alpha=0.4, linewidths=0, rasterized=True)

highlight_genes = ["MIF", "VEGFA", "SLC2A1", "HK2", "CD74", "TGFB1", "CD274", "IDO1"]
texts = []
for gene in highlight_genes:
    row = plot_df[plot_df["gene"] == gene]
    if len(row) == 0:
        continue
    x, y = row["log2FC_cold_vs_hot"].values[0], row["neglog10FDR"].values[0]
    ax1.scatter([x], [y], s=35, c="black", zorder=5, edgecolors="white", linewidths=0.8)
    texts.append(ax1.text(x, y, gene, fontsize=8, fontweight="bold", zorder=10))

# Fixed: shrinkA/shrinkB pull leader lines back from both the dot and the
# text so lines no longer cut through label characters (the "JK2" /
# strikethrough issue); stronger expand/force values separate crowded labels
# (MIF/VEGFA, HK2/SLC2A1) before a connector line is even needed.
adjust_text(
    texts, ax=ax1,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.5, shrinkA=5, shrinkB=5),
    expand_points=(2.0, 2.0), expand_text=(1.3, 1.3),
    force_text=(0.8, 0.9), force_points=(0.3, 0.3)
)

ax1.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.6)
ax1.axvline(0, color="grey", linestyle="-", linewidth=0.5)
x_range = plot_df["log2FC_cold_vs_hot"].abs().quantile(0.999)
ax1.set_xlim(-x_range, x_range)
ax1.set_xlabel("log$_2$FC (immune-cold vs immune-hot niche)")
ax1.set_ylabel("-log$_{10}$(FDR)")
ax1.set_title("A. Full-transcriptome differential expression\n(n=33 paired sections, 17,558 genes tested; sparse/near-silent\ngenes excluded from axis range, retained in supplementary table)",
              fontsize=8.5, loc="left", fontweight="bold")

# --- Panel B: GSEA pathways ---
ax2 = fig.add_subplot(gs[0, 1])
IMMUNE_PATHWAYS = {"Allograft Rejection", "Interferon Gamma Response", "Interferon Alpha Response",
                    "IL-6/JAK/STAT3 Signaling", "Inflammatory Response", "Complement",
                    "IL-2/STAT5 Signaling", "TNF-alpha Signaling via NF-kB", "Coagulation"}
sig_gsea = gsea_df[gsea_df["FDR q-val"] < 0.05].copy()
sig_gsea["category"] = sig_gsea["Term"].apply(lambda t: "Immune activity (expected)" if t in IMMUNE_PATHWAYS else "Metabolic / hypoxia / stromal (deeper signal)")
sig_gsea = sig_gsea.sort_values("NES")
plot_subset = pd.concat([sig_gsea.head(10), sig_gsea.tail(15)]).drop_duplicates()
colors = plot_subset["category"].map({"Immune activity (expected)": "#7f8c8d", "Metabolic / hypoxia / stromal (deeper signal)": "#27ae60"})

ax2.barh(plot_subset["Term"], plot_subset["NES"], color=colors, height=0.65)
ax2.axvline(0, color="black", linewidth=0.6)
ax2.set_xlabel("Normalized Enrichment Score (NES)")
ax2.set_title("B. Hallmark pathway enrichment\n(robust across immune-gene-filtered sensitivity check)", fontsize=9, loc="left", fontweight="bold")
legend_patches = [mpatches.Patch(color="#7f8c8d", label="Immune activity (expected)"),
                   mpatches.Patch(color="#27ae60", label="Metabolic / hypoxia / stromal (deeper signal)")]
ax2.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=7, frameon=False)
ax2.tick_params(axis="y", labelsize=6.5)

# --- Panel C: Hypoxia leading-edge genes ---
ax3 = fig.add_subplot(gs[1, 0])
hypoxia_leading = ["MIF", "VEGFA", "SLC2A1", "HK2", "PGK1", "ENO1", "ENO2", "PFKP", "NDRG1", "BNIP3L", "DDIT3", "ADM"]
leading_data = de_df[de_df["gene"].isin(hypoxia_leading)].set_index("gene").reindex(hypoxia_leading).reset_index()
bar_colors = ["#e74c3c" if g == "MIF" else "#f39c12" for g in leading_data["gene"]]
ax3.barh(leading_data["gene"], leading_data["log2FC_cold_vs_hot"], color=bar_colors, height=0.6)
ax3.axvline(0, color="black", linewidth=0.6)
ax3.set_xlabel("log$_2$FC (immune-cold vs immune-hot niche)")
ax3.set_title("C. Hallmark Hypoxia leading-edge genes\n(MIF highlighted: pathway member, not a pre-selected gene)", fontsize=9, loc="left", fontweight="bold")

# --- Panel D: Organ composition ---
ax4 = fig.add_subplot(gs[1, 1])
organ_counts = pseudobulk[pseudobulk["niche_type"] == "immune_cold"]["organ"].value_counts()
organ_order = ["breast", "pancreas", "ovary", "lung", "kidney", "colorectal"]
organ_counts = organ_counts.reindex(organ_order).fillna(0)
bar_colors_d = ["#2c3e50" if o in ["breast", "pancreas", "ovary"] else "#bdc3c7" for o in organ_order]
ax4.bar(organ_counts.index, organ_counts.values, color=bar_colors_d)
ax4.set_ylabel("Paired sections (n)")
ax4.set_title("D. Section distribution across cancer types\n(dark: well-represented; grey: underpowered, excluded from sensitivity analysis)", fontsize=9, loc="left", fontweight="bold")
ax4.tick_params(axis="x", rotation=30)
for i, v in enumerate(organ_counts.values):
    ax4.text(i, v + 0.3, str(int(v)), ha="center", fontsize=8)

fig.suptitle("Full-transcriptome characterization of the MIF-high immune-excluded spatial niche", fontsize=12, fontweight="bold", y=0.995)

fig.savefig("results/Figure4_full_transcriptome_composite.png", dpi=600, bbox_inches="tight")
fig.savefig("results/Figure4_full_transcriptome_composite.eps", format="eps", bbox_inches="tight")
print("Saved: results/Figure4_full_transcriptome_composite.png and .eps")

de_df.sort_values("FDR").to_csv("results/SuppTable_full_transcriptome_DE.csv", index=False)
gsea_df.sort_values("FDR q-val").to_csv("results/SuppTable_GSEA_hallmark_results.csv", index=False)
print("Saved: results/SuppTable_full_transcriptome_DE.csv and results/SuppTable_GSEA_hallmark_results.csv")
