# Cancer_atlas_study

Analysis code for the manuscript:

**Spatial Transcriptomic Analysis Reveals Conserved MIF-Associated Immune Exclusion Across Six Human Cancers**
(manuscript in preparation)

This repository contains the scripts used to test whether tumor regions with high expression of macrophage migration inhibitory factor (MIF) spatially exclude immune cells, using 10x Genomics Visium spatial transcriptomics data from six human cancer types, with an independent external validation cohort.

---

## Study overview

The analysis has three tiers, each reported separately in the manuscript and never pooled statistically:

1. **Primary spatial analysis** (Human Tumor Atlas Network; 151 sections, 451,290 spots; breast, ovary, pancreas, lung, kidney, colorectal). MIF spatial clustering (Moran's I), MIF-high versus immune-marker neighborhood exclusion, an EPCAM-based specificity control, a tumor-restricted confound correction, and unsupervised spatial niche clustering.
2. **Full-transcriptome mechanistic extension** (same cohort; 33 sections with both niches represented). Paired pseudobulk differential expression and Hallmark gene set enrichment between the MIF-high, immune-excluded niche and the immune-infiltrated niche, with an immune-gene-filtered sensitivity analysis and a tumor-type-restricted sensitivity analysis.
3. **Independent external validation** (Zenodo; 3 kidney and 5 lung tumor sections with pathologist annotation). Replication of the exclusion analysis and the full-transcriptome pathway analysis using pathologist-confirmed tumor regions in place of the EPCAM-based proxy.

---

## Repository structure

```
Cancer_atlas_study/
├── data_acquisition/            HTAN download and expression-matrix reconstruction
├── spatial_clustering/          Moran's I spatial autocorrelation (Figure 1)
├── exclusion_confound_control/  Neighborhood exclusion and EPCAM confound control (Figure 2)
├── niche_analysis/              Unsupervised spatial niche clustering (Figure 4)
├── full_transcriptome_analysis/ Pseudobulk DE and GSEA between niches (Figure 5)
├── external_validation/         Kidney/lung external cohort replication (Figure 6)
├── figures/                     Composite figure scripts for Figures 5 and 6
└── supplementary_tables/        Result tables underlying the manuscript tables
```

---

## Scripts, in run order

### 1. Data acquisition

| Script | Purpose |
|---|---|
| `data_acquisition/step4b_download_and_build.py` | Retrieves HTAN Visium files, groups files by capture area, detects compression from file contents, and reconstructs one full-transcriptome AnnData (`.h5ad`) object per section. |

### 2. Spatial clustering (Figure 1)

| Script | Purpose |
|---|---|
| `spatial_clustering/step5_morans_i.py` | Moran's I for MIF and immune markers per section (grid neighbor graph, 6 neighbors, 100 permutations). |
| `spatial_clustering/step5b_morans_i_binary.py` | Binary (detected/not detected) Moran's I for sparse immune markers. |
| `spatial_clustering/step5c_full_statistics.py` | Cross-organ and cross-gene statistical tests on Moran's I results. |
| `spatial_clustering/step9_moransi_epcam_check.py` | EPCAM specificity control for MIF spatial clustering. |

### 3. Neighborhood exclusion and confound control (Figure 2)

| Script | Purpose |
|---|---|
| `exclusion_confound_control/step6_exclusion_test.py` | Neighborhood enrichment between MIF-high and immune-marker-positive spots (whole tissue). |
| `exclusion_confound_control/step7_specificity_control.py` | Repeats the exclusion test with EPCAM in place of MIF to measure the generic tumor-versus-stroma component. |
| `exclusion_confound_control/step8_tumor_restricted_specificity.py` | Tumor-restricted exclusion test (EPCAM-above-median spots only), reported as the primary confound-controlled estimate. |

### 4. Unsupervised niche analysis (Figure 4)

| Script | Purpose |
|---|---|
| `niche_analysis/step11_niche_clustering.py` | Spatially smoothed seven-gene profiles clustered into six niches (k-means). Produces `step11_niche_assignments.csv`. |
| `niche_analysis/step12_niche_stats_and_figure.py` | Paired comparison of immune markers between the MIF-high/immune-excluded and immune-infiltrated niches. |
| `niche_analysis/step12b_niche_figure_v2.py` | Final version of the niche figure (Figure 4). |

### 5. Full-transcriptome analysis (Figure 5)

Run in this order:

| Script | Purpose |
|---|---|
| `full_transcriptome_analysis/step13_full_transcriptome_pseudobulk.py` | Rebuilds niche labels on the full-transcriptome data, validates spot alignment (correlation check against saved niche assignments), and builds per-section pseudobulk profiles for the two niches. |
| `full_transcriptome_analysis/step14_pseudobulk_de.py` | Initial paired differential expression across all genes. |
| `full_transcriptome_analysis/step14b_fix_fdr.py` | Recomputes FDR after excluding untestable genes (zero paired variance), which otherwise invalidate the correction. Run after `step14`. |
| `full_transcriptome_analysis/step15_pathway_enrichment.py` | First GSEA pass using a Wilcoxon-derived ranking. Retained for transparency; superseded by `step15b` because the ranking contained extensive ties at this sample size. |
| `full_transcriptome_analysis/step15b_continuous_rank_gsea.py` | Final GSEA: paired t-statistic ranking, run on the full gene list and on an immune-identity-gene-filtered list. Results reported in the manuscript. |
| `full_transcriptome_analysis/step16_organ_sensitivity.py` | Repeats key gene tests restricted to breast, pancreas, and ovary (n=28 paired sections). |
| `figures/figure5_full_transcriptome_composite.py` | Composite Figure 5 and supplementary tables. |

### 6. External validation (Figure 6)

| Script | Purpose |
|---|---|
| `external_validation/step18_external_validation.py` | Tumor-restricted exclusion test in the external cohort, using pathologist ground-truth tumor annotation. |
| `external_validation/step19_external_validation_full.py` | FDR correction across section-marker tests and per-marker direction summaries. |
| `external_validation/step20_external_pseudobulk_DE_GSEA.py` | MIF-high versus MIF-low pseudobulk DE and Hallmark GSEA within tumor regions. |
| `external_validation/step21_external_sensitivity_checks.py` | Organ-stratified consistency check (kidney versus lung) and preparation of the immune-gene-filtered ranking. |
| `figures/step19b_external_validation_figure.py` | Earlier two-panel draft of the external validation figure. Superseded by the composite below; retained for transparency. |
| `figures/figure6_external_validation_composite.py` | Composite Figure 6 and supplementary tables. |

The immune-gene-filtered GSEA for the external cohort was run as a short command after `step21`:

```python
import pandas as pd
import gseapy as gp

rank_df = pd.read_csv("results/step21_external_filtered_rank.csv")
res = gp.prerank(rnk=rank_df, gene_sets="MSigDB_Hallmark_2020",
                 outdir="results/gsea_external_filtered",
                 seed=42, permutation_num=1000, min_size=10, max_size=500)
res.res2d.sort_values("FDR q-val").to_csv("results/step21_external_filtered_GSEA.csv", index=False)
```

---

## Steps not included as standalone scripts

Two analyses described in the manuscript are not included here as standalone scripts:

- **Cross-center data harmonization and quality control**, beyond the file-handling logic contained in `step4b_download_and_build.py`.
- **Bulk-spatial concordance analysis** (Figure 3), which compared TCGA bulk MIF-PDCD1 correlations (via cBioPortal) with spatial exclusion estimates.

The methods for both are described in full in the manuscript's Materials and Methods.

---

## Data sources

Raw and processed spatial data are **not** included in this repository because of their size. They are publicly available from:

- **Human Tumor Atlas Network (HTAN)**: Visium spatial transcriptomics, accessed through the ISB Cancer Gateway in the Cloud BigQuery tables (`isb-cgc-bq.HTAN`) and Synapse. Only open-access data were used.
- **External validation cohort**: Dawo S, Nonchev K, Silina K. *10x Visium Spatial Transcriptomics Dataset: Kidney (3) and Lung (5) Cancer with Tertiary Lymphoid Structures*. Zenodo, 2025. doi:[10.5281/zenodo.14620362](https://doi.org/10.5281/zenodo.14620362)
- **TCGA PanCancer Atlas bulk RNA-seq** (bulk-spatial comparison): accessed through the cBioPortal API.
- **MSigDB Hallmark gene sets**: accessed through GSEApy (`MSigDB_Hallmark_2020`).

---

## Expected directory layout

Scripts assume they are run from the project root, with this layout:

```
project_root/
├── data/
│   ├── processed/{organ}/*.h5ad          one file per HTAN section (output of step4b)
│   └── external_test/h5ad_only/*.h5ad    external cohort sections (KC1-3, LC1-5)
└── results/                              all intermediate and final outputs
```

Run every script from the project root (for example, `python3 external_validation/step18_external_validation.py`), so that relative paths to `data/` and `results/` resolve correctly.

---

## Software

Analyses were run in Python 3 on a Linux HPC cluster. Main packages:

- `scanpy`, `anndata`, `squidpy` (spatial data handling, neighbor graphs, Moran's I, neighborhood enrichment)
- `gseapy` (preranked GSEA)
- `pandas`, `numpy`, `scipy`, `statsmodels` (statistics, Benjamini-Hochberg correction)
- `scikit-learn` (k-means niche clustering)
- `matplotlib`, `adjustText` (figures)

Exact package versions are listed in `requirements.txt`.

---

## Supplementary tables

| File | Contents |
|---|---|
| `step6_exclusion_zscores.csv` | Per-section, per-marker exclusion z-scores (primary cohort). |
| `step11_niche_assignments.csv` | Niche assignment and smoothed panel expression for every spot. |
| `SuppTable_full_transcriptome_DE.csv` | Full-transcriptome DE results between niches. |
| `SuppTable_GSEA_hallmark_results.csv` | Hallmark GSEA results between niches. |
| `SuppTable_external_exclusion_zscores.csv` | External cohort exclusion z-scores with FDR. |
| `SuppTable_external_DE.csv` | External cohort DE results (MIF-high versus MIF-low tumor spots). |
| `SuppTable_external_GSEA.csv` | External cohort Hallmark GSEA results (immune-gene-filtered). |

---

## Contact

Muyiwa S. Adegbaju, Department of Biomedical Sciences, Rochester Institute of Technology.
Questions about the code can be raised through the [Issues](https://github.com/Maxwellgbaju/Cancer_atlas_study/issues) tab of this repository.

