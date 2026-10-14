#!/usr/bin/env python3
"""
step4b_download_and_build.py

Reads a manifest CSV (from step4a_get_manifest.sh), downloads the Visium
bundle (barcodes, features, matrix, tissue_positions) for each capture-area
section via the htan CLI's Synapse downloader, assembles each section into a
scanpy AnnData object, subsets to the immune/MIF gene panel, and saves one
.h5ad per section plus a combined long-format CSV of the panel genes with
spatial coordinates.

USAGE:
    python step4b_download_and_build.py results/step4_manifest_lung.csv lung
    python step4b_download_and_build.py results/step4_manifest_ovary.csv ovary

REQUIRES:
    pip install scanpy pandas scipy
    htan CLI already configured (as in this project)
"""

import sys
import re
import subprocess
import shutil
import gzip
from pathlib import Path

import pandas as pd
import scanpy as sc
import anndata as ad

# Gene panel from the work plan (Step 2): MIF + core immune markers
GENE_PANEL = ["MIF", "PDCD1", "CD8A", "CD8B", "FOXP3", "CD163"]


def download_file(entity_id: str, dest_dir: Path) -> Path:
    """Download a single Synapse entity via the htan CLI, return its local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["htan", "download", "synapse", entity_id, "-o", str(dest_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed for {entity_id}: {result.stderr}")

    # htan's CLI always ends its stdout with the bare, final resolved file
    # path on its own line -- true whether it freshly downloaded the file
    # ("Downloaded to X") or served it from local cache ("Copying existing
    # file from Y to X"). Parsing the last non-empty line handles both.
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    downloaded_path = None
    if lines:
        candidate = Path(lines[-1])
        if candidate.is_file():
            downloaded_path = candidate

    if downloaded_path is not None:
        return downloaded_path

    print(f"  WARNING: could not parse downloaded filename from htan output for "
          f"{entity_id}; falling back to directory scan (less reliable). "
          f"CLI stdout was: {result.stdout!r}")
    candidates = list(dest_dir.rglob("*"))
    files = [p for p in candidates if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No file found after downloading {entity_id} into {dest_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def standardize_name(path: Path, file_type: str) -> Path:
    """Rename a downloaded file to the canonical 10x name scanpy expects."""
    mapping = {
        "barcodes": "barcodes.tsv.gz",
        "features": "features.tsv.gz",
        "filtered mex": "matrix.mtx.gz",
        "unfiltered mex": "matrix.mtx.gz",
        "tissue_positions": "tissue_positions_list.csv",
    }
    target_name = mapping[file_type]
    target_path = path.parent / target_name

    with open(path, "rb") as f_check:
        magic = f_check.read(2)
    already_gzipped = magic == b"\x1f\x8b"

    if target_name.endswith(".gz") and not already_gzipped:
        with open(path, "rb") as f_in, gzip.open(target_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(path, target_path)
    return target_path


def build_section_adata(section_dir: Path) -> ad.AnnData:
    """Load one Visium capture area's mex bundle into an AnnData object."""
    adata = sc.read_10x_mtx(section_dir, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()

    positions_path = section_dir / "tissue_positions_list.csv"
    # HTAN mixes old Space Ranger format (no header) and 2.0+ format (header
    # row present) across sections, even within the same organ/center.
    with open(positions_path, "r") as f_peek:
        first_line = f_peek.readline()
    has_header = first_line.split(",")[0].strip().lower() == "barcode"

    positions = pd.read_csv(
        positions_path,
        header=0 if has_header else None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"] if not has_header else None,
    ).set_index("barcode")
    if has_header:
        positions.columns = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    positions["in_tissue"] = positions["in_tissue"].astype(int)

    positions = positions.loc[positions.index.intersection(adata.obs_names)]
    adata = adata[positions.index].copy()
    adata.obsm["spatial"] = positions[["pxl_col", "pxl_row"]].to_numpy()
    adata.obs["in_tissue"] = positions["in_tissue"].to_numpy()

    adata = adata[adata.obs["in_tissue"] == 1].copy()
    return adata


def _capture_area_token(filename):
    """Extract HTAN's 'AS_N' capture-area identifier from a filename, if present."""
    m = re.search(r'(AS_\d+)', str(filename))
    return m.group(1) if m else None


def main(manifest_csv: str, organ: str):
    manifest = pd.read_csv(manifest_csv)
    raw_dir = Path("data") / "raw" / organ
    processed_dir = Path("data") / "processed" / organ
    processed_dir.mkdir(parents=True, exist_ok=True)

    panel_rows = []

    for section_id, group in manifest.groupby("HTAN_Parent_Biospecimen_ID"):
        print(f"--- Processing section {section_id} ({organ}) ---")
        section_dir = raw_dir / section_id
        section_dir.mkdir(parents=True, exist_ok=True)

        # When a section has more than one candidate row for a file type,
        # naively picking "the first" independently per file type can mix
        # files from DIFFERENT physical capture areas together (observed:
        # barcodes from one capture area paired with a matrix from another,
        # producing a technically valid but silently WRONG result). To avoid
        # this, extract a capture-area token from each filename (HTAN's
        # "AS_N" style substring) and require all file types for a section
        # to come from the SAME capture-area token wherever possible.
        group = group.copy()
        group["_capture_area"] = group["Filename"].apply(_capture_area_token)
        n_file_types = group["Visium_File_Type"].nunique()
        area_counts = group.dropna(subset=["_capture_area"])["_capture_area"].value_counts()
        complete_areas = area_counts[area_counts >= n_file_types].index.tolist()
        chosen_area = complete_areas[0] if complete_areas else (area_counts.index[0] if len(area_counts) else None)

        chosen_rows = {}
        for file_type, sub_group in group.groupby("Visium_File_Type"):
            if len(sub_group) > 1 and chosen_area is not None:
                same_area = sub_group[sub_group["_capture_area"] == chosen_area]
                if len(same_area) >= 1:
                    chosen_rows[file_type] = same_area.iloc[0]
                    continue
                raw_named = sub_group[sub_group["Filename"].str.contains("raw", case=False, na=False)]
                chosen_rows[file_type] = raw_named.iloc[0] if len(raw_named) > 0 else sub_group.iloc[0]
                print(f"  WARNING: could not find '{file_type}' matching capture area {chosen_area} "
                      f"in {section_id}; falling back, verify this section's result carefully.")
            else:
                chosen_rows[file_type] = sub_group.iloc[0]

        for file_type, chosen_row in chosen_rows.items():
            local_path = download_file(chosen_row["entityId"], section_dir)
            standardize_name(local_path, file_type)

        try:
            adata = build_section_adata(section_dir)
        except Exception as e:
            print(f"  SKIPPED {section_id}: {e}")
            continue

        h5ad_path = processed_dir / f"{section_id}.h5ad"
        adata.write_h5ad(h5ad_path)
        print(f"  Saved {h5ad_path} ({adata.n_obs} spots under tissue)")

        present_genes = [g for g in GENE_PANEL if g in adata.var_names]
        missing_genes = [g for g in GENE_PANEL if g not in adata.var_names]
        if missing_genes:
            print(f"  WARNING: genes not found in {section_id}: {missing_genes}")

        panel_df = pd.DataFrame(
            adata[:, present_genes].X.toarray(),
            columns=present_genes,
            index=adata.obs_names,
        )
        panel_df["organ"] = organ
        panel_df["section_id"] = section_id
        panel_df["spatial_x"] = adata.obsm["spatial"][:, 0]
        panel_df["spatial_y"] = adata.obsm["spatial"][:, 1]
        panel_rows.append(panel_df)

    if panel_rows:
        combined = pd.concat(panel_rows)
        out_csv = Path("results") / f"step4_panel_expression_{organ}.csv"
        combined.to_csv(out_csv)
        print(f"\nCombined panel expression written to {out_csv} ({len(combined)} spots total)")
    else:
        print("\nNo sections processed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step4b_download_and_build.py <manifest_csv> <organ_label>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
