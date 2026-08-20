#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate STMSC spot-to-image alignment and save overlays.")
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "code_Xin"))
    from config_utils import resolve_cohort_config_path
    from loader import discover_samples
    from spatial_alignment import resolve_image_alignment

    samples = discover_samples(resolve_cohort_config_path(root, args.cohort), project_root=root)
    outdir = root / "results" / "STMSC" / args.cohort
    overlay_dir = outdir / "alignment_overlays"
    rows = []
    for sample in samples:
        adata = sc.read_h5ad(sample["h5ad_path"])
        result = resolve_image_alignment(sample, adata, overlay_dir)
        rows.append(
            {
                "cohort": args.cohort,
                "sample_id": sample["sample_id"],
                "image_path": sample.get("image_path"),
                "image_enabled": result["image_enabled"],
                "alignment_mode": result["mode"],
                "note": result["note"],
                "score": json.dumps(result.get("score")),
                "summary": json.dumps(result.get("summary")),
                "overlay_path": result.get("overlay_path"),
            }
        )
    outdir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "image_alignment_check.csv", index=False)
    print(table[["sample_id", "image_enabled", "alignment_mode", "overlay_path"]].to_string(index=False))
    print(f"Saved: {outdir / 'image_alignment_check.csv'}")


if __name__ == "__main__":
    main()

