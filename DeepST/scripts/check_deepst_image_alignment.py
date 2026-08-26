#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scanpy as sc

from code_Xin.loader import discover_samples
from code_Xin.spatial_alignment import resolve_image_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual check for DeepST image/spot alignment.")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--sample", required=True, help="Canonical sample id, e.g. 151507 or A1")
    parser.add_argument(
        "--outdir",
        default="results/alignment_checks",
        help="Directory to write overlay figures.",
    )
    args = parser.parse_args()

    cohort_cfg = PROJECT_ROOT / "configs" / "cohorts" / f"{args.cohort}.yaml"
    if not cohort_cfg.exists():
        raise FileNotFoundError(f"Cohort config not found: {cohort_cfg}")

    samples = discover_samples(cohort_cfg, project_root=PROJECT_ROOT)
    sample = next((s for s in samples if str(s["sample_id"]) == args.sample), None)
    if sample is None:
        raise ValueError(f"Sample {args.sample} not found in cohort {args.cohort}")

    adata = sc.read_h5ad(sample["h5ad_path"])
    adata.obs_names = adata.obs_names.astype(str)

    alignment = resolve_image_alignment(
        sample=sample,
        adata=adata,
        overlay_outdir=PROJECT_ROOT / args.outdir / args.cohort,
    )

    print(f"Cohort     : {args.cohort}")
    print(f"Sample     : {args.sample}")
    print(f"Image path : {alignment['image_path']}")
    print(f"Coord mode : {alignment['mode']}")
    print(f"Image used : {alignment['image_enabled']}")
    print(f"Note       : {alignment['note']}")
    print(f"Score      : {alignment['score']}")
    print(f"Summary    : {alignment['summary']}")
    print(f"Overlay    : {alignment['overlay_path']}")


if __name__ == "__main__":
    main()
