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
    parser = argparse.ArgumentParser(
        description="Verify stGCL spot-to-H&E alignment before H-ViT extraction."
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--sample",
        default=None,
        help="Canonical sample ID. Omit to check every sample in the cohort.",
    )
    parser.add_argument("--outdir", default="results/alignment_checks/stGCL")
    parser.add_argument(
        "--require-image",
        action="store_true",
        help="Exit nonzero if any checked sample cannot safely use its image.",
    )
    args = parser.parse_args()

    cohort_cfg = PROJECT_ROOT / "configs" / "cohorts" / f"{args.cohort}.yaml"
    samples = discover_samples(cohort_cfg, project_root=PROJECT_ROOT)
    if args.sample is not None:
        samples = [s for s in samples if str(s["sample_id"]) == args.sample]
        if not samples:
            raise ValueError(f"Sample {args.sample} not found in {args.cohort}")

    failures: list[str] = []
    for sample in samples:
        adata = sc.read_h5ad(sample["h5ad_path"])
        adata.obs_names = adata.obs_names.astype(str)
        alignment = resolve_image_alignment(
            sample,
            adata,
            overlay_outdir=PROJECT_ROOT / args.outdir / args.cohort,
        )
        aligned = alignment["aligned_coords"].copy()
        aligned.index = aligned.index.astype(str)
        matched = aligned.reindex(adata.obs_names.astype(str))
        complete = not matched[["x", "y"]].isna().any().any()
        usable = bool(alignment["image_enabled"] and complete)
        print(
            f"sample={sample['sample_id']} mode={alignment['mode']} "
            f"image_enabled={alignment['image_enabled']} spot_order_complete={complete} "
            f"overlay={alignment['overlay_path']}"
        )
        if not usable:
            failures.append(str(sample["sample_id"]))

    print(f"checked={len(samples)} usable={len(samples) - len(failures)} failures={failures}")
    if args.require_image and failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

