from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Starfysh partial group jobs.")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--project_root", default="/path/to/MOCHA")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(project_root / "code_Xin"))

    from result_utils import (
        build_performance_df,
        save_evaluation_summary,
        save_true_vs_pred_plots,
        write_performance_tables,
        write_predictions_tables,
    )

    outdir = project_root / "results" / "Starfysh" / args.cohort
    partial_root = outdir / "_partials"
    if not partial_root.exists():
        raise FileNotFoundError(f"Partial directory not found: {partial_root}")

    all_pred_paths = sorted(partial_root.rglob("predictions.csv"))
    pred_paths = []
    for pred_path in all_pred_paths:
        child_prediction_paths = list(pred_path.parent.glob("*/predictions.csv"))
        if child_prediction_paths:
            continue
        pred_paths.append(pred_path)
    if not pred_paths:
        raise FileNotFoundError(f"No partial predictions found under {partial_root}")

    parts = []
    runtime_seconds = 0.0
    memory_mb = 0.0
    partial_meta = []
    for pred_path in pred_paths:
        part = pd.read_csv(pred_path)
        parts.append(part)

        meta_path = pred_path.parent / "run_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            partial_meta.append(meta)
            runtime_seconds += float(meta.get("runtime_seconds") or 0.0)
            memory_mb = max(memory_mb, float(meta.get("memory_mb") or 0.0))

    pred = pd.concat(parts, ignore_index=True)
    pred = write_predictions_tables(pred, outdir)

    perf = build_performance_df(
        cohort_name=args.cohort,
        method_name="Starfysh",
        runtime_seconds=round(runtime_seconds, 4),
        memory_mb=round(memory_mb, 3),
    )
    write_performance_tables(perf, outdir)
    save_evaluation_summary(pred, outdir, args.cohort, "Starfysh")
    save_true_vs_pred_plots(pred, outdir, args.cohort, "Starfysh")

    merged_meta = {
        "method": "Starfysh",
        "cohort": args.cohort,
        "status": "success",
        "merged_from_partials": True,
        "n_partials": len(pred_paths),
        "partial_prediction_files": [str(p) for p in pred_paths],
        "runtime_seconds_sum": round(runtime_seconds, 4),
        "memory_mb_max": round(memory_mb, 3),
        "partials": partial_meta,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(merged_meta, indent=2))
    print(f"Merged {len(pred_paths)} Starfysh partials into {outdir}")


if __name__ == "__main__":
    main()

