from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_shard(shard: str | None) -> tuple[int, int] | tuple[None, None]:
    if not shard:
        return None, None
    idx, total = shard.split("/", 1)
    return int(idx), int(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Starfysh shard/group checkpoint progress.")
    parser.add_argument("cohort")
    parser.add_argument("--project_root", default="/path/to/MOCHA")
    parser.add_argument("--method", default="Starfysh")
    parser.add_argument("--shard", default=None, help="Optional shard spec like 0/8.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(project_root / "code_Xin"))

    from config_utils import load_pipeline_config, resolve_cohort_config_path
    from loader import discover_samples
    from runners.run_starfysh import select_group_shard, split_samples_for_integration

    cfg = load_pipeline_config(project_root, args.method, args.cohort)
    samples = discover_samples(resolve_cohort_config_path(project_root, args.cohort), project_root)
    scope = str(cfg.get("runtime", {}).get("integration_scope", "cohort"))
    groups = split_samples_for_integration(samples, scope)

    shard_index, shard_total = parse_shard(args.shard)
    if shard_index is not None and shard_total is not None:
        groups = select_group_shard(groups, shard_index, shard_total)
        outdir = (
            project_root
            / "results"
            / args.method
            / args.cohort
            / "_partials"
            / f"shard_{shard_index}_of_{shard_total}"
        )
    else:
        outdir = project_root / "results" / args.method / args.cohort

    rows = []
    total_spots = 0
    for idx, (group_name, group_samples) in enumerate(groups.items(), start=1):
        pred_path = outdir / group_name / "predictions.csv"
        n_spots = None
        status = "pending"
        mtime = None
        if pred_path.exists():
            status = "done"
            mtime = pd.Timestamp.fromtimestamp(pred_path.stat().st_mtime).isoformat()
            try:
                n_spots = int(sum(1 for _ in pred_path.open("r", encoding="utf-8")) - 1)
                total_spots += max(n_spots, 0)
            except OSError:
                n_spots = None
        rows.append(
            {
                "index": idx,
                "group": group_name,
                "sample_ids": ",".join(str(x["sample_id"]) for x in group_samples),
                "status": status,
                "n_spots": n_spots,
                "checkpoint_mtime": mtime,
                "checkpoint": str(pred_path),
            }
        )

    done = sum(row["status"] == "done" for row in rows)
    total = len(rows)
    print(f"cohort={args.cohort}")
    print(f"scope={scope}")
    print(f"outdir={outdir}")
    print(f"groups_done={done}/{total}")
    print(f"spots_done={total_spots}")
    print("")

    for row in rows:
        n_spots = "" if row["n_spots"] is None else row["n_spots"]
        mtime = "" if row["checkpoint_mtime"] is None else row["checkpoint_mtime"]
        print(
            f"{row['index']:>3}/{total:<3} {row['status']:<7} "
            f"{row['group']} spots={n_spots} mtime={mtime}"
        )


if __name__ == "__main__":
    main()

