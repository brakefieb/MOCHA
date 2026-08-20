from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="List Starfysh integration groups for a cohort.")
    parser.add_argument("cohort")
    parser.add_argument("--project_root", default="/path/to/MOCHA")
    parser.add_argument(
        "--shard",
        default=None,
        help="Optional shard spec like 0/4; prints only groups assigned to that shard.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(project_root / "code_Xin"))

    from config_utils import load_pipeline_config, resolve_cohort_config_path
    from loader import discover_samples
    from runners.run_starfysh import select_group_shard, split_samples_for_integration

    cfg = load_pipeline_config(project_root, "Starfysh", args.cohort)
    samples = discover_samples(resolve_cohort_config_path(project_root, args.cohort), project_root)
    scope = str(cfg.get("runtime", {}).get("integration_scope", "cohort"))
    groups = split_samples_for_integration(samples, scope)
    if args.shard:
        idx, total = args.shard.split("/", 1)
        groups = select_group_shard(groups, int(idx), int(total))

    for group_name in groups.keys():
        print(group_name)


if __name__ == "__main__":
    main()

