#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "code_Xin"))
    from config_utils import load_pipeline_config, resolve_cohort_config_path
    from loader import discover_samples

    samples = discover_samples(resolve_cohort_config_path(root, args.cohort), root)
    cfg = load_pipeline_config(root, "STMSC", args.cohort)
    mode = cfg.get("runtime", {}).get("integration_scope", "subgroup_or_sample")
    groups = {}
    for sample in samples:
        if mode == "cohort":
            key = "all_samples"
        elif mode == "sample":
            key = str(sample["sample_id"])
        else:
            key = str(sample.get("subgroup") or sample["sample_id"])
        groups.setdefault(key, []).append(str(sample["sample_id"]))
    groups = dict(sorted(groups.items()))
    if args.count:
        print(len(groups))
    else:
        for idx, (name, members) in enumerate(groups.items()):
            print(f"{idx}\t{name}\t{','.join(members)}")


if __name__ == "__main__":
    main()

