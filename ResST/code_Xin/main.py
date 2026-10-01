from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import load_pipeline_config, resolve_cohort_config_path
from loader import discover_samples
from registry import METHOD_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified MOCHA pipeline entry point.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Optional: only run the first N discovered samples",
)
    
    # optional CLI overrides
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--w", type=float, default=None)
    parser.add_argument("--n_neighbor", type=int, default=None)
    parser.add_argument("--mcmc_iter", type=int, default=None)
    parser.add_argument("--store_thin", type=int, default=None)
    parser.add_argument("--preprocessing_workers", type=int, default=None)
    parser.add_argument(
        "--group_idx",
        type=int,
        default=None,
        help="Optional cohort-runner group index (used by STMSC Slurm arrays)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--mapping_epochs", type=int, default=None)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    pipeline_cfg = load_pipeline_config(
        project_root=project_root,
        method_name=args.method,
        cohort_name=args.cohort,
    )

    # CLI override
    if args.k is not None:
        pipeline_cfg.setdefault("parameters", {})["k"] = args.k
    if args.w is not None:
        pipeline_cfg.setdefault("parameters", {})["w"] = args.w
    if args.n_neighbor is not None:
        pipeline_cfg.setdefault("parameters", {})["n_neighbor"] = args.n_neighbor
    if args.mcmc_iter is not None:
        pipeline_cfg.setdefault("runtime", {})["mcmc_iter"] = args.mcmc_iter
    if args.store_thin is not None:
        pipeline_cfg.setdefault("runtime", {})["store_thin"] = args.store_thin
    if args.preprocessing_workers is not None:
        pipeline_cfg.setdefault("runtime", {})["preprocessing_workers"] = args.preprocessing_workers
    if args.group_idx is not None:
        pipeline_cfg.setdefault("runtime", {})["group_idx"] = args.group_idx
    if args.epochs is not None:
        pipeline_cfg.setdefault("runtime", {})["epochs"] = args.epochs
    if args.mapping_epochs is not None:
        pipeline_cfg.setdefault("runtime", {})["mapping_epochs"] = args.mapping_epochs

    cohort_cfg_path = resolve_cohort_config_path(project_root, args.cohort)
    samples = discover_samples(cohort_cfg_path, project_root=project_root)

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        samples = samples[:args.max_samples]

    method_name = args.method
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}")

    runner = METHOD_REGISTRY[method_name]
    runtime_level = pipeline_cfg.get("runtime", {}).get("level", "sample")

    print(f"Method : {method_name}")
    print(f"Cohort : {args.cohort}")
    print(f"Samples: {len(samples)}")
    print(f"Level  : {runtime_level}")

    if args.sample_idx is not None:
        if runtime_level == "cohort":
            raise ValueError(
                f"{method_name} is configured as cohort-level; do not use --sample_idx"
            )
        if args.sample_idx < 0 or args.sample_idx >= len(samples):
            raise IndexError("sample_idx out of range")
        samples = [samples[args.sample_idx]]

    result = None
    if runtime_level == "cohort":
        result = runner(
            samples=samples,
            project_root=project_root,
            pipeline_cfg=pipeline_cfg,
            rscript_bin=args.rscript,
        )
    else:
        results = []
        for sample in samples:
            results.append(
                runner(
                sample=sample,
                project_root=project_root,
                pipeline_cfg=pipeline_cfg,
                rscript_bin=args.rscript,
                )
            )
        result = results[-1] if results else None

    if isinstance(result, dict) and result.get("status") == "failed":
        raise RuntimeError(
            f"{method_name} failed for cohort {args.cohort}: {result.get('error_message')}"
        )


if __name__ == "__main__":
    main()
