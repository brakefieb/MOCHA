from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc

from result_utils import (
    build_performance_df,
    save_evaluation_summary,
    save_true_vs_pred_plots,
    validate_prediction_schema,
    write_performance_tables,
    write_predictions_tables,
)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_run_metadata(outdir: Path, meta: dict[str, Any]) -> None:
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def get_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def extract_coords(adata: sc.AnnData) -> pd.DataFrame:
    """
    Prefer the same style of spatial coordinates used in the official BayeSMART
    DLPFC tutorial: x/y spot coordinates. Fall back to spatial embeddings, then
    array indices only when true spatial coordinates are unavailable.
    """
    obs = adata.obs

    candidate_pairs = [
        ("x", "y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("pxl_col", "pxl_row"),
        ("array_col", "array_row"),
        ("col", "row"),
    ]

    for c1, c2 in candidate_pairs:
        if c1 in obs.columns and c2 in obs.columns:
            coords = obs[[c1, c2]].copy()
            coords.columns = ["x", "y"]
            return coords

    for key in ["spatial", "X_spatial", "S"]:
        if key in adata.obsm:
            spatial = adata.obsm[key]

            if isinstance(spatial, pd.DataFrame):
                coords = spatial.iloc[:, :2].copy()
                coords.index = adata.obs_names
                coords.columns = ["x", "y"]
                return coords

            if hasattr(spatial, "toarray"):
                spatial = spatial.toarray()

            spatial = np.asarray(spatial)
            if spatial.ndim != 2 or spatial.shape[1] < 2:
                raise ValueError(
                    f"obsm['{key}'] must be 2D with at least 2 columns, got {spatial.shape}"
                )

            return pd.DataFrame(
                spatial[:, :2],
                index=adata.obs_names,
                columns=["x", "y"],
            )

    raise ValueError(
        "Could not find coordinate columns in adata.obs or adata.obsm. "
        f"obs columns: {list(obs.columns)}, obsm keys: {list(adata.obsm.keys())}"
    )


def apply_label_mapping(
    labels: pd.Series,
    label_mapping: dict[str, Any],
    unmapped_default: str | None = None,
) -> pd.Series:
    if not label_mapping:
        return labels.astype(str)

    def normalize_label(value: Any) -> str:
        return " ".join(str(value).strip().split()).lower()

    reverse_mapping: dict[str, str] = {}
    for target_label, source_labels in label_mapping.items():
        target = str(target_label).strip()
        reverse_mapping[normalize_label(target)] = target
        if not isinstance(source_labels, list):
            source_labels = [source_labels]
        for source_label in source_labels:
            reverse_mapping[normalize_label(source_label)] = target

    if unmapped_default is None:
        mapped = labels.astype(str).map(
            lambda x: reverse_mapping.get(normalize_label(x), str(x).strip())
        )
    else:
        mapped = labels.astype(str).map(
            lambda x: reverse_mapping.get(normalize_label(x), unmapped_default)
        )
    return pd.Series(mapped, index=labels.index, name=labels.name)


def extract_ground_truth(
    obs: pd.DataFrame,
    label_mapping: dict[str, Any] | None = None,
    unmapped_default: str | None = None,
) -> pd.Series:
    candidates = [
        "annotation",
        "Classification",
        "classification",
        "ground_truth",
        "layer_guess",
        "spatialLIBD",
        "manual_label",
        "label",
        "region",
        "z",
    ]
    for col in candidates:
        if col in obs.columns:
            return apply_label_mapping(
                obs[col].astype(str),
                label_mapping or {},
                unmapped_default=unmapped_default,
            )

    return pd.Series(["NA"] * obs.shape[0], index=obs.index, name="ground_truth")


def get_dense_matrix(adata: sc.AnnData) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D expression matrix, got shape {X.shape}")

    return X


def is_git_lfs_pointer(path: Path) -> bool:
    with open(path, "rb") as f:
        header = f.read(128)
    return header.startswith(b"version https://git-lfs.github.com/spec/")


def stream_subprocess(cmd: list[str], stdout_path: Path, stderr_path: Path) -> int:
    """
    Run a command while teeing stdout/stderr to both Slurm logs and per-run logs.
    This keeps long BayeSMART MCMC jobs visible in the HPC .out/.err files.
    """
    with open(stdout_path, "a", encoding="utf-8") as out_f, open(
        stderr_path, "a", encoding="utf-8"
    ) as err_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def forward(pipe, log_file, console):
            try:
                for line in iter(pipe.readline, ""):
                    log_file.write(line)
                    log_file.flush()
                    console.write(line)
                    console.flush()
            finally:
                pipe.close()

        out_thread = threading.Thread(
            target=forward, args=(proc.stdout, out_f, sys.stdout), daemon=True
        )
        err_thread = threading.Thread(
            target=forward, args=(proc.stderr, err_f, sys.stderr), daemon=True
        )
        out_thread.start()
        err_thread.start()
        return_code = proc.wait()
        out_thread.join()
        err_thread.join()

    return return_code


def split_samples_for_integration(
    samples: list[dict[str, Any]],
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
    if mode == "subgroup":
        groups: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            subgroup = sample.get("subgroup") or "ungrouped"
            groups.setdefault(str(subgroup), []).append(sample)
        return groups

    return {"all_samples": samples}


def export_one_sample(sample: dict[str, Any], interm_dir: Path) -> dict[str, Any]:
    h5ad_path = Path(sample["h5ad_path"])
    if is_git_lfs_pointer(h5ad_path):
        raise ValueError(
            f"{h5ad_path} is a Git LFS pointer, not a real .h5ad file. "
            "Run git lfs pull or copy the real data files before running BayeSMART."
        )

    adata = sc.read_h5ad(h5ad_path)

    coords = extract_coords(adata)
    gt = extract_ground_truth(
        adata.obs,
        sample.get("label_mapping", {}),
        unmapped_default=sample.get("label_unmapped_default"),
    )
    X = get_dense_matrix(adata)

    counts_df = pd.DataFrame(
        X,
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )
    coords.index = adata.obs_names.astype(str)
    gt_df = pd.DataFrame({"ground_truth": gt.astype(str)}, index=adata.obs_names.astype(str))

    sample_prefix = sample["sample_id"]

    counts_path = interm_dir / f"{sample_prefix}__counts.csv"
    coords_path = interm_dir / f"{sample_prefix}__coords.csv"
    gt_path = interm_dir / f"{sample_prefix}__ground_truth.csv"

    counts_df.to_csv(counts_path, index_label="spot_id")
    coords.to_csv(coords_path, index_label="spot_id")
    gt_df.to_csv(gt_path, index_label="spot_id")

    record = {
        "sample_id": sample["sample_id"],
        "subgroup": sample.get("subgroup"),
        "counts_path": str(counts_path),
        "coords_path": str(coords_path),
        "ground_truth_path": str(gt_path),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }
    return record


def export_group_samples(
    group_samples: list[dict[str, Any]],
    interm_dir: Path,
    n_workers: int,
) -> list[dict[str, Any]]:
    if n_workers <= 1 or len(group_samples) <= 1:
        return [export_one_sample(sample, interm_dir) for sample in group_samples]

    max_workers = min(n_workers, len(group_samples))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        records = list(ex.map(lambda s: export_one_sample(s, interm_dir), group_samples))
    return records


def run_group(
    group_name: str,
    group_samples: list[dict[str, Any]],
    outdir: Path,
    method_root: Path,
    r_runner: Path,
    cohort_name: str,
    rscript_bin: str,
    params: dict[str, Any],
    preproc: dict[str, Any],
    runtime_cfg: dict[str, Any],
    outputs_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    k = params.get("k", 7)
    w = params.get("w", 0.05)
    n_neighbor = params.get("n_neighbor", 8)
    f_val = params.get("f_val", 1)

    gene_select = preproc.get("gene_select", "hvgs")
    n_gene = preproc.get("n_gene", 2000)
    pcn = preproc.get("pcn", 3)

    mcmc_iter = int(runtime_cfg.get("mcmc_iter", params.get("mcmc_iter", 20000)))
    store_thin = int(runtime_cfg.get("store_thin", params.get("store_thin", 1)))
    n_workers = int(runtime_cfg.get("preprocessing_workers", os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    n_workers = max(1, n_workers)

    save_posterior = outputs_cfg.get("save_posterior_rds", False)
    save_posterior_z = outputs_cfg.get("save_posterior_z", False)
    save_trace_plot = outputs_cfg.get("save_trace_plot", False)
    save_mu_median = outputs_cfg.get("save_mu_median", False)
    save_omega_median = outputs_cfg.get("save_omega_median", False)
    save_posterior_summary = outputs_cfg.get("save_posterior_summary", False)
    save_sample_predictions = outputs_cfg.get("save_sample_predictions", False)
    save_intermediate = outputs_cfg.get("save_intermediate", False)
    save_group_predictions = outputs_cfg.get("save_group_predictions", False)

    group_dir = outdir / group_name
    safe_mkdir(group_dir)
    interm_dir = group_dir / "_group_intermediate"
    safe_mkdir(interm_dir)

    prep_log = outdir / "stdout.log"
    with open(prep_log, "a", encoding="utf-8") as f:
        f.write(f"[python] Starting BayeSMART export for group {group_name}\n")
        f.write(f"[python] Group samples: {len(group_samples)}\n")
        f.write(f"[python] Preprocessing workers: {n_workers}\n")

    manifest_records = export_group_samples(group_samples, interm_dir, n_workers=n_workers)

    with open(prep_log, "a", encoding="utf-8") as f:
        for idx, record in enumerate(manifest_records, start=1):
            f.write(
                f"[python] Exported sample {record['sample_id']} "
                f"({idx}/{len(manifest_records)}) in group {group_name}\n"
            )

    manifest_path = interm_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

    cmd = [
        rscript_bin,
        str(r_runner),
        "--manifest", str(manifest_path),
        "--outdir", str(group_dir),
        "--method_root", str(method_root),
        "--cohort", cohort_name,
        "--k", str(k),
        "--w", str(w),
        "--n_neighbor", str(n_neighbor),
        "--f_val", str(f_val),
        "--gene_select", str(gene_select),
        "--n_gene", str(n_gene),
        "--pcn", str(pcn),
        "--save_posterior_rds", str(save_posterior).lower(),
        "--save_posterior_z", str(save_posterior_z).lower(),
        "--save_trace_plot", str(save_trace_plot).lower(),
        "--save_mu_median", str(save_mu_median).lower(),
        "--save_omega_median", str(save_omega_median).lower(),
        "--save_posterior_summary", str(save_posterior_summary).lower(),
        "--save_sample_predictions", str(save_sample_predictions).lower(),
        "--mcmc_iter", str(mcmc_iter),
        "--store_thin", str(store_thin),
        "--n_workers", str(n_workers),
    ]

    return_code = stream_subprocess(
        cmd,
        stdout_path=outdir / "stdout.log",
        stderr_path=outdir / "stderr.log",
    )
    if return_code != 0:
        raise RuntimeError(
            f"Rscript failed with code {return_code} for group {group_name}. "
            "See stderr.log for details."
        )

    pred_path = group_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"BayeSMART finished but predictions.csv is missing: {pred_path}")

    predictions_df = pd.read_csv(pred_path)
    predictions_df["group_name"] = group_name
    if not save_group_predictions and pred_path.exists():
        pred_path.unlink()

    group_record = {
        "group_name": group_name,
        "n_samples": len(group_samples),
        "sample_ids": [x["sample_id"] for x in group_samples],
        "n_spots": int(predictions_df.shape[0]),
        "subgroups": sorted({str(x.get("subgroup") or "ungrouped") for x in group_samples}),
        "preprocessing_workers": n_workers,
        "intermediate_dir": str(interm_dir),
    }

    intermediate_removed = False
    if not save_intermediate and interm_dir.exists():
        shutil.rmtree(interm_dir)
        intermediate_removed = True
    if not save_group_predictions and group_dir.exists() and not any(group_dir.iterdir()):
        group_dir.rmdir()

    return predictions_df, group_record, intermediate_removed


def run_bayesmart(
    samples: list[dict[str, Any]],
    project_root: Path,
    pipeline_cfg: dict[str, Any],
    rscript_bin: str = "Rscript",
) -> dict[str, Any]:
    """BayeSMART runner with optional cohort- or subgroup-level integration."""
    if not samples:
        raise ValueError("Empty samples list passed to run_bayesmart")

    method_name = pipeline_cfg.get("method_name", "BayeSMART")
    cohort_name = samples[0]["cohort_name"]

    params = pipeline_cfg.get("parameters", {})
    preproc = pipeline_cfg.get("preprocessing", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outputs_cfg = pipeline_cfg.get("outputs", {})
    k = params.get("k", 7)
    w = params.get("w", 0.05)
    n_neighbor = params.get("n_neighbor", 8)
    f_val = params.get("f_val", 1)
    gene_select = preproc.get("gene_select", "hvgs")
    n_gene = preproc.get("n_gene", 2000)
    pcn = preproc.get("pcn", 3)
    mcmc_iter = int(runtime_cfg.get("mcmc_iter", params.get("mcmc_iter", 20000)))
    store_thin = int(runtime_cfg.get("store_thin", params.get("store_thin", 1)))
    n_workers = int(runtime_cfg.get("preprocessing_workers", os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    n_workers = max(1, n_workers)
    integration_scope = str(runtime_cfg.get("integration_scope", "cohort"))
    save_posterior = outputs_cfg.get("save_posterior_rds", False)
    save_posterior_z = outputs_cfg.get("save_posterior_z", False)
    save_trace_plot = outputs_cfg.get("save_trace_plot", False)
    save_mu_median = outputs_cfg.get("save_mu_median", False)
    save_omega_median = outputs_cfg.get("save_omega_median", False)
    save_posterior_summary = outputs_cfg.get("save_posterior_summary", False)
    save_sample_predictions = outputs_cfg.get("save_sample_predictions", False)
    save_intermediate = outputs_cfg.get("save_intermediate", False)

    method_root = project_root / "methods" / "BayeSMART" / "code"
    r_runner = project_root / "code_Xin" / "runners" / "run_bayesmart.R"

    if not method_root.exists():
        raise FileNotFoundError(f"Method root not found: {method_root}")
    if not r_runner.exists():
        raise FileNotFoundError(f"R runner not found: {r_runner}")

    outdir = project_root / "results" / method_name / cohort_name
    safe_mkdir(outdir)

    t0 = time.time()

    meta: dict[str, Any] = {
        "method": method_name,
        "cohort": cohort_name,
        "status": "running",
        "runtime_level": runtime_cfg.get("level", "cohort"),
        "integration_scope": integration_scope,
        "n_samples": len(samples),
        "parameters": {
            "k": k,
            "w": w,
            "n_neighbor": n_neighbor,
            "f_val": f_val,
        },
        "preprocessing": {
            "gene_select": gene_select,
            "n_gene": n_gene,
            "pcn": pcn,
        },
        "outputs": {
            "save_posterior_rds": save_posterior,
            "save_posterior_z": save_posterior_z,
            "save_trace_plot": save_trace_plot,
            "save_mu_median": save_mu_median,
            "save_omega_median": save_omega_median,
            "save_posterior_summary": save_posterior_summary,
            "save_sample_predictions": save_sample_predictions,
            "save_intermediate": save_intermediate,
        },
        "mcmc": {
            "iter": mcmc_iter,
            "store_thin": store_thin,
        },
        "parallel": {
            "preprocessing_workers": n_workers,
        },
        "runtime_seconds": None,
        "memory_mb": None,
        "error_message": None,
        "groups": [],
    }
    write_run_metadata(outdir, meta)

    prep_log = outdir / "stdout.log"
    with open(prep_log, "a", encoding="utf-8") as f:
        f.write(f"[python] Starting BayeSMART run for {cohort_name}\n")
        f.write(f"[python] Integration scope: {integration_scope}\n")
        f.write(f"[python] Samples to export: {len(samples)}\n")

    all_predictions: list[pd.DataFrame] = []
    predictions_df: pd.DataFrame | None = None
    any_intermediate_removed = False
    try:
        groups = split_samples_for_integration(samples, integration_scope)
        for group_name, group_samples in groups.items():
            group_predictions, group_record, intermediate_removed = run_group(
                group_name=group_name,
                group_samples=group_samples,
                outdir=outdir,
                method_root=method_root,
                r_runner=r_runner,
                cohort_name=cohort_name,
                rscript_bin=rscript_bin,
                params=params,
                preproc=preproc,
                runtime_cfg=runtime_cfg,
                outputs_cfg=outputs_cfg,
            )
            meta["groups"].append(group_record)
            any_intermediate_removed = any_intermediate_removed or intermediate_removed
            all_predictions.append(group_predictions)
            write_run_metadata(outdir, meta)

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df = write_predictions_tables(predictions_df, outdir)
        meta["status"] = "success"
        meta["intermediate_removed"] = any_intermediate_removed

    except Exception as e:
        meta["status"] = "failed"
        meta["error_message"] = str(e)
        with open(outdir / "stderr.log", "a", encoding="utf-8") as f:
            f.write(f"[python] Exception before/around R run: {e}\n")

    meta["runtime_seconds"] = round(time.time() - t0, 4)
    meta["memory_mb"] = round(get_memory_mb(), 3)

    if meta["status"] == "success":
        if predictions_df is None:
            pred_path = outdir / "predictions.csv"
            if not pred_path.exists():
                raise FileNotFoundError(f"BayeSMART finished but predictions.csv is missing: {pred_path}")
            predictions_df = pd.read_csv(pred_path)
        validate_prediction_schema(predictions_df, source="BayeSMART predictions")

        performance_df = build_performance_df(
            cohort_name=cohort_name,
            method_name=method_name,
            runtime_seconds=meta["runtime_seconds"],
            memory_mb=meta["memory_mb"],
        )
        write_performance_tables(performance_df, outdir)
        save_evaluation_summary(predictions_df, outdir, cohort_name, method_name)
        save_true_vs_pred_plots(predictions_df, outdir, cohort_name, method_name)

    write_run_metadata(outdir, meta)

    return meta
