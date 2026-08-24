from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import resource
import shutil
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image, ImageOps
from scipy import sparse

from result_utils import (
    adjusted_rand_index,
    build_performance_df,
    save_evaluation_summary,
    save_true_vs_pred_plots,
    write_performance_tables,
    write_predictions_tables,
)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_progress(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def get_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def apply_label_mapping(
    labels: pd.Series,
    label_mapping: dict[str, Any],
    unmapped_default: str | None = None,
) -> pd.Series:
    if not label_mapping:
        return labels.astype(str)

    reverse_mapping: dict[str, str] = {}
    for target_label, source_labels in label_mapping.items():
        target = str(target_label).strip()
        reverse_mapping[target] = target
        if not isinstance(source_labels, list):
            source_labels = [source_labels]
        for source_label in source_labels:
            reverse_mapping[str(source_label).strip()] = target

    if unmapped_default is None:
        mapped = labels.astype(str).str.strip().map(lambda x: reverse_mapping.get(x, x))
    else:
        mapped = labels.astype(str).str.strip().map(
            lambda x: reverse_mapping.get(x, unmapped_default)
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


def infer_n_domains(labels: pd.Series) -> int | None:
    clean = (
        labels.astype(str)
        .str.strip()
        .replace({"": "NA", "nan": "NA", "None": "NA", "NA": "NA"})
    )
    uniques = sorted(x for x in clean.unique() if x not in {"NA", "Filtered"})
    return len(uniques) if uniques else None


def resolve_n_domains(adata: sc.AnnData, configured_value: Any) -> int:
    if configured_value not in (None, "auto"):
        return int(configured_value)
    inferred = infer_n_domains(adata.obs["ground_truth"])
    if inferred is None:
        raise ValueError(
            "Could not infer SpaConTDS K from ground-truth labels. "
            "Please set parameters.n_domains or parameters.k explicitly."
        )
    return inferred


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch_threads = os.environ.get("SLURM_CPUS_PER_TASK")
        if torch_threads:
            torch.set_num_threads(max(1, int(torch_threads)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def choose_device(configured: str) -> str:
    if configured != "auto":
        return configured
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def ensure_spacontds_path(method_root: Path) -> None:
    method_root_str = str(method_root)
    sys.path = [p for p in sys.path if p != method_root_str]
    sys.path.insert(0, method_root_str)


def load_spacontds_main(method_root: Path):
    ensure_spacontds_path(method_root)
    module_path = method_root / "main.py"
    module_name = "_mocha_spacontds_main"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SpaConTDS main.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prepare_spacontds_workdir(method_root: Path, sample_dir: Path) -> Path:
    """Create a per-sample cwd for upstream relative model paths."""
    workdir = sample_dir / "workdir"
    model_dir = workdir / "model"
    decoder_dir = workdir / "Decoder"
    pretrain_dir = model_dir / "pretrain"
    safe_mkdir(pretrain_dir)
    safe_mkdir(decoder_dir)

    for folder in [model_dir, decoder_dir]:
        for path in folder.glob("*.pth"):
            path.unlink(missing_ok=True)

    source_pretrain = method_root / "model" / "pretrain" / "Hist2ST_convmixer.pth"
    target_pretrain = pretrain_dir / "Hist2ST_convmixer.pth"
    if source_pretrain.exists() and not target_pretrain.exists():
        try:
            target_pretrain.symlink_to(source_pretrain)
        except OSError:
            shutil.copy2(source_pretrain, target_pretrain)

    return workdir


def prepare_spacontds_input(
    sample: dict[str, Any],
    sample_dir: Path,
    patch_size: int,
    overlay_outdir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    from spatial_alignment import extract_coords, resolve_image_alignment

    sample_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(sample["h5ad_path"])
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs["spot_id"] = adata.obs_names.astype(str)
    adata.obs["sample_id"] = str(sample["sample_id"])
    adata.obs["ground_truth"] = extract_ground_truth(
        adata.obs,
        sample.get("label_mapping", {}),
        unmapped_default=sample.get("label_unmapped_default"),
    ).astype(str).values

    alignment = resolve_image_alignment(
        sample=sample,
        adata=adata,
        overlay_outdir=overlay_outdir,
    )
    coords = alignment["aligned_coords"].reindex(adata.obs_names.astype(str))
    if coords.isnull().any().any():
        coords = extract_coords(adata)
    coords = coords[["x", "y"]].astype(float)

    image_array = alignment.get("image_array")
    image_enabled = bool(alignment.get("image_enabled")) and image_array is not None
    pathology_image_used = image_enabled
    image_note = str(alignment.get("note"))
    if not pathology_image_used:
        x_span = max(float(coords["x"].max() - coords["x"].min()), 512.0)
        y_span = max(float(coords["y"].max() - coords["y"].min()), 512.0)
        width = int(np.ceil(x_span + 4 * patch_size))
        height = int(np.ceil(y_span + 4 * patch_size))
        image = Image.new("RGB", (width, height), color=(245, 245, 245))
        coords["x"] = coords["x"] - coords["x"].min() + 2 * patch_size
        coords["y"] = coords["y"] - coords["y"].min() + 2 * patch_size
        image_note = (
            f"{image_note}; pathology_image_not_used; "
            "generated blank image for SpaConTDS input."
        )
    else:
        image = Image.fromarray(np.asarray(image_array).astype(np.uint8)).convert("RGB")
        width, height = image.size
        min_x, max_x = float(coords["x"].min()), float(coords["x"].max())
        min_y, max_y = float(coords["y"].min()), float(coords["y"].max())
        left = max(0, int(np.ceil(patch_size - min_x)))
        top = max(0, int(np.ceil(patch_size - min_y)))
        right = max(0, int(np.ceil(max_x + patch_size - width + 1)))
        bottom = max(0, int(np.ceil(max_y + patch_size - height + 1)))
        if any(v > 0 for v in [left, top, right, bottom]):
            image = ImageOps.expand(
                image,
                border=(left, top, right, bottom),
                fill=(245, 245, 245),
            )
            coords["x"] = coords["x"] + left
            coords["y"] = coords["y"] + top

    coords_int = np.rint(coords[["x", "y"]].to_numpy()).astype(int)
    adata.obsm["spatial"] = coords_int
    adata.obs["imagecol"] = coords_int[:, 0]
    adata.obs["imagerow"] = coords_int[:, 1]
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    h5ad_path = sample_dir / "spacontds_input.h5ad"
    image_path = sample_dir / "spacontds_image.png"
    adata.write_h5ad(h5ad_path)
    image.save(image_path)

    info = {
        "sample_id": sample["sample_id"],
        "subgroup": sample.get("subgroup"),
        "h5ad_path": sample["h5ad_path"],
        "image_path": sample.get("image_path"),
        "spacontds_h5ad_path": str(h5ad_path),
        "spacontds_image_path": str(image_path),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "image_enabled": image_enabled,
        "pathology_image_used": pathology_image_used,
        "image_note": image_note,
        "alignment_mode": str(alignment.get("mode")),
        "alignment_overlay_path": alignment.get("overlay_path"),
        "alignment_score": alignment.get("score"),
        "alignment_summary": alignment.get("summary"),
    }
    return h5ad_path, image_path, info


def make_spacontds_args(
    method_root: Path,
    h5ad_path: Path,
    image_path: Path,
    sample_id: str,
    sample_dir: Path,
    params: dict[str, Any],
    runtime_cfg: dict[str, Any],
    n_domains: int,
) -> SimpleNamespace:
    ensure_spacontds_path(method_root)
    utils = importlib.import_module("utils")
    parser = utils.ArgumentParserTupleST()
    parser.set_params()
    args = parser.parse_args([])
    args = utils.parse_special_params(args)

    args.dataset = "MOCHA"
    args.dataset_dir = str(h5ad_path)
    args.img_dir = str(image_path)
    args.is_10x = False
    args.save_dir = str(sample_dir / "upstream_output")
    args.valid_cluster = int(n_domains)
    args.pseudo_cluster = int(n_domains)
    args.countfile_name = ""
    args.slice_name = str(sample_id)
    args.device = choose_device(str(runtime_cfg.get("device", "auto")))
    args.num_epochs = int(runtime_cfg.get("num_epochs", params.get("num_epochs", 50)))

    for key in [
        "patch_size",
        "graph_type",
        "rad_cutoff",
        "k_cutoff",
        "subgraph_neighbors",
        "gene_in_dim",
        "gene_hidden_dim",
        "gene_out_dim",
        "gene_num_layers",
        "gene_augtype",
        "gene_mask_pct",
        "gene_sigma_noise",
        "img_sigma_noise",
        "neg_type",
        "img_out_dim",
        "fc_dim",
        "hidden_dim",
        "mlp_layers",
        "p",
        "p_round",
        "iter_epochs",
        "tau",
        "batch_size",
        "lr",
        "alpha",
        "clone",
        "internal",
        "alpha_lr",
        "reawrd_alphalr",
        "reawrd_emdklr",
        "emb_weight",
        "k",
        "gene_only",
        "img_only",
        "img_feat",
    ]:
        if key in params:
            setattr(args, key, params[key])

    if isinstance(args.subgraph_neighbors, str):
        args.subgraph_neighbors = eval(args.subgraph_neighbors)
    if isinstance(args.fc_dim, str):
        args.fc_dim = eval(args.fc_dim)
    safe_mkdir(Path(args.save_dir))
    return args


def build_predictions(
    adata: sc.AnnData,
    cohort_name: str,
    method_name: str,
    sample_info: dict[str, Any],
) -> pd.DataFrame:
    coords = pd.DataFrame(
        adata.obsm["spatial"][:, :2],
        index=adata.obs_names.astype(str),
        columns=["x", "y"],
    )
    df = pd.DataFrame(
        {
            "cohort": cohort_name,
            "sampleID": adata.obs["sample_id"].astype(str).values,
            "spotID": adata.obs["spot_id"].astype(str).values,
            "x": coords["x"].astype(float).values,
            "y": coords["y"].astype(float).values,
            "method": method_name,
            "z": adata.obs["ground_truth"].astype(str).values,
            "z_pred": adata.obs["predict"].astype(str).values,
            "obs_name": adata.obs_names.astype(str),
            "spot_id": adata.obs["spot_id"].astype(str).values,
            "sample_id": adata.obs["sample_id"].astype(str).values,
            "ground_truth": adata.obs["ground_truth"].astype(str).values,
            "SpaConTDS_domain": adata.obs["predict"].astype(str).values,
            "image_enabled": bool(sample_info.get("image_enabled", False)),
            "alignment_mode": sample_info.get("alignment_mode"),
        }
    )
    return df


def maybe_compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    clean = df.loc[df["ground_truth"].astype(str) != "NA"].copy()
    if clean.empty:
        return {"n_labeled_spots": 0, "ari": None}
    return {
        "n_labeled_spots": int(clean.shape[0]),
        "ari": float(adjusted_rand_index(clean["z"].astype(str), clean["z_pred"].astype(str))),
    }


def run_sample(
    sample: dict[str, Any],
    project_root: Path,
    outdir: Path,
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method_root = project_root / "methods" / "SpaConTDS"
    params = pipeline_cfg.get("parameters", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outputs_cfg = pipeline_cfg.get("outputs", {})
    method_name = pipeline_cfg.get("method_name", "SpaConTDS")
    cohort_name = sample["cohort_name"]
    sample_id = str(sample["sample_id"])
    sample_dir = outdir / sample_id
    checkpoint_path = sample_dir / "checkpoint_predictions.csv"
    meta_path = sample_dir / "sample_metadata.json"
    success_marker = sample_dir / "_SUCCESS"

    if (
        bool(outputs_cfg.get("resume_from_checkpoints", True))
        and success_marker.exists()
        and checkpoint_path.exists()
    ):
        log_progress(f"[SpaConTDS] sample={sample_id} checkpoint found; skipping")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return pd.read_csv(checkpoint_path), meta

    patch_size = int(params.get("patch_size", 56))
    h5ad_path, image_path, sample_info = prepare_spacontds_input(
        sample=sample,
        sample_dir=sample_dir,
        patch_size=patch_size,
        overlay_outdir=outdir / "alignment_overlays",
    )
    adata_for_k = sc.read_h5ad(h5ad_path)
    k_value = params.get("k", params.get("n_domains", "auto"))
    n_domains = resolve_n_domains(adata_for_k, k_value)
    sample_info["n_domains"] = int(n_domains)

    args = make_spacontds_args(
        method_root=method_root,
        h5ad_path=h5ad_path,
        image_path=image_path,
        sample_id=sample_id,
        sample_dir=sample_dir,
        params=params,
        runtime_cfg=runtime_cfg,
        n_domains=n_domains,
    )

    set_seeds(int(runtime_cfg.get("seed", 42)))
    log_progress(
        f"[SpaConTDS] sample={sample_id} training spots={adata_for_k.n_obs} "
        f"genes={adata_for_k.n_vars} K={n_domains} epochs={args.num_epochs} device={args.device}"
    )
    workdir = prepare_spacontds_workdir(method_root, sample_dir)
    sample_info["spacontds_workdir"] = str(workdir)

    with pushd(workdir):
        spacon_main = load_spacontds_main(method_root)
        utils = importlib.import_module("utils")
        from torch.utils.data import DataLoader

        dataset = utils.TupleDataset(args)
        spacon_main.dataset = dataset
        dataloader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            collate_fn=lambda x: utils.collate_fn(x, dataset),
            shuffle=True,
        )
        spacon_main.train(args, dataloader)

    output_adata_path = Path(args.save_dir) / f"{sample_id}_adata.h5ad"
    if not output_adata_path.exists():
        raise FileNotFoundError(f"SpaConTDS output h5ad not found: {output_adata_path}")
    pred_adata = sc.read_h5ad(output_adata_path)
    predictions = build_predictions(pred_adata, cohort_name, method_name, sample_info)
    predictions.to_csv(checkpoint_path, index=False)

    sample_info.update(
        {
            "status": "success",
            "checkpoint_predictions": str(checkpoint_path),
            "output_adata_path": str(output_adata_path),
            "metrics": maybe_compute_metrics(predictions),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_json(meta_path, sample_info)
    success_marker.write_text("success\n", encoding="utf-8")

    if not bool(outputs_cfg.get("save_upstream_output", False)):
        reconstructed = Path(args.save_dir) / f"{sample_id}_reconstructed_matrix.csv"
        if reconstructed.exists():
            reconstructed.unlink()
    if not bool(outputs_cfg.get("save_prepared_inputs", True)):
        for path in [h5ad_path, image_path]:
            if path.exists():
                path.unlink()
    if not bool(outputs_cfg.get("save_model_snapshots", False)):
        for folder in [workdir / "model", workdir / "Decoder"]:
            for path in folder.glob("*.pth"):
                if path.name not in {
                    "baseline_weight.pth",
                    "Decoder_weight.pth",
                    "0.0031187241874542173.pth",
                }:
                    path.unlink(missing_ok=True)

    return predictions, sample_info


def write_partial_outputs(
    all_predictions: list[pd.DataFrame],
    outdir: Path,
    meta: dict[str, Any],
) -> None:
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            outdir / "predictions.partial.csv",
            index=False,
        )
    write_json(outdir / "run_metadata.partial.json", meta)


def run_spacontds(
    samples: list[dict[str, Any]],
    project_root: Path,
    pipeline_cfg: dict[str, Any],
    rscript_bin: str = "Rscript",
) -> dict[str, Any]:
    del rscript_bin
    if not samples:
        raise ValueError("Empty samples list passed to run_spacontds")

    method_name = pipeline_cfg.get("method_name", "SpaConTDS")
    cohort_name = samples[0]["cohort_name"]
    params = pipeline_cfg.get("parameters", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outdir = project_root / "results" / method_name / cohort_name
    safe_mkdir(outdir)

    stdout_path = outdir / "stdout.log"
    stderr_path = outdir / "stderr.log"
    meta: dict[str, Any] = {
        "method": method_name,
        "cohort": cohort_name,
        "status": "running",
        "runtime_level": runtime_cfg.get("level", "cohort"),
        "n_samples": len(samples),
        "parameters": params,
        "runtime": runtime_cfg,
        "samples": [],
        "runtime_seconds": None,
        "memory_mb": None,
        "error_message": None,
        "input_requirements": {
            "required_h5ad": True,
            "required_spatial_coordinates": True,
            "pathology_image_required_by_upstream": True,
            "blank_image_fallback_available": True,
        },
    }

    t0 = time.time()
    all_predictions: list[pd.DataFrame] = []
    with open(stdout_path, "a", encoding="utf-8", buffering=1) as stdout_f, open(
        stderr_path, "a", encoding="utf-8", buffering=1
    ) as stderr_f, redirect_stdout(stdout_f), redirect_stderr(stderr_f):
        try:
            log_progress(f"[SpaConTDS] cohort={cohort_name} start n_samples={len(samples)}")
            for sample in samples:
                predictions, sample_info = run_sample(
                    sample=sample,
                    project_root=project_root,
                    outdir=outdir,
                    pipeline_cfg=pipeline_cfg,
                )
                all_predictions.append(predictions)
                meta["samples"].append(sample_info)
                meta["completed_samples"] = len(meta["samples"])
                meta["completed_spots"] = int(sum(df.shape[0] for df in all_predictions))
                write_partial_outputs(all_predictions, outdir, meta)
                log_progress(
                    f"[SpaConTDS] cohort={cohort_name} checkpointed "
                    f"samples={meta['completed_samples']}/{len(samples)}"
                )

            predictions_df = pd.concat(all_predictions, ignore_index=True)
            write_predictions_tables(predictions_df, outdir)
            build_performance_df(cohort_name, method_name, None, None).to_csv(
                outdir / "performance.csv",
                index=False,
            )
            meta["status"] = "success"

        except Exception as exc:
            meta["status"] = "failed"
            meta["error_message"] = str(exc)
            meta["traceback"] = traceback.format_exc()
            write_partial_outputs(all_predictions, outdir, meta)
            log_progress(f"[SpaConTDS] cohort={cohort_name} failed error={exc}")
            traceback.print_exc()

    meta["runtime_seconds"] = round(time.time() - t0, 4)
    meta["memory_mb"] = round(get_memory_mb(), 3)

    if meta["status"] == "success":
        predictions_df = pd.read_csv(outdir / "predictions.csv")
        performance_df = build_performance_df(
            cohort_name=cohort_name,
            method_name=method_name,
            runtime_seconds=meta["runtime_seconds"],
            memory_mb=meta["memory_mb"],
        )
        write_performance_tables(performance_df, outdir)
        save_evaluation_summary(predictions_df, outdir, cohort_name, method_name)
        save_true_vs_pred_plots(predictions_df, outdir, cohort_name, method_name)

    write_json(outdir / "run_metadata.json", meta)
    return meta
