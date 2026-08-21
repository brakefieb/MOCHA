from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from result_utils import (
    adjusted_rand_index,
    build_performance_df,
    save_evaluation_summary,
    save_true_vs_pred_plots,
    write_performance_tables,
    write_predictions_tables,
)
from spatial_alignment import extract_coords, resolve_image_alignment


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def apply_label_mapping(
    labels: pd.Series,
    label_mapping: dict[str, Any],
    unmapped_default: str | None,
) -> pd.Series:
    if not label_mapping:
        return labels.astype(str)
    reverse: dict[str, str] = {}
    for target, sources in label_mapping.items():
        target = str(target).strip()
        reverse[target] = target
        if not isinstance(sources, list):
            sources = [sources]
        for source in sources:
            reverse[str(source).strip()] = target
    if unmapped_default is None:
        return labels.astype(str).str.strip().map(lambda x: reverse.get(x, x))
    return labels.astype(str).str.strip().map(
        lambda x: reverse.get(x, unmapped_default)
    )


def extract_ground_truth(sample: dict[str, Any], obs: pd.DataFrame) -> pd.Series:
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
    for column in candidates:
        if column in obs.columns:
            return apply_label_mapping(
                obs[column],
                sample.get("label_mapping", {}),
                sample.get("label_unmapped_default"),
            )
    return pd.Series("NA", index=obs.index, dtype="string")


def resolve_n_domains(adata: sc.AnnData, configured: Any) -> int:
    if configured not in (None, "auto"):
        return int(configured)
    labels = adata.obs["ground_truth"].astype(str).str.strip()
    labels = labels[~labels.isin(["", "NA", "nan", "None", "Filtered"])]
    if labels.empty:
        raise ValueError("Cannot infer K; set parameters.n_domains in the experiment config.")
    return int(labels.nunique())


def _patch_box(x: float, y: float, crop_size: int) -> tuple[int, int, int, int]:
    half = crop_size / 2.0
    return (
        int(round(x - half)),
        int(round(y - half)),
        int(round(x + half)),
        int(round(y + half)),
    )


def extract_hvit_features(
    image_path: Path,
    aligned_coords: pd.DataFrame,
    cache_dir: Path,
    crop_size: int,
    patch_size: int,
    pca_components: int,
    batch_size: int,
    seed: int,
    device: Any,
    save_raw_features: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract one H-ViT feature row per spot, preserving AnnData row order."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from stGCL.modules import extract_model
    from stGCL.process import set_seed

    cache_path = cache_dir / "hvit_features.npz"
    cache_meta_path = cache_dir / "hvit_features_metadata.json"
    spot_ids = aligned_coords.index.astype(str).tolist()
    signature = {
        "image_path": str(image_path.resolve()),
        "image_mtime_ns": image_path.stat().st_mtime_ns,
        "spot_ids": spot_ids,
        "coordinates": np.round(aligned_coords[["x", "y"]].to_numpy(), 4).tolist(),
        "crop_size": crop_size,
        "patch_size": patch_size,
        "pca_components_requested": pca_components,
        "seed": seed,
    }
    if cache_path.exists() and cache_meta_path.exists():
        cached_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if cached_meta.get("signature") == signature:
            cached = np.load(cache_path)
            features = cached["features"]
            if features.shape[0] != len(spot_ids):
                raise ValueError("H-ViT cache row count does not match spot count")
            cached_meta["cache_hit"] = True
            return features, cached_meta

    set_seed(seed)
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010],
            ),
        ]
    )

    class SpotPatchDataset(Dataset):
        def __len__(self) -> int:
            return len(aligned_coords)

        def __getitem__(self, index: int):
            row = aligned_coords.iloc[index]
            # PIL pads out-of-bounds areas with black. Explicit white padding better
            # matches a pathology slide background and keeps every spot in the batch.
            box = _patch_box(float(row["x"]), float(row["y"]), crop_size)
            patch = Image.new("RGB", (crop_size, crop_size), "white")
            clipped = (
                max(box[0], 0),
                max(box[1], 0),
                min(box[2], image.width),
                min(box[3], image.height),
            )
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                source = image.crop(clipped)
                patch.paste(source, (clipped[0] - box[0], clipped[1] - box[1]))
            return transform(patch), index

    loader = DataLoader(
        SpotPatchDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )
    model = extract_model("ViT", crop_size, patch_size, device=device).to(device)
    model.eval()
    raw_parts: list[np.ndarray] = []
    observed_order: list[int] = []
    with torch.inference_mode():
        for patches, indices in loader:
            patches = patches.to(device, non_blocking=True)
            raw_parts.append(model(patches).cpu().numpy())
            observed_order.extend(indices.numpy().tolist())
    raw = np.concatenate(raw_parts, axis=0).astype(np.float32, copy=False)
    if observed_order != list(range(len(spot_ids))):
        raise RuntimeError("H-ViT patch order no longer matches gene-expression row order")

    n_components = min(pca_components, raw.shape[0] - 1, raw.shape[1])
    if n_components < 1:
        raise ValueError("At least two spots are required for H-ViT PCA")
    features = PCA(n_components=n_components, random_state=seed).fit_transform(raw)
    features = features.astype(np.float32, copy=False)
    safe_mkdir(cache_dir)
    if save_raw_features:
        np.save(cache_dir / "hvit_raw_features.npy", raw)
    np.savez_compressed(cache_path, features=features)
    metadata = {
        "signature": signature,
        "cache_hit": False,
        "n_spots": len(spot_ids),
        "raw_feature_dim": int(raw.shape[1]),
        "pca_feature_dim": int(features.shape[1]),
        "spot_order_verified": True,
        "coordinate_convention": "x=column, y=row; patch=image[y, x]",
    }
    write_json(cache_meta_path, metadata)
    return features, metadata


def preprocess_adata(adata: sc.AnnData, params: dict[str, Any]) -> None:
    min_cells = int(params.get("min_cells", 3))
    sc.pp.filter_genes(adata, min_cells=min_cells)
    keep = np.asarray(
        [
            not (str(g).startswith("ERCC") or str(g).startswith("MT-") or str(g).startswith("mt-"))
            for g in adata.var_names
        ]
    )
    adata._inplace_subset_var(keep)
    top_genes = min(int(params.get("top_genes", 3000)), adata.n_vars)
    try:
        sc.pp.highly_variable_genes(
            adata,
            flavor=str(params.get("hvg_flavor", "seurat_v3")),
            n_top_genes=top_genes,
        )
    except ImportError:
        sc.pp.highly_variable_genes(
            adata,
            flavor="cell_ranger",
            n_top_genes=top_genes,
        )
    sc.pp.normalize_total(adata, target_sum=float(params.get("target_sum", 1e4)))
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=float(params.get("scale_max_value", 10)))


def refine_labels(coords: np.ndarray, labels: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_neighbors = min(max(1, int(n_neighbors)), len(labels))
    indices = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(coords)
        .kneighbors(coords, return_distance=False)
    )
    refined: list[str] = []
    labels = labels.astype(str)
    for row in indices:
        values, counts = np.unique(labels[row], return_counts=True)
        refined.append(str(values[np.argmax(counts)]))
    return np.asarray(refined)


def build_predictions(
    adata: sc.AnnData,
    cohort_name: str,
    method_name: str,
    image_info: dict[str, Any],
) -> pd.DataFrame:
    coords = extract_coords(adata)
    return pd.DataFrame(
        {
            "cohort": cohort_name,
            "sampleID": adata.obs["sample_id"].astype(str).values,
            "spotID": adata.obs["spot_id"].astype(str).values,
            "x": coords["x"].astype(float).values,
            "y": coords["y"].astype(float).values,
            "method": method_name,
            "z": adata.obs["ground_truth"].astype(str).values,
            "z_pred": adata.obs["stGCL_refined"].astype(str).values,
            "obs_name": adata.obs_names.astype(str),
            "spot_id": adata.obs["spot_id"].astype(str).values,
            "sample_id": adata.obs["sample_id"].astype(str).values,
            "ground_truth": adata.obs["ground_truth"].astype(str).values,
            "stGCL_domain": adata.obs["stGCL_domain"].astype(str).values,
            "stGCL_refined": adata.obs["stGCL_refined"].astype(str).values,
            "image_used": bool(image_info.get("image_used", False)),
            "image_path": image_info.get("image_path"),
            "alignment_mode": image_info.get("alignment_mode"),
        }
    )


def run_one_sample(
    sample: dict[str, Any],
    project_root: Path,
    outdir: Path,
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch

    method_root = project_root / "methods" / "stGCL"
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))
    import stGCL

    params = pipeline_cfg.get("parameters", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outputs_cfg = pipeline_cfg.get("outputs", {})
    sample_id = str(sample["sample_id"])
    sample_dir = outdir / sample_id
    prediction_cache = sample_dir / "predictions.csv"
    metadata_cache = sample_dir / "sample_metadata.json"
    h5ad_path = Path(sample["h5ad_path"])
    image_path = Path(sample["image_path"]) if sample.get("image_path") else None
    checkpoint_signature = {
        "h5ad_path": str(h5ad_path.resolve()),
        "h5ad_mtime_ns": h5ad_path.stat().st_mtime_ns,
        "image_path": str(image_path.resolve()) if image_path is not None else None,
        "image_mtime_ns": image_path.stat().st_mtime_ns if image_path is not None else None,
        "parameters": params,
        "seed": int(runtime_cfg.get("seed", 0)),
    }
    if (
        bool(outputs_cfg.get("resume_from_checkpoints", True))
        and prediction_cache.exists()
        and metadata_cache.exists()
    ):
        cached_meta = json.loads(metadata_cache.read_text(encoding="utf-8"))
        if cached_meta.get("checkpoint_signature") == checkpoint_signature:
            print(f"[stGCL] sample={sample_id} checkpoint hit", flush=True)
            return pd.read_csv(prediction_cache), cached_meta
        print(f"[stGCL] sample={sample_id} stale checkpoint ignored", flush=True)

    safe_mkdir(sample_dir)
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs["spot_id"] = adata.obs_names.astype(str)
    adata.obs["sample_id"] = sample_id
    adata.obs["ground_truth"] = extract_ground_truth(sample, adata.obs).astype(str).values
    base_coords = extract_coords(adata)
    adata.obsm["spatial"] = base_coords[["x", "y"]].to_numpy(dtype=float)
    n_domains = resolve_n_domains(adata, params.get("n_domains"))

    use_image_cfg = params.get("use_image", "auto")
    image_requested = str(use_image_cfg).lower() != "false" and bool(use_image_cfg)
    image_info: dict[str, Any] = {
        "image_requested": image_requested,
        "image_used": False,
        "image_path": sample.get("image_path"),
        "alignment_mode": "disabled",
        "alignment_note": "Image branch disabled by configuration.",
        "alignment_overlay_path": None,
        "spot_image_match_verified": False,
        "image_error": None,
    }
    if image_requested:
        alignment = resolve_image_alignment(
            sample,
            adata,
            overlay_outdir=outdir / "alignment_overlays",
        )
        image_info.update(
            {
                "alignment_mode": alignment["mode"],
                "alignment_note": alignment["note"],
                "alignment_overlay_path": alignment["overlay_path"],
                "alignment_score": alignment.get("score"),
                "alignment_summary": alignment.get("summary"),
            }
        )
        if alignment["image_enabled"]:
            try:
                aligned = alignment["aligned_coords"].copy()
                aligned.index = aligned.index.astype(str)
                aligned = aligned.reindex(adata.obs_names.astype(str))
                if aligned[["x", "y"]].isna().any().any():
                    missing = aligned.index[aligned[["x", "y"]].isna().any(axis=1)].tolist()
                    raise ValueError(
                        f"Aligned image coordinates missing for {len(missing)} spots; examples={missing[:5]}"
                    )
                device_name = str(runtime_cfg.get("device", "auto"))
                if device_name == "auto":
                    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
                device = torch.device(device_name)
                features, feature_meta = extract_hvit_features(
                    image_path=Path(str(alignment["image_path"])),
                    aligned_coords=aligned,
                    cache_dir=sample_dir / "image_features",
                    crop_size=int(params.get("image_crop_size", 256)),
                    patch_size=int(params.get("vit_patch_size", 64)),
                    pca_components=int(params.get("image_pca_n_comps", 50)),
                    batch_size=int(runtime_cfg.get("image_batch_size", 64)),
                    seed=int(runtime_cfg.get("seed", 0)),
                    device=device,
                    save_raw_features=bool(outputs_cfg.get("save_raw_image_features", False)),
                )
                if features.shape[0] != adata.n_obs:
                    raise RuntimeError("Image feature count does not match gene-expression spot count")
                adata.obsm["im_re"] = features
                image_info.update(
                    {
                        "image_used": True,
                        "spot_image_match_verified": True,
                        "feature_metadata": feature_meta,
                    }
                )
            except Exception as exc:
                image_info["image_error"] = str(exc)
                if not bool(params.get("fallback_to_expression_only", True)):
                    raise
                print(f"[stGCL] image branch failed for {sample_id}; fallback enabled: {exc}", flush=True)

    preprocess_adata(adata, params)
    graph_model = str(params.get("graph_model", "KNN"))
    if graph_model == "Radius":
        stGCL.Cal_Spatial_Net(
            adata,
            rad_cutoff=float(params.get("rad_cutoff", 150)),
            model="Radius",
        )
    else:
        stGCL.Cal_Spatial_Net(
            adata,
            k_cutoff=min(int(params.get("graph_k", 6)), adata.n_obs - 1),
            model="KNN",
        )

    epoch_cfg = params.get("n_epochs", "auto")
    n_epochs = (1200 if adata.n_obs > 3600 else 1000) if epoch_cfg == "auto" else int(epoch_cfg)
    device_name = str(runtime_cfg.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    adata = stGCL.train(
        adata,
        knn=n_domains,
        hidden_dims=[int(x) for x in params.get("hidden_dims", [100, 30])],
        n_epochs=n_epochs,
        alph=float(params.get("contrastive_weight", 0.04)),
        lr=float(params.get("learning_rate", 0.001)),
        gradient_clipping=float(params.get("gradient_clipping", 5.0)),
        weight_decay=float(params.get("weight_decay", 0.0001)),
        use_image=bool(image_info["image_used"]),
        random_seed=int(runtime_cfg.get("seed", 0)),
        loadmin=False,
        early_stop=bool(params.get("early_stop", True)),
        device=device,
    )

    embedding = np.asarray(adata.obsm["stGCL"], dtype=np.float64)
    clusterer = GaussianMixture(
        n_components=n_domains,
        covariance_type="tied",  # mclust EEE: shared volume, shape and orientation
        n_init=int(params.get("gmm_n_init", 10)),
        reg_covar=float(params.get("gmm_reg_covar", 1e-6)),
        random_state=int(runtime_cfg.get("seed", 0)),
    )
    labels = clusterer.fit_predict(embedding).astype(str)
    adata.obs["stGCL_domain"] = labels
    adata.obs["stGCL_refined"] = refine_labels(
        np.asarray(adata.obsm["spatial"], dtype=float),
        labels,
        int(params.get("refine_neighbors", 50)),
    )

    predictions = build_predictions(
        adata,
        cohort_name=sample["cohort_name"],
        method_name=pipeline_cfg.get("method_name", "stGCL"),
        image_info=image_info,
    )
    valid = predictions.loc[~predictions["z"].isin(["NA", "Filtered", "nan"])]
    sample_meta = {
        "sample_id": sample_id,
        "h5ad_path": sample["h5ad_path"],
        "checkpoint_signature": checkpoint_signature,
        "n_spots": int(adata.n_obs),
        "n_genes_after_filter": int(adata.n_vars),
        "n_domains": n_domains,
        "n_epochs": n_epochs,
        "device": str(device),
        "graph_model": graph_model,
        "image": image_info,
        "cluster_model": "GaussianMixture(covariance_type=tied), equivalent to mclust EEE",
        "ground_truth_not_used_for_training_or_cluster_label_matching": True,
        "ari": (
            float(adjusted_rand_index(valid["z"], valid["z_pred"]))
            if not valid.empty
            else None
        ),
    }
    predictions.to_csv(prediction_cache, index=False)
    write_json(metadata_cache, sample_meta)
    if not bool(outputs_cfg.get("save_embedding", False)):
        adata.obsm.pop("stGCL", None)
    else:
        np.save(sample_dir / "stgcl_embedding.npy", embedding)
    return predictions, sample_meta


def run_stgcl(
    samples: list[dict[str, Any]],
    project_root: Path,
    pipeline_cfg: dict[str, Any],
    rscript_bin: str = "Rscript",
) -> dict[str, Any]:
    del rscript_bin
    if not samples:
        raise ValueError("Empty samples list passed to run_stgcl")

    method_name = pipeline_cfg.get("method_name", "stGCL")
    cohort_name = samples[0]["cohort_name"]
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outdir = project_root / "results" / method_name / cohort_name
    safe_mkdir(outdir)
    meta: dict[str, Any] = {
        "method": method_name,
        "cohort": cohort_name,
        "status": "running",
        "runtime_level": runtime_cfg.get("level", "cohort"),
        "n_samples": len(samples),
        "parameters": pipeline_cfg.get("parameters", {}),
        "runtime": runtime_cfg,
        "samples": [],
        "runtime_seconds": None,
        "memory_mb": None,
        "error_message": None,
        "input_requirements": {
            "required_h5ad": True,
            "required_spatial_coordinates": True,
            "pathology_image_optional": True,
            "hvit_branch_uses_image": True,
            "hvit_branch_requires_spot_image_matching": True,
            "expression_spatial_fallback_available": bool(
                pipeline_cfg.get("parameters", {}).get("fallback_to_expression_only", True)
            ),
        },
    }

    t0 = time.time()
    stdout_path = outdir / "stdout.log"
    stderr_path = outdir / "stderr.log"
    with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_f, redirect_stdout(stdout_f), redirect_stderr(stderr_f):
        try:
            parts: list[pd.DataFrame] = []
            for index, sample in enumerate(samples, start=1):
                print(
                    f"[stGCL] cohort={cohort_name} sample={index}/{len(samples)} id={sample['sample_id']}",
                    flush=True,
                )
                prediction, sample_meta = run_one_sample(
                    sample, project_root, outdir, pipeline_cfg
                )
                parts.append(prediction)
                meta["samples"].append(sample_meta)
            predictions = pd.concat(parts, ignore_index=True)
            write_predictions_tables(predictions, outdir)
            meta["status"] = "success"
        except Exception as exc:
            meta["status"] = "failed"
            meta["error_message"] = f"{type(exc).__name__}: {exc}"
            import traceback

            traceback.print_exc()

    meta["runtime_seconds"] = round(time.time() - t0, 4)
    meta["memory_mb"] = round(get_memory_mb(), 3)
    if meta["status"] == "success":
        predictions = pd.read_csv(outdir / "predictions.csv")
        performance = build_performance_df(
            cohort_name, method_name, meta["runtime_seconds"], meta["memory_mb"]
        )
        write_performance_tables(performance, outdir)
        save_evaluation_summary(predictions, outdir, cohort_name, method_name)
        save_true_vs_pred_plots(predictions, outdir, cohort_name, method_name)
    write_json(outdir / "run_metadata.json", meta)
    return meta
