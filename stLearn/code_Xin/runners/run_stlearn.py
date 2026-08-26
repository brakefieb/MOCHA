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
    uniques = sorted(x for x in clean.unique() if x != "NA")
    return len(uniques) if uniques else None


def resolve_n_domains(adata: sc.AnnData, configured_value: Any) -> int:
    if configured_value not in (None, "auto"):
        return int(configured_value)
    inferred = infer_n_domains(adata.obs["ground_truth"])
    if inferred is None:
        raise ValueError(
            "Could not infer n_domains from ground-truth labels. "
            "Please set parameters.n_domains explicitly."
        )
    return inferred


def resolve_pca_components(adata: sc.AnnData, requested: int) -> int:
    max_components = min(int(adata.n_obs), int(adata.n_vars))
    if max_components < 1:
        raise ValueError(
            f"Cannot run PCA with adata shape {adata.n_obs} x {adata.n_vars}."
        )
    if max_components == 1:
        return 1
    return max(1, min(int(requested), max_components - 1))


def ensure_array_grid_columns(adata: sc.AnnData) -> None:
    from spatial_alignment import extract_coords

    if "array_row" in adata.obs.columns and "array_col" in adata.obs.columns:
        return

    coords = extract_coords(adata)
    adata.obs["array_col"] = coords["x"].astype(float).values
    adata.obs["array_row"] = coords["y"].astype(float).values


def prepare_adata(
    sample: dict[str, Any],
    use_morphological_default: bool,
    overlay_outdir: Path,
) -> tuple[sc.AnnData, dict[str, Any]]:
    import scanpy as sc
    from spatial_alignment import (
        estimate_spot_diameter,
        extract_coords,
        resolve_image_alignment,
    )

    h5ad_path = Path(sample["h5ad_path"])
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs["spot_id"] = adata.obs_names.astype(str)
    adata.obs["sample_id"] = str(sample["sample_id"])

    gt = extract_ground_truth(
        adata.obs,
        sample.get("label_mapping", {}),
        unmapped_default=sample.get("label_unmapped_default"),
    )
    adata.obs["ground_truth"] = gt.astype(str).values

    coords = extract_coords(adata)
    adata.obsm["spatial"] = coords[["x", "y"]].to_numpy(dtype=float)
    ensure_array_grid_columns(adata)

    image_enabled = False
    image_note = "Morphology branch disabled by configuration."
    alignment_mode = "disabled"
    alignment_overlay_path = None
    alignment_score = None
    alignment_summary = None
    if use_morphological_default:
        alignment = resolve_image_alignment(
            sample=sample,
            adata=adata,
            overlay_outdir=overlay_outdir,
        )
        aligned_coords = alignment["aligned_coords"]
        adata.obs["imagecol"] = aligned_coords["x"].astype(float).values
        adata.obs["imagerow"] = aligned_coords["y"].astype(float).values
        if alignment["image_array"] is not None:
            adata.uns["spatial"] = {
                str(sample["sample_id"]): {
                    "images": {"hires": alignment["image_array"]},
                    "use_quality": "hires",
                    "scalefactors": {
                        "tissue_hires_scalef": 1.0,
                        "spot_diameter_fullres": estimate_spot_diameter(aligned_coords),
                    },
                }
            }
        image_enabled = bool(alignment["image_enabled"])
        image_note = str(alignment["note"])
        alignment_mode = str(alignment["mode"])
        alignment_overlay_path = alignment["overlay_path"]
        alignment_score = alignment["score"]
        alignment_summary = alignment["summary"]
        print(
            f"[alignment] sample={sample['sample_id']} "
            f"mode={alignment_mode} image_enabled={image_enabled} "
            f"note={image_note} overlay={alignment_overlay_path}"
        )

    adata.obs_names = [f"{sample['sample_id']}::{spot_id}" for spot_id in adata.obs["spot_id"]]

    info = {
        "sample_id": sample["sample_id"],
        "subgroup": sample.get("subgroup"),
        "h5ad_path": str(h5ad_path),
        "image_path": sample.get("image_path"),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "image_enabled": bool(image_enabled),
        "image_note": image_note,
        "alignment_mode": alignment_mode,
        "alignment_overlay_path": alignment_overlay_path,
        "alignment_score": alignment_score,
        "alignment_summary": alignment_summary,
    }
    return adata, info


def maybe_compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    clean = df.loc[df["ground_truth"].astype(str) != "NA"].copy()
    if clean.empty:
        return {"n_labeled_spots": 0, "ari": None}

    truth = clean["z"].astype(str)
    pred = clean["z_pred"].astype(str)
    return {
        "n_labeled_spots": int(clean.shape[0]),
        "ari": float(adjusted_rand_index(truth, pred)),
    }


def build_sample_predictions(
    adata: sc.AnnData,
    cohort_name: str,
    method_name: str,
    cluster_key: str,
    sample_info: dict[str, Any],
) -> pd.DataFrame:
    from spatial_alignment import extract_coords

    coords = extract_coords(adata)
    morphology_used = bool(sample_info.get("morphology_used", False))
    df = pd.DataFrame(
        {
            "cohort": cohort_name,
            "sampleID": adata.obs["sample_id"].astype(str).values,
            "spotID": adata.obs["spot_id"].astype(str).values,
            "x": coords["x"].astype(float).values,
            "y": coords["y"].astype(float).values,
            "method": method_name,
            "z": adata.obs["ground_truth"].astype(str).values,
            "z_pred": adata.obs[cluster_key].astype(str).values,
            "obs_name": adata.obs_names.astype(str),
            "spot_id": adata.obs["spot_id"].astype(str).values,
            "sample_id": adata.obs["sample_id"].astype(str).values,
            "ground_truth": adata.obs["ground_truth"].astype(str).values,
            "stlearn_cluster": adata.obs[cluster_key].astype(str).values,
            "cluster_key": cluster_key,
            "morphology_used": morphology_used,
            "image_available": bool(sample_info.get("image_enabled", False)),
            "image_path": sample_info.get("image_path"),
            "alignment_mode": sample_info.get("alignment_mode"),
        }
    )
    return df


def run_one_sample(
    sample: dict[str, Any],
    project_root: Path,
    outdir: Path,
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method_root = project_root / "methods" / "stLearn"
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))
    import stlearn as st

    params = pipeline_cfg.get("parameters", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outputs_cfg = pipeline_cfg.get("outputs", {})
    cohort_name = sample["cohort_name"]
    method_name = pipeline_cfg.get("method_name", "stLearn")

    sample_dir = outdir / str(sample["sample_id"])
    safe_mkdir(sample_dir)

    use_morphological_default = bool(params.get("use_morphology", True))
    if str(params.get("use_morphology", "true")).lower() == "auto":
        use_morphological_default = True

    adata, sample_info = prepare_adata(
        sample=sample,
        use_morphological_default=use_morphological_default,
        overlay_outdir=outdir / "alignment_overlays",
    )

    cluster_key = str(params.get("cluster_key", "stlearn_kmeans"))
    seed = int(runtime_cfg.get("seed", 1))
    n_domains = resolve_n_domains(adata, params.get("n_domains"))
    sample_info["n_domains"] = n_domains

    work = adata.copy()
    st.pp.filter_genes(work, min_cells=int(params.get("filter_genes_min_cells", 1)))
    st.pp.normalize_total(work)
    st.pp.log1p(work)

    pre_pca_n_comps = resolve_pca_components(work, int(params.get("pre_pca_n_comps", 50)))
    st.em.run_pca(work, n_comps=pre_pca_n_comps, random_state=seed)

    morphology_requested = bool(use_morphological_default)
    morphology_available = bool(sample_info["image_enabled"])
    morphology_used = False
    morphology_error = None
    spot_image_match_required = False

    if morphology_requested and morphology_available:
        try:
            tiles_dir = sample_dir / "tiles"
            st.pp.tiling(
                work,
                tiles_dir,
                crop_size=int(params.get("tiling_crop_size", 40)),
                target_size=int(params.get("tiling_target_size", 299)),
                img_fmt=str(params.get("tiling_image_format", "JPEG")),
                quality=int(params.get("tiling_jpeg_quality", 75)),
            )
            st.pp.extract_feature(
                work,
                cnn_base=str(params.get("cnn_base", "resnet50")),
                n_components=int(params.get("morphology_n_components", 50)),
                seeds=seed,
            )
            st.spatial.sme.sme_normalize(
                work,
                use_data=str(params.get("sme_use_data", "raw")),
                weights=str(params.get("sme_weights", "weights_matrix_all")),
                platform=str(params.get("sme_platform", "Visium")),
            )
            sme_key = f"{params.get('sme_use_data', 'raw')}_SME_normalized"
            work.X = np.asarray(work.obsm[sme_key], dtype=np.float32)
            morphology_used = True
            spot_image_match_required = True
        except Exception as exc:
            morphology_error = str(exc)
            if not bool(params.get("fallback_to_expression_only", True)):
                raise
            print(
                f"[stLearn] morphology path failed for {sample['sample_id']}; "
                f"falling back to expression-only clustering. Error: {morphology_error}"
            )

    st.pp.scale(work)
    post_pca_n_comps = resolve_pca_components(work, int(params.get("post_pca_n_comps", 50)))
    st.em.run_pca(work, n_comps=post_pca_n_comps, random_state=seed)
    st.tl.clustering.kmeans(
        work,
        n_clusters=n_domains,
        use_data=str(params.get("kmeans_use_data", "X_pca")),
        random_state=seed,
        key_added=cluster_key,
    )

    predictions = build_sample_predictions(
        adata=work,
        cohort_name=cohort_name,
        method_name=method_name,
        cluster_key=cluster_key,
        sample_info={
            **sample_info,
            "morphology_used": morphology_used,
        },
    )

    if bool(outputs_cfg.get("save_sample_predictions", False)):
        predictions.to_csv(sample_dir / "predictions.csv", index=False)

    if not bool(outputs_cfg.get("save_intermediate_artifacts", False)):
        tiles_dir = sample_dir / "tiles"
        if tiles_dir.exists():
            shutil.rmtree(tiles_dir, ignore_errors=True)
        if sample_dir.exists() and not any(sample_dir.iterdir()):
            sample_dir.rmdir()

    sample_info.update(
        {
            "cluster_key": cluster_key,
            "morphology_requested": morphology_requested,
            "morphology_available": morphology_available,
            "morphology_used": morphology_used,
            "morphology_error": morphology_error,
            "spot_image_match_required": spot_image_match_required,
            "pre_pca_n_comps": pre_pca_n_comps,
            "post_pca_n_comps": post_pca_n_comps,
            "sme_use_data": str(params.get("sme_use_data", "raw")),
            "sme_weights": str(params.get("sme_weights", "weights_matrix_all")),
            "sme_platform": str(params.get("sme_platform", "Visium")),
            "cnn_base": str(params.get("cnn_base", "resnet50")),
            "metrics": maybe_compute_metrics(predictions),
        }
    )
    return predictions, sample_info


def run_stlearn(
    samples: list[dict[str, Any]],
    project_root: Path,
    pipeline_cfg: dict[str, Any],
    rscript_bin: str = "Rscript",
) -> dict[str, Any]:
    del rscript_bin

    if not samples:
        raise ValueError("Empty samples list passed to run_stlearn")

    method_name = pipeline_cfg.get("method_name", "stLearn")
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
            "pathology_image_optional": True,
            "morphology_branch_uses_image": True,
            "morphology_branch_requires_spot_image_matching": True,
            "expression_only_fallback_available": bool(
                params.get("fallback_to_expression_only", True)
            ),
            "current_h5ad_expectation": (
                "Current runner expects coordinates from obs[x/y], obs[imagecol/imagerow], "
                "or obsm[spatial/X_spatial/S]."
            ),
        },
    }

    t0 = time.time()
    with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_f, redirect_stdout(stdout_f), redirect_stderr(stderr_f):
        try:
            predictions_parts: list[pd.DataFrame] = []
            for sample in samples:
                sample_predictions, sample_info = run_one_sample(
                    sample=sample,
                    project_root=project_root,
                    outdir=outdir,
                    pipeline_cfg=pipeline_cfg,
                )
                meta["samples"].append(sample_info)
                predictions_parts.append(sample_predictions)

            predictions_df = pd.concat(predictions_parts, ignore_index=True)
            write_predictions_tables(predictions_df, outdir)

            performance_df = build_performance_df(
                cohort_name=cohort_name,
                method_name=method_name,
                runtime_seconds=None,
                memory_mb=None,
            )
            performance_df.to_csv(outdir / "performance.csv", index=False)
            meta["status"] = "success"
        except Exception as exc:
            meta["status"] = "failed"
            meta["error_message"] = str(exc)

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

    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
