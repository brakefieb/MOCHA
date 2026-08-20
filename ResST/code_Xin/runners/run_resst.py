from __future__ import annotations

import json
import os
import random
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
from scipy.spatial import distance

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


def resolve_pca_components(n_obs: int, n_vars: int, requested: int) -> int:
    max_components = min(int(n_obs), int(n_vars))
    if max_components < 2:
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
        alignment_score = alignment.get("score")
        alignment_summary = alignment.get("summary")
        log_progress(
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


def harmonize_group_genes(
    adata_list: list[sc.AnnData],
) -> tuple[list[sc.AnnData], list[str]]:
    if not adata_list:
        return adata_list, []

    common_var_names = adata_list[0].var_names
    for adata in adata_list[1:]:
        common_var_names = common_var_names.intersection(adata.var_names)

    common_var_names = common_var_names.astype(str)
    if len(common_var_names) == 0:
        raise ValueError("No shared genes found across samples in this integration group.")

    harmonized = [adata[:, common_var_names].copy() for adata in adata_list]
    return harmonized, list(common_var_names)


def infer_platform(cohort_name: str) -> str:
    upper = cohort_name.upper()
    if upper.endswith("_ST") or "MOB_ST" in upper:
        return "ST"
    return "Visium"


def infer_refine_shape(cohort_name: str) -> str:
    upper = cohort_name.upper()
    if upper.endswith("_ST") or "MOB_ST" in upper:
        return "square"
    return "hexagon"


def preprocess_resst_sample(
    adata: sc.AnnData,
    info: dict[str, Any],
    sample_dir: Path,
    params: dict[str, Any],
    outputs_cfg: dict[str, Any],
    cohort_name: str,
    legacy_cache_dir: Path | None = None,
) -> tuple[sc.AnnData, dict[str, Any]]:
    from resst.get_adata import image_crop, image_feature
    from resst.preprocess import get_enhance_feature

    sample_dir.mkdir(parents=True, exist_ok=True)
    cache_path = sample_dir / "resst_enhanced.h5ad"
    cache_meta_path = sample_dir / "resst_enhanced_metadata.json"
    platform_cfg = str(params.get("platform", "auto"))
    platform = infer_platform(cohort_name) if platform_cfg == "auto" else platform_cfg

    if bool(outputs_cfg.get("resume_from_checkpoints", True)) and cache_path.exists():
        log_progress(f"[ResST] sample={info['sample_id']} loading cached enhanced data: {cache_path}")
        cached = sc.read_h5ad(cache_path)
        if cache_meta_path.exists():
            cached_meta = json.loads(cache_meta_path.read_text())
            info.update(cached_meta)
        info["sample_cache_hit"] = True
        info["sample_cache_path"] = str(cache_path)
        return cached, info

    if (
        bool(outputs_cfg.get("resume_from_checkpoints", True))
        and legacy_cache_dir is not None
        and (legacy_cache_dir / "resst_enhanced.h5ad").exists()
    ):
        legacy_cache_path = legacy_cache_dir / "resst_enhanced.h5ad"
        legacy_meta_path = legacy_cache_dir / "resst_enhanced_metadata.json"
        log_progress(
            f"[ResST] sample={info['sample_id']} loading legacy all_samples cache: "
            f"{legacy_cache_path}"
        )
        cached = sc.read_h5ad(legacy_cache_path)
        if legacy_meta_path.exists():
            cached_meta = json.loads(legacy_meta_path.read_text())
            info.update(cached_meta)
        info["sample_cache_hit"] = True
        info["sample_cache_path"] = str(legacy_cache_path)
        return cached, info

    morphology_used = False
    morphology_error = None
    weights_key = "weights_matrix_nomd"
    enhance_platform = "MERFISH"

    if bool(info["image_enabled"]) and "spatial" in adata.uns:
        try:
            log_progress(f"[ResST] sample={info['sample_id']} extracting image patches/features")
            crop_dir = sample_dir / "Image_crop"
            adata = image_crop(adata, save_path=crop_dir)
            image_pca = resolve_pca_components(
                adata.n_obs,
                1000,
                int(params.get("image_pca_n_comps", 200)),
            )
            adata = image_feature(
                adata,
                pca_components=image_pca,
                cnnType=str(params.get("cnn_type", "ResNet50")),
                seeds=int(params.get("image_seed", 88)),
            ).extract_image_feat()
            weights_key = "weights_matrix_all"
            enhance_platform = platform
            morphology_used = True
            info["image_pca_n_comps_used"] = image_pca
        except Exception as exc:
            morphology_error = str(exc)
            log_progress(
                f"[ResST] morphology path failed for {info['sample_id']}; "
                f"falling back to expression+spatial weights. Error: {morphology_error}"
            )

    log_progress(
        f"[ResST] sample={info['sample_id']} enhancing expression "
        f"weights={weights_key} platform={enhance_platform}"
    )
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata = get_enhance_feature(
        adata,
        adjacent_weight=float(params.get("adjacent_weight", 0.3)),
        neighbour_k=int(params.get("neighbour_k", 4)),
        weights=weights_key,
        spatial_k=int(params.get("spatial_k", 30)),
        platform=enhance_platform,
    )

    info.update(
        {
            "morphology_used": morphology_used,
            "morphology_error": morphology_error,
            "weights_key": weights_key,
            "enhance_platform": enhance_platform,
            "resst_platform": platform,
            "sample_cache_hit": False,
            "sample_cache_path": str(cache_path),
        }
    )

    if bool(outputs_cfg.get("save_sample_cache", True)):
        cache_adata = adata.copy()
        for key in [
            "adjacent_data",
            "weights_matrix_all",
            "weights_matrix_nomd",
            "image_feat",
            "image_feat_pca",
            "gene_correlation",
            "physical_distance",
            "morphological_similarity",
        ]:
            if key in cache_adata.obsm:
                del cache_adata.obsm[key]
        if "spatial" in cache_adata.uns:
            del cache_adata.uns["spatial"]
        if "slices_path" in cache_adata.obs.columns:
            cache_adata.obs = cache_adata.obs.drop(columns=["slices_path"])
        cache_adata.write_h5ad(cache_path)
        write_json(cache_meta_path, info)
        log_progress(f"[ResST] sample={info['sample_id']} cached enhanced data: {cache_path}")

    return adata, info


def build_predictions(
    adata: sc.AnnData,
    group_name: str,
    cohort_name: str,
    method_name: str,
    cluster_key: str,
    refined_key: str,
) -> pd.DataFrame:
    from spatial_alignment import extract_coords

    coords = extract_coords(adata)
    df = pd.DataFrame(
        {
            "cohort": cohort_name,
            "sampleID": adata.obs["sample_id"].astype(str).values,
            "spotID": adata.obs["spot_id"].astype(str).values,
            "x": coords["x"].astype(float).values,
            "y": coords["y"].astype(float).values,
            "method": method_name,
            "z": adata.obs["ground_truth"].astype(str).values,
            "z_pred": adata.obs[refined_key].astype(str).values,
            "obs_name": adata.obs_names.astype(str),
            "spot_id": adata.obs["spot_id"].astype(str).values,
            "sample_id": adata.obs["sample_id"].astype(str).values,
            "group_name": group_name,
            "ground_truth": adata.obs["ground_truth"].astype(str).values,
            "ResST_domain": adata.obs[cluster_key].astype(str).values,
            "ResST_refine_domain": adata.obs[refined_key].astype(str).values,
        }
    )
    return df


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


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_threads = os.environ.get("SLURM_CPUS_PER_TASK")
    if torch_threads:
        torch.set_num_threads(max(1, int(torch_threads)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def split_samples_for_integration(
    samples: list[dict[str, Any]],
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
    if mode == "sample":
        return {str(sample["sample_id"]): [sample] for sample in samples}

    if mode == "subgroup":
        groups: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            subgroup = sample.get("subgroup") or "ungrouped"
            groups.setdefault(str(subgroup), []).append(sample)
        return groups

    return {"all_samples": samples}


def run_group(
    group_name: str,
    group_samples: list[dict[str, Any]],
    project_root: Path,
    outdir: Path,
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    method_root = project_root / "methods" / "ResST"
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))

    import anndata
    from resst.get_adata import refine
    from resst.model_ST_utils import priori_cluster, trainer

    params = pipeline_cfg.get("parameters", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outputs_cfg = pipeline_cfg.get("outputs", {})
    cohort_name = group_samples[0]["cohort_name"]
    method_name = pipeline_cfg.get("method_name", "ResST")
    seed = int(runtime_cfg.get("seed", 0))
    set_seeds(seed)

    group_dir = outdir / group_name
    safe_mkdir(group_dir)
    checkpoint_file = str(outputs_cfg.get("checkpoint_predictions_file", "checkpoint_predictions.csv"))
    group_predictions_path = group_dir / checkpoint_file
    group_meta_path = group_dir / "group_metadata.json"
    success_marker = group_dir / "_SUCCESS"

    if (
        bool(outputs_cfg.get("resume_from_checkpoints", True))
        and success_marker.exists()
        and group_predictions_path.exists()
    ):
        log_progress(f"[ResST] group={group_name} checkpoint found; skipping completed group")
        predictions = pd.read_csv(group_predictions_path)
        sample_infos = []
        if group_meta_path.exists():
            group_meta = json.loads(group_meta_path.read_text())
            sample_infos = list(group_meta.get("samples", []))
        return predictions, sample_infos

    use_morphological_default = bool(params.get("use_morphology", True))
    if str(params.get("use_morphology", "true")).lower() == "auto":
        use_morphological_default = True

    prepared_adatas: list[sc.AnnData] = []
    sample_infos: list[dict[str, Any]] = []
    log_progress(
        f"[ResST] group={group_name} start n_samples={len(group_samples)} "
        f"sample_ids={[x['sample_id'] for x in group_samples]}"
    )
    for sample in group_samples:
        log_progress(f"[ResST] group={group_name} preparing sample={sample['sample_id']}")
        adata, info = prepare_adata(
            sample=sample,
            use_morphological_default=use_morphological_default,
            overlay_outdir=outdir / "alignment_overlays",
        )
        prepared_adatas.append(adata)
        sample_infos.append(info)

    prepared_adatas, common_genes = harmonize_group_genes(prepared_adatas)
    log_progress(f"[ResST] group={group_name} shared_genes={len(common_genes)}")
    for info in sample_infos:
        info["shared_gene_count"] = len(common_genes)

    enhanced_adatas: list[sc.AnnData] = []
    enhanced_infos: list[dict[str, Any]] = []
    for adata, info in zip(prepared_adatas, sample_infos):
        log_progress(f"[ResST] group={group_name} preprocessing sample={info['sample_id']}")
        legacy_cache_dir = None
        if group_name != "all_samples":
            legacy_cache_dir = outdir / "all_samples" / str(info["sample_id"])
        enhanced, enhanced_info = preprocess_resst_sample(
            adata=adata,
            info=info,
            sample_dir=group_dir / str(info["sample_id"]),
            params=params,
            outputs_cfg=outputs_cfg,
            cohort_name=cohort_name,
            legacy_cache_dir=legacy_cache_dir,
        )
        enhanced_adatas.append(enhanced)
        enhanced_infos.append(enhanced_info)

    if len(enhanced_adatas) == 1:
        merged_adata = enhanced_adatas[0].copy()
        domains = None
    else:
        merged_adata = sc.concat(
            enhanced_adatas,
            axis=0,
            join="inner",
            merge="same",
            uns_merge=None,
            label="batch_name",
            keys=[str(x["sample_id"]) for x in enhanced_infos],
            index_unique=None,
        )
        domains = np.asarray(
            pd.Categorical(
                merged_adata.obs["batch_name"].astype(str),
                categories=sorted(merged_adata.obs["batch_name"].astype(str).unique()),
            ).codes,
            dtype=np.int64,
        )

    n_domains = resolve_n_domains(merged_adata, params.get("n_domains"))
    data_name = str(group_name)
    pca_n_comps = resolve_pca_components(
        merged_adata.n_obs,
        merged_adata.n_vars,
        int(params.get("pca_n_comps", 50)),
    )

    log_progress(
        f"[ResST] group={group_name} training spots={merged_adata.n_obs} "
        f"genes={merged_adata.n_vars} n_domains={n_domains} pca={pca_n_comps}"
    )
    merged_adata = trainer(
        merged_adata,
        data_name=data_name,
        save_path=str(group_dir),
        domains=domains,
        pre_epochs=int(runtime_cfg.get("pre_epochs", 1000)),
        epochs=int(runtime_cfg.get("epochs", 500)),
        min_cells=int(params.get("min_cells", 3)),
        pca_n_comps=pca_n_comps,
        linear_encoder_hidden=list(params.get("linear_encoder_hidden", [32, 20])),
        linear_decoder_hidden=list(params.get("linear_decoder_hidden", [32, 60])),
        conv_hidden=list(params.get("conv_hidden", [32, 8])),
        p_drop=float(params.get("p_drop", 0.01)),
        dec_cluster_n=int(params.get("dec_cluster_n", 20)),
        lr=float(params.get("lr", 0.0005)),
        weight_decay=float(params.get("weight_decay", 0.0001)),
        grad_down=float(params.get("grad_down", 5)),
        store=bool(params.get("store_model", True)),
        use_model=bool(params.get("use_model", False)),
        graph_dist_type=str(params.get("graph_dist_type", "BallTree")),
        graph_k=int(params.get("graph_k", 12)),
        rad_cutoff=float(params.get("rad_cutoff", 150)),
    )

    cluster_adata = anndata.AnnData(merged_adata.obsm["embed"])
    cluster_adata.obs_names = merged_adata.obs_names
    sc.pp.neighbors(
        cluster_adata,
        n_neighbors=int(params.get("cluster_neighbors", 15)),
        random_state=seed,
    )
    log_progress(f"[ResST] group={group_name} clustering embeddings")
    resolution = priori_cluster(
        cluster_adata,
        n_domains=n_domains,
        cluster_type=str(params.get("cluster_type", "leiden")),
        increment=float(params.get("cluster_increment", 0.01)),
    )

    cluster_key = str(params.get("cluster_key", "pred"))
    refined_key = str(params.get("refined_key", "refine_pred"))
    cluster_type = str(params.get("cluster_type", "leiden"))
    if cluster_type == "louvain":
        try:
            import louvain  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ResST is configured with cluster_type='louvain', but the louvain "
                "package is not installed. Use cluster_type='leiden' (the MOCHA "
                "default) or install louvain in the environment."
            ) from exc
        sc.tl.louvain(cluster_adata, key_added=cluster_key, resolution=resolution, random_state=seed)
    else:
        sc.tl.leiden(cluster_adata, key_added=cluster_key, resolution=resolution, random_state=seed)
    merged_adata.obs[cluster_key] = cluster_adata.obs[cluster_key].astype(str).values

    refine_shape = str(params.get("refine_shape", "auto"))
    if refine_shape == "auto":
        refine_shape = infer_refine_shape(cohort_name)
    adj_2d = distance.cdist(merged_adata.obsm["spatial"], merged_adata.obsm["spatial"], "euclidean")
    merged_adata.obs[refined_key] = refine(
        sample_id=merged_adata.obs.index.tolist(),
        pred=merged_adata.obs[cluster_key].tolist(),
        dis=adj_2d,
        shape=refine_shape,
    )

    predictions = build_predictions(
        adata=merged_adata,
        group_name=group_name,
        cohort_name=cohort_name,
        method_name=method_name,
        cluster_key=cluster_key,
        refined_key=refined_key,
    )

    if bool(outputs_cfg.get("save_group_predictions", False)):
        predictions.to_csv(group_dir / "predictions.csv", index=False)
    predictions.to_csv(group_predictions_path, index=False)
    if bool(outputs_cfg.get("save_adata", False)):
        merged_adata.write_h5ad(group_dir / "adata_resst.h5ad")

    if not bool(outputs_cfg.get("save_model", False)):
        model_dir = group_dir / "Model"
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
    if not bool(outputs_cfg.get("save_intermediate_artifacts", False)):
        for sample_info in enhanced_infos:
            sample_dir = group_dir / str(sample_info["sample_id"])
            image_crop_dir = sample_dir / "Image_crop"
            if image_crop_dir.exists():
                shutil.rmtree(image_crop_dir, ignore_errors=True)
        if not bool(outputs_cfg.get("save_group_predictions", False)) and group_dir.exists():
            remaining = [p for p in group_dir.iterdir() if p.name not in {"Model"}]
            if not remaining:
                group_dir.rmdir()

    for info in enhanced_infos:
        info["n_domains"] = n_domains
        info["pca_n_comps_used"] = pca_n_comps
        info["refine_shape"] = refine_shape

    group_meta = {
        "group_name": group_name,
        "n_samples": len(group_samples),
        "sample_ids": [x["sample_id"] for x in group_samples],
        "n_spots": int(predictions.shape[0]),
        "n_domains": int(predictions["ResST_refine_domain"].nunique()),
        "metrics": maybe_compute_metrics(predictions),
        "samples": enhanced_infos,
        "checkpoint_predictions": str(group_predictions_path),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(group_meta_path, group_meta)
    success_marker.write_text("success\n", encoding="utf-8")
    log_progress(f"[ResST] group={group_name} complete checkpoint={group_predictions_path}")

    return predictions, enhanced_infos


def write_partial_outputs(
    all_predictions: list[pd.DataFrame],
    outdir: Path,
    meta: dict[str, Any],
) -> None:
    if all_predictions:
        partial_df = pd.concat(all_predictions, ignore_index=True)
        partial_df.to_csv(outdir / "predictions.partial.csv", index=False)
    write_json(outdir / "run_metadata.partial.json", meta)


def run_resst(
    samples: list[dict[str, Any]],
    project_root: Path,
    pipeline_cfg: dict[str, Any],
    rscript_bin: str = "Rscript",
) -> dict[str, Any]:
    del rscript_bin

    if not samples:
        raise ValueError("Empty samples list passed to run_resst")

    method_name = pipeline_cfg.get("method_name", "ResST")
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
        "integration_scope": runtime_cfg.get("integration_scope", "cohort"),
        "n_samples": len(samples),
        "parameters": params,
        "runtime": runtime_cfg,
        "groups": [],
        "runtime_seconds": None,
        "memory_mb": None,
        "error_message": None,
        "input_requirements": {
            "required_h5ad": True,
            "required_spatial_coordinates": True,
            "pathology_image_optional": True,
            "morphology_branch_uses_image": True,
            "expression_spatial_fallback_available": True,
            "current_h5ad_expectation": (
                "Current runner expects coordinates from obs[x/y], obs[imagecol/imagerow], "
                "or obsm[spatial/X_spatial/S]."
            ),
        },
    }

    t0 = time.time()
    all_predictions: list[pd.DataFrame] = []
    with open(stdout_path, "a", encoding="utf-8", buffering=1) as stdout_f, open(
        stderr_path, "a", encoding="utf-8", buffering=1
    ) as stderr_f, redirect_stdout(stdout_f), redirect_stderr(stderr_f):
        try:
            integration_scope = str(runtime_cfg.get("integration_scope", "cohort"))
            groups = split_samples_for_integration(samples, integration_scope)
            log_progress(
                f"[ResST] cohort={cohort_name} start groups={list(groups.keys())} "
                f"integration_scope={integration_scope}"
            )

            for group_name, group_samples in groups.items():
                log_progress(f"[ResST] cohort={cohort_name} running group={group_name}")
                group_predictions, sample_infos = run_group(
                    group_name=group_name,
                    group_samples=group_samples,
                    project_root=project_root,
                    outdir=outdir,
                    pipeline_cfg=pipeline_cfg,
                )
                group_record = {
                    "group_name": group_name,
                    "n_samples": len(group_samples),
                    "sample_ids": [x["sample_id"] for x in group_samples],
                    "n_spots": int(group_predictions.shape[0]),
                    "n_domains": int(group_predictions["ResST_refine_domain"].nunique()),
                    "metrics": maybe_compute_metrics(group_predictions),
                    "samples": sample_infos,
                }
                meta["groups"].append(group_record)
                all_predictions.append(group_predictions)
                meta["completed_groups"] = len(meta["groups"])
                meta["completed_spots"] = int(sum(df.shape[0] for df in all_predictions))
                write_partial_outputs(all_predictions, outdir, meta)
                log_progress(
                    f"[ResST] cohort={cohort_name} checkpointed "
                    f"groups={meta['completed_groups']}/{len(groups)}"
                )

            predictions_df = pd.concat(all_predictions, ignore_index=True)
            predictions_df = write_predictions_tables(predictions_df, outdir)

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
            write_partial_outputs(all_predictions, outdir, meta)
            log_progress(f"[ResST] cohort={cohort_name} failed error={exc}")

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
