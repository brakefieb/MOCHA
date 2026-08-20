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

from result_utils import (
    build_performance_df,
    save_evaluation_summary,
    save_true_vs_pred_plots,
    write_performance_tables,
    write_predictions_tables,
)
from spatial_alignment import estimate_spot_diameter, extract_coords, resolve_image_alignment


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


def infer_default_shape(cohort_name: str) -> str:
    upper = cohort_name.upper()
    if upper.endswith("_ST") or "SLIDE" in upper or "STEREO" in upper:
        return "square"
    return "hexagon"


def infer_n_domains(labels: pd.Series) -> int | None:
    clean = (
        labels.astype(str)
        .str.strip()
        .replace({"": "NA", "nan": "NA", "None": "NA", "NA": "NA"})
    )
    uniques = sorted(x for x in clean.unique() if x != "NA")
    return len(uniques) if uniques else None


def prepare_adata(
    sample: dict[str, Any],
    use_morphological_default: bool,
    overlay_outdir: Path,
) -> tuple[sc.AnnData, dict[str, Any]]:
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


def build_sample_predictions(
    adata: sc.AnnData,
    group_name: str,
    cohort_name: str,
    method_name: str,
) -> pd.DataFrame:
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
            "z_pred": adata.obs["DeepST_refine_domain"].astype(str).values,
            "obs_name": adata.obs_names.astype(str),
            "spot_id": adata.obs["spot_id"].astype(str).values,
            "sample_id": adata.obs["sample_id"].astype(str).values,
            "group_name": group_name,
            "ground_truth": adata.obs["ground_truth"].astype(str).values,
            "DeepST_domain": adata.obs["DeepST_domain"].astype(str).values,
            "DeepST_refine_domain": adata.obs["DeepST_refine_domain"].astype(str).values,
        }
    )
    return df


def maybe_compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    clean = df.loc[df["ground_truth"].astype(str) != "NA"].copy()
    if clean.empty:
        return {"n_labeled_spots": 0, "ari": None}

    truth = clean["z"].astype(str)
    pred = clean["z_pred"].astype(str)
    from result_utils import adjusted_rand_index

    return {
        "n_labeled_spots": int(clean.shape[0]),
        "ari": float(adjusted_rand_index(truth, pred)),
    }


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


def compact_adata_for_integration(adata: sc.AnnData) -> sc.AnnData:
    if "augment_gene_data" in adata.obsm:
        adata.X = np.asarray(adata.obsm["augment_gene_data"], dtype=np.float32)

    for key in ["augment_gene_data", "adjacent_data", "weights_matrix_all", "image_feat", "image_feat_pca"]:
        if key in adata.obsm:
            del adata.obsm[key]

    if "slices_path" in adata.obs.columns:
        adata.obs = adata.obs.drop(columns=["slices_path"])

    if "spatial" in adata.uns:
        del adata.uns["spatial"]

    return adata


def run_group(
    group_name: str,
    group_samples: list[dict[str, Any]],
    project_root: Path,
    outdir: Path,
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    method_root = project_root / "methods" / "DeepST"
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))
    import deepstkit as dt

    params = pipeline_cfg.get("parameters", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outputs_cfg = pipeline_cfg.get("outputs", {})
    cohort_name = group_samples[0]["cohort_name"]
    method_name = pipeline_cfg.get("method_name", "DeepST")

    use_morphological_default = bool(params.get("use_morphological", True))
    if str(params.get("use_morphological", "true")).lower() == "auto":
        use_morphological_default = True

    dt.utils_func.seed_torch(int(runtime_cfg.get("seed", 0)))

    group_dir = outdir / group_name
    safe_mkdir(group_dir)

    prepared_adatas: list[sc.AnnData] = []
    graph_list: list[dict[str, Any]] = []
    sample_infos: list[dict[str, Any]] = []

    task = "Integration" if len(group_samples) > 1 else "Identify_Domain"
    deepst = dt.main.run(
        save_path=str(group_dir),
        task=task,
        pre_epochs=int(runtime_cfg.get("pre_epochs", 500)),
        epochs=int(runtime_cfg.get("epochs", 500)),
        use_gpu=bool(runtime_cfg.get("use_gpu", True)),
    )

    for sample in group_samples:
        adata, info = prepare_adata(
            sample=sample,
            use_morphological_default=use_morphological_default,
            overlay_outdir=outdir / "alignment_overlays",
        )
        prepared_adatas.append(adata)
        sample_infos.append(info)

    common_genes: list[str] = []
    if len(prepared_adatas) > 1:
        prepared_adatas, common_genes = harmonize_group_genes(prepared_adatas)
        print(
            f"[integration] group={group_name} harmonized shared genes across "
            f"{len(prepared_adatas)} samples: {len(common_genes)}"
        )
        for info in sample_infos:
            info["shared_gene_count"] = len(common_genes)
    else:
        common_genes = prepared_adatas[0].var_names.astype(str).tolist()
        sample_infos[0]["shared_gene_count"] = len(common_genes)

    adata_list: list[sc.AnnData] = []
    for adata, info, sample in zip(prepared_adatas, sample_infos, group_samples):
        if bool(info["image_enabled"]):
            adata = deepst._get_image_crop(
                adata,
                data_name=str(sample["sample_id"]),
                cnn_type=str(params.get("cnn_type", "ResNet50")),
                pca_n_comps=int(params.get("image_pca_n_comps", 50)),
            )

        adata = deepst._get_augment(
            adata,
            adjacent_weight=float(params.get("adjacent_weight", 0.3)),
            neighbour_k=int(params.get("neighbour_k", 4)),
            spatial_k=int(params.get("spatial_k", 30)),
            n_components=int(params.get("augment_n_components", 100)),
            md_dist_type=str(params.get("md_dist_type", "cosine")),
            gb_dist_type=str(params.get("gb_dist_type", "correlation")),
            use_morphological=bool(info["image_enabled"]),
            use_data=str(params.get("use_data", "raw")),
            spatial_type=str(params.get("augment_spatial_type", "KDTree")),
        )

        graph_dict = deepst._get_graph(
            adata.obsm["spatial"],
            distType=str(params.get("graph_dist_type", "KDTree")),
            k=int(params.get("graph_k", 12)),
            rad_cutoff=float(params.get("rad_cutoff", 150)),
        )

        adata = compact_adata_for_integration(adata)

        adata_list.append(adata)
        graph_list.append(graph_dict)

    if len(adata_list) == 1:
        merged_adata = adata_list[0]
        merged_graph = graph_list[0]
    else:
        merged_adata, merged_graph = deepst._get_multiple_adata(
            adata_list=adata_list,
            data_name_list=[str(x["sample_id"]) for x in group_samples],
            graph_list=graph_list,
        )

    data = deepst._data_process(
        merged_adata,
        pca_n_comps=int(params.get("pca_n_comps", 200)),
    )

    fit_kwargs: dict[str, Any] = {
        "data": data,
        "graph_dict": merged_graph,
        "conv_type": str(params.get("conv_type", "GATConv")),
        "linear_encoder_hidden": list(params.get("linear_encoder_hidden", [32, 20])),
        "linear_decoder_hidden": list(params.get("linear_decoder_hidden", [32])),
        "conv_hidden": list(params.get("conv_hidden", [32, 8])),
        "p_drop": float(params.get("p_drop", 0.01)),
        "dec_cluster_n": int(params.get("dec_cluster_n", 20)),
        "kl_weight": float(params.get("kl_weight", 1.0)),
        "mse_weight": float(params.get("mse_weight", 1.0)),
        "bce_kld_weight": float(params.get("bce_kld_weight", 1.0)),
        "domain_weight": float(params.get("domain_weight", 1.0)),
    }
    if len(adata_list) > 1:
        fit_kwargs["domains"] = merged_adata.obs["batch"].values
        fit_kwargs["n_domains"] = int(merged_adata.obs["batch"].nunique())

    embeddings = deepst._fit(**fit_kwargs)
    merged_adata.obsm["DeepST_embed"] = embeddings

    requested_n_domains = params.get("n_domains")
    if requested_n_domains in (None, "auto"):
        n_domains = infer_n_domains(merged_adata.obs["ground_truth"])
    else:
        n_domains = int(requested_n_domains)

    if n_domains is None:
        raise ValueError(
            f"Could not infer n_domains for group {group_name}. "
            "Please set parameters.n_domains explicitly."
        )

    merged_adata = deepst._get_cluster_data(
        merged_adata,
        n_domains=n_domains,
        priori=bool(params.get("priori", True)),
        batch_key="batch_name" if len(adata_list) > 1 else None,
        shape=str(params.get("shape", infer_default_shape(group_samples[0]["cohort_name"]))),
    )

    predictions = build_sample_predictions(
        merged_adata,
        group_name=group_name,
        cohort_name=cohort_name,
        method_name=method_name,
    )
    save_group_predictions = bool(outputs_cfg.get("save_group_predictions", False))
    if save_group_predictions:
        predictions.to_csv(group_dir / "predictions.csv", index=False)

    if not bool(outputs_cfg.get("save_intermediate_artifacts", False)):
        for dirname in ["Image_crop", "Data"]:
            target = group_dir / dirname
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        if not save_group_predictions and group_dir.exists() and not any(group_dir.iterdir()):
            group_dir.rmdir()

    return predictions, sample_infos


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


def run_deepst(
    samples: list[dict[str, Any]],
    project_root: Path,
    pipeline_cfg: dict[str, Any],
    rscript_bin: str = "Rscript",
) -> dict[str, Any]:
    del rscript_bin

    if not samples:
        raise ValueError("Empty samples list passed to run_deepst")

    method_name = pipeline_cfg.get("method_name", "DeepST")
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
            "current_h5ad_expectation": (
                "Current runner expects coordinates from obs[x/y], obs[imagecol/imagerow], "
                "or obsm[spatial/X_spatial/S]."
            ),
        },
    }

    t0 = time.time()
    all_predictions: list[pd.DataFrame] = []
    with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_f, redirect_stdout(stdout_f), redirect_stderr(stderr_f):
        try:
            integration_scope = str(runtime_cfg.get("integration_scope", "cohort"))
            groups = split_samples_for_integration(samples, integration_scope)

            for group_name, group_samples in groups.items():
                group_predictions, sample_infos = run_group(
                    group_name=group_name,
                    group_samples=group_samples,
                    project_root=project_root,
                    outdir=outdir,
                    pipeline_cfg=pipeline_cfg,
                )
                group_metrics = maybe_compute_metrics(group_predictions)
                group_record = {
                    "group_name": group_name,
                    "n_samples": len(group_samples),
                    "sample_ids": [x["sample_id"] for x in group_samples],
                    "n_spots": int(group_predictions.shape[0]),
                    "n_domains": int(group_predictions["DeepST_refine_domain"].nunique()),
                    "metrics": group_metrics,
                    "samples": sample_infos,
                }
                meta["groups"].append(group_record)
                all_predictions.append(group_predictions)

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
