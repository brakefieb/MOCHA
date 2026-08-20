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
from scipy import sparse
from sklearn.cluster import KMeans

from result_utils import (
    adjusted_rand_index,
    build_performance_df,
    save_evaluation_summary,
    save_true_vs_pred_plots,
    write_performance_tables,
    write_predictions_tables,
)
from spatial_alignment import estimate_spot_diameter, extract_coords, resolve_image_alignment


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def log_progress(message: str) -> None:
    print(message, flush=True)


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


def select_group_shard(
    groups: dict[str, list[dict[str, Any]]],
    shard_index: int,
    shard_total: int,
) -> dict[str, list[dict[str, Any]]]:
    if shard_total <= 0:
        raise ValueError("STARFYSH_GROUP_SHARD_TOTAL must be positive")
    if shard_index < 0 or shard_index >= shard_total:
        raise ValueError(
            "STARFYSH_GROUP_SHARD_INDEX must satisfy "
            f"0 <= index < total, got {shard_index}/{shard_total}"
        )
    selected = {
        group_name: group_samples
        for idx, (group_name, group_samples) in enumerate(sorted(groups.items()))
        if idx % shard_total == shard_index
    }
    if not selected:
        raise ValueError(
            f"Shard {shard_index}/{shard_total} selected no groups from {len(groups)} groups"
        )
    return selected


def dense_float32(adata: sc.AnnData) -> sc.AnnData:
    out = adata.copy()
    X = out.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[X < 0] = 0.0
    out.X = X
    return out


def set_count_matrix(adata: sc.AnnData) -> sc.AnnData:
    """
    Starfysh expects raw, finite, nonnegative counts. Several benchmark h5ad
    files store transformed data in X, so prefer explicit count-like layers.
    """
    out = adata.copy()
    count_layer_candidates = [
        "counts",
        "count",
        "raw_counts",
        "raw_count",
        "spliced",
    ]
    selected = None
    for key in count_layer_candidates:
        if key in out.layers:
            selected = out.layers[key]
            out.uns["starfysh_count_source"] = f"layers/{key}"
            break
    if selected is None and out.raw is not None:
        selected = out.raw.X
        out.uns["starfysh_count_source"] = "raw/X"
    if selected is None:
        selected = out.X
        out.uns["starfysh_count_source"] = "X"

    if sparse.issparse(selected):
        selected = selected.toarray()
    X = np.asarray(selected, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[X < 0] = 0.0
    out.X = X
    return out


def drop_empty_counts(adata: sc.AnnData) -> sc.AnnData:
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    keep_cells = np.asarray(X.sum(axis=1)).ravel() > 0
    keep_genes = np.asarray(X.sum(axis=0)).ravel() > 0
    if not keep_cells.all() or not keep_genes.all():
        adata = adata[keep_cells, keep_genes].copy()
    return dense_float32(adata)


def assert_finite_nonnegative_counts(adata: sc.AnnData, label: str) -> None:
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if not np.isfinite(X).all():
        raise ValueError(f"{label} contains non-finite values after sanitization.")
    if (X < 0).any():
        raise ValueError(f"{label} contains negative values after sanitization.")


def resolve_use_image(params: dict[str, Any]) -> bool:
    value = str(params.get("use_image", "auto")).lower()
    if value in {"false", "0", "no", "off"}:
        return False
    return True


def prepare_one_sample(
    sample: dict[str, Any],
    use_image: bool,
    overlay_outdir: Path,
) -> tuple[sc.AnnData, dict[str, Any]]:
    h5ad_path = Path(sample["h5ad_path"])
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()
    adata = set_count_matrix(adata)
    adata = drop_empty_counts(adata)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs["spot_id"] = adata.obs_names.astype(str)
    adata.obs["sample_id"] = str(sample["sample_id"])
    adata.obs["sample"] = str(sample["sample_id"])

    gt = extract_ground_truth(
        adata.obs,
        sample.get("label_mapping", {}),
        unmapped_default=sample.get("label_unmapped_default"),
    )
    adata.obs["ground_truth"] = gt.astype(str).values

    coords = extract_coords(adata)
    adata.obsm["spatial"] = coords[["x", "y"]].to_numpy(dtype=float)
    adata.obs["array_col"] = coords["x"].astype(float).values
    adata.obs["array_row"] = coords["y"].astype(float).values
    adata.obs["imagecol"] = coords["x"].astype(float).values
    adata.obs["imagerow"] = coords["y"].astype(float).values

    image_enabled = False
    image_note = "PoE image branch disabled by configuration."
    alignment_mode = "disabled"
    alignment_overlay_path = None
    alignment_score = None
    alignment_summary = None
    image_array = None
    scalefactor = {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": estimate_spot_diameter(coords)}

    if use_image:
        alignment = resolve_image_alignment(
            sample=sample,
            adata=adata,
            overlay_outdir=overlay_outdir,
        )
        aligned_coords = alignment["aligned_coords"]
        adata.obs["imagecol"] = aligned_coords["x"].astype(float).values
        adata.obs["imagerow"] = aligned_coords["y"].astype(float).values
        image_enabled = bool(alignment["image_enabled"])
        image_note = str(alignment["note"])
        alignment_mode = str(alignment["mode"])
        alignment_overlay_path = alignment["overlay_path"]
        alignment_score = alignment["score"]
        alignment_summary = alignment["summary"]
        image_array = alignment["image_array"] if image_enabled else None
        scalefactor = {
            "tissue_hires_scalef": 1.0,
            "spot_diameter_fullres": estimate_spot_diameter(aligned_coords),
        }
        print(
            f"[alignment] sample={sample['sample_id']} "
            f"mode={alignment_mode} image_enabled={image_enabled} "
            f"note={image_note} overlay={alignment_overlay_path}"
        )

    adata.obs_names = [
        f"{sample['sample_id']}::{spot_id}" for spot_id in adata.obs["spot_id"].astype(str)
    ]

    info = {
        "sample_id": sample["sample_id"],
        "subgroup": sample.get("subgroup"),
        "h5ad_path": str(h5ad_path),
        "image_path": sample.get("image_path"),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "count_source": str(adata.uns.get("starfysh_count_source", "unknown")),
        "image_enabled": bool(image_enabled),
        "image_note": image_note,
        "image_array": image_array,
        "scalefactor": scalefactor,
        "alignment_mode": alignment_mode,
        "alignment_overlay_path": alignment_overlay_path,
        "alignment_score": alignment_score,
        "alignment_summary": alignment_summary,
    }
    return dense_float32(adata), info


def harmonize_group_genes(adatas: list[sc.AnnData]) -> tuple[list[sc.AnnData], list[str]]:
    if not adatas:
        return adatas, []
    common = adatas[0].var_names
    for adata in adatas[1:]:
        common = common.intersection(adata.var_names)
    common = common.astype(str)
    if len(common) == 0:
        raise ValueError("No shared genes found across samples in this Starfysh group.")
    return [dense_float32(adata[:, common].copy()) for adata in adatas], list(common)


def concat_group_adatas(adatas: list[sc.AnnData]) -> sc.AnnData:
    if len(adatas) == 1:
        return adatas[0].copy()
    return dense_float32(
        sc.concat(
            adatas,
            join="inner",
            merge="same",
            uns_merge="first",
            index_unique=None,
        )
    )


def build_sample_img_metadata(
    adata: sc.AnnData,
    sample_info: dict[str, Any],
    use_real_image: bool,
) -> dict[str, Any]:
    obs = adata.obs
    image_array = sample_info.get("image_array") if use_real_image else None
    if image_array is None:
        image_array = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    scalefactor = sample_info.get("scalefactor") or {
        "tissue_hires_scalef": 1.0,
        "spot_diameter_fullres": estimate_spot_diameter(
            pd.DataFrame(
                {
                    "x": obs["imagecol"].astype(float).values,
                    "y": obs["imagerow"].astype(float).values,
                },
                index=adata.obs_names,
            )
        ),
    }
    map_info = pd.DataFrame(
        {
            "array_row": obs["array_row"].astype(float).values,
            "array_col": obs["array_col"].astype(float).values,
            "imagerow": obs["imagerow"].astype(float).values,
            "imagecol": obs["imagecol"].astype(float).values,
            "sample": adata.obs["sample_id"].astype(str).values,
        },
        index=adata.obs_names,
    )
    return {
        "img": image_array,
        "map_info": map_info,
        "scalefactor": scalefactor,
    }


def marker_signatures_from_kmeans(
    adata_norm: sc.AnnData,
    n_factors: int,
    n_markers: int,
    seed: int,
) -> pd.DataFrame:
    work = adata_norm.copy()
    max_pcs = max(1, min(30, work.n_obs - 1, work.n_vars - 1))
    if "X_pca" not in work.obsm and max_pcs > 1:
        sc.pp.pca(work, n_comps=max_pcs, random_state=seed)
        X = work.obsm["X_pca"]
    else:
        X = work.X.toarray() if sparse.issparse(work.X) else np.asarray(work.X)

    n_clusters = max(1, min(int(n_factors), work.n_obs))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(X)
    work.obs["_starfysh_init_cluster"] = pd.Categorical(labels.astype(str))
    sc.tl.rank_genes_groups(
        work,
        "_starfysh_init_cluster",
        use_raw=False,
        method="wilcoxon",
    )
    names = work.uns["rank_genes_groups"]["names"]
    signatures: dict[str, list[str]] = {}
    for cluster in work.obs["_starfysh_init_cluster"].cat.categories:
        signatures[f"arch_{cluster}"] = [str(g) for g in names[cluster][:n_markers]]
    return pd.DataFrame(signatures)


def build_reference_free_signatures(
    adata_norm: sc.AnnData,
    params: dict[str, Any],
    seed: int,
    method_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    strategy = str(params.get("signature_strategy", "archetypes"))
    n_domains = int(params.get("n_domains_resolved"))
    n_markers = int(params.get("n_markers", 30))
    meta: dict[str, Any] = {"signature_strategy_requested": strategy}

    if strategy == "archetypes":
        try:
            # py_pcha still calls np.mat, which was removed in NumPy 2.0.
            # Keep Starfysh's AA path usable in newer shared HPC envs.
            if not hasattr(np, "mat"):
                np.mat = np.asmatrix  # type: ignore[attr-defined]
            from starfysh import AA

            r = int(params.get("archetype_r", 100))
            r = max(2, min(r, max(2, adata_norm.n_obs - 1)))
            aa_model = AA.ArchetypalAnalysis(
                adata_orig=adata_norm,
                r=r,
                verbose=bool(params.get("verbose", True)),
            )
            _, arche_dict, major_idx, evs = aa_model.compute_archetypes(
                cn=int(params.get("archetype_cn", 30)),
                n_iters=int(params.get("archetype_n_iters", 20)),
                converge=float(params.get("archetype_converge", 1e-3)),
                display=False,
            )
            aa_model.find_archetypal_spots(major=True)
            gene_sig = aa_model.find_markers(n_markers=n_markers, display=False)
            meta.update(
                {
                    "signature_strategy_used": "archetypes",
                    "n_signature_factors": int(gene_sig.shape[1]),
                    "archetype_r": r,
                    "archetype_major_idx": [int(x) for x in major_idx],
                    "archetype_dict": {
                        str(k): [int(x) for x in v] for k, v in arche_dict.items()
                    },
                    "archetype_explained_variance": [float(x) for x in evs],
                }
            )
            return gene_sig, meta
        except Exception as exc:
            if not bool(params.get("fallback_to_marker_signatures", True)):
                raise
            meta["archetype_error"] = str(exc)
            print(
                "[Starfysh] Archetypal signature discovery failed; "
                f"falling back to KMeans marker signatures. Error: {exc}"
            )

    gene_sig = marker_signatures_from_kmeans(
        adata_norm=adata_norm,
        n_factors=n_domains,
        n_markers=n_markers,
        seed=seed,
    )
    meta.update(
        {
            "signature_strategy_used": "kmeans_markers",
            "n_signature_factors": int(gene_sig.shape[1]),
        }
    )
    return gene_sig, meta


def build_predictions(
    adata: sc.AnnData,
    cohort_name: str,
    method_name: str,
    group_name: str,
    cluster_key: str,
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
            "z_pred": adata.obs[cluster_key].astype(str).values,
            "obs_name": adata.obs_names.astype(str),
            "spot_id": adata.obs["spot_id"].astype(str).values,
            "sample_id": adata.obs["sample_id"].astype(str).values,
            "group_name": group_name,
            "ground_truth": adata.obs["ground_truth"].astype(str).values,
            "starfysh_cluster": adata.obs[cluster_key].astype(str).values,
            "cluster_key": cluster_key,
        }
    )


def maybe_compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    clean = df.loc[df["ground_truth"].astype(str) != "NA"].copy()
    if clean.empty:
        return {"n_labeled_spots": 0, "ari": None}
    return {
        "n_labeled_spots": int(clean.shape[0]),
        "ari": float(adjusted_rand_index(clean["z"].astype(str), clean["z_pred"].astype(str))),
    }


def strip_nonserializable_sample_info(info: dict[str, Any]) -> dict[str, Any]:
    clean = dict(info)
    clean.pop("image_array", None)
    return clean


def ensure_histomicstk_importable() -> None:
    try:
        import histomicstk  # noqa: F401
    except Exception:
        import types

        sys.modules.setdefault("histomicstk", types.ModuleType("histomicstk"))


def patch_starfysh_train_poe(sf_model: Any, utils: Any, utils_integrate: Any) -> None:
    import torch
    import torch.nn as nn

    def train_poe_safe(model, dataloader, device, optimizer):
        model.train()

        running_loss = 0.0
        running_z = 0.0
        running_c = 0.0
        running_l = 0.0
        running_u = 0.0
        running_reconst = 0.0
        counter = 0
        valid_counter = 0
        corr_list = []

        for x, x_peri, library_i, img, data_loc, xs_k in dataloader:
            del x_peri, data_loc
            counter += 1
            mini_batch, _ = x.shape

            x = x.float().to(device)
            library_i = library_i.to(device)
            xs_k = xs_k.to(device)

            img = img.reshape(mini_batch, -1).float().to(device)
            if torch.max(img) > 1:
                img = img / 255.0
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            img = torch.clamp(img, min=0.0, max=1.0)

            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("NaNs detected in Starfysh PoE model parameters")

            inference_outputs = model.inference(x, img)
            generative_outputs = model.generative(inference_outputs, xs_k)
            img_outputs = model.predictor_img(img)
            poe_outputs = model.predictor_poe(inference_outputs, img_outputs)

            result = model.get_loss(
                generative_outputs,
                inference_outputs,
                img_outputs,
                poe_outputs,
                x,
                library_i,
                img,
                device,
            )
            loss, reconst_loss, kl_u, kl_z, kl_c, kl_l = result
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Starfysh PoE parameters became non-finite after optimizer step")

            valid_counter += 1
            running_loss += loss.item()
            running_reconst += reconst_loss.item()
            running_z += kl_z.item()
            running_c += kl_c.item()
            running_l += kl_l.item()
            running_u += kl_u.item()

        denom = max(valid_counter, 1)
        if valid_counter == 0:
            raise FloatingPointError("All Starfysh PoE mini-batches had non-finite losses")

        return (
            running_loss / denom,
            running_reconst / denom,
            running_u / denom,
            running_z / denom,
            running_c / denom,
            running_l / denom,
            corr_list,
        )

    sf_model.train_poe = train_poe_safe
    utils.train_poe = train_poe_safe
    utils_integrate.train_poe = train_poe_safe


def run_group(
    group_name: str,
    group_samples: list[dict[str, Any]],
    project_root: Path,
    outdir: Path,
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method_root = project_root / "methods" / "starfysh"
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))

    ensure_histomicstk_importable()

    import torch
    from starfysh import utils
    from starfysh import utils_integrate
    from starfysh import starfysh as sf_model
    patch_starfysh_train_poe(sf_model, utils, utils_integrate)

    params = pipeline_cfg.get("parameters", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    outputs_cfg = pipeline_cfg.get("outputs", {})
    cohort_name = group_samples[0]["cohort_name"]
    method_name = pipeline_cfg.get("method_name", "Starfysh")
    seed = int(runtime_cfg.get("seed", 0))

    group_dir = outdir / group_name
    safe_mkdir(group_dir)
    group_t0 = time.time()

    use_image_requested = resolve_use_image(params)
    prepared: list[sc.AnnData] = []
    sample_infos: list[dict[str, Any]] = []
    for sample in group_samples:
        adata, info = prepare_one_sample(
            sample=sample,
            use_image=use_image_requested,
            overlay_outdir=outdir / "alignment_overlays",
        )
        prepared.append(adata)
        sample_infos.append(info)

    use_poe = use_image_requested and all(bool(info["image_enabled"]) for info in sample_infos)
    if use_image_requested and not use_poe:
        disabled = [
            f"{info['sample_id']}({info.get('alignment_mode')}: {info.get('image_note')})"
            for info in sample_infos
            if not bool(info["image_enabled"])
        ]
        log_progress(
            "[Starfysh] PoE image branch disabled for group "
            f"{group_name}; unmatched/missing images: {disabled}"
        )

    prepared, common_genes = harmonize_group_genes(prepared)
    raw_adata = concat_group_adatas(prepared)
    raw_adata = drop_empty_counts(raw_adata)
    raw_adata.obs["ground_truth"] = raw_adata.obs["ground_truth"].astype(str)
    raw_adata.obs["sample_id"] = raw_adata.obs["sample_id"].astype(str)
    raw_adata.obs["sample"] = raw_adata.obs["sample_id"].astype(str)
    assert_finite_nonnegative_counts(raw_adata, f"{group_name} raw counts")

    n_domains = resolve_n_domains(raw_adata, params.get("n_domains"))
    params = {**params, "n_domains_resolved": n_domains}

    adata, adata_norm = utils.preprocess(
        raw_adata,
        n_top_genes=int(params.get("n_top_genes", 2000)),
        mt_thld=float(params.get("mt_thld", 100)),
        verbose=bool(params.get("verbose", True)),
        multiple_data=len(group_samples) > 1,
    )
    adata = drop_empty_counts(adata)
    adata_norm = dense_float32(adata_norm[adata.obs_names, adata.var_names].copy())
    assert_finite_nonnegative_counts(adata, f"{group_name} Starfysh preprocessed counts")
    carry_obs_cols = [
        "spot_id",
        "sample_id",
        "sample",
        "ground_truth",
        "array_col",
        "array_row",
        "imagecol",
        "imagerow",
    ]
    for col in carry_obs_cols:
        if col in raw_adata.obs.columns:
            adata.obs[col] = raw_adata.obs.loc[adata.obs_names, col].values
    adata.obs["spot_id"] = adata.obs["spot_id"].astype(str)
    adata.obs["sample_id"] = adata.obs["sample_id"].astype(str)
    adata.obs["sample"] = adata.obs["sample_id"].astype(str).values
    adata.obs["ground_truth"] = adata.obs["ground_truth"].astype(str)
    adata_norm.obs = adata.obs.copy()
    adata.obsm["spatial"] = raw_adata.obsm["spatial"][
        raw_adata.obs_names.get_indexer(adata.obs_names)
    ]
    adata_norm.obsm["spatial"] = adata.obsm["spatial"].copy()

    gene_sig, signature_meta = build_reference_free_signatures(
        adata_norm=adata_norm,
        params=params,
        seed=seed,
        method_root=method_root,
    )
    gene_sig_path = group_dir / "gene_signatures.csv"
    if bool(outputs_cfg.get("save_gene_signatures", True)):
        gene_sig.to_csv(gene_sig_path)

    sample_info_by_id = {str(info["sample_id"]): info for info in sample_infos}
    img_metadata_by_sample: dict[str, dict[str, Any]] = {}
    for sample_id, sample_adata in adata.obs.groupby("sample_id", sort=False):
        mask = adata.obs["sample_id"].astype(str) == str(sample_id)
        img_metadata_by_sample[str(sample_id)] = build_sample_img_metadata(
            adata[mask].copy(),
            sample_info_by_id[str(sample_id)],
            use_real_image=use_poe,
        )

    n_anchors_param = params.get("n_anchors")
    n_anchors = adata.n_obs if n_anchors_param in (None, "auto") else int(n_anchors_param)
    if len(group_samples) > 1:
        individual_args = {}
        for sample_id in sorted(adata.obs["sample_id"].astype(str).unique()):
            mask = (adata.obs["sample_id"].astype(str) == sample_id).to_numpy()
            individual_args[sample_id] = utils.VisiumArguments(
                adata[mask].copy(),
                adata_norm[mask].copy(),
                gene_sig,
                img_metadata_by_sample[sample_id],
                n_anchors=min(n_anchors, int(mask.sum())),
                window_size=int(params.get("window_size", 1)),
                signif_level=float(params.get("signif_level", 3)),
                sample_id=sample_id,
            )

        visium_args = utils_integrate.VisiumArguments_integrate(
            adata,
            adata_norm,
            gene_sig,
            img_metadata_by_sample,
            individual_args,
            n_anchors=n_anchors,
            window_size=int(params.get("window_size", 1)),
            signif_level=float(params.get("signif_level", 3)),
            sample_id=pd.Series(sorted(adata.obs["sample_id"].astype(str).unique())),
        )
        run_func = utils_integrate.run_starfysh
        eval_func = sf_model.model_eval_integrate
    else:
        only_sample_id = str(adata.obs["sample_id"].astype(str).iloc[0])
        visium_args = utils.VisiumArguments(
            adata,
            adata_norm,
            gene_sig,
            img_metadata_by_sample[only_sample_id],
            n_anchors=n_anchors,
            window_size=int(params.get("window_size", 1)),
            signif_level=float(params.get("signif_level", 3)),
            sample_id=only_sample_id,
        )
        run_func = utils.run_starfysh
        eval_func = sf_model.model_eval
    adata, adata_norm = visium_args.get_adata()
    adata = dense_float32(adata)

    use_gpu = bool(runtime_cfg.get("use_gpu", torch.cuda.is_available()))
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    train_lr = float(runtime_cfg.get("lr", 1e-4))
    if use_poe:
        train_lr = float(runtime_cfg.get("poe_lr", params.get("poe_lr", train_lr * 0.2)))

    model, loss = run_func(
        visium_args,
        n_repeats=int(runtime_cfg.get("n_repeats", 3)),
        lr=train_lr,
        epochs=int(runtime_cfg.get("epochs", 100)),
        batch_size=int(runtime_cfg.get("batch_size", 32)),
        alpha_mul=float(params.get("alpha_mul", 50)),
        poe=use_poe,
        device=device,
        verbose=bool(params.get("verbose", True)),
    )

    _, _, adata_starfysh = eval_func(
        model,
        adata,
        visium_args,
        poe=use_poe,
        device=device,
    )

    qc_m = np.asarray(adata_starfysh.obsm["qc_m"], dtype=np.float32)
    if qc_m.ndim == 1:
        qc_m = qc_m.reshape(-1, 1)
    cluster_key = str(params.get("cluster_key", "starfysh_kmeans"))
    labels = KMeans(
        n_clusters=n_domains,
        random_state=seed,
        n_init=int(params.get("kmeans_n_init", 20)),
    ).fit_predict(qc_m)
    adata_starfysh.obs[cluster_key] = labels.astype(str)

    predictions = build_predictions(
        adata=adata_starfysh,
        cohort_name=cohort_name,
        method_name=method_name,
        group_name=group_name,
        cluster_key=cluster_key,
    )

    # Always checkpoint group predictions. This makes long shard jobs resumable
    # under Juno's 48h wall-time limit.
    write_csv_atomic(predictions, group_dir / "predictions.csv")

    if bool(outputs_cfg.get("save_latent", False)):
        pd.DataFrame(qc_m, index=adata_starfysh.obs_names).to_csv(group_dir / "qc_m.csv")
        if "qz_m" in adata_starfysh.obsm:
            pd.DataFrame(adata_starfysh.obsm["qz_m"], index=adata_starfysh.obs_names).to_csv(
                group_dir / "qz_m.csv"
            )
    if bool(outputs_cfg.get("save_loss", True)):
        pd.DataFrame(loss).to_csv(group_dir / "loss.csv", index=False)

    group_record = {
        "group_name": group_name,
        "n_samples": len(group_samples),
        "sample_ids": [x["sample_id"] for x in group_samples],
        "n_spots": int(predictions.shape[0]),
        "n_domains": int(n_domains),
        "n_signature_factors": int(signature_meta.get("n_signature_factors", gene_sig.shape[1])),
        "shared_gene_count": int(len(common_genes)),
        "device": str(device),
        "use_image_requested": bool(use_image_requested),
        "poe_used": bool(use_poe),
        "metrics": maybe_compute_metrics(predictions),
        "signature_metadata": signature_meta,
        "samples": [strip_nonserializable_sample_info(info) for info in sample_infos],
        "runtime_seconds": round(time.time() - group_t0, 4),
    }

    if not bool(outputs_cfg.get("save_intermediate_artifacts", False)):
        if not bool(outputs_cfg.get("save_gene_signatures", True)) and gene_sig_path.exists():
            gene_sig_path.unlink()
        if not bool(outputs_cfg.get("save_group_predictions", False)) and group_dir.exists():
            remaining = [p for p in group_dir.iterdir() if p.name != "gene_signatures.csv"]
            if not remaining and not bool(outputs_cfg.get("save_gene_signatures", True)):
                group_dir.rmdir()

    return predictions, group_record


def run_starfysh(
    samples: list[dict[str, Any]],
    project_root: Path,
    pipeline_cfg: dict[str, Any],
    rscript_bin: str = "Rscript",
) -> dict[str, Any]:
    del rscript_bin

    if not samples:
        raise ValueError("Empty samples list passed to run_starfysh")

    method_name = pipeline_cfg.get("method_name", "Starfysh")
    cohort_name = samples[0]["cohort_name"]
    params = pipeline_cfg.get("parameters", {})
    runtime_cfg = pipeline_cfg.get("runtime", {})
    integration_scope = str(runtime_cfg.get("integration_scope", "cohort"))
    only_group = os.environ.get("STARFYSH_ONLY_GROUP")
    shard_index_env = os.environ.get("STARFYSH_GROUP_SHARD_INDEX")
    shard_total_env = os.environ.get("STARFYSH_GROUP_SHARD_TOTAL")
    shard_label = None
    if shard_index_env is not None or shard_total_env is not None:
        if shard_index_env is None or shard_total_env is None:
            raise ValueError(
                "Set both STARFYSH_GROUP_SHARD_INDEX and STARFYSH_GROUP_SHARD_TOTAL"
            )
        shard_index = int(shard_index_env)
        shard_total = int(shard_total_env)
        shard_label = f"shard_{shard_index}_of_{shard_total}"
    else:
        shard_index = None
        shard_total = None

    if only_group and shard_label:
        raise ValueError("Use either STARFYSH_ONLY_GROUP or STARFYSH_GROUP_SHARD_*, not both")

    outdir = project_root / "results" / method_name / cohort_name
    if only_group:
        outdir = outdir / "_partials" / only_group
    elif shard_label:
        outdir = outdir / "_partials" / shard_label
    safe_mkdir(outdir)

    stdout_path = outdir / "stdout.log"
    stderr_path = outdir / "stderr.log"

    meta: dict[str, Any] = {
        "method": method_name,
        "cohort": cohort_name,
        "status": "running",
        "runtime_level": runtime_cfg.get("level", "cohort"),
        "integration_scope": integration_scope,
        "only_group": only_group,
        "shard": {
            "index": shard_index,
            "total": shard_total,
            "label": shard_label,
        } if shard_label else None,
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
            "poe_image_branch_uses_image": True,
            "poe_image_branch_requires_spot_image_matching": True,
            "current_h5ad_expectation": (
                "Current runner expects coordinates from obs[x/y], obs[imagecol/imagerow], "
                "or obsm[spatial/X_spatial/S]. Starfysh uses matched pathology image "
                "patches with PoE when configured images align successfully."
            ),
        },
    }

    t0 = time.time()
    predictions_df: pd.DataFrame | None = None
    with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_f, redirect_stdout(stdout_f), redirect_stderr(stderr_f):
        try:
            predictions_parts: list[pd.DataFrame] = []
            groups = split_samples_for_integration(samples, integration_scope)
            if only_group:
                if only_group not in groups:
                    raise ValueError(
                        f"STARFYSH_ONLY_GROUP={only_group!r} not found. "
                        f"Available groups: {sorted(groups)}"
                    )
                groups = {only_group: groups[only_group]}
            elif shard_label:
                groups = select_group_shard(groups, int(shard_index), int(shard_total))

            group_items = list(groups.items())
            log_progress(
                "[Starfysh] Selected "
                f"{len(group_items)} group(s): {', '.join(name for name, _ in group_items)}"
            )
            for group_idx, (group_name, group_samples) in enumerate(group_items, start=1):
                group_pred_path = outdir / group_name / "predictions.csv"
                if bool(runtime_cfg.get("resume_completed_groups", True)) and group_pred_path.exists():
                    log_progress(
                        f"[Starfysh] SKIP group {group_idx}/{len(group_items)} "
                        f"{group_name}: checkpoint exists at {group_pred_path}"
                    )
                    group_predictions = pd.read_csv(group_pred_path)
                    group_record = {
                        "group_name": group_name,
                        "n_samples": len(group_samples),
                        "sample_ids": [x["sample_id"] for x in group_samples],
                        "n_spots": int(group_predictions.shape[0]),
                        "resumed_from": str(group_pred_path),
                        "metrics": maybe_compute_metrics(group_predictions),
                    }
                    meta["groups"].append(group_record)
                    predictions_parts.append(group_predictions)
                    continue

                log_progress(
                    f"[Starfysh] START group {group_idx}/{len(group_items)} "
                    f"{group_name}: samples={[x['sample_id'] for x in group_samples]}"
                )
                group_t0 = time.time()
                group_predictions, group_record = run_group(
                    group_name=group_name,
                    group_samples=group_samples,
                    project_root=project_root,
                    outdir=outdir,
                    pipeline_cfg=pipeline_cfg,
                )
                log_progress(
                    f"[Starfysh] DONE group {group_idx}/{len(group_items)} "
                    f"{group_name}: spots={group_predictions.shape[0]}, "
                    f"seconds={time.time() - group_t0:.1f}"
                )
                meta["groups"].append(group_record)
                predictions_parts.append(group_predictions)

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
        if predictions_df is None:
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
