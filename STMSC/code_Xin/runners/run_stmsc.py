from __future__ import annotations

import fcntl
import json
import os
import random
import resource
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import distance
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from result_utils import (
    adjusted_rand_index,
    build_performance_df,
    save_evaluation_summary,
    save_true_vs_pred_plots,
    write_performance_tables,
    write_predictions_tables,
)


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024.0


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    torch.set_num_threads(max(1, n_threads))


def extract_ground_truth(sample: dict[str, Any], adata: sc.AnnData) -> pd.Series:
    candidates = [
        "annotation", "Classification", "classification", "ground_truth",
        "layer_guess", "spatialLIBD", "manual_label", "label", "region", "z",
    ]
    raw = None
    for column in candidates:
        if column in adata.obs:
            raw = adata.obs[column].astype(str).str.strip()
            break
    if raw is None:
        return pd.Series("NA", index=adata.obs_names, dtype=str)

    mapping = sample.get("label_mapping", {}) or {}
    if not mapping:
        return raw
    reverse: dict[str, str] = {}
    for target, sources in mapping.items():
        reverse[str(target).strip()] = str(target).strip()
        for source in sources if isinstance(sources, list) else [sources]:
            reverse[str(source).strip()] = str(target).strip()
    default = sample.get("label_unmapped_default")
    return raw.map(lambda value: reverse.get(value, value if default is None else str(default)))


def normalize_grid(coords: np.ndarray) -> np.ndarray:
    """Put non-array spatial coordinates on a nearest-neighbour scale."""
    coords = np.asarray(coords, dtype=float)[:, :2]
    if len(coords) < 2:
        return coords.copy()
    sample = coords[: min(512, len(coords))]
    d = distance.cdist(sample, coords)
    d[d == 0] = np.inf
    nearest = np.min(d, axis=1)
    scale = float(np.median(nearest[np.isfinite(nearest)]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (coords - np.nanmin(coords, axis=0)) / scale


def prepare_sample(
    sample: dict[str, Any],
    outdir: Path,
    use_histology: bool,
) -> tuple[sc.AnnData, np.ndarray | None, dict[str, Any]]:
    from spatial_alignment import extract_coords, resolve_image_alignment

    adata = sc.read_h5ad(sample["h5ad_path"])
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs["spot_id"] = adata.obs_names
    adata.obs["sample_id"] = str(sample["sample_id"])
    adata.obs["ground_truth"] = extract_ground_truth(sample, adata).astype(str).values

    coords = extract_coords(adata)
    adata.obs["mocha_x"] = coords["x"].astype(float).values
    adata.obs["mocha_y"] = coords["y"].astype(float).values

    if {"array_row", "array_col"}.issubset(adata.obs.columns):
        grid = adata.obs[["array_col", "array_row"]].to_numpy(dtype=float)
    else:
        grid = normalize_grid(coords[["x", "y"]].to_numpy(dtype=float))
        adata.obs["array_col"] = grid[:, 0]
        adata.obs["array_row"] = grid[:, 1]
    adata.obsm["spatial"] = grid

    image = None
    alignment: dict[str, Any] = {
        "image_enabled": False,
        "mode": "disabled",
        "note": "Histology disabled by STMSC configuration.",
        "overlay_path": None,
    }
    if use_histology:
        alignment = resolve_image_alignment(
            sample=sample,
            adata=adata,
            overlay_outdir=outdir / "alignment_overlays",
        )
        if alignment["image_enabled"]:
            aligned = alignment["aligned_coords"]
            # Upstream indexes image[row, col]. MOCHA alignment returns x=col, y=row.
            adata.obs["x_pixel"] = np.rint(aligned["y"]).astype(int).values
            adata.obs["y_pixel"] = np.rint(aligned["x"]).astype(int).values
            image = np.asarray(alignment["image_array"])

    info = {
        "sample_id": str(sample["sample_id"]),
        "subgroup": sample.get("subgroup"),
        "h5ad_path": str(sample["h5ad_path"]),
        "image_path": sample.get("image_path"),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "image_enabled": bool(alignment["image_enabled"]),
        "alignment_mode": alignment["mode"],
        "alignment_note": alignment["note"],
        "alignment_overlay_path": alignment.get("overlay_path"),
        "alignment_score": alignment.get("score"),
        "alignment_summary": alignment.get("summary"),
    }
    return adata, image, info


def build_pseudo_reference(
    adatas: list[sc.AnnData],
    n_types: int,
    max_spots: int,
    seed: int,
) -> sc.AnnData:
    """Build a label-free pseudo-cell reference used only by upstream HVG selection.

    STMSC requires a labelled scRNA reference in ``preprocess`` even though its
    released latent model does not consume the resulting basis. MOCHA therefore
    clusters a deterministic subsample of the input spots without using benchmark
    labels. This keeps all ten cohorts runnable and records the adaptation in the
    run metadata.
    """
    ref = sc.concat(adatas, axis=0, join="inner", merge="same", index_unique="-")
    if ref.n_obs > max_spots:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(ref.n_obs, size=max_spots, replace=False))
        ref = ref[keep].copy()
    else:
        ref = ref.copy()

    x = ref.X
    if sparse.issparse(x):
        lib = np.asarray(x.sum(axis=1)).ravel()
        x_norm = sparse.diags(1e4 / np.maximum(lib, 1.0)) @ x
        x_norm = x_norm.copy()
        x_norm.data = np.log1p(x_norm.data)
    else:
        x = np.asarray(x, dtype=np.float32)
        lib = x.sum(axis=1)
        x_norm = np.log1p(x * (1e4 / np.maximum(lib, 1.0))[:, None])

    n_types = max(2, min(int(n_types), ref.n_obs // 2))
    n_components = max(2, min(50, ref.n_obs - 1, ref.n_vars - 1))
    embedding = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(x_norm)
    labels = MiniBatchKMeans(
        n_clusters=n_types,
        random_state=seed,
        n_init=10,
        batch_size=min(1024, ref.n_obs),
    ).fit_predict(embedding)
    ref.obs["celltype"] = pd.Categorical([f"pseudo_{x}" for x in labels])
    return ref


def split_groups(samples: list[dict[str, Any]], mode: str) -> dict[str, list[dict[str, Any]]]:
    if mode == "cohort":
        return {"all_samples": samples}
    if mode == "sample":
        return {str(s["sample_id"]): [s] for s in samples}
    if mode not in {"subgroup", "subgroup_or_sample"}:
        raise ValueError(f"Unsupported STMSC integration_scope={mode!r}")

    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        subgroup = sample.get("subgroup")
        if subgroup:
            key = str(subgroup)
        elif mode == "subgroup":
            key = "ungrouped"
        else:
            key = str(sample["sample_id"])
        groups.setdefault(key, []).append(sample)
    return dict(sorted(groups.items()))


def nearest_radius(coords: np.ndarray, coefficient: float) -> float:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return 1.0
    sample = coords[: min(512, len(coords))]
    d = distance.cdist(sample, coords)
    d[d == 0] = np.inf
    nn = np.min(d, axis=1)
    base = float(np.median(nn[np.isfinite(nn)]))
    return max(base * coefficient, np.finfo(float).eps)


def detach_aligned_dataframe_indexes(adata: sc.AnnData) -> None:
    """Make upstream STMSC's direct obs/var index renaming AnnData-safe.

    Some benchmark h5ad files store ``obsm`` entries as DataFrames indexed by
    barcode. STMSC.preprocess directly replaces ``obs.index`` on an AnnData view;
    recent AnnData then rejects the still-old DataFrame index. Array conversion
    preserves row/column order and values while removing that redundant index.
    """
    for mapping_name in ("obsm", "varm", "obsp", "varp", "layers"):
        mapping = getattr(adata, mapping_name)
        for key in list(mapping.keys()):
            value = mapping[key]
            if isinstance(value, pd.DataFrame):
                mapping[key] = value.to_numpy()


def coerce_coordinate_matrix(value: Any, n_obs: int, key: str) -> np.ndarray:
    """Convert upstream object/list coordinates into an n_obs x d float array."""
    if isinstance(value, pd.DataFrame):
        value = value.to_numpy()
    raw = np.asarray(value)
    if raw.ndim == 1:
        try:
            raw = np.vstack([np.asarray(item) for item in raw])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Could not expand object coordinates in {key}") from exc
    elif raw.ndim == 2 and raw.shape[1] == 1 and raw.dtype == object:
        first = raw[0, 0] if raw.size else None
        if isinstance(first, (list, tuple, np.ndarray)):
            raw = np.vstack([np.asarray(item) for item in raw[:, 0]])
    try:
        coords = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Coordinates in {key} are not numeric; shape={raw.shape}") from exc
    if coords.ndim != 2 or coords.shape[0] != n_obs or coords.shape[1] < 2:
        raise ValueError(
            f"Invalid coordinate matrix {key}: shape={coords.shape}; "
            f"expected ({n_obs}, >=2)"
        )
    if not np.isfinite(coords).all():
        raise ValueError(f"Coordinate matrix {key} contains NaN or infinite values")
    return coords


def build_predictions(
    adata: sc.AnnData,
    labels: np.ndarray,
    cohort: str,
    group_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort": cohort,
            "sampleID": adata.obs["sample_id"].astype(str).values,
            "spotID": adata.obs["spot_id"].astype(str).values,
            "x": adata.obs["mocha_x"].astype(float).values,
            "y": adata.obs["mocha_y"].astype(float).values,
            "method": "STMSC",
            "z": adata.obs["ground_truth"].astype(str).values,
            "z_pred": labels.astype(str),
            "obs_name": adata.obs_names.astype(str),
            "spot_id": adata.obs["spot_id"].astype(str).values,
            "sample_id": adata.obs["sample_id"].astype(str).values,
            "group_name": group_name,
            "ground_truth": adata.obs["ground_truth"].astype(str).values,
            "STMSC_domain": labels.astype(str),
        }
    )


def run_group(
    group_name: str,
    samples: list[dict[str, Any]],
    project_root: Path,
    outdir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    method_root = project_root / "methods" / "STMSC"
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))

    import torch
    from STMSC.load_data_preprocess import align_spots, align_spots_3D, extract_histology_features, preprocess
    from STMSC.train import learn_mapping_matrix, train_stmsc_model
    from STMSC.utils import construct_combined_graph

    params = dict(cfg.get("parameters", {}))
    params.update(dict(params.get("group_hyperparameters", {}).get(group_name, {})))
    runtime = cfg.get("runtime", {})
    outputs = cfg.get("outputs", {})
    seed = int(runtime.get("seed", 2024))
    set_seeds(seed)

    group_dir = outdir / "groups" / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    pred_path = group_dir / "checkpoint_predictions.csv"
    meta_path = group_dir / "group_metadata.json"
    success = group_dir / "_SUCCESS"
    if bool(outputs.get("resume_from_checkpoints", True)) and success.exists() and pred_path.exists():
        log(f"[STMSC] group={group_name} checkpoint found; skipping")
        return json.loads(meta_path.read_text())

    started = time.time()
    use_histology = str(params.get("use_histology", "auto")).lower() not in {"false", "0", "no"}
    adatas: list[sc.AnnData] = []
    images: list[np.ndarray | None] = []
    sample_info: list[dict[str, Any]] = []
    for sample in samples:
        adata, image, info = prepare_sample(sample, outdir, use_histology)
        if image is not None:
            extract_histology_features(
                [adata], [image],
                beta=int(params.get("image_patch_size", 49)),
                alpha=float(params.get("histology_alpha", 1.0)),
            )
            z = adata.obs["z_coord"].to_numpy(dtype=float)
            if not np.isfinite(z).all():
                log(
                    f"[STMSC] sample={sample['sample_id']} produced non-finite histology "
                    "features; using z=0 while retaining the verified pixel alignment"
                )
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                adata.obs["z_coord"] = z
                adata.obs["loc"] = np.c_[
                    adata.obs["array_row"].to_numpy(dtype=float),
                    adata.obs["array_col"].to_numpy(dtype=float),
                    z,
                ].astype(np.float32).tolist()
        else:
            loc = np.c_[
                adata.obs["array_row"].to_numpy(dtype=float),
                adata.obs["array_col"].to_numpy(dtype=float),
                np.zeros(adata.n_obs),
            ].astype(np.float32)
            adata.obs["z_coord"] = 0.0
            adata.obs["loc"] = loc.tolist()
        adatas.append(adata)
        images.append(image)
        sample_info.append(info)

    n_input_spots = sum(int(adata.n_obs) for adata in adatas)
    max_dense_spots = int(params.get("max_dense_spots", 18000))
    if n_input_spots > max_dense_spots:
        raise MemoryError(
            f"STMSC uses dense O(n_spots^2) graphs; group {group_name} has "
            f"{n_input_spots} input spots, above parameters.max_dense_spots="
            f"{max_dense_spots}. Split the biological group or raise the limit only "
            "after checking host and GPU memory."
        )

    data_type = str(params.get("data_type", "auto"))
    if data_type == "auto":
        data_type = "Visium" if "10x" in samples[0]["cohort_name"] else "ST"

    if len(adatas) == 1:
        adatas[0].obsm["spatial_aligned"] = np.asarray(adatas[0].obs["loc"].tolist(), dtype=float)
    elif all(image is not None for image in images):
        adatas = align_spots_3D(
            adatas,
            method=str(params.get("alignment_method", "icp")),
            data_type=data_type,
            tol=float(params.get("alignment_tolerance", 0.01)),
            test_all_angles=bool(params.get("test_all_angles", False)),
        )
    else:
        adatas = align_spots(
            adatas,
            method=str(params.get("alignment_method", "icp")),
            data_type=data_type,
            coor_key="spatial",
            tol=float(params.get("alignment_tolerance", 0.01)),
            test_all_angles=bool(params.get("test_all_angles", False)),
            plot=False,
        )

    # Upstream align_spots_3D stores the first slice via
    # np.array(adata.obs['loc']), which is a 1-D object array of Python lists
    # on current pandas. Normalize every slice before STMSC.preprocess calls
    # sklearn.pairwise_distances.
    for adata in adatas:
        adata.obsm["spatial_aligned"] = coerce_coordinate_matrix(
            adata.obsm["spatial_aligned"],
            n_obs=adata.n_obs,
            key="obsm['spatial_aligned']",
        )

    reference = build_pseudo_reference(
        adatas,
        n_types=int(params.get("pseudo_reference_types", 8)),
        max_spots=int(params.get("pseudo_reference_max_spots", 4000)),
        seed=seed,
    )
    for adata in adatas:
        detach_aligned_dataframe_indexes(adata)
    detach_aligned_dataframe_indexes(reference)
    slice_distance = float(params.get("slice_distance_microns", 10.0))
    slice_distances = [slice_distance] * (len(adatas) - 1)
    st, basis, _ = preprocess(
        adatas,
        reference,
        celltype_ref_col="celltype",
        n_hvg_group=int(params.get("n_hvg_group", 500)),
        coor_key="spatial_aligned",
        rad_coef=float(params.get("rad_coef", 1.1)),
        slice_dist_micron=slice_distances,
        prune_graph_cos=bool(params.get("prune_graph_cos", False)),
        cos_threshold=float(params.get("cos_threshold", 0.5)),
        c2c_dist=float(params.get("c2c_dist", 100.0)),
    )
    # Released STMSC training functions explicitly call .toarray(). Keep X as
    # CSR without passing a sparse object through np.asarray (which creates an
    # object scalar on recent SciPy/NumPy combinations).
    if sparse.issparse(st.X):
        st.X = st.X.astype(np.float32).tocsr()
    else:
        st.X = sparse.csr_matrix(np.asarray(st.X, dtype=np.float32))
    if sparse.issparse(basis.X):
        basis.X = basis.X.astype(np.float32).toarray()
    else:
        basis.X = np.asarray(basis.X, dtype=np.float32)

    n_spots = int(st.n_obs)
    if n_spots > max_dense_spots:
        raise MemoryError(
            f"STMSC uses dense O(n_spots^2) graphs; group {group_name} has {n_spots} spots, "
            f"above parameters.max_dense_spots={max_dense_spots}. Split the biological group "
            "or raise the limit only after checking GPU memory."
        )

    device_cfg = str(runtime.get("device", "auto"))
    device = "cuda:0" if device_cfg == "auto" and torch.cuda.is_available() else device_cfg
    if device == "auto":
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("STMSC was configured for CUDA, but torch.cuda.is_available() is false")

    map_path = group_dir / "mapping_matrix.npy"
    if bool(outputs.get("resume_from_checkpoints", True)) and map_path.exists():
        soft_map = np.load(map_path)
        st.obsm["map_matrix"] = soft_map
        log(f"[STMSC] group={group_name} loaded mapping checkpoint {map_path}")
    else:
        log(f"[STMSC] group={group_name} mapping epochs={runtime.get('mapping_epochs', 5000)}")
        learn_mapping_matrix(
            st,
            basis,
            lam=float(params.get("lam", 7.0)),
            device=device,
            epoch=int(runtime.get("mapping_epochs", 5000)),
        )
        soft_map = np.asarray(st.obsm["map_matrix"], dtype=np.float32)
        np.save(map_path, soft_map)

    hist_loc = np.vstack([np.asarray(x, dtype=float) for x in st.obs["loc"]])
    radius_cfg = params.get("combined_graph_radius", "auto")
    radius = (
        nearest_radius(np.asarray(st.obsm["3D_coor"]), float(params.get("combined_radius_coef", 1.5)))
        if radius_cfg in {None, "auto"}
        else float(radius_cfg)
    )
    st.obsm["graph"] = construct_combined_graph(
        st,
        hist_loc,
        str(map_path),
        bl=float(params.get("bl", 0.1)),
        bll=float(params.get("bll", 0.1)),
        radius=radius,
    ).astype(np.float32)

    latent_checkpoint = group_dir / "latent_checkpoint.npy"
    if bool(outputs.get("resume_from_checkpoints", True)) and latent_checkpoint.exists():
        latent = np.load(latent_checkpoint)
        if latent.shape[0] != st.n_obs:
            raise ValueError(
                f"Latent checkpoint row count {latent.shape[0]} does not match "
                f"preprocessed spots {st.n_obs}: {latent_checkpoint}"
            )
        log(f"[STMSC] group={group_name} loaded latent checkpoint {latent_checkpoint}")
    else:
        log(f"[STMSC] group={group_name} latent training epochs={runtime.get('epochs', 5000)} device={device}")
        _, latent = train_stmsc_model(
            st,
            basis,
            device=device,
            epochs=int(runtime.get("epochs", 5000)),
            lr=float(runtime.get("learning_rate", 0.01)),
        )
        # Write this before clustering/export so a late numerical or I/O failure
        # never forces another 5,000-epoch latent training run.
        np.save(latent_checkpoint, np.asarray(latent, dtype=np.float32))

    n_domains = int(params["n_domains"])
    cluster_method = str(params.get("cluster_method", "gaussian_mixture"))
    cluster_method_used = cluster_method
    gmm_reg_covar_used: float | None = None
    if cluster_method == "gaussian_mixture":
        latent_for_cluster = np.asarray(latent, dtype=np.float64)
        if not np.isfinite(latent_for_cluster).all():
            raise ValueError("STMSC latent contains NaN or infinite values")
        if bool(params.get("standardize_latent_for_clustering", True)):
            latent_for_cluster = StandardScaler().fit_transform(latent_for_cluster)

        configured_reg = float(params.get("gmm_reg_covar", 1e-5))
        regularizers = []
        for value in [configured_reg, 1e-4, 1e-3, 1e-2]:
            if value not in regularizers:
                regularizers.append(value)
        last_gmm_error: Exception | None = None
        labels = None
        for reg_covar in regularizers:
            try:
                labels = GaussianMixture(
                    n_components=n_domains,
                    covariance_type=str(params.get("gmm_covariance_type", "tied")),
                    reg_covar=reg_covar,
                    random_state=seed,
                    n_init=int(params.get("gmm_n_init", 10)),
                ).fit_predict(latent_for_cluster)
                gmm_reg_covar_used = reg_covar
                log(f"[STMSC] group={group_name} GMM succeeded reg_covar={reg_covar}")
                break
            except (ValueError, np.linalg.LinAlgError) as exc:
                last_gmm_error = exc
                log(
                    f"[STMSC] group={group_name} GMM failed reg_covar={reg_covar}: {exc}"
                )
        if labels is None:
            if not bool(params.get("fallback_to_kmeans", True)):
                raise RuntimeError("All STMSC GMM regularization attempts failed") from last_gmm_error
            cluster_method_used = "kmeans_fallback_after_gmm"
            log(f"[STMSC] group={group_name} falling back to KMeans after GMM failures")
            labels = KMeans(n_clusters=n_domains, random_state=seed, n_init=20).fit_predict(
                latent_for_cluster
            )
    elif cluster_method == "kmeans":
        labels = KMeans(n_clusters=n_domains, random_state=seed, n_init=20).fit_predict(
            np.asarray(latent, dtype=np.float64)
        )
    else:
        raise ValueError(f"Unsupported STMSC cluster_method={cluster_method!r}")
    pred = build_predictions(st, labels, samples[0]["cohort_name"], group_name)
    pred.to_csv(pred_path, index=False)
    if bool(outputs.get("save_embedding", False)):
        np.save(group_dir / "latent.npy", latent.astype(np.float32))

    clean = pred.loc[~pred["z"].isin(["NA", "Filtered", "Unknown", "nan"])]
    ari = float(adjusted_rand_index(clean["z"], clean["z_pred"])) if len(clean) else None
    meta = {
        "group_name": group_name,
        "sample_ids": [str(s["sample_id"]) for s in samples],
        "n_samples": len(samples),
        "n_spots": n_spots,
        "n_genes_used": int(st.n_vars),
        "n_domains": n_domains,
        "device": device,
        "combined_graph_radius": radius,
        "effective_parameters": params,
        "cluster_method_requested": cluster_method,
        "cluster_method_used": cluster_method_used,
        "gmm_reg_covar_used": gmm_reg_covar_used,
        "ari": ari,
        "runtime_seconds": time.time() - started,
        "memory_mb": memory_mb(),
        "samples": sample_info,
        "reference_adaptation": {
            "mode": "label_free_pseudo_reference",
            "purpose": "upstream STMSC HVG selection only",
            "uses_ground_truth": False,
            "pseudo_reference_types": int(params.get("pseudo_reference_types", 8)),
        },
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(meta_path, meta)
    success.write_text("success\n", encoding="utf-8")
    # Remove restart-only checkpoints only after every required group artifact
    # and the success marker have been committed.
    if not bool(outputs.get("save_embedding", False)):
        latent_checkpoint.unlink(missing_ok=True)
    if not bool(outputs.get("save_mapping_matrix", False)):
        map_path.unlink(missing_ok=True)
    log(f"[STMSC] group={group_name} complete")
    return meta


def aggregate_if_complete(
    outdir: Path,
    group_names: list[str],
    cohort: str,
    cfg: dict[str, Any],
) -> bool:
    lock_path = outdir / ".aggregate.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (outdir / "_SUCCESS").exists() and (outdir / "predictions.parquet").exists():
            log(f"[STMSC] cohort={cohort} is already aggregated")
            return True
        missing = [name for name in group_names if not (outdir / "groups" / name / "_SUCCESS").exists()]
        if missing:
            log(f"[STMSC] {len(missing)}/{len(group_names)} groups remain; cohort aggregation deferred")
            return False

        predictions = []
        group_meta = []
        for name in group_names:
            group_dir = outdir / "groups" / name
            predictions.append(pd.read_csv(group_dir / "checkpoint_predictions.csv"))
            group_meta.append(json.loads((group_dir / "group_metadata.json").read_text()))
        pred = write_predictions_tables(pd.concat(predictions, ignore_index=True), outdir)
        runtime_seconds = float(sum(x["runtime_seconds"] for x in group_meta))
        max_memory = float(max(x["memory_mb"] for x in group_meta))
        write_performance_tables(
            build_performance_df(cohort, "STMSC", runtime_seconds, max_memory), outdir
        )
        save_evaluation_summary(pred, outdir, cohort, "STMSC")
        save_true_vs_pred_plots(pred, outdir, cohort, "STMSC")
        write_json(
            outdir / "run_metadata.json",
            {
                "method": "STMSC",
                "cohort": cohort,
                "status": "success",
                "n_samples": int(pred["sampleID"].nunique()),
                "n_groups": len(group_names),
                "integration_scope": cfg.get("runtime", {}).get("integration_scope"),
                "runtime_seconds_sum": runtime_seconds,
                "peak_memory_mb_max": max_memory,
                "parameters": cfg.get("parameters", {}),
                "runtime": cfg.get("runtime", {}),
                "groups": group_meta,
                "image_usage": (
                    "STMSC histology patches are used only after MOCHA spot-to-pixel alignment; "
                    "low-confidence or missing image alignment falls back to expression+spatial."
                ),
                "reference_adaptation": (
                    "No common scRNA reference exists for the ten benchmark cohorts. A label-free "
                    "pseudo-reference is used only for the released STMSC preprocess/HVG interface; "
                    "benchmark ground truth is never used."
                ),
            },
        )
        (outdir / "_SUCCESS").write_text("success\n", encoding="utf-8")
        log(f"[STMSC] cohort={cohort} aggregation complete")
        return True


def run_stmsc(
    samples: list[dict[str, Any]],
    project_root: Path,
    pipeline_cfg: dict[str, Any],
    rscript_bin: str = "Rscript",
) -> dict[str, Any]:
    del rscript_bin
    if not samples:
        raise ValueError("Empty samples list passed to STMSC")
    cohort = str(samples[0]["cohort_name"])
    outdir = project_root / "results" / "STMSC" / cohort
    outdir.mkdir(parents=True, exist_ok=True)
    groups = split_groups(
        samples,
        str(pipeline_cfg.get("runtime", {}).get("integration_scope", "subgroup_or_sample")),
    )
    names = list(groups)
    group_idx = pipeline_cfg.get("runtime", {}).get("group_idx")
    selected = names
    if group_idx is not None:
        group_idx = int(group_idx)
        if group_idx < 0 or group_idx >= len(names):
            raise IndexError(f"STMSC group_idx={group_idx} outside [0, {len(names) - 1}]")
        selected = [names[group_idx]]

    stdout_path = outdir / (f"stdout.group_{group_idx}.log" if group_idx is not None else "stdout.log")
    stderr_path = outdir / (f"stderr.group_{group_idx}.log" if group_idx is not None else "stderr.log")
    status = "partial"
    try:
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                log(f"[STMSC] cohort={cohort} groups={len(names)} selected={selected}")
                for name in selected:
                    run_group(name, groups[name], project_root, outdir, pipeline_cfg)
                status = "success" if aggregate_if_complete(outdir, names, cohort, pipeline_cfg) else "partial"
    except Exception as exc:
        write_json(
            outdir / (f"failure.group_{group_idx}.json" if group_idx is not None else "failure.json"),
            {"method": "STMSC", "cohort": cohort, "status": "failed", "error_message": str(exc)},
        )
        raise
    return {"status": status, "cohort": cohort, "groups_selected": selected}
