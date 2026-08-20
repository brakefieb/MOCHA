from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image


def extract_coords(adata: sc.AnnData) -> pd.DataFrame:
    obs = adata.obs

    candidate_pairs = [
        ("x", "y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("pxl_col", "pxl_row"),
        ("imagecol", "imagerow"),
        ("array_col", "array_row"),
        ("col", "row"),
    ]

    for c1, c2 in candidate_pairs:
        if c1 in obs.columns and c2 in obs.columns:
            coords = obs[[c1, c2]].copy()
            coords.columns = ["x", "y"]
            coords.index = adata.obs_names.astype(str)
            return coords

    for key in ["spatial", "X_spatial", "S"]:
        if key not in adata.obsm:
            continue
        spatial = adata.obsm[key]
        if isinstance(spatial, pd.DataFrame):
            if {"x", "y"}.issubset(spatial.columns):
                coords = spatial[["x", "y"]].copy()
            else:
                coords = spatial.iloc[:, :2].copy()
                coords.columns = ["x", "y"]
            coords.index = adata.obs_names.astype(str)
            return coords

        if hasattr(spatial, "toarray"):
            spatial = spatial.toarray()
        spatial = np.asarray(spatial)
        if spatial.ndim == 2 and spatial.shape[1] >= 2:
            return pd.DataFrame(
                spatial[:, :2],
                index=adata.obs_names.astype(str),
                columns=["x", "y"],
            )

    raise ValueError(
        "Could not find spatial coordinates in adata.obs or adata.obsm. "
        f"obs columns: {list(obs.columns)}, obsm keys: {list(adata.obsm.keys())}"
    )


def estimate_spot_diameter(coords: pd.DataFrame) -> float:
    if coords.shape[0] < 2:
        return 50.0

    xy = coords[["x", "y"]].to_numpy(dtype=float)
    sample_n = min(256, xy.shape[0])
    anchor = xy[:sample_n]
    distances = np.sqrt(((anchor[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    distances[distances == 0] = np.inf
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 50.0
    return float(max(np.median(nearest), 16.0))


def normalize_alignment_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    return {
        "mode": str(cfg.get("mode", "auto")),
        "fallback_mode": str(cfg.get("fallback_mode", "disable_image")),
        "min_direct_span_ratio": float(cfg.get("min_direct_span_ratio", 0.25)),
        "min_tissue_coverage": float(cfg.get("min_tissue_coverage", 0.15)),
        "save_overlay": bool(cfg.get("save_overlay", True)),
        "transform_matrix_glob": cfg.get("transform_matrix_glob", "{sample_id}_st_transform_matrix.txt"),
        "visium_scale_preference": str(cfg.get("visium_scale_preference", "auto")),
    }


def summarize_coords(coords: pd.DataFrame, image: Image.Image, min_direct_span_ratio: float) -> dict[str, Any]:
    width, height = image.size
    x = coords["x"].astype(float).to_numpy()
    y = coords["y"].astype(float).to_numpy()

    min_x, max_x = float(np.nanmin(x)), float(np.nanmax(x))
    min_y, max_y = float(np.nanmin(y)), float(np.nanmax(y))
    x_span = max_x - min_x
    y_span = max_y - min_y
    x_span_ratio = x_span / max(width, 1)
    y_span_ratio = y_span / max(height, 1)
    within_bounds = min_x >= 0 and min_y >= 0 and max_x < width and max_y < height
    plausible_pixel_scale = (
        within_bounds
        and x_span_ratio >= min_direct_span_ratio
        and y_span_ratio >= min_direct_span_ratio
    )

    return {
        "image_width": width,
        "image_height": height,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "x_span": x_span,
        "y_span": y_span,
        "x_span_ratio": x_span_ratio,
        "y_span_ratio": y_span_ratio,
        "within_image_bounds": within_bounds,
        "plausible_pixel_scale": plausible_pixel_scale,
    }


def build_tissue_mask(image_array: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image_array)
    mean_intensity = image_array.mean(axis=2)
    channel_range = image_array.max(axis=2) - image_array.min(axis=2)
    # Tissue tends to be darker and/or more chromatic than background glass.
    return (mean_intensity < 245.0) | (channel_range > 10.0)


def score_overlay(coords: pd.DataFrame, image_array: np.ndarray) -> dict[str, float]:
    mask = build_tissue_mask(image_array)
    height, width = mask.shape

    x = np.rint(coords["x"].astype(float).to_numpy()).astype(int)
    y = np.rint(coords["y"].astype(float).to_numpy()).astype(int)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    in_bounds_frac = float(in_bounds.mean()) if len(in_bounds) else 0.0
    if in_bounds.sum() == 0:
        return {"score": 0.0, "in_bounds_frac": 0.0, "tissue_frac": 0.0}

    tissue_frac = float(mask[y[in_bounds], x[in_bounds]].mean())
    score = in_bounds_frac * tissue_frac
    return {
        "score": score,
        "in_bounds_frac": in_bounds_frac,
        "tissue_frac": tissue_frac,
    }


def find_visium_sidecars(sample: dict[str, Any]) -> tuple[Path | None, Path | None]:
    image_path = Path(sample["image_path"])
    image_dir = image_path.parent
    sample_id = str(sample["sample_id"])

    tissue_candidates = [
        image_dir / f"{sample_id}_tissue_positions_list.csv",
        image_dir / f"{sample_id}_tissue_positions.csv",
    ]
    scale_path = image_dir / f"{sample_id}_scalefactors_json.json"
    tissue_path = next((p for p in tissue_candidates if p.exists()), None)
    if tissue_path is not None:
        return tissue_path, scale_path if scale_path.exists() else None
    return None, None


def choose_best_visium_scale(
    coords_fullres: pd.DataFrame,
    image: Image.Image,
    scale_info: dict[str, Any],
    preference: str = "auto",
) -> tuple[float, str]:
    preference = str(preference).lower()
    if preference == "hires" and scale_info.get("tissue_hires_scalef") is not None:
        return float(scale_info["tissue_hires_scalef"]), "hires"
    if preference == "lowres" and scale_info.get("tissue_lowres_scalef") is not None:
        return float(scale_info["tissue_lowres_scalef"]), "lowres"
    if preference == "fullres":
        return 1.0, "fullres"

    width, height = image.size
    max_x = float(coords_fullres["x"].max())
    max_y = float(coords_fullres["y"].max())

    candidates: list[tuple[str, float]] = [("fullres", 1.0)]
    if scale_info.get("tissue_hires_scalef") is not None:
        candidates.append(("hires", float(scale_info["tissue_hires_scalef"])))
    if scale_info.get("tissue_lowres_scalef") is not None:
        candidates.append(("lowres", float(scale_info["tissue_lowres_scalef"])))

    best_label = "fullres"
    best_scale = 1.0
    best_score = float("inf")
    for label, scale in candidates:
        scaled_w = max_x * scale
        scaled_h = max_y * scale
        overflow = max(0.0, scaled_w - width) + max(0.0, scaled_h - height)
        size_gap = abs(scaled_w - width) + abs(scaled_h - height)
        score = overflow * 1000.0 + size_gap
        if score < best_score:
            best_score = score
            best_label = label
            best_scale = scale
    return best_scale, best_label


def load_visium_coords(
    sample: dict[str, Any],
    adata: sc.AnnData,
    image: Image.Image,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame | None, str]:
    tissue_path, scale_path = find_visium_sidecars(sample)
    if tissue_path is None:
        return None, "no_visium_sidecar"

    if tissue_path.name.endswith("_tissue_positions.csv"):
        tissue = pd.read_csv(tissue_path)
    else:
        tissue = pd.read_csv(
            tissue_path,
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )

    required_cols = {
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    }
    if not required_cols.issubset(tissue.columns):
        raise ValueError(
            f"Visium tissue positions file is missing required columns: {tissue_path}"
        )

    tissue["barcode"] = tissue["barcode"].astype(str)
    tissue = tissue.set_index("barcode")
    common = tissue.index.intersection(adata.obs_names.astype(str))
    if common.empty:
        return None, "visium_sidecar_no_overlap"

    coords = tissue.loc[common, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].copy()
    coords.columns = ["x", "y"]

    scale_info: dict[str, Any] = {}
    if scale_path is not None:
        scale_info = json.loads(scale_path.read_text())
    scale_value, scale_label = choose_best_visium_scale(
        coords,
        image,
        scale_info,
        preference=str(cfg.get("visium_scale_preference", "auto")),
    )
    coords["x"] = coords["x"].astype(float) * scale_value
    coords["y"] = coords["y"].astype(float) * scale_value
    return coords, f"visium_sidecar_{scale_label}; scale={scale_value:.8f}"


def find_st_transform_matrix(sample: dict[str, Any], cfg: dict[str, Any]) -> Path | None:
    image_path = Path(sample["image_path"])
    image_dir = image_path.parent
    sample_id = str(sample["sample_id"])
    pattern = str(cfg.get("transform_matrix_glob", "{sample_id}_st_transform_matrix.txt"))
    path = image_dir / pattern.format(sample_id=sample_id)
    return path if path.exists() else None


def read_st_transform_matrix(path: Path) -> np.ndarray:
    raw = path.read_text().strip().replace(",", " ")
    values = [float(x) for x in raw.split()]
    if len(values) != 9:
        raise ValueError(f"Expected 9 numeric values in {path}, found {len(values)}")
    return np.asarray(values, dtype=float).reshape(3, 3)


def apply_st_transform_matrix(coords: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    xy1 = np.c_[coords["x"].astype(float).to_numpy(), coords["y"].astype(float).to_numpy(), np.ones(coords.shape[0])]
    transformed = xy1 @ matrix
    out = pd.DataFrame(transformed[:, :2], index=coords.index, columns=["x", "y"])
    return out


def load_st_transform_matrix_coords(
    sample: dict[str, Any],
    adata: sc.AnnData,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame | None, str]:
    matrix_path = find_st_transform_matrix(sample, cfg)
    if matrix_path is None:
        return None, "no_st_transform_matrix"
    coords = extract_coords(adata)
    matrix = read_st_transform_matrix(matrix_path)
    transformed = apply_st_transform_matrix(coords, matrix)
    return transformed, f"st_transform_matrix:{matrix_path.name}"


def choose_direct_alignment(
    coords: pd.DataFrame,
    image: Image.Image,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    image_array = np.asarray(image)
    transformed = coords.copy().astype(float)
    summary = summarize_coords(
        transformed,
        image,
        min_direct_span_ratio=float(cfg["min_direct_span_ratio"]),
    )
    score = score_overlay(transformed, image_array)
    note = (
        "direct_coords;"
        + f" tissue_score={score['score']:.4f}"
        + f"; tissue_frac={score['tissue_frac']:.4f}"
        + f"; in_bounds_frac={score['in_bounds_frac']:.4f}"
    )
    return {
        "coords": transformed,
        "summary": summary,
        "score": score,
        "note": note,
        "usable": (
            summary["plausible_pixel_scale"]
            and score["tissue_frac"] >= float(cfg["min_tissue_coverage"])
        ),
    }


def save_overlay_figure(
    image: Image.Image,
    coords: pd.DataFrame,
    outpath: Path,
    title: str,
    alpha: float = 0.45,
    size: float = 8.0,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(image)
    ax.scatter(coords["x"], coords["y"], s=size, c="cyan", alpha=alpha, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_image_alignment(
    sample: dict[str, Any],
    adata: sc.AnnData,
    overlay_outdir: Path | None = None,
) -> dict[str, Any]:
    base_coords = extract_coords(adata)
    image_path = sample.get("image_path")
    cfg = normalize_alignment_config(sample.get("spatial_alignment"))

    if not image_path:
        return {
            "image_enabled": False,
            "mode": "no_image",
            "note": "No image_path available; using expression-only DeepST.",
            "base_coords": base_coords,
            "aligned_coords": base_coords,
            "image_path": None,
            "image_array": None,
            "overlay_path": None,
        }

    img_path = Path(image_path)
    if not img_path.exists():
        return {
            "image_enabled": False,
            "mode": "missing_image",
            "note": f"Image not found: {img_path}",
            "base_coords": base_coords,
            "aligned_coords": base_coords,
            "image_path": str(img_path),
            "image_array": None,
            "overlay_path": None,
        }

    image = Image.open(img_path).convert("RGB")
    image_array = np.asarray(image)
    mode = cfg["mode"].lower()
    fallback_mode = cfg["fallback_mode"].lower()

    aligned_coords: pd.DataFrame | None = None
    note = ""
    resolved_mode = ""
    score: dict[str, float] | None = None
    summary: dict[str, Any] | None = None

    if mode in {"auto", "visium_sidecar"}:
        visium_coords, visium_note = load_visium_coords(sample, adata, image, cfg)
        if visium_coords is not None:
            aligned_coords = visium_coords
            note = visium_note
            resolved_mode = "visium_sidecar"
            summary = summarize_coords(
                aligned_coords,
                image,
                min_direct_span_ratio=float(cfg["min_direct_span_ratio"]),
            )
            score = score_overlay(aligned_coords, image_array)

    if aligned_coords is None and mode in {"auto", "direct"}:
        direct = choose_direct_alignment(base_coords, image, cfg)
        aligned_coords = direct["coords"]
        note = direct["note"]
        resolved_mode = "direct"
        summary = direct["summary"]
        score = direct["score"]
        if not direct["usable"] and fallback_mode == "disable_image":
            note += "; image_disabled_due_to_low_alignment_confidence"
        elif not direct["usable"] and fallback_mode == "use_direct_anyway":
            note += "; using_low_confidence_direct_alignment"

    if aligned_coords is None and mode in {"auto", "st_transform_matrix"}:
        st_coords, st_note = load_st_transform_matrix_coords(sample, adata, cfg)
        if st_coords is not None:
            aligned_coords = st_coords
            note = st_note
            resolved_mode = "st_transform_matrix"
            summary = summarize_coords(
                aligned_coords,
                image,
                min_direct_span_ratio=float(cfg["min_direct_span_ratio"]),
            )
            score = score_overlay(aligned_coords, image_array)

    if aligned_coords is None:
        aligned_coords = base_coords.copy()
        resolved_mode = "unaligned"
        note = f"Could not resolve alignment with mode={mode}; using expression-only DeepST."
        summary = summarize_coords(
            aligned_coords,
            image,
            min_direct_span_ratio=float(cfg["min_direct_span_ratio"]),
        )
        score = score_overlay(aligned_coords, image_array)

    image_enabled = resolved_mode in {"visium_sidecar", "direct", "st_transform_matrix"}
    if resolved_mode == "direct" and fallback_mode == "disable_image":
        image_enabled = (
            summary is not None
            and bool(summary["plausible_pixel_scale"])
            and score is not None
            and float(score["tissue_frac"]) >= float(cfg["min_tissue_coverage"])
        )
    elif resolved_mode == "unaligned":
        image_enabled = False

    overlay_path = None
    if overlay_outdir is not None and bool(cfg.get("save_overlay", True)):
        overlay_path = overlay_outdir / f"{sample['sample_id']}__overlay.png"
        save_overlay_figure(
            image=image,
            coords=aligned_coords,
            outpath=overlay_path,
            title=f"{sample['cohort_name']} / {sample['sample_id']}\nmode={resolved_mode}; {note}",
        )

    return {
        "image_enabled": image_enabled,
        "mode": resolved_mode,
        "note": note,
        "base_coords": base_coords,
        "aligned_coords": aligned_coords,
        "image_path": str(img_path),
        "image_array": image_array,
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
        "score": score,
        "summary": summary,
    }
