from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs"}


def load_yaml(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML structure in {path}")

    return config


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _list_files(directory: Path, pattern: str) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([p for p in directory.glob(pattern) if p.is_file()])


def _index_by_stem(files: List[Path]) -> Dict[str, Path]:
    out = {}
    for f in files:
        out[f.stem] = f
    return out


def _canonical_sample_id_from_stem(stem: str) -> str:
    """
    Convert stems like:
      SCE_151507 -> 151507
      HE_151507 -> 151507
      SCE_SN048_A121573_Rep1 -> SN048_A121573_Rep1
      HE_SN048_A121573_Rep1 -> SN048_A121573_Rep1
    Falls back to the original stem if no clear normalization is available.
    """
    stem = stem.strip()

    lowered = stem.lower()
    for prefix in ["sce_", "he_", "sample_", "tissue_", "img_"]:
        if lowered.startswith(prefix):
            stripped = stem[len(prefix):]
            if re.fullmatch(r"\d{6,}", stripped):
                return stripped
            return stripped

    m = re.search(r'(\d{6,})$', stem)
    if m:
        return m.group(1)

    m = re.search(r'(\d{6,})', stem)
    if m:
        return m.group(1)

    return stem


def _build_subgroup_lookup(subgroup_map: Dict[str, List[Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for subgroup, members in subgroup_map.items():
        if not isinstance(members, list):
            raise ValueError(f"subgroup_map[{subgroup}] must be a list")
        for member in members:
            key = str(member)
            if key in lookup and lookup[key] != subgroup:
                raise ValueError(
                    f"Sample {key} appears in multiple subgroups: "
                    f"{lookup[key]} and {subgroup}"
                )
            lookup[key] = subgroup
    return lookup


def _find_image_for_h5ad(
    h5ad_file: Path,
    image_files: List[Path],
    match_by_stem: bool = True,
) -> Optional[Path]:
    if not image_files:
        return None

    if match_by_stem:
        image_map = _index_by_stem(image_files)
        exact = image_map.get(h5ad_file.stem)
        if exact is not None:
            return exact

    # robust fallback: match by canonical sample id
    h5ad_key = _canonical_sample_id_from_stem(h5ad_file.stem)
    candidates = []
    for img in image_files:
        if _canonical_sample_id_from_stem(img.stem) == h5ad_key:
            candidates.append(img)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple images matched sample {h5ad_file.stem} ({h5ad_key}): "
            f"{[str(x) for x in candidates]}"
        )

    # fallback: if only one image exists, use it
    if len(image_files) == 1:
        return image_files[0]

    return None


def _validate_required_fields(config: Dict[str, Any], required: List[str]) -> None:
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")


def discover_samples_from_standard_config(
    config: Dict[str, Any],
    project_root: Path,
) -> List[Dict[str, Any]]:
    _validate_required_fields(config, ["cohort_name", "data_dir", "h5ad_glob"])

    cohort_name = config["cohort_name"]
    has_he_image = bool(config.get("has_he_image", False))
    match_by_stem = bool(config.get("match_by_stem", True))
    label_mapping = config.get("label_mapping", {})
    label_unmapped_default = config.get("label_unmapped_default")
    subgroup_map = config.get("subgroup_map", {})
    subgroup_lookup = _build_subgroup_lookup(subgroup_map) if subgroup_map else {}
    spatial_alignment = config.get("spatial_alignment", {})

    data_dir = _resolve_path(config["data_dir"], project_root)
    h5ad_glob = config["h5ad_glob"]
    h5ad_files = _list_files(data_dir, h5ad_glob)

    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files found in {data_dir} with glob={h5ad_glob}")

    image_files: List[Path] = []
    if has_he_image:
        if "image_dir" not in config:
            raise ValueError(f"{cohort_name}: has_he_image=true but image_dir is missing")
        if "image_glob" not in config:
            raise ValueError(f"{cohort_name}: has_he_image=true but image_glob is missing")

        image_dir = _resolve_path(config["image_dir"], project_root)
        image_glob = config["image_glob"]
        image_files = [
            path
            for path in _list_files(image_dir, image_glob)
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    samples: List[Dict[str, Any]] = []
    for h5ad_file in h5ad_files:
        image_path = _find_image_for_h5ad(h5ad_file, image_files, match_by_stem=match_by_stem)
        sample_id = _canonical_sample_id_from_stem(h5ad_file.stem)
        subgroup = subgroup_lookup.get(sample_id)

        sample_record = {
            "cohort_name": cohort_name,
            "sample_id": sample_id,
            "subgroup": subgroup,
            "h5ad_path": str(h5ad_file),
            "image_path": str(image_path) if image_path else None,
            "label_mapping": label_mapping,
            "label_unmapped_default": label_unmapped_default,
            "spatial_alignment": spatial_alignment,
        }
        samples.append(sample_record)

    return samples


def discover_samples_from_subgroups(
    config: Dict[str, Any],
    project_root: Path,
) -> List[Dict[str, Any]]:
    _validate_required_fields(config, ["cohort_name", "subgroups"])

    cohort_name = config["cohort_name"]
    has_he_image = bool(config.get("has_he_image", False))
    match_by_stem = bool(config.get("match_by_stem", True))
    label_mapping = config.get("label_mapping", {})
    label_unmapped_default = config.get("label_unmapped_default")
    spatial_alignment = config.get("spatial_alignment", {})

    subgroups = config["subgroups"]
    if not isinstance(subgroups, dict) or not subgroups:
        raise ValueError(f"{cohort_name}: subgroups must be a non-empty dict")

    all_samples: List[Dict[str, Any]] = []

    for subgroup_name, subgroup_cfg in subgroups.items():
        required = ["data_dir", "h5ad_glob"]
        missing = [k for k in required if k not in subgroup_cfg]
        if missing:
            raise ValueError(
                f"{cohort_name}/{subgroup_name}: missing subgroup fields {missing}"
            )

        data_dir = _resolve_path(subgroup_cfg["data_dir"], project_root)
        h5ad_glob = subgroup_cfg["h5ad_glob"]
        h5ad_files = _list_files(data_dir, h5ad_glob)

        if not h5ad_files:
            raise FileNotFoundError(
                f"No h5ad files found in subgroup {subgroup_name}: {data_dir}"
            )

        image_files: List[Path] = []
        if has_he_image:
            if "image_dir" not in subgroup_cfg:
                raise ValueError(
                    f"{cohort_name}/{subgroup_name}: has_he_image=true but image_dir missing"
                )
            if "image_glob" not in subgroup_cfg:
                raise ValueError(
                    f"{cohort_name}/{subgroup_name}: has_he_image=true but image_glob missing"
                )

            image_dir = _resolve_path(subgroup_cfg["image_dir"], project_root)
            image_glob = subgroup_cfg["image_glob"]
            image_files = [
                path
                for path in _list_files(image_dir, image_glob)
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]

        for h5ad_file in h5ad_files:
            image_path = _find_image_for_h5ad(h5ad_file, image_files, match_by_stem=match_by_stem)

            sample_record = {
                "cohort_name": cohort_name,
                "sample_id": _canonical_sample_id_from_stem(h5ad_file.stem),
                "subgroup": subgroup_name,
                "h5ad_path": str(h5ad_file),
                "image_path": str(image_path) if image_path else None,
                "label_mapping": label_mapping,
                "label_unmapped_default": label_unmapped_default,
                "spatial_alignment": spatial_alignment,
            }
            all_samples.append(sample_record)

    return all_samples


def discover_samples(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> List[Dict[str, Any]]:
    config = load_yaml(config_path)

    if project_root is None:
        # assume config is under PROJECT_ROOT/configs/cohorts/
        config_file = Path(config_path).expanduser().resolve()
        project_root = config_file.parents[2]
    project_root = Path(project_root).expanduser().resolve()

    if "subgroups" in config:
        samples = discover_samples_from_subgroups(config, project_root)
    else:
        samples = discover_samples_from_standard_config(config, project_root)

    if not samples:
        raise RuntimeError(f"No samples discovered from config: {config_path}")

    return samples


def print_sample_summary(samples: List[Dict[str, Any]]) -> None:
    print(f"Discovered {len(samples)} samples")
    for s in samples:
        print(
            f"- sample_id={s['sample_id']} | "
            f"subgroup={s['subgroup']} | "
            f"h5ad={s['h5ad_path']} | "
            f"image={s['image_path']}"
        )
