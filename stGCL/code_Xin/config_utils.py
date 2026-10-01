from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _resolve_yaml_by_name_or_field(directory: Path, name: str, field: str) -> Path:
    exact = directory / f"{name}.yaml"
    if exact.exists():
        return exact

    lower_name = name.lower()
    for path in sorted(directory.glob("*.yaml")):
        if path.stem.lower() == lower_name:
            return path

    for path in sorted(directory.glob("*.yaml")):
        data = load_yaml(path)
        if str(data.get(field, "")).lower() == lower_name:
            return path

    raise FileNotFoundError(
        f"Could not find YAML for {field}={name!r} under {directory}"
    )


def _resolve_optional_yaml_by_name_or_field(
    directory: Path,
    name: str,
    field: str,
) -> Path | None:
    try:
        return _resolve_yaml_by_name_or_field(directory, name, field)
    except FileNotFoundError:
        return None


def _resolve_child_dir_case_insensitive(parent: Path, name: str) -> Path:
    exact = parent / name
    if exact.exists():
        return exact

    lower_name = name.lower()
    for path in sorted(parent.iterdir()):
        if path.is_dir() and path.name.lower() == lower_name:
            return path

    return exact


def resolve_cohort_config_path(project_root: Path, cohort_name: str) -> Path:
    return _resolve_yaml_by_name_or_field(
        project_root / "configs" / "cohorts",
        cohort_name,
        "cohort_name",
    )


def load_pipeline_config(
    project_root: Path,
    method_name: str,
    cohort_name: str,
) -> Dict[str, Any]:
    method_key = method_name.lower()

    cohort_cfg_path = resolve_cohort_config_path(project_root, cohort_name)
    method_cfg_path = _resolve_yaml_by_name_or_field(
        project_root / "configs" / "methods",
        method_key,
        "method_name",
    )
    exp_dir = _resolve_child_dir_case_insensitive(
        project_root / "configs" / "experiments",
        method_key,
    )
    exp_cfg_path = _resolve_optional_yaml_by_name_or_field(
        exp_dir,
        cohort_name,
        "cohort_name",
    )

    cohort_cfg = load_yaml(cohort_cfg_path)
    method_cfg = load_yaml(method_cfg_path)

    final_cfg = deep_update(method_cfg, {"cohort": cohort_cfg})

    if exp_cfg_path is not None:
        exp_cfg = load_yaml(exp_cfg_path)
        overrides = exp_cfg.get("overrides", {})
        final_cfg = deep_update(final_cfg, overrides)
        final_cfg["experiment"] = exp_cfg
    else:
        final_cfg["experiment"] = {}

    return final_cfg
