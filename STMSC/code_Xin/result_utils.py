from __future__ import annotations

from math import comb
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = [
    "cohort",
    "sampleID",
    "spotID",
    "x",
    "y",
    "method",
    "z",
    "z_pred",
]

PREDICTION_STRING_COLUMNS = [
    "cohort",
    "sampleID",
    "spotID",
    "method",
    "z",
    "z_pred",
    "obs_name",
    "spot_id",
    "sample_id",
    "group_name",
    "ground_truth",
    "DeepST_domain",
    "DeepST_refine_domain",
    "ResST_domain",
    "ResST_refine_domain",
    "SpaConTDS_domain",
    "stlearn_cluster",
    "starfysh_cluster",
    "stGCL_domain",
    "stGCL_refined",
    "STMSC_domain",
    "cluster_key",
]

INVALID_LABELS = {"", "NA", "NaN", "nan", "None", "Filtered", "Unknown", "unknown"}


def clean_label_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def validate_prediction_schema(df: pd.DataFrame, source: str = "predictions") -> None:
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def normalize_prediction_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep identifiers and labels stable across CSV resume and Parquet export."""
    out = df.copy()
    for column in PREDICTION_STRING_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype("string")
    return out


def sort_predictions(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["cohort", "sampleID", "spotID"] if c in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols).reset_index(drop=True)


def require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise ImportError(
            "Parquet export requires pyarrow. Install it in the runtime environment "
            "before running the MOCHA runners."
        ) from e


def write_predictions_tables(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    validate_prediction_schema(df)
    require_parquet_engine()
    outdir.mkdir(parents=True, exist_ok=True)
    df = sort_predictions(normalize_prediction_dtypes(df))
    df.to_csv(outdir / "predictions.csv", index=False)
    df.to_parquet(outdir / "predictions.parquet", index=False)
    return df


def build_performance_df(
    cohort_name: str,
    method_name: str,
    runtime_seconds: float | int | None,
    memory_mb: float | int | None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cohort": cohort_name,
                "method": method_name,
                "runtime": runtime_seconds,
                "memory": memory_mb,
            }
        ]
    )


def write_performance_tables(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    require_parquet_engine()
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "performance.csv", index=False)
    df.to_parquet(outdir / "performance.parquet", index=False)
    return df


def valid_label_mask(df: pd.DataFrame) -> pd.Series:
    z = clean_label_series(df["z"])
    z_pred = clean_label_series(df["z_pred"])
    return z.notna() & z_pred.notna() & ~z.isin(INVALID_LABELS) & ~z_pred.isin(INVALID_LABELS)


def adjusted_rand_index(labels_true: Iterable[str], labels_pred: Iterable[str]) -> float:
    labels_true = pd.Series(labels_true).astype("string")
    labels_pred = pd.Series(labels_pred).astype("string")
    n = len(labels_true)
    if n < 2:
        return float("nan")

    contingency = pd.crosstab(labels_true, labels_pred)
    sum_comb_cells = sum(comb(int(v), 2) for v in contingency.to_numpy().ravel() if v >= 2)
    row_sums = contingency.sum(axis=1).to_numpy()
    col_sums = contingency.sum(axis=0).to_numpy()
    sum_comb_rows = sum(comb(int(v), 2) for v in row_sums if v >= 2)
    sum_comb_cols = sum(comb(int(v), 2) for v in col_sums if v >= 2)
    total_pairs = comb(n, 2)
    expected_index = (sum_comb_rows * sum_comb_cols) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denominator = max_index - expected_index

    if denominator == 0:
        return 1.0 if contingency.shape == (1, 1) else 0.0
    return (sum_comb_cells - expected_index) / denominator


def build_evaluation_summary(
    pred: pd.DataFrame,
    cohort_name: str,
    method_name: str,
) -> pd.DataFrame:
    validate_prediction_schema(pred)

    records: list[dict[str, object]] = []
    evaluated_parts: list[pd.DataFrame] = []

    for sample_id, sample_df in pred.groupby("sampleID", dropna=False, sort=True):
        sample_mask = valid_label_mask(sample_df)
        eval_df = sample_df.loc[sample_mask].copy()
        if eval_df.empty:
            ari = float("nan")
            n_true = 0
            n_pred = 0
        else:
            eval_df["z"] = clean_label_series(eval_df["z"])
            eval_df["z_pred"] = clean_label_series(eval_df["z_pred"])
            ari = adjusted_rand_index(eval_df["z"], eval_df["z_pred"])
            n_true = int(eval_df["z"].nunique(dropna=True))
            n_pred = int(eval_df["z_pred"].nunique(dropna=True))
            evaluated_parts.append(eval_df)

        records.append(
            {
                "cohort": cohort_name,
                "method": method_name,
                "metric": "ARI",
                "scope": "section",
                "sampleID": str(sample_id),
                "value": ari,
                "n_sections": 1,
                "n_spots_total": int(len(sample_df)),
                "n_spots_evaluated": int(len(eval_df)),
                "n_true_labels": n_true,
                "n_pred_labels": n_pred,
                "aggregation": "per_section",
            }
        )

    section_df = pd.DataFrame(records)
    valid_section_df = section_df.dropna(subset=["value"]).copy()
    if valid_section_df.empty:
        return section_df

    eval_all = pd.concat(evaluated_parts, ignore_index=True) if evaluated_parts else pd.DataFrame()
    pooled_ari = adjusted_rand_index(eval_all["z"], eval_all["z_pred"]) if not eval_all.empty else float("nan")
    denom = valid_section_df["n_spots_evaluated"].sum()
    weighted_mean_ari = (
        (valid_section_df["value"] * valid_section_df["n_spots_evaluated"]).sum() / denom
        if denom
        else float("nan")
    )

    aggregate_records = pd.DataFrame(
        [
            {
                "cohort": cohort_name,
                "method": method_name,
                "metric": "ARI_mean",
                "scope": "cohort",
                "sampleID": "ALL",
                "value": valid_section_df["value"].mean(),
                "n_sections": int(len(valid_section_df)),
                "n_spots_total": int(len(pred)),
                "n_spots_evaluated": int(valid_section_df["n_spots_evaluated"].sum()),
                "n_true_labels": int(eval_all["z"].nunique(dropna=True)),
                "n_pred_labels": int(eval_all["z_pred"].nunique(dropna=True)),
                "aggregation": "mean_section_ari_unweighted",
            },
            {
                "cohort": cohort_name,
                "method": method_name,
                "metric": "ARI_weighted_mean",
                "scope": "cohort",
                "sampleID": "ALL",
                "value": weighted_mean_ari,
                "n_sections": int(len(valid_section_df)),
                "n_spots_total": int(len(pred)),
                "n_spots_evaluated": int(valid_section_df["n_spots_evaluated"].sum()),
                "n_true_labels": int(eval_all["z"].nunique(dropna=True)),
                "n_pred_labels": int(eval_all["z_pred"].nunique(dropna=True)),
                "aggregation": "mean_section_ari_weighted_by_spots",
            },
            {
                "cohort": cohort_name,
                "method": method_name,
                "metric": "ARI_pooled",
                "scope": "cohort",
                "sampleID": "ALL",
                "value": pooled_ari,
                "n_sections": int(len(valid_section_df)),
                "n_spots_total": int(len(pred)),
                "n_spots_evaluated": int(len(eval_all)),
                "n_true_labels": int(eval_all["z"].nunique(dropna=True)),
                "n_pred_labels": int(eval_all["z_pred"].nunique(dropna=True)),
                "aggregation": "pooled_spots",
            },
        ]
    )

    return pd.concat([section_df, aggregate_records], ignore_index=True)


def save_evaluation_summary(
    pred: pd.DataFrame,
    outdir: Path,
    cohort_name: str,
    method_name: str,
) -> pd.DataFrame:
    summary = build_evaluation_summary(
        pred=pred,
        cohort_name=cohort_name,
        method_name=method_name,
    )
    summary.to_csv(outdir / "evaluation_summary.csv", index=False)
    return summary


def save_true_vs_pred_plots(
    pred: pd.DataFrame,
    outdir: Path,
    cohort_name: str,
    method_name: str,
) -> list[Path]:
    validate_prediction_schema(pred)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError(
            "Plot generation requires matplotlib in the runtime environment."
        ) from e

    plot_df = pred.dropna(subset=["sampleID", "x", "y", "z", "z_pred"]).copy()
    plot_df["sampleID"] = plot_df["sampleID"].astype(str)
    plot_df["z"] = plot_df["z"].astype(str)
    plot_df["z_pred"] = plot_df["z_pred"].astype(str)

    if plot_df.empty:
        return []

    colors = list(plt.cm.tab20.colors)
    if not colors:
        colors = ["#1f77b4"]

    def build_color_map(labels: list[str]) -> dict[str, object]:
        return {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    true_labels = sorted(plot_df["z"].unique())
    pred_labels = sorted(plot_df["z_pred"].unique(), key=lambda x: (not str(x).isdigit(), str(x)))
    true_cmap = build_color_map(true_labels)
    pred_cmap = build_color_map(pred_labels)

    def scatter_panel(ax, df: pd.DataFrame, label_col: str, color_map: dict[str, object], title: str) -> None:
        for label, group in df.groupby(label_col, sort=True):
            key = str(label)
            ax.scatter(
                group["x"],
                group["y"],
                s=9,
                c=[color_map[key]],
                label=key,
                linewidths=0,
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            markerscale=1.8,
        )

    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for sample_id, df in plot_df.groupby("sampleID", sort=True):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        scatter_panel(axes[0], df, "z", true_cmap, f"{sample_id} true label")
        scatter_panel(axes[1], df, "z_pred", pred_cmap, f"{sample_id} prediction")
        fig.suptitle(f"{method_name} / {cohort_name} / {sample_id}", fontsize=13)
        out_path = fig_dir / f"{sample_id}_true_vs_pred.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved
