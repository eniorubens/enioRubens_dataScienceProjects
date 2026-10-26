"""Generate artifact-backed SVG figures for the executive HTML report.

This module only reads materialized CSV files. It does not retrain models or
recompute metrics from raw data. SVG is written directly so the report does
not depend on a plotting backend being available in the runtime environment.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


COLORS = {"blue": "#2563eb", "green": "#16805b", "amber": "#a66300", "purple": "#7c4d9e", "muted": "#9aa8ba", "ink": "#14263e", "grid": "#e6ebf2"}


def _artifact(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Required report artifact not found: {path}")
    return path


def _svg(name: str, width: int, height: int, content: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = f'''<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">
<title id="title">{escape(name)}</title><desc id="desc">Generated from materialized project artifacts.</desc>
<rect width="100%" height="100%" fill="white"/>{content}</svg>'''
    path = output_dir / f"{name}.svg"
    path.write_text(document, encoding="utf-8")
    return path


def _text(x: float, y: float, value: str, size: int = 12, anchor: str = "start", weight: str = "normal", fill: str = COLORS["ink"], transform: str = "") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}px" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{transform}>{escape(str(value))}</text>'


def _frame(title: str, width: int, height: int, left: int = 62, right: int = 24, top: int = 52, bottom: int = 70) -> tuple[str, dict[str, float]]:
    x0, x1, y0, y1 = left, width - right, height - bottom, top
    body = _text(left, 25, title, 15, weight="bold")
    body += f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#dce3ec"/><line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="#dce3ec"/>'
    return body, {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def _grid(frame: dict[str, float], width: int, y_max: float = 1.0, y_min: float = 0.0, ticks: int = 5) -> str:
    body = ""
    span = y_max - y_min
    decimals = 1 if span >= 0.5 else 2 if span >= 0.1 else 3
    for index in range(ticks + 1):
        value = y_min + (y_max - y_min) * index / ticks
        y = frame["y0"] - (frame["y0"] - frame["y1"]) * index / ticks
        body += f'<line x1="{frame["x0"]}" y1="{y:.1f}" x2="{frame["x1"]}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>'
        body += _text(frame["x0"] - 8, y + 4, f"{value:.{decimals}f}", 10, anchor="end", fill="#526276")
    return body


def _line_chart(output_dir: Path, filename: str, title: str, x_values: list[float], series: list[tuple[str, list[float], str]], y_max: float = 1.0, x_label: str = "", y_label: str = "") -> Path:
    width, height = 760, 410
    body, frame = _frame(title, width, height)
    body += _grid(frame, width, y_max)
    x_span = frame["x1"] - frame["x0"]
    y_span = frame["y0"] - frame["y1"]
    def sx(value: float) -> float:
        return frame["x0"] + (value - min(x_values)) / (max(x_values) - min(x_values)) * x_span
    def sy(value: float) -> float:
        return frame["y0"] - value / y_max * y_span
    for label, values, color in series:
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, values))
        body += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
    for x in [min(x_values), (min(x_values) + max(x_values)) / 2, max(x_values)]:
        body += _text(sx(x), frame["y0"] + 22, f"{x:.2f}", 10, anchor="middle", fill="#526276")
    if x_label:
        body += _text((frame["x0"] + frame["x1"]) / 2, height - 10, x_label, 11, anchor="middle", fill="#526276")
    if y_label:
        body += _text(15, (frame["y0"] + frame["y1"]) / 2, y_label, 11, anchor="middle", fill="#526276", transform=f' transform="rotate(-90 15 {(frame["y0"] + frame["y1"]) / 2:.1f}"')
    legend_x = frame["x0"]
    for label, _, color in series:
        body += f'<line x1="{legend_x}" y1="40" x2="{legend_x + 20}" y2="40" stroke="{color}" stroke-width="2"/>'
        body += _text(legend_x + 25, 44, label, 10, fill="#526276")
        legend_x += 115 + len(label) * 2.2
    return _svg(filename, width, height, body, output_dir)


def _bar_chart(output_dir: Path, filename: str, title: str, labels: list[str], values: list[float], colors: list[str], y_max: float = 1.0, y_min: float = 0.0, value_format: str = ".3f") -> Path:
    width, height = 760, 390
    body, frame = _frame(title, width, height, bottom=90)
    body += _grid(frame, width, y_max, y_min)
    span = frame["x1"] - frame["x0"]
    step = span / max(len(labels), 1)
    bar_width = step * 0.62
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = frame["x0"] + index * step + (step - bar_width) / 2
        y = frame["y0"] - (value - y_min) / (y_max - y_min) * (frame["y0"] - frame["y1"])
        body += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{frame["y0"] - y:.1f}" fill="{color}" rx="3"/>'
        body += _text(x + bar_width / 2, y - 7, format(value, value_format), 10, anchor="middle")
        body += _text(x + bar_width / 2, frame["y0"] + 18, label.replace("_", "\n"), 10, anchor="middle", fill="#526276")
    return _svg(filename, width, height, body, output_dir)


def _data_overview(root: Path, output_dir: Path) -> Path:
    data = pd.read_csv(_artifact(root, "data/processed/eda_initial/split_comparison.csv"))
    labels = ["train_official", "test_official"]
    values = [float(data.loc[data["split"] == label, "rows"].iloc[0]) / 1000 for label in labels]
    return _bar_chart(output_dir, "data_overview", "Official split size (thousands of rows)", labels, values, [COLORS["blue"], COLORS["green"]], y_max=36, value_format=".1f")


def _baseline(root: Path, output_dir: Path) -> Path:
    data = pd.read_csv(_artifact(root, "data/processed/baseline/baseline_model_comparison.csv"))
    data = data[data["model"] != "dummy_prior"].copy()
    labels = [str(value) for value in data["model"]]
    return _bar_chart(output_dir, "baseline_comparison", "Baseline ROC-AUC", labels, data["roc_auc"].tolist(), [COLORS["blue"]] * len(labels), y_min=0.7, y_max=1, value_format=".3f")


def _calibration(root: Path, output_dir: Path) -> Path:
    data = pd.read_csv(_artifact(root, "data/processed/validation/validation_calibration_bins.csv"))
    data = data[data["model"] == "hist_gradient_boosting"].copy()
    return _line_chart(output_dir, "calibration_reliability", "Calibration diagnostic · validation holdout", data["mean_predicted_probability"].tolist(), [("observed rate", data["observed_rate"].tolist(), COLORS["blue"])], x_label="Mean predicted probability", y_label="Observed rate")


def _tuning(root: Path, output_dir: Path) -> Path:
    data = pd.read_csv(_artifact(root, "data/processed/tuning/hgb_tuning_metrics.csv")).sort_values("pr_auc")
    return _bar_chart(output_dir, "tuning_pr_auc", "Fine-tuning candidates · OOF PR-AUC", data["candidate_id"].tolist(), data["pr_auc"].tolist(), [COLORS["blue"] if value == data["pr_auc"].max() else COLORS["muted"] for value in data["pr_auc"]], y_min=0.81, y_max=0.82, value_format=".5f")


def _threshold(root: Path, output_dir: Path) -> Path:
    data = pd.read_csv(_artifact(root, "data/processed/selection/selection_threshold_metrics.csv"))
    return _line_chart(output_dir, "threshold_tradeoff", "Threshold trade-off · selection partition", data["threshold"].tolist(), [("precision", data["precision"].tolist(), COLORS["blue"]), ("recall / TPR", data["recall_tpr"].tolist(), COLORS["green"]), ("F1", data["f1"].tolist(), COLORS["purple"]), ("selection rate", data["selection_rate"].tolist(), COLORS["amber"])], x_label="Threshold", y_label="Metric")


def _final_metrics(root: Path, output_dir: Path) -> Path:
    data = pd.read_csv(_artifact(root, "data/processed/final_evaluation/final_test_metrics.csv")).iloc[0]
    labels = ["ROC-AUC", "PR-AUC", "precision", "recall / TPR", "F1", "selection"]
    values = [data["roc_auc"], data["pr_auc"], data["precision"], data["recall_tpr"], data["f1"], data["selection_rate"]]
    return _bar_chart(output_dir, "final_metrics", f"Official test · threshold {data['threshold']:.2f}", labels, [float(value) for value in values], [COLORS["blue"], COLORS["blue"], COLORS["green"], COLORS["green"], COLORS["purple"], COLORS["amber"]], value_format=".3f")


def _group_audit(root: Path, output_dir: Path) -> Path:
    data = pd.read_csv(_artifact(root, "data/processed/final_evaluation/final_test_group_metrics.csv"))
    data = data[(data["metric"].isin(["tpr", "fpr"])) & (data["audit_column"].isin(["sex", "race"]))].copy()
    order = [("sex", "Female"), ("sex", "Male"), ("race", "Amer-Indian-Eskimo"), ("race", "Asian-Pac-Islander"), ("race", "Black"), ("race", "Other"), ("race", "White")]
    width, height = 820, 450
    body, frame = _frame("Group audit · Wilson 95% intervals", width, height, bottom=105)
    body += _grid(frame, width)
    span = frame["x1"] - frame["x0"]
    step = span / len(order)
    for offset, metric, color in [(-0.16, "tpr", COLORS["green"]), (0.16, "fpr", COLORS["amber"])]:
        for index, (audit, group) in enumerate(order):
            row = data[(data["audit_column"] == audit) & (data["group"] == group) & (data["metric"] == metric)].iloc[0]
            value = float(row["estimate"])
            x = frame["x0"] + (index + 0.5) * step + offset * step
            y = frame["y0"] - value * (frame["y0"] - frame["y1"])
            low = frame["y0"] - float(row["ci_low"]) * (frame["y0"] - frame["y1"])
            high = frame["y0"] - float(row["ci_high"]) * (frame["y0"] - frame["y1"])
            body += f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}" stroke="{color}"/><line x1="{x-4:.1f}" y1="{low:.1f}" x2="{x+4:.1f}" y2="{low:.1f}" stroke="{color}"/><line x1="{x-4:.1f}" y1="{high:.1f}" x2="{x+4:.1f}" y2="{high:.1f}" stroke="{color}"/><circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
    for index, (_, group) in enumerate(order):
        x = frame["x0"] + (index + 0.5) * step
        body += _text(x, frame["y0"] + 20, group, 9, anchor="middle", fill="#526276")
    body += '<circle cx="650" cy="35" r="4" fill="#16805b"/>' + _text(660, 39, "TPR", 10, fill="#526276") + '<circle cx="705" cy="35" r="4" fill="#a66300"/>' + _text(715, 39, "FPR", 10, fill="#526276")
    return _svg("group_audit", width, height, body, output_dir)


def write_executive_report_assets(project_root: str | Path) -> list[Path]:
    """Generate all static report figures from materialized project artifacts."""

    root = Path(project_root).resolve()
    output_dir = root / "site" / "assets" / "figures"
    generated = [_data_overview(root, output_dir), _baseline(root, output_dir), _calibration(root, output_dir), _tuning(root, output_dir), _threshold(root, output_dir), _final_metrics(root, output_dir), _group_audit(root, output_dir)]
    return generated


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    generated = write_executive_report_assets(project)
    print(f"Generated {len(generated)} executive report figures in {project / 'site' / 'assets' / 'figures'}")
