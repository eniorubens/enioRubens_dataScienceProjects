"""Aplicação prática do modelo final a uma observação e a um grupo.

O módulo separa a probabilidade individual da proporção binomial observada em
um grupo. O intervalo de Wilson é calculado somente para a segunda quantidade.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adult_data import AUDIT_COLUMNS, TARGET, load_adult
from baseline_models import PREDICTOR_COLUMNS, RANDOM_STATE
from threshold_selection import (
    SigmoidProbabilityCalibrator,
    build_model_pipeline,
    make_group_aware_partitions,
    model_frame,
)
from validation_audit import wilson_interval


DEFAULT_GROUP_COLUMN = "sex"
DEFAULT_GROUP = "Female"


def group_calibration_summary(
    probabilities: np.ndarray,
    target: pd.Series,
    audit: pd.DataFrame,
    threshold: float,
    audit_column: str = DEFAULT_GROUP_COLUMN,
    group: str = DEFAULT_GROUP,
) -> dict[str, object]:
    """Compara score médio e proporção observada de um grupo.

    O intervalo de Wilson pertence à proporção observada do alvo no grupo.
    A média das probabilidades é reportada como comparação agregada, não como
    uma quantidade que receba intervalo de Wilson.
    """
    values = np.asarray(probabilities, dtype=float)
    target_values = np.asarray(target, dtype=int)
    if len(values) != len(target_values) or len(values) != len(audit):
        raise ValueError("Probabilities, target and audit must have the same length")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a non-empty finite array")
    if audit_column not in audit.columns:
        raise ValueError(f"Unknown audit column: {audit_column}")

    mask = audit[audit_column].astype(str).eq(group).to_numpy()
    count = int(mask.sum())
    if count == 0:
        raise ValueError(f"Group has no observations: {audit_column}={group}")
    group_target = target_values[mask]
    group_probabilities = values[mask]
    positive_count = int(group_target.sum())
    observed_rate = float(group_target.mean())
    wilson_low, wilson_high = wilson_interval(positive_count, count)
    predictions = (group_probabilities >= threshold).astype(int)
    selection_count = int(predictions.sum())
    selection_rate = float(predictions.mean())
    selection_ci_low, selection_ci_high = wilson_interval(selection_count, count)
    return {
        "audit_column": audit_column,
        "group": group,
        "count": count,
        "positive_count": positive_count,
        "observed_target_rate": observed_rate,
        "wilson_ci_low": wilson_low,
        "wilson_ci_high": wilson_high,
        "wilson_confidence": 0.95,
        "mean_calibrated_probability": float(group_probabilities.mean()),
        "mean_probability_inside_wilson_interval": bool(
            wilson_low <= group_probabilities.mean() <= wilson_high
        ),
        "threshold": threshold,
        "selection_count": selection_count,
        "selection_rate": selection_rate,
        "selection_wilson_ci_low": selection_ci_low,
        "selection_wilson_ci_high": selection_ci_high,
        "reference": "test_official_same_frozen_evaluation_protocol",
        "wilson_scope": "observed binary target proportion in the group",
    }


def _fit_final_protocol(
    frame: pd.DataFrame, selection_contract: dict[str, object]
) -> tuple[object, SigmoidProbabilityCalibrator]:
    """Reproduz o protocolo congelado para gerar previsões ilustrativas."""
    partitions = make_group_aware_partitions(frame)
    model_name = str(selection_contract["selected_model"])
    tuned_params = selection_contract.get("tuned_model_params")
    pipeline = build_model_pipeline(model_name, tuned_params)
    pipeline.fit(partitions["x_fit"], partitions["y_fit"])
    calibration_raw = pipeline.predict_proba(partitions["x_calibration"])[:, 1]
    calibrator = SigmoidProbabilityCalibrator().fit(
        calibration_raw, partitions["y_calibration"]
    )
    return pipeline, calibrator


def run_practical_case_study(
    frame: pd.DataFrame,
    selection_contract: dict[str, object],
    row_selector: str = "closest_to_threshold",
    group_column: str = DEFAULT_GROUP_COLUMN,
    group: str = DEFAULT_GROUP,
) -> dict[str, object]:
    """Aplica o modelo congelado a uma linha e resume um grupo do teste."""
    if selection_contract["test_used"] is not False:
        raise ValueError("The selection contract must keep the official test unused")
    if row_selector != "closest_to_threshold":
        raise ValueError("Only the deterministic closest_to_threshold selector is supported")

    pipeline, calibrator = _fit_final_protocol(frame, selection_contract)
    test_x, test_y, test_audit = model_frame(frame, "test_official")
    raw_probabilities = pipeline.predict_proba(test_x)[:, 1]
    probabilities = calibrator.predict(raw_probabilities)
    threshold = float(selection_contract["selected_threshold"])
    selected_position = int(np.argmin(np.abs(probabilities - threshold)))
    selected_index = test_x.index[selected_position]
    selected_probability = float(probabilities[selected_position])
    predicted_positive = int(selected_probability >= threshold)
    actual_positive = int(test_y.loc[selected_index])

    source_row = frame.loc[selected_index]
    case = source_row[PREDICTOR_COLUMNS + AUDIT_COLUMNS].to_dict()
    case.update(
        {
            "source_split": "test_official",
            "source_row_index": int(selected_index),
            "model": str(selection_contract["selected_model"]),
            "threshold": threshold,
            "raw_probability": float(raw_probabilities[selected_position]),
            "calibrated_probability": selected_probability,
            "predicted_label": ">50K" if predicted_positive else "<=50K",
            "actual_label": ">50K" if actual_positive else "<=50K",
            "prediction_correct": bool(predicted_positive == actual_positive),
        }
    )
    group_summary = group_calibration_summary(
        probabilities,
        test_y,
        test_audit,
        threshold,
        audit_column=group_column,
        group=group,
    )
    contract = {
        "protocol_version": "practical-model-application-mvp-2",
        "model": str(selection_contract["selected_model"]),
        "threshold": threshold,
        "calibrator": "sigmoid_platt_on_calibration_holdout",
        "row_selector": row_selector,
        "source_split": "test_official",
        "source_row_index": int(selected_index),
        "test_case_study_post_evaluation": True,
        "test_used_for_training_or_selection": False,
        "wilson_confidence": 0.95,
        "wilson_scope": "observed binary target proportion in the selected audit group",
        "individual_confidence_interval_provided": False,
        "target": TARGET,
        "target_interpretation": "binary income label: >50K versus <=50K",
        "group_reference": {
            "audit_column": group_column,
            "group": group,
            "source": "test_official_same_frozen_evaluation_protocol",
        },
        "random_state": RANDOM_STATE,
        "case": case,
        "group_summary": group_summary,
    }
    return {
        "case": pd.DataFrame([case]),
        "group_summary": pd.DataFrame([group_summary]),
        "contract": contract,
    }


def _svg_text(
    x: float, y: float, text: str, size: int = 14, anchor: str = "start"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" fill="#243247">'
        f"{html.escape(text)}</text>"
    )


def build_group_calibration_svg(group_summary: dict[str, object]) -> str:
    """Constrói gráfico do score médio e Wilson do grupo, sem backend gráfico."""
    width, height = 860, 340
    left, right = 200, 800
    scale = lambda value: left + float(value) * (right - left)
    observed = float(group_summary["observed_target_rate"])
    mean_probability = float(group_summary["mean_calibrated_probability"])
    ci_low = float(group_summary["wilson_ci_low"])
    ci_high = float(group_summary["wilson_ci_high"])
    threshold = float(group_summary["threshold"])
    group_label = f'{group_summary["audit_column"]}={group_summary["group"]}'
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 30, f"Calibração agregada do grupo {group_label}", 18, "middle"),
        _svg_text(left, 72, "0,0", 12, "middle"),
        _svg_text((left + right) / 2, 72, "0,5", 12, "middle"),
        _svg_text(right, 72, "1,0", 12, "middle"),
        f'<line x1="{left}" y1="90" x2="{right}" y2="90" stroke="#cbd5e1" stroke-width="2"/>',
        f'<line x1="{scale(threshold):.1f}" y1="82" x2="{scale(threshold):.1f}" y2="260" stroke="#d97706" stroke-width="2" stroke-dasharray="6 5"/>',
        _svg_text(scale(threshold), 282, f"threshold {threshold:.2f}", 12, "middle"),
        _svg_text(20, 145, "Média prevista", 14),
        _svg_text(20, 216, "Taxa observada", 14),
        _svg_text(20, 235, "+ IC Wilson", 14),
        f'<circle cx="{scale(mean_probability):.1f}" cy="150" r="8" fill="#173f5f"/>',
        _svg_text(scale(mean_probability), 125, f"média = {mean_probability:.3f}", 13, "middle"),
        f'<line x1="{scale(ci_low):.1f}" y1="220" x2="{scale(ci_high):.1f}" y2="220" stroke="#2a9d8f" stroke-width="5"/>',
        f'<line x1="{scale(ci_low):.1f}" y1="208" x2="{scale(ci_low):.1f}" y2="232" stroke="#2a9d8f" stroke-width="2"/>',
        f'<line x1="{scale(ci_high):.1f}" y1="208" x2="{scale(ci_high):.1f}" y2="232" stroke="#2a9d8f" stroke-width="2"/>',
        f'<circle cx="{scale(observed):.1f}" cy="220" r="7" fill="#2a9d8f"/>',
        _svg_text(scale(observed), 195, f"taxa = {observed:.3f}", 13, "middle"),
        _svg_text(width / 2, 315, "Wilson: proporção observada do alvo no grupo; não é IC da média prevista nem individual.", 12, "middle"),
        "</svg>",
    ]
    return "".join(elements)


def write_practical_case_outputs(
    frame: pd.DataFrame,
    selection_contract_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Gera a observação, o resumo de grupo, o gráfico e o contrato."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_contract = json.loads(
        Path(selection_contract_path).read_text(encoding="utf-8")
    )
    results = run_practical_case_study(frame, selection_contract)
    outputs = {
        "case": output_dir / "practical_case_observation.csv",
        "group_summary": output_dir / "practical_group_summary.csv",
        "svg": output_dir / "practical_group_calibration.svg",
        "contract": output_dir / "practical_case_contract.json",
    }
    results["case"].to_csv(outputs["case"], index=False, encoding="utf-8")
    results["group_summary"].to_csv(
        outputs["group_summary"], index=False, encoding="utf-8"
    )
    outputs["svg"].write_text(
        build_group_calibration_svg(results["contract"]["group_summary"]),
        encoding="utf-8",
    )
    outputs["contract"].write_text(
        json.dumps(results["contract"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return outputs


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = load_adult(root / "data" / "raw" / "adult_uci")
    paths = write_practical_case_outputs(
        data,
        root / "data" / "processed" / "selection" / "threshold_selection_contract.json",
        root / "data" / "processed" / "case_study",
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
