"""Avaliação final, única e pré-especificada no teste oficial."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adult_data import load_adult
from baseline_models import RANDOM_STATE
from threshold_selection import (
    SigmoidProbabilityCalibrator,
    build_model_pipeline,
    classification_metrics,
    group_metric_rows,
    make_group_aware_partitions,
    model_frame,
    model_group_signature,
)


def run_final_evaluation(
    frame: pd.DataFrame, selection_contract: dict[str, object]
) -> dict[str, object]:
    """Avalia no teste o modelo e threshold congelados no desenvolvimento."""
    if selection_contract["test_used"]:
        raise ValueError("Selection contract already used the official test")
    if not selection_contract["final_test_evaluation_pending"]:
        raise ValueError("Final test evaluation is not pending")

    partitions = make_group_aware_partitions(frame)
    selected_model = str(selection_contract["selected_model"])
    selected_threshold = float(selection_contract["selected_threshold"])
    tuned_params = selection_contract.get("tuned_model_params")
    pipeline = build_model_pipeline(selected_model, tuned_params)
    pipeline.fit(partitions["x_fit"], partitions["y_fit"])

    calibration_raw = pipeline.predict_proba(partitions["x_calibration"])[:, 1]
    calibrator = SigmoidProbabilityCalibrator().fit(
        calibration_raw, partitions["y_calibration"]
    )
    test_x, test_y, test_audit = model_frame(frame, "test_official")
    test_raw = pipeline.predict_proba(test_x)[:, 1]
    test_probabilities = calibrator.predict(test_raw)
    global_metrics = pd.DataFrame([{
        "model": selected_model,
        "split": "test_official",
        "rows": int(len(test_y)),
        "prevalence": float(test_y.mean()),
        **classification_metrics(test_y, test_probabilities, selected_threshold),
    }])
    group_metrics = pd.DataFrame(group_metric_rows(
        test_y, test_probabilities, test_audit, selected_model, selected_threshold
    ))

    train_x, _, _ = model_frame(frame, "train_official")
    train_keys = set(model_group_signature(train_x).tolist())
    test_keys = model_group_signature(test_x)
    overlap_mask = test_keys.isin(train_keys).to_numpy()
    sensitivity_mask = ~overlap_mask
    sensitivity_y = test_y.loc[test_y.index[sensitivity_mask]]
    sensitivity_audit = test_audit.loc[sensitivity_y.index]
    sensitivity_probabilities = test_probabilities[sensitivity_mask]
    sensitivity_global_metrics = pd.DataFrame([{
        "model": selected_model,
        "split": "test_official_excluding_train_overlap",
        "rows": int(len(sensitivity_y)),
        "prevalence": float(sensitivity_y.mean()),
        **classification_metrics(
            sensitivity_y, sensitivity_probabilities, selected_threshold
        ),
    }])
    sensitivity_group_metrics = pd.DataFrame(group_metric_rows(
        sensitivity_y,
        sensitivity_probabilities,
        sensitivity_audit,
        selected_model,
        selected_threshold,
    ))
    sensitivity_contract = {
        "performed": True,
        "method": "exclude_test_rows_whose_exact_predictor_signature_occurs_in_official_train",
        "identity": "exact_model_predictor_signature_with_missing_marker; target_excluded",
        "excluded_test_rows": int(overlap_mask.sum()),
        "retained_test_rows": int(sensitivity_mask.sum()),
        "retained_source_rows": True,
    }
    contract = {
        "protocol_version": "final-test-evaluation-mvp-1",
        "model": selected_model,
        "threshold": selected_threshold,
        "tuned_model_params": tuned_params,
        "fit_rows": int(len(partitions["y_fit"])),
        "calibration_rows": int(len(partitions["y_calibration"])),
        "selection_rows": int(len(partitions["y_selection"])),
        "official_test_rows": int(len(test_y)),
        "random_state": RANDOM_STATE,
        "calibrator": "sigmoid_platt_on_calibration_holdout",
        "test_used": True,
        "evaluation_count": 1,
        "final_model_refit": False,
        "selection_contract_frozen": True,
        "official_test_overlap_diagnostic_only": True,
        "duplicate_policy": "exact_predictor_signature_groups_adopted; source_rows_retained; official_train_test_overlap_reported_as_diagnostic",
        "duplicate_policy_status": "adopted",
        "duplicate_identity": "exact_model_predictor_signature_with_missing_marker; target_excluded",
        "source_rows_removed": 0,
        "cross_official_split_predictor_signature_keys": selection_contract[
            "cross_official_split_predictor_signature_keys"
        ],
        "cross_official_split_train_rows": selection_contract[
            "cross_official_split_train_rows"
        ],
        "cross_official_split_test_rows": selection_contract[
            "cross_official_split_test_rows"
        ],
        "cross_official_split_target_discordant_keys": selection_contract[
            "cross_official_split_target_discordant_keys"
        ],
        "sensitivity_analysis": sensitivity_contract,
        "production_eligible": False,
    }
    return {
        "global_metrics": global_metrics,
        "group_metrics": group_metrics,
        "sensitivity_global_metrics": sensitivity_global_metrics,
        "sensitivity_group_metrics": sensitivity_group_metrics,
        "contract": contract,
    }


def write_final_evaluation_outputs(
    frame: pd.DataFrame,
    selection_contract_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Executa e salva a avaliação final pré-especificada."""
    selection_contract_path = Path(selection_contract_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_contract = json.loads(
        selection_contract_path.read_text(encoding="utf-8")
    )
    results = run_final_evaluation(frame, selection_contract)
    outputs = {
        "global_metrics": output_dir / "final_test_metrics.csv",
        "group_metrics": output_dir / "final_test_group_metrics.csv",
        "sensitivity_global_metrics": output_dir / "final_test_sensitivity_metrics.csv",
        "sensitivity_group_metrics": output_dir / "final_test_sensitivity_group_metrics.csv",
        "contract": output_dir / "final_test_contract.json",
    }
    results["global_metrics"].to_csv(
        outputs["global_metrics"], index=False, encoding="utf-8"
    )
    results["group_metrics"].to_csv(
        outputs["group_metrics"], index=False, encoding="utf-8"
    )
    results["sensitivity_global_metrics"].to_csv(
        outputs["sensitivity_global_metrics"], index=False, encoding="utf-8"
    )
    results["sensitivity_group_metrics"].to_csv(
        outputs["sensitivity_group_metrics"], index=False, encoding="utf-8"
    )
    outputs["contract"].write_text(
        json.dumps(results["contract"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return outputs


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = load_adult(root / "data" / "raw" / "adult_uci")
    paths = write_final_evaluation_outputs(
        data,
        root / "data" / "processed" / "selection" / "threshold_selection_contract.json",
        root / "data" / "processed" / "final_evaluation",
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
