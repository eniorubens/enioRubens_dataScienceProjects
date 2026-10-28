"""Testes mínimos dos contratos de dados, EDA e separação do baseline."""

from pathlib import Path
import sys
import tempfile
import unittest
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_data import COLUMNS, TARGET, load_adult, quality_profile, sha256, validate_contract  # noqa: E402
from baseline_models import AUDIT_COLUMNS, PREDICTOR_COLUMNS, run_baseline  # noqa: E402
from eda_report import write_eda_outputs  # noqa: E402
from group_audit_plots import build_group_audit_svgs  # noqa: E402
from hyperparameter_tuning import HGB_TUNING_CANDIDATES  # noqa: E402
from practical_application import group_calibration_summary  # noqa: E402
from threshold_plots import build_threshold_metric_svgs  # noqa: E402
from threshold_selection import (  # noqa: E402
    make_group_aware_partitions,
    official_split_overlap_diagnostic,
    run_threshold_selection,
    select_threshold,
)


class ProjectContractTests(unittest.TestCase):
    """Protege invariantes que não devem depender do notebook."""

    @classmethod
    def setUpClass(cls):
        raw_dir = ROOT / "data" / "raw" / "adult_uci"
        if not (raw_dir / "adult.data").exists():
            raise unittest.SkipTest(
                "UCI Adult is not versioned; download it into data/raw/adult_uci/ (see README, seção Reprodução)"
            )
        cls.frame = load_adult(raw_dir)

    def test_dataset_contract(self):
        self.assertEqual(self.frame.shape, (48842, 16))
        self.assertEqual(self.frame[TARGET].value_counts().to_dict(), {"<=50K": 37155, ">50K": 11687})
        self.assertEqual(self.frame["split"].value_counts().to_dict(), {"train_official": 32561, "test_official": 16281})
        self.assertEqual(list(self.frame.columns), COLUMNS + ["split"])
        profile = quality_profile(self.frame)
        self.assertEqual(profile["duplicates_without_split"], 52)
        self.assertEqual(profile["duplicate_policy_status"], "adopted")
        self.assertEqual(profile["duplicate_identity"], "exact_model_predictor_signature_with_missing_marker; target_excluded")
        self.assertEqual(profile["missing_by_column"]["occupation"], 2809)
        self.assertEqual(profile["missing_by_column"]["workclass"], 2799)
        self.assertEqual(profile["missing_by_column"]["native_country"], 857)
        self.assertEqual(profile["audit_cardinality"], {"sex": 2, "race": 5})
        self.assertEqual(
            sha256(ROOT / "data" / "raw" / "adult.zip"),
            "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
        )

    def test_loader_rejects_semantic_contract_break(self):
        invalid = self.frame.copy()
        invalid.loc[invalid.index[0], TARGET] = "unexpected"
        with self.assertRaises(ValueError):
            validate_contract(invalid)

    def test_audit_attributes_are_not_baseline_predictors(self):
        self.assertTrue(set(AUDIT_COLUMNS).isdisjoint(PREDICTOR_COLUMNS))
        self.assertEqual(len(PREDICTOR_COLUMNS), len(COLUMNS) - 1 - len(AUDIT_COLUMNS))

    def test_eda_outputs_are_reproducible_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_eda_outputs(self.frame, directory)
            expected = {
                "feature_overview",
                "categorical_target_rates",
                "numeric_by_target",
                "split_comparison",
                "conditional_target_rates",
                "category_risk_diagnostics",
                "split_distribution_drift",
            }
            self.assertEqual(set(outputs), expected)
            for path in outputs.values():
                self.assertTrue(Path(path).exists())

            conditional = pd.read_csv(outputs["conditional_target_rates"])
            self.assertEqual(set(conditional["audit_column"]), {"sex", "race"})
            self.assertTrue((conditional["count"] > 0).all())

            diagnostics = pd.read_csv(outputs["category_risk_diagnostics"])
            self.assertTrue(diagnostics["is_rare"].any())
            self.assertTrue(diagnostics["sex_proxy_flag"].any())
            self.assertTrue(diagnostics["race_proxy_flag"].any())

            drift = pd.read_csv(outputs["split_distribution_drift"])
            self.assertEqual(set(drift["comparison"]), {"categorical", "numeric"})

    def test_validation_contract_keeps_test_reserved_and_marks_support(self):
        validation_dir = ROOT / "data" / "processed" / "validation"
        contract = json.loads(
            (validation_dir / "validation_split_contract.json").read_text(encoding="utf-8")
        )
        self.assertFalse(contract["test_used"])
        self.assertTrue(contract["thresholds_not_selected"])
        self.assertTrue(contract["calibration_is_diagnostic_only"])
        self.assertEqual(contract["official_test_rows_reserved"], 16281)
        self.assertIn("duplicate_keys_crossing_fit_validation", contract)
        self.assertTrue(contract["protected_attributes_excluded_but_proxies_retained"])

        group_metrics = pd.read_csv(validation_dir / "validation_group_metrics.csv")
        self.assertIn("low_support", group_metrics.columns)
        self.assertTrue(group_metrics["low_support"].any())
        self.assertGreater(group_metrics["threshold"].nunique(), 1)

    def test_baseline_pipeline_executes_and_contract_is_explicit(self):
        comparison, contract = run_baseline(self.frame)
        self.assertEqual(set(comparison["model"]), {
            "dummy_prior",
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
        })
        self.assertTrue(comparison.drop(columns=["model"]).notna().all().all())
        self.assertEqual(contract["fit_rows"], 26048)
        self.assertEqual(contract["calibration_rows"], 6513)
        self.assertEqual(contract["validation_rows"], 6513)
        self.assertFalse(contract["test_used"])
        self.assertEqual(contract["duplicate_policy_status"], "adopted")
        self.assertEqual(contract["source_rows_removed"], 0)
        self.assertTrue(contract["protected_attributes_excluded_but_proxies_retained"])
        self.assertIn("preprocessing_contract", contract)

    def test_threshold_plot_logic_is_reusable(self):
        validation_dir = ROOT / "data" / "processed" / "validation"
        threshold_metrics = pd.read_csv(validation_dir / "validation_threshold_metrics.csv")
        threshold_metrics = threshold_metrics[
            threshold_metrics["model"].eq("hist_gradient_boosting")
        ].sort_values("threshold")
        labels = {
            "precision": "Precision",
            "recall_tpr": "Recall / TPR",
            "f1": "F1",
            "selection_rate": "Selection rate",
        }
        svgs = build_threshold_metric_svgs(
            threshold_metrics, labels, "HistGradientBoosting"
        )
        self.assertEqual(set(svgs), set(labels))
        self.assertTrue(all(svg.startswith("<svg") for svg in svgs.values()))
        self.assertTrue(all("rotate(90" in svg for svg in svgs.values()))

    def test_group_wilson_plot_logic_is_reusable(self):
        validation_dir = ROOT / "data" / "processed" / "validation"
        group_metrics = pd.read_csv(validation_dir / "validation_group_metrics.csv")
        labels = {
            "selection_rate": "Selection rate",
            "tpr": "TPR",
            "fpr": "FPR",
            "fnr": "FNR",
            "ppv": "PPV",
        }
        svgs = build_group_audit_svgs(
            group_metrics, labels, "hist_gradient_boosting", 0.5
        )
        self.assertEqual(set(svgs), {"marginal", "intersection"})
        self.assertEqual(len(svgs["marginal"]), 10)
        self.assertEqual(len(svgs["intersection"]), 5)
        all_svgs = list(svgs["marginal"].values()) + list(svgs["intersection"].values())
        self.assertTrue(all(svg.startswith("<svg") for svg in all_svgs))

    def test_executive_report_bar_axes_match_bar_scale(self):
        figures_dir = ROOT / "site" / "assets" / "figures"
        baseline_svg = (figures_dir / "baseline_comparison.svg").read_text(
            encoding="utf-8"
        )
        tuning_svg = (figures_dir / "tuning_pr_auc.svg").read_text(encoding="utf-8")
        self.assertIn(">0.70<", baseline_svg)
        self.assertIn(">0.94<", baseline_svg)
        self.assertNotIn(">0.0<", baseline_svg)
        self.assertIn(">0.810<", tuning_svg)
        self.assertIn(">0.820<", tuning_svg)

    def test_threshold_selection_contract_is_explicit(self):
        selection_dir = ROOT / "data" / "processed" / "selection"
        contract = json.loads(
            (selection_dir / "threshold_selection_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(contract["group_aware_split"])
        self.assertNotIn("source_deduplication_pending", contract["duplicate_policy"])
        self.assertEqual(contract["duplicate_policy_status"], "adopted")
        self.assertEqual(contract["source_rows_removed"], 0)
        self.assertIn("exact_model_predictor_signature", contract["duplicate_identity"])
        self.assertEqual(len(contract["threshold_grid"]), 81)
        self.assertAlmostEqual(
            contract["threshold_grid"][1] - contract["threshold_grid"][0], 0.01
        )
        self.assertTrue(contract["official_test_overlap_diagnostic_only"])
        self.assertEqual(contract["cross_official_split_predictor_signature_keys"], 26)
        self.assertEqual(contract["cross_official_split_train_rows"], 28)
        self.assertEqual(contract["cross_official_split_test_rows"], 26)
        self.assertEqual(contract["cross_official_split_target_discordant_keys"], 3)
        self.assertFalse(contract["test_used"])
        self.assertTrue(contract["final_test_evaluation_pending"])
        self.assertFalse(contract["optuna_used"])
        self.assertTrue(contract["protected_attributes_excluded_but_proxies_retained"])
        self.assertEqual(contract["fnlwgt_role"], "predictor_not_sample_weight")
        self.assertIn("lowest_threshold_meeting", contract["threshold_rule"])
        self.assertTrue(contract["tuning_used"])
        self.assertEqual(contract["tuned_model_params"]["max_iter"], 250)
        self.assertTrue(contract["risk_gate_passed"])
        self.assertEqual(
            contract["decision_status"],
            "provisional_risk_floors_satisfied",
        )
        self.assertTrue(contract["selection_metrics_are_development_only"])
        self.assertTrue(contract["final_model_refit_policy_pending"])
        self.assertIn(contract["selected_threshold"], contract["threshold_grid"])
        self.assertIn(contract["selected_model"], {
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
        })

        threshold_metrics = pd.read_csv(selection_dir / "selection_threshold_metrics.csv")
        selected = threshold_metrics[
            threshold_metrics["threshold"].eq(contract["selected_threshold"])
        ]
        self.assertEqual(set(selected["model"]), {contract["selected_model"]})
        self.assertTrue(
            selected[["precision", "recall_tpr", "f1"]].notna().all().all()
        )

    def test_hgb_tuning_contract_is_explicit(self):
        tuning_dir = ROOT / "data" / "processed" / "tuning"
        contract = json.loads(
            (tuning_dir / "hgb_tuning_contract.json").read_text(encoding="utf-8")
        )
        metrics = pd.read_csv(tuning_dir / "hgb_tuning_metrics.csv")
        self.assertEqual(contract["tuned_model"], "hist_gradient_boosting")
        self.assertEqual(contract["candidate_count"], len(HGB_TUNING_CANDIDATES))
        self.assertEqual(len(metrics), len(HGB_TUNING_CANDIDATES))
        self.assertTrue(contract["group_aware"])
        self.assertFalse(contract["threshold_used"])
        self.assertFalse(contract["calibration_used"])
        self.assertFalse(contract["test_used"])
        self.assertEqual(
            contract["selected_candidate_id"], metrics.iloc[0]["candidate_id"]
        )
        self.assertEqual(contract["selected_params"]["max_iter"], 250)

    def test_final_test_evaluation_contract_is_frozen_and_single_use(self):
        evaluation_dir = ROOT / "data" / "processed" / "final_evaluation"
        contract = json.loads(
            (evaluation_dir / "final_test_contract.json").read_text(encoding="utf-8")
        )
        metrics = pd.read_csv(evaluation_dir / "final_test_metrics.csv")
        group_metrics = pd.read_csv(evaluation_dir / "final_test_group_metrics.csv")
        sensitivity_metrics = pd.read_csv(
            evaluation_dir / "final_test_sensitivity_metrics.csv"
        )
        sensitivity_group_metrics = pd.read_csv(
            evaluation_dir / "final_test_sensitivity_group_metrics.csv"
        )
        selection_contract = json.loads(
            (ROOT / "data" / "processed" / "selection" / "threshold_selection_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(contract["test_used"])
        self.assertEqual(contract["evaluation_count"], 1)
        self.assertTrue(contract["selection_contract_frozen"])
        self.assertFalse(contract["production_eligible"])
        self.assertEqual(contract["official_test_rows"], 16281)
        self.assertEqual(contract["model"], selection_contract["selected_model"])
        self.assertEqual(contract["threshold"], selection_contract["selected_threshold"])
        self.assertEqual(len(metrics), 1)
        self.assertEqual(len(group_metrics), 85)
        self.assertEqual(len(sensitivity_metrics), 1)
        self.assertEqual(sensitivity_metrics.iloc[0]["rows"], 16255)
        self.assertEqual(len(sensitivity_group_metrics), 85)
        self.assertEqual(contract["duplicate_policy_status"], "adopted")
        self.assertEqual(contract["sensitivity_analysis"]["excluded_test_rows"], 26)
        self.assertEqual(contract["sensitivity_analysis"]["retained_test_rows"], 16255)
        self.assertTrue(metrics[["roc_auc", "pr_auc", "brier"]].notna().all().all())

    def test_practical_case_study_contract_keeps_wilson_scope_collective(self):
        case_dir = ROOT / "data" / "processed" / "case_study"
        contract = json.loads(
            (case_dir / "practical_case_contract.json").read_text(encoding="utf-8")
        )
        case = pd.read_csv(case_dir / "practical_case_observation.csv")
        group_summary = pd.read_csv(case_dir / "practical_group_summary.csv")
        self.assertEqual(contract["source_split"], "test_official")
        self.assertTrue(contract["test_case_study_post_evaluation"])
        self.assertFalse(contract["test_used_for_training_or_selection"])
        self.assertFalse(contract["individual_confidence_interval_provided"])
        self.assertEqual(len(case), 1)
        self.assertNotIn("wilson_ci_low", case.columns)
        self.assertNotIn("wilson_ci_high", case.columns)
        self.assertEqual(len(group_summary), 1)
        self.assertEqual(group_summary.iloc[0]["group"], "Female")
        self.assertGreater(group_summary.iloc[0]["count"], 0)
        self.assertLessEqual(
            group_summary.iloc[0]["wilson_ci_low"],
            group_summary.iloc[0]["wilson_ci_high"],
        )
        self.assertTrue((case["calibrated_probability"] >= 0).all())
        self.assertTrue((case["calibrated_probability"] <= 1).all())
        self.assertTrue((case["threshold"] == 0.31).all())
        self.assertTrue((group_summary["mean_calibrated_probability"] >= 0).all())
        self.assertTrue((group_summary["mean_calibrated_probability"] <= 1).all())
        self.assertTrue(
            (case_dir / "practical_group_calibration.svg")
            .read_text(encoding="utf-8")
            .startswith("<svg")
        )

    def test_practical_group_summary_uses_binomial_wilson_interval(self):
        summary = group_calibration_summary(
            np.array([0.31, 0.32, 0.38, 0.41]),
            pd.Series([1, 0, 1, 0]),
            pd.DataFrame({"sex": ["Female", "Female", "Male", "Female"]}),
            threshold=0.31,
        )
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["reference"], "test_official_same_frozen_evaluation_protocol")
        self.assertLessEqual(summary["wilson_ci_low"], summary["observed_target_rate"])
        self.assertGreaterEqual(summary["wilson_ci_high"], summary["observed_target_rate"])

    def test_group_aware_partitions_do_not_cross_predictor_signatures(self):
        partitions = make_group_aware_partitions(self.frame)
        signatures = partitions["group_signature"]
        fit = set(signatures.loc[partitions["x_fit"].index])
        calibration = set(signatures.loc[partitions["x_calibration"].index])
        selection = set(signatures.loc[partitions["x_selection"].index])
        self.assertTrue(fit.isdisjoint(calibration))
        self.assertTrue(fit.isdisjoint(selection))
        self.assertTrue(calibration.isdisjoint(selection))

    def test_official_split_overlap_diagnostic_is_reproducible(self):
        diagnostic = official_split_overlap_diagnostic(self.frame)
        self.assertEqual(diagnostic, {
            "cross_official_split_predictor_signature_keys": 26,
            "cross_official_split_train_rows": 28,
            "cross_official_split_test_rows": 26,
            "cross_official_split_target_discordant_keys": 3,
        })

    def test_threshold_policy_uses_floor_then_explicit_fallback(self):
        metrics = pd.DataFrame(
            [
                {"threshold": 0.30, "precision": 0.63, "recall_tpr": 0.84, "f1": 0.72},
                {"threshold": 0.35, "precision": 0.66, "recall_tpr": 0.79, "f1": 0.719},
                {"threshold": 0.40, "precision": 0.70, "recall_tpr": 0.75, "f1": 0.725},
            ]
        )
        threshold, reason = select_threshold(
            metrics, recall_floor=0.80, precision_floor=0.65
        )
        self.assertEqual(threshold, 0.40)
        self.assertEqual(reason, "fallback_max_f1_no_threshold_met_floors")

        metrics.loc[1, "recall_tpr"] = 0.81
        threshold, reason = select_threshold(
            metrics, recall_floor=0.80, precision_floor=0.65
        )
        self.assertEqual(threshold, 0.35)
        self.assertEqual(reason, "risk_floors_satisfied")

    def test_threshold_selection_pipeline_executes_end_to_end(self):
        tuning_contract = json.loads(
            (ROOT / "data" / "processed" / "tuning" / "hgb_tuning_contract.json").read_text(
                encoding="utf-8"
            )
        )
        result = run_threshold_selection(
            self.frame, tuned_model_params=tuning_contract["selected_params"]
        )
        contract = result["contract"]
        self.assertEqual(
            (contract["fit_rows"], contract["calibration_rows"], contract["selection_rows"]),
            (19537, 6512, 6512),
        )
        self.assertEqual(contract["selected_model"], "hist_gradient_boosting")
        self.assertEqual(contract["selected_threshold"], 0.31)
        self.assertEqual(
            contract["selection_reason"],
            "risk_floors_satisfied",
        )
        self.assertTrue(contract["risk_gate_passed"])
        self.assertTrue(result["model_metrics"].shape[0] >= 3)
        self.assertEqual(result["threshold_metrics"].shape[0], 81)
        self.assertFalse(result["group_metrics"].empty)
        self.assertTrue(result["group_metrics"]["estimate"].notna().any())

        mutated = self.frame.copy()
        test_rows = mutated.index[mutated["split"].eq("test_official")][:3]
        mutated.loc[test_rows, "age"] = [17, 18, 19]
        mutated_result = run_threshold_selection(
            mutated, tuned_model_params=tuning_contract["selected_params"]
        )
        for key in ("model_metrics", "threshold_metrics", "group_metrics"):
            pd.testing.assert_frame_equal(
                result[key].reset_index(drop=True),
                mutated_result[key].reset_index(drop=True),
                check_dtype=False,
            )
        for key in ("selected_model", "selected_threshold", "selection_reason"):
            self.assertEqual(result["contract"][key], mutated_result["contract"][key])


if __name__ == "__main__":
    unittest.main()
