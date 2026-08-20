"""Check bilingual notebook structure without executing the full dataset."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class NotebookStructureTests(unittest.TestCase):
    """Verify PT-BR and EN-US notebooks remain structurally equivalent."""

    def test_bilingual_notebooks_have_matching_cell_shapes(self) -> None:
        """Keep code-cell structure aligned while allowing translated prose."""

        for stem in (
            "01_Data_Inspection",
            "02_S0_Audit",
            "03_S1_Taxonomy_Dedup",
            "04_S2_Temporal_Protocol",
            "05_S3_Baseline_Learning_Curve",
            "06_S4_Error_Representation_Challenge",
            "07_S5_Estimator_Benchmark",
            "08_S6_Calibrated_Classical_Challenge",
            "09_S7_Frozen_Model_Package",
            "10_S8_Confirmatory_Evaluation",
            "11_V2_Kaggle_Import",
            "12_V2_Frozen_Package",
            "13_V2_Stress_Confirmatory",
        ):
            pt_path = ROOT / "notebooks" / "pt-BR" / f"{stem}_PT.ipynb"
            en_path = ROOT / "notebooks" / "en-US" / f"{stem}_EN.ipynb"
            pt = json.loads(pt_path.read_text(encoding="utf-8"))
            en = json.loads(en_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [cell["cell_type"] for cell in pt["cells"]],
                [cell["cell_type"] for cell in en["cells"]],
            )
            pt_code = [
                cell["source"]
                for cell in pt["cells"]
                if cell["cell_type"] == "code"
            ]
            en_code = [
                cell["source"]
                for cell in en["cells"]
                if cell["cell_type"] == "code"
            ]
            self.assertEqual(pt_code, en_code)

    def test_legacy_notebook_remains_at_root(self) -> None:
        """Keep the executed smoke-test notebook traceable and untouched."""

        legacy = ROOT / "notebooks" / "01_CFPB_Data_Inspection_PT.ipynb"
        self.assertTrue(legacy.exists())

    def test_s2_notebooks_discover_source_path_before_import(self) -> None:
        """Require S2 notebooks to resolve the project root before imports."""

        path = ROOT / "notebooks" / "pt-BR" / "04_S2_Temporal_Protocol_PT.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        path_setup = "sys.path.insert(0, str(SRC_DIR))"
        package_import = "from consumer_complaint_intelligence"
        self.assertIn("PROJECT_ROOT / 'dataset' / 'processed'", code)
        self.assertIn(path_setup, code)
        self.assertLess(code.index(path_setup), code.index(package_import))
        self.assertIn("s2_report.json", code)
        self.assertIn("s2_report_pilot.json", code)

    def test_s3_notebooks_publish_cached_evidence_without_modeling_logic(self) -> None:
        """Require disabled mode, explicit full config, and Polars tables."""

        path = ROOT / "notebooks" / "pt-BR" / (
            "05_S3_Baseline_Learning_Curve_PT.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("RUN_MODE = 'disabled'", code)
        self.assertIn("BaselineConfig(", code)
        self.assertIn("config=baseline_config", code)
        self.assertNotIn("def show_cached_evidence", code)
        self.assertIn(
            "from consumer_complaint_intelligence.s3_reporting import "
            "load_s3_evidence_tables",
            code,
        )
        self.assertIn("display(evidence.curve)", code)
        self.assertIn("display(evidence.per_class)", code)

    def test_s4_notebooks_are_thin_and_disabled_by_default(self) -> None:
        """Require S4 notebooks to delegate execution and preserve the boundary."""

        path = ROOT / "notebooks" / "pt-BR" / (
            "06_S4_Error_Representation_Challenge_PT.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("RUN_MODE = 'disabled'", code)
        self.assertIn("run_s4_smoke", code)
        self.assertIn("run_s4(scientific_cache, artifact, config)", code)
        self.assertIn("load_s4_report_tables", code)
        self.assertNotIn("TfidfVectorizer", code)
        self.assertNotIn("SGDClassifier", code)
        self.assertIn("s4_results.json", code)
        self.assertNotIn("show_cached_evidence", code)
        self.assertNotIn("selected", code)

    def test_s5_notebooks_are_thin_and_disabled_by_default(self) -> None:
        """Require S5 notebooks to delegate execution and keep parity checks."""

        pt_path = ROOT / "notebooks" / "pt-BR" / "07_S5_Estimator_Benchmark_PT.ipynb"
        en_path = ROOT / "notebooks" / "en-US" / "07_S5_Estimator_Benchmark_EN.ipynb"
        pt = json.loads(pt_path.read_text(encoding="utf-8"))
        en = json.loads(en_path.read_text(encoding="utf-8"))
        pt_code_cells = [
            cell["source"] for cell in pt["cells"] if cell["cell_type"] == "code"
        ]
        en_code_cells = [
            cell["source"] for cell in en["cells"] if cell["cell_type"] == "code"
        ]
        self.assertEqual(pt_code_cells, en_code_cells)
        code = "\n".join(
            "".join(cell["source"])
            for cell in pt["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("RUN_MODE = 'disabled'", code)
        self.assertIn("{'disabled', 'smoke', 'full'}", code)
        self.assertIn("run_s5_smoke", code)
        self.assertIn(
            "run_s5(scientific_cache, artifact, config, "
            "reference_artifact_path=s4_reference)",
            code,
        )
        self.assertIn("load_s5_report_tables", code)
        for forbidden in ("TfidfVectorizer", "SGDClassifier", "LinearSVC", "ComplementNB"):
            self.assertNotIn(forbidden, code)
        self.assertIn("s5_results.json", code)
        self.assertIn("deferred_candidates", code)
        self.assertIn("display(evidence.reference_parity)", code)

    def test_s6_notebooks_are_thin_and_disabled_by_default(self) -> None:
        """Require S6 notebooks to publish cached calibrated evidence only."""

        pt_path = ROOT / "notebooks" / "pt-BR" / (
            "08_S6_Calibrated_Classical_Challenge_PT.ipynb"
        )
        en_path = ROOT / "notebooks" / "en-US" / (
            "08_S6_Calibrated_Classical_Challenge_EN.ipynb"
        )
        pt = json.loads(pt_path.read_text(encoding="utf-8"))
        en = json.loads(en_path.read_text(encoding="utf-8"))
        pt_code = [
            cell["source"] for cell in pt["cells"] if cell["cell_type"] == "code"
        ]
        en_code = [
            cell["source"] for cell in en["cells"] if cell["cell_type"] == "code"
        ]
        self.assertEqual(pt_code, en_code)
        code = "\n".join(
            "".join(cell["source"])
            for cell in pt["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("RUN_MODE = 'disabled'", code)
        self.assertIn("{'disabled', 'smoke', 'full'}", code)
        self.assertIn("run_s6_smoke", code)
        self.assertIn("run_s6(scientific_cache, artifact, config)", code)
        self.assertIn("load_s6_report_tables", code)
        self.assertIn(
            "scientific_cache = PROJECT_ROOT / 'temp' / 's3' / 'scientific.parquet'",
            code,
        )
        self.assertIn(
            "artifact = PROJECT_ROOT / 'temp' / 's6' / 's6_results.json'",
            code,
        )
        self.assertIn(
            "config = PROJECT_ROOT / 'config' / 's6_calibrated_classical.json'",
            code,
        )
        for forbidden in (
            "TfidfVectorizer",
            "SGDClassifier",
            "LinearSVC",
            "RidgeClassifier",
            "LogisticRegression",
        ):
            self.assertNotIn(forbidden, code)
        for table in (
            "calibration_summary",
            "outer_summary",
            "critical_confusions",
            "per_class",
            "statuses",
        ):
            self.assertIn(f"display(evidence.{table})", code)

    def test_s7_notebooks_are_thin_cached_and_bilingual(self) -> None:
        """Require identical code, Polars tables, and no estimator logic."""

        pt_path = ROOT / "notebooks" / "pt-BR" / (
            "09_S7_Frozen_Model_Package_PT.ipynb"
        )
        en_path = ROOT / "notebooks" / "en-US" / (
            "09_S7_Frozen_Model_Package_EN.ipynb"
        )
        pt = json.loads(pt_path.read_text(encoding="utf-8"))
        en = json.loads(en_path.read_text(encoding="utf-8"))
        pt_code = [
            cell["source"] for cell in pt["cells"] if cell["cell_type"] == "code"
        ]
        en_code = [
            cell["source"] for cell in en["cells"] if cell["cell_type"] == "code"
        ]
        self.assertEqual(pt_code, en_code)
        code = "\n".join("".join(cell["source"]) for cell in pt["cells"])
        self.assertIn("RUN_MODE = 'disabled'", code)
        self.assertIn("load_s7_report_tables", code)
        for table in ("statuses", "calibration_summary", "per_class"):
            self.assertIn(f"display(evidence.{table})", code)
        for forbidden in (
            "TfidfVectorizer",
            "LinearSVC",
            "SGDClassifier",
            "sklearn",
        ):
            self.assertNotIn(forbidden, code)
        self.assertNotIn("—", code)

    def test_s8_notebooks_are_thin_disabled_and_bilingual(self) -> None:
        """Require identical disabled code and aggregate reporting only."""

        pt_path = ROOT / "notebooks" / "pt-BR" / (
            "10_S8_Confirmatory_Evaluation_PT.ipynb"
        )
        en_path = ROOT / "notebooks" / "en-US" / (
            "10_S8_Confirmatory_Evaluation_EN.ipynb"
        )
        pt = json.loads(pt_path.read_text(encoding="utf-8"))
        en = json.loads(en_path.read_text(encoding="utf-8"))
        pt_code = [
            cell["source"] for cell in pt["cells"]
            if cell["cell_type"] == "code"
        ]
        en_code = [
            cell["source"] for cell in en["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertEqual(pt_code, en_code)
        code = "\n".join("".join(cell["source"]) for cell in pt["cells"])
        self.assertIn("RUN_MODE = 'disabled'", code)
        self.assertIn("load_s8_report_tables", code)
        self.assertIn("run_s8", code)
        self.assertNotIn("S8_CONFIRM_TEST_2025_H1_ONCE", code)
        for forbidden in (
            "sklearn",
            "duckdb",
            "TfidfVectorizer",
            "decision_function",
        ):
            self.assertNotIn(forbidden, code)
        for table in (
            "statuses",
            "primary_summary",
            "per_class",
            "confidence_intervals",
            "scientific_operational",
            "audit_counts",
        ):
            self.assertIn(f"display(evidence.{table})", code)

    def test_v2_package_notebooks_orchestrate_and_stay_bilingual(self) -> None:
        """Require identical text-only orchestration in both notebooks."""

        pt_path = ROOT / "notebooks" / "pt-BR" / (
            "12_V2_Frozen_Package_PT.ipynb"
        )
        en_path = ROOT / "notebooks" / "en-US" / (
            "12_V2_Frozen_Package_EN.ipynb"
        )
        pt = json.loads(pt_path.read_text(encoding="utf-8"))
        en = json.loads(en_path.read_text(encoding="utf-8"))
        pt_code = [
            cell["source"] for cell in pt["cells"]
            if cell["cell_type"] == "code"
        ]
        en_code = [
            cell["source"] for cell in en["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertEqual(pt_code, en_code)
        code = "\n".join("".join(source) for source in pt_code)
        self.assertIn("sys.path.insert(0, str(SRC_DIR))", code)
        self.assertIn("render_package_import_report", code)
        self.assertIn("PROJECT_ROOT / 'temp' / 'v2'", code)
        self.assertIn("PROJECT_ROOT / 'config'", code)
        self.assertIn("v2_package.json", code)
        self.assertIn("v2_results.json", code)
        self.assertNotIn("kaggle_output_p", code)
        for forbidden in (
            "sklearn",
            "joblib",
            "TfidfVectorizer",
            "LinearSVC",
            "run_v2_package",
            "display(",
        ):
            self.assertNotIn(forbidden, code)
        for notebook in (pt, en):
            prose = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            )
            self.assertIn("PACKAGE_FROZEN", prose)
            self.assertIn("REPRODUCTION_MISMATCH", prose)
            self.assertNotIn("—", prose)
        pt_prose = "\n".join(
            "".join(cell["source"])
            for cell in pt["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("-0,13949530151425016", pt_prose)
        self.assertNotIn("Ã", pt_prose)

    def test_v2_stress_notebooks_orchestrate_and_stay_bilingual(self) -> None:
        """Require identical text-only orchestration in both notebooks."""

        pt_path = ROOT / "notebooks" / "pt-BR" / (
            "13_V2_Stress_Confirmatory_PT.ipynb"
        )
        en_path = ROOT / "notebooks" / "en-US" / (
            "13_V2_Stress_Confirmatory_EN.ipynb"
        )
        pt = json.loads(pt_path.read_text(encoding="utf-8"))
        en = json.loads(en_path.read_text(encoding="utf-8"))
        pt_code = [
            cell["source"] for cell in pt["cells"]
            if cell["cell_type"] == "code"
        ]
        en_code = [
            cell["source"] for cell in en["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertEqual(pt_code, en_code)
        code = "\n".join("".join(source) for source in pt_code)
        self.assertIn("sys.path.insert(0, str(SRC_DIR))", code)
        self.assertIn("load_stress_payload", code)
        self.assertIn("render_stress_import_report", code)
        self.assertIn(
            "PROJECT_ROOT / 'config' / 'v2_stress_protocol.json'", code
        )
        self.assertIn(
            "PROJECT_ROOT / 'temp' / 'v2' / 'v2_stress_results.json'", code
        )
        self.assertIn(
            "PROJECT_ROOT / 'config' / 'v2_stress_results.json'", code
        )
        for forbidden in (
            "sklearn",
            "joblib",
            "duckdb",
            "TfidfVectorizer",
            "LinearSVC",
            "run_v2_stress",
            "V2_STRESS_UNLOCK",
            "os.environ",
            "getenv",
            "display(",
            "RUN_MODE",
        ):
            self.assertNotIn(forbidden, code)
        for notebook in (pt, en):
            prose = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            )
            self.assertIn("CONFIRMED", prose)
            self.assertIn("NOT_CONFIRMED", prose)
            self.assertIn("ADR-014", prose)
            self.assertNotIn("V2_STRESS_UNLOCK", prose)
            self.assertNotIn("—", prose)
        pt_prose = "\n".join(
            "".join(cell["source"])
            for cell in pt["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("0,047234", pt_prose)
        self.assertNotIn("Ã", pt_prose)

    def test_s4_manifest_and_notebooks_use_utf8_narrative(self):
        """Require the published manifest and readable bilingual narrative."""

        manifest = ROOT / "config" / "s4_results.json"
        text = manifest.read_text(encoding="utf-8")
        for value in ("três", "diagnóstico", "precisão", "não"):
            self.assertIn(value, text)
        for language, name in (
            ("pt-BR", "06_S4_Error_Representation_Challenge_PT.ipynb"),
            ("en-US", "06_S4_Error_Representation_Challenge_EN.ipynb"),
        ):
            notebook = ROOT / "notebooks" / language / name
            text = notebook.read_text(encoding="utf-8")
            self.assertNotIn("Ã", text)
            self.assertNotIn("Â", text)
