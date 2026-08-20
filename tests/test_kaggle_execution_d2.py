"""Operational checks for the V2-D2 Kaggle GPU execution plumbing.

These tests cover only the staging/bundling/orchestration surface added for
the D2 transformer challenge (ADR-012). They never import torch or
transformers, and never exercise the scientific runner itself, which lives
in ``v2_transformer`` and is covered by its own test module.
"""

import json
import unittest
from pathlib import Path

from consumer_complaint_intelligence import kaggle_execution as kx


ROOT = Path(__file__).parents[1]
KERNEL_D2_DIR = ROOT / "kaggle" / "kernel-d2"
FORBIDDEN_CODE_TERMS = ("stress", "monitor", "test_partition")


class BundleFilesD2Tests(unittest.TestCase):
    """Verify the D2 additions to the bundle manifest and output list."""

    def test_d2_bundle_entries_are_present_and_exist_on_disk(self) -> None:
        """Require the pinned D2 config and D1 incumbent to travel and exist."""

        expected = (
            "config/v2_d2_execution.json",
            "config/v2_classical_results.json",
            "temp/v2/v2_classical_benchmark.json",
        )
        for name in expected:
            self.assertIn(name, kx.BUNDLE_FILES)
            self.assertTrue(
                (ROOT / name).is_file(), f"missing bundle file: {name}"
            )

    def test_d2_output_files_are_the_two_expected_paths(self) -> None:
        """Require exactly the D2 artifact and manifest output paths."""

        self.assertEqual(
            kx.D2_OUTPUT_FILES,
            (
                "temp/v2/v2_transformer_challenge.json",
                "config/v2_transformer_results.json",
            ),
        )


class KernelMetadataD2Tests(unittest.TestCase):
    """Verify the new GPU kernel metadata is well-formed."""

    def test_kernel_metadata_parses_and_enables_gpu(self) -> None:
        """Require valid JSON, GPU enabled, and a co-located code file."""

        meta_path = KERNEL_D2_DIR / "kernel-metadata.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertIs(payload.get("enable_gpu"), True)
        code_file = payload.get("code_file")
        self.assertTrue(code_file)
        self.assertTrue((KERNEL_D2_DIR / str(code_file)).is_file())


class NotebookD2Tests(unittest.TestCase):
    """Verify the D2 notebook is valid, orchestration-only, and bounded."""

    def _load_notebook(self) -> dict:
        path = KERNEL_D2_DIR / "v2_d2_transformer.ipynb"
        return json.loads(path.read_text(encoding="utf-8"))

    def _cell_source(self, cell: dict) -> str:
        source = cell["source"]
        return source if isinstance(source, str) else "".join(source)

    def test_notebook_parses_as_json(self) -> None:
        """Require the notebook to be syntactically valid JSON."""

        notebook = self._load_notebook()
        self.assertEqual(notebook.get("nbformat"), 4)
        self.assertTrue(notebook.get("cells"))

    def test_code_cells_call_the_three_d2_orchestration_entry_points(self) -> None:
        """Require the notebook to call preflight, run, and collect for D2."""

        notebook = self._load_notebook()
        code = "\n".join(
            self._cell_source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("kx.preflight_d2", code)
        self.assertIn("kx.run_full_d2", code)
        self.assertIn("kx.collect_outputs_d2", code)

    def test_code_cells_never_mention_sealed_or_forbidden_partition_terms(
        self,
    ) -> None:
        """Keep executable cells free of sealed-partition or test-suite terms.

        Markdown prose is allowed to name ``test``/``stress``/``monitor``
        when stating the sealed-partition boundary (mirroring the D1
        notebook's intro cell); only the executable code cells are held to
        the stricter no-mention bar, since those are what could ever touch
        a partition in practice.
        """

        notebook = self._load_notebook()
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            lowered = self._cell_source(cell).lower()
            for term in FORBIDDEN_CODE_TERMS:
                self.assertNotIn(term, lowered, f"cell {cell.get('id')!r}")


class ReportGpuTests(unittest.TestCase):
    """Verify the notebook-facing GPU report never raises."""

    def test_report_gpu_returns_a_boolean_flag_without_torch(self) -> None:
        """Require a boolean cuda_available even when torch is unavailable."""

        report = kx.report_gpu()
        self.assertIsInstance(report, dict)
        self.assertIsInstance(report.get("cuda_available"), bool)
        self.assertIn("device_name", report)
        self.assertIn("torch_version", report)


if __name__ == "__main__":
    unittest.main()


class GpuProbeTests(unittest.TestCase):
    """Keep the accelerator gate honest when torch is absent or broken."""

    def test_report_gpu_reports_unusable_without_torch(self) -> None:
        """Distinguish 'no device' from 'device that cannot run a kernel'."""

        report = kx.report_gpu()
        self.assertIn("cuda_available", report)
        self.assertIn("cuda_usable", report)
        self.assertIsInstance(report["cuda_available"], bool)
        self.assertIsInstance(report["cuda_usable"], bool)
        self.assertFalse(report["cuda_usable"] and not report["cuda_available"])

    def test_assert_usable_gpu_raises_when_no_device_can_run(self) -> None:
        """Fail fast rather than proceed toward a first-launch crash."""

        report = kx.report_gpu()
        if report["cuda_usable"]:
            self.skipTest("a usable CUDA device is present")
        with self.assertRaises(ValueError):
            kx.assert_usable_gpu()

    def test_kernel_metadata_pins_a_supported_accelerator(self) -> None:
        """Pin T4: current torch builds ship no kernels for P100 sm_60."""

        path = ROOT / "kaggle" / "kernel-d2" / "kernel-metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(meta["enable_gpu"])

    def test_notebook_gates_the_gpu_before_the_expensive_pool_build(
        self,
    ) -> None:
        """Keep the kernel probe ahead of staging and pool generation."""

        path = ROOT / "kaggle" / "kernel-d2" / "v2_d2_transformer.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        gate = next(
            index for index, cell in enumerate(code)
            if "assert_usable_gpu" in cell
        )
        staging = next(
            index for index, cell in enumerate(code)
            if "stage_project" in cell
        )
        self.assertLess(gate, staging)
