"""Operational tests for the V2 frozen-package Kaggle helpers."""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from consumer_complaint_intelligence import kaggle_execution as kx


def _smoke_payload(**overrides):
    """Build a passing package smoke payload for injection."""

    payload = {
        "status": "DIAGNOSTIC_ONLY",
        "complete": True,
        "candidate": {"candidate_id": "word_char_tfidf_union_40000_60000_c_1"},
        "fallback_model_version": "consumer-complaint-classifier-s7",
        "provenance": {"protocol": {}, "cache": {}},
        "checks": {"frozen_config_validated": True, "s7_loaded": True},
    }
    payload.update(overrides)
    return payload


class PackageBundleContractTests(unittest.TestCase):
    """Keep the staged bundle and output contracts explicit."""

    def test_bundle_ships_the_frozen_package_config(self) -> None:
        """Require the frozen package config inside the Kaggle bundle."""

        self.assertIn("config/v2_frozen_package.json", kx.BUNDLE_FILES)

    def test_bundle_ships_the_pinned_d1_and_d2_evidence(self) -> None:
        """Require the pinned provenance files the gate verifies."""

        for name in (
            "temp/v2/v2_classical_benchmark.json",
            "config/v2_classical_results.json",
            "temp/v2/v2_transformer_challenge.json",
            "config/v2_transformer_results.json",
        ):
            self.assertIn(name, kx.BUNDLE_FILES)

    def test_output_contract_names_evidence_and_bundle(self) -> None:
        """Publish two evidence files plus the conditional joblib bundle."""

        self.assertEqual(
            kx.PACKAGE_OUTPUT_FILES,
            ("temp/v2/v2_package.json", "config/v2_results.json"),
        )
        self.assertEqual(
            kx.PACKAGE_BUNDLE_FILE,
            "artifacts/v2/consumer_complaint_detector_v2.joblib",
        )

    def test_package_helpers_are_exported(self) -> None:
        """Expose the package stage through the module contract."""

        for name in (
            "preflight_package",
            "run_full_package",
            "collect_outputs_package",
            "PACKAGE_OUTPUT_FILES",
            "PACKAGE_BUNDLE_FILE",
        ):
            self.assertIn(name, kx.__all__)


class CollectPackageOutputsTests(unittest.TestCase):
    """Collect published evidence and only a bundle that actually exists."""

    def _work_root(self, root: pathlib.Path, *, with_bundle: bool) -> None:
        """Populate a staged tree with published package outputs."""

        for name in kx.PACKAGE_OUTPUT_FILES:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"name": name}), encoding="utf-8")
        if with_bundle:
            bundle = root / kx.PACKAGE_BUNDLE_FILE
            bundle.parent.mkdir(parents=True, exist_ok=True)
            bundle.write_bytes(b"joblib-placeholder")

    def test_collects_evidence_and_bundle_when_frozen(self) -> None:
        """Copy both evidence files and the bundle after a passing gate."""

        with tempfile.TemporaryDirectory() as raw:
            base = pathlib.Path(raw)
            work = base / "project"
            self._work_root(work, with_bundle=True)
            copied = kx.collect_outputs_package(work, base / "out")
            names = sorted(pathlib.Path(item).name for item in copied)
            self.assertEqual(
                names,
                [
                    "consumer_complaint_detector_v2.joblib",
                    "v2_package.json",
                    "v2_results.json",
                ],
            )

    def test_collects_evidence_only_on_reproduction_mismatch(self) -> None:
        """Return the evidence files when no bundle was persisted."""

        with tempfile.TemporaryDirectory() as raw:
            base = pathlib.Path(raw)
            work = base / "project"
            self._work_root(work, with_bundle=False)
            copied = kx.collect_outputs_package(work, base / "out")
            names = sorted(pathlib.Path(item).name for item in copied)
            self.assertEqual(names, ["v2_package.json", "v2_results.json"])
            self.assertFalse(
                (base / "out" / "consumer_complaint_detector_v2.joblib").exists()
            )

    def test_missing_evidence_file_raises(self) -> None:
        """Refuse to report success when published evidence is absent."""

        with tempfile.TemporaryDirectory() as raw:
            base = pathlib.Path(raw)
            work = base / "project"
            self._work_root(work, with_bundle=True)
            (work / kx.PACKAGE_OUTPUT_FILES[0]).unlink()
            with self.assertRaises(ValueError):
                kx.collect_outputs_package(work, base / "out")


class PreflightPackageTests(unittest.TestCase):
    """Fail the preflight before any fitting when the tree is not ready."""

    def test_missing_frozen_config_raises(self) -> None:
        """Require the frozen package config in the staged tree."""

        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError) as caught:
                kx.preflight_package(pathlib.Path(raw))
            self.assertIn("Frozen package config", str(caught.exception))

    def _staged_root(self, base: pathlib.Path) -> pathlib.Path:
        """Create a staged tree carrying only the frozen package config."""

        work = base / "project"
        config = work / "config" / "v2_frozen_package.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("{}", encoding="utf-8")
        return work

    def test_ready_report_summarizes_the_smoke(self) -> None:
        """Report the pinned candidate and verified provenance when ready."""

        with tempfile.TemporaryDirectory() as raw:
            work = self._staged_root(pathlib.Path(raw))
            with mock.patch(
                "consumer_complaint_intelligence.v2_package"
                ".run_v2_package_smoke",
                return_value=_smoke_payload(),
            ):
                report = kx.preflight_package(work)
            self.assertEqual(report["status"], "READY")
            self.assertEqual(
                report["candidate_id"],
                "word_char_tfidf_union_40000_60000_c_1",
            )
            self.assertEqual(report["provenance_verified"], ["cache", "protocol"])

    def test_failed_smoke_check_raises(self) -> None:
        """Refuse to report READY when any smoke check is false."""

        with tempfile.TemporaryDirectory() as raw:
            work = self._staged_root(pathlib.Path(raw))
            payload = _smoke_payload(
                checks={"frozen_config_validated": True, "s7_loaded": False}
            )
            with mock.patch(
                "consumer_complaint_intelligence.v2_package"
                ".run_v2_package_smoke",
                return_value=payload,
            ):
                with self.assertRaises(ValueError) as caught:
                    kx.preflight_package(work)
            self.assertIn("s7_loaded", str(caught.exception))

    def test_incomplete_smoke_raises(self) -> None:
        """Refuse to proceed when the smoke did not complete."""

        with tempfile.TemporaryDirectory() as raw:
            work = self._staged_root(pathlib.Path(raw))
            payload = _smoke_payload(status="FAILED", complete=False)
            with mock.patch(
                "consumer_complaint_intelligence.v2_package"
                ".run_v2_package_smoke",
                return_value=payload,
            ):
                with self.assertRaises(ValueError):
                    kx.preflight_package(work)


class StressSealIntegrityTests(unittest.TestCase):
    """Keep the sealed stress protocol out of any remote execution bundle."""

    def test_bundle_never_ships_stress_artifacts(self) -> None:
        """Refuse to stage the stress protocol, results, or manifest."""

        for name in (
            "config/v2_stress_protocol.json",
            "config/v2_stress_results.json",
            "temp/v2/v2_stress_results.json",
        ):
            self.assertNotIn(name, kx.BUNDLE_FILES)

    def test_bundle_carries_no_stress_path_at_all(self) -> None:
        """Reject any staged path that mentions the sealed partition."""

        for name in kx.BUNDLE_FILES:
            self.assertNotIn("stress", name)

    def test_module_declares_no_stress_stage(self) -> None:
        """Keep the unlock contract and stress helpers out of the runner."""

        exported = set(kx.__all__)
        for name in exported:
            self.assertNotIn("stress", name.lower())
        source = pathlib.Path(kx.__file__).read_text(encoding="utf-8")
        self.assertNotIn("V2_STRESS_UNLOCK", source)
        self.assertNotIn("v2_stress", source)


if __name__ == "__main__":
    unittest.main()
