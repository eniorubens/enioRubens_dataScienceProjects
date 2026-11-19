"""Tests of the resume guard (``runs``) and of its use by ``scripts/run_evaluation.py`` (synthetic, no solver)."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from delivery_routing_vrptw import SolverConfig, runs
from delivery_routing_vrptw.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_evaluation():
    """Load ``scripts/run_evaluation.py`` as a module (no solver is executed)."""
    return _load_script("run_evaluation")


def _eval_cfg():
    cfg = load_config()
    return replace(cfg, solver=replace(cfg.solver, time_limit_s=cfg.evaluation.time_limit_s))


def _rows(instance: str, vehicles: int, replicates: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "instance": instance, "method": "ortools", "replicate": list(range(1, replicates + 1)),
        "vehicles": vehicles, "distance": 100.0 + vehicles, "routes": "[[1, 2]]",
    })


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


# --------------------------------------------------------------------------- fingerprint
def test_fingerprint_is_stable_and_key_order_free():
    assert runs.config_fingerprint({"a": 1, "b": [1, 2]}) == runs.config_fingerprint({"b": [1, 2], "a": 1})
    assert len(runs.config_fingerprint({"a": 1})) == 64


def test_fingerprint_changes_with_any_result_determining_field():
    base = {"scale": 100, "k": None, "time_limit_s": 30.0}
    fp = runs.config_fingerprint(base)
    for key, value in (("scale", 10000), ("k", 10.0), ("time_limit_s", 120.0)):
        assert runs.config_fingerprint({**base, key: value}) != fp


def test_solver_fields_separate_v1_absolute_and_v2_relative_fixed_cost():
    v1 = runs.solver_fields(SolverConfig())
    v2 = runs.solver_fields(SolverConfig(scale=10000, fixed_cost_k=10.0))
    assert v1["fixed_cost_rule"] == "absolute" and v1["vehicle_fixed_cost"] == 10**7
    assert v2["fixed_cost_rule"] == "relative_k_scale_Dub" and v2["vehicle_fixed_cost"] is None
    assert runs.config_fingerprint(v1) != runs.config_fingerprint(v2)
    assert "time_limit_s" not in runs.solver_fields(SolverConfig(), with_time_limit=False)


def test_evaluation_fields_differ_between_versions_and_include_workers(run_evaluation):
    cfg = _eval_cfg()
    v1, v2 = run_evaluation.evaluation_fields(cfg, "v1"), run_evaluation.evaluation_fields(cfg, "v2")
    assert runs.config_fingerprint(v1) != runs.config_fingerprint(v2)
    assert v1["scale"] == 100 and v2["scale"] == run_evaluation.V2_SCALE
    other = replace(cfg, solver=replace(cfg.solver, workers=cfg.solver.workers + 1))
    assert run_evaluation.evaluation_fields(other, "v1") != v1
    assert runs.config_fingerprint(run_evaluation.evaluation_fields(other, "v1")) != runs.config_fingerprint(v1)


# --------------------------------------------------------------------------- guard
def test_fresh_folder_writes_metadata_with_fingerprint(tmp_path):
    out = tmp_path / "new"
    fp = runs.start_run(out, {"a": 1}, {"x": 1})
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["config_fingerprint"] == fp and meta["config_fingerprint_fields"] == {"a": 1}


def test_same_fingerprint_resumes_and_skips_done_cells(tmp_path, run_evaluation):
    cfg = _eval_cfg()
    fields = run_evaluation.evaluation_fields(cfg, "v1")
    fp = runs.start_run(tmp_path, fields, {})
    runs.append_rows(tmp_path / "results.csv", _rows("RC207", 4), fp)
    assert runs.check_resume(tmp_path, fp) == "resume"
    assert runs.start_run(tmp_path, fields, {}) == fp
    cells, todo = run_evaluation.plan_cells(["RC207", "RC208"], 5, False, "v1", tmp_path / "results.csv")
    assert len(cells) == 10
    assert todo == [("RC208", "ortools", r) for r in range(1, 6)]
    assert pd.read_csv(tmp_path / "results.csv")["config_fingerprint"].eq(fp).all()


def test_v2_resume_into_v1_folder_is_refused_and_changes_nothing(tmp_path, run_evaluation):
    """Reviewer's case: five 4-vehicle v1 plans of RC207 must not be relabelled as v2."""
    cfg = _eval_cfg()
    fp1 = runs.start_run(tmp_path, run_evaluation.evaluation_fields(cfg, "v1"), {"version": "v1"})
    runs.append_rows(tmp_path / "results.csv", _rows("RC207", 4), fp1)
    before = _snapshot(tmp_path)
    with pytest.raises(runs.ResumeError, match="different configuration"):
        runs.start_run(tmp_path, run_evaluation.evaluation_fields(cfg, "v2"), {"version": "v2"})
    assert _snapshot(tmp_path) == before


def test_run_evaluation_main_refuses_v2_into_v1_folder(tmp_path, run_evaluation, monkeypatch):
    """The real main path, up to the guard: no solver, no pool, files untouched."""
    cfg = _eval_cfg()
    fp1 = runs.start_run(tmp_path, run_evaluation.evaluation_fields(cfg, "v1"), {"version": "v1"})
    runs.append_rows(tmp_path / "results.csv", _rows("RC207", 4), fp1)
    before = _snapshot(tmp_path)

    def no_pool(*a, **k):
        raise AssertionError("the solver pool must not start")

    monkeypatch.setattr(run_evaluation, "ProcessPoolExecutor", no_pool)
    argv = ["run_evaluation.py", "--version", "v2", "--confirm-eval", "--post-hoc", "--instances", "RC207",
            "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="different configuration"):
        run_evaluation.main()
    assert _snapshot(tmp_path) == before


def test_row_with_other_fingerprint_is_refused_even_if_metadata_matches(tmp_path):
    fp = runs.start_run(tmp_path, {"a": 1}, {})
    runs.append_rows(tmp_path / "results.csv", _rows("R101", 3, 1), fp)
    runs.append_rows(tmp_path / "results.csv", _rows("R102", 3, 1), "0" * 64)
    with pytest.raises(runs.ResumeError, match="row"):
        runs.check_resume(tmp_path, fp)


def test_legacy_folder_needs_explicit_flag_and_is_not_stamped(tmp_path):
    (tmp_path / "run_metadata.json").write_text(json.dumps({"timestamp": "old"}), encoding="utf-8")
    _rows("R101", 3, 2).to_csv(tmp_path / "results.csv", index=False)
    before = _snapshot(tmp_path)
    with pytest.raises(runs.ResumeError, match="--allow-legacy-resume"):
        runs.start_run(tmp_path, {"a": 1}, {})
    assert _snapshot(tmp_path) == before
    assert runs.check_resume(tmp_path, runs.config_fingerprint({"a": 1}), allow_legacy=True) == "legacy"

    fp = runs.start_run(tmp_path, {"a": 1}, {}, allow_legacy=True)
    assert json.loads((tmp_path / "run_metadata.legacy.json").read_text(encoding="utf-8")) == {"timestamp": "old"}
    runs.append_rows(tmp_path / "results.csv", _rows("R102", 3, 1), fp)
    out = pd.read_csv(tmp_path / "results.csv")
    assert "config_fingerprint" not in out.columns and len(out) == 3


def test_empty_results_file_counts_as_fresh(tmp_path):
    (tmp_path / "results.csv").write_text("", encoding="utf-8")
    assert runs.check_resume(tmp_path, "x") == "fresh"


def test_every_resuming_runner_calls_the_guard():
    names = ["run_evaluation", "run_time_sensitivity", "run_window_scenarios", "run_early_delivery",
             "run_dev_grid", "run_v2_solver_grid", "run_v2_layerb_warm", "run_phase_c_solver",
             "run_reoptimisation", "run_robustness"]
    for name in names:
        text = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert "runs.start_run(" in text and "runs.add_resume_args(" in text, name
        assert '"run_metadata.json").write_text' not in text, name


def test_resume_with_other_worker_count_is_refused_and_changes_nothing(tmp_path):
    """B4 compares 4 vs 8 processes: a 4-process folder must not be resumed (and relabelled) as 8."""
    fields = {"runner": "x", "time_limit_s": 30.0, "workers": 4}
    fp = runs.start_run(tmp_path, fields, {"workers": 4})
    runs.append_rows(tmp_path / "results.csv", _rows("RC207", 4), fp)
    before = _snapshot(tmp_path)
    with pytest.raises(runs.ResumeError, match="different configuration"):
        runs.start_run(tmp_path, {**fields, "workers": 8}, {"workers": 8})
    assert _snapshot(tmp_path) == before
    assert runs.start_run(tmp_path, fields, {"workers": 4}) == fp


def test_phase_c_field_functions_include_workers():
    cfg = load_config()
    other = replace(cfg, phase_c=replace(cfg.phase_c, workers=cfg.phase_c.workers + 1))
    for name, fn_name in (("run_phase_c_solver", "phase_c_fields"), ("run_reoptimisation", "reopt_fields")):
        fn = getattr(_load_script(name), fn_name)
        a, b = fn(cfg, "capacity", 30.0, None), fn(other, "capacity", 30.0, None)
        assert a["workers"] == cfg.phase_c.workers and b["workers"] == other.phase_c.workers
        assert runs.config_fingerprint(a) != runs.config_fingerprint(b), name


def test_every_runner_puts_workers_in_the_fingerprint_fields():
    for name in ["run_time_sensitivity", "run_window_scenarios", "run_early_delivery", "run_dev_grid",
                 "run_v2_solver_grid", "run_v2_layerb_warm", "run_robustness"]:
        text = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert f'"runner": "{name}", ' in text and '"workers": ' in text.split(f'"runner": "{name}"')[1][:60], name
