"""Run storage: fit once, reuse on identical fingerprint, refit on change or request."""

import json

import pandas as pd
import pytest

from retail_multilevel.runs import fingerprint, get_or_fit, load_run


@pytest.fixture
def inputs(tmp_path):
    p = tmp_path / "input.csv"
    p.write_text("a,b\n1,2\n")
    return p


@pytest.fixture
def pkg(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "m.py").write_text("x = 1\n")
    return d


def _fitter(calls):
    def fit(run):
        calls.append(run.run_id)
        run.save_table("coefs", pd.DataFrame({"term": ["price"], "est": [-1.5]}))
        run.save_json("info", {"n": 3})
        run.file("model.txt").write_text("model")
        return {"rmse": 0.42}

    return fit


def test_fit_then_reuse(tmp_path, inputs, pkg):
    root, calls = tmp_path / "runs", []
    kw = dict(inputs=[inputs], root=root, package_dir=pkg)
    r1 = get_or_fit("exp", {"alpha": 1}, _fitter(calls), **kw)
    r2 = get_or_fit("exp", {"alpha": 1}, _fitter(calls), **kw)
    assert len(calls) == 1 and r1.run_id == r2.run_id
    assert r2.metrics == {"rmse": 0.42}
    assert r2.load_table("coefs")["est"].iloc[0] == -1.5
    assert r2.load_json("info") == {"n": 3}
    ledger = (root / "LEDGER.jsonl").read_text().splitlines()
    assert len(ledger) == 1 and json.loads(ledger[0])["run_id"] == r1.run_id


@pytest.mark.parametrize("change", ["config", "input", "code"])
def test_any_change_triggers_refit(tmp_path, inputs, pkg, change):
    root, calls = tmp_path / "runs", []
    kw = dict(inputs=[inputs], root=root, package_dir=pkg)
    get_or_fit("exp", {"alpha": 1}, _fitter(calls), **kw)
    config = {"alpha": 2} if change == "config" else {"alpha": 1}
    if change == "input":
        inputs.write_text("a,b\n1,3\n")
    if change == "code":
        (pkg / "m.py").write_text("x = 2\n")
    get_or_fit("exp", config, _fitter(calls), **kw)
    assert len(calls) == 2


def test_train_mode_forces_refit_and_load_mode_never_fits(tmp_path, inputs, pkg):
    root, calls = tmp_path / "runs", []
    kw = dict(inputs=[inputs], root=root, package_dir=pkg)
    with pytest.raises(FileNotFoundError):
        get_or_fit("exp", {}, _fitter(calls), mode="load", **kw)
    get_or_fit("exp", {}, _fitter(calls), **kw)
    get_or_fit("exp", {}, _fitter(calls), mode="train", **kw)
    assert len(calls) == 2
    with pytest.raises(ValueError):
        get_or_fit("exp", {}, _fitter(calls), mode="refit", **kw)


def test_load_run_returns_latest_and_ignores_incomplete(tmp_path, inputs, pkg):
    root, calls = tmp_path / "runs", []
    kw = dict(inputs=[inputs], root=root, package_dir=pkg)
    get_or_fit("exp", {"v": 1}, _fitter(calls), **kw)
    r2 = get_or_fit("exp", {"v": 2}, _fitter(calls), **kw)
    (root / "exp" / "99999999-999999-deadbeef").mkdir()  # interrupted run: no metadata.json
    assert load_run("exp", root=root).run_id == r2.run_id
    with pytest.raises(FileNotFoundError):
        load_run("other", root=root)


def test_code_files_limit_refits_to_declared_modules(tmp_path, inputs, pkg):
    (pkg / "plots.py").write_text("y = 1\n")
    root, calls = tmp_path / "runs", []
    kw = dict(inputs=[inputs], root=root, package_dir=pkg, code_files=[pkg / "m.py"])
    get_or_fit("exp", {}, _fitter(calls), **kw)
    (pkg / "plots.py").write_text("y = 2\n")  # undeclared module: reuse
    get_or_fit("exp", {}, _fitter(calls), **kw)
    assert len(calls) == 1
    (pkg / "m.py").write_text("x = 3\n")  # declared module: refit
    get_or_fit("exp", {}, _fitter(calls), **kw)
    assert len(calls) == 2


def test_fingerprint_ignores_config_key_order(inputs, pkg):
    assert fingerprint({"a": 1, "b": 2}, [inputs], pkg) == fingerprint({"b": 2, "a": 1}, [inputs], pkg)
