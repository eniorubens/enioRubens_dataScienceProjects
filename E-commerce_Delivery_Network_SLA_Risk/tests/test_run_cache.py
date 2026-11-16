"""Tests for the notebook execution cache."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ecom_delivery_sla.run_cache import cached_run, describe_run, run_fingerprint


@pytest.fixture
def tmp_path():
    """Isolated directory via ``tempfile`` (pytest's own base dir may be locked)."""
    with tempfile.TemporaryDirectory(prefix="run_cache_") as directory:
        yield Path(directory)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


class Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> dict[str, pd.DataFrame]:
        self.calls += 1
        return {
            "one": pd.DataFrame({"v": [1.0, 2.0], "w": ["p", "q"]}),
            "two": pd.DataFrame({"m": [0.5]}),
        }


def _run(tmp_path, counter, frame, config=None, mode="auto"):
    return cached_run(
        "tourn",
        "V1",
        config or {"k": 1},
        counter,
        frames={"f": frame},
        mode=mode,
        cache_root=tmp_path,
    )


def test_first_use_computes_and_saves(tmp_path, frame):
    counter = Counter()
    results, meta = _run(tmp_path, counter, frame)
    assert counter.calls == 1
    assert meta["source"] == "computed"
    run_dir = tmp_path / "tourn" / "V1"
    for key in ("one", "two"):
        assert (run_dir / f"{key}.pkl").exists()
        assert (run_dir / f"{key}.csv").exists()
    assert (run_dir / "metadata.json").exists()
    assert set(results) == {"one", "two"}
    assert [p.name for p in (tmp_path / "tourn").iterdir()] == ["V1"]


def test_second_use_loads_without_compute(tmp_path, frame):
    counter = Counter()
    first, _ = _run(tmp_path, counter, frame)
    second, meta = _run(tmp_path, counter, frame)
    assert counter.calls == 1
    assert meta["source"] == "loaded"
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])


def test_config_change_invalidates(tmp_path, frame):
    counter = Counter()
    _run(tmp_path, counter, frame)
    _, meta = _run(tmp_path, counter, frame, config={"k": 2})
    assert counter.calls == 2
    assert meta["source"] == "computed"


def test_frame_change_invalidates(tmp_path, frame):
    counter = Counter()
    _run(tmp_path, counter, frame)
    changed = frame.copy()
    changed.loc[0, "a"] = 99
    _run(tmp_path, counter, changed)
    assert counter.calls == 2


def test_train_mode_forces_recompute(tmp_path, frame):
    counter = Counter()
    _run(tmp_path, counter, frame)
    _, meta = _run(tmp_path, counter, frame, mode="train")
    assert counter.calls == 2
    assert meta["source"] == "computed"


def test_load_without_cache_raises(tmp_path, frame):
    with pytest.raises(FileNotFoundError):
        _run(tmp_path, Counter(), frame, mode="load")


def test_load_with_diverging_fingerprint_raises(tmp_path, frame):
    counter = Counter()
    _run(tmp_path, counter, frame)
    with pytest.raises(ValueError):
        _run(tmp_path, counter, frame, config={"k": 2}, mode="load")
    assert counter.calls == 1


def test_invalid_mode_raises(tmp_path, frame):
    with pytest.raises(ValueError):
        _run(tmp_path, Counter(), frame, mode="nope")


def test_metadata_keys(tmp_path, frame):
    _run(tmp_path, Counter(), frame)
    meta = json.loads((tmp_path / "tourn" / "V1" / "metadata.json").read_text("utf-8"))
    expected = {
        "name",
        "run_version",
        "fingerprint",
        "created_at",
        "elapsed_seconds",
        "python",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "config",
        "keys",
    }
    assert expected <= set(meta)
    assert meta["keys"] == ["one", "two"]
    assert meta["config"] == {"k": 1}


def test_fingerprint_is_stable_and_order_independent(frame):
    a = run_fingerprint({"x": 1, "y": 2}, {"f": frame})
    b = run_fingerprint({"y": 2, "x": 1}, {"f": frame.copy()})
    assert a == b
    assert a != run_fingerprint({"x": 1, "y": 3}, {"f": frame})
    assert a != run_fingerprint({"x": 1, "y": 2}, {"f": frame.rename(columns={"a": "c"})})


def test_fingerprint_depends_on_package_code(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.py").write_text("x = 1\n", encoding="utf-8")
    before = run_fingerprint({}, None, package_dir=pkg)
    (pkg / "m.py").write_text("x = 2\n", encoding="utf-8")
    assert before != run_fingerprint({}, None, package_dir=pkg)


def test_language_does_not_change_fingerprint_or_cache(tmp_path, frame):
    counter = Counter()
    _, meta = _run(tmp_path, counter, frame)
    _, loaded = _run(tmp_path, counter, frame)
    pt = describe_run(loaded, language="pt")
    en = describe_run(loaded, language="en")
    short = meta["fingerprint"][:8]
    assert short in pt and short in en
    assert pt != en
    assert "carregados" in pt and "loaded" in en
    assert "treinados" in describe_run(meta, language="pt")
    assert "trained" in describe_run(meta, language="en")


def test_load_run_unchecked_ignores_fingerprint_and_reports_missing(tmp_path, frame):
    from ecom_delivery_sla.run_cache import load_run_unchecked

    counter = Counter()
    _, saved_meta = _run(tmp_path, counter, frame)

    results, meta = load_run_unchecked("tourn", "V1", tmp_path)
    assert counter.calls == 1
    assert meta["source"] == "loaded" and meta["fingerprint"] == saved_meta["fingerprint"]
    assert set(results) == {"one", "two"}
    with pytest.raises(ValueError):  # the checked loader would refuse a changed configuration
        _run(tmp_path, counter, frame, config={"k": 2}, mode="load")

    with pytest.raises(FileNotFoundError):
        load_run_unchecked("tourn", "V9", tmp_path)
    (tmp_path / "tourn" / "V1" / "one.pkl").unlink()
    with pytest.raises(FileNotFoundError):
        load_run_unchecked("tourn", "V1", tmp_path)
