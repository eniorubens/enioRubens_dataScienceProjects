"""The notebook builder produces PT/EN pairs that satisfy the notebook contract."""

import importlib.util
import re
import sys

import pytest

from retail_multilevel.nbcontract import check_notebooks
from retail_multilevel.paths import ROOT


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "scripts" / "build_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_writes_contract_compliant_pairs(builder, tmp_path):
    root = tmp_path / "notebooks"
    paths = builder.build(out_dir=root)
    assert len(paths) == 2 * len(builder.NOTEBOOKS)
    assert all(p.exists() for p in paths)
    assert check_notebooks(root) == []


def test_only_filters_by_number(builder, tmp_path):
    assert builder.build(out_dir=tmp_path, only="99") == []
    assert len(builder.build(out_dir=tmp_path, only="01")) == 2


def test_every_load_run_call_pins_a_run_id(builder):
    """Notebooks must read stored runs by explicit id, never the latest run of an experiment."""
    call = re.compile(r"load_run\(([^()]*)\)")
    offenders = []
    for spec in builder.NOTEBOOKS:
        for lang in ("pt", "en"):
            for i, cell in enumerate(builder.render(spec, lang).cells):
                if cell.cell_type != "code":
                    continue
                for args in call.findall(cell.source):
                    if len([a for a in args.split(",") if a.strip()]) < 2:
                        offenders.append(f"{spec.number}/{lang} cell {i}: load_run({args})")
    assert offenders == []
