"""O projeto deve estar registrado como suite propria no CI do portfolio.

No repositorio de portfolio (monorepo), o workflow raiz
``.github/workflows/tests.yml`` executa ``python -m pytest -q`` na pasta de
cada projeto listado em ``.github/ci/test-projects.json``. Fora do monorepo
(copia isolada do projeto), os testes deste arquivo sao pulados.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def monorepo_root():
    for parent in ROOT.parents:
        if (parent / ".github" / "ci" / "test-projects.json").exists():
            return parent
    return None


@pytest.fixture
def portfolio():
    root = monorepo_root()
    if root is None:
        pytest.skip("fora do repositorio de portfolio")
    return root


def test_project_is_a_matrix_suite_of_its_own(portfolio):
    config = json.loads((portfolio / ".github" / "ci" / "test-projects.json")
                        .read_text(encoding="utf-8"))
    name = ROOT.relative_to(portfolio).as_posix()
    entries = [e for e in config["matrix_projects"] if e["project"] == name]
    assert len(entries) == 1
    assert entries[0]["python"] == "3.12"
    assert ".[dev]" in entries[0]["install"]
    assert name not in config.get("no_suite", [])
    assert name not in config.get("conda_jobs", {}).values()


def test_root_workflow_runs_pytest_inside_the_project_folder(portfolio):
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (portfolio / ".github" / "workflows" / "tests.yml").read_text(
            encoding="utf-8"))
    job = workflow["jobs"]["tests"]
    assert job["defaults"]["run"]["working-directory"] == (
        "${{ matrix.project }}")
    runs = [s.get("run", "") for s in job["steps"]]
    assert "python -m pytest -q" in runs


def test_project_has_no_nested_workflow():
    # GitHub so le workflows na raiz do repositorio; um arquivo aqui seria
    # ignorado e daria a falsa impressao de um CI proprio.
    assert not (ROOT / ".github" / "workflows").exists()
