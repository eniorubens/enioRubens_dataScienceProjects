"""Tests for scripts/anonymize_local_paths.py on synthetic strings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import PureWindowsPath

SCRIPT = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "anonymize_local_paths.py"
_spec = importlib.util.spec_from_file_location("anonymize_local_paths", SCRIPT)
anonymize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(anonymize)

ROOT = PureWindowsPath(r"D:\Work\E-commerce-Delivery-Network-&-SLA-Risk")
RULES = anonymize.build_rules(ROOT)


def run(text: str) -> str:
    return anonymize.anonymize_text(text, RULES)[0]


def test_conda_env_temp_and_project_root_in_plain_text():
    text = (
        r"c:\Users\someone\miniforge3\envs\ecom-delivery-sla\Lib\site-packages\sklearn\x.py:262: UserWarning"
        "\n"
        r"C:\Users\someone\AppData\Local\Temp\ipykernel_3128\3819203259.py:1: FutureWarning"
        "\n"
        r"d:\Work\E-commerce-Delivery-Network-&-SLA-Risk\outputs\models\catboost_final\F1\model.joblib"
        "\n"
        r"runtime C:\Users\someone\miniforge3\python.exe"
    )
    assert run(text).splitlines() == [
        r"CONDA_ENV\Lib\site-packages\sklearn\x.py:262: UserWarning",
        r"TEMP\ipykernel_3128\3819203259.py:1: FutureWarning",
        r"PROJECT_ROOT\outputs\models\catboost_final\F1\model.joblib",
        r"runtime CONDA_BASE\python.exe",
    ]


def test_json_escaped_html_escaped_and_truncated_forms():
    payload = json.dumps({"text": r"d:\Work\E-commerce-Delivery-Network-&-SLA-Risk\outputs\runs"})
    assert json.loads(run(payload))["text"] == r"PROJECT_ROOT\outputs\runs"
    assert run(r"d:\Work\E-commerce-Delivery-Network-&amp;-SLA-Risk\outputs") == r"PROJECT_ROOT\outputs"
    assert run(r"d:\\Work\\E-commerce-Delivery-Network-&-SLA-...") == r"PROJECT_ROOT..."


def test_text_without_local_paths_is_unchanged():
    text = "outputs/runs/final_evaluation/F1 and C:/ is not a user folder; D:\\Data\\file.csv"
    assert anonymize.anonymize_text(text, RULES) == (text, 0)
