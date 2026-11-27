"""Tests for scripts/anonymize_local_paths.py on synthetic strings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PureWindowsPath

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "anonymize_local_paths.py"
_spec = importlib.util.spec_from_file_location("anonymize_local_paths", SCRIPT)
anonymize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(anonymize)

ROOT = PureWindowsPath(r"D:\Work\Delivery_Routing_&_Time_Window_Optimization")
RULES = anonymize.build_rules(ROOT)


def run(text: str) -> str:
    return anonymize.anonymize_text(text, RULES)[0]


def test_conda_env_temp_and_project_root_in_plain_text():
    text = (
        r"c:\Users\someone\miniforge3\envs\delivery-routing-vrptw\Lib\site-packages\ortools\x.py:262: UserWarning"
        "\n"
        r"C:\Users\someone\AppData\Local\Temp\ipykernel_3128\3819203259.py:1: FutureWarning"
        "\n"
        r"d:\Work\Delivery_Routing_&_Time_Window_Optimization\outputs\frozen\evaluation\run_metadata.json"
        "\n"
        r"runtime C:\Users\someone\miniforge3\python.exe"
    )
    assert run(text).splitlines() == [
        r"CONDA_ENV\Lib\site-packages\ortools\x.py:262: UserWarning",
        r"TEMP\ipykernel_3128\3819203259.py:1: FutureWarning",
        r"PROJECT_ROOT\outputs\frozen\evaluation\run_metadata.json",
        r"runtime CONDA_BASE\python.exe",
    ]


def test_json_escaped_html_escaped_and_truncated_forms():
    payload = json.dumps({"text": r"d:\Work\Delivery_Routing_&_Time_Window_Optimization\outputs\runs"})
    assert json.loads(run(payload))["text"] == r"PROJECT_ROOT\outputs\runs"
    assert run(r"d:\Work\Delivery_Routing_&amp;_Time_Window_Optimization\outputs") == r"PROJECT_ROOT\outputs"
    assert run(r"d:\\Work\\Delivery_Routing_&_Time_...") == r"PROJECT_ROOT..."


def test_frozen_metadata_path_in_json():
    path = r"D:\Work\Delivery_Routing_&_Time_Window_Optimization\reports\protocolo-fase-c.md"
    payload = json.dumps({"protocol": path})
    assert json.loads(run(payload))["protocol"] == r"PROJECT_ROOT\reports\protocolo-fase-c.md"


def test_text_without_local_paths_is_unchanged():
    text = "outputs/runs/final_evaluation/F1 and C:/ is not a user folder; D:\\Data\\file.csv"
    assert anonymize.anonymize_text(text, RULES) == (text, 0)
