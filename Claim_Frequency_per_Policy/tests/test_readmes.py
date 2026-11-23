"""Paridade estrutural dos READMEs PT-BR e EN-US."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def headings(text):
    return re.findall(r"^(#{1,3}) (\d+)?", text, flags=re.M)


def blocks(text):
    return [b.split("```")[0] for b in text.split("```powershell")[1:]]


def commands(text):
    out = []
    for block in blocks(text):
        out.append([re.sub(r"\s+#.*$", "", line).strip()
                    for line in block.splitlines() if line.strip()])
    return out


def test_readmes_have_the_same_structure():
    pt, en = read("README.pt-BR.md"), read("README.en-US.md")
    assert headings(pt) == headings(en)
    assert pt.count("```") == en.count("```")
    assert pt.count("|") == en.count("|")


def test_readmes_report_the_same_final_numbers():
    pt, en = read("README.pt-BR.md"), read("README.en-US.md")
    for value in ("0,31902", "0,33079", "−0,01177", "101.859", "5.415",
                  "−0,233", "+0,317", "−0,253", "+0,288", "0,050", "0,027"):
        assert value in pt
    for value in ("0.31902", "0.33079", "−0.01177", "101,859", "5,415",
                  "−0.233", "+0.317", "−0.253", "+0.288", "0.050", "0.027"):
        assert value in en


def test_readmes_use_the_same_commands_in_every_block():
    pt, en = commands(read("README.pt-BR.md")), commands(
        read("README.en-US.md"))
    assert len(pt) == len(en) == 3
    assert pt == en


def test_readmes_separate_consultation_from_reproduction():
    for name in ("README.pt-BR.md", "README.en-US.md"):
        text = read(name)
        consult, repro = commands(text)[1], commands(text)[2]
        assert any("verify-stage5" in c for c in consult)
        assert any("--final" in c for c in consult)
        assert not any(" stage5" in c or c.endswith("stage4")
                       for c in repro)
        assert any("verify-freeze" in c for c in repro)
        assert any("robocopy" in c for c in repro)


def test_readmes_declare_the_lock_local_data_and_ci_state():
    for name in ("README.pt-BR.md", "README.en-US.md"):
        text = read(name)
        assert "official_run.lock" in text
        assert "data/raw" in text and "data/splits" in text
        assert "production_eligible = false" in text
        assert "test-projects.json" in text
