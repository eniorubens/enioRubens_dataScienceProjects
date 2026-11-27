"""Replace machine-specific absolute paths in tracked text files with neutral placeholders.

Executed notebooks record local paths in their outputs (scikit-learn warnings show the conda
environment, IPython warnings show the kernel's temporary folder, and the F1/F2 provenance tables
show the absolute path of the frozen model). This script rewrites those paths before publishing:

===============================================  ==============
Local path (shown with forward slashes)          Placeholder
===============================================  ==============
``<drive>:/Users/<user>/miniforge3/envs/<env>``  ``CONDA_ENV``
``<drive>:/Users/<user>/miniforge3``             ``CONDA_BASE``
``<drive>:/Users/<user>/AppData/Local/Temp``     ``TEMP``
``<drive>:/Users/<user>``                        ``USER_HOME``
the project root (``<drive>:/.../<folder>``)     ``PROJECT_ROOT``
===============================================  ==============

It works on the raw file text, so JSON-escaped (doubled backslashes), HTML-escaped (``&amp;``) and
plain forms are all covered, as are project paths truncated by pandas (``...``); nothing else in the
file changes (code cells keep their bytes). Run it with ``--check`` to only report what would change
(exit code 1 when something is found).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEP = r"(?:\\{1,4}|/)"
SKIP = {Path(__file__).resolve(), (ROOT / "tests" / "test_anonymize_local_paths.py").resolve()}
TEXT_SUFFIXES = {".ipynb", ".md", ".csv", ".html", ".json", ".jsonl", ".txt", ".py", ".yml", ".yaml"}


def project_root_pattern(root: Path) -> re.Pattern:
    """Return a pattern for the project root in any escaping (backslashes, slashes, ``&amp;``)."""

    def escape(part: str) -> str:
        return re.escape(part).replace("\\&", "(?:&|&amp;)")

    parts = [escape(part) for part in root.parts[1:]]
    last = root.parts[-1]
    # Pandas may truncate long cells ("...-SLA-..."): accept a prefix of the folder name before "...".
    truncated = "|".join(escape(last[:size]) for size in range(len(last) - 1, 9, -1))
    parts[-1] = rf"(?:{parts[-1]}|(?:{truncated})(?=\.\.\.|…))"
    drive = re.escape(root.drive.rstrip(":"))
    body = SEP.join(parts)
    return re.compile(rf"(?i)\b{drive}:{SEP}{body}(?P<sep>{SEP})?")


def build_rules(root: Path, user: str | None = None) -> list[tuple[re.Pattern, str]]:
    """Return (pattern, placeholder) rules, most specific first."""
    user_part = re.escape(user) if user else r"[^\\/\s\"'<>|]+"
    home = rf"(?i)\b[a-z]:{SEP}users{SEP}{user_part}"
    return [
        (re.compile(rf"{home}{SEP}miniforge3{SEP}envs{SEP}[^\\/\s\"'<>|]+(?P<sep>{SEP})?"), "CONDA_ENV"),
        (re.compile(rf"{home}{SEP}miniforge3(?P<sep>{SEP})?"), "CONDA_BASE"),
        (re.compile(rf"{home}{SEP}AppData{SEP}Local{SEP}Temp(?P<sep>{SEP})?"), "TEMP"),
        (project_root_pattern(root), "PROJECT_ROOT"),
        (re.compile(rf"{home}(?P<sep>{SEP})?"), "USER_HOME"),
    ]


def anonymize_text(text: str, rules: list[tuple[re.Pattern, str]]) -> tuple[str, int]:
    """Apply the rules to a text and return the new text and the number of replacements."""
    total = 0
    for pattern, placeholder in rules:
        text, count = pattern.subn(lambda match, p=placeholder: p + (match.group("sep") or ""), text)
        total += count
    return text, total


def tracked_files(root: Path) -> list[Path]:
    """Return the tracked text files of the repository."""
    listing = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True)
    return [root / rel for rel in listing.stdout.splitlines() if Path(rel).suffix.lower() in TEXT_SUFFIXES]


def main(argv: list[str] | None = None) -> int:
    """Rewrite (or, with --check, report) local paths in the tracked text files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="report only; exit 1 if any path is found")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    rules = build_rules(root)
    found = 0
    for path in tracked_files(root):
        if path.resolve() in SKIP:  # the script and its tests hold synthetic example paths
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new_text, count = anonymize_text(text, rules)
        if count:
            found += count
            print(f"{count:5d}  {path.relative_to(root).as_posix()}")
            if not arguments.check:
                path.write_bytes(new_text.encode("utf-8"))
    print(f"{'found' if arguments.check else 'replaced'}: {found}")
    return 1 if arguments.check and found else 0


if __name__ == "__main__":
    sys.exit(main())
