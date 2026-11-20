#!/usr/bin/env python3
"""Select which test suites the Tests workflow must run.

Input is the output of ``git diff --name-status -z -M <base> <head>`` (read from a
file), or ``--full`` to force the whole battery. Output is written as
``key=value`` lines to ``$GITHUB_OUTPUT`` (or stdout when unset):

* ``matrix``     JSON ``{"include": [...]}`` for the pip matrix job
* ``has_matrix`` ``true`` when at least one matrix suite is selected
* ``bike``       ``true`` when the Bike-Sharing conda job must run
* ``mode``       ``full`` or ``selective``
* ``reason``     one-line human explanation

Rules (see .github/ci/test-projects.json):

1. A path inside a project folder selects that project's suite.
2. A path inside a ds_toolkit package selects the package's own tests (if any)
   plus every consumer listed under ``shared``. Any other ds_toolkit path
   (root files, unknown packages) is uncertain and forces the full battery.
3. Editorial root files (root READMEs, LICENSE, the static showcase) select
   nothing, and so do project folders listed under ``no_suite``.
4. Everything else (``.github/**``, root config such as .gitignore, unknown
   top-level folders) forces the full battery.
5. Renames and copies classify both the old and the new path.
6. An unparsable diff forces the full battery.

Uses only the standard library so the detect job needs no setup.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CONFIG = Path(__file__).with_name("test-projects.json")


class Uncertain(Exception):
    """The impact of a change cannot be bounded; run everything."""


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def all_suites(cfg: dict) -> set[str]:
    return {p["project"] for p in cfg["matrix_projects"]} | set(cfg["conda_jobs"].values())


def parse_name_status(raw: str) -> list[str]:
    """Return every path touched by a ``git diff --name-status -z`` dump.

    For renames (R) and copies (C) both the source and the destination are returned.
    """
    tokens = raw.split("\0")
    if tokens and tokens[-1] == "":
        tokens.pop()
    paths: list[str] = []
    i = 0
    while i < len(tokens):
        status = tokens[i]
        if not status or status[0] not in "ACDMRTUXB":
            raise Uncertain(f"unexpected diff status token {status!r}")
        n = 2 if status[0] in "RC" else 1
        chunk = tokens[i + 1 : i + 1 + n]
        if len(chunk) != n or any(not p for p in chunk):
            raise Uncertain("truncated diff output")
        if status[0] in "XUB":
            raise Uncertain(f"diff reports status {status!r}")
        paths.extend(chunk)
        i += 1 + n
    return paths


def suites_for_path(path: str, cfg: dict) -> set[str]:
    """Suites affected by one path. Raises Uncertain when it must be the full battery."""
    for entry in cfg["editorial"]:
        if (entry.endswith("/") and _under(path, entry)) or path == entry:
            return set()
    for folder in cfg["no_suite"]:
        if _under(path, folder):
            return set()
    # Shared packages first: their consumers live outside the package folder.
    for package, consumers in cfg["shared"].items():
        if _under(path, package):
            return set(consumers)
    if _under(path, "ds_toolkit"):
        raise Uncertain(f"ds_toolkit path with unmapped consumers: {path}")
    for suite in sorted(all_suites(cfg), key=len, reverse=True):
        if _under(path, suite):
            return {suite}
    raise Uncertain(f"global or unmapped path: {path}")


def select(paths: list[str], cfg: dict) -> tuple[set[str], str]:
    selected: set[str] = set()
    for path in paths:
        selected |= suites_for_path(path, cfg)
    if not paths:
        return selected, "no changed files"
    if not selected:
        return selected, "only editorial or suite-less paths changed"
    return selected, f"{len(paths)} changed path(s) map to {len(selected)} suite(s)"


def build_outputs(selected: set[str], mode: str, reason: str, cfg: dict) -> dict[str, str]:
    include = [p for p in cfg["matrix_projects"] if p["project"] in selected]
    include = [{k: v for k, v in p.items() if not k.startswith("_")} for p in include]
    bike = cfg["conda_jobs"]["bike-sharing"] in selected
    return {
        "matrix": json.dumps({"include": include}, ensure_ascii=False, separators=(",", ":")),
        "has_matrix": "true" if include else "false",
        "bike": "true" if bike else "false",
        "mode": mode,
        "reason": reason.replace("\n", " "),
        "suites": json.dumps(sorted(selected), ensure_ascii=False),
    }


def decide(cfg: dict, *, full_reason: str | None, diff_text: str | None) -> dict[str, str]:
    if full_reason is None:
        try:
            if diff_text is None:
                raise Uncertain("diff not available")
            selected, reason = select(parse_name_status(diff_text), cfg)
            return build_outputs(selected, "selective", reason, cfg)
        except Uncertain as exc:
            full_reason = str(exc)
    return build_outputs(all_suites(cfg), "full", full_reason, cfg)


def write_outputs(outputs: dict[str, str]) -> None:
    target = os.environ.get("GITHUB_OUTPUT")
    lines = "".join(f"{k}={v}\n" for k, v in outputs.items())
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(lines)
    else:
        sys.stdout.write(lines)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        suites = json.loads(outputs["suites"])
        body = [
            "## Test selection",
            f"- Mode: **{outputs['mode']}**",
            f"- Reason: {outputs['reason']}",
            f"- Suites ({len(suites)}): " + (", ".join(f"`{s}`" for s in suites) or "none"),
            "",
        ]
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("\n".join(body) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", metavar="REASON", help="force the full battery")
    ap.add_argument("--diff-file", type=Path, help="git diff --name-status -z -M output")
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    diff_text = None
    if args.full is None and args.diff_file is not None and args.diff_file.exists():
        diff_text = args.diff_file.read_text(encoding="utf-8", errors="surrogateescape")
    outputs = decide(cfg, full_reason=args.full, diff_text=diff_text)
    write_outputs(outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
