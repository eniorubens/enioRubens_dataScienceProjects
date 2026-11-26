"""Compare C0 and C1 by ML on the training panel and calibrate the joint LR under C0.

Use ``--replicates 0`` for the observed comparison, ``--replicates 5`` for the default pilot,
and ``--replicates 39 --resume-from RUN_ID`` to complete the fixed Monte Carlo design.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import pandas as pd

from retail_multilevel import (
    discount_slope_bootstrap as boot,
)
from retail_multilevel import (
    discount_slope_lr as slopes,
)
from retail_multilevel import features, fixedeffects, multilevel, p1, rbridge, split
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import RUNS_DIR, Run, fingerprint, get_or_fit, load_run, module_files

EXPERIMENT = "discount_slope_lr"
PANEL_PATH = PROCESSED_DIR / "panel.parquet"
DEFAULT_SEED = 20261125
DEFAULT_PLANNED = 39
_MAX_R_SEED = 2**31 - 1
_RUN_ID = re.compile(r"\d{8}-\d{6}-[a-f0-9]{8}(?:-\d+)?\Z")
_DRAW_COLUMNS = ("replicate", "seed", "lr", "valid", "singular_c0", "singular_c1", "seconds", "message")


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=5, help="cumulative attempted draws, 0 for observed LR only")
    parser.add_argument("--planned", type=int, default=DEFAULT_PLANNED, help="fixed size of the Monte Carlo design")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the deterministic sequence of R seeds")
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    parser.add_argument(
        "--resume-from", metavar="RUN_ID", help="reuse draws from a compatible completed or interrupted run"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.planned <= _MAX_R_SEED:
        parser.error(f"--planned must be between 1 and {_MAX_R_SEED}")
    if not 0 <= args.replicates <= args.planned:
        parser.error("--replicates must be between 0 and --planned")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.resume_from and not _RUN_ID.fullmatch(args.resume_from):
        parser.error("--resume-from must be a run ID such as 20261125-143000-01234567")
    if args.resume_from and args.replicates == 0:
        parser.error("--resume-from needs a positive --replicates target")
    return args


def _save_checkpoint(run: Run, draws: pd.DataFrame) -> None:
    """Replace the aggregate checkpoint only after its temporary Parquet is complete."""
    path = run.file("bootstrap_checkpoint.parquet")
    pending = path.with_suffix(".tmp")
    draws.to_parquet(pending, index=False)
    pending.replace(path)


def _resume(
    run_id: str | None, design_hash: str, seeds: dict[int, int], target: int
) -> tuple[pd.DataFrame, float | None]:
    if run_id is None:
        return pd.DataFrame(columns=_DRAW_COLUMNS), None
    directory = RUNS_DIR / EXPERIMENT / run_id
    checkpoint = directory / "files" / "bootstrap_checkpoint.parquet"
    manifest = Run(directory)
    details = manifest.load_json("bootstrap_design")
    if details.get("fingerprint") != design_hash:
        raise ValueError("cannot resume: code, panel, seed or planned Monte Carlo design differs")
    if manifest.is_complete:
        draws = load_run(EXPERIMENT, run_id).load_table("bootstrap")
    elif checkpoint.is_file():
        draws = pd.read_parquet(checkpoint)
    else:
        draws = pd.DataFrame(columns=_DRAW_COLUMNS)
    used = draws["replicate"].tolist()
    if sorted(used) != list(range(1, len(used) + 1)) or len(used) > target:
        raise ValueError("resume draws must be unique, contiguous and within the requested cumulative target")
    if draws["seed"].tolist() != [seeds[i] for i in used]:
        raise ValueError("saved replicate seeds differ from this Monte Carlo design")
    observed_lr = details["observed_lr"]
    if not isinstance(observed_lr, (int, float)) or not math.isfinite(observed_lr):
        raise ValueError("the resume manifest does not contain a finite observed LR")
    return draws.sort_values("replicate", ignore_index=True), float(observed_lr)


def main(argv: list[str] | None = None) -> None:
    """Run or read a training-only comparison and store aggregate results."""
    args = _arguments(argv)
    design = {
        "sample": "train",
        "week_min": 0,
        "week_max": split.FIRST_TEST_WEEK - 1,
        "in_model": True,
        "c0": slopes.C0_FORMULA,
        "c1": slopes.C1_FORMULA,
        "estimator": "ML for both models on the same rows",
        "bootstrap_null": "unconditional simulation under fitted C0; refit both models per draw",
        "bootstrap_limitations": "model-conditional calibration; C0 assumes independent residual errors",
        "seed": args.seed,
        "planned": args.planned,
        "tail": "inclusive",
    }
    code = module_files(slopes, boot, features, fixedeffects, multilevel, rbridge, split, p1)
    code.append(Path(__file__).resolve())
    design_hash = fingerprint(design, inputs=[PANEL_PATH], code_files=code)
    config = design | {"design_fingerprint": design_hash, "replicates": args.replicates}
    seed_order = random.Random(args.seed).sample(range(1, _MAX_R_SEED + 1), args.planned)
    seeds = dict(enumerate(seed_order, start=1))
    existing, previous_lr = _resume(args.resume_from, design_hash, seeds, args.replicates)

    def fit(run: Run) -> dict:
        started = time.perf_counter()
        print(f"run: {run.run_id}", flush=True)
        train = pd.read_parquet(
            PANEL_PATH,
            engine="pyarrow",
            filters=[("WEEK", ">=", 0), ("WEEK", "<", split.FIRST_TEST_WEEK), ("IN_MODEL", "==", True)],
        )
        counts = {"n_pairs": int(train["PAIR_ID"].nunique()), "max_week": int(train["WEEK"].max())}
        df = slopes.prepare_training(train)
        del train
        c0, c1 = slopes.fit_observed(df)
        if c0.nobs != len(df) or c1.nobs != len(df):
            raise RuntimeError("both models must include every prepared training row")
        observed_lr = slopes.likelihood_ratio(c0, c1)
        if previous_lr is not None and not math.isclose(observed_lr, previous_lr, rel_tol=1e-9, abs_tol=1e-6):
            raise RuntimeError("observed LR differs from the resumed run")
        run.save_table("fits", pd.DataFrame([m.summary_row() | {"messages": " | ".join(m.messages)} for m in (c0, c1)]))
        run.save_table("varcomp", pd.concat([m.varcomp.assign(model=m.name) for m in (c0, c1)], ignore_index=True))
        run.save_json("bootstrap_design", {"fingerprint": design_hash, "observed_lr": observed_lr})
        print(f"observed LR={observed_lr:.6f}; rows={c0.nobs:,}", flush=True)

        records = existing.to_dict(orient="records")
        if records:
            _save_checkpoint(run, existing)

        def record(row: dict) -> None:
            records.append(row)
            _save_checkpoint(run, pd.DataFrame(records))
            print(
                f"draw {row['replicate']}/{args.replicates}: valid={row['valid']} ({row['seconds']:.1f} s)", flush=True
            )

        pending = [(i, seeds[i]) for i in range(len(existing) + 1, args.replicates + 1)]
        if pending:
            boot.bootstrap(c0, c1, pending, on_result=record)
        draws = pd.DataFrame(records) if records else existing
        run.save_table("bootstrap", draws)
        summary = (
            boot.bootstrap_summary(observed_lr, draws, args.planned)
            | counts
            | {
                "nobs": len(df),
                "requested": args.replicates,
                "resumed_from": args.resume_from,
                "seconds_total": time.perf_counter() - started,
            }
        )
        run.save_json("summary", summary)
        return summary

    run = get_or_fit(EXPERIMENT, config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    print(json.dumps(run.load_json("summary"), indent=2, ensure_ascii=False))
    print(f"run: {run.path}")


if __name__ == "__main__":
    main()
