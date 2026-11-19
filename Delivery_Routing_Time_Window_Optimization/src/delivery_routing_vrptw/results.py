"""Loaders for the frozen result files (development instances only)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .io import Instance, Reference, read_archive, read_references


def assert_dev_only(instances: Iterable[str], split: Mapping[str, str]) -> None:
    """Raise if any instance is not a development instance of the split.

    Parameters
    ----------
    instances : iterable of str
        Instance names found in a result file.
    split : mapping
        Instance name -> ``"dev"`` or ``"eval"`` (see :func:`delivery_routing_vrptw.read_split`).

    Raises
    ------
    ValueError
        If an instance is missing from the split or belongs to the evaluation set.
    """
    bad = sorted({name for name in instances if split.get(name) != "dev"})
    if bad:
        raise ValueError(f"instances that are not development instances: {bad}")


def assert_eval_only(instances: Iterable[str], split: Mapping[str, str]) -> None:
    """Raise if any instance is not an evaluation instance of the split.

    Parameters
    ----------
    instances : iterable of str
        Instance names found in a result file.
    split : mapping
        Instance name -> ``"dev"`` or ``"eval"``.

    Raises
    ------
    ValueError
        If an instance is missing from the split or belongs to the development set.
    """
    bad = sorted({name for name in instances if split.get(name) != "eval"})
    if bad:
        raise ValueError(f"instances that are not evaluation instances: {bad}")


def _read(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)


def _json_or_none(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else None


def load_dev_grid(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the frozen development grid (tournament and reference cells).

    Parameters
    ----------
    path : str or pathlib.Path
        ``results.csv`` written by ``scripts/run_dev_grid.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``; every instance in the file must be ``dev``.

    Returns
    -------
    pandas.DataFrame
        One row per cell. ``routes`` and ``attempts`` are parsed from JSON into Python lists
        (``attempts`` is ``None`` outside the ``reduce_fleet`` strategy).

    Raises
    ------
    ValueError
        If any instance is not a development instance.
    """
    df = _read(path)
    assert_dev_only(df["instance"].unique(), split)
    df["routes"] = df["routes"].map(json.loads)
    df["attempts"] = df["attempts"].map(_json_or_none).astype(object)
    return df


def load_ranking(path: str | Path) -> pd.DataFrame:
    """Load the ranking of the 18 configurations (aggregated over instances, no instance column)."""
    return _read(path)


def load_gate(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the gate results; every instance must be a development instance.

    Parameters
    ----------
    path : str or pathlib.Path
        ``gate_results.csv`` written by ``scripts/run_gate.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, method).

    Raises
    ------
    ValueError
        If any instance is not a development instance.
    """
    df = _read(path)
    assert_dev_only(df["instance"].unique(), split)
    return df


def load_time_sensitivity(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the frozen time-budget sensitivity results; every instance must be a development instance.

    Parameters
    ----------
    path : str or pathlib.Path
        ``results.csv`` written by ``scripts/run_time_sensitivity.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, budget); ``routes`` is parsed from JSON into Python lists.

    Raises
    ------
    ValueError
        If any instance is not a development instance.
    """
    df = _read(path)
    assert_dev_only(df["instance"].unique(), split)
    df["routes"] = df["routes"].map(json.loads)
    return df


def load_selected_config(path: str | Path, split: Mapping[str, str] | None = None) -> dict:
    """Load the frozen configuration written by ``scripts/select_config.py``.

    Parameters
    ----------
    path : str or pathlib.Path
        ``selected_config.json``.
    split : mapping, optional
        When given, every instance listed in the file must be a development instance.

    Returns
    -------
    dict
        The parsed JSON (keys ``config``, ``selection_stats``, ``instances``, ...).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if split is not None:
        assert_dev_only(payload.get("instances", []), split)
    return payload


def get_routes(grid: pd.DataFrame, instance: str, **selection: str) -> list[list[int]]:
    """Return the routes of the single grid cell that matches the selection.

    Parameters
    ----------
    grid : pandas.DataFrame
        Output of :func:`load_dev_grid`.
    instance : str
        Instance name.
    **selection
        Column = value filters (for example ``method="ortools"``, ``first_solution="SAVINGS"``,
        ``metaheuristic="NONE"``).

    Returns
    -------
    list of list of int
        Customer ids per route.

    Raises
    ------
    ValueError
        If the selection does not match exactly one cell.
    """
    mask = grid["instance"] == instance
    for column, value in selection.items():
        mask &= grid[column] == value
    rows = grid[mask]
    if len(rows) != 1:
        raise ValueError(f"selection {instance} {selection} matches {len(rows)} cells, expected 1")
    return rows.iloc[0]["routes"]


def dev_names(split: Mapping[str, str]) -> list[str]:
    """Return the sorted names of the development instances of a split."""
    return sorted(name for name, which in split.items() if which == "dev")


def load_dev_instances(
    archive: str | Path, references: str | Path, split: Mapping[str, str]
) -> tuple[dict[str, Instance], dict[str, Reference]]:
    """Load only the development instances and their SINTEF references.

    Evaluation instances are dropped right after the archive is read and never returned.

    Parameters
    ----------
    archive : str or pathlib.Path
        SINTEF ``solomon-100.zip``.
    references : str or pathlib.Path
        SINTEF best-known-solution CSV.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    tuple of dict
        ``(instances, references)`` keyed by development instance name.
    """
    names = dev_names(split)
    instances = read_archive(archive)
    refs = read_references(references)
    return {n: instances[n] for n in names}, {n: refs[n] for n in names}


def load_dev_inputs(archive: str | Path, split: Mapping[str, str]) -> dict[str, Instance]:
    """Load only the development instances, without any SINTEF reference (for the business layer).

    Parameters
    ----------
    archive : str or pathlib.Path
        SINTEF ``solomon-100.zip`` (used only as the source of the instance files).
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    dict
        Development instances keyed by name; evaluation instances are never returned.
    """
    instances = read_archive(archive)
    return {n: instances[n] for n in dev_names(split)}


def load_evaluation(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the frozen final-evaluation results; every instance must be an evaluation instance.

    Parameters
    ----------
    path : str or pathlib.Path
        ``results.csv`` written by ``scripts/run_evaluation.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per cell (OR-Tools replicate or baseline); ``routes`` is parsed from JSON into lists.

    Raises
    ------
    ValueError
        If any instance is not an evaluation instance.
    """
    df = _read(path)
    assert_eval_only(df["instance"].unique(), split)
    df["routes"] = df["routes"].map(json.loads)
    return df


def load_eval_time_metrics(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the time metrics derived from the frozen evaluation routes (``time_metrics.csv``).

    Parameters
    ----------
    path : str or pathlib.Path
        File written by ``scripts/derive_eval_time_metrics.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per evaluation cell (instance, method, replicate) with the time metrics.

    Raises
    ------
    ValueError
        If any instance is not an evaluation instance.
    """
    df = _read(path)
    assert_eval_only(df["instance"].unique(), split)
    df["feasible"] = df["feasible"].astype(str).str.lower().eq("true")
    return df


def load_eval_instances(
    archive: str | Path, split: Mapping[str, str], names: Iterable[str]
) -> dict[str, Instance]:
    """Load the requested evaluation instances (for route maps of the final evaluation).

    Parameters
    ----------
    archive : str or pathlib.Path
        SINTEF ``solomon-100.zip``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.
    names : iterable of str
        Instances to return; each must be an evaluation instance.

    Returns
    -------
    dict
        Instances keyed by name.

    Raises
    ------
    ValueError
        If a requested instance is not an evaluation instance.
    """
    names = list(names)
    assert_eval_only(names, split)
    instances = read_archive(archive)
    return {n: instances[n] for n in names}


def load_window_scenarios(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the frozen window-widening results (layer B); every instance must be a development instance.

    Parameters
    ----------
    path : str or pathlib.Path
        ``results.csv`` written by ``scripts/run_window_scenarios.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, factor); ``routes`` is parsed from JSON into lists.

    Raises
    ------
    ValueError
        If any instance is not a development instance.
    """
    df = _read(path)
    assert_dev_only(df["instance"].unique(), split)
    df["routes"] = df["routes"].map(json.loads)
    return df


def load_early_delivery(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the frozen early-delivery results (layer B); every instance must be a development instance.

    Parameters
    ----------
    path : str or pathlib.Path
        ``results.csv`` written by ``scripts/run_early_delivery.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, tolerance, replicate); ``tolerance_label`` is text (``"0"``, ``"0.25"``,
        ..., ``"full"``) and ``routes`` is parsed from JSON into lists.

    Raises
    ------
    ValueError
        If any instance is not a development instance.
    """
    df = _read(path, dtype={"tolerance_label": str})
    assert_dev_only(df["instance"].unique(), split)
    df["routes"] = df["routes"].map(json.loads)
    return df


# --------------------------------------------------------------------------- protocol v2, phase C and phase D
def load_v2_block(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load a frozen protocol-v2 solver block (B1 to B4); every instance must be a development instance.

    Parameters
    ----------
    path : str or pathlib.Path
        ``results.csv`` of ``outputs/frozen/v2/<block>/`` (written by ``scripts/run_v2_solver_grid.py``).
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, level, replicate); ``routes`` is parsed from JSON into lists. The columns
        ``scale``, ``k`` and ``workers`` identify the level.

    Raises
    ------
    ValueError
        If any instance is not a development instance.
    """
    df = _read(path)
    assert_dev_only(df["instance"].unique(), split)
    df["routes"] = df["routes"].map(json.loads)
    return df


def load_v2_decision(path: str | Path, split: Mapping[str, str]) -> dict:
    """Load the ``decision.json`` of a protocol-v2 block; the instances it mentions must be development ones.

    Parameters
    ----------
    path : str or pathlib.Path
        ``decision.json`` written by ``scripts/select_v2.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    dict
        The decision record (``block``, ``winner``, ``decided_by``, ``ranking``, ``representatives``, ...).

    Raises
    ------
    ValueError
        If a representative plan belongs to a non-development instance.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_dev_only([r["instance"] for r in record.get("representatives", [])], split)
    return record


def load_evaluation_comparison(path: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the v1-versus-v2 per-instance comparison of the final evaluation (evaluation instances only).

    Parameters
    ----------
    path : str or pathlib.Path
        ``comparison_by_instance.csv`` written by ``scripts/compare_eval_versions.py``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per evaluation instance.

    Raises
    ------
    ValueError
        If any instance is not an evaluation instance.
    """
    df = _read(path)
    assert_eval_only(df["instance"].unique(), split)
    return df


def load_phase_c_cells(folder: str | Path, split: Mapping[str, str]) -> pd.DataFrame:
    """Load the cells of a phase C solver block (C2 or C3); every instance must be a development instance.

    Parameters
    ----------
    folder : str or pathlib.Path
        Block folder (for example ``outputs/frozen/phase_c/C2/capacity``) that holds ``results.csv``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    pandas.DataFrame
        One row per cell; ``level`` and ``variant`` are text, ``feasible`` is boolean and ``routes`` stays a
        JSON string (the summaries of :mod:`delivery_routing_vrptw.phase_c` parse it).

    Raises
    ------
    ValueError
        If any instance is not a development instance.
    """
    df = _read(Path(folder) / "results.csv", dtype={"level": str, "variant": str})
    assert_dev_only(df["instance"].unique(), split)
    df["variant"] = df["variant"].fillna("")
    df["feasible"] = df["feasible"].astype(str).str.lower().eq("true")
    for col in ("capacity_infeasible",):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def load_robustness(folder: str | Path, split: Mapping[str, str]) -> dict[str, pd.DataFrame]:
    """Load the phase C1 robustness run (stress tests of the evaluation plans, descriptive use only).

    Parameters
    ----------
    folder : str or pathlib.Path
        ``outputs/frozen/phase_c/C1``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.

    Returns
    -------
    dict of pandas.DataFrame
        ``cells`` (one row per plan and scenario), ``summary`` (means by scope, method and scenario),
        ``slack_lateness`` (late rate by nominal-slack bin) and ``slack_correlation`` (Spearman correlation
        between slack and late rate).

    Raises
    ------
    ValueError
        If any instance of the run is not an evaluation instance.
    """
    base = Path(folder)
    cells = _read(base / "results.csv")
    assert_eval_only(cells["instance"].unique(), split)
    cells["plan_feasible"] = cells["plan_feasible"].astype(str).str.lower().eq("true")
    return {
        "cells": cells,
        "summary": _read(base / "summary_by_family_method.csv"),
        "slack_lateness": _read(base / "slack_lateness.csv", dtype={"slack_bin": str}),
        "slack_correlation": _read(base / "slack_correlation.csv"),
    }
