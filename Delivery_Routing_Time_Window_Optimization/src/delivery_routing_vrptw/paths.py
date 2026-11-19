"""Project-relative locations, so that notebooks do not hard-code paths."""

from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    """Return the nearest ancestor folder (or ``start`` itself) that contains ``pyproject.toml``.

    Parameters
    ----------
    start : str or pathlib.Path, optional
        Folder where the search begins; defaults to the current working directory.

    Returns
    -------
    pathlib.Path
        The project root.

    Raises
    ------
    FileNotFoundError
        If no ancestor contains ``pyproject.toml``.
    """
    here = Path.cwd() if start is None else Path(start)
    here = here.resolve()
    for folder in (here, *here.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"no pyproject.toml found from {here} upwards")


def data_paths(root: str | Path | None = None) -> dict[str, Path]:
    """Return the locations of the versioned data files.

    Parameters
    ----------
    root : str or pathlib.Path, optional
        Project root; discovered with :func:`project_root` if omitted.

    Returns
    -------
    dict
        Name -> path. Keys: ``root``, ``sintef_zip``, ``sintef_manifest``, ``cvrplib_instances``,
        ``cvrplib_solutions``, ``cvrplib_manifest``, ``sintef_bks``, ``cvrplib_bks``, ``split``.
    """
    base = project_root() if root is None else Path(root)
    raw = base / "data" / "raw"
    ref = base / "data" / "reference"
    return {
        "root": base,
        "sintef_zip": raw / "sintef" / "solomon-100.zip",
        "sintef_manifest": raw / "sintef" / "MANIFEST.md",
        "cvrplib_instances": raw / "cvrplib_solomon_100" / "Solomon.7z",
        "cvrplib_solutions": raw / "cvrplib_solomon_100" / "Solomon-solutions.7z",
        "cvrplib_manifest": raw / "cvrplib_solomon_100" / "MANIFEST.md",
        "sintef_bks": ref / "sintef_bks_100.csv",
        "cvrplib_bks": ref / "cvrplib_bks_100.csv",
        "split": ref / "split.csv",
    }


def result_paths(root: str | Path | None = None) -> dict[str, Path]:
    """Return the locations of the frozen result files read by the notebooks.

    Parameters
    ----------
    root : str or pathlib.Path, optional
        Project root; discovered with :func:`project_root` if omitted.

    Returns
    -------
    dict
        Name -> path. Keys: ``root``, ``frozen``, ``selected_config``, ``dev_grid``, ``ranking``,
        ``dev_grid_metadata``, ``gate``, ``time_sensitivity``, ``evaluation``, ``window_scenarios``,
        ``window_scenarios_contended``, ``early_delivery``, ``early_delivery_lex`` (derived summary),
        ``eval_time_metrics`` (derived time metrics of the evaluation plans). Protocol v2 (phase B, development
        instances only): ``v2_B1`` ... ``v2_B4`` (result files), ``v2_B1_ranking`` ... ``v2_B3_ranking``,
        ``v2_B1_decision`` ... ``v2_B3_decision``, ``v2_B5_windows`` and ``v2_B5_early``. Phase C: ``phase_c_C1``
        (folder of the robustness run on evaluation plans), ``phase_c_capacity``, ``phase_c_fleet_priority``,
        ``phase_c_soft_windows``, ``phase_c_removal`` and ``phase_c_insertion`` (folders; development instances
        only). Phase D (post-hoc, evaluation instances): ``evaluation_v2`` and ``evaluation_v2_comparison``.
    """
    base = project_root() if root is None else Path(root)
    frozen = base / "outputs" / "frozen"
    return {
        "root": base,
        "frozen": frozen,
        "selected_config": frozen / "selected_config.json",
        "dev_grid": frozen / "dev_grid" / "results.csv",
        "ranking": frozen / "dev_grid" / "ranking.csv",
        "dev_grid_metadata": frozen / "dev_grid" / "run_metadata.json",
        "gate": frozen / "gate" / "gate_results.csv",
        "time_sensitivity": frozen / "time_sensitivity" / "results.csv",
        "evaluation": frozen / "evaluation" / "results.csv",
        "window_scenarios": frozen / "window_scenarios" / "results.csv",
        "window_scenarios_contended": frozen / "window_scenarios_run1_contended" / "results.csv",
        "early_delivery": frozen / "early_delivery" / "results.csv",
        "early_delivery_lex": frozen / "early_delivery" / "summary_lex.csv",
        "eval_time_metrics": frozen / "evaluation" / "time_metrics.csv",
        **{f"v2_{b}": frozen / "v2" / b / "results.csv" for b in ("B1", "B2", "B3", "B4")},
        **{f"v2_{b}_ranking": frozen / "v2" / b / "ranking.csv" for b in ("B1", "B2", "B3")},
        **{f"v2_{b}_decision": frozen / "v2" / b / "decision.json" for b in ("B1", "B2", "B3")},
        "v2_B5_windows": frozen / "v2" / "B5_windows" / "results.csv",
        "v2_B5_early": frozen / "v2" / "B5_early" / "results.csv",
        "phase_c_C1": frozen / "phase_c" / "C1",
        "phase_c_capacity": frozen / "phase_c" / "C2" / "capacity",
        "phase_c_fleet_priority": frozen / "phase_c" / "C2" / "fleet_priority",
        "phase_c_soft_windows": frozen / "phase_c" / "C2" / "soft_windows",
        "phase_c_removal": frozen / "phase_c" / "C3" / "removal",
        "phase_c_insertion": frozen / "phase_c" / "C3" / "insertion",
        "evaluation_v2": frozen / "evaluation_v2" / "results.csv",
        "evaluation_v2_comparison": frozen / "evaluation_v2" / "comparison_by_instance.csv",
    }
