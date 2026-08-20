"""Text-only importer for Kaggle V2-D1 run artifacts.

This module reads the operational evidence produced by a Kaggle kernel
run for the V2-D1 classical benchmark -- the raw execution log, the
full benchmark result, and the small published manifest -- and renders
each as plain text. It performs no plotting, no dataframe construction,
and no modeling; it exists purely as a validation aid for a human
reviewing what happened on Kaggle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping


def _read_log_text(path: str | Path) -> str:
    """Read one Kaggle log file, rejecting a missing or empty file.

    Args:
        path: Log file to read.

    Returns:
        The raw file text.

    Raises:
        ValueError: If the file is missing or empty.
    """

    target = Path(path)
    if not target.is_file():
        raise ValueError(f"Kaggle log file is missing: {target}")
    text = target.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Kaggle log file is empty: {target}")
    return text


def _parse_log_entries(raw_text: str) -> list[dict[str, Any]] | None:
    """Parse the JSON-array Kaggle log format, or signal plain text.

    Args:
        raw_text: Full file contents.

    Returns:
        The list of log entry objects, or ``None`` when ``raw_text`` is
        not a JSON array of objects.
    """

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list) and all(isinstance(e, dict) for e in payload):
        return payload
    return None


def _log_lines(raw_text: str, stream: str | None) -> list[str]:
    """Reconstruct log lines from chunked ``data`` fields, in order.

    Args:
        raw_text: Full file contents.
        stream: Optional ``stream_name`` filter (e.g. ``"stdout"``).

    Returns:
        The reconstructed lines, filtered by stream when the file is
        the JSON-array format; the plain-text lines otherwise.
    """

    entries = _parse_log_entries(raw_text)
    if entries is None:
        return raw_text.splitlines()
    if stream is not None:
        entries = [e for e in entries if e.get("stream_name") == stream]
    data = "".join(str(e.get("data", "")) for e in entries)
    return data.splitlines()


def read_kaggle_log(path: str | Path) -> str:
    """Render one downloaded Kaggle log as plain text.

    Parses the JSON-array log format (a list of ``{"stream_name",
    "time", "data"}`` objects) and concatenates the ``data`` chunks in
    order before splitting into lines, since each chunk is a partial
    line rather than a complete one. Falls back to the raw file text
    when the file is not that JSON format, since some Kaggle logs are
    plain text.

    Args:
        path: Downloaded Kaggle log file.

    Returns:
        The plain-text log, one reconstructed line per line of output.

    Raises:
        ValueError: If the file is missing or empty.
    """

    raw_text = _read_log_text(path)
    return "\n".join(_log_lines(raw_text, stream=None))


def summarize_kaggle_log(
    path: str | Path,
    *,
    stream: str | None = None,
    tail: int | None = None,
) -> str:
    """Render one downloaded Kaggle log, optionally filtered and tailed.

    Args:
        path: Downloaded Kaggle log file.
        stream: Keep only entries whose ``stream_name`` equals this
            value (e.g. ``"stdout"`` or ``"stderr"``). Ignored when the
            file is not the JSON-array format, since no stream metadata
            is available in that fallback.
        tail: Keep only the last ``tail`` lines when given. A
            non-positive value returns no lines.

    Returns:
        The plain-text log, filtered and tailed as requested.

    Raises:
        ValueError: If the file is missing or empty.
    """

    raw_text = _read_log_text(path)
    lines = _log_lines(raw_text, stream=stream)
    if tail is not None:
        lines = lines[-tail:] if tail > 0 else []
    return "\n".join(lines)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    """Load one non-empty UTF-8 JSON object file.

    Args:
        path: JSON file to load.

    Returns:
        The parsed top-level JSON object.

    Raises:
        ValueError: If the file is missing, empty, or not a JSON object.
    """

    target = Path(path)
    if not target.is_file():
        raise ValueError(f"Required JSON file is missing: {target}")
    text = target.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Required JSON file is empty: {target}")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in: {target}")
    return payload


def _dig(mapping: Any, *path: str, default: Any = None) -> Any:
    """Read a nested value from mappings, defaulting past any gap.

    Args:
        mapping: Starting mapping to walk.
        *path: Sequence of keys to follow.
        default: Value returned when any step is missing or not a
            mapping.

    Returns:
        The nested value, or ``default``.
    """

    current = mapping
    for key in path:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return default
    return current


def _fmt_num(value: Any) -> str:
    """Format one metric value as fixed-point text, or a placeholder.

    Args:
        value: Value to format.

    Returns:
        A six-decimal string for numbers, ``"N/A"`` for ``None``, and
        ``str(value)`` otherwise.
    """

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    if value is None:
        return "N/A"
    return str(value)


def _d2_model(payload: Mapping[str, Any], field: str) -> str:
    """Read one model descriptor field from a D2 result.

    The runner publishes the block as ``model_spec`` because the shared
    privacy validator rejects any persisted key named ``model``. Older
    or hand-built payloads may still use ``model``, so both are read.

    Args:
        payload: Full D2 result payload.
        field: Descriptor field to read.

    Returns:
        The field value as text, or ``"N/A"`` when absent.
    """

    for block in ("model_spec", "model"):
        value = _dig(payload, block, field)
        if value is not None:
            return str(value)
    return "N/A"


def _fmt_count(value: Any) -> str:
    """Format one integer count without a fixed-point tail.

    Args:
        value: Value to format.

    Returns:
        The integer as text, ``"N/A"`` for ``None``, else ``str(value)``.
    """

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if value is None:
        return "N/A"
    return str(value)


def _fmt_verdict(value: Any) -> str:
    """Format one gate/safety boolean as a readable verdict.

    Args:
        value: Value to format.

    Returns:
        ``"PASS"`` for ``True``, ``"FAIL"`` for ``False``, and
        ``"UNKNOWN"`` otherwise.
    """

    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "UNKNOWN"


def _outer_safety(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return one candidate's outer safety verdict, falling back to gates.

    Args:
        candidate: One candidate record.

    Returns:
        The ``outer.safety`` mapping, or ``outer.gates`` when ``safety``
        is absent (older result shape), or an empty mapping.
    """

    outer = candidate.get("outer")
    if not isinstance(outer, Mapping):
        return {}
    safety = outer.get("safety")
    if isinstance(safety, Mapping):
        return safety
    gates = outer.get("gates")
    if isinstance(gates, Mapping):
        return gates
    return {}


def _render_candidate_table(candidates: list[Any]) -> list[str]:
    """Render the fixed-width all-candidates comparison table.

    Args:
        candidates: Candidate records from the benchmark result.

    Returns:
        Lines of a fixed-width text table, header and separator first.
    """

    headers = {
        "candidate_id": "candidate_id",
        "threshold": "calib_threshold",
        "calib_critical_f1": "calib_crit_f1",
        "outer_critical_f1": "outer_crit_f1",
        "outer_macro_f1": "outer_macro_f1",
        "outer_safety": "outer_safety",
    }
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        safety = _outer_safety(candidate)
        rows.append({
            "candidate_id": str(candidate.get("candidate_id", "?")),
            "threshold": _fmt_num(
                _dig(candidate, "calibration", "threshold")
            ),
            "calib_critical_f1": _fmt_num(
                _dig(candidate, "calibration", "metrics", "critical_f1")
            ),
            "outer_critical_f1": _fmt_num(
                _dig(candidate, "outer", "metrics", "critical_f1")
            ),
            "outer_macro_f1": _fmt_num(
                _dig(candidate, "outer", "metrics", "macro_f1")
            ),
            "outer_safety": _fmt_verdict(safety.get("passed")),
        })
    if not rows:
        return ["  (no candidates available)"]
    widths = {
        key: max(len(header), max(len(row[key]) for row in rows))
        for key, header in headers.items()
    }
    header_line = "  " + "  ".join(
        headers[key].ljust(widths[key]) for key in headers
    )
    separator = "  " + "  ".join("-" * widths[key] for key in headers)
    lines = [header_line, separator]
    for row in rows:
        lines.append(
            "  " + "  ".join(row[key].ljust(widths[key]) for key in headers)
        )
    return lines


def _render_degeneracy_check(candidates: list[Any]) -> list[str]:
    """Flag candidates whose outer metrics never differ from each other.

    Args:
        candidates: Candidate records from the benchmark result.

    Returns:
        Lines summarizing distinct outer-metric tuples and distinct
        calibration thresholds, with a loud warning when every
        candidate produced identical outer metrics.
    """

    valid = [c for c in candidates if isinstance(c, Mapping)]
    tuples: set[tuple[Any, Any, Any, Any]] = set()
    thresholds: set[Any] = set()
    for candidate in valid:
        metrics = _dig(candidate, "outer", "metrics", default={})
        if not isinstance(metrics, Mapping):
            metrics = {}
        tuples.add((
            metrics.get("critical_f1"),
            metrics.get("critical_precision"),
            metrics.get("critical_recall"),
            metrics.get("macro_f1"),
        ))
        thresholds.add(_dig(candidate, "calibration", "threshold"))
    lines = [
        f"  distinct outer metric tuples: {len(tuples)} "
        f"(of {len(valid)} candidates)",
        f"  distinct calibration thresholds: {len(thresholds)} "
        f"(of {len(valid)} candidates)",
    ]
    if len(valid) > 1 and len(tuples) == 1:
        lines.append("")
        lines.append(
            "  *** WARNING: EVERY CANDIDATE PRODUCED IDENTICAL OUTER "
            "METRICS. ***"
        )
        lines.append(
            "  *** No candidate differentiated itself from the "
            "fallback. ***"
        )
    return lines


def summarize_v2_result(path: str | Path) -> str:
    """Render the V2 benchmark result JSON as a plain-text report.

    Reads defensively with ``.get()`` throughout so a partially-shaped
    or older result file still renders instead of raising.

    Args:
        path: Full benchmark result JSON (e.g.
            ``v2_classical_benchmark.json``).

    Returns:
        A plain-text report with a header, the selected candidate's
        metrics, a table of all candidates, and a degeneracy check.

    Raises:
        ValueError: If the file is missing, empty, or not a JSON object.
    """

    payload = _load_json_object(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    selected_id = payload.get("selected", "UNKNOWN")

    lines = ["V2 BENCHMARK RESULT"]
    lines.append(f"  status: {payload.get('status', 'UNKNOWN')}")
    lines.append(f"  complete: {payload.get('complete', 'UNKNOWN')}")
    lines.append(
        f"  runtime_seconds: {_fmt_num(payload.get('runtime_seconds'))}"
    )
    lines.append(f"  candidate_count: {len(candidates)}")
    lines.append(f"  selected: {selected_id}")
    lines.append("")

    selected_candidate = next(
        (
            c for c in candidates
            if isinstance(c, Mapping)
            and c.get("candidate_id") == selected_id
        ),
        None,
    )
    lines.append("SELECTED CANDIDATE")
    if not isinstance(selected_candidate, Mapping):
        lines.append(
            f"  (no candidate matches selected id: {selected_id})"
        )
    else:
        lines.append(
            f"  candidate_id: {selected_candidate.get('candidate_id')}"
        )
        lines.append("  calibration:")
        threshold = _dig(selected_candidate, "calibration", "threshold")
        lines.append(f"    threshold: {_fmt_num(threshold)}")
        for metric in (
            "critical_f1", "critical_precision", "critical_recall", "macro_f1"
        ):
            value = _dig(selected_candidate, "calibration", "metrics", metric)
            lines.append(f"    {metric}: {_fmt_num(value)}")
        lines.append("  outer:")
        for metric in (
            "critical_f1", "critical_precision", "critical_recall", "macro_f1"
        ):
            value = _dig(selected_candidate, "outer", "metrics", metric)
            lines.append(f"    {metric}: {_fmt_num(value)}")
        safety = _outer_safety(selected_candidate)
        verdict = _fmt_verdict(safety.get("passed"))
        lines.append(f"    safety_verdict: {verdict}")
    lines.append("")

    lines.append(f"ALL CANDIDATES ({len(candidates)})")
    lines.extend(_render_candidate_table(candidates))
    lines.append("")

    lines.append("DEGENERACY CHECK")
    lines.extend(_render_degeneracy_check(candidates))

    return "\n".join(lines)


def _render_mapping_lines(mapping: Mapping[str, Any], indent: int) -> list[str]:
    """Render one mapping as indented, sorted key/value text lines.

    Args:
        mapping: Mapping to render.
        indent: Current indentation depth, in two-space steps.

    Returns:
        Lines of readable key/value text, recursing into nested
        mappings.
    """

    prefix = "  " * indent
    lines: list[str] = []
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{key}:")
            lines.extend(_render_mapping_lines(value, indent + 1))
        elif isinstance(value, (list, tuple)):
            rendered = ", ".join(str(item) for item in value)
            lines.append(f"{prefix}{key}: {rendered}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines


def summarize_v2_manifest(path: str | Path) -> str:
    """Render the small published V2 manifest as key/value text.

    Args:
        path: Published manifest JSON (e.g.
            ``v2_classical_results.json``).

    Returns:
        A plain-text, indented key/value rendering of the manifest.

    Raises:
        ValueError: If the file is missing, empty, or not a JSON object.
    """

    payload = _load_json_object(path)
    lines = ["V2 PUBLISHED MANIFEST"]
    lines.extend(_render_mapping_lines(payload, indent=1))
    return "\n".join(lines)


def _fmt_list(value: Any) -> str:
    """Format a list-shaped value as comma-joined text, or a placeholder.

    Args:
        value: Value to format, expected to be a list or tuple.

    Returns:
        A comma-joined string for a non-empty list/tuple, and ``"N/A"``
        otherwise.
    """

    if isinstance(value, (list, tuple)) and value:
        return ", ".join(str(item) for item in value)
    return "N/A"


def summarize_d2_result(path: str | Path) -> str:
    """Render the D2 header block: schema, status, model, partitions.

    Reads defensively with ``.get()`` and ``_dig()`` throughout so a
    partially-shaped or malformed result file still renders instead of
    raising.

    Args:
        path: Full D2 transformer-challenge result JSON (e.g.
            ``v2_transformer_challenge.json``).

    Returns:
        A short plain-text header with the schema version, status,
        complete/diagnostic_only flags, runtime, the model id and its
        resolved revision, sealed access, and the allowed vs sealed
        partitions.

    Raises:
        ValueError: If the file is missing, empty, or not a JSON object.
    """

    payload = _load_json_object(path)
    lines = ["V2.1-D2 TRANSFORMER CHALLENGE"]
    lines.append(f"  schema_version: {payload.get('schema_version', 'N/A')}")
    lines.append(f"  status: {payload.get('status', 'UNKNOWN')}")
    lines.append(f"  complete: {payload.get('complete', 'UNKNOWN')}")
    lines.append(
        f"  diagnostic_only: {payload.get('diagnostic_only', 'UNKNOWN')}"
    )
    lines.append(
        f"  runtime_seconds: {_fmt_num(payload.get('runtime_seconds'))}"
    )
    lines.append(
        f"  model_id: {_d2_model(payload, 'model_id')}"
    )
    lines.append(
        "  resolved_revision: "
        f"{_d2_model(payload, 'resolved_revision')}"
    )
    sealed_access = payload.get("sealed_access")
    if isinstance(sealed_access, Mapping):
        lines.append("  sealed_access:")
        lines.extend(_render_mapping_lines(sealed_access, indent=2))
    elif sealed_access is None:
        lines.append("  sealed_access: N/A")
    else:
        lines.append(f"  sealed_access: {sealed_access}")
    lines.append(
        "  allowed_partitions: "
        f"{_fmt_list(payload.get('allowed_partitions'))}"
    )
    lines.append(
        "  sealed_partitions: "
        f"{_fmt_list(payload.get('sealed_partitions'))}"
    )
    return "\n".join(lines)


def _render_d2_incumbent(payload: Mapping[str, Any]) -> list[str]:
    """Render the D2 report's incumbent (V2.1-D1 winner) block.

    Args:
        payload: Parsed D2 result JSON object.

    Returns:
        Lines naming the incumbent candidate and its four outer
        metrics, read defensively.
    """

    incumbent = payload.get("incumbent")
    if not isinstance(incumbent, Mapping):
        incumbent = {}
    metrics = " ".join(
        f"{metric}={_fmt_num(_dig(incumbent, 'outer', metric))}"
        for metric in (
            "critical_f1", "critical_precision", "critical_recall", "macro_f1"
        )
    )
    return [
        "INCUMBENT (V2.1-D1 WINNER)",
        f"  candidate_id: {incumbent.get('candidate_id', 'N/A')}",
        f"  outer: {metrics}",
        "",
    ]


def _render_d2_fallback(payload: Mapping[str, Any]) -> list[str]:
    """Render the D2 report's fallback-baseline block for both windows.

    Args:
        payload: Parsed D2 result JSON object.

    Returns:
        Lines with the frozen S7 fallback's metrics on the
        ``inner_calibration`` and ``outer_evaluation`` windows.
    """

    fallback = payload.get("fallback_baseline")
    if not isinstance(fallback, Mapping):
        fallback = {}
    lines = ["FALLBACK BASELINE (FROZEN S7)"]
    for window in ("inner_calibration", "outer_evaluation"):
        lines.append(f"  {window}:")
        window_payload = fallback.get(window)
        if not isinstance(window_payload, Mapping):
            window_payload = {}
        for metric in (
            "critical_f1", "critical_precision", "critical_recall", "macro_f1"
        ):
            lines.append(f"    {metric}: {_fmt_num(window_payload.get(metric))}")
    lines.append("")
    return lines


def _render_d2_seed_table(seeds: list[Any], reported_seed: Any) -> list[str]:
    """Render the fixed-width per-seed comparison table.

    Args:
        seeds: Seed replicate records from the D2 result.
        reported_seed: The seed value marked as the pre-registered,
            median-aggregation seed, or ``None`` when unknown.

    Returns:
        Lines of a fixed-width text table, header and separator first,
        with the reported seed's row prefixed by ``"*"``.
    """

    headers = {
        "marker": " ",
        "seed": "seed",
        "threshold": "threshold",
        "cal_ovr": "cal_eff_ovr",
        "outer_ovr": "outer_eff_ovr",
        "outer_critical_f1": "outer_crit_f1",
        "outer_critical_precision": "outer_crit_prec",
        "outer_critical_recall": "outer_crit_rec",
        "outer_macro_f1": "outer_macro_f1",
    }
    rows: list[dict[str, str]] = []
    for seed_record in seeds:
        if not isinstance(seed_record, Mapping):
            continue
        seed_value = seed_record.get("seed", "?")
        marker = "*" if seed_value == reported_seed else ""
        rows.append({
            "marker": marker,
            "seed": str(seed_value),
            "threshold": _fmt_num(
                _dig(seed_record, "calibration", "threshold")
            ),
            "cal_ovr": _fmt_count(
                _dig(seed_record, "calibration", "effective_overrides")
            ),
            "outer_ovr": _fmt_count(
                _dig(seed_record, "outer", "effective_overrides")
            ),
            "outer_critical_f1": _fmt_num(
                _dig(seed_record, "outer", "metrics", "critical_f1")
            ),
            "outer_critical_precision": _fmt_num(
                _dig(seed_record, "outer", "metrics", "critical_precision")
            ),
            "outer_critical_recall": _fmt_num(
                _dig(seed_record, "outer", "metrics", "critical_recall")
            ),
            "outer_macro_f1": _fmt_num(
                _dig(seed_record, "outer", "metrics", "macro_f1")
            ),
        })
    if not rows:
        return ["  (no seeds available)"]
    widths = {
        key: max(len(header), max(len(row[key]) for row in rows))
        for key, header in headers.items()
    }
    header_line = "  " + "  ".join(
        headers[key].ljust(widths[key]) for key in headers
    )
    separator = "  " + "  ".join("-" * widths[key] for key in headers)
    lines = [header_line, separator]
    for row in rows:
        lines.append(
            "  " + "  ".join(row[key].ljust(widths[key]) for key in headers)
        )
    return lines


def _render_d2_seeds(payload: Mapping[str, Any]) -> list[str]:
    """Render the D2 report's per-seed table, legend, and spread block.

    Args:
        payload: Parsed D2 result JSON object.

    Returns:
        Lines with the per-seed comparison table (reported seed marked
        with ``"*"``), a legend explaining the marker, and the outer
        critical-F1 spread (min/median/max/range).
    """

    seeds = payload.get("seeds")
    if not isinstance(seeds, list):
        seeds = []
    reported_seed = _dig(payload, "reported", "seed")
    lines = ["SEEDS"]
    lines.extend(_render_d2_seed_table(seeds, reported_seed))
    lines.append(
        "  * marks the pre-registered median-aggregation seed "
        "(reported), not the best seed."
    )
    lines.append("")

    lines.append("SEED SPREAD (outer critical_f1)")
    spread = payload.get("seed_spread")
    if not isinstance(spread, Mapping):
        spread = {}
    for key, label in (
        ("critical_f1_min", "min"),
        ("critical_f1_median", "median"),
        ("critical_f1_max", "max"),
        ("critical_f1_range", "range"),
    ):
        lines.append(f"  {label}: {_fmt_num(spread.get(key))}")
    lines.append("")
    return lines


_D2_DECISION_CONDITIONS = (
    (
        "passes_three_of_three_development_safety_margins",
        "passes_margins",
    ),
    ("effective_overrides_greater_than_zero", "has_effective_overrides"),
    (
        "outer_critical_f1_greater_than_fallback_baseline",
        "beats_fallback",
    ),
    (
        "outer_critical_precision_at_least_precision_floor",
        "meets_precision_floor",
    ),
    (
        "outer_critical_f1_at_least_displacement_bar",
        "meets_displacement_threshold",
    ),
)


def _render_d2_decision(payload: Mapping[str, Any]) -> list[str]:
    """Render the D2 report's pre-registered decision block.

    Args:
        payload: Parsed D2 result JSON object.

    Returns:
        Lines with every boolean condition as PASS/FAIL/UNKNOWN, the
        displacement bar and increment, the precision floor, the
        reported deltas versus the incumbent and the fallback, the
        blocked reason, and the final outcome -- with an explicit note
        when the outcome is ``CLASSICAL_WINNER_STANDS``.
    """

    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        decision = {}
    reported = payload.get("reported")
    if not isinstance(reported, Mapping):
        reported = {}

    lines = ["DECISION (PRE-REGISTERED)"]
    lines.append(
        f"  pre_registered: {_fmt_verdict(decision.get('pre_registered'))}"
    )
    for key, label in _D2_DECISION_CONDITIONS:
        value = decision.get(key)
        if value is None:
            value = decision.get(label)
        lines.append(f"  {label}: {_fmt_verdict(value)}")
    lines.append(
        f"  displacement_bar: {_fmt_num(decision.get('displacement_bar'))}"
    )
    lines.append(
        "  displacement_increment: "
        f"{_fmt_num(decision.get('displacement_increment'))}"
    )
    lines.append(
        f"  precision_floor: {_fmt_num(decision.get('precision_floor'))}"
    )
    lines.append(
        "  critical_f1_vs_incumbent: "
        f"{_fmt_num(reported.get('critical_f1_vs_incumbent'))}"
    )
    lines.append(
        "  critical_f1_vs_fallback: "
        f"{_fmt_num(reported.get('critical_f1_vs_fallback'))}"
    )
    lines.append(f"  blocked_reason: {decision.get('blocked_reason')}")
    outcome = decision.get("outcome", "UNKNOWN")
    lines.append(f"  outcome: {outcome}")
    if outcome == "CLASSICAL_WINNER_STANDS":
        lines.append("")
        lines.append(
            "  This is a valid, published result: the Transformer did "
            "not displace the incumbent."
        )
        lines.append(
            "  The D1 classical winner remains the V2 candidate."
        )
    lines.append("")
    return lines


def _render_d2_integrity(payload: Mapping[str, Any]) -> list[str]:
    """Render the D2 report's provenance/integrity block.

    Args:
        payload: Parsed D2 result JSON object.

    Returns:
        Lines with the execution-config signature, the incumbent
        artifact's sha256, and the hard-negative pool signature with
        its pool_rows/positive_groups/hard_negative_groups counts.
    """

    hard_negative = payload.get("hard_negative")
    if not isinstance(hard_negative, Mapping):
        hard_negative = {}
    lines = ["INTEGRITY"]
    lines.append(
        "  execution_config_signature: "
        f"{payload.get('execution_config_signature', 'N/A')}"
    )
    lines.append(
        "  incumbent_artifact_sha256: "
        f"{_dig(payload, 'incumbent', 'artifact', 'sha256', default='N/A')}"
    )
    lines.append(
        "  hard_negative_pool_signature: "
        f"{hard_negative.get('pool_signature', 'N/A')}"
    )
    lines.append(f"    pool_rows: {hard_negative.get('pool_rows', 'N/A')}")
    lines.append(
        f"    positive_groups: {hard_negative.get('positive_groups', 'N/A')}"
    )
    lines.append(
        "    hard_negative_groups: "
        f"{hard_negative.get('hard_negative_groups', 'N/A')}"
    )
    return lines


def summarize_d2_manifest(path: str | Path) -> str:
    """Render the small published D2 manifest as key/value text.

    Args:
        path: Published manifest JSON (e.g.
            ``v2_transformer_results.json``).

    Returns:
        A plain-text, indented key/value rendering of the manifest.

    Raises:
        ValueError: If the file is missing, empty, or not a JSON object.
    """

    payload = _load_json_object(path)
    lines = ["V2.1-D2 PUBLISHED MANIFEST"]
    lines.extend(_render_mapping_lines(payload, indent=1))
    return "\n".join(lines)


_PACKAGE_OUTCOME_FROZEN = "PACKAGE_FROZEN"
_PACKAGE_OUTCOME_MISMATCH = "REPRODUCTION_MISMATCH"
_PACKAGE_OUTER_METRICS = (
    "critical_f1",
    "critical_precision",
    "critical_recall",
    "macro_f1",
)
_PACKAGE_MATRIX_FIELDS = (
    "comparable",
    "reason",
    "shape",
    "observed_shape",
    "expected_shape",
    "mismatched_cells",
    "total_absolute_difference",
    "max_absolute_cell_difference",
    "observed_total",
    "expected_total",
)
_PACKAGE_SCOPE_BLOCKS = ("fit_scope", "calibration_scope", "outer_scope")
_PACKAGE_SAFETY_BLOCKS = ("limits", "values", "headroom")


def _fmt_flag(value: Any) -> str:
    """Format one declarative boolean without ever hiding a ``False``.

    An absent field and a published ``False`` must never render the
    same way, so only ``None`` becomes the placeholder.

    Args:
        value: Value to format.

    Returns:
        ``"True"`` or ``"False"`` for a boolean, ``"N/A"`` when the
        field is absent, and ``str(value)`` otherwise.
    """

    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "N/A"
    return str(value)


def _fmt_scalar(value: Any) -> str:
    """Format one published value as a list, a metric, or a count.

    Args:
        value: Value to format.

    Returns:
        Comma-joined text for a list or tuple, six-decimal text for a
        float, the bare integer for an integer count, and ``"N/A"``
        when the value is absent.
    """

    if isinstance(value, (list, tuple)):
        return _fmt_list(value)
    if isinstance(value, float):
        return _fmt_num(value)
    return _fmt_count(value)


def _render_package_verdict(payload: Mapping[str, Any]) -> list[str]:
    """State plainly whether a fitted bundle was persisted, and why not.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines carrying the freeze verdict. ``PACKAGE_FROZEN`` confirms
        that every check passed and the joblib bundle was written; any
        other state says loudly that no bundle exists and names the
        failed checks, so a mismatch can never read as a success.
    """

    if payload.get("diagnostic_only") is True:
        return [
            "  DIAGNOSTIC PREFLIGHT ONLY: nothing was fitted, nothing was",
            "  frozen, and no joblib bundle was written.",
        ]
    outcome = payload.get("outcome")
    if payload.get("frozen") is True and outcome == _PACKAGE_OUTCOME_FROZEN:
        return [
            "  PACKAGE FROZEN: every reproduction check passed and the",
            "  fitted joblib bundle was written.",
        ]
    if outcome == _PACKAGE_OUTCOME_MISMATCH or payload.get("frozen") is False:
        failed = _dig(payload, "reproduction_gate", "failed_checks")
        return [
            "  *** REPRODUCTION MISMATCH: THE PACKAGE WAS NOT FROZEN. ***",
            "  *** NO joblib bundle was written. ***",
            "  The refit did not reproduce the V2.1-D1 published numbers",
            "  exactly, so the divergence is published as evidence instead",
            "  of a package.",
            f"  failed_checks: {_fmt_list(failed)}",
        ]
    return [
        "  *** NOT FROZEN: this artifact records no completed freeze. ***",
        "  No joblib bundle was written.",
        f"  recorded outcome: {outcome if outcome is not None else 'UNKNOWN'}",
    ]


def _render_package_outcome(payload: Mapping[str, Any]) -> list[str]:
    """Render the package report's outcome block, first and prominent.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines with the outcome, the frozen and complete flags, the
        status, the run mode, the runtime, and the deployment block,
        closed by the explicit freeze verdict.
    """

    deployment = payload.get("deployment")
    if not isinstance(deployment, Mapping):
        deployment = {}
    lines = ["OUTCOME"]
    lines.append(f"  outcome: {payload.get('outcome', 'UNKNOWN')}")
    lines.append(f"  frozen: {_fmt_flag(payload.get('frozen'))}")
    lines.append(f"  complete: {_fmt_flag(payload.get('complete'))}")
    lines.append(f"  status: {payload.get('status', 'UNKNOWN')}")
    lines.append(f"  run_mode: {payload.get('run_mode', 'UNKNOWN')}")
    lines.append(
        f"  runtime_seconds: {_fmt_num(payload.get('runtime_seconds'))}"
    )
    lines.append("  deployment:")
    lines.append(
        "    deployment_authorized: "
        f"{_fmt_flag(deployment.get('deployment_authorized'))}"
    )
    lines.append(f"    status: {deployment.get('status', 'N/A')}")
    lines.append(f"    next_step: {deployment.get('next_step', 'N/A')}")
    lines.append("")
    lines.extend(_render_package_verdict(payload))
    lines.append("")
    return lines


def _render_package_check_table(
    checks: Mapping[str, Any], required: Any
) -> list[str]:
    """Render the fixed-width reproduction-check table.

    Args:
        checks: Every evaluated check name mapped to its boolean.
        required: The canonical ``required_checks`` names, when
            published.

    Returns:
        Lines of a fixed-width text table, header and separator first,
        listing every canonical check first and marking it with ``"*"``.
    """

    canonical = [
        str(name)
        for name in (required if isinstance(required, (list, tuple)) else ())
    ]
    ordered = [name for name in canonical if name in checks]
    ordered.extend(sorted(name for name in checks if name not in canonical))
    headers = {"marker": " ", "check": "check", "verdict": "verdict"}
    rows = [
        {
            "marker": "*" if name in canonical else "",
            "check": str(name),
            "verdict": _fmt_verdict(checks[name]),
        }
        for name in ordered
    ]
    if not rows:
        return ["  (no checks available)"]
    widths = {
        key: max(len(header), max(len(row[key]) for row in rows))
        for key, header in headers.items()
    }
    header_line = "  " + "  ".join(
        headers[key].ljust(widths[key]) for key in headers
    )
    separator = "  " + "  ".join("-" * widths[key] for key in headers)
    lines = [header_line, separator]
    for row in rows:
        lines.append(
            "  " + "  ".join(row[key].ljust(widths[key]) for key in headers)
        )
    return lines


def _render_package_divergences(gate: Mapping[str, Any]) -> list[str]:
    """Render every published divergence without echoing any matrix.

    Scalar divergences render as observed versus expected. A confusion
    matrix divergence renders only its aggregate difference summary, so
    no matrix, narrative, identifier, or row index is ever printed.

    Args:
        gate: The reproduction-gate block of the result.

    Returns:
        Lines describing each failed check's divergence, or a single
        note when the gate published none.
    """

    divergences = gate.get("divergences")
    if not isinstance(divergences, Mapping) or not divergences:
        return ["  (no divergences published)"]
    lines: list[str] = []
    for name in sorted(divergences):
        entry = divergences[name]
        lines.append(f"  {name}:")
        if not isinstance(entry, Mapping):
            lines.append(f"    {entry}")
            continue
        if "observed" in entry or "expected" in entry:
            lines.append(f"    observed: {_fmt_scalar(entry.get('observed'))}")
            lines.append(f"    expected: {_fmt_scalar(entry.get('expected'))}")
            continue
        for field in _PACKAGE_MATRIX_FIELDS:
            if field in entry:
                lines.append(f"    {field}: {_fmt_scalar(entry[field])}")
    return lines


def _render_package_gate(payload: Mapping[str, Any]) -> list[str]:
    """Render the exact reproduction gate, its checks, and divergences.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines with the gate verdict, the evaluated check count, the
        comparison mode and its source of truth, the full check table,
        and, when any check failed, the published divergences.
    """

    gate = payload.get("reproduction_gate")
    if not isinstance(gate, Mapping):
        gate = {}
    checks = gate.get("checks")
    if not isinstance(checks, Mapping):
        checks = {}
    lines = ["REPRODUCTION GATE (EXACT, NO TOLERANCE)"]
    lines.append(f"  passed: {_fmt_verdict(gate.get('passed'))}")
    lines.append(f"  check_count: {_fmt_count(gate.get('check_count'))}")
    lines.append(f"  comparison: {gate.get('comparison', 'N/A')}")
    lines.append(f"  source_of_truth: {gate.get('source_of_truth', 'N/A')}")
    lines.append(f"  candidate_id: {gate.get('candidate_id', 'N/A')}")
    lines.append("  checks:")
    lines.extend(
        _render_package_check_table(checks, gate.get("required_checks"))
    )
    lines.append("  * marks a canonical check named in required_checks.")
    failed = gate.get("failed_checks")
    if isinstance(failed, (list, tuple)) and failed:
        lines.append("")
        lines.append(f"DIVERGENCES ({len(failed)} failed)")
        lines.append(f"  failed_checks: {_fmt_list(failed)}")
        lines.extend(_render_package_divergences(gate))
    lines.append("")
    return lines


def _render_package_candidate(payload: Mapping[str, Any]) -> list[str]:
    """Render the pinned candidate and the three frozen scopes.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines with the candidate descriptor and the fit, calibration,
        and outer scope windows, each read defensively.
    """

    lines = ["CANDIDATE AND SCOPES"]
    for block in ("candidate",) + _PACKAGE_SCOPE_BLOCKS:
        value = payload.get(block)
        lines.append(f"  {block}:")
        if isinstance(value, Mapping) and value:
            lines.extend(_render_mapping_lines(value, indent=2))
        else:
            lines.append("    (not available)")
    lines.append("")
    return lines


def _render_package_metrics(payload: Mapping[str, Any]) -> list[str]:
    """Render the calibrated threshold and both evaluation windows.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines with the calibrated threshold, the override decisions and
        effective overrides of both windows, the outer metric vector,
        the frozen S7 fallback's outer values, and the critical-F1
        delta against that fallback.
    """

    lines = ["CALIBRATION AND OUTER"]
    lines.append(
        "  calibrated_threshold: "
        f"{_fmt_num(_dig(payload, 'calibration', 'threshold'))}"
    )
    lines.append("  inner_calibration:")
    for field in ("override_decisions", "effective_overrides"):
        lines.append(
            f"    {field}: {_fmt_count(_dig(payload, 'calibration', field))}"
        )
    lines.append(
        "    row_count: "
        f"{_fmt_count(_dig(payload, 'calibration', 'metrics', 'row_count'))}"
    )
    lines.append("  outer_evaluation:")
    for field in ("override_decisions", "effective_overrides"):
        lines.append(
            f"    {field}: {_fmt_count(_dig(payload, 'outer', field))}"
        )
    for metric in _PACKAGE_OUTER_METRICS:
        lines.append(
            f"    {metric}: "
            f"{_fmt_num(_dig(payload, 'outer', 'metrics', metric))}"
        )
    for metric in ("critical_support", "row_count"):
        lines.append(
            f"    {metric}: "
            f"{_fmt_count(_dig(payload, 'outer', 'metrics', metric))}"
        )
    lines.append("  fallback_baseline (outer_evaluation):")
    for metric in _PACKAGE_OUTER_METRICS:
        value = _dig(payload, "fallback_baseline", "outer_evaluation", metric)
        lines.append(f"    {metric}: {_fmt_num(value)}")
    lines.append(
        "  critical_f1_vs_fallback: "
        f"{_fmt_num(_dig(payload, 'outer', 'critical_f1_vs_fallback'))}"
    )
    lines.append("")
    return lines


def _render_package_refit_margin(
    payload: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> list[str]:
    """Render the margin this run recomputed, next to the frozen one.

    The top-level ``safety_margin`` block is copied from the frozen
    configuration and therefore always shows D1's published margin. The
    refit recomputes its own margin at ``outer.safety``. When the
    reproduction gate passes the two are identical by construction, so
    the comparison is only interesting when it fails.

    Args:
        payload: Parsed V2 package result JSON object.
        frozen: The frozen ``safety_margin`` block already rendered.

    Returns:
        Lines with the recomputed values and an agreement verdict, or an
        empty list when this run recomputed no margin.
    """

    safety = _dig(payload, "outer", "safety")
    if not isinstance(safety, Mapping) or not safety:
        return []
    values = safety.get("values")
    if not isinstance(values, Mapping) or not values:
        return []
    lines = ["", "  recomputed by this run (outer.safety):"]
    lines.append(f"    passed: {_fmt_verdict(safety.get('passed'))}")
    for key in sorted(values):
        lines.append(f"    {key}: {_fmt_num(values[key])}")
    frozen_values = frozen.get("values")
    if not isinstance(frozen_values, Mapping) or not frozen_values:
        return lines
    differing = sorted(
        key
        for key in set(values) | set(frozen_values)
        if values.get(key) != frozen_values.get(key)
    )
    if differing:
        lines.append(
            f"    DIFFERS from the frozen margin at: {_fmt_list(differing)}"
        )
    else:
        lines.append("    agrees with the frozen margin on every value")
    return lines


def _render_package_safety(payload: Mapping[str, Any]) -> list[str]:
    """Render the development safety margin and its evidence status.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines with the three limits, the measured values, the headroom,
        the verdict, and the evidence status, followed by an explicit
        note that these margins were measured on the same window used
        for selection and are therefore development-optimistic.
    """

    margin = payload.get("safety_margin")
    if not isinstance(margin, Mapping):
        margin = {}
    lines = ["SAFETY MARGIN (DEVELOPMENT)"]
    lines.append(f"  passed: {_fmt_verdict(margin.get('passed'))}")
    lines.append(
        "  required_gate_count: "
        f"{_fmt_count(margin.get('required_gate_count'))}"
    )
    lines.append(f"  measured_on: {margin.get('measured_on', 'N/A')}")
    lines.append(f"  evidence_status: {margin.get('evidence_status', 'N/A')}")
    for block in _PACKAGE_SAFETY_BLOCKS:
        values = margin.get(block)
        lines.append(f"  {block}:")
        if not isinstance(values, Mapping) or not values:
            lines.append("    (not available)")
            continue
        for key in sorted(values):
            lines.append(f"    {key}: {_fmt_num(values[key])}")
    lines.extend(_render_package_refit_margin(payload, margin))
    lines.append("")
    lines.append(
        "  These margins were measured on the same outer window that "
        "served"
    )
    lines.append(
        "  as the selection surface, so they are development-optimistic "
        "and"
    )
    lines.append("  are not independent evidence of future performance.")
    lines.append("")
    return lines


def _render_package_pool(payload: Mapping[str, Any]) -> list[str]:
    """Render the deterministic hard-negative pool counts and signature.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines with the positive and hard-negative group counts, the
        pool row count, and the pool signature. No row position is ever
        printed, and the pool itself is never persisted.
    """

    pool = payload.get("hard_negative")
    if not isinstance(pool, Mapping):
        pool = {}
    lines = ["HARD NEGATIVE POOL"]
    for field in ("positive_groups", "hard_negative_groups", "pool_rows"):
        lines.append(f"  {field}: {_fmt_count(pool.get(field))}")
    lines.append(f"  pool_signature: {pool.get('pool_signature', 'N/A')}")
    lines.append("")
    return lines


def _render_package_bundle(payload: Mapping[str, Any]) -> list[str]:
    """Render the joblib bundle block, persisted or withheld.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines with the persisted path, hash, and size when the freeze
        succeeded, and otherwise the withholding reason and the failed
        checks that caused it.
    """

    bundle = payload.get("bundle")
    if not isinstance(bundle, Mapping):
        bundle = {}
    persisted = bundle.get("persisted")
    lines = ["BUNDLE"]
    lines.append(f"  persisted: {_fmt_flag(persisted)}")
    if persisted is True:
        lines.append(f"  path: {bundle.get('path', 'N/A')}")
        lines.append(f"  sha256: {bundle.get('sha256', 'N/A')}")
        lines.append(f"  size_bytes: {_fmt_count(bundle.get('size_bytes'))}")
    else:
        lines.append(f"  reason: {bundle.get('reason', 'N/A')}")
        lines.append(
            f"  failed_checks: {_fmt_list(bundle.get('failed_checks'))}"
        )
        lines.append("  No fitted artifact was written for this run.")
    lines.append("")
    return lines


def _render_package_integrity(payload: Mapping[str, Any]) -> list[str]:
    """Render the package report's provenance and boundary block.

    Args:
        payload: Parsed V2 package result JSON object.

    Returns:
        Lines with the run signature, both schema versions, the stage
        and its ADR, every pinned provenance signature, the sealed
        access map and sealed partitions, and the declared persistence
        boundary, with a note on the deliberate weight-persistence flag.
    """

    lines = ["INTEGRITY"]
    lines.append(f"  signature: {payload.get('signature', 'N/A')}")
    lines.append(f"  code_schema: {payload.get('code_schema', 'N/A')}")
    lines.append(f"  schema_version: {payload.get('schema_version', 'N/A')}")
    lines.append(f"  stage: {payload.get('stage', 'N/A')}")
    lines.append(f"  adr: {payload.get('adr', 'N/A')}")
    provenance = payload.get("provenance")
    lines.append("  provenance:")
    if isinstance(provenance, Mapping) and provenance:
        lines.extend(_render_mapping_lines(provenance, indent=2))
    else:
        lines.append("    (not available)")
    sealed_access = payload.get("sealed_access")
    if isinstance(sealed_access, Mapping):
        lines.append("  sealed_access:")
        lines.extend(_render_mapping_lines(sealed_access, indent=2))
    else:
        lines.append(f"  sealed_access: {_fmt_flag(sealed_access)}")
    lines.append(
        f"  sealed_partitions: {_fmt_list(payload.get('sealed_partitions'))}"
    )
    boundary = payload.get("boundary")
    lines.append("  boundary:")
    if isinstance(boundary, Mapping) and boundary:
        lines.extend(_render_mapping_lines(boundary, indent=2))
    else:
        lines.append("    (not available)")
    lines.append("")
    lines.append(
        "  boundary.persists_fitted_weights is deliberately True at step "
        "6:"
    )
    lines.append(
        "  freezing a package is precisely persisting fitted weights. What"
    )
    lines.append(
        "  does not change is persists_narratives_or_identifiers, which"
    )
    lines.append("  stays False.")
    return lines


def summarize_package_result(path: str | Path) -> str:
    """Render the V2 frozen-package result JSON as a plain-text report.

    Reads defensively with ``.get()`` and ``_dig()`` throughout, so a
    smoke-mode artifact -- which is ``diagnostic_only`` and carries no
    ``calibration``, ``outer``, or ``reproduction_gate`` block -- and a
    partially-shaped artifact both render instead of raising. No
    confusion matrix, narrative, identifier, or row index is printed.

    Args:
        path: Full V2 package result JSON (e.g. ``v2_package.json``).

    Returns:
        A plain-text report opening with the outcome and the explicit
        freeze verdict, then the exact reproduction gate with any
        divergences, the candidate and its three scopes, the
        calibration and outer evidence, the development safety margin,
        the hard-negative pool, the joblib bundle, and the integrity
        block.

    Raises:
        ValueError: If the file is missing, empty, or not a JSON object.
    """

    payload = _load_json_object(path)
    lines = ["V2.1-P FROZEN HIERARCHICAL PACKAGE", ""]
    lines.extend(_render_package_outcome(payload))
    lines.extend(_render_package_gate(payload))
    lines.extend(_render_package_candidate(payload))
    lines.extend(_render_package_metrics(payload))
    lines.extend(_render_package_safety(payload))
    lines.extend(_render_package_pool(payload))
    lines.extend(_render_package_bundle(payload))
    lines.extend(_render_package_integrity(payload))
    return "\n".join(lines)


def summarize_package_manifest(path: str | Path) -> str:
    """Render the small published V2 package manifest as key/value text.

    Args:
        path: Published manifest JSON (e.g. ``v2_results.json``).

    Returns:
        A plain-text, indented key/value rendering of the manifest,
        including its outcome, frozen flag, failed checks, and the
        bundle record, which is ``None`` when nothing was persisted.

    Raises:
        ValueError: If the file is missing, empty, or not a JSON object.
    """

    payload = _load_json_object(path)
    lines = ["V2.1-P PUBLISHED MANIFEST"]
    lines.extend(_render_mapping_lines(payload, indent=1))
    return "\n".join(lines)


def _render_section(
    title: str,
    path: str | Path | None,
    renderer: Callable[[Path], str],
) -> str:
    """Render one report section, skipping a missing or absent path.

    Args:
        title: Section heading text.
        path: Input path for this section, or ``None`` to skip it.
        renderer: Callable that turns the resolved path into text.

    Returns:
        The section text, headed by an ``=== TITLE ===`` line. Skipped
        sections carry a one-line explanatory note instead of a body.
    """

    header = f"=== {title} ==="
    if path is None:
        return f"{header}\n(skipped: no path provided)"
    target = Path(path)
    if not target.is_file():
        return f"{header}\n(skipped: file not found: {target})"
    try:
        body = renderer(target)
    except ValueError as error:
        return f"{header}\n(skipped: {error})"
    return f"{header}\n{body}"


def render_kaggle_import_report(
    result_path: str | Path,
    *,
    log_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    tail: int = 40,
) -> str:
    """Compose the Kaggle log, benchmark result, and manifest as text.

    Args:
        result_path: Full benchmark result JSON.
        log_path: Optional downloaded Kaggle log file.
        manifest_path: Optional small published manifest JSON.
        tail: Number of trailing log lines to include.

    Returns:
        One plain-text report with ``=== SECTION ===`` separators. A
        section whose path is ``None`` or missing is skipped with a
        one-line note instead of raising.
    """

    sections = [
        _render_section(
            f"KAGGLE LOG (last {tail} lines)",
            log_path,
            lambda target: summarize_kaggle_log(target, tail=tail),
        ),
        _render_section(
            "V2 BENCHMARK RESULT", result_path, summarize_v2_result
        ),
        _render_section(
            "PUBLISHED MANIFEST", manifest_path, summarize_v2_manifest
        ),
    ]
    return "\n\n".join(sections)


def render_d2_import_report(
    result_path: str | Path,
    *,
    log_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    tail: int = 40,
) -> str:
    """Compose the D2 transformer-challenge result, manifest, and log.

    Renders, in order: the D2 header, the incumbent (V2.1-D1 winner)
    line, the fallback baseline on both windows, a per-seed table with
    the reported (median) seed marked, the seed spread, the
    pre-registered decision block, and an integrity block with the
    execution-config, incumbent-artifact, and hard-negative-pool
    signatures. Reads defensively throughout, so a malformed or
    partial D2 artifact still renders instead of raising.

    Args:
        result_path: Full D2 transformer-challenge result JSON (e.g.
            ``v2_transformer_challenge.json``).
        log_path: Optional downloaded Kaggle log file.
        manifest_path: Optional small published D2 manifest JSON.
        tail: Number of trailing log lines to include.

    Returns:
        One plain-text report with ``=== SECTION ===`` separators. A
        section whose path is ``None`` or missing is skipped with a
        one-line note instead of raising.
    """

    def _render_full(target: Path) -> str:
        payload = _load_json_object(target)
        lines = [summarize_d2_result(target), ""]
        lines.extend(_render_d2_incumbent(payload))
        lines.extend(_render_d2_fallback(payload))
        lines.extend(_render_d2_seeds(payload))
        lines.extend(_render_d2_decision(payload))
        lines.extend(_render_d2_integrity(payload))
        return "\n".join(lines)

    sections = [
        _render_section(
            "V2.1-D2 TRANSFORMER CHALLENGE RESULT", result_path, _render_full
        ),
        _render_section(
            "D2 PUBLISHED MANIFEST", manifest_path, summarize_d2_manifest
        ),
        _render_section(
            f"KAGGLE LOG (last {tail} lines)",
            log_path,
            lambda target: summarize_kaggle_log(target, tail=tail),
        ),
    ]
    return "\n\n".join(sections)


def render_package_import_report(
    result_path: str | Path,
    *,
    log_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    tail: int = 40,
) -> str:
    """Compose the V2 frozen-package result, manifest, and Kaggle log.

    Renders, in order: the outcome and the explicit freeze verdict, the
    exact reproduction gate with its check table and any divergences,
    the pinned candidate and the three frozen scopes, the calibration
    and outer evidence against the frozen S7 fallback, the development
    safety margin and its evidence status, the hard-negative pool, the
    joblib bundle, and the provenance and boundary block. Reads
    defensively throughout, so a smoke-mode, malformed, or partial
    package artifact still renders instead of raising.

    Args:
        result_path: Full V2 package result JSON (e.g.
            ``v2_package.json``).
        log_path: Optional downloaded Kaggle log file.
        manifest_path: Optional small published package manifest JSON
            (e.g. ``v2_results.json``).
        tail: Number of trailing log lines to include.

    Returns:
        One plain-text report with ``=== SECTION ===`` separators. A
        section whose path is ``None`` or missing is skipped with a
        one-line note instead of raising.
    """

    sections = [
        _render_section(
            "V2.1-P FROZEN PACKAGE RESULT",
            result_path,
            summarize_package_result,
        ),
        _render_section(
            "V2.1-P PUBLISHED MANIFEST",
            manifest_path,
            summarize_package_manifest,
        ),
        _render_section(
            f"KAGGLE LOG (last {tail} lines)",
            log_path,
            lambda target: summarize_kaggle_log(target, tail=tail),
        ),
    ]
    return "\n\n".join(sections)


_STRESS_METRICS = (
    "macro_f1",
    "critical_f1",
    "critical_precision",
    "critical_recall",
)
_STRESS_ARMS = ("v2_combined", "s7_fallback_alone")


def load_stress_payload(path: str | Path) -> dict[str, Any]:
    """Load one V2.1-C stress JSON object, tolerating a missing file.

    Unlike ``_load_json_object``, a missing or empty file returns an
    empty mapping instead of raising, because the sealed ``stress``
    run may not have happened yet when a notebook calls this loader.

    Args:
        path: JSON file to load.

    Returns:
        The parsed top-level JSON object, or ``{}`` when the file is
        absent or empty.

    Raises:
        ValueError: If the file exists, is non-empty, and its parsed
            content is not a JSON object.
    """

    target = Path(path)
    if not target.is_file():
        return {}
    text = target.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in: {target}")
    return payload


def _fmt_interval(value: Any) -> str:
    """Format a two-element ``[lower, upper]`` bootstrap interval.

    Args:
        value: Value to format, expected to be a two-element sequence
            of numbers.

    Returns:
        ``"[lower, upper]"`` with six-decimal bounds, or ``"N/A"`` when
        the value is not a two-element numeric sequence.
    """

    if isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
        numeric = (
            isinstance(lower, (int, float))
            and isinstance(upper, (int, float))
            and not isinstance(lower, bool)
            and not isinstance(upper, bool)
        )
        if numeric:
            return f"[{lower:.6f}, {upper:.6f}]"
    return "N/A"


def summarize_stress_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the confirmatory verdict and evidence from a stress result.

    Unlike the path-based ``summarize_*`` helpers above, this function
    takes an already-parsed JSON object -- the V2.1-C stress result is
    consumed by the reporting layer as a payload, not a path, since the
    notebook must render cleanly even before the sealed result exists.
    Every field is read defensively via ``_dig``/``.get()`` so a
    missing or partial block yields ``None`` instead of raising.

    Args:
        payload: Parsed V2.1-C stress confirmatory result JSON object,
            or an empty mapping when the sealed run has not happened.

    Returns:
        A compact dict with the verdict fields (``status``,
        ``confirmed``, ``deploy``, ``complete``), the gate outcome,
        both arms' scientific-view metrics, the paired contrast, the
        override mechanics, and the expectation check.
    """

    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        gates = {}
    return {
        "schema_version": payload.get("schema_version"),
        "code_schema": payload.get("code_schema"),
        "stage": payload.get("stage"),
        "adr": payload.get("adr"),
        "status": payload.get("status"),
        "confirmed": payload.get("confirmed"),
        "complete": payload.get("complete"),
        "deploy": payload.get("deploy"),
        "gates": {
            "required_gate_count": gates.get("required_gate_count"),
            "passed_count": gates.get("passed_count"),
            "passed": gates.get("passed"),
        },
        "paired": {
            "critical_f1_gain": _dig(
                payload, "primary", "paired", "critical_f1_gain"
            ),
            "macro_f1_gain": _dig(
                payload, "primary", "paired", "macro_f1_gain"
            ),
            "critical_precision_gain": _dig(
                payload, "primary", "paired", "critical_precision_gain"
            ),
            "critical_recall_gain": _dig(
                payload, "primary", "paired", "critical_recall_gain"
            ),
        },
        "arms": {
            arm: _dig(
                payload, "primary", "arms", arm, "metrics", default={}
            )
            for arm in _STRESS_ARMS
        },
        "override": {
            "override_decisions": _dig(
                payload, "primary", "override", "override_decisions"
            ),
            "effective_overrides": _dig(
                payload, "primary", "override", "effective_overrides"
            ),
            "rows": _dig(payload, "primary", "rows"),
        },
        "expectation": dict(payload.get("expectation") or {}),
        "signature": payload.get("signature"),
        "opened_at": payload.get("opened_at"),
        "remaining_sealed": list(payload.get("remaining_sealed") or []),
    }


def summarize_stress_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the identity fields from the published stress manifest.

    Args:
        payload: Parsed small published stress manifest JSON object
            (``config/v2_stress_results.json``), or an empty mapping
            when not yet published.

    Returns:
        A compact dict with the manifest's schema version, stage,
        status, confirmed flag, deploy flag, signature, and opened_at
        timestamp, read defensively so a missing field yields ``None``
        instead of raising.
    """

    return {
        "schema_version": payload.get("schema_version"),
        "stage": payload.get("stage"),
        "status": payload.get("status"),
        "confirmed": payload.get("confirmed"),
        "deploy": payload.get("deploy"),
        "signature": payload.get("signature"),
        "opened_at": payload.get("opened_at"),
    }


def _render_stress_verdict(
    result_payload: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
) -> list[str]:
    """Render the stress verdict block, first and unmissable.

    Args:
        result_payload: Parsed V2.1-C stress confirmatory result JSON
            object, or an empty mapping when the sealed run has not
            happened yet.
        protocol_payload: Parsed frozen stress protocol object, used
            only to fall back to the pre-registered stage, ADR, scope,
            and required gate count before the result exists.

    Returns:
        Lines with the stage/ADR/scope header, the status, the
        confirmed flag, the always-false deploy flag, the gate pass
        count out of the required count, and an explicit
        CONFIRMED/NOT_CONFIRMED/no-verdict-yet banner.
    """

    status = result_payload.get("status", "UNKNOWN")
    confirmed = result_payload.get("confirmed")
    deploy = result_payload.get("deploy")
    gates = result_payload.get("gates")
    if not isinstance(gates, Mapping):
        gates = {}
    passed_count = gates.get("passed_count")
    required = gates.get("required_gate_count")
    if required is None:
        protocol_gates = protocol_payload.get("gates")
        if isinstance(protocol_gates, Mapping):
            required = protocol_gates.get("required_gate_count")
    stage = result_payload.get("stage") or protocol_payload.get("stage")
    adr = result_payload.get("adr") or protocol_payload.get("adr")
    scope = result_payload.get("stress_scope")
    if not isinstance(scope, Mapping) or not scope:
        scope = protocol_payload.get("stress_scope")
    if not isinstance(scope, Mapping):
        scope = {}

    lines = ["VERDICT"]
    lines.append(f"  stage: {stage or 'N/A'}   adr: {adr or 'N/A'}")
    lines.append(
        f"  stress_scope: {scope.get('start', 'N/A')} to "
        f"{scope.get('end', 'N/A')}"
    )
    lines.append(f"  status: {status}")
    lines.append(f"  confirmed: {_fmt_flag(confirmed)}")
    lines.append(f"  deploy: {_fmt_flag(deploy)}")
    lines.append(
        f"  gates_passed: {_fmt_count(passed_count)} / "
        f"{_fmt_count(required)}"
    )
    lines.append("")
    if status == "CONFIRMED":
        lines.append(
            "  *** CONFIRMED: the frozen V2 package passed all "
            "pre-registered gates on the sealed stress partition. ***"
        )
    elif status == "NOT_CONFIRMED":
        lines.append(
            "  *** NOT_CONFIRMED: at least one pre-registered gate "
            "failed on the sealed stress partition. ***"
        )
    else:
        lines.append(f"  *** NO SEALED VERDICT YET: status={status} ***")
    lines.append(
        "  A CONFIRMED verdict never authorizes deployment; deploy is "
        "always false."
    )
    lines.append("")
    return lines


def _render_stress_gate_table(results: list[Any]) -> list[str]:
    """Render the fixed-width table of the four pre-registered gates.

    Args:
        results: The ``gates.results`` list from the stress result.

    Returns:
        Lines of a fixed-width text table, header and separator first,
        marking every failing gate with ``!`` so a failure can never
        be missed, followed by a loud banner when any gate failed.
    """

    headers = {
        "marker": " ",
        "name": "gate",
        "observed": "observed",
        "limit": "limit",
        "strict": "strict",
        "verdict": "verdict",
    }
    rows: list[dict[str, str]] = []
    any_failed = False
    for entry in results:
        if not isinstance(entry, Mapping):
            continue
        passed = entry.get("passed")
        if passed is not True:
            any_failed = True
        rows.append({
            "marker": "" if passed is True else "!",
            "name": str(entry.get("name", "?")),
            "observed": _fmt_num(entry.get("observed")),
            "limit": _fmt_num(entry.get("limit")),
            "strict": _fmt_flag(entry.get("strict")),
            "verdict": _fmt_verdict(passed),
        })
    if not rows:
        return ["  (no gate results available)"]
    widths = {
        key: max(len(header), max(len(row[key]) for row in rows))
        for key, header in headers.items()
    }
    header_line = "  " + "  ".join(
        headers[key].ljust(widths[key]) for key in headers
    )
    separator = "  " + "  ".join("-" * widths[key] for key in headers)
    lines = [header_line, separator]
    for row in rows:
        lines.append(
            "  " + "  ".join(row[key].ljust(widths[key]) for key in headers)
        )
    lines.append("  ! marks a failed gate.")
    if any_failed:
        lines.append("")
        lines.append("  *** AT LEAST ONE GATE FAILED. ***")
    return lines


def _render_stress_paired(payload: Mapping[str, Any]) -> list[str]:
    """Render the paired contrast between the two arms, prominently.

    Args:
        payload: Parsed V2.1-C stress confirmatory result JSON object.

    Returns:
        Lines with each arm's critical F1 on identical rows, the
        paired gain (which may be negative), and its bootstrap
        interval, stating explicitly that this is the drift-controlled
        comparison.
    """

    v2_f1 = _dig(
        payload, "primary", "arms", "v2_combined", "metrics", "critical_f1"
    )
    s7_f1 = _dig(
        payload,
        "primary",
        "arms",
        "s7_fallback_alone",
        "metrics",
        "critical_f1",
    )
    gain = _dig(payload, "primary", "paired", "critical_f1_gain")
    interval = _dig(payload, "bootstrap", "paired_critical_f1_gain")

    lines = ["=" * 66]
    lines.append("THE PAIRED CONTRAST (PRIMARY, DRIFT-CONTROLLED)")
    lines.append("=" * 66)
    lines.append(f"  v2_combined critical_f1:       {_fmt_num(v2_f1)}")
    lines.append(f"  s7_fallback_alone critical_f1: {_fmt_num(s7_f1)}")
    lines.append(f"  paired critical_f1_gain:       {_fmt_num(gain)}")
    lines.append(
        f"  bootstrap 95% CI on the gain:  {_fmt_interval(interval)}"
    )
    lines.append("")
    lines.extend(
        (
            "  Both arms are scored in one pass over the identical",
            "  sealed rows, so this paired contrast is immune to the",
            "  class-mix drift between the stress, development, and",
            "  test windows. This is the drift-controlled comparison",
            "  and the scientific primary result of this stage.",
        )
    )
    lines.append("")
    return lines


def _render_stress_expectation(
    result_payload: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
) -> list[str]:
    """Render the pre-registered expectation check, marked diagnostic.

    Args:
        result_payload: Parsed V2.1-C stress confirmatory result JSON
            object, or an empty mapping when the sealed run has not
            happened yet.
        protocol_payload: Parsed frozen stress protocol object, used
            as the authoritative source for the pre-registered
            development gain when the result does not carry one.

    Returns:
        Lines comparing the pre-registered development paired gain
        against the observed stress paired gain and their sign
        agreement, labeled explicitly as diagnostic, not a gate.
    """

    expectation = result_payload.get("expectation")
    if not isinstance(expectation, Mapping):
        expectation = {}
    development_gain = expectation.get("development_paired_gain")
    if development_gain is None:
        protocol_expectation = protocol_payload.get("expectation")
        if isinstance(protocol_expectation, Mapping):
            development_gain = protocol_expectation.get(
                "development_paired_gain"
            )
    observed_gain = expectation.get("observed_paired_gain")
    agrees = expectation.get("agrees_in_sign")

    lines = ["EXPECTATION CHECK (DIAGNOSTIC, NOT A GATE)"]
    lines.append(
        "  development_paired_gain (pre-registered): "
        f"{_fmt_num(development_gain)}"
    )
    lines.append(
        "  observed_paired_gain (stress, sealed):     "
        f"{_fmt_num(observed_gain)}"
    )
    lines.append(f"  agrees_in_sign: {_fmt_flag(agrees)}")
    if agrees is False:
        lines.append("")
        lines.append(
            "  *** SIGN DISAGREEMENT: the observed stress gain does "
            "not agree in sign with the pre-registered development "
            "gain. This is diagnostic only and is not a gate. ***"
        )
    lines.append("")
    return lines


def _stress_cross_window_caveat() -> list[str]:
    """State the fixed cross-window comparability caveat.

    Returns:
        Lines warning that absolute metrics on ``stress`` are NOT
        comparable to the S8 numbers on ``test``, because the class
        mix differs between the two windows for the same frozen
        model, and that only the paired contrast between the two
        arms, on identical rows, is drift-controlled. This caveat is
        unconditional and always renders.
    """

    return [
        "  CAVEAT: absolute metrics on stress are NOT comparable to",
        "  the S8 numbers on test -- the class mix differs between",
        "  the two windows for the same frozen model. Only the",
        "  paired contrast (same rows, both arms) above is",
        "  drift-controlled; these absolute numbers are not.",
        "",
    ]


def _render_stress_arms_table(payload: Mapping[str, Any]) -> list[str]:
    """Render both arms' scientific-view metrics side by side.

    Args:
        payload: Parsed V2.1-C stress confirmatory result JSON object.

    Returns:
        Lines with macro_f1, critical_f1, critical_precision, and
        critical_recall for both arms, each metric followed by its
        bootstrap interval.
    """

    lines: list[str] = []
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        bootstrap = {}
    for arm in _STRESS_ARMS:
        metrics = _dig(payload, "primary", "arms", arm, "metrics", default={})
        if not isinstance(metrics, Mapping):
            metrics = {}
        arm_bootstrap = bootstrap.get(arm)
        if not isinstance(arm_bootstrap, Mapping):
            arm_bootstrap = {}
        lines.append(f"  {arm}:")
        for metric in _STRESS_METRICS:
            value = _fmt_num(metrics.get(metric))
            interval = _fmt_interval(arm_bootstrap.get(metric))
            lines.append(f"    {metric}: {value}  (95% CI: {interval})")
    return lines


def _render_stress_scientific_arms(payload: Mapping[str, Any]) -> list[str]:
    """Render block 5: the cross-window caveat and both arms' metrics.

    Args:
        payload: Parsed V2.1-C stress confirmatory result JSON object.

    Returns:
        Lines opening with the fixed cross-window caveat, then both
        arms' scientific-view metrics with bootstrap intervals.
    """

    lines = ["BOTH ARMS (SCIENTIFIC VIEW)"]
    lines.extend(_stress_cross_window_caveat())
    lines.extend(_render_stress_arms_table(payload))
    lines.append("")
    return lines


def _render_stress_override(payload: Mapping[str, Any]) -> list[str]:
    """Render the stage-A override mechanics on the scientific view.

    Args:
        payload: Parsed V2.1-C stress confirmatory result JSON object.

    Returns:
        Lines with the override decision count, the effective
        override count, the row count, and effective overrides as a
        share of rows.
    """

    decisions = _dig(payload, "primary", "override", "override_decisions")
    effective = _dig(payload, "primary", "override", "effective_overrides")
    rows = _dig(payload, "primary", "rows")
    share = None
    if isinstance(effective, (int, float)) and not isinstance(effective, bool):
        if isinstance(rows, (int, float)) and not isinstance(rows, bool):
            if rows > 0:
                share = effective / rows

    lines = ["OVERRIDE MECHANICS (SCIENTIFIC VIEW)"]
    lines.append(f"  override_decisions: {_fmt_count(decisions)}")
    lines.append(f"  effective_overrides: {_fmt_count(effective)}")
    lines.append(f"  rows: {_fmt_count(rows)}")
    lines.append(f"  effective_overrides_share_of_rows: {_fmt_num(share)}")
    lines.append("")
    return lines


def _render_stress_scope_counts(payload: Mapping[str, Any]) -> list[str]:
    """Render the scope counts block, including the S2 difference.

    Args:
        payload: Parsed V2.1-C stress confirmatory result JSON object.

    Returns:
        Lines with every scope count and, separately, the
        ``s2_difference`` sub-block comparing reconstructed counts
        against the frozen S2 evidence.
    """

    counts = payload.get("scope_counts")
    if not isinstance(counts, Mapping):
        counts = {}
    lines = ["SCOPE COUNTS"]
    for key in sorted(counts):
        if key == "s2_difference":
            continue
        lines.append(f"  {key}: {_fmt_scalar(counts[key])}")
    lines.append("  s2_difference:")
    difference = counts.get("s2_difference")
    if isinstance(difference, Mapping) and difference:
        lines.extend(_render_mapping_lines(difference, indent=2))
    else:
        lines.append("    (not available)")
    lines.append("")
    return lines


def _render_stress_operational(payload: Mapping[str, Any]) -> list[str]:
    """Render the operational secondary view, excluded from the verdict.

    Args:
        payload: Parsed V2.1-C stress confirmatory result JSON object.

    Returns:
        Lines with the operational view's row count and both arms'
        metrics, marked clearly as not participating in the decision.
    """

    operational = payload.get("operational_secondary")
    if not isinstance(operational, Mapping):
        operational = {}
    lines = ["OPERATIONAL SECONDARY VIEW"]
    lines.append(
        "  NOT part of the decision: gates are evaluated on the "
        "scientific view only."
    )
    lines.append(f"  rows: {_fmt_count(operational.get('rows'))}")
    for arm in _STRESS_ARMS:
        metrics = _dig(operational, "arms", arm, "metrics", default={})
        if not isinstance(metrics, Mapping):
            metrics = {}
        lines.append(f"  {arm}:")
        for metric in _STRESS_METRICS:
            lines.append(f"    {metric}: {_fmt_num(metrics.get(metric))}")
    paired_gain = _dig(operational, "paired", "critical_f1_gain")
    lines.append(f"  paired_critical_f1_gain: {_fmt_num(paired_gain)}")
    lines.append("")
    return lines


def _render_stress_integrity(
    result_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
) -> list[str]:
    """Render provenance, the sealed boundary, and manifest agreement.

    Args:
        result_payload: Parsed V2.1-C stress confirmatory result JSON
            object, or an empty mapping when the sealed run has not
            happened yet.
        manifest_payload: Parsed small published stress manifest JSON
            object, or an empty mapping when not yet published.
        protocol_payload: Parsed frozen stress protocol object, used
            only as a fallback for ``remaining_sealed`` before the
            result exists.

    Returns:
        Lines with both schema versions, the opened_at timestamp, the
        signature, the remaining sealed partitions, and an explicit
        agreement verdict between the result and the manifest.
    """

    remaining = result_payload.get("remaining_sealed")
    if not remaining:
        remaining = protocol_payload.get("remaining_sealed")
    lines = ["INTEGRITY"]
    lines.append(
        f"  schema_version: {result_payload.get('schema_version', 'N/A')}"
    )
    lines.append(f"  code_schema: {result_payload.get('code_schema', 'N/A')}")
    lines.append(f"  opened_at: {result_payload.get('opened_at', 'N/A')}")
    lines.append(f"  signature: {result_payload.get('signature', 'N/A')}")
    lines.append(f"  remaining_sealed: {_fmt_list(remaining)}")
    lines.append("")
    if not manifest_payload:
        lines.append(
            "  manifest_agreement: N/A (no published manifest available)"
        )
        return lines
    disagreements = [
        field
        for field in ("status", "confirmed", "signature")
        if manifest_payload.get(field) is not None
        and manifest_payload.get(field) != result_payload.get(field)
    ]
    if disagreements:
        lines.append(
            "  *** MANIFEST DISAGREES WITH RESULT AT: "
            f"{_fmt_list(disagreements)} ***"
        )
    else:
        lines.append(
            "  manifest_agreement: agrees on status/confirmed/signature"
        )
    return lines


def render_stress_import_report(
    result_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
) -> str:
    """Compose the V2.1-C stress confirmatory report as plain text.

    Unlike the path-based ``render_*_import_report`` functions above,
    this renders directly from already-parsed JSON objects. The
    notebook loads the frozen protocol (always present), the sealed
    result, and the published manifest itself, and this function must
    never raise on a result or manifest that is empty -- the sealed
    run may not have happened yet -- or on any partially-shaped
    payload. Every block reads defensively, so a missing key always
    shows a placeholder instead of raising, and the pre-registered
    protocol context (stage, scope, required gate count, development
    gain) still renders even before the sealed result exists.

    Renders, in order: the verdict, the gate table, the paired
    contrast (the primary scientific comparison), the pre-registered
    expectation check, both arms side by side on the scientific view
    (opening with the fixed cross-window comparability caveat), the
    override mechanics, the scope counts, the operational secondary
    view, and the integrity block.

    Args:
        result_payload: Parsed V2.1-C stress confirmatory result JSON
            object, or an empty mapping when the sealed run has not
            happened yet.
        manifest_payload: Parsed small published stress manifest JSON
            object, or an empty mapping when not yet published.
        protocol_payload: Parsed frozen ``v2_stress_protocol.json``
            object.

    Returns:
        One plain-text report with the nine blocks described above.
    """

    lines: list[str] = ["V2.1-C STRESS CONFIRMATORY EVALUATION", ""]
    lines.extend(_render_stress_verdict(result_payload, protocol_payload))
    lines.append("GATES (SCIENTIFIC VIEW, 4 REQUIRED)")
    gates = result_payload.get("gates")
    results = gates.get("results") if isinstance(gates, Mapping) else None
    lines.extend(
        _render_stress_gate_table(results if isinstance(results, list) else [])
    )
    lines.append("")
    lines.extend(_render_stress_paired(result_payload))
    lines.extend(_render_stress_expectation(result_payload, protocol_payload))
    lines.extend(_render_stress_scientific_arms(result_payload))
    lines.extend(_render_stress_override(result_payload))
    lines.extend(_render_stress_scope_counts(result_payload))
    lines.extend(_render_stress_operational(result_payload))
    lines.extend(
        _render_stress_integrity(
            result_payload, manifest_payload, protocol_payload
        )
    )
    return "\n".join(lines)


__all__ = [
    "load_stress_payload",
    "read_kaggle_log",
    "render_d2_import_report",
    "render_kaggle_import_report",
    "render_package_import_report",
    "render_stress_import_report",
    "summarize_d2_manifest",
    "summarize_d2_result",
    "summarize_kaggle_log",
    "summarize_package_manifest",
    "summarize_package_result",
    "summarize_stress_manifest",
    "summarize_stress_result",
    "summarize_v2_manifest",
    "summarize_v2_result",
]
