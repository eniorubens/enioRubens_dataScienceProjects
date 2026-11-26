"""Build the bilingual static executive report (``site/index.html`` and ``site/en.html``).

The page reads only results already stored in ``outputs/`` (tournament V1, distance ablation V1,
final evaluation F1 and F2, ledger) and the raw Olist CSVs (descriptive route network and
supervised counts). It never trains a model, never opens the test period and never writes to
``outputs/``, ``data/``, ``docs/`` or ``notebooks/``. Every number in the page text is computed
here and interpolated into the PT/EN templates; no result is typed by hand.

Usage: ``python scripts/build_site.py [--output site]``.
Exit codes: 0 success, 1 stored results or data missing, 2 stored results contradict the narrative.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
CANDIDATE = "CatBoost"
REFERENCE = "baseline_logistic_promise_plus_state"
RULE = "baseline_negative_promised_lead_days"
PREVALENCE = "baseline_prevalence"
GAIN_THRESHOLD = 0.05
MIN_WIN_SHARE = 0.75
GATE_RATIO = 0.95
CALENDAR_FEATURES = ["purchase_month", "days_to_black_friday", "days_to_christmas"]
RAW_TABLES = ("orders", "items", "customers", "sellers", "geolocation")
RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}
RUNS = (("tournament", "V1"), ("distance_ablation", "V1"), ("final_evaluation", "F1"), ("final_evaluation", "F2"))
NOTEBOOKS = {"pt": "notebooks/pt_br/11_relatorio_final.ipynb", "en": "notebooks/en_us/11_final_report.ipynb"}
REPORT_HTML = {"pt": "reports/relatorio_final_pt.html", "en": "reports/final_report_en.html"}
# Each page offers the README in its own language (falls back to README.md).
README_FILES = {"pt": "README.pt-BR.md", "en": "README.md"}
# Plotly basemap (Natural Earth, served by cdn.plot.ly/un/), vendored so the site works offline.
TOPOJSON_FILES = ("south-america_110m.json",)
TOPOJSON_SOURCE = Path(__file__).resolve().parent / "vendor"

SECTION_IDS = ["dados", "rotas", "torneio", "teste-oficial", "atualizacao", "decisao", "reproducao", "fontes"]

CSS = (
    ":root{--ink:#14263e;--muted:#526276;--blue:#2563eb;--paper:#f6f8fb}*{box-sizing:border-box}"
    "html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);"
    "font:17px/1.7 Arial,sans-serif}header,main,footer{max-width:1120px;margin:auto}"
    "header{display:flex;justify-content:space-between;align-items:center;padding:28px 28px;"
    "border-bottom:1px solid #dce3ec}.brand{font-size:20px;font-weight:bold;letter-spacing:1px}"
    ".brand span{font-size:12px;padding:6px 9px;background:#dce8ff;border-radius:6px;margin-left:6px}"
    ".language{font-size:15px}a{color:#1c56b7;text-underline-offset:3px}main{padding:0 28px}"
    ".hero{padding:65px 0 35px}.eyebrow{font-size:12px;font-weight:bold;letter-spacing:1.6px;"
    "text-transform:uppercase;color:var(--blue)}h1{font-size:clamp(34px,5vw,57px);line-height:1.12;"
    "max-width:850px;margin:18px 0 24px;letter-spacing:-1.8px}.lead{font-size:21px;max-width:790px;"
    "color:var(--muted)}.scope{font-size:14px;color:var(--muted)}.stats{display:grid;"
    "grid-template-columns:repeat(3,1fr);gap:14px;margin:32px 0 15px}.stats>div{background:white;"
    "border:1px solid #dce3ec;border-radius:12px;padding:22px}.stats strong{display:block;"
    "font-size:34px;line-height:1.2}.stats span{display:block;font-size:14px;color:var(--muted);"
    "margin-top:8px}nav{display:flex;flex-wrap:wrap;gap:18px;font-size:14px;padding:20px 0 30px}"
    "nav a{text-decoration:none}section{background:white;border:1px solid #dce3ec;border-radius:14px;"
    "padding:34px;margin:0 0 24px;scroll-margin-top:18px;min-width:0;overflow:hidden}"
    ".section-no{color:var(--blue);font-size:12px;font-weight:bold;letter-spacing:2px}"
    "h2{font-size:28px;line-height:1.3;margin:5px 0 16px}p{max-width:900px}"
    ".insight{background:#edf3fb;border-left:3px solid #648bd3;padding:16px 20px;color:#30465f;"
    "font-size:15px}.plotly-graph-div{width:100%}.downloads{display:flex;gap:15px;flex-wrap:wrap}"
    ".downloads a{border:1px solid #bfd0e8;border-radius:7px;padding:9px 14px;text-decoration:none;"
    "font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#13253d;color:#f5f7fa;"
    "border-radius:8px;padding:18px;font-size:13px;line-height:1.6}.sources{background:transparent}"
    ".sources h2{font-size:21px}.sources p,.sources summary{font-size:14px}"
    ".file-links{overflow-wrap:anywhere}footer{display:flex;justify-content:space-between;"
    "padding:15px 28px 40px;color:var(--muted);font-size:13px}.skip{position:absolute;top:-100px}"
    ".skip:focus{top:5px;background:white;padding:8px;z-index:2}a:focus-visible,"
    "summary:focus-visible{outline:3px solid #2563eb;outline-offset:4px}"
    "@media(max-width:650px){header,main,footer{padding-left:16px;padding-right:16px}"
    ".hero{padding-top:34px}section{padding:18px 12px}.stats{gap:8px}.stats>div{padding:12px}"
    ".stats strong{font-size:23px}.stats span{font-size:12px}.lead{font-size:18px}nav{gap:12px}"
    "h2{font-size:24px}}@media print{body{background:white}nav,.language,.downloads{display:none}"
    "section{break-inside:avoid;border:none}.hero{padding-top:20px}}\n"
    ".table-scroll{overflow-x:auto;margin:16px 0}table{border-collapse:collapse;width:100%;"
    "font-size:13px;line-height:1.5}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce3ec;"
    "white-space:nowrap}th{background:#edf3fb}summary{cursor:pointer;color:#1c56b7;font-size:14px}\n"
    "/* additions for this report */\n"
    ".stats strong.stat-text{font-size:25px}.refs h3{font-size:16px;margin:22px 0 8px}"
    ".refs p{font-size:13px;line-height:1.55;padding-left:26px;text-indent:-26px;margin:0 0 9px}"
    ".note{font-size:14px;color:var(--muted)}.chart-title{font-size:13px;color:var(--muted);margin:18px 0 0}"
    "td.num,th.num{text-align:right}.sources table{font-size:12px}"
    "@media(max-width:650px){.stats strong.stat-text{font-size:17px}}"
    "@media print{details>summary{display:none}details .table-scroll{display:block}}\n"
)

# ----------------------------------------------------------------------------------------------
# Number and date formatting
# ----------------------------------------------------------------------------------------------

MONTHS = {
    "pt": ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
           "outubro", "novembro", "dezembro"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"],
}
DASH = "—"


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA


def fmt_number(value: float, digits: int = 3, lang: str = "pt", signed: bool = False) -> str:
    """Format a number with decimal comma and thousand point (PT) or decimal point and comma (EN).

    Parameters
    ----------
    value : float
        Number to format; ``None``/NaN gives an em dash.
    digits : int, default 3
        Decimal places.
    lang : {"pt", "en"}, default "pt"
        Separator convention.
    signed : bool, default False
        Prefix non-negative numbers with ``+``. Negative numbers always use the minus sign (U+2212).

    Returns
    -------
    str
        Formatted number.
    """
    if _is_missing(value):
        return DASH
    rounded = round(float(value), digits)
    text = f"{abs(rounded):,.{digits}f}"
    if lang == "pt":
        text = text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    if rounded < 0:
        return "−" + text
    return "+" + text if signed else text


def fmt_int(value: float, lang: str = "pt") -> str:
    """Format an integer with thousand separators (see :func:`fmt_number`)."""
    return fmt_number(value, 0, lang)


def fmt_pct(value: float, digits: int = 1, lang: str = "pt", signed: bool = False) -> str:
    """Format a proportion as a percentage (``0.139`` becomes ``13,9%`` in PT)."""
    if _is_missing(value):
        return DASH
    return fmt_number(100 * float(value), digits, lang, signed) + "%"


def fmt_range(low: float, high: float, digits: int = 3, lang: str = "pt", signed: bool = False) -> str:
    """Format an interval as ``low a high`` (PT) or ``low to high`` (EN)."""
    joiner = " a " if lang == "pt" else " to "
    return fmt_number(low, digits, lang, signed) + joiner + fmt_number(high, digits, lang, signed)


def fmt_month(period: str, lang: str = "pt") -> str:
    """Format ``YYYY-MM`` as ``outubro de 2017`` (PT) or ``October 2017`` (EN)."""
    year, month = str(period)[:7].split("-")
    name = MONTHS[lang][int(month) - 1]
    return f"{name} de {year}" if lang == "pt" else f"{name} {year}"


def fmt_date(iso: str, lang: str = "pt") -> str:
    """Format ``YYYY-MM-DD`` as ``1º de junho de 2018`` (PT) or ``June 1, 2018`` (EN)."""
    year, month, day = (int(part) for part in str(iso)[:10].split("-"))
    name = MONTHS[lang][month - 1]
    if lang == "pt":
        return f"{'1º' if day == 1 else day} de {name} de {year}"
    return f"{name} {day}, {year}"


# ----------------------------------------------------------------------------------------------
# Hashing, manifest and small HTML helpers
# ----------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_sources(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    """Map each existing relative path (POSIX style) to its SHA-256, sorted by path.

    Parameters
    ----------
    root : pathlib.Path
        Base directory.
    relative_paths : iterable of str
        Paths relative to ``root``; a directory contributes every file below it.

    Returns
    -------
    dict
        ``{relative_path: sha256}``; missing paths raise ``FileNotFoundError``.
    """
    found: dict[str, str] = {}
    for relative in relative_paths:
        target = root / relative
        if not target.exists():
            raise FileNotFoundError(target)
        files = sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else [target]
        for file in files:
            found[file.relative_to(root).as_posix()] = sha256_file(file)
    return dict(sorted(found.items()))


def esc(value: object) -> str:
    """Escape a value for HTML text."""
    return html.escape(str(value), quote=True)


def table_html(headers: list[str], rows: list[list[str]], numeric: Iterable[int] = ()) -> str:
    """Render a plain HTML table; cells are escaped, numeric columns right-aligned."""
    numeric = set(numeric)

    def cell(tag: str, index: int, text: str) -> str:
        css = ' class="num"' if index in numeric else ""
        return f"<{tag}{css}>{esc(text)}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def details_html(summary: str, content: str) -> str:
    """Wrap content in a collapsed ``<details>`` block."""
    return f"<details><summary>{esc(summary)}</summary>{content}</details>"


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------


class MissingInputs(RuntimeError):
    """Stored results or raw data needed by the page are not available."""


class NarrativeMismatch(RuntimeError):
    """Stored numbers contradict a statement the page narrative relies on."""


@dataclass
class Data:
    """Everything the page reads, loaded once."""

    root: Path
    tournament: dict
    tournament_meta: dict
    ablation: dict
    f1: dict
    f2: dict
    ledger: list[dict]
    orders: pd.DataFrame
    supervised: pd.DataFrame
    pairs: pd.DataFrame
    geolocation: pd.DataFrame
    classification_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    regression_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime: pd.DataFrame = field(default_factory=pd.DataFrame)


def inputs_available(root: Path) -> bool:
    """Return True when the stored runs and the raw CSVs needed by the build exist."""
    runs = all((root / "outputs" / "runs" / name / version).is_dir() for name, version in RUNS)
    ledger = (root / "outputs" / "runs" / "final_evaluation" / "LEDGER.jsonl").is_file()
    raw = all((root / "data" / "raw" / "olist" / "csv" / file).is_file() for file in RAW_FILES.values())
    return runs and ledger and raw


def read_ledger(root: Path) -> list[dict]:
    """Read the append-only test ledger (one JSON object per line)."""
    path = root / "outputs" / "runs" / "final_evaluation" / "LEDGER.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_data(root: Path) -> Data:
    """Load stored runs and raw tables (read-only; nothing is trained or recomputed)."""
    if not inputs_available(root):
        raise MissingInputs("outputs/runs (V1, F1, F2, LEDGER) or data/raw/olist/csv is missing")
    from ecom_delivery_sla import (  # noqa: PLC0415 - imported late so --help works without the environment
        build_supervised_orders,
        promise_regime_table,
        read_olist_tables,
        tournament_summary,
    )
    from ecom_delivery_sla.run_cache import load_run_unchecked  # noqa: PLC0415
    from ecom_delivery_sla_viz import route_pairs  # noqa: PLC0415

    cache = root / "outputs" / "runs"
    tournament, tournament_meta = load_run_unchecked("tournament", "V1", cache_root=cache)
    ablation, _ = load_run_unchecked("distance_ablation", "V1", cache_root=cache)
    f1, _ = load_run_unchecked("final_evaluation", "F1", cache_root=cache)
    f2, _ = load_run_unchecked("final_evaluation", "F2", cache_root=cache)
    tables = read_olist_tables(root / "data" / "raw" / "olist" / "csv", table_names=RAW_TABLES)
    orders = tables["orders"]
    supervised = build_supervised_orders(orders)
    pairs = route_pairs(supervised, tables["items"], tables["sellers"], tables["customers"], tables["geolocation"])
    data = Data(
        root=root,
        tournament=tournament,
        tournament_meta=tournament_meta,
        ablation=ablation,
        f1=f1,
        f2=f2,
        ledger=read_ledger(root),
        orders=orders,
        supervised=supervised,
        pairs=pairs,
        geolocation=tables["geolocation"],
    )
    data.classification_summary = tournament_summary(tournament["classification_results"])
    data.regression_summary = tournament_summary(tournament["regression_results"])
    data.regime = promise_regime_table(supervised)
    return data


# ----------------------------------------------------------------------------------------------
# Raw (language-neutral) facts
# ----------------------------------------------------------------------------------------------


def _one(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column] == value
    if mask.sum() < 1:
        raise NarrativeMismatch(f"no row with {conditions}")
    return frame[mask].iloc[0]


def _metric(frame: pd.DataFrame, model: str, metric: str = "average_precision") -> pd.Series:
    return _one(frame, model=model, metric=metric)


def compute_raw(data: Data) -> dict:
    """Compute every number used in the narrative as plain Python values (no formatting).

    Raises
    ------
    NarrativeMismatch
        When the stored results contradict a statement of the narrative templates.
    """
    from ecom_delivery_sla_viz import route_table  # noqa: PLC0415

    f1, f2 = data.f1, data.f2
    cs = data.classification_summary.set_index("model")
    rs = data.regression_summary
    results = data.tournament["classification_results"]
    config = data.tournament_meta["config"]
    n_folds = int(results["fold"].nunique())
    min_wins = math.ceil(MIN_WIN_SHARE * n_folds)

    models = data.classification_summary[~data.classification_summary["is_baseline"]].copy()
    models = models.sort_values("average_precision_gain_vs_baseline", ascending=False).reset_index(drop=True)
    candidates = models[
        (models["average_precision_gain_vs_baseline"] >= GAIN_THRESHOLD)
        & (models["average_precision_folds_won"] >= min_wins)
    ]["model"].tolist()
    best_baseline = str(cs.loc[CANDIDATE, "average_precision_best_baseline"])

    reg_models = rs[~rs["is_baseline"]].sort_values("mae_gain_vs_baseline", ascending=False).reset_index(drop=True)
    reg_candidates = reg_models[
        (reg_models["mae_gain_vs_baseline"] >= 0.03) & (reg_models["mae_folds_won"] >= min_wins)
    ]["model"].tolist()
    reg_best_baseline = str(reg_models["mae_best_baseline"].iloc[0])
    reg_bb_mae = float(rs.set_index("model").loc[reg_best_baseline, "mae_mean"])

    from ecom_delivery_sla import ablation_summary  # noqa: PLC0415

    adopted = False
    for task in ("classification", "regression"):
        table = ablation_summary(data.ablation[f"{task}_ablation"], reference="mean", task=task)
        prefix = "average_precision" if task == "classification" else "mae"
        delta, won = table[f"{prefix}_delta_vs_mean"], table[f"{prefix}_folds_won"]
        adopted = adopted or bool(((delta > 0) & (won >= min_wins)).any())

    sup = data.supervised
    pairs = data.pairs
    edges = route_table(pairs, "state", min_orders=100)
    real_edges = edges[~edges["is_self_loop"]]
    rj = _one(edges, origin="SP", destination="RJ")
    mg = _one(edges, origin="SP", destination="MG")
    sp_sp = _one(edges, origin="SP", destination="SP")

    m1, p1 = f1["classification_metrics"], f1["paired_difference"].iloc[0]
    pop1 = f1["populations"].set_index("split")
    prov1 = f1["provenance"].iloc[0]
    cens = f1["censoring"].set_index("scenario")
    rel1 = f1["reliability"]
    bins1 = {round(float(b), 1): r for b, r in zip(rel1[rel1["model"] == CANDIDATE]["bin_lower"],
                                                   rel1[rel1["model"] == CANDIDATE].to_dict("records"), strict=True)}
    topk = f1["top_k"]
    top5 = {m: float(_one(topk, model=m, top_fraction=0.05)["precision"]) for m in (CANDIDATE, REFERENCE, RULE)}
    psi = f1["psi"].set_index("feature")["psi"]

    m2 = f2["classification_metrics"]
    pop2 = f2["populations"].set_index("variant")
    labels = f2["label_availability"].set_index("purchase_month")
    gate = f2["gate_summary"].set_index("variant")
    variants_scored = [v for v in f2["variants"]["variant"] if bool(_one(f2["variants"], variant=v)["scored"])]
    variants_failed = [v for v in f2["variants"]["variant"] if v not in variants_scored]
    ap2 = {v: float(_metric(m2, v)["estimate"]) for v in variants_scored}
    paired2 = f2["paired_difference"]
    vs_log = paired2[(paired2["reference"] == REFERENCE) & paired2["difference"].notna()]
    vs_rule = paired2[(paired2["reference"] == RULE) & paired2["difference"].notna()]
    rel2 = f2["reliability"]

    def profile(model: str) -> dict:
        part = rel2[(rel2["model"] == model) & (rel2["n"] > 0)]
        total = float(part["n"].sum())
        return {
            "first_bin_share": float(part.iloc[0]["n"]) / total,
            "mean_pred": float((part["n"] * part["mean_predicted"]).sum() / total),
            "high": part[part["bin_lower"] >= 0.1 - 1e-9],
        }

    prof_a = profile("A_f1_frozen")
    prof_other = [profile(v) for v in ("B_retrain", "C_recency", "E_promise_feature")]
    a_high = prof_a["high"][prof_a["high"]["n"] >= 10]  # bins with at least 10 orders

    regime = data.regime.copy()
    regime["purchase_month"] = regime["purchase_month"].astype(str)
    dev = regime[(regime["purchase_month"] >= config["first_val_month"]) & (regime["purchase_month"] <= config["last_val_month"])]
    month = regime.set_index("purchase_month")
    aug = "2018-08"  # evaluation month, fixed by the F2 design (P-06)

    raw = {
        "n_orders": len(data.orders),
        "n_delivered": int((data.orders["order_status"] == "delivered").sum()),
        "n_sup": len(sup),
        "first_purchase": str(sup["order_purchase_timestamp"].min())[:7],
        "last_purchase": str(sup["order_purchase_timestamp"].max())[:7],
        "first_year": int(str(sup["order_purchase_timestamp"].min())[:4]),
        "last_year": int(str(sup["order_purchase_timestamp"].max())[:4]),
        "n_breach": int(sup["sla_breach"].sum()),
        "prev_all": float(sup["sla_breach"].mean()),
        "n_pairs": len(pairs),
        "n_routes": len(real_edges),
        "sp_share": float((pairs["origin_state"] == "SP").mean()),
        "rj_rate": float(rj["breach_rate"]), "rj_km": float(rj["distance_median_km"]),
        "mg_rate": float(mg["breach_rate"]), "mg_km": float(mg["distance_median_km"]),
        "rj_mg_ratio": float(rj["breach_rate"] / mg["breach_rate"]),
        "sp_sp_rate": float(sp_sp["breach_rate"]),
        "n_folds": n_folds, "min_wins": min_wins, "gain_threshold": GAIN_THRESHOLD,
        "first_val": config["first_val_month"], "last_val": config["last_val_month"],
        "embargo": int(config["embargo_days"]),
        "candidates": candidates, "reg_candidates": reg_candidates,
        "best_baseline": best_baseline,
        "ap_cat": float(cs.loc[CANDIDATE, "average_precision_mean"]),
        "ap_bb": float(cs.loc[best_baseline, "average_precision_mean"]),
        "gain_cat": float(cs.loc[CANDIDATE, "average_precision_gain_vs_baseline"]),
        "won_cat": int(cs.loc[CANDIDATE, "average_precision_folds_won"]),
        "runner_ups": [
            (r["model"], float(r["average_precision_gain_vs_baseline"]), int(r["average_precision_folds_won"]))
            for _, r in models.iloc[1:3].iterrows()
        ],
        "rule_dev_ap": float(cs.loc[RULE, "average_precision_mean"]),
        "prev_dev_ap": float(cs.loc[PREVALENCE, "average_precision_mean"]),
        "reg_best_model": str(reg_models.iloc[0]["model"]),
        "reg_best_gain": float(reg_models.iloc[0]["mae_gain_vs_baseline"]),
        "reg_bb": reg_best_baseline, "reg_bb_mae": reg_bb_mae,
        "ablation_adopted": adopted,
        "f1_train_n": int(pop1.loc["train", "orders"]), "f1_test_n": int(pop1.loc["test", "orders"]),
        "f1_test_breach": int(pop1.loc["test", "breaches"]),
        "f1_weeks": int(pop1.loc["test", "weeks"]),
        "prev_train": float(pop1.loc["train", "prevalence"]), "prev_test": float(pop1.loc["test", "prevalence"]),
        "train_end": str(prov1["train_end_exclusive"]),
        "n_boot": int(prov1["n_boot"]),
        "n_censored": int(cens.loc["exclude", "censored_orders"]),
        "prev_test_incl": float(cens.loc["include_as_breach", "prevalence"]),
        "ap_f1_cat": _metric(m1, CANDIDATE), "ap_f1_log": _metric(m1, REFERENCE), "ap_f1_rule": _metric(m1, RULE),
        "auc_f1_rule": float(_metric(m1, RULE, "roc_auc")["estimate"]),
        "ece_f1_cat": float(_metric(m1, CANDIDATE, "ece_10bins")["estimate"]),
        "delta": float(p1["difference"]), "delta_low": float(p1["ci_low"]), "delta_high": float(p1["ci_high"]),
        "outcome_pt": str(p1["outcome"]), "outcome_en": str(p1["outcome_en"]),
        "top5": top5,
        "psi_promise": float(psi["promised_lead_days"]), "psi_dest": float(psi["dest_hist_breach_rate"]),
        "bin1": bins1[0.1], "bin2": bins1[0.2],
        "asof": str(labels["as_of"].iloc[0]),
        "n_aug": int(pop2.loc["all", "orders"]), "aug_breach": int(pop2.loc["all", "breaches"]),
        "prev_aug": float(pop2.loc["all", "prevalence"]),
        "ap2": ap2,
        "ap2_log": _metric(m2, REFERENCE), "ap2_rule": _metric(m2, RULE),
        "auc2_rule": float(_metric(m2, RULE, "roc_auc")["estimate"]),
        "variants_failed": variants_failed,
        "d_ratio": float(gate["ap_ratio"].min()) if variants_failed else float("nan"),
        "gate_ratio": GATE_RATIO,
        "vs_log_min": float(vs_log["difference"].min()), "vs_log_max": float(vs_log["difference"].max()),
        "vs_log_ci_excludes_zero": bool((vs_log["ci_high"] < 0).all()),
        "vs_rule_mean": float(vs_rule["difference"].mean()),
        "label_jun": labels.loc["2018-06"].to_dict(), "label_jul": labels.loc["2018-07"].to_dict(),
        "add_orders": int(pop2.loc["B_retrain", "orders"] - pop2.loc["A_f1_frozen", "orders"]),
        "add_breach": int(pop2.loc["B_retrain", "breaches"] - pop2.loc["A_f1_frozen", "breaches"]),
        "a_pred_min": float(a_high["mean_predicted"].min()), "a_pred_max": float(a_high["mean_predicted"].max()),
        "a_obs_min": float(a_high["observed_rate"].min()), "a_obs_max": float(a_high["observed_rate"].max()),
        "bce_pred_min": min(p["mean_pred"] for p in prof_other), "bce_pred_max": max(p["mean_pred"] for p in prof_other),
        "bce_share_min": min(p["first_bin_share"] for p in prof_other),
        "bce_share_max": max(p["first_bin_share"] for p in prof_other),
        "pm_dev_min": float(dev["promise_median"].min()), "pm_dev_max": float(dev["promise_median"].max()),
        "pm_jul": float(month.loc["2018-07", "promise_median"]), "pm_aug": float(month.loc[aug, "promise_median"]),
        "auc_dev_min": float(dev["auc_negative_promise"].min()), "auc_dev_max": float(dev["auc_negative_promise"].max()),
        "prod_eligible": bool(f2["provenance"]["production_eligible"].iloc[0]),
        "ledger_rows": len(data.ledger),
    }
    check_narrative(raw)
    return raw


def check_narrative(raw: dict) -> None:
    """Raise :class:`NarrativeMismatch` when stored results contradict what the templates assert."""
    problems = []
    if raw["candidates"] != [CANDIDATE]:
        problems.append(f"expected CatBoost as the only classification candidate, got {raw['candidates']}")
    if raw["reg_candidates"]:
        problems.append(f"expected no regression candidate, got {raw['reg_candidates']}")
    if raw["outcome_en"] != "inconclusive":
        problems.append(f"F1 outcome is {raw['outcome_en']!r}, narrative assumes 'inconclusive'")
    if not raw["delta_low"] < 0 < raw["delta_high"]:
        problems.append("F1 paired interval does not include zero")
    if raw["ap_f1_rule"]["estimate"] <= max(raw["ap_f1_cat"]["estimate"], raw["ap_f1_log"]["estimate"]):
        problems.append("the promised lead time rule is not the strongest AP in F1")
    if any(raw["ap2"][v] >= raw["ap2_log"]["estimate"] for v in raw["ap2"]):
        problems.append("an F2 variant reached the logistic reference")
    if not raw["vs_log_ci_excludes_zero"]:
        problems.append("F2 paired differences against the logistic baseline do not all exclude zero")
    if not raw["rj_km"] < raw["mg_km"] or not raw["rj_rate"] > raw["mg_rate"]:
        problems.append("SP→RJ is no longer the shorter and higher-rate route versus SP→MG")
    if raw["prod_eligible"]:
        problems.append("production_eligible is true; the narrative assumes false")
    if raw["ablation_adopted"]:
        problems.append("a distance variant met the adoption rule; the narrative assumes mean distance was kept")
    if not raw["label_jul"]["prevalence_unknown"] > raw["label_jul"]["prevalence_known"]:
        problems.append("unknown July labels no longer have higher prevalence than known ones")
    if problems:
        raise NarrativeMismatch("; ".join(problems))


# ----------------------------------------------------------------------------------------------
# Citations and references
# ----------------------------------------------------------------------------------------------

# key -> (prefix of the entry in the notebook references cell, PT citation, EN citation)
CITATIONS = {
    "olist": ("Olist (2018)", "(Olist, 2018)", "(Olist, 2018)"),
    "nosek": ("Nosek, B. A.", "(Nosek et al., 2018)", "(Nosek et al., 2018)"),
    "klein": ("Klein, J. P.", "(Klein e Moeschberger, 2003)", "(Klein and Moeschberger, 2003)"),
    "wilson": ("Wilson, E. B.", "(Wilson, 1927)", "(Wilson, 1927)"),
    "bergmeir": ("Bergmeir, C.", "(Bergmeir e Benítez, 2012)", "(Bergmeir and Benítez, 2012)"),
    "davis": ("Davis, J.", "(Davis e Goadrich, 2006)", "(Davis and Goadrich, 2006)"),
    "prok": ("Prokhorenkova, L.", "(Prokhorenkova et al., 2018)", "(Prokhorenkova et al., 2018)"),
    "kunsch": ("Künsch, H. R.", "(Künsch, 1989)", "(Künsch, 1989)"),
    "degroot": ("DeGroot, M. H.", "(DeGroot e Fienberg, 1983)", "(DeGroot and Fienberg, 1983)"),
    "siddiqi": ("Siddiqi, N.", "(Siddiqi, 2006)", "(Siddiqi, 2006)"),
    "dwork": ("Dwork, C.", "(Dwork et al., 2015)", "(Dwork et al., 2015)"),
    "chapelle": ("Chapelle, O.", "(Chapelle, 2014)", "(Chapelle, 2014)"),
    "kaplan": ("Kaplan, E. L.", "(Kaplan e Meier, 1958)", "(Kaplan and Meier, 1958)"),
}


def parse_references(notebook_path: Path) -> list[str]:
    """Return the reference entries (markdown) of the final cell of notebook 11.

    The entries are returned verbatim, in the order of the cell, without the heading.
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in reversed(notebook["cells"]):
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and re.match(r"##\s+(Refer[êe]ncias|References)\b", text):
            return [block.strip() for block in text.split("\n\n")[1:] if block.strip()]
    raise MissingInputs(f"no references cell in {notebook_path}")


def reference_html(entry: str) -> str:
    """Convert one markdown reference entry (italics and bare URLs) to safe HTML."""
    text = html.escape(entry, quote=False)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', text)


def select_references(entries: list[str], used: Iterable[str]) -> list[str]:
    """Return the entries cited on the page (by citation key), in alphabetical order of the notebook."""
    prefixes = [CITATIONS[key][0] for key in used]
    chosen = []
    for prefix in prefixes:
        matches = [entry for entry in entries if entry.startswith(prefix)]
        if not matches:
            raise MissingInputs(f"reference not found in notebook 11: {prefix}")
        chosen.append(matches[0])
    return [entry for entry in entries if entry in chosen]


class Facts(Mapping):
    """Formatted values for one language; records which citations a template actually uses."""

    def __init__(self, values: dict[str, str], lang: str):
        self.values, self.lang, self.cited = values, lang, set()

    def __getitem__(self, key: str) -> str:
        """Return a formatted fact, or a citation (recording that it was used) for ``c_<key>``."""
        if key.startswith("c_"):
            name = key[2:]
            self.cited.add(name)
            return CITATIONS[name][1 if self.lang == "pt" else 2]
        return self.values[key]

    def __iter__(self):
        """Iterate over the fact names."""
        return iter(self.values)

    def __len__(self) -> int:
        """Return the number of facts."""
        return len(self.values)


def render(template: str, facts: Facts) -> str:
    """Fill a template; every placeholder must exist in ``facts`` (KeyError otherwise)."""
    return Formatter().vformat(template, (), facts)


def make_facts(raw: dict, lang: str) -> Facts:
    """Format the raw facts for one language (decimal comma in PT, decimal point in EN)."""

    def n(value: float, digits: int = 3, signed: bool = False) -> str:
        return fmt_number(value, digits, lang, signed)

    def pct(value: float, digits: int = 1, signed: bool = False) -> str:
        return fmt_pct(value, digits, lang, signed)

    def i(value: float) -> str:
        return fmt_int(value, lang)

    def ci(metric: pd.Series, digits: int = 3) -> str:
        return fmt_range(metric["ci_low"], metric["ci_high"], digits, lang)

    ap2 = raw["ap2"]
    ru = raw["runner_ups"]
    outcome = raw["outcome_pt"] if lang == "pt" else raw["outcome_en"]
    jun, jul = raw["label_jun"], raw["label_jul"]
    values = {
        "n_orders": i(raw["n_orders"]), "n_delivered": i(raw["n_delivered"]), "n_sup": i(raw["n_sup"]),
        "sup_share": pct(raw["n_sup"] / raw["n_orders"]),
        "n_not_sup": i(raw["n_orders"] - raw["n_sup"]),
        "first_purchase": fmt_month(raw["first_purchase"], lang), "last_purchase": fmt_month(raw["last_purchase"], lang),
        "years": f"{raw['first_year']}–{raw['last_year']}",
        "n_breach": i(raw["n_breach"]), "prev_all": pct(raw["prev_all"], 2),
        "n_pairs": i(raw["n_pairs"]), "n_routes": i(raw["n_routes"]), "sp_share": pct(raw["sp_share"]),
        "rj_rate": pct(raw["rj_rate"]), "rj_km": i(raw["rj_km"]), "mg_rate": pct(raw["mg_rate"]), "mg_km": i(raw["mg_km"]),
        "rj_mg_ratio": n(raw["rj_mg_ratio"], 1), "sp_sp_rate": pct(raw["sp_sp_rate"]),
        "n_folds": i(raw["n_folds"]), "min_wins": i(raw["min_wins"]), "gain_threshold": pct(raw["gain_threshold"], 0),
        "first_val": fmt_month(raw["first_val"], lang), "last_val": fmt_month(raw["last_val"], lang),
        "embargo": i(raw["embargo"]),
        "ap_cat": n(raw["ap_cat"]), "ap_bb": n(raw["ap_bb"]), "bb_name": raw["best_baseline"],
        "gain_cat": pct(raw["gain_cat"], 1, True), "won_cat": i(raw["won_cat"]),
        "ru1_name": ru[0][0], "ru1_gain": pct(ru[0][1], 1, True), "ru1_won": i(ru[0][2]),
        "ru2_name": ru[1][0], "ru2_gain": pct(ru[1][1], 1, True), "ru2_won": i(ru[1][2]),
        "rule_dev_ap": n(raw["rule_dev_ap"]), "prev_dev_ap": n(raw["prev_dev_ap"]),
        "reg_best_model": raw["reg_best_model"], "reg_best_gain_abs": pct(abs(raw["reg_best_gain"])),
        "reg_bb_mae": n(raw["reg_bb_mae"], 2),
        "f1_train_n": i(raw["f1_train_n"]), "f1_test_n": i(raw["f1_test_n"]), "f1_test_breach": i(raw["f1_test_breach"]),
        "f1_weeks": i(raw["f1_weeks"]), "prev_train": pct(raw["prev_train"], 2), "prev_test": pct(raw["prev_test"], 2),
        "train_end": fmt_date(raw["train_end"], lang), "n_boot": i(raw["n_boot"]),
        "n_censored": i(raw["n_censored"]), "prev_test_incl": pct(raw["prev_test_incl"], 2),
        "ap_f1_cat": n(raw["ap_f1_cat"]["estimate"]), "ap_f1_cat_ci": ci(raw["ap_f1_cat"]),
        "ap_f1_log": n(raw["ap_f1_log"]["estimate"]), "ap_f1_log_ci": ci(raw["ap_f1_log"]),
        "ap_f1_rule": n(raw["ap_f1_rule"]["estimate"]), "ap_f1_rule_ci": ci(raw["ap_f1_rule"]),
        "auc_f1_rule": n(raw["auc_f1_rule"]), "ece_f1_cat": n(raw["ece_f1_cat"]),
        "delta": n(raw["delta"], 3, True), "delta_ci": fmt_range(raw["delta_low"], raw["delta_high"], 3, lang, True),
        "f1_outcome": outcome, "f1_outcome_cap": outcome[:1].upper() + outcome[1:],
        "p5_cat": pct(raw["top5"][CANDIDATE]), "p5_log": pct(raw["top5"][REFERENCE]), "p5_rule": pct(raw["top5"][RULE]),
        "psi_promise": n(raw["psi_promise"], 2), "psi_dest": n(raw["psi_dest"], 2),
        "b1_pred": pct(raw["bin1"]["mean_predicted"]), "b1_obs": pct(raw["bin1"]["observed_rate"]),
        "b2_pred": pct(raw["bin2"]["mean_predicted"]), "b2_obs": pct(raw["bin2"]["observed_rate"]),
        "asof": fmt_date(raw["asof"], lang), "n_aug": i(raw["n_aug"]), "aug_breach": i(raw["aug_breach"]),
        "prev_aug": pct(raw["prev_aug"], 2),
        "ap_A": n(ap2["A_f1_frozen"]), "ap_B": n(ap2["B_retrain"]), "ap_C": n(ap2["C_recency"]),
        "ap_E": n(ap2["E_promise_feature"]),
        "ap2_log": n(raw["ap2_log"]["estimate"]), "ap2_log_ci": ci(raw["ap2_log"]),
        "ap2_rule": n(raw["ap2_rule"]["estimate"]), "ap2_rule_ci": ci(raw["ap2_rule"]),
        "auc2_rule": n(raw["auc2_rule"]), "d_ratio": n(raw["d_ratio"]), "gate_ratio": n(raw["gate_ratio"], 2),
        "vs_log_min": n(raw["vs_log_min"], 3, True), "vs_log_max": n(raw["vs_log_max"], 3, True),
        "vs_rule_mean": n(raw["vs_rule_mean"], 2, True),
        "jun_known_share": pct(jun["share_known"]), "jun_known_prev": pct(jun["prevalence_known"], 2),
        "jun_unknown_n": i(jun["unknown_labels"]), "jun_unknown_prev": pct(jun["prevalence_unknown"]),
        "jul_known_share": pct(jul["share_known"]), "jul_known_prev": pct(jul["prevalence_known"], 2),
        "jul_unknown_n": i(jul["unknown_labels"]), "jul_unknown_prev": pct(jul["prevalence_unknown"], 2),
        "add_orders": i(raw["add_orders"]), "add_breach": i(raw["add_breach"]),
        "add_prev": pct(raw["add_breach"] / raw["add_orders"], 2),
        "a_pred_min": pct(raw["a_pred_min"], 0), "a_pred_max": pct(raw["a_pred_max"], 0),
        "a_obs_min": pct(raw["a_obs_min"], 0), "a_obs_max": pct(raw["a_obs_max"], 0),
        "bce_pred_min": pct(raw["bce_pred_min"]), "bce_pred_max": pct(raw["bce_pred_max"]),
        "bce_share_min": pct(raw["bce_share_min"]), "bce_share_max": pct(raw["bce_share_max"]),
        "pm_dev_min": n(raw["pm_dev_min"], 1), "pm_dev_max": n(raw["pm_dev_max"], 1),
        "pm_jul": n(raw["pm_jul"], 1), "pm_aug": n(raw["pm_aug"], 1),
        "auc_dev_min": n(raw["auc_dev_min"], 2), "auc_dev_max": n(raw["auc_dev_max"], 2),
        "prod_eligible": str(raw["prod_eligible"]).lower(), "ledger_rows": i(raw["ledger_rows"]),
        "kaggle": KAGGLE_URL,
    }
    return Facts(values, lang)


# ----------------------------------------------------------------------------------------------
# Narrative templates (placeholders are filled from computed facts; no result is typed here)
# ----------------------------------------------------------------------------------------------

TEXT = {
    "pt": {
        "html_lang": "pt-BR", "other_lang": "en", "other_file": "en.html", "other_label": "English",
        "brand": "Olist", "brand_tag": "SLA de entrega",
        "title": "Dá para saber, na hora da compra, se a entrega vai atrasar?",
        "eyebrow": "Olist · E-commerce brasileiro · Portfólio de ciência de dados",
        "lead": (
            "Durante o desenvolvimento, um modelo CatBoost previu o atraso melhor que regras simples, mas esse ganho "
            "não foi confirmado no único teste oficial, cujo resultado foi {f1_outcome}. No período de teste, o sinal "
            "mais forte foi o de uma regra simples sobre o prazo prometido ao cliente. O modelo não está habilitado "
            "para uso operacional."
        ),
        "scope": "Pedidos entregues {years} · previsão no momento da compra · CatBoost vs. baselines",
        "stat1": "pedidos supervisionados (entregues, com datas válidas)",
        "stat2": "ganho de AP do CatBoost sobre a melhor baseline, com folds vencidos (desenvolvimento)",
        "stat3": "leitura pré-registrada do teste oficial (F1), com diferença pareada de AP",
        "nav_label": "Neste relatório", "skip": "Ir para o conteúdo",
        "nav": {"dados": "Dados", "rotas": "Rotas", "torneio": "Torneio", "teste-oficial": "Teste oficial",
                "atualizacao": "Atualização", "decisao": "Decisão", "reproducao": "Reprodução",
                "fontes": "Fontes e rastreabilidade"},
        "noscript": "É necessário JavaScript para os gráficos interativos. As tabelas com os dados estão disponíveis abaixo de cada gráfico.",
        "h_dados": "De onde vêm os dados?",
        "dados_1": (
            "Os dados são o conjunto público <em>Brazilian E-Commerce Public Dataset by Olist</em>, publicado no Kaggle "
            "{c_olist}. A tabela de pedidos reúne {n_orders} pedidos de um marketplace brasileiro, com compras de "
            "{first_purchase} a {last_purchase}. Os dados brutos não são versionados neste repositório, e o download é manual."
        ),
        "dados_2": (
            "A população supervisionada é formada pelos {n_sup} pedidos entregues com datas real e estimada válidas "
            "({sup_share} da tabela). Pedidos cancelados, indisponíveis ou ainda em trânsito ficam fora do alvo e são "
            "tratados em análise separada (decisão P-02). A censura à direita no fim da série {c_klein}, isto é, "
            "pedidos de julho e agosto de 2018 ainda sem entrega na extração, foi tratada pela exclusão desses pedidos: "
            "no teste oficial, {n_censored} pedidos foram excluídos, e, se fossem contados como violações, a prevalência "
            "subiria de {prev_test} para {prev_test_incl}."
        ),
        "dados_3": (
            "A violação do SLA é definida por dia de calendário: o pedido é marcado como violação quando a data de "
            "entrega, truncada ao dia, é posterior à data estimada, também truncada, de modo que uma entrega no próprio "
            "dia prometido não é atraso. Na população supervisionada, {n_breach} pedidos ({prev_all}) violaram o prazo. "
            "A previsão é feita no momento da compra, com informações disponíveis nesse instante: características do "
            "pedido, distância, calendário e histórico da rede vendedor → UF de destino."
        ),
        "dados_link": "Dataset no Kaggle",
        "dados_details": "Ver a distribuição de status dos pedidos",
        "h_rotas": "Onde os atrasos acontecem?",
        "rotas_1": (
            "O mapa mostra a rede de rotas entre estados (UF): cada seta liga a UF do vendedor à UF do cliente, com "
            "largura proporcional ao volume e cor dada pela taxa de atraso. A unidade é o par pedido–vendedor "
            "({n_pairs} pares), e o intervalo de confiança de 95% de cada taxa é calculado pelo método do escore "
            "{c_wilson}, visível ao passar o cursor. O menu do gráfico altera o volume mínimo por rota. O mapa descreve o período inteiro e não "
            "alimenta o modelo."
        ),
        "rotas_details": "Ver as maiores rotas entre UFs",
        "rotas_insight": (
            "A rede tem forma de estrela centrada em São Paulo: {sp_share} dos pares saem de vendedores paulistas. Com a "
            "mesma origem, as taxas diferem muito por destino: SP → RJ tem {rj_rate} de atrasos, com distância mediana "
            "de {rj_km} km, enquanto SP → MG tem {mg_rate}, a {mg_km} km. A rota mais curta tem, portanto, taxa "
            "{rj_mg_ratio} vezes maior, e a distância sozinha não explica o risco. A leitura é descritiva: cada rota "
            "mistura destino e período, e nenhuma relação causal é inferida."
        ),
        "h_torneio": "O modelo supera regras simples?",
        "torneio_1": (
            "No desenvolvimento, {n_folds} janelas mensais ({first_val} a {last_val}) foram avaliadas em ordem temporal, "
            "com treino acumulado e embargo de {embargo} dias entre treino e validação {c_bergmeir}. A métrica principal "
            "é a precisão média (<em>average precision</em>, AP) {c_davis}, adequada a uma classe rara como o atraso. "
            "Cada modelo foi comparado com baselines obrigatórias: a prevalência, a regra do prazo prometido (quanto "
            "menor o prazo, maior o risco) e a regressão logística com prazo prometido e UF, que foi a melhor baseline."
        ),
        "torneio_2": (
            "O critério foi aprovado antes da execução (P-01): um modelo só é candidato se o ganho relativo médio de AP "
            "sobre a melhor baseline for de pelo menos {gain_threshold} e se essa baseline for vencida em pelo menos "
            "{min_wins} dos {n_folds} folds. O primeiro gráfico mostra o ganho médio de cada modelo, com a variação "
            "entre folds e a linha do critério, e o segundo, a AP de cada candidato em cada fold."
        ),
        "torneio_details_cls": "Ver a tabela da classificação (AP)",
        "torneio_details_reg": "Ver a tabela da regressão do atraso em dias (MAE)",
        "torneio_insight": (
            "O CatBoost foi o único candidato: AP média de {ap_cat}, contra {ap_bb} da melhor baseline "
            "(<code>{bb_name}</code>), ganho de {gain_cat} e vitória em {won_cat} de {n_folds} folds. "
            "<code>{ru1_name}</code> ({ru1_gain}, {ru1_won}/{n_folds}) e <code>{ru2_name}</code> ({ru2_gain}, "
            "{ru2_won}/{n_folds}) ficaram abaixo do ganho mínimo. Na regressão do atraso em dias, nenhum modelo foi "
            "candidato: o melhor ({reg_best_model}) ficou {reg_best_gain_abs} pior que a baseline por UF (MAE de "
            "{reg_bb_mae} dias), e a regressão permanece descritiva. A distância também foi testada em variantes "
            "(máxima; máxima mais amplitude), e nenhuma atendeu à regra de adoção (P-03)."
        ),
        "h_teste": "O ganho se confirmou no teste?",
        "teste_1": (
            "O teste oficial (F1) foi aberto uma única vez, sob pré-registro {c_nosek}: o CatBoost, com a configuração "
            "padrão do torneio, foi treinado com {f1_train_n} compras anteriores a {train_end} (prevalência de "
            "{prev_train}) e avaliado em {f1_test_n} pedidos de julho e agosto de 2018 ({f1_test_breach} violações, "
            "prevalência de {prev_test}, em {f1_weeks} semanas). A leitura foi fixada antes do resultado: o ganho seria "
            "confirmado se o intervalo de confiança (IC) de 95% da diferença pareada de AP contra a regressão logística "
            "ficasse acima de zero, com IC obtido por bootstrap em blocos semanais {c_kunsch}."
        ),
        "teste_2": (
            "O primeiro gráfico mostra a AP de cada modelo com seu IC e, ao lado, a diferença pareada em relação à linha "
            "de zero. O segundo mostra a curva de confiabilidade {c_degroot}, que compara a probabilidade prevista com a "
            "taxa observada de atraso. O terceiro mostra o PSI, o índice de estabilidade populacional {c_siddiqi}, que "
            "mede o quanto cada feature mudou entre treino e teste. As features de calendário aparecem em cinza porque "
            "o PSI delas é alto por construção, já que o teste cobre apenas dois meses."
        ),
        "teste_details": "Ver as tabelas de métricas do teste oficial",
        "teste_insight": (
            "O resultado foi {f1_outcome}: AP de {ap_f1_cat} (IC 95% {ap_f1_cat_ci}) para o CatBoost e {ap_f1_log} "
            "(IC 95% {ap_f1_log_ci}) para a logística, com diferença pareada de {delta} (IC 95% {delta_ci}). O IC "
            "inclui zero, e a vantagem de {gain_cat} observada no desenvolvimento não se reproduziu. A regra do prazo "
            "prometido foi o sinal mais forte do teste (AP de {ap_f1_rule}, AUC de {auc_f1_rule}), embora tenha ficado "
            "abaixo da prevalência nos folds de desenvolvimento (AP de {rule_dev_ap}, contra {prev_dev_ap}). Essa "
            "inversão é coerente com a mudança na política de promessa: o PSI de <code>promised_lead_days</code> foi "
            "{psi_promise}. Entre os 5% de pedidos de maior risco, a precisão foi de {p5_cat} no CatBoost, {p5_log} na "
            "logística e {p5_rule} na regra. O CatBoost também superestimou o risco: na faixa de 0,1 a 0,2, previu "
            "{b1_pred} e observou {b1_obs}, e na faixa de 0,2 a 0,3, previu {b2_pred} e observou {b2_obs}. Julho e "
            "agosto de 2018 estão esgotados para novas avaliações."
        ),
        "h_f2": "Atualizar o modelo resolveria?",
        "f2_1": (
            "O F2 é uma análise pós-hoc e exploratória, desenhada depois de o período do F1 ter sido observado. Ele não "
            "confirma nem refuta o F1, porque o reuso adaptativo de um mesmo período de teste invalida a leitura "
            "confirmatória {c_dwork}. Uma atualização do modelo foi simulada em {asof}, com os rótulos já conhecidos "
            "nessa data, e avaliada nas {n_aug} compras de agosto de 2018 ({aug_breach} violações, prevalência de "
            "{prev_aug}). Cinco variantes formam uma lista fechada: A (F1 congelado), B (retreino mensal), C (pesos de "
            "recência), D (recalibração recente) e E (feature de promessa relativa)."
        ),
        "f2_2": (
            "O primeiro gráfico mostra, por mês de compra, a mediana do prazo prometido e o AUC da regra do prazo "
            "prometido, calculado só até junho de 2018. O segundo mostra, na data da atualização, quantas compras recentes "
            "já tinham rótulo e qual era a prevalência de cada grupo, o problema do <em>feedback</em> atrasado "
            "{c_chapelle}: os desfechos mais lentos são os últimos a chegar."
        ),
        "f2_details": "Ver as tabelas do F2 (exploratório)",
        "f2_insight": (
            "Nenhuma variante superou as referências em agosto: a AP foi de {ap_A} (A), {ap_B} (B), {ap_C} (C) e {ap_E} "
            "(E), contra {ap2_log} da logística e {ap2_rule} (IC 95% {ap2_rule_ci}) da regra do prazo prometido, e a "
            "variante D foi reprovada no gate de não inferioridade (razão de AP de {d_ratio}, contra mínimo de "
            "{gate_ratio}). Atualizar o modelo não ajudou: B ({ap_B}) ficou igual a A ({ap_A}). A diferença pareada "
            "contra a logística ficou entre {vs_log_min} e {vs_log_max} em todas as variantes, com IC abaixo de zero. "
            "Uma explicação compatível com as tabelas, mas não testada, é o viés de rótulos lentos: na data da "
            "atualização, só {jul_known_share} das compras de julho tinham rótulo, com prevalência de {jul_known_prev}, "
            "enquanto os {jul_unknown_n} pedidos ainda em trânsito tinham {jul_unknown_prev}. O retreino acrescentou "
            "{add_orders} pedidos com {add_breach} violações ({add_prev}). O erro de calibração mudou de direção sem "
            "diminuir: A superestimou o risco (previu de {a_pred_min} a {a_pred_max} nas faixas altas, contra {a_obs_min} "
            "a {a_obs_max} observados), enquanto B, C e E subestimaram (previram de {bce_pred_min} a {bce_pred_max} em "
            "média, contra {prev_aug} observados, com {bce_share_min} a {bce_share_max} dos pedidos na faixa abaixo de "
            "0,1). A mediana do prazo prometido, que ficou entre {pm_dev_min} e {pm_dev_max} dias nos meses de "
            "desenvolvimento, caiu para {pm_jul} em julho e {pm_aug} em agosto, e o AUC da regra, que oscilou entre "
            "{auc_dev_min} e {auc_dev_max} nos folds, foi de {auc2_rule} em agosto. Os resultados do F2 são exploratórios "
            "e servem apenas para formular hipóteses."
        ),
        "h_decisao": "O que fica para decisão?",
        "decisao_1": (
            "A tabela reúne as estimativas centrais das três etapas: o ganho do desenvolvimento, as estimativas do F1 com "
            "seus intervalos e a diferença pareada com a leitura pré-registrada, e as estimativas do F2 com a leitura "
            "exploratória. Todos os valores vêm das tabelas gravadas."
        ),
        "decisao_2": (
            "Antes de julho de 2018, o CatBoost superou a melhor baseline de forma consistente entre os folds e foi o "
            "único candidato pelo critério pré-registrado, enquanto a regressão do atraso não teve candidato. Esse "
            "ganho não foi confirmado no teste oficial, que foi {f1_outcome}, e a regra simples do prazo prometido "
            "foi uma referência forte nos períodos recentes. Os dados não permitem afirmar que o CatBoost supere as "
            "baselines fora do período de desenvolvimento, nem que deva ser usado operacionalmente. Também não "
            "permitem afirmar que algum mecanismo de atualização do F2 seja melhor que o modelo congelado, que o viés "
            "de rótulos lentos seja a causa da piora, nem que haja relações causais entre rota, vendedor, promessa e "
            "atraso. Por isso, <code>production_eligible = {prod_eligible}</code>."
        ),
        "decisao_3": (
            "Duas hipóteses ficam para um próximo ciclo e só podem ser testadas em dados posteriores a agosto de 2018, "
            "que o Olist não contém. A primeira é treinar, na atualização, apenas com coortes de desfecho maduro, ou "
            "tratar os pedidos em trânsito como censurados em um modelo de sobrevivência {c_kaplan}. A segunda é usar a "
            "regra do prazo prometido como alerta de mudança na política de promessa, e não como modelo. Cada uma exige "
            "protocolo próprio, registrado antes da avaliação."
        ),
        "decisao_insight": (
            "Leitura para decisão: nenhum modelo está habilitado para uso operacional. O sinal mais confiável do período "
            "recente é o próprio prazo prometido, e a mudança de regime dele sugere que monitorar a política de "
            "promessa e a prevalência rende mais do que treinar novos modelos com os dados atuais."
        ),
        "h_repro": "Reprodução",
        "repro_1": (
            "Esta página é gerada por <code>scripts/build_site.py</code> a partir de resultados já gravados em "
            "<code>outputs/</code> e dos CSVs brutos do Olist, sem treinar modelos nem reabrir o período de teste. Os "
            "comandos abaixo recriam o ambiente, verificam os dados, executam os notebooks e regeneram a página."
        ),
        "repro_code": (
            "conda env create -f environment.yml\n"
            "conda activate ecom-delivery-sla\n"
            "python scripts/verify_data.py --extract\n"
            "# notebooks 01 a 11 em ordem, português primeiro (cache em outputs/runs)\n"
            "pytest\n"
            "ruff check src tests scripts\n"
            "python scripts/export_report.py   # exporta o notebook 11 para HTML (reports/)\n"
            "python scripts/build_site.py --output site"
        ),
        "repro_insight": (
            "Limitações: um único marketplace, cerca de dois anos de dados terminando em 2018, sem informações de "
            "capacidade, tráfego ou rota real, e período de teste esgotado para novas avaliações."
        ),
        "dl_readme": "Baixar o README (.md)", "dl_notebook": "Baixar o notebook 11 (.ipynb)",
        "dl_html": "Baixar o relatório exportado (.html)",
        "h_fontes": "Fontes e rastreabilidade",
        "fontes_1": (
            "Os gráficos e as tabelas foram gerados a partir de resultados gravados localmente, sem novo treino e sem "
            "nova avaliação do período de teste. O manifesto registra o SHA-256 de cada arquivo de origem, com "
            "<code>training_performed = false</code> e <code>test_reopened = false</code>; o ledger do teste tem "
            "{ledger_rows} linhas, uma por versão avaliada."
        ),
        "fontes_details": "Dados dos gráficos, manifesto e ledger",
        "fontes_charts": "Tabelas plotadas (CSV)",
        "fontes_manifest": "Manifesto",
        "fontes_ledger": "Ledger do período de teste (uma linha por avaliação)",
        "fontes_dataset": (
            "Proveniência do dataset: Kaggle, {kaggle}. O SHA-256 do ZIP original está registrado em "
            "<code>data/raw/olist/README.md</code>, e o SHA-256 dos CSVs usados está no manifesto."
        ),
        "refs": "Referências",
        "back": "Voltar ao topo", "footer": "Olist · Risco de SLA de entrega",
        "chart_label": "Gráfico interativo",
    },
    "en": {
        "html_lang": "en", "other_lang": "pt-BR", "other_file": "index.html", "other_label": "Português",
        "brand": "Olist", "brand_tag": "Delivery SLA",
        "title": "Can a late delivery be predicted at purchase time?",
        "eyebrow": "Olist · Brazilian e-commerce · Data science portfolio",
        "lead": (
            "During development, a CatBoost model predicted delays better than simple rules, but that gain was not "
            "confirmed in the single official test, whose result was {f1_outcome}. In the test period, the strongest "
            "signal was a simple rule on the lead time promised to the customer. The model is not cleared for "
            "operational use."
        ),
        "scope": "Delivered orders {years} · prediction at purchase time · CatBoost vs. baselines",
        "stat1": "supervised orders (delivered, with valid dates)",
        "stat2": "AP gain of CatBoost over the best baseline, with folds won (development)",
        "stat3": "pre-registered reading of the official test (F1), with the paired AP difference",
        "nav_label": "In this report", "skip": "Skip to content",
        "nav": {"dados": "Data", "rotas": "Routes", "torneio": "Tournament", "teste-oficial": "Official test",
                "atualizacao": "Update", "decisao": "Decision", "reproducao": "Reproduction",
                "fontes": "Sources and traceability"},
        "noscript": "JavaScript is required for the interactive charts. The data tables are available below each chart.",
        "h_dados": "Where does the data come from?",
        "dados_1": (
            "The data are the public <em>Brazilian E-Commerce Public Dataset by Olist</em>, published on Kaggle "
            "{c_olist}. The orders table holds {n_orders} orders of a Brazilian marketplace, with purchases from "
            "{first_purchase} to {last_purchase}. The raw data are not versioned in this repository, and the download "
            "is manual."
        ),
        "dados_2": (
            "The supervised population consists of the {n_sup} delivered orders with valid actual and estimated "
            "delivery dates ({sup_share} of the table). Canceled, unavailable or still in-transit orders stay outside "
            "the target and are handled in a separate analysis (decision P-02). Right censoring at the end of the "
            "series {c_klein}, that is, orders of July and August 2018 still undelivered at extraction, was handled by "
            "excluding those orders: in the official test, {n_censored} orders were excluded, and, if counted as "
            "breaches, the prevalence would rise from {prev_test} to {prev_test_incl}."
        ),
        "dados_3": (
            "The SLA breach is defined by calendar day: an order is flagged as a breach when the delivery date, "
            "truncated to the day, is later than the estimated date, also truncated, so a delivery on the promised day "
            "itself is not a delay. In the supervised population, {n_breach} orders ({prev_all}) breached the deadline. "
            "The prediction is made at purchase time, with information available at that moment: order characteristics, "
            "distance, calendar and the history of the seller → destination-state network."
        ),
        "dados_link": "Dataset on Kaggle",
        "dados_details": "View the order status distribution",
        "h_rotas": "Where do the delays happen?",
        "rotas_1": (
            "The map shows the network of routes between states (UF): each arrow links the seller's state to the "
            "customer's state, with width proportional to volume and color given by the breach rate. The unit is the "
            "order-seller pair ({n_pairs} pairs), and the 95% confidence interval of each rate is computed by the "
            "score method {c_wilson}, visible on hover. The chart menu changes the minimum volume per route. The map describes the "
            "whole period and does not feed the model."
        ),
        "rotas_details": "View the largest routes between states",
        "rotas_insight": (
            "The network is a star centered on São Paulo: {sp_share} of the pairs leave from São Paulo sellers. From the "
            "same origin, rates differ widely by destination: SP → RJ has {rj_rate} of breaches, with a median distance "
            "of {rj_km} km, while SP → MG has {mg_rate}, at {mg_km} km. The shorter route thus has a rate "
            "{rj_mg_ratio} times higher, and distance alone does not explain the risk. The reading is descriptive: each "
            "route mixes destination and period, and no causal relation is inferred."
        ),
        "h_torneio": "Does the model beat simple rules?",
        "torneio_1": (
            "In development, {n_folds} monthly windows ({first_val} to {last_val}) were evaluated in time order, with "
            "accumulated training and a {embargo}-day embargo between training and validation {c_bergmeir}. The main "
            "metric is average precision (AP) {c_davis}, suited to a rare class such as the delay. Every model was "
            "compared with mandatory baselines: the prevalence, the promised lead time rule (the shorter the lead time, "
            "the higher the risk) and the logistic regression with promised lead time and state, which was the best "
            "baseline."
        ),
        "torneio_2": (
            "The criterion was approved before the run (P-01): a model is a candidate only if the mean relative AP gain "
            "over the best baseline is at least {gain_threshold} and that baseline is beaten in at least {min_wins} of "
            "the {n_folds} folds. The first chart shows the mean gain of each model, with the spread across folds and "
            "the criterion line, and the second shows the AP of each candidate in each fold."
        ),
        "torneio_details_cls": "View the classification table (AP)",
        "torneio_details_reg": "View the delay-in-days regression table (MAE)",
        "torneio_insight": (
            "CatBoost was the only candidate: mean AP of {ap_cat}, against {ap_bb} for the best baseline "
            "(<code>{bb_name}</code>), a gain of {gain_cat} and {won_cat} of {n_folds} folds won. "
            "<code>{ru1_name}</code> ({ru1_gain}, {ru1_won}/{n_folds}) and <code>{ru2_name}</code> ({ru2_gain}, "
            "{ru2_won}/{n_folds}) fell below the minimum gain. In the regression of the delay in days, no model was a "
            "candidate: the best ({reg_best_model}) was {reg_best_gain_abs} worse than the per-state baseline (MAE of "
            "{reg_bb_mae} days), and the regression remains descriptive. Distance was also tested in variants (maximum; "
            "maximum plus range), and none met the adoption rule (P-03)."
        ),
        "h_teste": "Was the gain confirmed in the test?",
        "teste_1": (
            "The official test (F1) was opened a single time, under pre-registration {c_nosek}: CatBoost, with the "
            "default tournament configuration, was trained on {f1_train_n} purchases before {train_end} (prevalence of "
            "{prev_train}) and evaluated on {f1_test_n} orders of July and August 2018 ({f1_test_breach} breaches, "
            "prevalence of {prev_test}, over {f1_weeks} weeks). The reading was fixed before the result: the gain would "
            "be confirmed if the 95% confidence interval (CI) of the paired AP difference against the logistic "
            "regression were above zero, with the CI obtained by a bootstrap in weekly blocks {c_kunsch}."
        ),
        "teste_2": (
            "The first chart shows the AP of each model with its CI and, beside it, the paired difference relative to "
            "the zero line. The second shows the reliability curve {c_degroot}, which compares the predicted "
            "probability with the observed breach rate. The third shows the PSI, the population stability index "
            "{c_siddiqi}, which measures how much each feature changed between training and test. Calendar features "
            "appear in gray because their PSI is high by construction, since the test covers only two months."
        ),
        "teste_details": "View the official test metric tables",
        "teste_insight": (
            "The result was {f1_outcome}: AP of {ap_f1_cat} (95% CI {ap_f1_cat_ci}) for CatBoost and {ap_f1_log} "
            "(95% CI {ap_f1_log_ci}) for the logistic baseline, with a paired difference of {delta} (95% CI "
            "{delta_ci}). The CI includes zero, and the {gain_cat} advantage seen in development was not reproduced. "
            "The promised lead time rule was the strongest signal in the test (AP of {ap_f1_rule}, AUC of "
            "{auc_f1_rule}), although it fell below the prevalence in the development folds (AP of {rule_dev_ap}, "
            "against {prev_dev_ap}). This reversal is consistent with the change in the promise policy: the PSI of "
            "<code>promised_lead_days</code> was {psi_promise}. Among the 5% highest-risk orders, precision was "
            "{p5_cat} for CatBoost, {p5_log} for the logistic baseline and {p5_rule} for the rule. CatBoost also "
            "overestimated the risk: in the 0.1 to 0.2 bin it predicted {b1_pred} and observed {b1_obs}, and in the 0.2 "
            "to 0.3 bin it predicted {b2_pred} and observed {b2_obs}. July and August 2018 are exhausted for new "
            "evaluations."
        ),
        "h_f2": "Would updating the model fix it?",
        "f2_1": (
            "F2 is a post-hoc, exploratory analysis, designed after the F1 period had been observed. It neither "
            "confirms nor refutes F1, because the adaptive reuse of the same test period invalidates a confirmatory "
            "reading {c_dwork}. A model update was simulated on {asof}, with the labels already known on that date, "
            "and evaluated on the {n_aug} purchases of August 2018 ({aug_breach} breaches, prevalence of {prev_aug}). "
            "Five variants form a closed list: A (frozen F1), B (monthly retraining), C (recency weights), D (recent "
            "recalibration) and E (relative promise feature)."
        ),
        "f2_2": (
            "The first chart shows, by purchase month, the median promised lead time and the AUC of the promised lead "
            "time rule, computed only up to June 2018. The second shows, on the update date, how many recent purchases "
            "already had a label and the prevalence of each group, the delayed feedback problem {c_chapelle}: the "
            "slowest outcomes are the last to arrive."
        ),
        "f2_details": "View the F2 tables (exploratory)",
        "f2_insight": (
            "No variant beat the references in August: AP was {ap_A} (A), {ap_B} (B), {ap_C} (C) and {ap_E} (E), "
            "against {ap2_log} for the logistic baseline and {ap2_rule} (95% CI {ap2_rule_ci}) for the promised lead "
            "time rule, and variant D failed the non-inferiority gate (AP ratio of {d_ratio}, against a minimum of "
            "{gate_ratio}). Updating the model did not help: B ({ap_B}) matched A ({ap_A}). The paired difference "
            "against the logistic baseline stayed between {vs_log_min} and {vs_log_max} in every variant, with the CI "
            "below zero. An explanation consistent with the tables, but not tested, is slow-label bias: on the update "
            "date, only {jul_known_share} of the July purchases had a label, with a prevalence of {jul_known_prev}, "
            "while the {jul_unknown_n} orders still in transit had {jul_unknown_prev}. Retraining added {add_orders} "
            "orders with {add_breach} breaches ({add_prev}). The calibration error changed direction without "
            "shrinking: A overestimated the risk (it predicted {a_pred_min} to {a_pred_max} in the high bins, against "
            "{a_obs_min} to {a_obs_max} observed), while B, C and E underestimated it (mean predictions of "
            "{bce_pred_min} to {bce_pred_max}, against {prev_aug} observed, with {bce_share_min} to {bce_share_max} of "
            "the orders in the bin below 0.1). The median promised lead time, which stayed between {pm_dev_min} and "
            "{pm_dev_max} days in the development months, fell to {pm_jul} in July and {pm_aug} in August, and the "
            "AUC of the rule, which oscillated between {auc_dev_min} and {auc_dev_max} in the folds, was {auc2_rule} "
            "in August. The F2 results are exploratory and serve only to formulate hypotheses."
        ),
        "h_decisao": "What is left for decision?",
        "decisao_1": (
            "The table gathers the central estimates of the three stages: the development gain, the F1 estimates with "
            "their intervals and the paired difference with the pre-registered reading, and the F2 estimates with the "
            "exploratory reading. All values come from the stored tables."
        ),
        "decisao_2": (
            "Before July 2018, CatBoost beat the best baseline consistently across folds and was the only candidate "
            "under the pre-registered criterion, while the delay regression had no candidate. That gain was not "
            "confirmed in the official test, which was {f1_outcome}, and the simple promised lead time rule was a "
            "strong reference in recent periods. The data do not show that CatBoost beats the baselines outside the "
            "development period, or that it should be used operationally. They also do not show that any F2 update "
            "mechanism is better than the frozen model, that slow-label bias caused the deterioration, or that route, "
            "seller, promise and delay are causally related. For these reasons, "
            "<code>production_eligible = {prod_eligible}</code>."
        ),
        "decisao_3": (
            "Two hypotheses are left for a next cycle and can be tested only on data after August 2018, which the Olist "
            "dataset does not contain. The first is to train, at update time, only on cohorts with a mature outcome, "
            "or to treat in-transit orders as censored in a survival model {c_kaplan}. The second is to use the promised "
            "lead time rule as an alert of a change in the promise policy, not as a model. Each requires its own "
            "protocol, registered before the evaluation."
        ),
        "decisao_insight": (
            "Reading for decision: no model is cleared for operational use. The most reliable signal of the recent "
            "period is the promised lead time itself, and its change of regime suggests that monitoring the promise "
            "policy and the prevalence would pay off more than training new models on the current data."
        ),
        "h_repro": "Reproduction",
        "repro_1": (
            "This page is generated by <code>scripts/build_site.py</code> from results already stored in "
            "<code>outputs/</code> and the raw Olist CSVs, without training models or reopening the test period. The "
            "commands below recreate the environment, verify the data, run the notebooks and regenerate the page."
        ),
        "repro_code": (
            "conda env create -f environment.yml\n"
            "conda activate ecom-delivery-sla\n"
            "python scripts/verify_data.py --extract\n"
            "# notebooks 01 to 11 in order, Portuguese first (cache in outputs/runs)\n"
            "pytest\n"
            "ruff check src tests scripts\n"
            "python scripts/export_report.py   # export notebook 11 to HTML (reports/)\n"
            "python scripts/build_site.py --output site"
        ),
        "repro_insight": (
            "Limitations: a single marketplace, about two years of data ending in 2018, no capacity, traffic or actual "
            "route information, and a test period exhausted for new evaluations."
        ),
        "dl_readme": "Download the README (.md)", "dl_notebook": "Download notebook 11 (.ipynb)",
        "dl_html": "Download the exported report (.html)",
        "h_fontes": "Sources and traceability",
        "fontes_1": (
            "Charts and tables were generated from locally stored results, without new training and without a new "
            "evaluation of the test period. The manifest records the SHA-256 of every source file, with "
            "<code>training_performed = false</code> and <code>test_reopened = false</code>; the test ledger has "
            "{ledger_rows} rows, one per evaluated version."
        ),
        "fontes_details": "Chart data, manifest and ledger",
        "fontes_charts": "Plotted tables (CSV)",
        "fontes_manifest": "Manifest",
        "fontes_ledger": "Test-period ledger (one row per evaluation)",
        "fontes_dataset": (
            "Dataset provenance: Kaggle, {kaggle}. The SHA-256 of the original ZIP is recorded in "
            "<code>data/raw/olist/README.md</code>, and the SHA-256 of the CSVs used is in the manifest."
        ),
        "refs": "References",
        "back": "Back to top", "footer": "Olist · Delivery SLA risk",
        "chart_label": "Interactive chart",
    },
}

COLS = {
    "pt": {
        "status": "Status do pedido", "orders": "Pedidos", "share": "Participação",
        "route": "Rota (origem → destino)", "pairs": "Pares", "rate": "Taxa de atraso (IC 95%)",
        "dist": "Distância mediana (km)", "promised": "Prazo prometido mediano (dias)",
        "actual": "Prazo realizado mediano (dias)",
        "model": "Modelo", "ap_mean": "AP média", "ap_sd": "Desvio da AP", "gain": "Ganho sobre a melhor baseline",
        "won": "Folds vencidos", "ece": "ECE médio", "mae_mean": "MAE médio (dias)", "mae_gain": "Ganho de MAE",
        "ap_ci": "AP (IC 95%)", "auc": "AUC", "brier": "Brier", "ece1": "ECE", "p5": "Precisão no top 5%",
        "r5": "Recall no top 5%", "diff": "Diferença pareada de AP (IC 95%)", "reading": "Leitura",
        "variant": "Variante", "status_v": "Situação", "vs_log": "Diferença contra a logística",
        "vs_rule": "Diferença contra a regra", "month": "Mês de compra", "known": "Rótulos conhecidos",
        "prev_known": "Prevalência (conhecidos)", "unknown": "Sem rótulo", "prev_unknown": "Prevalência (sem rótulo)",
        "version": "Versão", "timestamp": "Data/hora", "post_hoc": "Pós-hoc", "train": "Treino", "test": "Avaliação",
        "stage": "Etapa", "item": "Item", "estimate": "Estimativa", "ci": "IC 95%", "note": "Nota",
        "yes": "sim", "no": "não", "failed": "reprovada no gate", "evaluated": "avaliada", "baseline": "baseline",
    },
    "en": {
        "status": "Order status", "orders": "Orders", "share": "Share",
        "route": "Route (origin → destination)", "pairs": "Pairs", "rate": "Breach rate (95% CI)",
        "dist": "Median distance (km)", "promised": "Median promised lead time (days)",
        "actual": "Median actual lead time (days)",
        "model": "Model", "ap_mean": "Mean AP", "ap_sd": "AP SD", "gain": "Gain over the best baseline",
        "won": "Folds won", "ece": "Mean ECE", "mae_mean": "Mean MAE (days)", "mae_gain": "MAE gain",
        "ap_ci": "AP (95% CI)", "auc": "AUC", "brier": "Brier", "ece1": "ECE", "p5": "Precision at top 5%",
        "r5": "Recall at top 5%", "diff": "Paired AP difference (95% CI)", "reading": "Reading",
        "variant": "Variant", "status_v": "Status", "vs_log": "Difference vs. logistic",
        "vs_rule": "Difference vs. rule", "month": "Purchase month", "known": "Known labels",
        "prev_known": "Prevalence (known)", "unknown": "Unlabeled", "prev_unknown": "Prevalence (unlabeled)",
        "version": "Version", "timestamp": "Timestamp", "post_hoc": "Post-hoc", "train": "Train", "test": "Evaluation",
        "stage": "Stage", "item": "Item", "estimate": "Estimate", "ci": "95% CI", "note": "Note",
        "yes": "yes", "no": "no", "failed": "failed the gate", "evaluated": "evaluated", "baseline": "baseline",
    },
}

STATUS_PT = {
    "delivered": "entregue", "shipped": "enviado", "canceled": "cancelado", "unavailable": "indisponível",
    "invoiced": "faturado", "processing": "em processamento", "created": "criado", "approved": "aprovado",
}
MODEL_LABELS = {
    "pt": {
        REFERENCE: "Baseline: logística (prazo + UF)", RULE: "Baseline: regra do prazo prometido",
        PREVALENCE: "Baseline: prevalência",
        "A_f1_frozen": "A · F1 congelado", "B_retrain": "B · retreino", "C_recency": "C · pesos de recência",
        "D_recalibration": "D · recalibração", "E_promise_feature": "E · promessa relativa",
    },
    "en": {
        REFERENCE: "Baseline: logistic (lead time + state)", RULE: "Baseline: promised lead time rule",
        PREVALENCE: "Baseline: prevalence",
        "A_f1_frozen": "A · frozen F1", "B_retrain": "B · retraining", "C_recency": "C · recency weights",
        "D_recalibration": "D · recalibration", "E_promise_feature": "E · relative promise",
    },
}


def model_label(name: str, lang: str) -> str:
    """Return a readable model/baseline label; unknown names are returned unchanged."""
    return MODEL_LABELS[lang].get(name, name)


# ----------------------------------------------------------------------------------------------
# Charts and chart data
# ----------------------------------------------------------------------------------------------


def finalize_figure(fig) -> None:
    """Apply the report look (font, transparent paper) without changing sizes or legends."""
    fig.update_layout(font={"family": "Arial, sans-serif", "color": "#24364b"}, autosize=True,
                      paper_bgcolor="rgba(0,0,0,0)")


def geo_topojson_script() -> str:
    """Return a script that preloads the vendored basemaps into Plotly.

    Browsers block ``fetch`` of local JSON when the page is opened from disk
    (``file://``); a ``<script>`` file is allowed, so the topojson is injected into
    ``window.PlotlyGeoAssets``, which plotly.js checks before fetching.
    """
    lines = ["window.PlotlyGeoAssets = window.PlotlyGeoAssets || {topojson: {}};",
             "window.PlotlyGeoAssets.topojson = window.PlotlyGeoAssets.topojson || {};"]
    for name in TOPOJSON_FILES:
        payload = json.loads((TOPOJSON_SOURCE / name).read_text(encoding="utf-8"))
        key = name.removesuffix(".json")
        lines.append(f"window.PlotlyGeoAssets.topojson[{json.dumps(key)}] = {json.dumps(payload, separators=(',', ':'))};")
    return chr(10).join(lines) + chr(10)


def chart_div(fig, div_id: str) -> str:
    """Return the Plotly div (no embedded library, no CDN)."""
    finalize_figure(fig)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"responsive": True, "displaylogo": False, "topojsonURL": "assets/"})


def build_charts(data: Data, lang: str) -> dict:
    """Build every figure for one language.

    Returns
    -------
    dict
        ``{name: plotly Figure}`` in page order.
    """
    from ecom_delivery_sla_viz import (  # noqa: PLC0415
        plot_gain_over_baseline,
        plot_label_availability,
        plot_metric_by_fold,
        plot_metric_intervals,
        plot_promise_regime,
        plot_psi,
        plot_route_network,
    )
    from ecom_delivery_sla_viz.report import plot_reliability_curves  # noqa: PLC0415

    results = data.tournament["classification_results"]
    summary = data.classification_summary
    candidates = summary.loc[~summary["is_baseline"], "model"].tolist()
    best_baseline = summary["average_precision_best_baseline"].replace("", pd.NA).dropna().iloc[0]
    f1, f2 = data.f1, data.f2
    return {
        "network": plot_route_network(data.pairs, data.geolocation, language=lang),
        "gain": plot_gain_over_baseline(summary, metric="average_precision", threshold=GAIN_THRESHOLD,
                                        fold_results=results, language=lang),
        "folds": plot_metric_by_fold(results, fold_col="fold", model_col="model", metric_col="average_precision",
                                     reference=best_baseline, models=candidates, language=lang),
        "f1_intervals": plot_metric_intervals(f1["classification_metrics"], metric="average_precision",
                                              paired=f1["paired_difference"], language=lang),
        "f1_reliability": plot_reliability_curves(f1["reliability"], language=lang),
        "f1_psi": plot_psi(f1["psi"], neutral_features=CALENDAR_FEATURES, highlight_col="highlight", language=lang),
        "regime": plot_promise_regime(data.regime, language=lang),
        "labels": plot_label_availability(f2["label_availability"], language=lang),
    }


def chart_tables(data: Data) -> dict[str, pd.DataFrame]:
    """Return the exact (language-neutral) tables plotted, keyed by CSV file name."""
    from ecom_delivery_sla_viz import node_table, route_table  # noqa: PLC0415

    results = data.tournament["classification_results"]
    summary = data.classification_summary
    candidates = summary.loc[~summary["is_baseline"], "model"].tolist()
    all_edges = route_table(data.pairs, "state", min_orders=1)
    edges = route_table(data.pairs, "state", min_orders=100)
    nodes = node_table(data.pairs, all_edges[all_edges["orders"] >= 100], data.geolocation)
    gain_columns = ["model", "n_folds", "average_precision_mean", "average_precision_std",
                    "average_precision_gain_vs_baseline", "average_precision_folds_won",
                    "average_precision_best_baseline"]
    f1, f2 = data.f1, data.f2
    return {
        "chart-route-edges.csv": edges,
        "chart-route-nodes.csv": nodes,
        "chart-tournament-gain.csv": summary.loc[~summary["is_baseline"], gain_columns],
        "chart-tournament-ap-by-fold.csv": results.loc[results["model"].isin(candidates), ["fold", "model",
                                                                                              "average_precision"]],
        "chart-f1-metrics.csv": f1["classification_metrics"],
        "chart-f1-paired-difference.csv": f1["paired_difference"],
        "chart-f1-reliability.csv": f1["reliability"],
        "chart-f1-psi.csv": f1["psi"],
        "chart-promise-regime.csv": data.regime,
        "chart-f2-label-availability.csv": f2["label_availability"],
        "chart-f2-metrics.csv": f2["classification_metrics"],
        "chart-f2-paired-difference.csv": f2["paired_difference"],
    }


# ----------------------------------------------------------------------------------------------
# Detail tables (formatted per language)
# ----------------------------------------------------------------------------------------------


def status_table(data: Data, lang: str) -> str:
    """Order status distribution of the raw orders table."""
    c = COLS[lang]
    counts = data.orders["order_status"].value_counts()
    rows = [[STATUS_PT.get(s, s) if lang == "pt" else s, fmt_int(v, lang), fmt_pct(v / counts.sum(), 2, lang)]
            for s, v in counts.items()]
    return table_html([c["status"], c["orders"], c["share"]], rows, numeric=(1, 2))


def routes_table(data: Data, lang: str) -> str:
    """Top 10 state routes by volume."""
    from ecom_delivery_sla_viz import route_table  # noqa: PLC0415

    c = COLS[lang]
    edges = route_table(data.pairs, "state", min_orders=100).head(10)
    rows = [[f"{r.origin} → {r.destination}", fmt_int(r.orders, lang),
             f"{fmt_pct(r.breach_rate, 1, lang)} ({fmt_pct(r.ci_low, 1, lang)} – {fmt_pct(r.ci_high, 1, lang)})",
             fmt_int(r.distance_median_km, lang), fmt_number(r.promised_median_days, 1, lang),
             fmt_number(r.actual_median_days, 1, lang)] for r in edges.itertuples()]
    return table_html([c["route"], c["pairs"], c["rate"], c["dist"], c["promised"], c["actual"]], rows,
                      numeric=(1, 2, 3, 4, 5))


def tournament_tables(data: Data, lang: str) -> tuple[str, str]:
    """Classification (AP) and regression (MAE) tournament summaries."""
    c = COLS[lang]
    cls_rows = []
    for r in data.classification_summary.itertuples():
        gain = DASH if r.is_baseline else fmt_pct(r.average_precision_gain_vs_baseline, 1, lang, True)
        won = DASH if r.is_baseline else f"{int(r.average_precision_folds_won)}/{int(r.n_folds)}"
        cls_rows.append([model_label(r.model, lang), fmt_number(r.average_precision_mean, 3, lang),
                         fmt_number(r.average_precision_std, 3, lang), gain, won, fmt_number(r.ece_10bins_mean, 3, lang)])
    reg_rows = []
    for r in data.regression_summary.sort_values("mae_mean").itertuples():
        gain = DASH if r.is_baseline else fmt_pct(r.mae_gain_vs_baseline, 1, lang, True)
        won = DASH if r.is_baseline else f"{int(r.mae_folds_won)}/{int(r.n_folds)}"
        reg_rows.append([model_label(r.model, lang), fmt_number(r.mae_mean, 2, lang), gain, won])
    cls = table_html([c["model"], c["ap_mean"], c["ap_sd"], c["gain"], c["won"], c["ece"]], cls_rows,
                     numeric=(1, 2, 3, 4, 5))
    reg = table_html([c["model"], c["mae_mean"], c["mae_gain"], c["won"]], reg_rows, numeric=(1, 2, 3))
    return cls, reg


def f1_tables(data: Data, lang: str) -> str:
    """F1 metrics (AP with CI, AUC, Brier, ECE, top 5%) and the paired difference."""
    c = COLS[lang]
    metrics = data.f1["classification_metrics"]
    rows = []
    for model in (CANDIDATE, REFERENCE, RULE, PREVALENCE):
        ap = _metric(metrics, model)

        def get(name: str, model=model) -> float:
            return float(_metric(metrics, model, name)["estimate"])

        rows.append([model_label(model, lang),
                     f"{fmt_number(ap['estimate'], 3, lang)} ({fmt_range(ap['ci_low'], ap['ci_high'], 3, lang)})",
                     fmt_number(get("roc_auc"), 3, lang), fmt_number(get("brier"), 3, lang),
                     fmt_number(get("ece_10bins"), 3, lang), fmt_pct(get("precision_at_5pct"), 1, lang),
                     fmt_pct(get("recall_at_5pct"), 1, lang)])
    first = table_html([c["model"], c["ap_ci"], c["auc"], c["brier"], c["ece1"], c["p5"], c["r5"]], rows,
                       numeric=(1, 2, 3, 4, 5, 6))
    p = data.f1["paired_difference"].iloc[0]
    outcome = p["outcome"] if lang == "pt" else p["outcome_en"]
    second = table_html(
        [c["model"], c["diff"], c["reading"]],
        [[f"{CANDIDATE} − {model_label(p['reference'], lang)}",
          f"{fmt_number(p['difference'], 3, lang, True)} ({fmt_range(p['ci_low'], p['ci_high'], 3, lang, True)})",
          outcome]], numeric=(1,))
    return first + second


def f2_tables(data: Data, lang: str) -> str:
    """F2 metrics by variant, paired differences and label availability."""
    c = COLS[lang]
    metrics, paired = data.f2["classification_metrics"], data.f2["paired_difference"]
    order = ["A_f1_frozen", "B_retrain", "C_recency", "D_recalibration", "E_promise_feature", REFERENCE, RULE, PREVALENCE]
    rows = []
    for model in order:
        ap = _metric(metrics, model)
        failed = pd.isna(ap["estimate"])
        auc = float("nan") if failed else _metric(metrics, model, "roc_auc")["estimate"]
        ece = float("nan") if failed else _metric(metrics, model, "ece_10bins")["estimate"]
        diffs = []
        for ref in (REFERENCE, RULE):
            part = paired[(paired["model"] == model) & (paired["reference"] == ref)]
            if len(part) and pd.notna(part.iloc[0]["difference"]):
                d = part.iloc[0]
                diffs.append(f"{fmt_number(d['difference'], 3, lang, True)} "
                             f"({fmt_range(d['ci_low'], d['ci_high'], 3, lang, True)})")
            else:
                diffs.append(DASH)
        status = c["failed"] if failed else (c["baseline"] if model.startswith("baseline_") else c["evaluated"])
        rows.append([model_label(model, lang),
                     DASH if failed else f"{fmt_number(ap['estimate'], 3, lang)} "
                                         f"({fmt_range(ap['ci_low'], ap['ci_high'], 3, lang)})",
                     fmt_number(auc, 3, lang), fmt_number(ece, 3, lang), diffs[0], diffs[1], status])
    first = table_html([c["variant"], c["ap_ci"], c["auc"], c["ece1"], c["vs_log"], c["vs_rule"], c["status_v"]],
                       rows, numeric=(1, 2, 3, 4, 5))
    labels = data.f2["label_availability"]
    lab_rows = [[fmt_month(r.purchase_month, lang), fmt_int(r.orders, lang), fmt_pct(r.share_known, 1, lang),
                 fmt_pct(r.prevalence_known, 2, lang), fmt_int(r.unknown_labels, lang),
                 fmt_pct(r.prevalence_unknown, 2, lang)] for r in labels.itertuples()]
    second = table_html([c["month"], c["orders"], c["known"], c["prev_known"], c["unknown"], c["prev_unknown"]],
                        lab_rows, numeric=(1, 2, 3, 4, 5))
    return first + second


def evidence_html(data: Data, lang: str) -> str:
    """Evidence table of the three stages (``evidence_table`` of the viz package)."""
    from ecom_delivery_sla_viz.report import evidence_table  # noqa: PLC0415

    c = COLS[lang]
    table = evidence_table(data.classification_summary, data.f1["classification_metrics"],
                           data.f1["paired_difference"], data.f2["classification_metrics"],
                           data.f2["paired_difference"], language=lang)
    rows = []
    for index, r in table.iterrows():
        if pd.isna(r["estimate"]):
            estimate = DASH
        elif index == 2:  # the development gain row is a proportion
            estimate = fmt_pct(r["estimate"], 1, lang, True)
        else:
            estimate = fmt_number(r["estimate"], 3, lang, r["item"].lower().startswith(("diferença", "paired")))
        interval = DASH if pd.isna(r["ci_low"]) else fmt_range(r["ci_low"], r["ci_high"], 3, lang)
        rows.append([r["stage"], r["item"], estimate, interval, r["note"]])
    return table_html([c["stage"], c["item"], c["estimate"], c["ci"], c["note"]], rows, numeric=(2,))


def ledger_html(ledger: list[dict], lang: str) -> str:
    """Render the test ledger as a table."""
    c = COLS[lang]
    rows = []
    for entry in ledger:
        rows.append([entry["version"], entry["timestamp"], c["yes"] if entry["post_hoc"] else c["no"],
                     fmt_int(entry["train_orders"], lang), fmt_int(entry["test_orders"], lang),
                     fmt_int(entry["n_boot"], lang), str(entry["seed"]), entry["fingerprint"][:16] + "…"])
    return table_html([c["version"], c["timestamp"], c["post_hoc"], c["train"], c["test"], "n_boot", "seed",
                       "fingerprint"], rows, numeric=(3, 4, 5, 6))


# ----------------------------------------------------------------------------------------------
# Page assembly
# ----------------------------------------------------------------------------------------------


def section(section_id: str, number: str | None, heading: str, body: str, css: str = "") -> str:
    """Return one white report section."""
    label = f'<span class="section-no">{number}</span>' if number else ""
    cls = f' class="{css}"' if css else ""
    return f'<section id="{section_id}"{cls}>{label}<h2>{esc(heading)}</h2>{body}</section>'


def render_page(lang: str, raw: dict, data: Data, figs: dict, assets: dict) -> tuple[str, set[str]]:
    """Render one language page.

    Parameters
    ----------
    lang : {"pt", "en"}
        Page language.
    raw : dict
        Output of :func:`compute_raw`.
    data : Data
        Loaded data (for tables).
    figs : dict
        Output of :func:`build_charts` for this language.
    assets : dict
        ``{"readme": path, "notebook": path, "report_html": path or None, "charts": [file names]}``
        relative to ``site/``.

    Returns
    -------
    tuple
        ``(html, cited_keys)``.
    """
    t = TEXT[lang]
    facts = make_facts(raw, lang)

    def p(key: str, css: str = "") -> str:
        cls = f' class="{css}"' if css else ""
        return f"<p{cls}>{render(t[key], facts)}</p>"

    def chart(name: str) -> str:
        return f'<div class="chart">{chart_div(figs[name], f"chart-{lang}-{name.replace('_', '-')}")}</div>'

    cls_table, reg_table = tournament_tables(data, lang)
    sections = [
        section("dados", None, t["h_dados"],
                p("dados_1") + p("dados_2") + p("dados_3")
                + f'<p><a href="{KAGGLE_URL}">{esc(t["dados_link"])}</a></p>'
                + details_html(t["dados_details"], status_table(data, lang))),
        section("rotas", "01", t["h_rotas"],
                p("rotas_1") + chart("network") + details_html(t["rotas_details"], routes_table(data, lang))
                + p("rotas_insight", "insight")),
        section("torneio", "02", t["h_torneio"],
                p("torneio_1") + p("torneio_2") + chart("gain") + chart("folds")
                + details_html(t["torneio_details_cls"], cls_table) + details_html(t["torneio_details_reg"], reg_table)
                + p("torneio_insight", "insight")),
        section("teste-oficial", "03", t["h_teste"],
                p("teste_1") + p("teste_2") + chart("f1_intervals") + chart("f1_reliability") + chart("f1_psi")
                + details_html(t["teste_details"], f1_tables(data, lang)) + p("teste_insight", "insight")),
        section("atualizacao", "04", t["h_f2"],
                p("f2_1") + p("f2_2") + chart("regime") + chart("labels")
                + details_html(t["f2_details"], f2_tables(data, lang)) + p("f2_insight", "insight")),
        section("decisao", "05", t["h_decisao"],
                p("decisao_1") + evidence_html(data, lang)
                + p("decisao_2") + p("decisao_3") + p("decisao_insight", "insight")),
    ]
    downloads = [f'<a href="{assets["readme"]}" download>{esc(t["dl_readme"])}</a>',
                 f'<a href="{assets["notebook"]}" download>{esc(t["dl_notebook"])}</a>']
    if assets.get("report_html"):
        downloads.append(f'<a href="{assets["report_html"]}" download>{esc(t["dl_html"])}</a>')
    sections.append(section(
        "reproducao", "06", t["h_repro"],
        p("repro_1") + f"<pre><code>{esc(t['repro_code'])}</code></pre>"
        + f'<div class="downloads">{"".join(downloads)}</div>' + p("repro_insight", "insight")))

    # the sources section comes last so that every citation used above is known
    nav = "".join(f'<a href="#{sid}">{esc(t["nav"][sid])}</a>' for sid in SECTION_IDS)
    references = select_references(assets["references"], sorted(facts.cited, key=list(CITATIONS).index))
    chart_links = " · ".join(f'<a href="assets/{name}">{esc(name)}</a>' for name in assets["charts"])
    sources = section(
        "fontes", None, t["h_fontes"],
        f"<p>{render(t['fontes_1'], facts)}</p>"
        + details_html(t["fontes_details"],
                       f'<p>{esc(t["fontes_charts"])}:</p><p class="file-links">{chart_links}</p>'
                       f'<p><a href="assets/build-manifest.json">{esc(t["fontes_manifest"])} (build-manifest.json)</a></p>'
                       f'<p>{esc(t["fontes_ledger"])}:</p>{ledger_html(data.ledger, lang)}')
        + f"<p>{render(t['fontes_dataset'], facts)}</p>"
        + f'<div class="refs"><h3>{esc(t["refs"])}</h3>{"".join(f"<p>{reference_html(r)}</p>" for r in references)}</div>',
        css="sources")
    stats = (
        f'<div><strong>{facts["n_sup"]}</strong><span>{esc(t["stat1"])}</span></div>'
        f'<div><strong class="stat-text">{facts["gain_cat"]} · {facts["won_cat"]}/{facts["n_folds"]}</strong>'
        f'<span>{esc(t["stat2"])}</span></div>'
        f'<div><strong class="stat-text">{esc(facts["f1_outcome_cap"])} · ΔAP {facts["delta"]}</strong>'
        f'<span>{esc(t["stat3"])}</span></div>'
    )
    lead = render(t["lead"], facts)
    other = t["other_lang"]
    document = f"""<!doctype html>
<html lang="{t['html_lang']}">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Olist SLA — {esc(t['title'])}</title>
<meta name="description" content="{esc(lead)}">
<link rel="alternate" hreflang="pt-BR" href="index.html"><link rel="alternate" hreflang="en" href="en.html">
<link rel="stylesheet" href="assets/report.css"><script src="assets/plotly.min.js"></script><script src="assets/geo-topojson.js"></script></head>
<body id="top"><a class="skip" href="#dados">{esc(t['skip'])}</a>
<header><div class="brand">{esc(t['brand'])} <span>{esc(t['brand_tag'])}</span></div>\
<a class="language" href="{t['other_file']}" hreflang="{other}" lang="{other}">{esc(t['other_label'])} ↗</a></header>
<main><div class="hero"><p class="eyebrow">{esc(t['eyebrow'])}</p><h1>{esc(t['title'])}</h1>
<p class="lead">{lead}</p><p class="scope">{render(t['scope'], facts)}</p>
<div class="stats">{stats}</div></div>
<nav aria-label="{esc(t['nav_label'])}">{nav}</nav>
<noscript><p>{esc(t['noscript'])}</p></noscript>
{''.join(sections)}
{sources}
</main><footer><span>{esc(t['footer'])}</span><a href="#top">{esc(t['back'])} ↑</a></footer></body></html>
"""
    return document, set(facts.cited)


# ----------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with LF newlines (deterministic output)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def source_paths(root: Path) -> list[str]:
    """Relative paths whose SHA-256 is recorded in the manifest."""
    paths = [f"outputs/runs/{name}/{version}" for name, version in RUNS]
    paths += ["outputs/runs/final_evaluation/LEDGER.jsonl"]
    paths += [f"data/raw/olist/csv/{file}" for file in RAW_FILES.values()]
    paths += ["data/raw/olist/README.md", "docs/REGISTRO-DE-DECISOES.md", "README.md"]
    if (root / README_FILES["pt"]).is_file():
        paths.append(README_FILES["pt"])
    paths += list(NOTEBOOKS.values())
    paths += [p for p in REPORT_HTML.values() if (root / p).is_file()]
    return paths


def build_manifest(root: Path, output: Path, raw: dict, ledger_unchanged: bool) -> dict:
    """Assemble the build manifest (hashes of sources and of generated assets, no timestamps)."""
    import plotly  # noqa: PLC0415

    assets = output / "assets"
    generated = {
        p.relative_to(output).as_posix(): sha256_file(p)
        for p in sorted(assets.rglob("*")) if p.is_file() and p.name != "build-manifest.json"
    }
    return {
        "sources": hash_sources(root, source_paths(root)),
        "generated": generated,
        "builder": {"script": "scripts/build_site.py", "sha256": sha256_file(Path(__file__)),
                    "plotly_version": plotly.__version__},
        "languages": ["pt-BR", "en"],
        "training_performed": False,
        "test_reopened": not ledger_unchanged,
        "production_eligible": raw["prod_eligible"],
        "ledger": {"rows": raw["ledger_rows"], "path": "outputs/runs/final_evaluation/LEDGER.jsonl"},
        "dataset_provenance": {
            "repository": KAGGLE_URL,
            "citation": "Olist (2018), Brazilian E-Commerce Public Dataset by Olist, Kaggle",
            "local_readme": "data/raw/olist/README.md",
        },
    }


def build_site(root: Path = ROOT, output: Path | None = None) -> Path:
    """Build ``index.html``, ``en.html`` and ``assets/`` under ``output`` (default ``root/site``).

    Parameters
    ----------
    root : pathlib.Path, default project root
        Project directory with ``outputs/``, ``data/``, ``docs/`` and ``notebooks/``.
    output : pathlib.Path, optional
        Destination directory.

    Returns
    -------
    pathlib.Path
        The output directory.
    """
    from plotly.offline import get_plotlyjs  # noqa: PLC0415

    output = (output or root / "site").resolve()
    ledger_before = read_ledger(root) if inputs_available(root) else []
    data = load_data(root)
    raw = compute_raw(data)
    assets_dir = output / "assets"
    write_text(assets_dir / "report.css", CSS)
    write_text(assets_dir / "plotly.min.js", get_plotlyjs())
    for name in TOPOJSON_FILES:
        shutil.copyfile(TOPOJSON_SOURCE / name, assets_dir / name)
    write_text(assets_dir / "geo-topojson.js", geo_topojson_script())

    chart_files = []
    for name, table in chart_tables(data).items():
        write_text(assets_dir / name, table.to_csv(index=False, lineterminator="\n"))
        chart_files.append(name)
    for lang in ("pt", "en"):
        from ecom_delivery_sla_viz.report import evidence_table  # noqa: PLC0415

        evidence = evidence_table(data.classification_summary, data.f1["classification_metrics"],
                                  data.f1["paired_difference"], data.f2["classification_metrics"],
                                  data.f2["paired_difference"], language=lang)
        name = f"chart-evidence-{lang}.csv"
        write_text(assets_dir / name, evidence.to_csv(index=False, lineterminator="\n"))
        chart_files.append(name)

    readme_assets = {}
    for lang, name in README_FILES.items():
        source = root / name if (root / name).is_file() else root / "README.md"
        shutil.copyfile(source, assets_dir / source.name)
        readme_assets[lang] = f"assets/{source.name}"
    cited_all: set[str] = set()
    for lang in ("pt", "en"):
        notebook = root / NOTEBOOKS[lang]
        shutil.copyfile(notebook, assets_dir / notebook.name)
        report_html = None
        if (root / REPORT_HTML[lang]).is_file():
            report_html = f"assets/{Path(REPORT_HTML[lang]).name}"
            shutil.copyfile(root / REPORT_HTML[lang], output / report_html)
        assets = {"readme": readme_assets[lang], "notebook": f"assets/{notebook.name}", "report_html": report_html,
                  "charts": sorted(chart_files), "references": parse_references(notebook)}
        document, cited = render_page(lang, raw, data, build_charts(data, lang), assets)
        write_text(output / ("index.html" if lang == "pt" else "en.html"), document)
        cited_all |= cited

    ledger_after = read_ledger(root)
    manifest = build_manifest(root, output, raw, ledger_unchanged=ledger_before == ledger_after)
    write_text(assets_dir / "build-manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and build the site."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=ROOT / "site", help="output directory (default: site/)")
    parser.add_argument("--root", type=Path, default=ROOT, help="project root (default: this repository)")
    arguments = parser.parse_args(argv)
    try:
        output = build_site(arguments.root, arguments.output)
    except (MissingInputs, FileNotFoundError) as error:
        print(f"ERROR: stored results or data are missing: {error}", file=sys.stderr)
        return 1
    except NarrativeMismatch as error:
        print(f"ERROR: stored results contradict the narrative: {error}", file=sys.stderr)
        return 2
    print(f"site written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
