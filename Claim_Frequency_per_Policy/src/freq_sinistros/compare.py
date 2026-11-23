"""Comparacao em validacao: taxa global, Poisson e NB2 (etapa 3).

O pre-processamento e o ajuste usam so as linhas de ajuste de cada
comparacao; a avaliacao usa as linhas de validacao com contagens e
exposicoes originais. O teste nunca e carregado.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from freq_sinistros import (
    __version__,
    audit,
    metrics,
    models,
    prep,
    profiles,
    segments,
)
from freq_sinistros.audit import _peak_mb, safe_ratio
from freq_sinistros.data import load_partitions
from freq_sinistros.protocol import (
    FIT_LIMITS,
    SAMPLE_LADDER,
    amendment_stage3,
    nested_train_sample,
)
from freq_sinistros.stage3 import OUTPUTS_DIR, write_json, write_table

CANDIDATES = ("global", "poisson", "nb2")
PAIRS = [("poisson", "global"), ("nb2", "poisson"), ("nb2", "global")]
FAMILIES = segments.FAMILIES
MAX_RAM_MIB = FIT_LIMITS["max_peak_ram_gib"] * 1024


# ------------------------------------------------------------ utilitarios
def ids_hash(ids: pd.Series) -> str:
    """SHA-256 dos ``IDpol`` ordenados (um por linha, UTF-8)."""
    text = "\n".join(str(int(i)) for i in sorted(ids.tolist())) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_sample(train: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """Linhas da amostra aninhada aprovada (``None`` = treino inteiro)."""
    if n is None or n >= len(train):
        return train
    chosen = set(nested_train_sample(train["IDpol"], n))
    return train[train["IDpol"].isin(chosen)]


def fit_candidates(sample: pd.DataFrame, pre: prep.Preprocessor,
                   context: dict) -> tuple[dict, list[dict], np.ndarray]:
    """Ajuste os tres candidatos na mesma amostra, medindo custo.

    Args:
        sample: linhas de ajuste.
        pre: pre-processador **ja ajustado** nessas linhas.
        context: campos extras para o registro de cada ajuste.

    Returns:
        Ajustes por nome, registros de custo/convergencia e a matriz de
        desenho de ajuste.
    """
    matrix, _ = pre.transform(sample)
    y = sample["ClaimNb"].to_numpy(dtype="float64")
    e = sample["Exposure"].to_numpy(dtype="float64")
    fits, records = {}, []
    for name in CANDIDATES:
        with models.ResourceMonitor() as monitor:
            if name == "global":
                fit = models.fit_global(y, e)
            elif name == "poisson":
                fit = models.fit_poisson(matrix, y, e, pre.columns)
            else:
                fit = models.fit_nb2(matrix, y, e, pre.columns,
                                     start=fits["poisson"])
        fits[name] = fit
        within = (monitor.wall_s <= FIT_LIMITS["max_fit_wall_s"]
                  and monitor.peak_rss_mib <= MAX_RAM_MIB)
        records.append({
            **context, "candidate": name, "n_fit": int(len(y)),
            "converged": bool(fit.converged), "iterations": fit.iterations,
            "wall_s": round(monitor.wall_s, 3),
            "peak_rss_mib": round(monitor.peak_rss_mib, 1),
            "delta_rss_mib": round(monitor.delta_rss_mib, 1),
            "within_limits": bool(within),
            "loglik_fit": fit.loglik, "alpha": fit.alpha,
            "alpha_at_poisson_limit": fit.extras.get(
                "alpha_at_poisson_limit"),
            "max_abs_score_per_row": fit.extras.get(
                "max_abs_score_beta_per_row",
                fit.extras.get("max_abs_score_per_row")),
            "score_alpha_per_row": fit.extras.get("score_alpha_per_row"),
            "warnings": "; ".join(fit.warnings),
        })
    return fits, records, matrix


def predict_all(fits: dict, pre: prep.Preprocessor,
                evaluation: pd.DataFrame) -> tuple[dict, dict]:
    """Previsoes ``mu`` de cada candidato nas linhas de avaliacao."""
    matrix, report = pre.transform(evaluation)
    e = evaluation["Exposure"].to_numpy(dtype="float64")
    return {name: fit.predict(matrix, e) for name, fit in fits.items()}, \
        report


# ------------------------------------------------------- estatistica do IC
class EvalSet:
    """Linhas de avaliacao e quantidades por linha reutilizadas no bootstrap.

    Args:
        frame: linhas de avaliacao (originais).
        mu: ``{modelo: medias previstas}``.
        alphas: ``{modelo: alpha}`` para modelos NB2 (``None`` = Poisson).
        pairs: pares ``(b, a)`` para diferencas pareadas ``b - a``.
    """

    def __init__(self, frame: pd.DataFrame, mu: dict, alphas: dict,
                 pairs: list[tuple[str, str]]) -> None:
        """Calcule contribuicoes por linha e codigos de segmento."""
        self.frame = frame
        self.y = frame["ClaimNb"].to_numpy(dtype="float64")
        self.e = frame["Exposure"].to_numpy(dtype="float64")
        self.n = len(self.y)
        self.names = list(mu)
        self.mu = {m: np.asarray(v, dtype="float64") for m, v in mu.items()}
        self.pairs = [p for p in pairs if p[0] in mu and p[1] in mu]
        self.dev, self.ll, self.p0 = {}, {}, {}
        for m in self.names:
            self.dev[m] = metrics.poisson_deviance_contributions(
                self.y, self.mu[m])
            alpha = alphas.get(m)
            self.ll[m] = (metrics.poisson_logpmf(self.y, self.mu[m])
                          if alpha is None else
                          metrics.nb2_logpmf(self.y, self.mu[m], alpha))
            self.p0[m] = metrics.zero_prob(self.mu[m], alpha)
        seg = segments.segment_columns(frame)
        self.codes, self.labels, self.keep = {}, {}, {}
        for fam in FAMILIES:
            codes, uniques = pd.factorize(seg[fam], sort=True)
            self.codes[fam], self.labels[fam] = codes, list(uniques)
            rows = np.bincount(codes, minlength=len(uniques))
            expo = np.bincount(codes, weights=self.e,
                               minlength=len(uniques))
            self.keep[fam] = [not segments.low_support(r, x)
                              for r, x in zip(rows, expo, strict=True)]
        self.zero = (self.y == 0).astype("float64")

    def statistic(self, idx: np.ndarray) -> dict[str, float]:
        """Indicadores da amostra ``idx`` (uma replica ou a amostra toda)."""
        y, e, zero = self.y[idx], self.e[idx], self.zero[idx]
        out, e_sum = {}, e.sum()
        mus = {m: self.mu[m][idx] for m in self.names}
        for m in self.names:
            out[f"dev|{m}"] = self.dev[m][idx].mean()
            out[f"ll|{m}"] = self.ll[m][idx].mean()
            out[f"ratediff|{m}|overall"] = 100 * (
                mus[m].sum() - y.sum()) / e_sum
            out[f"zerosdiff|{m}|overall"] = self.p0[m][idx].sum() - zero.sum()
        for b, a in self.pairs:
            out[f"d_dev|{b}-{a}"] = out[f"dev|{b}"] - out[f"dev|{a}"]
            out[f"d_ll|{b}-{a}"] = out[f"ll|{b}"] - out[f"ll|{a}"]
        for fam in FAMILIES:
            codes = self.codes[fam][idx]
            k = len(self.labels[fam])
            seg_e = np.bincount(codes, weights=e, minlength=k)
            seg_y = np.bincount(codes, weights=y, minlength=k)
            seg_z = np.bincount(codes, weights=zero, minlength=k)
            seg_mu = {m: np.bincount(codes, weights=mus[m], minlength=k)
                      for m in self.names}
            seg_p0 = ({m: np.bincount(codes, weights=self.p0[m][idx],
                                      minlength=k) for m in self.names}
                      if fam == "exposure_band" else {})
            for j, label in enumerate(self.labels[fam]):
                if not self.keep[fam][j]:
                    continue
                for m in self.names:
                    out[f"ratediff|{m}|{fam}|{label}"] = safe_ratio(
                        100 * (seg_mu[m][j] - seg_y[j]), seg_e[j])
                    if fam == "exposure_band":
                        out[f"zerosdiff|{m}|{fam}|{label}"] = (
                            seg_p0[m][j] - seg_z[j])
        return out


def interval_table(ev: EvalSet, degree, population: str,
                   clusters=None) -> pd.DataFrame:
    """Bootstrap pareado (1.000 replicas) dos indicadores de ``ev``."""
    table = metrics.bootstrap_intervals(
        ev.statistic, ev.n, clusters=clusters)
    table.insert(0, "population", population)
    table.insert(0, "degree", degree)
    return table


def metric_rows(ev: EvalSet, degree, population: str,
                mask: np.ndarray | None = None,
                label: str = "all_rows") -> pd.DataFrame:
    """Metricas pontuais por candidato (opcionalmente em subconjunto)."""
    m_ = np.ones(ev.n, dtype=bool) if mask is None else mask
    y, e = ev.y[m_], ev.e[m_]
    rows = []
    for m in ev.names:
        mu = ev.mu[m][m_]
        obs_rate = 100 * y.sum() / e.sum()
        pred_rate = 100 * mu.sum() / e.sum()
        rows.append({
            "degree": degree, "population": population, "subset": label,
            "candidate": m, "rows": int(m_.sum()),
            "exposure": float(e.sum()), "observed_claims": float(y.sum()),
            "predicted_claims": float(mu.sum()),
            "observed_rate_per_100": obs_rate,
            "predicted_rate_per_100": pred_rate,
            "rate_diff_pred_minus_obs": pred_rate - obs_rate,
            "abs_rate_error": abs(pred_rate - obs_rate),
            "mean_poisson_deviance": float(ev.dev[m][m_].mean()),
            "mean_loglik": float(ev.ll[m][m_].mean()),
            "zeros_observed": float((y == 0).sum()),
            "zeros_predicted": float(ev.p0[m][m_].sum()),
        })
    return pd.DataFrame(rows)


def calibration_rows(ev: EvalSet, degree, population: str,
                     intervals: pd.DataFrame) -> pd.DataFrame:
    """Calibracao por segmento e candidato, com IC quando ha suporte."""
    ci = intervals.set_index("indicator")
    seg = segments.segment_columns(ev.frame)
    rows = []
    for m in ev.names:
        for fam in FAMILIES:
            table = metrics.calibration_table(seg[fam], ev.y, ev.mu[m], ev.e)
            for r in table.to_dict(orient="records"):
                key = f"ratediff|{m}|{fam}|{r['segment']}"
                has = key in ci.index and bool(
                    ci.loc[key, "interval_reported"])
                rows.append({
                    "degree": degree, "population": population,
                    "candidate": m, "family": fam, **r,
                    "ci_low": ci.loc[key, "ci_low"] if has else np.nan,
                    "ci_high": ci.loc[key, "ci_high"] if has else np.nan})
    return pd.DataFrame(rows)


def zero_rows(ev: EvalSet, degree, population: str,
              intervals: pd.DataFrame) -> pd.DataFrame:
    """Resumo preliminar dos zeros por faixa de exposicao (sem decisao)."""
    ci = intervals.set_index("indicator")
    band = segments.segment_columns(ev.frame)["exposure_band"].to_numpy()
    rows = []
    for m in ev.names:
        for label in ["overall"] + list(ev.labels["exposure_band"]):
            mask = np.ones(ev.n, bool) if label == "overall" else (
                band == label)
            key = (f"zerosdiff|{m}|overall" if label == "overall" else
                   f"zerosdiff|{m}|exposure_band|{label}")
            has = key in ci.index and bool(ci.loc[key, "interval_reported"])
            obs = float(ev.zero[mask].sum())
            pred = float(ev.p0[m][mask].sum())
            rows.append({
                "degree": degree, "population": population, "candidate": m,
                "band": label, "rows": int(mask.sum()),
                "zeros_observed": obs, "zeros_predicted": pred,
                "diff_pred_minus_obs": pred - obs,
                "ci_low": ci.loc[key, "ci_low"] if has else np.nan,
                "ci_high": ci.loc[key, "ci_high"] if has else np.nan})
    return pd.DataFrame(rows)


def reconcile(ev: EvalSet, degree, population: str) -> pd.DataFrame:
    """Reconcilie registros, exposicao, observados e previstos por familia."""
    seg = segments.segment_columns(ev.frame)
    rows = []
    for m in ev.names:
        totals = {"rows": float(ev.n), "exposure": float(ev.e.sum()),
                  "observed_claims": float(ev.y.sum()),
                  "predicted_claims": float(ev.mu[m].sum())}
        for fam in FAMILIES:
            data = pd.DataFrame({"s": seg[fam].to_numpy(), "e": ev.e,
                                 "y": ev.y, "mu": ev.mu[m]})
            sums = {"rows": float(len(data)), "exposure": data["e"].sum(),
                    "observed_claims": data["y"].sum(),
                    "predicted_claims": data["mu"].sum()}
            # soma dos grupos (nao do total) para provar a reconciliacao
            grouped = data.groupby("s").agg(
                rows=("e", "size"), exposure=("e", "sum"),
                observed_claims=("y", "sum"), predicted_claims=("mu", "sum"))
            for k, total in totals.items():
                value = float(grouped[k].sum())
                rows.append({"degree": degree, "population": population,
                             "candidate": m, "family": fam, "indicator": k,
                             "segments_sum": value, "total": total,
                             "ok": bool(np.isclose(value, total, rtol=0,
                                                   atol=1e-6)),
                             "direct_sum_check": float(sums[k])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- decisoes
def evaluate_rules(metrics_df: pd.DataFrame,
                   intervals: pd.DataFrame) -> dict:
    """Aplique as regras de escolha do parecer a um degrau (provisorio).

    Args:
        metrics_df: metricas pontuais do degrau (um registro por modelo).
        intervals: intervalos do mesmo degrau.

    Returns:
        Condicoes individuais e o modelo preferido pelas regras.
    """
    m = metrics_df.set_index("candidate")
    ci = intervals.set_index("indicator")

    def interval(key):
        return float(ci.loc[key, "ci_low"]), float(ci.loc[key, "ci_high"])

    d_pg = interval("d_dev|poisson-global")
    d_np = interval("d_ll|nb2-poisson")
    d_ng = interval("d_dev|nb2-global")
    poisson_wins = bool(
        d_pg[1] < 0
        and m.loc["poisson", "abs_rate_error"]
        <= m.loc["global", "abs_rate_error"])
    nb2_vs_poisson = bool(
        d_np[0] > 0
        and m.loc["nb2", "mean_poisson_deviance"]
        <= m.loc["poisson", "mean_poisson_deviance"]
        and m.loc["nb2", "abs_rate_error"]
        <= m.loc["poisson", "abs_rate_error"])
    reference = "poisson" if poisson_wins else "global"
    nb2_vs_global_ok = bool(d_ng[1] < 0)
    nb2_wins = nb2_vs_poisson and (poisson_wins or nb2_vs_global_ok)
    return {
        "poisson_over_global": poisson_wins,
        "d_dev_poisson_minus_global_ci": d_pg,
        "nb2_over_poisson_conditions": nb2_vs_poisson,
        "d_ll_nb2_minus_poisson_ci": d_np,
        "d_dev_nb2_minus_global_ci": d_ng,
        "reference_before_nb2": reference,
        "nb2_selected": nb2_wins,
        "preferred_by_rules": "nb2" if nb2_wins else reference,
        "note": "resultado provisorio; a decisao de modelo pertence a "
                "etapa 4",
    }


def point_signs(metrics_df: pd.DataFrame, intervals: pd.DataFrame) -> dict:
    """Sinais das estimativas pontuais usadas na estabilidade entre degraus."""
    ci = intervals.set_index("indicator")["estimate"]
    return {"d_dev_poisson_minus_global": float(ci["d_dev|poisson-global"]),
            "d_ll_nb2_minus_poisson": float(ci["d_ll|nb2-poisson"]),
            "d_dev_nb2_minus_global": float(ci["d_dev|nb2-global"])}


# ------------------------------------------------------------- execucao
def _profile_codes(frame: pd.DataFrame) -> np.ndarray:
    """Codigos inteiros da chave de perfil de cada linha."""
    return pd.factorize(profiles.profile_keys(frame))[0]


def _evaluate_degree(train, val, n, degree_label, fits_log, outputs):
    """Ajuste, preveja e avalie um degrau; devolva tabelas e objetos."""
    sample = select_sample(train, n)
    pre = prep.Preprocessor().fit(sample)
    context = {"degree": degree_label, "sample_rows": int(len(sample)),
               "sample_ids_sha256": ids_hash(sample["IDpol"]),
               "context": "main"}
    fits, records, matrix = fit_candidates(sample, pre, context)
    fits_log.extend(records)
    checks = prep.design_checks(matrix, pre.columns)
    y_s = sample["ClaimNb"].to_numpy(dtype="float64")
    e_s = sample["Exposure"].to_numpy(dtype="float64")
    mu_fit = fits["poisson"].predict(matrix, e_s)
    dispersion = models.dispersion_diagnostics(y_s, mu_fit,
                                               matrix.shape[1])
    ok = all(r["converged"] and r["within_limits"] for r in records)
    result = {"degree": degree_label, "n": n, "sample": sample,
              "pre": pre, "fits": fits, "records": records,
              "checks": checks, "dispersion": dispersion, "eligible": ok,
              "matrix_columns": matrix.shape[1]}
    del matrix
    write_json(outputs / f"preprocessing_n{degree_label}.json",
               {**pre.summary(), "design_checks": checks})
    coef = pd.DataFrame({"column": pre.columns})
    for name in ("poisson", "nb2"):
        coef[f"{name}_coef"] = fits[name].beta
        coef[f"{name}_naive_se"] = fits[name].extras["std_errors"]
    write_table(outputs / f"coefficients_n{degree_label}.csv", coef)
    return result


def run_comparison(root: Path) -> int:
    """Execute a comparacao completa (principal e sensibilidades).

    Efeitos colaterais: le o ARFF local, grava ``outputs/etapa_03/*``.

    Args:
        root: raiz do projeto.

    Returns:
        0 em sucesso; 1 se o primeiro degrau nao for elegivel.
    """
    root = Path(root)
    outputs = root / OUTPUTS_DIR
    outputs.mkdir(parents=True, exist_ok=True)
    stages, t_all = [], time.perf_counter()

    def mark(name, t0):
        stages.append({"stage": name,
                       "wall_s": round(time.perf_counter() - t0, 3),
                       "peak_rss_mib_cumulative": round(_peak_mb(), 1)})

    eda_path = outputs / "eda_summary.json"
    eda_summary = (json.loads(eda_path.read_text("utf-8"))
                   if eda_path.exists() else None)
    amendment = amendment_stage3(eda_summary)
    amendment["eda_summary_sha256"] = (
        hashlib.sha256(eda_path.read_bytes()).hexdigest()
        if eda_path.exists() else None)
    write_json(outputs / "protocol_amendment.json", amendment)

    t0 = time.perf_counter()
    parts, provenance = load_partitions(root, keep=("train", "validation"))
    train, val = parts["train"], parts["validation"]
    overview = json.loads((root / "outputs" / "etapa_02"
                           / "overview.json").read_text("utf-8"))
    stage2_ok = (len(val) == overview["validation"]["rows"]
                 and np.isclose(val["Exposure"].sum(),
                                overview["validation"]["exposure"]))
    mark("load_verify", t0)

    ladder = [(n, "train_full" if n is None else str(n))
              for n in SAMPLE_LADDER]
    fits_log, metric_tables, interval_tables = [], [], []
    calib_tables, zero_tables, recon_tables = [], [], []
    degrees, dispersion_rows, rules, signs = [], [], {}, {}
    last_ok = None
    for n, label in ladder:
        t0 = time.perf_counter()
        res = _evaluate_degree(train, val, n, label, fits_log, outputs)
        dispersion_rows.append({"degree": label, **res["dispersion"]})
        degrees.append({"degree": label, "n_fit": int(len(res["sample"])),
                        "eligible": res["eligible"],
                        "sample_ids_sha256": ids_hash(
                            res["sample"]["IDpol"])})
        if not res["eligible"]:
            mark(f"degree_{label}_failed", t0)
            break
        mu, report = predict_all(res["fits"], res["pre"], val)
        alphas = {"nb2": res["fits"]["nb2"].alpha}
        ev = EvalSet(val, mu, alphas, PAIRS)
        iv = interval_table(ev, label, "validation")
        metric_tables.append(metric_rows(ev, label, "validation"))
        interval_tables.append(iv)
        calib_tables.append(calibration_rows(ev, label, "validation", iv))
        zero_tables.append(zero_rows(ev, label, "validation", iv))
        recon_tables.append(reconcile(ev, label, "validation"))
        rules[label] = evaluate_rules(metric_tables[-1], iv)
        signs[label] = point_signs(metric_tables[-1], iv)
        degrees[-1]["validation_unknown_levels"] = report["unknown_levels"]
        last_ok = {"label": label, "n": n, **res, "ev": ev, "mu": mu,
                   "alphas": alphas}
        mark(f"degree_{label}", t0)
    summary_status = 1
    if last_ok is not None:
        summary_status = 0
        extra = _sensitivities(root, train, val, last_ok, fits_log,
                               outputs, mark)
        metric_tables += extra["metrics"]
        interval_tables += extra["intervals"]
        calib_tables += extra["calibration"]
        zero_tables += extra["zeros"]
        recon_tables += extra["reconciliation"]
    t0 = time.perf_counter()
    write_table(outputs / "fits.csv", pd.DataFrame(fits_log))
    write_table(outputs / "validation_metrics.csv",
                pd.concat(metric_tables, ignore_index=True)
                if metric_tables else pd.DataFrame())
    write_table(outputs / "intervals.csv",
                pd.concat(interval_tables, ignore_index=True)
                if interval_tables else pd.DataFrame())
    write_table(outputs / "calibration_by_segment.csv",
                pd.concat(calib_tables, ignore_index=True)
                if calib_tables else pd.DataFrame())
    write_table(outputs / "zeros_preliminary.csv",
                pd.concat(zero_tables, ignore_index=True)
                if zero_tables else pd.DataFrame())
    write_table(outputs / "reconciliation.csv",
                pd.concat(recon_tables, ignore_index=True)
                if recon_tables else pd.DataFrame())
    write_table(outputs / "dispersion_diagnostics.csv",
                pd.DataFrame(dispersion_rows))
    viable = [d["degree"] for d in degrees if d["eligible"]]
    stability = _stability(signs, viable)
    config = {
        "stage": 3, "package_version": __version__,
        "production_eligible": False,
        "provenance": provenance,
        "stage2_validation_totals_match": bool(stage2_ok),
        "ladder": degrees, "viable_degrees": viable,
        "largest_viable_degree": viable[-1] if viable else None,
        "rules_by_degree": rules, "stability": stability,
        "limits": FIT_LIMITS,
        "bootstrap": {"replicas": metrics.BOOTSTRAP_REPLICAS,
                      "seed": metrics.BOOTSTRAP_SEED},
        "validation": {"rows": int(len(val)),
                       "exposure": float(val["Exposure"].sum()),
                       "observed_claims": float(val["ClaimNb"].sum()),
                       "observed_zeros": int((val["ClaimNb"] == 0).sum())},
        "amendment_sha256": hashlib.sha256(
            (outputs / "protocol_amendment.json").read_bytes()).hexdigest(),
    }
    write_json(outputs / "comparison_config.json", config)
    mark("write_outputs", t0)
    import psutil

    write_json(outputs / "cost.json", {
        "stages": stages,
        "total_wall_s": round(time.perf_counter() - t_all, 3),
        "peak_rss_mib": round(_peak_mb(), 1),
        "environment": {"python": sys.version,
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "psutil": psutil.__version__,
                        "cpu_logical": psutil.cpu_count(),
                        "ram_total_gib": round(
                            psutil.virtual_memory().total / 2**30, 1)},
        "note": "peak_wset do processo; CPU; sem rede"})
    print("viable degrees:", viable, "status", summary_status)
    return summary_status


def _stability(signs: dict, viable: list[str]) -> dict:
    """Sinal das diferencas pontuais em cada degrau viavel."""
    out = {}
    for key in ("d_dev_poisson_minus_global", "d_ll_nb2_minus_poisson",
                "d_dev_nb2_minus_global"):
        values = {d: signs[d][key] for d in viable if d in signs}
        sgn = {np.sign(v) for v in values.values()}
        out[key] = {"estimates": values,
                    "same_sign_across_viable": len(sgn) <= 1,
                    "verified": len(values) > 1}
    return out


def _sensitivities(root, train, val, base, fits_log, outputs, mark) -> dict:
    """Execute as sensibilidades no maior degrau viavel."""
    acc = {"metrics": [], "intervals": [], "calibration": [], "zeros": [],
           "reconciliation": []}
    label, n = base["label"], base["n"]
    sample = base["sample"]
    ev0 = base["ev"]
    alpha_main = base["fits"]["nb2"].alpha

    # (c) bootstrap por perfil no modelo principal
    t0 = time.perf_counter()
    iv = interval_table(ev0, label, "validation_cluster_bootstrap",
                        clusters=_profile_codes(val))
    iv["sensitivity"] = "profile_cluster_bootstrap"
    acc["intervals"].append(iv)
    mark("sens_profile_bootstrap", t0)

    # (e) faixa muito curta fora, so descritivo
    band = segments.segment_columns(val)["exposure_band"].to_numpy()
    mask = band != audit.BAND_LABELS[0]
    m = metric_rows(ev0, label, "validation", mask,
                    "exposure_gt_0.02_descriptive")
    m["sensitivity"] = "short_band_excluded_descriptive"
    acc["metrics"].append(m)

    # (a) BonusMalus
    t0 = time.perf_counter()
    pre_b = prep.Preprocessor(with_bonus_malus=True).fit(sample)
    ctx = {"degree": label, "sample_rows": int(len(sample)),
           "sample_ids_sha256": ids_hash(sample["IDpol"]),
           "context": "sens_bonus_malus"}
    fits_b, rec, mat = fit_candidates(sample, pre_b, ctx)
    del mat
    fits_log.extend(rec)
    mu_b, _ = predict_all(fits_b, pre_b, val)
    mu_all = {"poisson": ev0.mu["poisson"], "nb2": ev0.mu["nb2"],
              "poisson_bonus": mu_b["poisson"], "nb2_bonus": mu_b["nb2"]}
    ev_b = EvalSet(val, mu_all, {"nb2": alpha_main,
                                 "nb2_bonus": fits_b["nb2"].alpha},
                   [("poisson_bonus", "poisson"), ("nb2_bonus", "nb2"),
                    ("nb2_bonus", "poisson_bonus")])
    _collect(acc, ev_b, label, "validation_bonus_malus",
             "bonus_malus")
    mark("sens_bonus_malus", t0)

    # (b) limites didaticos so no ajuste
    t0 = time.perf_counter()
    clipped = sample.copy()
    clipped["ClaimNb"] = clipped["ClaimNb"].clip(upper=4)
    clipped["Exposure"] = clipped["Exposure"].clip(upper=1.0)
    pre_c = prep.Preprocessor().fit(clipped)
    ctx["context"] = "sens_clipped_fit"
    fits_c, rec, mat = fit_candidates(clipped, pre_c, ctx)
    del mat
    fits_log.extend(rec)
    mu_c, _ = predict_all(fits_c, pre_c, val)  # exposicao original
    mu_all = {"poisson": ev0.mu["poisson"], "nb2": ev0.mu["nb2"],
              "poisson_clipfit": mu_c["poisson"],
              "nb2_clipfit": mu_c["nb2"], "global_clipfit": mu_c["global"]}
    ev_c = EvalSet(val, mu_all, {"nb2": alpha_main,
                                 "nb2_clipfit": fits_c["nb2"].alpha},
                   [("poisson_clipfit", "poisson"), ("nb2_clipfit", "nb2"),
                    ("global_clipfit", "poisson_clipfit")])
    _collect(acc, ev_c, label, "validation_clipped_fit", "clipped_fit")
    mark("sens_clipped_fit", t0)

    # diagnostico: log(Exposure) livre (nao candidato)
    t0 = time.perf_counter()
    mat_fit, _ = base["pre"].transform(sample)
    design = np.hstack([mat_fit, np.log(sample["Exposure"].to_numpy())
                        .reshape(-1, 1)])
    with models.ResourceMonitor() as mon:
        free = models.fit_poisson(
            design, sample["ClaimNb"].to_numpy(dtype="float64"),
            np.ones(len(sample)), base["pre"].columns + ["log_Exposure"])
    fits_log.append({"degree": label, "context": "diagnostic_free_exposure",
                     "candidate": "poisson_free_exposure",
                     "n_fit": int(len(sample)),
                     "converged": bool(free.converged),
                     "iterations": free.iterations,
                     "wall_s": round(mon.wall_s, 3),
                     "peak_rss_mib": round(mon.peak_rss_mib, 1),
                     "within_limits": True, "loglik_fit": free.loglik,
                     "warnings": "; ".join(free.warnings)})
    del mat_fit, design
    mat_v, _ = base["pre"].transform(val)
    mu_free = free.predict(np.hstack([mat_v, np.log(
        val["Exposure"].to_numpy()).reshape(-1, 1)]), np.ones(len(val)))
    mu_all = {"poisson": ev0.mu["poisson"], "nb2": ev0.mu["nb2"],
              "poisson_free_exposure": mu_free}
    ev_f = EvalSet(val, mu_all, {"nb2": alpha_main},
                   [("poisson_free_exposure", "poisson"),
                    ("poisson_free_exposure", "nb2")])
    _collect(acc, ev_f, label, "validation_free_exposure_diagnostic",
             "free_exposure_diagnostic")
    acc["metrics"][-1]["free_log_exposure_coef"] = float(free.beta[-1])
    mark("sens_free_exposure", t0)

    # (d) particao agrupada do desenvolvimento
    t0 = time.perf_counter()
    grouped = _grouped(root, train, val, n, label, fits_log, outputs)
    for key in acc:
        acc[key].extend(grouped[key])
    mark("sens_grouped_partition", t0)
    return acc


def _collect(acc, ev, label, population, sensitivity) -> None:
    """Gere as tabelas de uma sensibilidade e acumule em ``acc``."""
    iv = interval_table(ev, label, population)
    iv["sensitivity"] = sensitivity
    met = metric_rows(ev, label, population)
    met["sensitivity"] = sensitivity
    cal = calibration_rows(ev, label, population, iv)
    cal["sensitivity"] = sensitivity
    zer = zero_rows(ev, label, population, iv)
    zer["sensitivity"] = sensitivity
    rec = reconcile(ev, label, population)
    rec["sensitivity"] = sensitivity
    acc["intervals"].append(iv)
    acc["metrics"].append(met)
    acc["calibration"].append(cal)
    acc["zeros"].append(zer)
    acc["reconciliation"].append(rec)


def _grouped(root, train, val, n, label, fits_log, outputs) -> dict:
    """Sensibilidade agrupada: refaz pre-processamento, ajustes e avaliacao."""
    config = json.loads((root / "outputs" / "etapa_02"
                         / "partition_config.json").read_text("utf-8"))
    path = root / config["grouped_assignment_file"]["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config["grouped_assignment_file"]["sha256"]:
        raise ValueError("atribuicoes agrupadas divergem da etapa 2")
    assign = pd.read_csv(path)
    dev = pd.concat([train, val], ignore_index=True)
    part = assign.set_index("IDpol")["grouped_partition"].reindex(
        dev["IDpol"]).to_numpy()
    g_train = dev[part == "group_train"].reset_index(drop=True)
    g_val = dev[part == "group_validation"].reset_index(drop=True)
    sample = select_sample(g_train, n)
    pre = prep.Preprocessor().fit(sample)
    ctx = {"degree": label, "sample_rows": int(len(sample)),
           "sample_ids_sha256": ids_hash(sample["IDpol"]),
           "context": "sens_grouped_partition"}
    fits, rec, mat = fit_candidates(sample, pre, ctx)
    del mat
    fits_log.extend(rec)
    mu, _ = predict_all(fits, pre, g_val)
    ev = EvalSet(g_val, mu, {"nb2": fits["nb2"].alpha}, PAIRS)
    acc = {"metrics": [], "intervals": [], "calibration": [], "zeros": [],
           "reconciliation": []}
    _collect(acc, ev, label, "grouped_validation", "grouped_partition")
    acc["metrics"][-1]["group_train_rows"] = int(len(g_train))
    acc["metrics"][-1]["group_validation_rows"] = int(len(g_val))
    write_json(outputs / "grouped_partition_info.json", {
        "assignment_sha256": digest, "group_train_rows": int(len(g_train)),
        "group_validation_rows": int(len(g_val)),
        "fit_rows": int(len(sample)),
        "profile_key_overlap": int(len(
            set(profiles.profile_keys(g_train))
            & set(profiles.profile_keys(g_val))))})
    return acc
