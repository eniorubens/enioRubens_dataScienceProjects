"""Congelamento e verificacao do modelo selecionado (etapa 4).

O artefato congelado guarda o pre-processador ja ajustado, a ordem das
colunas, os coeficientes do Poisson com offset fixo, a taxa global e os
hashes de origem. A verificacao restaura tudo sem reaprender o
pre-processamento nem reajustar o estimador e reproduz as metricas de
validacao aprovadas.

Precisao: o JSON guarda cada ``float`` com ``repr`` (17 digitos
significativos), que e exato na ida e na volta; os coeficientes restaurados
sao identicos bit a bit. A tolerancia da verificacao cobre apenas a
reordenacao de somas de ponto flutuante (``rtol = 1e-9``, ``atol = 1e-12``).
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from freq_sinistros import __version__, metrics, prep, segments
from freq_sinistros.models import ETA_MAX

CRITICAL_MODULES = ["prep", "models", "metrics", "segments", "partition",
                    "data", "protocol", "audit"]
RTOL = 1e-9
ATOL = 1e-12
MODEL_NAME = "poisson_offset_fixed"
REFERENCE_NAME = "global_rate"


def code_hashes(root: Path | None = None) -> dict:
    """SHA-256 dos modulos que definem previsao e metricas.

    Os modulos sao lidos do pacote efetivamente importado (``root`` e
    ignorado e mantido so por compatibilidade).
    """
    base = Path(__file__).resolve().parent
    return {name: hashlib.sha256((base / f"{name}.py").read_bytes())
            .hexdigest() for name in CRITICAL_MODULES}


def file_hash(path: Path) -> str:
    """SHA-256 de um arquivo."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(frozen: dict) -> tuple[prep.Preprocessor, np.ndarray, float]:
    """Restaure pre-processador, coeficientes e taxa global.

    Args:
        frozen: artefato congelado (dicionario lido do JSON).

    Returns:
        Pre-processador, vetor ``beta`` na ordem das colunas congeladas e
        a taxa global.

    Raises:
        ValueError: se a ordem das colunas restauradas diferir da
            congelada ou se o numero de coeficientes nao conferir.
    """
    pre = prep.Preprocessor.from_summary(frozen["preprocessor"])
    columns = frozen["estimator"]["columns"]
    if pre.columns != columns:
        raise ValueError("ordem das colunas restauradas difere da congelada")
    beta = np.array(frozen["estimator"]["coefficients"], dtype="float64")
    if beta.shape[0] != len(columns):
        raise ValueError("numero de coeficientes difere do de colunas")
    return pre, beta, float(frozen["global_rate"]["rate"])


def predict_frozen(frozen: dict, frame: pd.DataFrame) -> dict:
    """Previsoes ``mu`` do Poisson congelado e da taxa global.

    ``mu = Exposure * exp(x'beta)`` (offset ``log(Exposure)`` com
    coeficiente 1); a taxa global preve ``taxa * Exposure``.

    Args:
        frozen: artefato congelado.
        frame: linhas a prever (covariaveis e ``Exposure`` originais).

    Returns:
        ``{"poisson": mu, "global": mu, "report": relatorio do
        pre-processador}``.
    """
    pre, beta, rate = restore(frozen)
    matrix, report = pre.transform(frame)
    exposure = frame["Exposure"].to_numpy(dtype="float64")
    eta = np.clip(matrix @ beta, -50.0, ETA_MAX)
    return {"poisson": exposure * np.exp(eta), "global": rate * exposure,
            "report": report}


def aggregate_reference(frame: pd.DataFrame, mu: dict) -> dict:
    """Agregados de validacao usados para verificar o congelamento.

    Args:
        frame: linhas de validacao (contagens e exposicoes originais).
        mu: previsoes por candidato (``poisson`` e ``global``).

    Returns:
        Metricas globais por candidato e somas por faixa de exposicao.
    """
    y = frame["ClaimNb"].to_numpy(dtype="float64")
    e = frame["Exposure"].to_numpy(dtype="float64")
    band = segments.segment_columns(frame)["exposure_band"].to_numpy()
    out = {"rows": int(len(y)), "exposure": float(e.sum()),
           "observed_claims": float(y.sum()), "candidates": {}}
    for name, values in mu.items():
        if name == "report":
            continue
        per_band = {}
        for label in dict.fromkeys(band):
            m = band == label
            per_band[label] = {"rows": int(m.sum()),
                               "exposure": float(e[m].sum()),
                               "observed_claims": float(y[m].sum()),
                               "predicted_claims": float(values[m].sum())}
        out["candidates"][name] = {
            "mean_poisson_deviance": metrics.mean_poisson_deviance(y, values),
            "mean_loglik": float(metrics.poisson_logpmf(y, values).mean()),
            "predicted_claims": float(values.sum()),
            "rate_diff_pred_minus_obs": float(
                100 * (values.sum() - y.sum()) / e.sum()),
            "zeros_predicted": float(np.exp(-values).sum()),
            "by_exposure_band": per_band}
    return out


def compare_references(a: dict, b: dict, rtol: float = RTOL,
                       atol: float = ATOL) -> dict:
    """Compare dois agregados de validacao dentro da tolerancia.

    Returns:
        ``max_abs_diff``, ``max_rel_diff``, ``n_compared`` e ``ok``.
    """
    pairs: list[tuple[float, float]] = []
    for key in ("rows", "exposure", "observed_claims"):
        pairs.append((a[key], b[key]))
    for name, cand in a["candidates"].items():
        other = b["candidates"][name]
        for key in ("mean_poisson_deviance", "mean_loglik",
                    "predicted_claims", "rate_diff_pred_minus_obs",
                    "zeros_predicted"):
            pairs.append((cand[key], other[key]))
        for label, values in cand["by_exposure_band"].items():
            for key, value in values.items():
                pairs.append((value, other["by_exposure_band"][label][key]))
    left = np.array([p[0] for p in pairs], dtype="float64")
    right = np.array([p[1] for p in pairs], dtype="float64")
    diff = np.abs(left - right)
    rel = diff / np.maximum(np.abs(right), 1e-300)
    return {"n_compared": int(len(pairs)),
            "max_abs_diff": float(diff.max()),
            "max_rel_diff": float(rel.max()),
            "ok": bool(np.allclose(left, right, rtol=rtol, atol=atol))}


def environment_record() -> dict:
    """Versoes do ambiente que geraram o artefato."""
    return {"package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.platform()}
