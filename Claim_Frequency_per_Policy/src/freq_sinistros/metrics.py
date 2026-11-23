"""Metricas e bootstrap do protocolo de avaliacao (etapa 2).

Nesta etapa as funcoes so sao verificadas com dados sinteticos. Elas nao
ajustam modelos nem produzem previsoes: recebem contagens observadas,
medias previstas e exposicoes ja calculadas.

Convencoes: ``mu`` e a contagem esperada da linha (taxa x exposicao);
``alpha`` e o parametro de dispersao de NB2, ``Var = mu + alpha * mu**2``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

from freq_sinistros.audit import safe_ratio
from freq_sinistros.segments import low_support

BOOTSTRAP_REPLICAS = 1000
BOOTSTRAP_SEED = 20261121
CI_LEVEL = 0.95
MAX_UNDEFINED_SHARE = 0.01


def check_inputs(y, mu, exposure=None) -> tuple[np.ndarray, ...]:
    """Valide e converta as entradas das metricas.

    Args:
        y: contagens observadas (finitas, inteiras, ``>= 0``).
        mu: medias previstas (finitas e ``> 0``).
        exposure: exposicoes (finitas e ``> 0``), opcional.

    Returns:
        Arrays ``float64`` de ``y``, ``mu`` (e ``exposure`` se dado).

    Raises:
        ValueError: se houver valor invalido ou tamanhos diferentes. Nenhuma
            linha e descartada em silencio.
    """
    y = np.asarray(y, dtype="float64")
    mu = np.asarray(mu, dtype="float64")
    if y.shape != mu.shape or y.size == 0:
        raise ValueError("y e mu devem ter o mesmo tamanho, nao vazio")
    if not (np.isfinite(y).all() and (y >= 0).all()
            and (y == np.floor(y)).all()):
        raise ValueError("ClaimNb invalido na avaliacao")
    if not (np.isfinite(mu).all() and (mu > 0).all()):
        raise ValueError("media prevista deve ser finita e positiva")
    if exposure is None:
        return y, mu
    exposure = np.asarray(exposure, dtype="float64")
    if exposure.shape != y.shape or not (
            np.isfinite(exposure).all() and (exposure > 0).all()):
        raise ValueError("Exposure invalida na avaliacao")
    return y, mu, exposure


def poisson_deviance_contributions(y, mu) -> np.ndarray:
    """Contribuicao de cada linha ao desvio de Poisson.

    ``2*(y*log(y/mu) - (y - mu))``, com ``y*log(y/mu) = 0`` se ``y = 0``.
    """
    y, mu = check_inputs(y, mu)
    ratio = np.where(y > 0, y / mu, 1.0)
    return 2.0 * (np.where(y > 0, y * np.log(ratio), 0.0) - (y - mu))


def mean_poisson_deviance(y, mu) -> float:
    """Desvio medio de Poisson por registro (metrica primaria).

    ``D = (2/n) * sum(y*log(y/mu) - (y - mu))``, com o termo
    ``y*log(y/mu) = 0`` quando ``y = 0`` (a linha contribui ``2*mu``).
    """
    return float(poisson_deviance_contributions(y, mu).mean())


def _lgamma(values: np.ndarray) -> np.ndarray:
    """``log(Gamma(x))`` por valores unicos (sem depender de scipy)."""
    unique, inverse = np.unique(values, return_inverse=True)
    table = np.array([math.lgamma(float(v)) for v in unique])
    return table[inverse].reshape(values.shape)


def poisson_logpmf(y, mu) -> np.ndarray:
    """Log da probabilidade de Poisson por linha."""
    y, mu = check_inputs(y, mu)
    return y * np.log(mu) - mu - _lgamma(y + 1.0)


def nb2_logpmf(y, mu, alpha: float) -> np.ndarray:
    """Log da probabilidade NB2 por linha (``alpha > 0``).

    Forma numericamente estavel, inclusive para ``alpha`` proximo de zero
    (limite de Poisson): com ``r = 1/alpha`` e ``y`` inteiro,
    ``lgamma(y + r) - lgamma(r) = sum_{k<y} log(r + k)`` e
    ``r*log(r/(r + mu)) = -r*log1p(mu/r)``.
    """
    if not (np.isfinite(alpha) and alpha > 0):
        raise ValueError("alpha deve ser finito e positivo")
    y, mu = check_inputs(y, mu)
    r = 1.0 / alpha
    unique = np.unique(y)
    rising = np.array([np.log(r + np.arange(int(v))).sum()
                       for v in unique])
    head = rising[np.searchsorted(unique, y)]
    return (head - _lgamma(y + 1.0) - r * np.log1p(mu / r)
            + y * (np.log(mu) - np.log(r + mu)))


def mean_loglik(logpmf: np.ndarray) -> float:
    """Log-verossimilhanca preditiva media por registro."""
    return float(np.mean(logpmf))


def zero_prob(mu, alpha: float | None = None) -> np.ndarray:
    """``P(Y=0)`` por linha: Poisson (``alpha=None``) ou NB2."""
    mu = np.asarray(mu, dtype="float64")
    if alpha is None:
        return np.exp(-mu)
    return np.power(1.0 + alpha * mu, -1.0 / alpha)


def zero_calibration(y, p0) -> dict:
    """Zeros observados contra zeros previstos ``sum P(Y_i = 0)``."""
    y = np.asarray(y, dtype="float64")
    observed = float((y == 0).sum())
    predicted = float(np.sum(p0))
    return {"zeros_observed": observed, "zeros_predicted": predicted,
            "difference_predicted_minus_observed": predicted - observed}


def calibration_table(segment: pd.Series, y, mu, exposure) -> pd.DataFrame:
    """Calibracao por segmento: observado, previsto e diferenca de taxas.

    A taxa de cada grupo usa a exposicao total do grupo e a diferenca e
    **prevista menos observada**, por 100 apolices-ano. Linhas de baixo
    suporte permanecem visiveis e sinalizadas.
    """
    y, mu, exposure = check_inputs(y, mu, exposure)
    data = pd.DataFrame({"segment": np.asarray(segment), "y": y, "mu": mu,
                         "exposure": exposure})
    out = []
    for name, part in data.groupby("segment", sort=True):
        e, o, p = (part["exposure"].sum(), part["y"].sum(),
                   part["mu"].sum())
        obs_rate = safe_ratio(100 * o, e)
        pred_rate = safe_ratio(100 * p, e)
        out.append({
            "segment": name, "rows": int(len(part)), "exposure": float(e),
            "observed_claims": float(o), "predicted_claims": float(p),
            "observed_rate_per_100": obs_rate,
            "predicted_rate_per_100": pred_rate,
            "rate_difference_pred_minus_obs": pred_rate - obs_rate,
            "low_support": low_support(len(part), e),
        })
    return pd.DataFrame(out)


def bootstrap_indices(n: int, seed: int, replicas: int) -> Iterator[
        np.ndarray]:
    """Gere os indices de reamostragem por registro (com reposicao)."""
    rng = np.random.default_rng(seed)
    for _ in range(replicas):
        yield rng.integers(0, n, size=n)


def cluster_bootstrap_indices(codes, seed: int, replicas: int) -> Iterator[
        np.ndarray]:
    """Gere indices reamostrando grupos inteiros (chave de perfil).

    Args:
        codes: rotulo do grupo de cada linha.
        seed: semente.
        replicas: numero de replicas.
    """
    codes = np.asarray(codes)
    order = np.argsort(codes, kind="stable")
    _, starts, counts = np.unique(codes[order], return_index=True,
                                  return_counts=True)
    rng = np.random.default_rng(seed)
    n_groups = len(starts)
    for _ in range(replicas):
        pick = rng.integers(0, n_groups, size=n_groups)
        yield np.concatenate([order[starts[g]:starts[g] + counts[g]]
                              for g in pick])


def bootstrap_intervals(
    statistic: Callable[[np.ndarray], dict[str, float]],
    n: int,
    replicas: int = BOOTSTRAP_REPLICAS,
    seed: int = BOOTSTRAP_SEED,
    clusters=None,
) -> pd.DataFrame:
    """Intervalos percentil de 95% com modelos fixos.

    O mesmo sorteio e usado para todos os indicadores (e, portanto, para
    todos os modelos), o que permite diferencas pareadas. Replicas em que
    um indicador nao esta definido (``NaN``) sao excluidas desse indicador e
    contadas; se passarem de 1% das replicas, o intervalo nao e informado.

    Args:
        statistic: funcao que recebe os indices da amostra (``None`` para a
            amostra completa) e devolve ``{indicador: valor}``.
        n: numero de registros avaliados.
        replicas: numero de replicas.
        seed: semente do gerador.
        clusters: rotulos de grupo para o bootstrap por perfil (opcional).

    Returns:
        Uma linha por indicador: estimativa, limites, replicas indefinidas
        e ``interval_reported``.
    """
    point = statistic(np.arange(n))
    draws = (cluster_bootstrap_indices(clusters, seed, replicas)
             if clusters is not None
             else bootstrap_indices(n, seed, replicas))
    values = {k: [] for k in point}
    for idx in draws:
        result = statistic(idx)
        for key in point:
            values[key].append(result[key])
    alpha = (1 - CI_LEVEL) / 2 * 100
    rows = []
    for key, estimate in point.items():
        arr = np.asarray(values[key], dtype="float64")
        undefined = int(np.isnan(arr).sum())
        ok = undefined <= MAX_UNDEFINED_SHARE * replicas and (
            undefined < replicas)
        lo, hi = (np.nanpercentile(arr, [alpha, 100 - alpha])
                  if ok else (np.nan, np.nan))
        rows.append({"indicator": key, "estimate": estimate,
                     "ci_low": float(lo), "ci_high": float(hi),
                     "undefined_replicas": undefined,
                     "replicas": replicas, "interval_reported": bool(ok)})
    return pd.DataFrame(rows)
