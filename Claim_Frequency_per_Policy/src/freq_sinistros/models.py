"""Estimadores: taxa global, Poisson e NB2 com offset log(Exposure).

Poisson e NB2 usam ligacao log e ``log(Exposure)`` como offset de
coeficiente fixo 1: ``mu_i = Exposure_i * exp(x_i' beta)``. A estimacao e
por minimos quadrados ponderados iterativos (IRLS) com meio-passo de
seguranca; no NB2, ``alpha`` (``Var = mu + alpha*mu^2``) e atualizado por
maxima verossimilhanca unidimensional, alternando com ``beta`` ate a
convergencia conjunta (verificada pelo escore).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
from scipy.optimize import minimize_scalar

from freq_sinistros import metrics

ALPHA_MIN = 1e-8
ALPHA_MAX = 100.0
ETA_MAX = 30.0


class ResourceMonitor:
    """Mede tempo de parede e pico de memoria residente durante um bloco.

    Uma thread amostra o RSS do processo a cada ``interval`` segundos. O
    pico reportado e o maximo amostrado; o acrescimo e o pico menos o RSS
    inicial. Use como gerenciador de contexto.
    """

    def __init__(self, interval: float = 0.02) -> None:
        """Prepare o monitor (``interval`` em segundos)."""
        self.interval = interval
        self.wall_s = 0.0
        self.peak_rss_mib = 0.0
        self.delta_rss_mib = 0.0

    def __enter__(self) -> ResourceMonitor:
        """Inicie a medicao."""
        self._proc = psutil.Process()
        self._start_rss = self._proc.memory_info().rss
        self._peak = self._start_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def _sample(self) -> None:
        """Amostre o RSS ate o bloco terminar."""
        while not self._stop.is_set():
            self._peak = max(self._peak, self._proc.memory_info().rss)
            self._stop.wait(self.interval)

    def __exit__(self, *exc) -> None:
        """Encerre a medicao e calcule tempo e pico."""
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._proc.memory_info().rss)
        self.wall_s = time.perf_counter() - self._t0
        self.peak_rss_mib = self._peak / 2**20
        self.delta_rss_mib = (self._peak - self._start_rss) / 2**20


@dataclass
class FitResult:
    """Resultado de um ajuste.

    Attributes:
        name: ``global``, ``poisson`` ou ``nb2``.
        beta: coeficientes (``None`` na taxa global).
        columns: nomes das colunas de ``beta``.
        rate: taxa global (so no candidato ``global``).
        alpha: dispersao NB2 (``None`` nos demais).
        loglik: log-verossimilhanca na amostra de ajuste.
        converged: criterio de convergencia atendido.
        iterations: iteracoes IRLS (externas, no NB2).
        warnings: avisos do ajuste.
        extras: diagnosticos (escore, limite de Poisson, erros padrao).
    """

    name: str
    beta: np.ndarray | None = None
    columns: list[str] = field(default_factory=list)
    rate: float | None = None
    alpha: float | None = None
    loglik: float = float("nan")
    converged: bool = True
    iterations: int = 0
    warnings: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def predict(self, matrix: np.ndarray | None,
                exposure: np.ndarray) -> np.ndarray:
        """Contagem esperada ``mu = Exposure * exp(x'beta)`` (ou taxa)."""
        exposure = np.asarray(exposure, dtype="float64")
        if self.name == "global":
            return self.rate * exposure
        eta = np.clip(matrix @ self.beta, -50.0, ETA_MAX)
        return exposure * np.exp(eta)


def fit_global(y: np.ndarray, exposure: np.ndarray) -> FitResult:
    """Taxa global ``sum(y)/sum(exposure)`` como candidato de referencia."""
    y = np.asarray(y, dtype="float64")
    exposure = np.asarray(exposure, dtype="float64")
    rate = float(y.sum() / exposure.sum())
    result = FitResult("global", rate=rate, iterations=0)
    result.loglik = float(metrics.poisson_logpmf(y, rate * exposure).sum())
    return result


def _loglik(y, mu, alpha) -> float:
    """Log-verossimilhanca total (Poisson se ``alpha`` e ``None``)."""
    if alpha is None:
        return float(metrics.poisson_logpmf(y, mu).sum())
    return float(metrics.nb2_logpmf(y, mu, alpha).sum())


def _irls(matrix, y, offset, alpha, beta0, max_iter=100, tol=1e-10):
    """IRLS com ``alpha`` fixo (``None`` = Poisson).

    Returns:
        ``beta``, ``loglik``, iteracoes, convergiu e contagem de
        meio-passos.
    """
    a = 0.0 if alpha is None else alpha
    beta = beta0.copy()
    eta = np.clip(matrix @ beta + offset, -50.0, ETA_MAX)
    mu = np.exp(eta)
    ll = _loglik(y, mu, alpha)
    halvings = 0
    for it in range(1, max_iter + 1):
        w = mu / (1.0 + a * mu)
        z = (eta - offset) + (y - mu) / mu
        gram = matrix.T @ (matrix * w[:, None])
        rhs = matrix.T @ (w * z)
        try:
            new = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        step = 1.0
        while True:
            cand = beta + step * (new - beta)
            eta_c = np.clip(matrix @ cand + offset, -50.0, ETA_MAX)
            mu_c = np.exp(eta_c)
            ll_c = _loglik(y, mu_c, alpha)
            if np.isfinite(ll_c) and ll_c >= ll - 1e-9 * abs(ll):
                break
            step /= 2.0
            halvings += 1
            if step < 1e-6:
                return beta, ll, it, False, halvings
        converged = abs(ll_c - ll) <= tol * (abs(ll) + 1.0)
        beta, eta, mu, ll = cand, eta_c, mu_c, ll_c
        if converged:
            return beta, ll, it, True, halvings
    return beta, ll, max_iter, False, halvings


def _update_alpha(y, mu, start) -> tuple[float, bool]:
    """Maximize a log-verossimilhanca em ``log(alpha)`` com ``mu`` fixo.

    Returns:
        ``alpha`` e se ficou no limite inferior (limite de Poisson).
    """
    def neg(log_alpha: float) -> float:
        return -float(metrics.nb2_logpmf(y, mu, np.exp(log_alpha)).sum())

    result = minimize_scalar(
        neg, bounds=(np.log(ALPHA_MIN), np.log(ALPHA_MAX)),
        method="bounded", options={"xatol": 1e-9})
    alpha = float(np.exp(result.x))
    return alpha, bool(alpha <= ALPHA_MIN * 1.01)


def _standard_errors(matrix, mu, alpha) -> np.ndarray:
    """Erros padrao ingenuos ``sqrt(diag((X'WX)^-1))``."""
    w = mu / (1.0 + (alpha or 0.0) * mu)
    gram = matrix.T @ (matrix * w[:, None])
    return np.sqrt(np.diag(np.linalg.pinv(gram)))


def _initial_beta(matrix, y, exposure) -> np.ndarray:
    """Inicio: so intercepto, com a taxa global."""
    beta = np.zeros(matrix.shape[1])
    beta[0] = np.log(max(y.sum(), 1e-12) / exposure.sum())
    return beta


def fit_poisson(matrix, y, exposure, columns) -> FitResult:
    """Ajuste Poisson com offset ``log(exposure)``.

    Args:
        matrix: matriz de desenho (intercepto na primeira coluna).
        y: contagens.
        exposure: exposicoes (``> 0``).
        columns: nomes das colunas.
    """
    y = np.asarray(y, dtype="float64")
    exposure = np.asarray(exposure, dtype="float64")
    offset = np.log(exposure)
    beta, ll, it, ok, halvings = _irls(
        matrix, y, offset, None, _initial_beta(matrix, y, exposure))
    mu = exposure * np.exp(np.clip(matrix @ beta, -50.0, ETA_MAX))
    score = matrix.T @ (y - mu)
    result = FitResult("poisson", beta=beta, columns=list(columns),
                       loglik=ll, converged=ok, iterations=it)
    result.extras = {
        "step_halvings": halvings,
        "max_abs_score_per_row": float(np.abs(score).max() / len(y)),
        "std_errors": _standard_errors(matrix, mu, None).tolist(),
    }
    if not ok:
        result.warnings.append("IRLS Poisson nao convergiu")
    return result


def fit_nb2(matrix, y, exposure, columns, start: FitResult | None = None,
            max_outer: int = 50, tol: float = 1e-9) -> FitResult:
    """Ajuste NB2 (``Var = mu + alpha*mu^2``) com offset ``log(exposure)``.

    Alterna IRLS em ``beta`` (``alpha`` fixo) e maxima verossimilhanca
    em ``alpha`` (``mu`` fixo). Inicia no Poisson (``start``) com ``alpha``
    de maxima verossimilhanca dado o ``mu`` do Poisson. Se ``alpha`` fica
    no limite inferior, o
    ajuste esta no limite de Poisson, o que e registrado.

    Args:
        matrix: matriz de desenho.
        y: contagens.
        exposure: exposicoes.
        columns: nomes das colunas.
        start: ajuste Poisson para inicializar ``beta`` (opcional).
        max_outer: maximo de iteracoes externas.
        tol: tolerancia relativa da log-verossimilhanca.
    """
    y = np.asarray(y, dtype="float64")
    exposure = np.asarray(exposure, dtype="float64")
    offset = np.log(exposure)
    beta = (start.beta.copy() if start is not None
            else _initial_beta(matrix, y, exposure))
    mu = exposure * np.exp(np.clip(matrix @ beta, -50.0, ETA_MAX))
    alpha, at_limit = _update_alpha(y, mu, None)
    ll_prev = metrics.nb2_logpmf(y, mu, alpha).sum()
    ok, outer, halvings = False, 0, 0
    for _ in range(max_outer):
        outer += 1
        beta, ll, _, inner_ok, h = _irls(matrix, y, offset, alpha, beta,
                                         max_iter=50)
        halvings += h
        mu = exposure * np.exp(np.clip(matrix @ beta, -50.0, ETA_MAX))
        alpha, at_limit = _update_alpha(y, mu, alpha)
        ll = float(metrics.nb2_logpmf(y, mu, alpha).sum())
        if inner_ok and abs(ll - ll_prev) <= tol * (abs(ll) + 1.0):
            ok = True
            break
        ll_prev = ll
    score_beta = matrix.T @ ((y - mu) / (1.0 + alpha * mu))
    eps = 1e-5
    ll_hi = metrics.nb2_logpmf(y, mu, alpha * (1 + eps)).sum()
    ll_lo = metrics.nb2_logpmf(y, mu, alpha * (1 - eps)).sum()
    score_alpha = float((ll_hi - ll_lo) / (2 * eps * alpha))
    result = FitResult("nb2", beta=beta, columns=list(columns), alpha=alpha,
                       loglik=ll, converged=ok, iterations=outer)
    result.extras = {
        "step_halvings": halvings,
        "alpha_at_poisson_limit": bool(at_limit),
        "max_abs_score_beta_per_row": float(np.abs(score_beta).max()
                                            / len(y)),
        "score_alpha_per_row": score_alpha / len(y),
        "std_errors": _standard_errors(matrix, mu, alpha).tolist(),
    }
    if not ok:
        result.warnings.append("NB2 nao convergiu")
    if at_limit:
        result.warnings.append("alpha no limite inferior: limite de Poisson")
    return result


def dispersion_diagnostics(y, mu, n_parameters: int) -> dict:
    """Diagnostico de dispersao condicional a partir do Poisson ajustado.

    Calcula a estatistica de Pearson ``X2 = sum((y-mu)^2/mu)``, os graus de
    liberdade ``n - p`` e ``phi = X2/(n-p)``; e o estimador de
    Cameron e Trivedi de ``alpha`` (regressao de ``((y-mu)^2 - y)/mu`` em
    ``mu`` sem constante) com estatistica t robusta (HC0).

    Args:
        y: contagens da populacao de ajuste.
        mu: medias ajustadas do Poisson na mesma populacao.
        n_parameters: numero de parametros estimados (colunas de ``X``).

    Returns:
        Dicionario com populacao, graus de liberdade e estatisticas.
    """
    y, mu = metrics.check_inputs(y, mu)
    n = int(y.size)
    pearson = float(np.sum((y - mu) ** 2 / mu))
    df = n - n_parameters
    z = ((y - mu) ** 2 - y) / mu
    alpha_hat = float(np.sum(mu * z) / np.sum(mu * mu))
    resid = z - alpha_hat * mu
    var = float(np.sum((mu * resid) ** 2) / np.sum(mu * mu) ** 2)
    se = float(np.sqrt(var))
    return {
        "population_rows": n, "parameters": n_parameters,
        "degrees_of_freedom": df, "pearson_chi2": pearson,
        "pearson_dispersion_phi": pearson / df,
        "cameron_trivedi_alpha": alpha_hat,
        "cameron_trivedi_se_hc0": se,
        "cameron_trivedi_t": alpha_hat / se if se > 0 else float("nan"),
    }
