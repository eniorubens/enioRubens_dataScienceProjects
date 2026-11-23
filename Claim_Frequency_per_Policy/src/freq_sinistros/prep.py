"""Pre-processamento ajustado somente nas linhas de ajuste (etapa 3).

Passos aprovados no protocolo: niveis categoricos com menos de 500 linhas
de ajuste vao para ``OTHER``; ``VehAge``, ``DrivAge`` (e ``BonusMalus`` na
sensibilidade) viram faixas por quantis; ``Density`` entra como ``log``
padronizado. ``IDpol`` nunca e preditor. Tudo que depende dos dados e
aprendido em :meth:`Preprocessor.fit`; :meth:`Preprocessor.transform` so
aplica o aprendido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["Area", "VehPower", "VehBrand", "VehGas", "Region"]
BINNED = ["VehAge", "DrivAge"]
MAIN_COVARIATES = ["Area", "VehPower", "VehAge", "DrivAge", "VehBrand",
                   "VehGas", "Density", "Region"]
OTHER = "OTHER"
MISSING = "__MISSING__"
RARE_MIN_ROWS = 500
N_BINS = 10


def _as_text(series: pd.Series, column: str) -> pd.Series:
    """Texto estavel de uma coluna categorica (``VehPower`` inteiro)."""
    if column == "VehPower":
        values = series.to_numpy(dtype="float64")
        out = np.full(len(values), MISSING, dtype=object)
        ok = np.isfinite(values)
        out[ok] = np.round(values[ok]).astype("int64").astype(str)
        return pd.Series(out, index=series.index)
    text = series.astype("object").where(series.notna(), MISSING)
    return text.astype(str)


class Preprocessor:
    """Transformacao ajustada nas linhas de ajuste e aplicada a outras.

    Args:
        with_bonus_malus: inclui ``BonusMalus`` em faixas (sensibilidade).
        rare_min_rows: minimo de linhas de ajuste por nivel.
        n_bins: numero de faixas de quantis.
    """

    def __init__(self, with_bonus_malus: bool = False,
                 rare_min_rows: int = RARE_MIN_ROWS,
                 n_bins: int = N_BINS) -> None:
        """Defina a configuracao; nenhum dado e lido aqui."""
        self.with_bonus_malus = with_bonus_malus
        self.rare_min_rows = rare_min_rows
        self.n_bins = n_bins
        self.binned = BINNED + (["BonusMalus"] if with_bonus_malus else [])
        self.fitted = False

    # ---------------------------------------------------------- ajuste
    def fit(self, frame: pd.DataFrame) -> Preprocessor:
        """Aprenda niveis, cortes, referencias e medianas nas linhas dadas.

        Args:
            frame: linhas de ajuste (com as covariaveis).

        Returns:
            O proprio objeto, ajustado.

        Raises:
            ValueError: se ``frame`` estiver vazio.
        """
        if len(frame) == 0:
            raise ValueError("sem linhas de ajuste")
        self.n_fit = int(len(frame))
        self.levels: dict[str, list[str]] = {}
        self.reference: dict[str, str] = {}
        self.pooled: dict[str, list[str]] = {}
        self.has_other: dict[str, bool] = {}
        for col in CATEGORICAL:
            text = _as_text(frame[col], col)
            counts = text.value_counts()
            kept = sorted(counts[counts >= self.rare_min_rows].index)
            if not kept:  # garante ao menos o nivel mais frequente
                kept = [str(counts.idxmax())]
            pooled = sorted(set(counts.index) - set(kept))
            self.pooled[col] = pooled
            self.has_other[col] = bool(pooled)
            self.reference[col] = str(counts.loc[kept].idxmax())
            self.levels[col] = kept + ([OTHER] if pooled else [])
        self.medians = {c: float(frame[c].median())
                        for c in self.binned + ["Density"]}
        self.edges: dict[str, list[float]] = {}
        self.bin_reference: dict[str, int] = {}
        for col in self.binned:
            values = frame[col].fillna(self.medians[col]).to_numpy(
                dtype="float64")
            q = np.quantile(values, np.linspace(0, 1, self.n_bins + 1)[1:-1],
                            method="averaged_inverted_cdf")
            edges = sorted(set(float(v) for v in q))
            self.edges[col] = edges
            codes = np.searchsorted(edges, values, side="right")
            self.bin_reference[col] = int(np.bincount(codes).argmax())
        logd = np.log(frame["Density"].fillna(
            self.medians["Density"]).clip(lower=1e-9))
        self.log_density_mean = float(logd.mean())
        self.log_density_std = float(logd.std(ddof=0)) or 1.0
        self.fitted = True
        self.columns = self._column_names()
        return self

    def _column_names(self) -> list[str]:
        """Nomes das colunas da matriz de desenho, na ordem de saida."""
        names = ["const"]
        for col in CATEGORICAL:
            names += [f"{col}={lv}" for lv in self.levels[col]
                      if lv != self.reference[col]]
        for col in self.binned:
            n_levels = len(self.edges[col]) + 1
            names += [f"{col}=bin{k}" for k in range(n_levels)
                      if k != self.bin_reference[col]]
        return names + ["log_Density_std"]

    # ----------------------------------------------------- transformacao
    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """Monte a matriz de desenho com o que foi aprendido no ajuste.

        Niveis fora do ajuste (ou raros) vao para ``OTHER`` quando existe;
        caso contrario, para a referencia. Ausentes categoricos viram
        ``__MISSING__`` (tratado como desconhecido) e numericos recebem a
        mediana do ajuste.

        Args:
            frame: linhas a transformar.

        Returns:
            Matriz ``float64`` e dicionario com contagens de
            desconhecidos e imputados por coluna.

        Raises:
            RuntimeError: se ``fit`` ainda nao foi chamado.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor nao ajustado")
        n = len(frame)
        parts = [np.ones((n, 1))]
        report = {"unknown_levels": {}, "imputed_numeric": {}}
        for col in CATEGORICAL:
            text = _as_text(frame[col], col)
            known = set(self.levels[col]) - {OTHER}
            seen = text.isin(known)
            report["unknown_levels"][col] = int((~seen).sum())
            fallback = OTHER if self.has_other[col] else self.reference[col]
            mapped = text.where(seen, fallback)
            for lv in self.levels[col]:
                if lv != self.reference[col]:
                    parts.append((mapped == lv).to_numpy(
                        dtype="float64").reshape(-1, 1))
        for col in self.binned:
            raw = frame[col]
            report["imputed_numeric"][col] = int(
                (~np.isfinite(raw.to_numpy(dtype="float64"))).sum())
            values = raw.where(np.isfinite(raw), self.medians[col]).to_numpy(
                dtype="float64")
            codes = np.searchsorted(self.edges[col], values, side="right")
            for k in range(len(self.edges[col]) + 1):
                if k != self.bin_reference[col]:
                    parts.append((codes == k).astype("float64").reshape(
                        -1, 1))
        dens = frame["Density"].to_numpy(dtype="float64")
        report["imputed_numeric"]["Density"] = int((~np.isfinite(dens)
                                                    | (dens <= 0)).sum())
        dens = np.where(np.isfinite(dens) & (dens > 0), dens,
                        self.medians["Density"])
        parts.append(((np.log(dens) - self.log_density_mean)
                      / self.log_density_std).reshape(-1, 1))
        return np.hstack(parts), report

    @classmethod
    def from_summary(cls, state: dict) -> Preprocessor:
        """Restaure um pre-processador ja ajustado a partir de ``summary()``.

        Nenhum dado e lido: niveis, referencias, cortes, medianas e a
        padronizacao de ``Density`` vem do estado persistido, de modo que
        ``transform`` reproduz exatamente o ajuste original.

        Args:
            state: dicionario produzido por :meth:`summary` (ou lido do
                JSON congelado).

        Returns:
            Um ``Preprocessor`` pronto para ``transform``.
        """
        pre = cls(with_bonus_malus=bool(state["with_bonus_malus"]),
                  rare_min_rows=int(state["rare_min_rows"]))
        pre.n_fit = int(state["n_fit"])
        pre.levels = {k: list(v) for k, v in state["levels"].items()}
        pre.reference = dict(state["reference_level"])
        pre.pooled = {k: list(v) for k, v in
                      state["pooled_into_OTHER"].items()}
        pre.has_other = {k: bool(v) for k, v in pre.pooled.items()}
        pre.medians = {k: float(v) for k, v in
                       state["medians_for_imputation"].items()}
        pre.edges = {k: [float(x) for x in v]
                     for k, v in state["bin_edges"].items()}
        pre.bin_reference = {k: int(v) for k, v in
                             state["bin_reference_index"].items()}
        pre.log_density_mean = float(state["log_density_mean"])
        pre.log_density_std = float(state["log_density_std"])
        pre.fitted = True
        pre.columns = pre._column_names()
        return pre

    # ------------------------------------------------------- relatorios
    def summary(self) -> dict:
        """Resumo serializavel do que foi aprendido no ajuste."""
        return {
            "n_fit": self.n_fit, "n_columns": len(self.columns),
            "rare_min_rows": self.rare_min_rows,
            "levels": self.levels, "reference_level": self.reference,
            "pooled_into_OTHER": self.pooled,
            "bin_edges": self.edges,
            "bin_reference_index": self.bin_reference,
            "medians_for_imputation": self.medians,
            "log_density_mean": self.log_density_mean,
            "log_density_std": self.log_density_std,
            "with_bonus_malus": self.with_bonus_malus,
        }


def design_checks(matrix: np.ndarray, columns: list[str]) -> dict:
    """Verifique a matriz de desenho: finitude, posto e colunas constantes.

    Args:
        matrix: matriz de desenho com intercepto na primeira coluna.
        columns: nomes das colunas.

    Returns:
        Dicionario com as verificacoes e o numero de condicao de ``X'X``
        (colunas padronizadas).
    """
    finite = bool(np.isfinite(matrix).all())
    gram = matrix.T @ matrix
    rank = int(np.linalg.matrix_rank(gram))
    scale = np.sqrt(np.diag(gram))
    scaled = gram / np.outer(scale, scale)
    sums = matrix[:, 1:].sum(axis=0)
    return {
        "rows": int(matrix.shape[0]), "columns": int(matrix.shape[1]),
        "all_finite": finite, "rank": rank,
        "full_column_rank": rank == matrix.shape[1],
        "condition_number_scaled_gram": float(np.linalg.cond(scaled)),
        "empty_columns": [columns[i + 1] for i in
                          np.where(sums == 0)[0].tolist()],
    }
