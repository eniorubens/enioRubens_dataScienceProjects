"""Segmentos predefinidos e suporte (etapa 2).

Familias: faixa de exposicao (reuso da auditoria), idade do veiculo e
regiao. Cada segmentacao reconcilia registros, exposicao e contagens com o
total. O limite de baixo suporte e verificado no treino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from freq_sinistros.audit import (
    BAND_LABELS,
    INVALID_BAND,
    assign_band,
    safe_ratio,
)

LOW_SUPPORT_ROWS = 500
LOW_SUPPORT_EXPOSURE = 25.0
VEHAGE_LABELS = ["0-1", "2-5", "6-10", "11-20", "21+"]
VEHAGE_INVALID = "idade_ausente_ou_invalida"
# Limites inferiores inclusivos; o superior de cada faixa e o inferior da
# seguinte (exclusivo): [0,2) [2,6) [6,11) [11,21) [21,inf).
VEHAGE_LOWER = [0, 2, 6, 11, 21]
REGION_MISSING = "__MISSING__"
FAMILIES = ("exposure_band", "vehicle_age", "region")
_FIXED_ORDER = {
    "exposure_band": list(BAND_LABELS) + [INVALID_BAND],
    "vehicle_age": VEHAGE_LABELS + [VEHAGE_INVALID],
}


def vehicle_age_band(vehage: pd.Series) -> pd.Series:
    """Atribua a faixa de idade do veiculo (0-1, 2-5, 6-10, 11-20, 21+).

    Ausentes, nao finitos e negativos recebem ``VEHAGE_INVALID``.
    """
    v = vehage.to_numpy(dtype="float64")
    valid = np.isfinite(v) & (v >= 0)
    safe = np.where(valid, v, 0.0)
    idx = np.searchsorted(VEHAGE_LOWER, safe, side="right") - 1
    labels = np.array(VEHAGE_LABELS, dtype=object)[idx]
    return pd.Series(np.where(valid, labels, VEHAGE_INVALID),
                     index=vehage.index)


def region_segment(region: pd.Series) -> pd.Series:
    """Codigo cadastral da regiao; ausente ou desconhecido vira categoria."""
    text = region.astype("object")
    return text.where(region.notna(), REGION_MISSING).astype(str)


def segment_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Devolva as tres segmentacoes predefinidas para ``frame``."""
    return {
        "exposure_band": assign_band(frame["Exposure"]),
        "vehicle_age": vehicle_age_band(frame["VehAge"]),
        "region": region_segment(frame["Region"]),
    }


def low_support(rows: float, exposure: float) -> bool:
    """Sinalize baixo suporte: < 500 registros ou < 25 apolices-ano."""
    return bool(rows < LOW_SUPPORT_ROWS or exposure < LOW_SUPPORT_EXPOSURE)


def segment_table(frame: pd.DataFrame, with_outcome: bool) -> pd.DataFrame:
    """Tabela por segmento de cada familia, com linha de total.

    Args:
        frame: linhas da particao (treino ou validacao).
        with_outcome: inclui contagens e taxa observada. Deve ser
            ``False`` para qualquer particao cujo desfecho nao possa ser
            visto.

    Returns:
        Colunas ``family``, ``segment``, ``rows``, ``exposure``, opcional
        ``claims`` e ``observed_rate_per_100``, e ``low_support``.
    """
    rows = []
    for family, seg in segment_columns(frame).items():
        data = pd.DataFrame({"segment": seg.to_numpy(),
                             "exposure": frame["Exposure"].to_numpy()})
        if with_outcome:
            data["claims"] = frame["ClaimNb"].to_numpy()
        fixed = _FIXED_ORDER.get(family)
        observed = sorted(set(data["segment"]) - set(fixed or []))
        for name in (fixed + observed if fixed else observed):
            part = data[data["segment"] == name]
            row = {"family": family, "segment": name,
                   "rows": int(len(part)),
                   "exposure": float(part["exposure"].sum())}
            if with_outcome:
                row["claims"] = float(part["claims"].sum())
                row["observed_rate_per_100"] = safe_ratio(
                    100 * row["claims"], row["exposure"])
            row["low_support"] = low_support(row["rows"], row["exposure"])
            rows.append(row)
    return pd.DataFrame(rows)


def sort_segments(table: pd.DataFrame) -> pd.DataFrame:
    """Ordene por familia e pela ordem natural dos segmentos.

    Faixas de exposicao e de idade seguem a ordem declarada; regioes, a
    ordem alfabetica do codigo.
    """
    def position(row) -> tuple[int, int, str]:
        fixed = _FIXED_ORDER.get(row["family"])
        inside = fixed.index(row["segment"]) if fixed and (
            row["segment"] in fixed) else 0
        return (FAMILIES.index(row["family"]), inside, str(row["segment"]))

    keys = table.apply(position, axis=1)
    return table.iloc[sorted(range(len(table)), key=lambda i: keys.iloc[i])
                      ].reset_index(drop=True)


def reconcile_segments(table: pd.DataFrame, frame: pd.DataFrame,
                       with_outcome: bool) -> pd.DataFrame:
    """Reconcilie cada familia com os totais da particao.

    Returns:
        Uma linha por familia e indicador, com ``ok`` booleano.
    """
    totals = {"rows": float(len(frame)),
              "exposure": float(frame["Exposure"].sum())}
    if with_outcome:
        totals["claims"] = float(frame["ClaimNb"].sum())
    out = []
    for family, part in table.groupby("family"):
        for column, total in totals.items():
            value = float(part[column].sum())
            out.append({"family": family, "indicator": column,
                        "segments_sum": value, "partition_total": total,
                        "ok": bool(np.isclose(value, total, rtol=0,
                                              atol=1e-6))})
    return pd.DataFrame(out)


__all__ = ["BAND_LABELS", "INVALID_BAND", "segment_table", "segment_columns",
           "reconcile_segments", "vehicle_age_band", "region_segment",
           "low_support"]
