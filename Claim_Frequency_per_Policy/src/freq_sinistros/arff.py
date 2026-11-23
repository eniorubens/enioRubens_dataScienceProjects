"""Leitor incremental de ARFF com diagnostico de conversao.

O leitor le o artefato bruto sem altera-lo. Todas as colunas entram como
texto; a conversao de tipos e feita por bloco, com contagem explicita de
valores ausentes, nao numericos e fora do dominio declarado. Nada e
descartado: toda linha de dados do arquivo aparece no resultado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_TOKEN = "?"
CHUNK_ROWS = 100_000


@dataclass
class Attribute:
    """Atributo declarado no cabecalho ARFF.

    Attributes:
        name: nome da coluna.
        arff_type: ``numeric``, ``string`` ou ``nominal``.
        levels: niveis declarados (apenas para ``nominal``).
    """

    name: str
    arff_type: str
    levels: list[str] | None = None


@dataclass
class ArffHeader:
    """Cabecalho ARFF interpretado."""

    relation: str
    attributes: list[Attribute]
    data_line: int  # indice (base 0) da primeira linha apos ``@data``

    @property
    def names(self) -> list[str]:
        """Nomes das colunas na ordem do arquivo."""
        return [a.name for a in self.attributes]


@dataclass
class ReadResult:
    """Resultado da leitura tipada.

    Attributes:
        frame: tabela com tipos finais (numericas float64, nominais
            categoricas; ``IDpol`` em int64 quando a conversao e exata).
        header: cabecalho interpretado.
        diagnostics: contagens de leitura e conversao.
    """

    frame: pd.DataFrame
    header: ArffHeader
    diagnostics: dict = field(default_factory=dict)


def _split_nominal(spec: str) -> list[str]:
    """Extrai os niveis de uma declaracao ``{'a','b'}``."""
    body = spec.strip()[1:-1]
    return [t.strip().strip("'\"") for t in body.split(",")]


def parse_header(path: Path) -> ArffHeader:
    """Interpreta o cabecalho de um arquivo ARFF.

    Args:
        path: caminho do arquivo.

    Returns:
        O cabecalho com relacao, atributos e posicao dos dados.

    Raises:
        ValueError: se nao houver a secao ``@data``.
    """
    relation, attributes = "", []
    with open(path, encoding="utf-8", newline="") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            lower = text.lower()
            if lower.startswith("@relation"):
                relation = text.split(None, 1)[1].strip("'\"")
            elif lower.startswith("@attribute"):
                _, name, spec = text.split(None, 2)
                name = name.strip("'\"")
                if spec.startswith("{"):
                    attributes.append(
                        Attribute(name, "nominal", _split_nominal(spec)))
                else:
                    attributes.append(Attribute(name, spec.strip().lower()))
            elif lower.startswith("@data"):
                return ArffHeader(relation, attributes, index + 1)
    raise ValueError("secao @data ausente")


def count_data_lines(path: Path, data_line: int) -> int:
    """Conta as linhas de dados nao vazias e que nao sao comentario."""
    total = 0
    with open(path, encoding="utf-8", newline="") as handle:
        for index, line in enumerate(handle):
            if index < data_line:
                continue
            text = line.strip()
            if text and not text.startswith("%"):
                total += 1
    return total


def _convert_numeric(raw: pd.Series) -> tuple[pd.Series, dict]:
    """Converta texto em float64 contando ausentes e nao numericos."""
    missing = raw.isna()
    parsed = pd.to_numeric(raw, errors="coerce").astype("float64")
    non_numeric = parsed.isna() & ~missing
    # 'nan' textual converte sem erro e produz NaN: conta como nao finito.
    return parsed, {
        "missing": int(missing.sum()),
        "non_numeric": int(non_numeric.sum()),
    }


def read_arff(path: Path, chunk_rows: int = CHUNK_ROWS) -> ReadResult:
    """Le um ARFF de forma incremental e tipa as colunas.

    Args:
        path: caminho do arquivo ARFF.
        chunk_rows: linhas por bloco de leitura.

    Returns:
        ``ReadResult`` com a tabela e as contagens de diagnostico.

    Raises:
        ValueError: se o cabecalho nao for valido.
    """
    path = Path(path)
    header = parse_header(path)
    names = header.names
    numeric = [a.name for a in header.attributes if a.arff_type == "numeric"]
    nominal = {a.name: a for a in header.attributes
               if a.arff_type == "nominal"}
    strings = [a.name for a in header.attributes if a.arff_type == "string"]
    diag = {
        "numeric": {n: {"missing": 0, "non_numeric": 0} for n in numeric},
        "out_of_domain": {n: 0 for n in nominal},
        "nominal_missing": {n: 0 for n in nominal},
        "string_missing": {n: 0 for n in strings},
        "id_roundtrip_mismatch": 0,
        "id_non_integer_or_unsafe": 0,
        "id_noncanonical_tokens": [],
        "chunks": 0,
    }
    frames = []
    reader = pd.read_csv(
        path, header=None, names=names, skiprows=header.data_line,
        dtype=str, na_values=[MISSING_TOKEN], keep_default_na=False,
        quotechar="'", skipinitialspace=True, chunksize=chunk_rows,
        engine="c", on_bad_lines="error",
    )
    for chunk in reader:
        diag["chunks"] += 1
        out = pd.DataFrame(index=chunk.index)
        for name in names:
            raw = chunk[name]
            if name in diag["numeric"]:
                parsed, counts = _convert_numeric(raw)
                for key, value in counts.items():
                    diag["numeric"][name][key] += value
                out[name] = parsed
                if name == "IDpol":
                    ok = parsed.notna() & np.isfinite(parsed)
                    integral = ok & (parsed == np.floor(parsed)) & (
                        parsed.abs() < 2**53)
                    diag["id_non_integer_or_unsafe"] += int(
                        (~integral).sum())
                    canon = parsed[integral].astype("int64").astype(str)
                    differs = canon != raw[integral]
                    diag["id_roundtrip_mismatch"] += int(differs.sum())
                    for token in raw[integral][differs].head(20):
                        if len(diag["id_noncanonical_tokens"]) < 20:
                            diag["id_noncanonical_tokens"].append(token)
            elif name in nominal:
                levels = nominal[name].levels
                diag["nominal_missing"][name] += int(raw.isna().sum())
                diag["out_of_domain"][name] += int(
                    (raw.notna() & ~raw.isin(levels)).sum())
                out[name] = pd.Categorical(raw, categories=levels)
            else:
                diag["string_missing"][name] += int(raw.isna().sum())
                out[name] = raw.astype("category")
        frames.append(out)
    frame = pd.concat(frames, ignore_index=True)
    for name in strings:
        frame[name] = frame[name].astype("category")
    if "IDpol" in frame and diag["id_non_integer_or_unsafe"] == 0 and (
            frame["IDpol"].notna().all()):
        frame["IDpol"] = frame["IDpol"].astype("int64")
    diag["raw_data_lines"] = count_data_lines(path, header.data_line)
    diag["parsed_rows"] = int(len(frame))
    diag["rows_rejected"] = diag["raw_data_lines"] - diag["parsed_rows"]
    return ReadResult(frame, header, diag)
