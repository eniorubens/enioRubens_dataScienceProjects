"""Particao deterministica por ``IDpol`` (etapa 2).

A alocacao depende apenas do identificador canonico: nao ha estado, ordem
de linhas nem semente aleatoria. Regra fixa:

``bucket = int.from_bytes(sha256("freMTPL2freq-v1-etapa2|<IDpol>")[:8],
"big") % 10000``; ``[0, 7000)`` treino, ``[7000, 8500)`` validacao,
``[8500, 10000)`` teste.
"""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal, InvalidOperation

import pandas as pd

RULE_PREFIX = "freMTPL2freq-v1-etapa2|"
MODULUS = 10_000
TRAIN_END = 7_000
VALIDATION_END = 8_500
PARTITIONS = ("train", "validation", "test")
RULE_TEXT = (
    "bucket = int.from_bytes(sha256(utf8('freMTPL2freq-v1-etapa2|' + "
    "str(IDpol)))[:8], 'big') % 10000; train [0,7000), validation "
    "[7000,8500), test [8500,10000)"
)


def canonical_id(value) -> int:
    """Converta um identificador em inteiro canonico, sem perda.

    Aceita inteiros, ``float`` integrais e texto (inclusive notacao
    cientifica como ``1e+05``, lida exatamente via ``Decimal``).

    Args:
        value: identificador lido do ARFF.

    Returns:
        O inteiro correspondente.

    Raises:
        ValueError: se o valor for ausente, nao finito, fracionario,
            negativo ou nao numerico.
    """
    if value is None or isinstance(value, bool):
        raise ValueError(f"IDpol invalido: {value!r}")
    try:
        if isinstance(value, str):
            number = Decimal(value.strip())
        elif isinstance(value, float):
            if not math.isfinite(value) or abs(value) >= 2**53:
                raise ValueError
            number = Decimal(value)
        else:
            number = Decimal(int(value))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        raise ValueError(f"IDpol invalido: {value!r}") from None
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"IDpol nao inteiro: {value!r}")
    result = int(number)
    if result < 0:
        raise ValueError(f"IDpol negativo: {value!r}")
    return result


def bucket(identifier: int) -> int:
    """Devolva o bucket 0-9999 do identificador canonico."""
    digest = hashlib.sha256(
        f"{RULE_PREFIX}{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % MODULUS


def partition_of(identifier: int) -> str:
    """Devolva a particao (``train``, ``validation`` ou ``test``)."""
    b = bucket(identifier)
    if b < TRAIN_END:
        return "train"
    return "validation" if b < VALIDATION_END else "test"


def validate_ids(ids: pd.Series) -> pd.Series:
    """Valide e canonize os identificadores antes de particionar.

    Args:
        ids: coluna ``IDpol``.

    Returns:
        Serie ``int64`` com os identificadores canonicos.

    Raises:
        ValueError: se houver ausente, invalido ou repetido.
    """
    canon = pd.Series([canonical_id(v) for v in ids], index=ids.index,
                      dtype="int64")
    duplicated = canon[canon.duplicated(keep=False)]
    if len(duplicated):
        raise ValueError(
            f"{duplicated.nunique()} IDpol repetidos; "
            f"exemplo: {int(duplicated.iloc[0])}")
    return canon


def build_assignments(ids: pd.Series) -> pd.DataFrame:
    """Calcule a atribuicao de todos os identificadores.

    O resultado e ordenado por ``IDpol``; por isso nao depende da ordem de
    entrada.

    Args:
        ids: coluna ``IDpol`` (valida e unica).

    Returns:
        Tabela com ``IDpol`` (``int64``) e ``partition``.
    """
    canon = validate_ids(ids)
    out = pd.DataFrame({"IDpol": canon.to_numpy()})
    out = out.sort_values("IDpol", kind="stable").reset_index(drop=True)
    out["partition"] = [partition_of(int(i)) for i in out["IDpol"]]
    return out


def assignment_bytes(assignments: pd.DataFrame) -> bytes:
    """Serialize as atribuicoes ordenadas em CSV canonico (UTF-8, LF)."""
    ordered = assignments.sort_values("IDpol", kind="stable")
    return ordered[["IDpol", "partition"]].to_csv(
        index=False, lineterminator="\n").encode("utf-8")


def assignment_hash(assignments: pd.DataFrame) -> str:
    """SHA-256 das atribuicoes ordenadas (``IDpol,partition``)."""
    return hashlib.sha256(assignment_bytes(assignments)).hexdigest()


def partition_sizes(assignments: pd.DataFrame) -> pd.DataFrame:
    """Tamanho e proporcao de cada particao (somente contagem de IDs)."""
    counts = assignments["partition"].value_counts()
    total = int(len(assignments))
    rows = [{"partition": p, "ids": int(counts.get(p, 0)),
             "share": counts.get(p, 0) / total if total else float("nan")}
            for p in PARTITIONS]
    return pd.DataFrame(rows)


def check_partition(assignments: pd.DataFrame, ids: pd.Series) -> dict:
    """Verifique cobertura integral, disjuncao e unicidade.

    Args:
        assignments: atribuicoes calculadas.
        ids: identificadores da populacao.

    Returns:
        Dicionario de verificacoes booleanas e contagens.
    """
    canon = set(validate_ids(ids).tolist())
    assigned = assignments["IDpol"]
    by_part = {p: set(assigned[assignments["partition"] == p].tolist())
               for p in PARTITIONS}
    union = set().union(*by_part.values())
    overlap = sum(len(by_part[a] & by_part[b])
                  for i, a in enumerate(PARTITIONS)
                  for b in PARTITIONS[i + 1:])
    return {
        "population_ids": len(canon),
        "assigned_rows": int(len(assignments)),
        "assigned_ids_unique": bool(assigned.is_unique),
        "full_coverage": union == canon,
        "overlap_ids": int(overlap),
        "disjoint": overlap == 0,
        "sizes_sum_to_population": sum(map(len, by_part.values()))
        == len(canon),
    }
