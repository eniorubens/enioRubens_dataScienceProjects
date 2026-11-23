"""Acesso verificado aos dados de desenvolvimento (etapas 3 em diante).

O carregador confere o SHA-256 do ARFF contra o manifesto da etapa 1 e o
hash das atribuicoes contra a configuracao da etapa 2. O leitor interpreta o
ARFF integral (incluindo as linhas de teste) em memoria, mas o carregador
devolve apenas as particoes pedidas e descarta o restante logo apos a
leitura: nenhuma funcao da etapa 3 recebe linhas de teste.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from freq_sinistros.acquire import hash_file
from freq_sinistros.arff import read_arff
from freq_sinistros.partition import assignment_hash


def load_partitions(root: Path, keep: tuple[str, ...] = ("train",
                    "validation")) -> tuple[dict[str, pd.DataFrame], dict]:
    """Leia o ARFF e devolva apenas as particoes pedidas.

    Args:
        root: raiz do projeto.
        keep: particoes a manter; ``test`` e recusada.

    Returns:
        Dicionario particao -> tabela e a procedencia verificada (hashes
        do ARFF e das atribuicoes, tamanhos).

    Raises:
        ValueError: se ``keep`` incluir o teste ou se algum hash divergir.
    """
    root = Path(root)
    if "test" in keep:
        raise ValueError("o teste permanece reservado ate a etapa 5")
    manifest = json.loads((root / "outputs" / "etapa_01"
                           / "manifest.json").read_text("utf-8"))
    config = json.loads((root / "outputs" / "etapa_02"
                         / "partition_config.json").read_text("utf-8"))
    raw = root / manifest["artifact"]["local_path"]
    digest = hash_file(raw)
    if digest["sha256"] != manifest["artifact"]["sha256"]:
        raise ValueError("ARFF diverge do manifesto da etapa 1")
    assign_path = root / config["assignment_file"]["path"]
    assignments = pd.read_csv(assign_path)
    expected = config["assignment_file"]["sha256_of_ordered_assignments"]
    if assignment_hash(assignments) != expected:
        raise ValueError("atribuicoes divergem da configuracao da etapa 2")
    frame = read_arff(raw).frame
    part = assignments.set_index("IDpol")["partition"].reindex(
        frame["IDpol"]).to_numpy()
    if pd.isna(part).any():
        raise ValueError("IDpol sem particao")
    out = {p: frame[part == p].reset_index(drop=True) for p in keep}
    sizes = {p: int(len(t)) for p, t in out.items()}
    expected_sizes = {r["partition"]: r["ids"] for r in config["sizes"]}
    for p, n in sizes.items():
        if n != expected_sizes[p]:
            raise ValueError(f"tamanho de {p} diverge da configuracao")
    provenance = {
        "arff_sha256": digest["sha256"],
        "arff_size_bytes": digest["size_bytes"],
        "assignment_sha256": expected,
        "partition_rule": config["rule"],
        "sizes": sizes,
        "protocol_sha256": hashlib.sha256(
            (root / "outputs" / "etapa_02" / "protocol_config.json")
            .read_bytes()).hexdigest(),
    }
    del frame
    return out, provenance
