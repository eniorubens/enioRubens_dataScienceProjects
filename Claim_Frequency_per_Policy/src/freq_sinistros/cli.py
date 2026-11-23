"""Interface de linha de comando de todas as etapas do projeto.

Aquisicao e auditoria (etapa 1), particao (2), EDA e comparacao (3),
congelamento e sua verificacao (4), portao de integridade, avaliacao
oficial unica e verificacao offline (5).

Codigos de saida: 0 sucesso; 1 falha de verificacao; 2 auditoria
divergente; 3 aquisicao recusada; 4 ``stage5`` recusado; 5
``check-stage5`` ou ``verify-freeze`` recusado apos a trava.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos da CLI."""
    parser = argparse.ArgumentParser(prog="freq_sinistros")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="raiz do projeto (padrao: diretorio atual)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("acquire", help="baixa o ARFF e grava o manifesto")
    sub.add_parser("audit", help="audita o ARFF local (sem rede)")
    sub.add_parser("partition",
                   help="materializa as particoes da etapa 2 (sem rede)")
    sub.add_parser("eda", help="EDA descritiva so com o treino (sem rede)")
    sub.add_parser("compare",
                   help="taxa global, Poisson e NB2 em validacao")
    sub.add_parser("stage4", help="diagnostico de zeros, decisao e "
                   "congelamento (sem rede)")
    sub.add_parser("verify-freeze", help="restaura o artefato congelado e "
                   "reproduz a validacao aprovada (recusado apos a trava)")
    sub.add_parser("check-stage5", help="portao de integridade da etapa 5 "
                   "antes da avaliacao oficial (recusado apos a trava)")
    sub.add_parser("stage5", help="avaliacao oficial unica no teste "
                   "reservado")
    sub.add_parser("verify-stage5", help="verifica offline os agregados "
                   "da execucao oficial, a trava, o modelo e o roteiro")
    sub.add_parser("verify-partition",
                   help="recria as atribuicoes e compara o hash")
    return parser


def verify_stage5_full(root: Path) -> int:
    """Verificacao offline completa da avaliacao oficial (somente leitura).

    Combina ``stage5.verify_official`` (registro, hashes, reconciliacao e
    arquivos listados) com ``artifacts.verify_stage5`` (trava, modelo e
    roteiro contra o registro, portao e numero de registros do teste).
    Nao le o ARFF nem o teste.

    Args:
        root: raiz do projeto.

    Returns:
        0 se as duas verificacoes passarem; 1 se alguma falhar ou se faltar
        ``official_run.json``, a lista de checksums ou um arquivo listado.
    """
    import json

    from freq_sinistros import artifacts, stage5

    try:
        code = stage5.verify_official(root)
        full = artifacts.verify_stage5(root)
    except (FileNotFoundError, KeyError, json.JSONDecodeError,
            ValueError) as error:
        print(f"ERRO: verificacao impossivel ({type(error).__name__}: "
              f"{error})")
        return 1
    print("verify-stage5 (conferencia completa) ok:", full["all_ok"],
          full["checks"])
    return 0 if code == 0 and full["all_ok"] else 1


def _locked_refusal(root: Path, command: str) -> int | None:
    """Recusa ``command`` se a avaliacao oficial ja foi travada.

    Os dois comandos protegidos regravariam arquivos listados em checksums
    selados (``integrity_gate.json`` e ``verify_freeze.json``).

    Args:
        root: raiz do projeto.
        command: nome do subcomando, usado na mensagem.

    Returns:
        5 se ``official_run.lock`` existir (mensagem impressa); ``None`` se
        o comando puder prosseguir.
    """
    from freq_sinistros import stage5

    lock = Path(root) / stage5.OUT5 / stage5.LOCK_NAME
    if not lock.exists():
        return None
    print(f"ERRO: a avaliacao oficial ja foi concluida ({lock.name} "
          f"existe); {command} nao e mais executado porque regravaria "
          "artefatos selados. Use verify-stage5 (ou, para o "
          "congelamento, uma copia isolada sem outputs/etapa_05).")
    return 5


def main(argv: list[str] | None = None) -> int:
    """Executa o subcomando solicitado.

    Args:
        argv: argumentos; ``None`` usa ``sys.argv``.

    Returns:
        Codigo de saida: 0 em sucesso; 3 se a aquisicao for recusada
        (MD5 divergente ou bruto ja existente); 2 se a auditoria
        encontrar o bruto divergente do manifesto; 4 se a avaliacao oficial
        for recusada (integridade ou execucao ja realizada); 5 se
        ``check-stage5`` ou ``verify-freeze`` for recusado porque
        ``official_run.lock`` existe (avaliacao oficial concluida; use
        ``verify-stage5``, que devolve 1 em vez de excecao quando falta
        registro, lista ou arquivo listado).
    """
    args = build_parser().parse_args(argv)
    if args.command == "acquire":
        from freq_sinistros.acquire import ChecksumMismatchError, acquire

        try:
            manifest = acquire(args.root)
        except (ChecksumMismatchError, FileExistsError) as error:
            print(f"ERRO: {error}")
            return 3
        art = manifest["artifact"]
        print(f"arquivo: {art['local_path']}")
        print(f"bytes: {art['size_bytes']}")
        print(f"sha256: {art['sha256']}")
        print("md5 confere:", manifest["checksum_verification"]["match"])
        return 0
    if args.command in ("eda", "compare"):
        from freq_sinistros import stage3

        return (stage3.run_eda(args.root) if args.command == "eda"
                else stage3.run_compare(args.root))
    if args.command in ("check-stage5", "stage5", "verify-stage5"):
        from freq_sinistros import stage5

        if args.command == "check-stage5":
            refused = _locked_refusal(args.root, args.command)
            return refused if refused is not None else stage5.check_only(
                args.root)
        if args.command == "verify-stage5":
            return verify_stage5_full(args.root)
        try:
            return stage5.run_official(args.root)
        except (stage5.IntegrityError, stage5.AlreadyEvaluatedError) as error:
            print(f"ERRO: {error}")
            return 4
    if args.command in ("stage4", "verify-freeze"):
        from freq_sinistros import stage4

        if args.command == "stage4":
            return stage4.run_stage4(args.root)
        refused = _locked_refusal(args.root, args.command)
        return refused if refused is not None else stage4.verify_freeze(
            args.root)
    if args.command in ("partition", "verify-partition"):
        from freq_sinistros import design

        if args.command == "partition":
            return design.run_partition(args.root)
        return design.verify_partition(args.root)
    from freq_sinistros.audit import run_audit

    return run_audit(args.root)
