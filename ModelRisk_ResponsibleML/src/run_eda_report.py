"""Executa a EDA tabular inicial e salva as tabelas de análise."""

from pathlib import Path

from adult_data import load_adult
from eda_report import write_eda_outputs


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Carrega os dados brutos e gera os artefatos tabulares da EDA."""
    frame = load_adult(ROOT / "data" / "raw" / "adult_uci")
    outputs = write_eda_outputs(frame, ROOT / "data" / "processed" / "eda_initial")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
