"""Prevê RUL FD001 a partir de um CSV de histórico observado."""
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmapss_final_inference import load_final_bundle, predict_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/models/extra_trees_fd001_v1/model.joblib")
    parser.add_argument("--all-cycles", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Arquivo de saída já existe; escolha outro nome para preservá-lo.")
    prediction = predict_history(pd.read_csv(args.input), load_final_bundle(args.model),
                                 latest_only=not args.all_cycles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(prediction.to_string(index=False))


if __name__ == "__main__":
    main()

