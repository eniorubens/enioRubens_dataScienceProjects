"""Build the cleaned analytical panel: data/interim/*.parquet -> data/processed/panel.parquet.

Usage:
    python scripts/build_panel.py
"""

from retail_multilevel.cleaning import build_panel, cleaning_report
from retail_multilevel.io import load_interim
from retail_multilevel.paths import PROCESSED_DIR


def main() -> None:
    stores, products, tx = (load_interim(n) for n in ("stores", "products", "transactions"))
    panel = build_panel(stores, products, tx)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "panel.parquet"
    panel.to_parquet(out, index=False)
    print(cleaning_report(tx, panel).to_string())
    print(f"-> {out} ({panel.shape[0]:,d} x {panel.shape[1]})")


if __name__ == "__main__":
    main()
