"""Audit categorical values unseen in training, per development fold and in the F1 test window.

The shared preprocessor (``ecom_delivery_sla.modeling.make_preprocessor``) one-hot encodes the
categorical features with ``OneHotEncoder(drop="first", handle_unknown="ignore")``. A value that
did not appear in training is therefore encoded as an all-zero vector, which is the same encoding
as the dropped reference category (the first category in sorted order). This script measures how
often that happens, which reference category it collides with, and whether those rows are flagged
by the network cold-start indicators. It reads the raw CSVs only; no model is trained and the
encoder is not changed (the test period was opened with it, see P-07 and T-11 in
``docs/REGISTRO-DE-DECISOES.md``).

Output: ``reports/unknown_categories.csv`` (one row per split and categorical feature).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("reports/unknown_categories.csv")
COLD_START_COLUMNS = ("seller_cold_start", "edge_cold_start")


def unknown_category_stats(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    split: str,
    target: str = "sla_breach",
) -> pd.DataFrame:
    """Count evaluation rows whose categorical value is absent from the training rows.

    Missing values are filled with the training mode first, as the pipeline's
    ``SimpleImputer(strategy="most_frequent")`` does, so they never count as unknown.

    Parameters
    ----------
    train, evaluation : pandas.DataFrame
        Training and evaluation partitions (not modified).
    columns : list of str
        Categorical feature names.
    split : str
        Label of the evaluation partition.
    target : str, default "sla_breach"
        Binary target used for the breach rate of the unknown rows.

    Returns
    -------
    pandas.DataFrame
        One row per feature with the unknown count and share, the unseen values,
        the dropped reference category, its row count in the evaluation partition,
        the breach rate of the unknown rows and the share of them flagged as cold start.
    """
    rows = []
    for column in columns:
        mode = train[column].mode(dropna=True).iloc[0]
        seen_values = train[column].fillna(mode).astype(str)
        values = evaluation[column].fillna(mode).astype(str)
        seen = set(seen_values.unique())
        reference = sorted(seen)[0]
        unknown = ~values.isin(seen)
        subset = evaluation.loc[unknown.to_numpy()]
        row = {
            "split": split,
            "feature": column,
            "n_evaluation": len(values),
            "unknown_rows": int(unknown.sum()),
            "unknown_share": float(unknown.mean()) if len(values) else float("nan"),
            "unknown_values": ", ".join(sorted(values[unknown].unique())),
            "reference_category": reference,
            "reference_rows_train": int((seen_values == reference).sum()),
            "reference_rows_evaluation": int((values == reference).sum()),
            "breach_rate_unknown": float(subset[target].mean()) if len(subset) and target in subset else float("nan"),
        }
        for flag in COLD_START_COLUMNS:
            row[f"{flag}_share_unknown"] = (
                float(subset[flag].mean()) if len(subset) and flag in subset else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def audit(root: Path = ROOT) -> pd.DataFrame:
    """Build the modeling frame from the raw CSVs and audit every fold and the F1 test window."""
    sys.path.insert(0, str(root / "src"))
    from ecom_delivery_sla import (  # noqa: PLC0415
        build_order_features,
        build_supervised_orders,
        read_olist_tables,
    )
    from ecom_delivery_sla.modeling import CATEGORICAL_FEATURES, build_modeling_frame  # noqa: PLC0415
    from ecom_delivery_sla.temporal import expanding_window_folds  # noqa: PLC0415

    tables = read_olist_tables(
        root / "data" / "raw" / "olist" / "csv",
        table_names=("orders", "items", "products", "payments", "customers", "sellers", "geolocation"),
    )
    supervised = build_supervised_orders(tables["orders"])
    features = build_order_features(
        orders=tables["orders"],
        items=tables["items"],
        products=tables["products"],
        payments=tables["payments"],
        customers=tables["customers"],
        sellers=tables["sellers"],
        geolocation=tables["geolocation"],
    )
    frame = build_modeling_frame(features, supervised)
    folds = expanding_window_folds(
        frame,
        timestamp_col="order_purchase_timestamp",
        first_val_month="2017-10",
        last_val_month="2018-06",
        embargo_days=30,
    )
    parts = [
        unknown_category_stats(fold["train"], fold["validation"], CATEGORICAL_FEATURES, f"fold {fold['fold']}")
        for fold in folds
    ]
    purchase = pd.to_datetime(frame["order_purchase_timestamp"])
    parts.append(
        unknown_category_stats(
            frame[purchase < "2018-06-01"],
            frame[(purchase >= "2018-07-01") & (purchase < "2018-09-01")],
            CATEGORICAL_FEATURES,
            "F1 test",
        )
    )
    return pd.concat(parts, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    """Run the audit and write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    table = audit(arguments.root)
    output = arguments.root / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, lineterminator="\n")
    flagged = table[table["unknown_rows"] > 0]
    print(flagged[["split", "feature", "unknown_rows", "unknown_values", "reference_category",
                   "reference_rows_evaluation", "seller_cold_start_share_unknown"]].to_string(index=False))
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
