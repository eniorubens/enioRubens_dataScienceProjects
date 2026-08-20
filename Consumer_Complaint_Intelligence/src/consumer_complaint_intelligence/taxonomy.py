"""Versioned product-family taxonomy for CFPB complaint routing."""

from dataclasses import dataclass
from typing import Mapping


TAXONOMY_VERSION = "cfpb-product-family-v1.0.0"
"""Stable identifier for the conservative product-family registry."""


PRODUCT_FAMILY_REGISTRY: Mapping[str, str] = {
    "Credit reporting": "credit_reporting",
    "Credit reporting, credit repair services, or other personal consumer "
    "reports": "credit_reporting",
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Debt collection": "debt_collection",
    "Mortgage": "mortgage",
    "Bank account or service": "deposit_accounts",
    "Checking or savings account": "deposit_accounts",
    "Credit card": "cards_prepaid",
    "Credit card or prepaid card": "cards_prepaid",
    "Prepaid card": "cards_prepaid",
    "Money transfers": "money_services",
    "Virtual currency": "money_services",
    "Money transfer, virtual currency, or money service": "money_services",
    "Student loan": "student_loan",
    "Consumer Loan": "consumer_lending",
    "Vehicle loan or lease": "consumer_lending",
    "Payday loan": "consumer_lending",
    "Payday loan, title loan, or personal loan": "consumer_lending",
    "Payday loan, title loan, personal loan, or advance loan": (
        "consumer_lending"
    ),
    "Debt or credit management": "debt_credit_management",
    "Other financial service": "other_financial_services",
}


@dataclass(frozen=True)
class ProductMapping:
    """Represent one raw Product mapping decision.

    Args:
        raw_product: Original Product value supplied to the mapper.
        family: Stable family name, or ``None`` when unmapped.
        mapping_status: ``mapped`` or ``unmapped``.
        taxonomy_version: Version of the registry that made the decision.
    """

    raw_product: str | None
    family: str | None
    mapping_status: str
    taxonomy_version: str = TAXONOMY_VERSION


def map_product(
    raw_product: str | None,
    mode: str = "strict",
) -> ProductMapping:
    """Map one raw Product while making unknown labels explicit.

    Args:
        raw_product: Raw Product label from the CFPB database.
        mode: ``strict`` raises for unknown values; ``unmapped`` returns an
            explicit unmapped result for audit workflows.

    Returns:
        A versioned mapping decision that preserves the raw label.

    Raises:
        ValueError: If ``mode`` is invalid or the label is unknown in strict
            mode.
    """

    if mode not in {"strict", "unmapped"}:
        raise ValueError("mode must be 'strict' or 'unmapped'")

    normalized = raw_product.strip() if raw_product is not None else None
    family = PRODUCT_FAMILY_REGISTRY.get(normalized)
    if family is not None:
        return ProductMapping(raw_product, family, "mapped")
    if mode == "strict":
        raise ValueError(
            f"Unknown Product label for {TAXONOMY_VERSION}: {raw_product!r}"
        )
    return ProductMapping(raw_product, None, "unmapped")


def build_issue_key(product_family: str | None, issue: str | None) -> str:
    """Build the hierarchical key ``product_family + Issue``.

    Args:
        product_family: Stable family name from :func:`map_product`.
        issue: Raw Issue label, which is intentionally not merged.

    Returns:
        A readable hierarchical key with explicit tokens for missing values.
    """

    family_value = product_family or "<UNMAPPED_FAMILY>"
    issue_value = issue.strip() if issue and issue.strip() else "<NULL>"
    return f"{family_value} :: {issue_value}"


def product_family_case_sql(column: str) -> str:
    """Return a DuckDB CASE expression for the versioned registry.

    Args:
        column: Already quoted DuckDB identifier.

    Returns:
        SQL expression returning a family or ``NULL`` for an unmapped label.
    """

    clauses = []
    for label, family in PRODUCT_FAMILY_REGISTRY.items():
        escaped_label = label.replace("'", "''")
        clauses.append(
            f"WHEN trim(CAST({column} AS VARCHAR)) = '{escaped_label}' "
            f"THEN '{family}'"
        )
    return "CASE " + " ".join(clauses) + " ELSE NULL END"


def mapping_status_case_sql(column: str) -> str:
    """Return a DuckDB CASE expression for mapped status."""

    family_sql = product_family_case_sql(column)
    return f"CASE WHEN {family_sql} IS NULL THEN 'unmapped' ELSE 'mapped' END"
