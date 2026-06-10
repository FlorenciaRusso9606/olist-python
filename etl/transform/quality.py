"""
transform/quality.py
--------------------
Profiling y validación de calidad de datos.

Dos niveles de severidad:
  - warn  → se loguea como WARNING, el pipeline continúa
  - error → se loguea como ERROR, lanza DataQualityError al final del check

Dos puntos de control:
  - validate_clean(clean)       → después de raw → clean
  - validate_dwh(dwh, clean)    → después de clean → dwh
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    pass


@dataclass
class QualityIssue:
    table: str
    check: str
    detail: str
    level: str  # "warn" | "error"


# ─────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────

def _null_rate(series: pd.Series) -> tuple[int, float]:
    n = int(series.isna().sum())
    pct = n / len(series) * 100 if len(series) else 0.0
    return n, pct


def _profile(name: str, df: pd.DataFrame, null_cols: list[str] | None = None) -> None:
    """Loguea row count y tasa de nulos para las columnas indicadas."""
    logger.info("  [quality] %-32s %s filas", name, f"{len(df):,}")
    for col in null_cols or []:
        if col not in df.columns:
            continue
        n, pct = _null_rate(df[col])
        if n > 0:
            logger.warning("    %-32s nulos: %6d (%5.1f%%)", col, n, pct)
        else:
            logger.info("    %-32s nulos: %6d (%5.1f%%)", col, n, pct)


def _numeric_summary(label: str, series: pd.Series) -> None:
    s = series.dropna()
    if s.empty:
        return
    logger.info(
        "    %-32s min=%s  max=%s  mean=%s",
        label,
        f"{s.min():,.2f}",
        f"{s.max():,.2f}",
        f"{s.mean():,.2f}",
    )


def _check_empty(table: str, df: pd.DataFrame) -> list[QualityIssue]:
    if len(df) == 0:
        return [QualityIssue(table, "row_count", "tabla vacía", "error")]
    return []


def _check_range(
    table: str,
    series: pd.Series,
    min_val: float | None = None,
    max_val: float | None = None,
) -> list[QualityIssue]:
    issues = []
    s = series.dropna()
    if min_val is not None:
        n = int((s < min_val).sum())
        if n:
            issues.append(QualityIssue(
                table, "range",
                f"{series.name} < {min_val}: {n:,} filas", "warn",
            ))
    if max_val is not None:
        n = int((s > max_val).sum())
        if n:
            issues.append(QualityIssue(
                table, "range",
                f"{series.name} > {max_val}: {n:,} filas", "warn",
            ))
    return issues


def _check_fk(
    child_table: str,
    child_col: pd.Series,
    parent_col: pd.Series,
) -> list[QualityIssue]:
    """Verifica que todos los valores de child_col existan en parent_col."""
    orphans = int((~child_col.isin(set(parent_col))).sum())
    if orphans:
        return [QualityIssue(
            child_table, "referential_integrity",
            f"{child_col.name}: {orphans:,} valores sin match en {parent_col.name}",
            "error",
        )]
    return []


# ─────────────────────────────────────────────
# Puntos de control del pipeline
# ─────────────────────────────────────────────

def validate_clean(clean: dict[str, pd.DataFrame]) -> list[QualityIssue]:
    """
    Perfila y valida los DataFrames clean.*.
    Devuelve la lista de issues encontrados (sin lanzar excepciones).
    Llamar a report() después para loguear y lanzar si hay errores.
    """
    logger.info("  --- Profiling clean.* ---")
    issues: list[QualityIssue] = []

    # ── Perfiles de nulos ──────────────────────────────────────────────
    _profile("clean.orders", clean["orders"], [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    _profile("clean.order_items", clean["order_items"], [
        "price", "freight_value", "product_id", "seller_id",
    ])
    _profile("clean.order_payments", clean["order_payments"], [
        "payment_value", "payment_installments",
    ])
    _profile("clean.order_reviews", clean["order_reviews"], ["review_score"])
    _profile("clean.products", clean["products"], [
        "product_category_name", "product_weight_g",
    ])
    _profile("clean.customers", clean["customers"], ["customer_city", "customer_state"])
    _profile("clean.sellers",   clean["sellers"],   ["seller_city",   "seller_state"])

    # ── Tablas vacías (error crítico) ──────────────────────────────────
    for name, df in clean.items():
        issues += _check_empty(f"clean.{name}", df)

    # ── Rangos de valores ──────────────────────────────────────────────
    oi = clean["order_items"]
    issues += _check_range("clean.order_items", oi["price"],         min_val=0)
    issues += _check_range("clean.order_items", oi["freight_value"], min_val=0)

    op = clean["order_payments"]
    issues += _check_range("clean.order_payments", op["payment_value"],        min_val=0)
    issues += _check_range("clean.order_payments", op["payment_installments"], min_val=1)

    rev = clean["order_reviews"]
    issues += _check_range("clean.order_reviews", rev["review_score"], min_val=1, max_val=5)

    # ── Integridad referencial ─────────────────────────────────────────
    issues += _check_fk("clean.order_items",    oi["order_id"], clean["orders"]["order_id"])
    issues += _check_fk("clean.order_payments", op["order_id"], clean["orders"]["order_id"])

    return issues


def validate_dwh(
    dwh: dict[str, pd.DataFrame],
    clean: dict[str, pd.DataFrame],
) -> list[QualityIssue]:
    """
    Perfila y valida los DataFrames dwh.*.
    Recibe también los clean para calcular la tasa de retención.
    """
    logger.info("  --- Profiling dwh.* ---")
    issues: list[QualityIssue] = []

    for name, df in dwh.items():
        _profile(f"dwh.{name}", df)

    fact = dwh["fact_sales"]
    issues += _check_empty("dwh.fact_sales", fact)

    if len(fact) == 0:
        return issues  # no tiene sentido seguir validando

    # ── Nulos en columnas clave de fact ────────────────────────────────
    key_cols = ["order_id", "order_item_id", "customer_id", "product_id", "seller_id", "date_id"]
    for col in key_cols:
        n, pct = _null_rate(fact[col])
        if n:
            issues.append(QualityIssue(
                "dwh.fact_sales", "null_key",
                f"{col}: {n:,} nulos ({pct:.1f}%)", "error",
            ))

    # ── Rangos de medidas ──────────────────────────────────────────────
    issues += _check_range("dwh.fact_sales", fact["item_price"],              min_val=0)
    issues += _check_range("dwh.fact_sales", fact["payment_value_allocated"], min_val=0)
    issues += _check_range("dwh.fact_sales", fact["freight_value"],           min_val=0)

    # ── Integridad referencial dims ────────────────────────────────────
    issues += _check_fk("dwh.fact_sales", fact["customer_id"], dwh["dim_customer"]["customer_id"])
    issues += _check_fk("dwh.fact_sales", fact["product_id"],  dwh["dim_product"]["product_id"])
    issues += _check_fk("dwh.fact_sales", fact["date_id"],     dwh["dim_date"]["date_id"])

    # ── Resumen de métricas clave ──────────────────────────────────────
    _numeric_summary("item_price",              fact["item_price"])
    _numeric_summary("payment_value_allocated", fact["payment_value_allocated"])

    # ── Tasa de retención order_items → fact_sales ─────────────────────
    # Si cae por debajo del 80% algo raro pasó en los joins o filtros
    n_items = len(clean["order_items"])
    n_fact  = len(fact)
    retention = n_fact / n_items * 100 if n_items else 0.0
    logger.info(
        "  [quality] retención order_items → fact_sales: %s/%s (%.1f%%)",
        f"{n_fact:,}", f"{n_items:,}", retention,
    )
    if retention < 80:
        issues.append(QualityIssue(
            "dwh.fact_sales", "retention",
            f"solo {retention:.1f}% de order_items llegaron a fact_sales", "warn",
        ))

    return issues


def report(issues: list[QualityIssue]) -> None:
    """
    Loguea todos los issues encontrados.
    Lanza DataQualityError si alguno es de nivel 'error'.
    """
    if not issues:
        logger.info("  [quality] ✓ Sin issues")
        return

    errors = []
    for issue in issues:
        msg = f"[{issue.table}] {issue.check}: {issue.detail}"
        if issue.level == "error":
            logger.error("  ✗ %s", msg)
            errors.append(msg)
        else:
            logger.warning("  ⚠ %s", msg)

    if errors:
        raise DataQualityError(
            f"{len(errors)} validación/es fallida/s:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
