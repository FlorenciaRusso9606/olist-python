"""
transform/dwh.py
----------------
Transforma clean.* → dwh.* (dimensiones + fact_sales)

El orden importa: dimensiones primero, fact_sales al final
porque depende de todas las dimensiones.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_dim_date(orders: pd.DataFrame) -> pd.DataFrame:
    logger.info("  build_dim_date...")

    dates = (
        orders["order_purchase_timestamp"]
        .dropna()
        .dt.normalize()      # trunca al día: 2018-07-04 15:32 → 2018-07-04
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    df = pd.DataFrame({"date_id": dates})
    df["year"]        = df["date_id"].dt.year
    df["quarter"]     = df["date_id"].dt.quarter
    df["month"]       = df["date_id"].dt.month
    df["month_name"]  = df["date_id"].dt.strftime("%B")
    df["week"]        = df["date_id"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date_id"].dt.dayofweek   # 0=lunes (pandas), no 0=domingo (SQL)
    df["day_name"]    = df["date_id"].dt.strftime("%A")
    df["is_weekend"]  = df["day_of_week"].isin([5, 6])

    logger.info("    ✓ %d fechas únicas", len(df))
    return df


def build_dim_customer(customers: pd.DataFrame) -> pd.DataFrame:
    logger.info("  build_dim_customer...")
    df = customers[[
        "customer_id", "customer_unique_id",
        "customer_city", "customer_state", "customer_zip_code_prefix",
    ]].copy()
    logger.info("    ✓ %d filas", len(df))
    return df


def build_dim_product(products: pd.DataFrame) -> pd.DataFrame:
    logger.info("  build_dim_product...")
    df = products.copy()

    all_dims_present = (
        df["product_length_cm"].notna() &
        df["product_height_cm"].notna() &
        df["product_width_cm"].notna()
    )
    df["product_volume_cm3"] = np.where(
        all_dims_present,
        (df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]).round(2),
        np.nan,
    )

    df = df[[
        "product_id", "product_category_name", "product_category_name_english",
        "product_weight_g", "product_volume_cm3",
    ]]
    logger.info("    ✓ %d filas", len(df))
    return df


def build_dim_seller(sellers: pd.DataFrame) -> pd.DataFrame:
    logger.info("  build_dim_seller...")
    df = sellers[[
        "seller_id", "seller_city", "seller_state", "seller_zip_code_prefix",
    ]].copy()
    logger.info("    ✓ %d filas", len(df))
    return df


def build_dim_order(orders: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    logger.info("  build_dim_order...")

    df = pd.merge(
        orders,
        reviews[["order_id", "review_score"]],
        on="order_id",
        how="left",
    )

    delivered = df["order_delivered_customer_date"]
    purchased = df["order_purchase_timestamp"]
    estimated = df["order_estimated_delivery_date"]

    df["delivery_days"]       = (delivered - purchased).dt.days
    df["delivery_delay_days"] = (delivered - estimated).dt.days  # positivo = llegó tarde

    df = df[[
        "order_id", "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "delivery_days", "delivery_delay_days",
        "review_score",
    ]]
    logger.info("    ✓ %d filas", len(df))
    return df


def build_fact_sales(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    order_payments: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_customer: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye fact_sales.

    El pago total de una orden se distribuye proporcionalmente al precio
    de cada ítem:
      payment_value_allocated = total_payment × (item_price / sum_prices)
    """
    logger.info("  build_fact_sales...")

    # ── Totales por orden ────────────────────────────────────────────────────
    sum_prices = (
        order_items.groupby("order_id")["price"]
        .sum().reset_index().rename(columns={"price": "sum_prices"})
    )
    total_payment = (
        order_payments.groupby("order_id")["payment_value"]
        .sum().reset_index().rename(columns={"payment_value": "total_payment"})
    )
    item_count = (
        order_items.groupby("order_id")["order_item_id"]
        .count().reset_index().rename(columns={"order_item_id": "order_item_count"})
    )

    # ── Método de pago dominante (el de mayor valor) ─────────────────────────
    dominant_payment = (
        order_payments
        .sort_values("payment_value", ascending=False)
        .drop_duplicates(subset=["order_id"], keep="first")
        [["order_id", "payment_type", "payment_installments"]]
    )

    # ── Joins en cadena ──────────────────────────────────────────────────────
    fact = order_items.copy()

    fact = pd.merge(fact, orders[[
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]], on="order_id", how="inner")

    fact = fact[fact["order_purchase_timestamp"].notna()]

    fact = pd.merge(fact, sum_prices,        on="order_id", how="inner")
    fact = pd.merge(fact, total_payment,     on="order_id", how="inner")
    fact = pd.merge(fact, item_count,        on="order_id", how="inner")
    fact = pd.merge(fact, dominant_payment,  on="order_id", how="left")

    fact = fact[fact["sum_prices"] > 0]

    # ── Columnas calculadas ──────────────────────────────────────────────────
    fact["date_id"] = fact["order_purchase_timestamp"].dt.normalize()

    fact["payment_value_allocated"] = (
        fact["total_payment"] * (fact["price"] / fact["sum_prices"])
    ).round(2)

    fact["is_delivered"] = fact["order_status"] == "delivered"
    fact["is_canceled"]  = fact["order_status"].isin(["canceled", "unavailable"])

    both_dates = (
        fact["order_delivered_customer_date"].notna() &
        fact["order_estimated_delivery_date"].notna()
    )
    fact["is_on_time"] = np.where(
        both_dates,
        fact["order_delivered_customer_date"] <= fact["order_estimated_delivery_date"],
        None,
    )

    # ── Filtrar productos y clientes que existen en las dimensiones ──────────
    fact = fact[
        fact["product_id"].isin(set(dim_product["product_id"])) &
        fact["customer_id"].isin(set(dim_customer["customer_id"]))
    ]

    fact = fact[[
        "order_id", "order_item_id", "date_id",
        "customer_id", "product_id", "seller_id",
        "price", "freight_value", "payment_value_allocated",
        "is_delivered", "is_canceled", "is_on_time",
        "order_item_count", "payment_type", "payment_installments",
    ]].rename(columns={"price": "item_price"})

    fact = fact.drop_duplicates(subset=["order_id", "order_item_id"])

    logger.info("    ✓ %d filas", len(fact))
    return fact


def run_dwh(clean: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("--- Transformando clean → dwh ---")

    dim_date     = build_dim_date(clean["orders"])
    dim_customer = build_dim_customer(clean["customers"])
    dim_product  = build_dim_product(clean["products"])
    dim_seller   = build_dim_seller(clean["sellers"])
    dim_order    = build_dim_order(clean["orders"], clean["order_reviews"])

    fact_sales = build_fact_sales(
        orders         = clean["orders"],
        order_items    = clean["order_items"],
        order_payments = clean["order_payments"],
        dim_product    = dim_product,
        dim_customer   = dim_customer,
    )

    return {
        "dim_date":     dim_date,
        "dim_customer": dim_customer,
        "dim_product":  dim_product,
        "dim_seller":   dim_seller,
        "dim_order":    dim_order,
        "fact_sales":   fact_sales,
    }
