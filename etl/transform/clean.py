"""
transform/clean.py
------------------
Transforma raw.* → clean.*

Cada función recibe un DataFrame crudo (todo texto, como llegó del CSV)
y devuelve un DataFrame limpio con tipos correctos y sin duplicados.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _to_numeric(series: pd.Series, decimals: int = 2) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(decimals)


def _drop_duplicates_on(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    dropped = before - len(df)
    if dropped:
        logger.debug("    drop_duplicates: %d filas eliminadas", dropped)
    return df


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  clean_customers...")
    out = df.copy()

    mask = out["customer_id"].notna() & (out["customer_id"] != "")
    out = out[mask]

    out["customer_city"]  = out["customer_city"].str.strip().str.lower()
    out["customer_state"] = out["customer_state"].str.strip().str.upper()
    out = _drop_duplicates_on(out, ["customer_id"])

    out = out[[
        "customer_id", "customer_unique_id",
        "customer_zip_code_prefix", "customer_city", "customer_state",
    ]]

    logger.info("    ✓ %d filas", len(out))
    return out


def clean_products(
    df_products: pd.DataFrame,
    df_translation: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("  clean_products...")
    out = df_products.copy()

    mask = out["product_id"].notna() & (out["product_id"] != "")
    out = out[mask]
    out = _drop_duplicates_on(out, ["product_id"])

    out = pd.merge(
        out,
        df_translation[["product_category_name", "product_category_name_english"]],
        on="product_category_name",
        how="left",
    )

    out["product_weight_g"]   = _to_numeric(out["product_weight_g"])
    out["product_length_cm"]  = _to_numeric(out["product_length_cm"])
    out["product_height_cm"]  = _to_numeric(out["product_height_cm"])
    out["product_width_cm"]   = _to_numeric(out["product_width_cm"])

    # Int64 (con mayúscula) admite NaN; int64 no
    out["product_photos_qty"] = pd.to_numeric(
        out["product_photos_qty"], errors="coerce"
    ).astype("Int64")

    out = out[[
        "product_id", "product_category_name", "product_category_name_english",
        "product_weight_g", "product_length_cm", "product_height_cm",
        "product_width_cm", "product_photos_qty",
    ]]

    logger.info("    ✓ %d filas", len(out))
    return out


def clean_sellers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  clean_sellers...")
    out = df.copy()

    mask = out["seller_id"].notna() & (out["seller_id"] != "")
    out = out[mask]

    out["seller_city"]  = out["seller_city"].str.strip().str.lower()
    out["seller_state"] = out["seller_state"].str.strip().str.upper()
    out = _drop_duplicates_on(out, ["seller_id"])

    out = out[[
        "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state",
    ]]

    logger.info("    ✓ %d filas", len(out))
    return out


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  clean_orders...")
    out = df.copy()

    mask = (
        out["order_id"].notna() & (out["order_id"] != "") &
        out["customer_id"].notna() & (out["customer_id"] != "")
    )
    out = out[mask]
    out = _drop_duplicates_on(out, ["order_id"])

    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in timestamp_cols:
        out[col] = _to_timestamp(out[col])

    out = out[["order_id", "customer_id", "order_status", *timestamp_cols]]

    logger.info("    ✓ %d filas", len(out))
    return out


def clean_order_items(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  clean_order_items...")
    out = df.copy()

    mask = (
        out["order_id"].notna() & (out["order_id"] != "") &
        out["order_item_id"].notna() &
        out["price"].notna() & (out["price"] != "") &
        out["freight_value"].notna() & (out["freight_value"] != "")
    )
    out = out[mask]

    out["order_item_id"]       = pd.to_numeric(out["order_item_id"], errors="coerce").astype("Int64")
    out["price"]               = _to_numeric(out["price"])
    out["freight_value"]       = _to_numeric(out["freight_value"])
    out["shipping_limit_date"] = _to_timestamp(out["shipping_limit_date"])

    out = _drop_duplicates_on(out, ["order_id", "order_item_id"])

    out = out[[
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    ]]

    logger.info("    ✓ %d filas", len(out))
    return out


def clean_order_payments(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  clean_order_payments...")
    out = df.copy()

    mask = (
        out["order_id"].notna() & (out["order_id"] != "") &
        out["payment_value"].notna() & (out["payment_value"] != "")
    )
    out = out[mask]

    out["payment_sequential"]   = pd.to_numeric(out["payment_sequential"], errors="coerce").astype("Int64")
    out["payment_installments"] = pd.to_numeric(out["payment_installments"], errors="coerce").fillna(1).astype("Int64")
    out["payment_value"]        = _to_numeric(out["payment_value"])

    out = _drop_duplicates_on(out, ["order_id", "payment_sequential"])

    out = out[[
        "order_id", "payment_sequential", "payment_type",
        "payment_installments", "payment_value",
    ]]

    logger.info("    ✓ %d filas", len(out))
    return out


def clean_order_reviews(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  clean_order_reviews...")
    out = df.copy()

    mask = (
        out["review_id"].notna() & (out["review_id"] != "") &
        out["order_id"].notna() & (out["order_id"] != "")
    )
    out = out[mask]
    out = _drop_duplicates_on(out, ["review_id"])

    out["review_score"]            = pd.to_numeric(out["review_score"], errors="coerce").astype("Int64")
    out["review_creation_date"]    = _to_timestamp(out["review_creation_date"])
    out["review_answer_timestamp"] = _to_timestamp(out["review_answer_timestamp"])

    out = out[[
        "review_id", "order_id", "review_score",
        "review_creation_date", "review_answer_timestamp",
    ]]

    logger.info("    ✓ %d filas", len(out))
    return out


def run_clean(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("--- Transformando raw → clean ---")
    return {
        "customers":      clean_customers(raw["customers"]),
        "products":       clean_products(raw["products"], raw["product_category_translation"]),
        "sellers":        clean_sellers(raw["sellers"]),
        "orders":         clean_orders(raw["orders"]),
        "order_items":    clean_order_items(raw["order_items"]),
        "order_payments": clean_order_payments(raw["order_payments"]),
        "order_reviews":  clean_order_reviews(raw["order_reviews"]),
    }
