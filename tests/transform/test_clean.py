import pandas as pd
import pytest

from etl.transform.clean import (
    clean_customers,
    clean_order_items,
    clean_order_payments,
    clean_order_reviews,
    clean_orders,
    clean_products,
    clean_sellers,
)


# ── clean_customers ───────────────────────────────────────────────────────────

def _make_customer(**overrides):
    base = {
        "customer_id": ["c1"],
        "customer_unique_id": ["u1"],
        "customer_zip_code_prefix": ["01310"],
        "customer_city": ["São Paulo"],
        "customer_state": ["sp"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_customers_drops_empty_id():
    df = pd.concat([
        _make_customer(customer_id=["c1"]),
        _make_customer(customer_id=[""]),
        _make_customer(customer_id=[None]),
    ], ignore_index=True)
    out = clean_customers(df)
    assert list(out["customer_id"]) == ["c1"]


def test_clean_customers_normalizes_city_to_lowercase():
    df = _make_customer(customer_city=["  RIO DE JANEIRO  "])
    out = clean_customers(df)
    assert out.iloc[0]["customer_city"] == "rio de janeiro"


def test_clean_customers_normalizes_state_to_uppercase():
    df = _make_customer(customer_state=["  sp  "])
    out = clean_customers(df)
    assert out.iloc[0]["customer_state"] == "SP"


def test_clean_customers_keeps_first_on_duplicate_id():
    df = pd.concat([
        _make_customer(customer_id=["c1"], customer_unique_id=["u_first"]),
        _make_customer(customer_id=["c1"], customer_unique_id=["u_second"]),
    ], ignore_index=True)
    out = clean_customers(df)
    assert len(out) == 1
    assert out.iloc[0]["customer_unique_id"] == "u_first"


# ── clean_sellers ─────────────────────────────────────────────────────────────

def test_clean_sellers_drops_empty_id():
    df = pd.DataFrame({
        "seller_id": ["s1", "", None],
        "seller_zip_code_prefix": ["01310", "01311", "01312"],
        "seller_city": ["sao paulo", "rio", "bh"],
        "seller_state": ["SP", "RJ", "MG"],
    })
    out = clean_sellers(df)
    assert list(out["seller_id"]) == ["s1"]


def test_clean_sellers_normalizes_city_and_state():
    df = pd.DataFrame({
        "seller_id": ["s1"],
        "seller_zip_code_prefix": ["01310"],
        "seller_city": ["  Curitiba  "],
        "seller_state": ["  pr  "],
    })
    out = clean_sellers(df)
    assert out.iloc[0]["seller_city"] == "curitiba"
    assert out.iloc[0]["seller_state"] == "PR"


# ── clean_orders ──────────────────────────────────────────────────────────────

def _make_raw_order(**overrides):
    base = {
        "order_id": ["o1"],
        "customer_id": ["c1"],
        "order_status": ["delivered"],
        "order_purchase_timestamp": ["2023-01-10 10:00:00"],
        "order_approved_at": ["2023-01-10 11:00:00"],
        "order_delivered_carrier_date": ["2023-01-12 00:00:00"],
        "order_delivered_customer_date": ["2023-01-15 00:00:00"],
        "order_estimated_delivery_date": ["2023-01-20 00:00:00"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_orders_drops_empty_order_id():
    df = pd.concat([
        _make_raw_order(order_id=["o1"]),
        _make_raw_order(order_id=[""]),
    ], ignore_index=True)
    out = clean_orders(df)
    assert list(out["order_id"]) == ["o1"]


def test_clean_orders_drops_empty_customer_id():
    df = pd.concat([
        _make_raw_order(customer_id=["c1"]),
        _make_raw_order(order_id=["o2"], customer_id=[""]),
    ], ignore_index=True)
    out = clean_orders(df)
    assert list(out["order_id"]) == ["o1"]


def test_clean_orders_converts_purchase_timestamp():
    out = clean_orders(_make_raw_order())
    assert pd.api.types.is_datetime64_any_dtype(out["order_purchase_timestamp"])
    assert out.iloc[0]["order_purchase_timestamp"] == pd.Timestamp("2023-01-10 10:00:00")


def test_clean_orders_coerces_invalid_timestamp_to_nat():
    out = clean_orders(_make_raw_order(order_approved_at=["not-a-date"]))
    assert pd.isna(out.iloc[0]["order_approved_at"])


# ── clean_order_items ─────────────────────────────────────────────────────────

def _make_raw_item(**overrides):
    base = {
        "order_id": ["o1"],
        "order_item_id": ["1"],
        "product_id": ["p1"],
        "seller_id": ["s1"],
        "shipping_limit_date": ["2023-01-20 00:00:00"],
        "price": ["99.99"],
        "freight_value": ["10.50"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_order_items_drops_empty_price():
    df = pd.concat([
        _make_raw_item(order_id=["o1"], price=["99.99"]),
        _make_raw_item(order_id=["o2"], price=[""]),
    ], ignore_index=True)
    out = clean_order_items(df)
    assert list(out["order_id"]) == ["o1"]


def test_clean_order_items_drops_empty_freight():
    df = pd.concat([
        _make_raw_item(order_id=["o1"]),
        _make_raw_item(order_id=["o2"], freight_value=[""]),
    ], ignore_index=True)
    out = clean_order_items(df)
    assert list(out["order_id"]) == ["o1"]


def test_clean_order_items_converts_price_to_float():
    out = clean_order_items(_make_raw_item(price=["149.90"]))
    assert out.iloc[0]["price"] == 149.90


def test_clean_order_items_converts_order_item_id_to_int():
    out = clean_order_items(_make_raw_item(order_item_id=["3"]))
    assert out.iloc[0]["order_item_id"] == 3


# ── clean_order_payments ──────────────────────────────────────────────────────

def _make_raw_payment(**overrides):
    base = {
        "order_id": ["o1"],
        "payment_sequential": ["1"],
        "payment_type": ["credit_card"],
        "payment_installments": ["3"],
        "payment_value": ["100.00"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_order_payments_drops_empty_payment_value():
    df = pd.concat([
        _make_raw_payment(order_id=["o1"]),
        _make_raw_payment(order_id=["o2"], payment_value=[""]),
    ], ignore_index=True)
    out = clean_order_payments(df)
    assert list(out["order_id"]) == ["o1"]


def test_clean_order_payments_coalesces_null_installments_to_1():
    # COALESCE(payment_installments, 1) — None → 1
    df = _make_raw_payment(payment_installments=[None])
    out = clean_order_payments(df)
    assert out.iloc[0]["payment_installments"] == 1


def test_clean_order_payments_keeps_explicit_installments():
    df = _make_raw_payment(payment_installments=["6"])
    out = clean_order_payments(df)
    assert out.iloc[0]["payment_installments"] == 6


def test_clean_order_payments_deduplicates_by_order_and_sequential():
    df = pd.concat([
        _make_raw_payment(payment_value=["100.00"]),
        _make_raw_payment(payment_value=["999.00"]),  # duplicate key
    ], ignore_index=True)
    out = clean_order_payments(df)
    assert len(out) == 1
    assert out.iloc[0]["payment_value"] == 100.00


# ── clean_order_reviews ───────────────────────────────────────────────────────

def _make_raw_review(**overrides):
    base = {
        "review_id": ["r1"],
        "order_id": ["o1"],
        "review_score": ["5"],
        "review_creation_date": ["2023-01-12 00:00:00"],
        "review_answer_timestamp": ["2023-01-13 00:00:00"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_order_reviews_drops_empty_review_id():
    df = pd.concat([
        _make_raw_review(review_id=["r1"]),
        _make_raw_review(review_id=[""]),
    ], ignore_index=True)
    out = clean_order_reviews(df)
    assert list(out["review_id"]) == ["r1"]


def test_clean_order_reviews_converts_score_to_int():
    out = clean_order_reviews(_make_raw_review(review_score=["4"]))
    assert out.iloc[0]["review_score"] == 4


def test_clean_order_reviews_converts_timestamps():
    out = clean_order_reviews(_make_raw_review())
    assert pd.api.types.is_datetime64_any_dtype(out["review_creation_date"])


# ── clean_products ────────────────────────────────────────────────────────────

def _make_raw_products(**overrides):
    base = {
        "product_id": ["p1"],
        "product_category_name": ["cama_mesa_banho"],
        "product_weight_g": ["1000"],
        "product_length_cm": ["30"],
        "product_height_cm": ["10"],
        "product_width_cm": ["20"],
        "product_photos_qty": ["3"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_translation(mapping: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "product_category_name": list(mapping.keys()),
        "product_category_name_english": list(mapping.values()),
    })


def test_clean_products_joins_english_translation():
    out = clean_products(
        _make_raw_products(product_category_name=["cama_mesa_banho"]),
        _make_translation({"cama_mesa_banho": "bed_bath_table"}),
    )
    assert out.iloc[0]["product_category_name_english"] == "bed_bath_table"


def test_clean_products_unknown_category_gets_null_translation():
    out = clean_products(
        _make_raw_products(product_category_name=["categoria_inexistente"]),
        _make_translation({"cama_mesa_banho": "bed_bath_table"}),
    )
    assert pd.isna(out.iloc[0]["product_category_name_english"])


def test_clean_products_converts_numeric_dimensions():
    out = clean_products(
        _make_raw_products(product_weight_g=["750"], product_length_cm=["25.5"]),
        _make_translation({"cama_mesa_banho": "bed_bath_table"}),
    )
    assert out.iloc[0]["product_weight_g"] == 750.0
    assert out.iloc[0]["product_length_cm"] == 25.5


def test_clean_products_coerces_invalid_numeric_to_nan():
    out = clean_products(
        _make_raw_products(product_weight_g=["invalid"]),
        _make_translation({"cama_mesa_banho": "bed_bath_table"}),
    )
    assert pd.isna(out.iloc[0]["product_weight_g"])
