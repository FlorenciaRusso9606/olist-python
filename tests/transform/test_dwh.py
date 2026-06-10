import numpy as np
import pandas as pd
import pytest

from etl.transform.dwh import (
    build_dim_date,
    build_dim_order,
    build_dim_product,
    build_fact_sales,
)


# ── build_dim_date ────────────────────────────────────────────────────────────

def _orders_with_dates(*date_strings):
    return pd.DataFrame({
        "order_purchase_timestamp": pd.to_datetime(list(date_strings)),
    })


def test_build_dim_date_extracts_year_month_quarter():
    out = build_dim_date(_orders_with_dates("2023-07-15"))
    row = out.iloc[0]
    assert row["year"] == 2023
    assert row["month"] == 7
    assert row["quarter"] == 3


def test_build_dim_date_saturday_is_weekend():
    out = build_dim_date(_orders_with_dates("2023-07-08"))  # Saturday
    assert out.iloc[0]["is_weekend"] == True


def test_build_dim_date_monday_is_not_weekend():
    out = build_dim_date(_orders_with_dates("2023-07-10"))  # Monday
    assert out.iloc[0]["is_weekend"] == False


def test_build_dim_date_deduplicates_same_day():
    # Two timestamps on the same day → one row in dim_date
    out = build_dim_date(_orders_with_dates("2023-01-01 09:00", "2023-01-01 18:00"))
    assert len(out) == 1


def test_build_dim_date_drops_nat():
    orders = pd.DataFrame({
        "order_purchase_timestamp": pd.to_datetime(["2023-01-01", None]),
    })
    out = build_dim_date(orders)
    assert len(out) == 1


def test_build_dim_date_day_name_is_correct():
    out = build_dim_date(_orders_with_dates("2023-07-10"))  # Monday
    assert out.iloc[0]["day_name"] == "Monday"


# ── build_dim_product ─────────────────────────────────────────────────────────

def _make_clean_product(**overrides):
    base = {
        "product_id": ["p1"],
        "product_category_name": ["cat"],
        "product_category_name_english": ["category"],
        "product_weight_g": [500.0],
        "product_length_cm": [10.0],
        "product_height_cm": [5.0],
        "product_width_cm": [4.0],
        "product_photos_qty": pd.array([2], dtype="Int64"),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_build_dim_product_computes_volume():
    # 10 × 5 × 4 = 200
    out = build_dim_product(_make_clean_product(
        product_length_cm=[10.0],
        product_height_cm=[5.0],
        product_width_cm=[4.0],
    ))
    assert out.iloc[0]["product_volume_cm3"] == 200.0


def test_build_dim_product_volume_nan_when_length_missing():
    out = build_dim_product(_make_clean_product(product_length_cm=[np.nan]))
    assert pd.isna(out.iloc[0]["product_volume_cm3"])


def test_build_dim_product_volume_nan_when_height_missing():
    out = build_dim_product(_make_clean_product(product_height_cm=[np.nan]))
    assert pd.isna(out.iloc[0]["product_volume_cm3"])


def test_build_dim_product_volume_nan_when_width_missing():
    out = build_dim_product(_make_clean_product(product_width_cm=[np.nan]))
    assert pd.isna(out.iloc[0]["product_volume_cm3"])


# ── build_dim_order ───────────────────────────────────────────────────────────

def _make_clean_order(**overrides):
    base = {
        "order_id": ["o1"],
        "customer_id": ["c1"],
        "order_status": ["delivered"],
        "order_purchase_timestamp": pd.to_datetime(["2023-01-01"]),
        "order_approved_at": pd.to_datetime(["2023-01-01"]),
        "order_delivered_carrier_date": pd.to_datetime(["2023-01-09"]),
        "order_delivered_customer_date": pd.to_datetime(["2023-01-11"]),
        "order_estimated_delivery_date": pd.to_datetime(["2023-01-15"]),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_clean_review(**overrides):
    base = {
        "review_id": ["r1"],
        "order_id": ["o1"],
        "review_score": pd.array([5], dtype="Int64"),
        "review_creation_date": pd.to_datetime(["2023-01-12"]),
        "review_answer_timestamp": pd.to_datetime(["2023-01-13"]),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_build_dim_order_computes_delivery_days():
    orders = _make_clean_order(
        order_purchase_timestamp=pd.to_datetime(["2023-01-01"]),
        order_delivered_customer_date=pd.to_datetime(["2023-01-11"]),
    )
    out = build_dim_order(orders, _make_clean_review())
    assert out.iloc[0]["delivery_days"] == 10


def test_build_dim_order_positive_delay_when_late():
    # Llegó 5 días después de la estimada → delay = +5
    orders = _make_clean_order(
        order_delivered_customer_date=pd.to_datetime(["2023-01-20"]),
        order_estimated_delivery_date=pd.to_datetime(["2023-01-15"]),
    )
    out = build_dim_order(orders, _make_clean_review())
    assert out.iloc[0]["delivery_delay_days"] == 5


def test_build_dim_order_negative_delay_when_early():
    # Llegó 3 días antes de la estimada → delay = -3
    orders = _make_clean_order(
        order_delivered_customer_date=pd.to_datetime(["2023-01-12"]),
        order_estimated_delivery_date=pd.to_datetime(["2023-01-15"]),
    )
    out = build_dim_order(orders, _make_clean_review())
    assert out.iloc[0]["delivery_delay_days"] == -3


def test_build_dim_order_joins_review_score():
    out = build_dim_order(_make_clean_order(), _make_clean_review(review_score=pd.array([4], dtype="Int64")))
    assert out.iloc[0]["review_score"] == 4


def test_build_dim_order_null_review_score_when_no_match():
    # LEFT JOIN: si no hay review para la orden → NaN
    reviews = _make_clean_review(order_id=["o_other"])
    out = build_dim_order(_make_clean_order(order_id=["o1"]), reviews)
    assert pd.isna(out.iloc[0]["review_score"])


# ── build_fact_sales ──────────────────────────────────────────────────────────

def _base_orders(**overrides):
    base = {
        "order_id": ["o1"],
        "customer_id": ["c1"],
        "order_status": ["delivered"],
        "order_purchase_timestamp": pd.to_datetime(["2023-01-10"]),
        "order_delivered_customer_date": pd.to_datetime(["2023-01-20"]),
        "order_estimated_delivery_date": pd.to_datetime(["2023-01-25"]),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _base_items(prices, order_id="o1", product_id="p1"):
    n = len(prices)
    return pd.DataFrame({
        "order_id": [order_id] * n,
        "order_item_id": pd.array(list(range(1, n + 1)), dtype="Int64"),
        "product_id": [product_id] * n,
        "seller_id": ["s1"] * n,
        "shipping_limit_date": pd.to_datetime(["2023-01-15"] * n),
        "price": [float(p) for p in prices],
        "freight_value": [10.0] * n,
    })


def _base_payments(total, order_id="o1"):
    return pd.DataFrame({
        "order_id": [order_id],
        "payment_sequential": pd.array([1], dtype="Int64"),
        "payment_type": ["credit_card"],
        "payment_installments": pd.array([1], dtype="Int64"),
        "payment_value": [float(total)],
    })


def _dim_product(product_ids=("p1",)):
    return pd.DataFrame({"product_id": list(product_ids)})


def _dim_customer(customer_ids=("c1",)):
    return pd.DataFrame({"customer_id": list(customer_ids)})


def test_fact_sales_prorates_payment_value_proportionally():
    # 2 ítems: $100 y $200 → sum=300, total_payment=$150
    # item1: 150 × (100/300) = 50.00
    # item2: 150 × (200/300) = 100.00
    out = build_fact_sales(
        orders=_base_orders(),
        order_items=_base_items([100, 200]),
        order_payments=_base_payments(150),
        dim_product=_dim_product(),
        dim_customer=_dim_customer(),
    )
    out = out.sort_values("order_item_id").reset_index(drop=True)
    assert out.iloc[0]["payment_value_allocated"] == 50.0
    assert out.iloc[1]["payment_value_allocated"] == 100.0


def test_fact_sales_single_item_gets_full_payment():
    out = build_fact_sales(
        orders=_base_orders(),
        order_items=_base_items([200]),
        order_payments=_base_payments(180),
        dim_product=_dim_product(),
        dim_customer=_dim_customer(),
    )
    assert out.iloc[0]["payment_value_allocated"] == 180.0


def test_fact_sales_is_delivered_true_for_delivered_orders():
    out = build_fact_sales(
        orders=_base_orders(order_status=["delivered"]),
        order_items=_base_items([100]),
        order_payments=_base_payments(100),
        dim_product=_dim_product(),
        dim_customer=_dim_customer(),
    )
    assert out.iloc[0]["is_delivered"] == True
    assert out.iloc[0]["is_canceled"] == False


def test_fact_sales_is_canceled_true_for_canceled_orders():
    out = build_fact_sales(
        orders=_base_orders(order_status=["canceled"]),
        order_items=_base_items([100]),
        order_payments=_base_payments(100),
        dim_product=_dim_product(),
        dim_customer=_dim_customer(),
    )
    assert out.iloc[0]["is_canceled"] == True
    assert out.iloc[0]["is_delivered"] == False


def test_fact_sales_is_on_time_true_when_delivered_before_estimate():
    out = build_fact_sales(
        orders=_base_orders(
            order_delivered_customer_date=pd.to_datetime(["2023-01-20"]),
            order_estimated_delivery_date=pd.to_datetime(["2023-01-25"]),
        ),
        order_items=_base_items([100]),
        order_payments=_base_payments(100),
        dim_product=_dim_product(),
        dim_customer=_dim_customer(),
    )
    assert out.iloc[0]["is_on_time"] == True


def test_fact_sales_is_on_time_false_when_delivered_late():
    out = build_fact_sales(
        orders=_base_orders(
            order_delivered_customer_date=pd.to_datetime(["2023-01-30"]),
            order_estimated_delivery_date=pd.to_datetime(["2023-01-25"]),
        ),
        order_items=_base_items([100]),
        order_payments=_base_payments(100),
        dim_product=_dim_product(),
        dim_customer=_dim_customer(),
    )
    assert out.iloc[0]["is_on_time"] == False


def test_fact_sales_filters_out_unknown_product():
    out = build_fact_sales(
        orders=_base_orders(),
        order_items=_base_items([100], product_id="p_unknown"),
        order_payments=_base_payments(100),
        dim_product=_dim_product(["p1"]),   # p_unknown no está
        dim_customer=_dim_customer(),
    )
    assert len(out) == 0


def test_fact_sales_filters_out_unknown_customer():
    out = build_fact_sales(
        orders=_base_orders(customer_id=["c_unknown"]),
        order_items=_base_items([100]),
        order_payments=_base_payments(100),
        dim_product=_dim_product(),
        dim_customer=_dim_customer(["c1"]),  # c_unknown no está
    )
    assert len(out) == 0


def test_fact_sales_price_column_renamed_to_item_price():
    out = build_fact_sales(
        orders=_base_orders(),
        order_items=_base_items([100]),
        order_payments=_base_payments(100),
        dim_product=_dim_product(),
        dim_customer=_dim_customer(),
    )
    assert "item_price" in out.columns
    assert "price" not in out.columns
