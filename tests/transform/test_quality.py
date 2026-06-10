import pandas as pd
import pytest

from etl.transform.quality import (
    DataQualityError,
    QualityIssue,
    _check_empty,
    _check_fk,
    _check_range,
    report,
    validate_clean,
    validate_dwh,
)


# ── _check_empty ──────────────────────────────────────────────────────────────

def test_check_empty_no_issues_when_df_has_rows():
    assert _check_empty("t", pd.DataFrame({"a": [1, 2]})) == []


def test_check_empty_returns_error_when_df_is_empty():
    issues = _check_empty("clean.orders", pd.DataFrame())
    assert len(issues) == 1
    assert issues[0].level == "error"
    assert issues[0].table == "clean.orders"


# ── _check_range ──────────────────────────────────────────────────────────────

def test_check_range_no_issues_when_all_values_in_range():
    s = pd.Series([1.0, 5.0, 10.0], name="price")
    assert _check_range("t", s, min_val=0) == []


def test_check_range_warn_when_value_below_min():
    s = pd.Series([1.0, -5.0, 3.0], name="price")
    issues = _check_range("clean.order_items", s, min_val=0)
    assert len(issues) == 1
    assert issues[0].level == "warn"
    assert "price" in issues[0].detail


def test_check_range_warn_when_value_above_max():
    s = pd.Series([1, 3, 6], name="review_score")
    issues = _check_range("clean.order_reviews", s, max_val=5)
    assert len(issues) == 1
    assert issues[0].level == "warn"


def test_check_range_both_bounds_generate_separate_issues():
    # -1 viola min=1, 6 viola max=5 → 2 issues
    s = pd.Series([-1, 3, 6], name="review_score")
    issues = _check_range("t", s, min_val=1, max_val=5)
    assert len(issues) == 2


def test_check_range_ignores_nan():
    s = pd.Series([1.0, float("nan"), 3.0], name="price")
    assert _check_range("t", s, min_val=0) == []


# ── _check_fk ─────────────────────────────────────────────────────────────────

def test_check_fk_no_issues_when_all_values_exist_in_parent():
    child  = pd.Series(["o1", "o2"], name="order_id")
    parent = pd.Series(["o1", "o2", "o3"])
    assert _check_fk("t", child, parent) == []


def test_check_fk_error_on_orphan_values():
    child  = pd.Series(["o1", "o_unknown"], name="order_id")
    parent = pd.Series(["o1"])
    issues = _check_fk("clean.order_items", child, parent)
    assert len(issues) == 1
    assert issues[0].level == "error"


def test_check_fk_detail_includes_orphan_count():
    child  = pd.Series(["o1", "o2", "o3"], name="order_id")
    parent = pd.Series(["o1"])
    issues = _check_fk("t", child, parent)
    assert "2" in issues[0].detail  # 2 orphans


# ── report ────────────────────────────────────────────────────────────────────

def test_report_does_not_raise_when_no_issues():
    report([])  # no debe lanzar nada


def test_report_does_not_raise_for_warn_only_issues():
    issues = [QualityIssue("t", "range", "price < 0: 2 filas", "warn")]
    report(issues)  # no debe lanzar nada


def test_report_raises_data_quality_error_on_error_issue():
    issues = [QualityIssue("clean.orders", "row_count", "tabla vacía", "error")]
    with pytest.raises(DataQualityError):
        report(issues)


def test_report_raises_even_with_mixed_warn_and_error():
    issues = [
        QualityIssue("t", "range", "price < 0", "warn"),
        QualityIssue("t", "row_count", "tabla vacía", "error"),
    ]
    with pytest.raises(DataQualityError):
        report(issues)


def test_report_error_message_includes_issue_count():
    issues = [
        QualityIssue("t", "row_count", "tabla vacía", "error"),
        QualityIssue("t", "null_key", "order_id nulo", "error"),
    ]
    with pytest.raises(DataQualityError) as exc_info:
        report(issues)
    assert "2" in str(exc_info.value)


# ── validate_clean ────────────────────────────────────────────────────────────

def _minimal_clean():
    return {
        "orders": pd.DataFrame({
            "order_id": ["o1"],
            "customer_id": ["c1"],
            "order_status": ["delivered"],
            "order_purchase_timestamp": pd.to_datetime(["2023-01-01"]),
            "order_delivered_customer_date": pd.to_datetime(["2023-01-10"]),
            "order_estimated_delivery_date": pd.to_datetime(["2023-01-15"]),
        }),
        "order_items": pd.DataFrame({
            "order_id": ["o1"],
            "order_item_id": pd.array([1], dtype="Int64"),
            "product_id": ["p1"],
            "seller_id": ["s1"],
            "shipping_limit_date": pd.to_datetime(["2023-01-05"]),
            "price": [99.0],
            "freight_value": [10.0],
        }),
        "order_payments": pd.DataFrame({
            "order_id": ["o1"],
            "payment_sequential": pd.array([1], dtype="Int64"),
            "payment_type": ["credit_card"],
            "payment_installments": pd.array([1], dtype="Int64"),
            "payment_value": [109.0],
        }),
        "order_reviews": pd.DataFrame({
            "review_id": ["r1"],
            "order_id": ["o1"],
            "review_score": pd.array([5], dtype="Int64"),
            "review_creation_date": pd.to_datetime(["2023-01-11"]),
            "review_answer_timestamp": pd.to_datetime(["2023-01-12"]),
        }),
        "products": pd.DataFrame({
            "product_id": ["p1"],
            "product_category_name": ["eletronicos"],
            "product_category_name_english": ["electronics"],
            "product_weight_g": [500.0],
            "product_length_cm": [10.0],
            "product_height_cm": [5.0],
            "product_width_cm": [4.0],
            "product_photos_qty": pd.array([2], dtype="Int64"),
        }),
        "customers": pd.DataFrame({
            "customer_id": ["c1"],
            "customer_unique_id": ["u1"],
            "customer_zip_code_prefix": ["01310"],
            "customer_city": ["sao paulo"],
            "customer_state": ["SP"],
        }),
        "sellers": pd.DataFrame({
            "seller_id": ["s1"],
            "seller_zip_code_prefix": ["01310"],
            "seller_city": ["sao paulo"],
            "seller_state": ["SP"],
        }),
    }


def test_validate_clean_no_errors_on_valid_data():
    errors = [i for i in validate_clean(_minimal_clean()) if i.level == "error"]
    assert errors == []


def test_validate_clean_error_on_empty_table():
    clean = _minimal_clean()
    clean["orders"] = clean["orders"].iloc[0:0]  # vacía pero con las mismas columnas
    issues = validate_clean(clean)
    errors = [i for i in issues if i.level == "error" and "orders" in i.table]
    assert len(errors) >= 1


def test_validate_clean_warn_on_negative_price():
    clean = _minimal_clean()
    clean["order_items"]["price"] = -5.0
    issues = validate_clean(clean)
    warns = [i for i in issues if i.level == "warn" and "price" in i.detail]
    assert len(warns) >= 1


def test_validate_clean_error_on_fk_violation_order_items():
    clean = _minimal_clean()
    clean["order_items"]["order_id"] = "o_inexistente"
    issues = validate_clean(clean)
    errors = [i for i in issues if i.check == "referential_integrity"]
    assert len(errors) >= 1


def test_validate_clean_warn_on_review_score_out_of_range():
    clean = _minimal_clean()
    clean["order_reviews"]["review_score"] = pd.array([6], dtype="Int64")
    issues = validate_clean(clean)
    warns = [i for i in issues if i.level == "warn" and "review_score" in i.detail]
    assert len(warns) >= 1


# ── validate_dwh ──────────────────────────────────────────────────────────────

def _minimal_dwh():
    return {
        "dim_date": pd.DataFrame({
            "date_id": pd.to_datetime(["2023-01-01"]),
            "year": [2023], "quarter": [1], "month": [1],
            "month_name": ["January"], "week": [52],
            "day_of_week": [6], "day_name": ["Sunday"], "is_weekend": [True],
        }),
        "dim_customer": pd.DataFrame({"customer_id": ["c1"]}),
        "dim_product":  pd.DataFrame({
            "product_id": ["p1"], "product_category_name": ["cat"],
            "product_category_name_english": ["cat"],
            "product_weight_g": [500.0], "product_volume_cm3": [200.0],
        }),
        "dim_seller": pd.DataFrame({
            "seller_id": ["s1"], "seller_city": ["sp"],
            "seller_state": ["SP"], "seller_zip_code_prefix": ["01310"],
        }),
        "dim_order": pd.DataFrame({
            "order_id": ["o1"], "order_status": ["delivered"],
            "order_purchase_timestamp": pd.to_datetime(["2023-01-01"]),
            "order_delivered_customer_date": pd.to_datetime(["2023-01-10"]),
            "order_estimated_delivery_date": pd.to_datetime(["2023-01-15"]),
            "delivery_days": [9], "delivery_delay_days": [-5],
            "review_score": pd.array([5], dtype="Int64"),
        }),
        "fact_sales": pd.DataFrame({
            "order_id": ["o1"],
            "order_item_id": pd.array([1], dtype="Int64"),
            "date_id": pd.to_datetime(["2023-01-01"]),
            "customer_id": ["c1"],
            "product_id": ["p1"],
            "seller_id": ["s1"],
            "item_price": [99.0],
            "freight_value": [10.0],
            "payment_value_allocated": [109.0],
            "is_delivered": [True],
            "is_canceled": [False],
            "is_on_time": [True],
            "order_item_count": pd.array([1], dtype="Int64"),
            "payment_type": ["credit_card"],
            "payment_installments": pd.array([1], dtype="Int64"),
        }),
    }


def _minimal_clean_for_dwh():
    # validate_dwh solo usa clean["order_items"] para la tasa de retención
    return {"order_items": pd.DataFrame({"order_id": ["o1"], "price": [99.0]})}


def test_validate_dwh_no_errors_on_valid_data():
    errors = [i for i in validate_dwh(_minimal_dwh(), _minimal_clean_for_dwh()) if i.level == "error"]
    assert errors == []


def test_validate_dwh_error_on_empty_fact_sales():
    dwh = _minimal_dwh()
    dwh["fact_sales"] = dwh["fact_sales"].iloc[0:0]
    issues = validate_dwh(dwh, _minimal_clean_for_dwh())
    errors = [i for i in issues if i.level == "error" and "fact_sales" in i.table]
    assert len(errors) >= 1


def test_validate_dwh_error_on_null_in_key_column():
    dwh = _minimal_dwh()
    dwh["fact_sales"]["customer_id"] = None
    issues = validate_dwh(dwh, _minimal_clean_for_dwh())
    errors = [i for i in issues if i.check == "null_key" and "customer_id" in i.detail]
    assert len(errors) >= 1


def test_validate_dwh_error_on_fk_violation_customer():
    dwh = _minimal_dwh()
    dwh["fact_sales"]["customer_id"] = "c_desconocido"
    issues = validate_dwh(dwh, _minimal_clean_for_dwh())
    errors = [i for i in issues if i.check == "referential_integrity" and "customer_id" in i.detail]
    assert len(errors) >= 1


def test_validate_dwh_warn_on_low_retention_rate():
    dwh = _minimal_dwh()  # fact_sales tiene 1 fila
    # 1 de 10 order_items llegaron → 10% retención → warn
    clean = {"order_items": pd.DataFrame({"order_id": [f"o{i}" for i in range(10)]})}
    issues = validate_dwh(dwh, clean)
    warns = [i for i in issues if i.check == "retention"]
    assert len(warns) >= 1
