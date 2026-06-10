"""
pipeline.py
-----------
Orquesta el ETL completo:
  1. Extract  — CSV → raw.*
  2. Transform — raw.* → clean.* (en memoria)
  3. Load clean — DataFrames → clean.*
  4. Transform — clean.* → dwh.* (en memoria)
  5. Load dwh  — DataFrames → dwh.*
"""

import logging
import sys

import pandas as pd
from sqlalchemy import Engine, text

from etl.db import get_engine, test_connection
from etl.extract.load_csvs import load_all_csvs
from etl.transform.clean import run_clean
from etl.transform.dwh import run_dwh
from etl.transform.quality import report, validate_clean, validate_dwh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _read_tables(engine: Engine, schema: str, tables: list[str]) -> dict[str, pd.DataFrame]:
    """Lee una lista de tablas de un schema y las devuelve como dict de DataFrames."""
    logger.info("Leyendo %s.* desde PostgreSQL...", schema)
    return {
        t: pd.read_sql(f"SELECT * FROM {schema}.{t}", engine)
        for t in tables
    }


def _write_tables(
    dfs: dict[str, pd.DataFrame],
    schema: str,
    engine: Engine,
) -> None:
    """
    Escribe un dict de DataFrames en un schema de PostgreSQL.
    Trunca las tablas antes de insertar para garantizar idempotencia.
    """
    tables_sql = ", ".join(f"{schema}.{t}" for t in dfs)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables_sql} CASCADE"))
    logger.info("  ✓ Tablas %s.* vaciadas", schema)

    for table_name, df in dfs.items():
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi",
        )
        logger.info("  ✓ %s.%-28s %d filas", schema, table_name, len(df))


def run() -> None:
    logger.info("=== Iniciando ETL de Olist ===")

    engine = get_engine()
    test_connection(engine)

    # ── Paso 1: Extract ──────────────────────────────────────────────
    logger.info("--- PASO 1: Extract (CSV → raw.*) ---")
    load_all_csvs(engine)

    # ── Paso 2 + 3: raw → clean ──────────────────────────────────────
    logger.info("--- PASO 2: Transform raw → clean ---")
    raw_dfs   = _read_tables(engine, "raw", [
        "orders", "order_items", "order_payments",
        "customers", "products", "sellers",
        "product_category_translation", "order_reviews",
    ])
    clean_dfs = run_clean(raw_dfs)

    logger.info("--- PASO 2b: Quality checks clean.* ---")
    report(validate_clean(clean_dfs))

    logger.info("--- PASO 3: Load clean.* ---")
    _write_tables(clean_dfs, "clean", engine)

    # ── Paso 4 + 5: clean → dwh ──────────────────────────────────────
    logger.info("--- PASO 4: Transform clean → dwh ---")
    dwh_dfs = run_dwh(clean_dfs)  # usamos los DataFrames ya en memoria

    logger.info("--- PASO 4b: Quality checks dwh.* ---")
    report(validate_dwh(dwh_dfs, clean_dfs))

    logger.info("--- PASO 5: Load dwh.* ---")
    _write_tables(dwh_dfs, "dwh", engine)

    # ── Resumen  ─────────────────────────────────────────────────
    logger.info("=== ETL completo ===")
    logger.info("Resumen dwh.*:")
    for table, df in dwh_dfs.items():
        logger.info("  %-30s %d filas", table, len(df))


if __name__ == "__main__":
    run()