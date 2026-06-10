"""
extract/load_csvs.py
--------------------
Carga los CSVs de Olist en las tablas raw.* de PostgreSQL.
"""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from etl.config import CSV_TO_TABLE, get_csv_dir

logger = logging.getLogger(__name__)


def _truncate_raw_tables(engine: Engine) -> None:
    tables_sql = ", ".join(CSV_TO_TABLE.values())
    logger.info("Truncando tablas raw.*...")
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables_sql} CASCADE"))
    logger.info("✓ Tablas raw.* vaciadas")


def _load_one(csv_path: Path, table: str, engine: Engine) -> int:
    if not csv_path.exists():
        logger.warning("  ⚠ %s no encontrado — saltando %s", csv_path.name, table)
        return 0

    logger.info("  → Cargando %s en %s...", csv_path.name, table)

    # dtype=str: lee todo como texto para no perder ceros a la izquierda
    # ni que pandas infiera tipos incorrectamente antes de la limpieza
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8")

    schema, table_name = table.split(".")
    df.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        method="multi",
    )

    logger.info("    ✓ %d filas", len(df))
    return len(df)


def load_all_csvs(engine: Engine) -> dict[str, int]:
    csv_dir = get_csv_dir()
    logger.info("Directorio de CSVs: %s", csv_dir)

    _truncate_raw_tables(engine)

    results: dict[str, int] = {}
    for filename, table in CSV_TO_TABLE.items():
        results[table] = _load_one(csv_dir / filename, table, engine)

    total = sum(results.values())
    logger.info("Extract completo — %d filas totales cargadas en raw.*", total)
    return results
