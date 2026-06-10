"""
db.py
-----
Crea y expone la conexión a PostgreSQL.

Usamos SQLAlchemy como capa de abstracción por dos razones:
1. pandas tiene integración nativa con SQLAlchemy (to_sql, read_sql)
2. Maneja el pool de conexiones automáticamente

"""

from sqlalchemy import create_engine, Engine, text
from etl.config import get_database_url


def get_engine() -> Engine:
    """
    Crea un engine de SQLAlchemy usando DATABASE_URL.

    pool_pre_ping=True: antes de usar una conexión del pool,
    verifica que siga viva. 
    """
    url = get_database_url()
    return create_engine(url, pool_pre_ping=True)


def test_connection(engine: Engine) -> None:
    """
    Verifica que la conexión funciona.
    """

    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        row = result.fetchone()
        if row and row[0] == 1:
            print("✓ Conexión a PostgreSQL OK")
        else:
            raise RuntimeError("La conexión respondió de forma inesperada")