"""
config.py
---------
Centraliza toda la configuración del proyecto.
Un solo lugar para cambiar rutas, URLs, y constantes.

"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent

# Carga el archivo .env en la raíz del proyecto.
# Si no existe, no falla: usa las variables de entorno del sistema.
load_dotenv(PROJECT_ROOT / ".env")


def get_database_url() -> str:
    """
    Devuelve la URL de conexión a PostgreSQL.
    Lanza un error claro si no está configurada,
    en vez de un error críptico más adelante.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL no está definida. "
        )
    return url


def get_csv_dir() -> Path:
    """
    Devuelve la carpeta donde están los CSVs.
    Si no está configurada, usa ./data por defecto.
    """
    raw = os.environ.get("CSV_DIR", str(PROJECT_ROOT / "data"))
    path = Path(raw)
    if not path.is_absolute():
        # Si la ruta es relativa, la resolvemos desde la raíz del proyecto
        path = PROJECT_ROOT / path
    return path


# Nombres exactos de los CSVs de Olist → tabla destino en raw.*
CSV_TO_TABLE: dict[str, str] = {
    "olist_orders_dataset.csv":              "raw.orders",
    "olist_order_items_dataset.csv":         "raw.order_items",
    "olist_order_payments_dataset.csv":      "raw.order_payments",
    "olist_customers_dataset.csv":           "raw.customers",
    "olist_products_dataset.csv":            "raw.products",
    "olist_sellers_dataset.csv":             "raw.sellers",
    "product_category_name_translation.csv": "raw.product_category_translation",
    "olist_order_reviews_dataset.csv":       "raw.order_reviews",
}