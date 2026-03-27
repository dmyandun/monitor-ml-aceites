"""
load_prices.py
--------------
Descarga e ingesta el histórico de precios del aceite de palma desde FRED
(Federal Reserve Bank of St. Louis) — serie PPOILUSDM (USD/tonelada métrica).

Fuente: https://fred.stlouisfed.org/series/PPOILUSDM
Frecuencia: mensual | Unidad: USD/MT | Desde: 2003

Ejecutar:
    python -m scripts.load_prices
    python -m scripts.load_prices --years 5    (solo últimos 5 años)
"""

import argparse
import logging
from datetime import datetime, date
from dotenv import load_dotenv
load_dotenv()

import httpx
from database.supabase_client import get_supabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PPOILUSDM"


def fetch_fred_prices(years: int | None = None) -> list[dict]:
    """Descarga la serie PPOILUSDM de FRED y devuelve lista de registros."""
    logger.info(f"Descargando datos de FRED: {FRED_URL}")

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(FRED_URL)
        response.raise_for_status()

    lines = response.text.strip().split("\n")
    records = []
    cutoff = None
    if years:
        cutoff = date(datetime.now().year - years, 1, 1)

    for line in lines[1:]:  # Skip header
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        date_str, price_str = parts
        try:
            record_date = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
            price = float(price_str.strip())
            if price <= 0:
                continue
            if cutoff and record_date < cutoff:
                continue
            records.append({
                "date": record_date.isoformat(),
                "actual_price": round(price, 2),
                "source": "FRED",
                "currency": "USD",
                "unit": "USD/MT",
            })
        except (ValueError, TypeError):
            continue

    logger.info(f"Registros obtenidos de FRED: {len(records)}")
    return records


def load_to_supabase(records: list[dict]) -> int:
    """Inserta registros en price_data (upsert por date+source)."""
    db = get_supabase()
    inserted = 0
    batch_size = 50

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        db.table("price_data").upsert(batch, on_conflict="date,source").execute()
        inserted += len(batch)
        logger.info(f"  Insertados {inserted}/{len(records)} registros...")

    return inserted


def run(years: int | None = None):
    records = fetch_fred_prices(years)
    if not records:
        logger.warning("No se obtuvieron registros de FRED.")
        return

    total = load_to_supabase(records)
    logger.info(f"\n✅ Carga completada: {total} registros en price_data (fuente: FRED).")

    # Mostrar rango de fechas cargado
    dates = [r["date"] for r in records]
    logger.info(f"Rango: {min(dates)} → {max(dates)}")
    logger.info(f"Último precio: ${records[-1]['actual_price']} USD/MT ({records[-1]['date']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carga histórico de precios de palma desde FRED")
    parser.add_argument("--years", type=int, default=None,
                        help="Cargar solo los últimos N años (default: todo desde 2003)")
    args = parser.parse_args()
    run(years=args.years)
