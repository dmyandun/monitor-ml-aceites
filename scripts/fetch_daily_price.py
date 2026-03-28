"""
fetch_daily_price.py
--------------------
Descarga precios del aceite de palma (CPO) desde Commodities-API.

ESTRATEGIA:
  - Durante el trial (7 días): correr con --bulk para descargar TODO el histórico diario
    disponible. Esto siembra la base de datos con datos de alta frecuencia para Prophet.
  - Después del trial: usar --today para el precio del día (si se renueva) o simplemente
    dejar que FRED/World Bank sirvan de actualización mensual.

Uso:
    # Descarga histórica completa (usar durante el trial de 7 días)
    python -m scripts.fetch_daily_price --bulk

    # Solo el precio de hoy
    python -m scripts.fetch_daily_price --today

    # Rango específico (YYYY-MM-DD)
    python -m scripts.fetch_daily_price --from 2020-01-01 --to 2024-12-31

Variable de entorno requerida:
    COMMODITIES_API_KEY=tu_api_key_aqui

API docs: https://commodities-api.com/documentation
Símbolo CPO: Crude Palm Oil (USD por tonelada métrica × factor de conversión)
"""

import os
import logging
import argparse
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

import httpx
from database.supabase_client import get_supabase

API_KEY = os.environ.get("COMMODITIES_API_KEY", "")
BASE_URL = "https://commodities-api.com/api"

# CPO cotiza en USD/tonelada métrica en commodities-api (verificar factor al recibir datos)
SYMBOL = "CPO"


def _fetch_latest() -> dict | None:
    """Obtiene el precio más reciente de CPO."""
    if not API_KEY:
        raise EnvironmentError("COMMODITIES_API_KEY no configurada en .env")

    url = f"{BASE_URL}/latest"
    params = {"access_key": API_KEY, "base": "USD", "symbols": SYMBOL}

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if not data.get("success"):
        error = data.get("error", {})
        raise RuntimeError(f"API error: {error.get('type')} — {error.get('info')}")

    rate = data["data"]["rates"].get(SYMBOL)
    if not rate:
        raise RuntimeError(f"Símbolo {SYMBOL} no encontrado en la respuesta")

    # La API devuelve 1/precio cuando base=USD: rates[CPO] = cantidad de CPO por 1 USD
    # Convertir a USD/MT: precio = 1 / rate (si rate < 1)
    # Verificar el valor: precio palma mundial ~ $600-1200 USD/MT
    price_usd_mt = _normalize_price(rate)

    return {
        "date": date.today().isoformat(),
        "actual_price": round(price_usd_mt, 2),
        "source": "commodities-api",
        "currency": "USD",
        "unit": "USD/MT",
    }


def _fetch_historical(start: date, end: date) -> list[dict]:
    """
    Descarga histórico diario de CPO entre start y end.
    La API solo permite 1 día por llamada en el endpoint /historical,
    pero tiene un endpoint /timeseries para rangos (verificar en docs de tu plan).
    """
    if not API_KEY:
        raise EnvironmentError("COMMODITIES_API_KEY no configurada en .env")

    records = []
    current = start

    # Intentar primero con endpoint timeseries (más eficiente)
    try:
        records = _fetch_timeseries(start, end)
        if records:
            return records
    except Exception as e:
        logger.warning(f"Timeseries falló ({e}), intentando día a día...")

    # Fallback: día a día
    total_days = (end - start).days + 1
    logger.info(f"Descargando {total_days} días individualmente ({start} → {end})...")

    while current <= end:
        try:
            record = _fetch_single_day(current)
            if record:
                records.append(record)
        except Exception as e:
            logger.warning(f"Error en {current}: {e}")

        current += timedelta(days=1)

        # Rate limiting: pausa breve cada 10 requests
        if len(records) % 10 == 0 and records:
            import time
            time.sleep(0.5)

    return records


def _fetch_timeseries(start: date, end: date) -> list[dict]:
    """Endpoint timeseries: obtiene rango en una sola llamada."""
    url = f"{BASE_URL}/timeseries"
    params = {
        "access_key": API_KEY,
        "base": "USD",
        "symbols": SYMBOL,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }

    with httpx.Client(timeout=60) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if not data.get("success"):
        error = data.get("error", {})
        raise RuntimeError(f"Timeseries error: {error.get('type')} — {error.get('info')}")

    rates_by_date = data["data"]["rates"]
    records = []
    for date_str, rates in rates_by_date.items():
        rate = rates.get(SYMBOL)
        if rate:
            price_usd_mt = _normalize_price(rate)
            records.append({
                "date": date_str,
                "actual_price": round(price_usd_mt, 2),
                "source": "commodities-api",
                "currency": "USD",
                "unit": "USD/MT",
            })

    logger.info(f"Timeseries: {len(records)} registros descargados")
    return records


def _fetch_single_day(target_date: date) -> dict | None:
    """Obtiene precio para un día específico."""
    url = f"{BASE_URL}/historical/{target_date.isoformat()}"
    params = {"access_key": API_KEY, "base": "USD", "symbols": SYMBOL}

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        if resp.status_code == 422:
            return None  # Fecha no disponible (fin de semana/feriado)
        resp.raise_for_status()
        data = resp.json()

    if not data.get("success"):
        return None

    rate = data["data"]["rates"].get(SYMBOL)
    if not rate:
        return None

    return {
        "date": target_date.isoformat(),
        "actual_price": round(_normalize_price(rate), 2),
        "source": "commodities-api",
        "currency": "USD",
        "unit": "USD/MT",
    }


def _normalize_price(rate: float) -> float:
    """
    Commodities-API puede devolver el precio en diferentes formas según el plan.
    El precio de palma debe estar entre $400 y $1500 USD/MT.
    Si rate < 1, probablemente es 1/precio → invertir.
    Si rate está en rango correcto, usar directamente.
    """
    if 400 <= rate <= 1500:
        return rate
    elif rate > 0 and rate < 1:
        return 1.0 / rate
    elif rate > 1500:
        # Podría estar en USD/10MT o unidad diferente
        # CPO puede cotizar en diferentes unidades según la fuente
        # Loguear para verificación manual
        logger.warning(f"Precio CPO inusual: {rate} — verifica la unidad en la respuesta API")
        return rate
    else:
        return rate


def load_to_supabase(records: list[dict]) -> int:
    """Inserta/actualiza registros en price_data."""
    db = get_supabase()
    inserted = 0
    batch_size = 50

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        db.table("price_data").upsert(batch, on_conflict="date,source").execute()
        inserted += len(batch)
        logger.info(f"  Insertados {inserted}/{len(records)} registros...")

    return inserted


def run_today():
    """Descarga solo el precio de hoy."""
    record = _fetch_latest()
    if not record:
        logger.error("No se obtuvo precio.")
        return

    logger.info(f"Precio hoy ({record['date']}): ${record['actual_price']} USD/MT")
    total = load_to_supabase([record])
    logger.info(f"✅ {total} registro insertado en price_data.")


def run_bulk(start_year: int = 2018):
    """
    Descarga TODO el histórico disponible desde start_year hasta hoy.
    Usar durante el trial para sembrar la base de datos.
    """
    start = date(start_year, 1, 1)
    end = date.today()
    logger.info(f"Iniciando bulk download: {start} → {end}")

    records = _fetch_historical(start, end)
    if not records:
        logger.error("No se obtuvieron registros.")
        return

    # Filtrar precios inválidos
    valid = [r for r in records if 200 <= r["actual_price"] <= 2000]
    logger.info(f"Registros válidos: {len(valid)}/{len(records)}")

    if valid:
        total = load_to_supabase(valid)
        dates = [r["date"] for r in valid]
        logger.info(f"✅ Bulk completado: {total} registros")
        logger.info(f"   Rango: {min(dates)} → {max(dates)}")
        logger.info(f"   Último precio: ${valid[-1]['actual_price']} USD/MT")


def run_range(from_date: str, to_date: str):
    """Descarga rango específico."""
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)
    records = _fetch_historical(start, end)

    if records:
        valid = [r for r in records if 200 <= r["actual_price"] <= 2000]
        total = load_to_supabase(valid)
        logger.info(f"✅ {total} registros insertados ({from_date} → {to_date})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch precios CPO desde Commodities-API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--today", action="store_true", help="Solo el precio de hoy")
    group.add_argument("--bulk", action="store_true", help="Histórico completo desde 2018")
    group.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD",
                       help="Fecha inicio (usar con --to)")
    parser.add_argument("--to", dest="to_date", metavar="YYYY-MM-DD",
                        help="Fecha fin (usar con --from)", default=date.today().isoformat())
    parser.add_argument("--start-year", type=int, default=2018,
                        help="Año inicio para --bulk (default: 2018)")

    args = parser.parse_args()

    if args.today:
        run_today()
    elif args.bulk:
        run_bulk(start_year=args.start_year)
    elif args.from_date:
        run_range(args.from_date, args.to_date)
