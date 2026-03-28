"""
train_price_model.py
--------------------
Entrena el modelo Prophet con datos históricos de Supabase
y almacena los forecasts resultantes en la tabla price_forecasts.

El modelo NO se persiste en disco (HF Spaces tiene filesystem efímero).
Los FORECASTS sí se persisten en Supabase → el bot/dashboard los lee de ahí.

Ejecutar manualmente o via GitHub Actions cron (mensual recomendado):
    python -m scripts.train_price_model

Output en Supabase:
    - price_forecasts: 4 filas nuevas (tomorrow, next_week, month_1, month_2)
    - ml_models: métricas actualizadas (mape, last_evaluated)
    - ml_model_runs: registro de la ejecución
"""

import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Silenciar logs verbosos de cmdstanpy/prophet
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

from database.supabase_client import get_supabase
from models.price_forecast import PriceForecastModel


def load_price_data() -> list[dict]:
    """Carga todos los registros de price_data desde Supabase."""
    db = get_supabase()
    result = (
        db.table("price_data")
        .select("date, actual_price, source")
        .not_.is_("actual_price", "null")
        .order("date", desc=False)
        .execute()
    )
    records = result.data or []
    logger.info(f"Registros cargados de Supabase: {len(records)}")

    # Si hay datos diarios (commodities-api) y mensuales (FRED), combinar
    # Preferir datos diarios cuando haya solapamiento
    seen_dates = {}
    for r in records:
        d = r["date"]
        # Datos diarios tienen prioridad sobre mensuales
        if d not in seen_dates or r["source"] == "commodities-api":
            seen_dates[d] = r

    deduped = sorted(seen_dates.values(), key=lambda x: x["date"])
    if len(deduped) < len(records):
        logger.info(f"Después de deduplicar (prioridad datos diarios): {len(deduped)} registros")

    return deduped


def save_forecasts(forecasts: dict, mape: float | None, n_records: int):
    """Almacena los forecasts en Supabase."""
    db = get_supabase()
    now = datetime.utcnow().isoformat()

    rows = []
    for horizon, data in forecasts.items():
        rows.append({
            "forecast_run_at": now,
            "target_date": data["target_date"],
            "horizon": horizon,
            "predicted_price": data["predicted_price"],
            "lower_bound": data["lower_bound"],
            "upper_bound": data["upper_bound"],
            "model_name": "palm_price_forecast",
            "data_points": n_records,
            "mape": mape,
        })

    db.table("price_forecasts").insert(rows).execute()
    logger.info(f"Forecasts guardados en Supabase: {len(rows)} horizontes")

    for h, d in forecasts.items():
        logger.info(
            f"  {h:12s}: ${d['predicted_price']:8.2f} "
            f"[${d['lower_bound']:.2f} – ${d['upper_bound']:.2f}] "
            f"→ {d['target_date']}"
        )


def update_model_metadata(metrics: dict, n_records: int):
    """Actualiza métricas del modelo en ml_models y registra en ml_model_runs."""
    db = get_supabase()

    # Obtener model_id
    result = db.table("ml_models").select("id").eq("name", "palm_price_forecast").execute()
    if not result.data:
        logger.warning("Modelo 'palm_price_forecast' no encontrado en ml_models.")
        return
    model_id = result.data[0]["id"]

    # Actualizar métricas en ml_models
    db.table("ml_models").update({
        "metrics": metrics,
        "last_evaluated": datetime.utcnow().isoformat(),
    }).eq("id", model_id).execute()

    # Registrar ejecución en ml_model_runs
    db.table("ml_model_runs").insert({
        "model_id": model_id,
        "run_date": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "data_source": "FRED+commodities-api",
        "rows_processed": n_records,
        "status": "success",
        "notes": f"Entrenamiento Prophet. MAPE: {metrics.get('mape')}%",
    }).execute()

    logger.info(f"Metadata actualizada — MAPE: {metrics.get('mape')}%, MAE: ${metrics.get('mae')}")


def run():
    logger.info("=== Entrenamiento modelo de precios de palma ===")
    start = datetime.utcnow()

    # 1. Cargar datos
    records = load_price_data()
    if len(records) < 12:
        logger.error(f"Datos insuficientes: {len(records)} registros. Mínimo 12.")
        return

    # 2. Entrenar
    model = PriceForecastModel()
    metrics = model.train(records)

    if "error" in metrics:
        logger.error(f"Error en entrenamiento: {metrics['error']}")
        return

    # 3. Predecir
    forecasts = model.predict()

    # 4. Guardar forecasts en Supabase
    save_forecasts(forecasts, mape=metrics.get("mape"), n_records=model.n_records)

    # 5. Actualizar metadata del modelo
    update_model_metadata(metrics, n_records=model.n_records)

    elapsed = (datetime.utcnow() - start).seconds
    logger.info(f"=== Completado en {elapsed}s ===")
    logger.info(f"    Datos: {model.n_records} registros")
    logger.info(f"    MAPE:  {metrics.get('mape')}%")
    logger.info(f"    MAE:   ${metrics.get('mae')} USD/MT")


if __name__ == "__main__":
    run()
