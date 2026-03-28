"""
price_monitor.py
----------------
Agente especialista: monitoreo del modelo de forecasting de precios
del aceite de palma (datos FRED / World Bank).

Herramientas disponibles:
  - get_latest_price_metrics   Últimas métricas del modelo (MAE, RMSE, MAPE)
  - get_price_history          Serie histórica de precios
  - detect_price_anomalies     Detecta anomalías en predicciones recientes
  - get_model_drift_status     Estado de drift del modelo
"""

import logging
from agents.base_agent import run_agent, MODEL_SMART
from database.supabase_client import get_supabase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres el agente monitor del modelo de forecasting de precios del aceite de palma.

Tu rol:
- Reportar el estado actual del modelo de predicción de precios
- Detectar y alertar sobre anomalías, drift de datos o degradación de métricas
- Explicar las tendencias de precio y su impacto en el negocio
- Sugerir acciones cuando el modelo necesita reentrenamiento

El negocio es una empresa ecuatoriana de producción y venta de aceites comestibles y mantecas.
El precio mundial del aceite de palma (USD/tonelada métrica) es el insumo principal del modelo.
Fuentes de datos: FRED (PPOILUSDM), World Bank Pink Sheet, CFN Ecuador, ANCUPA.

Cuando reportes métricas, sé específico con números. Cuando detectes problemas, prioriza claridad.
Responde siempre en español."""

# Definiciones de herramientas
TOOLS = [
    {
        "name": "get_latest_price_metrics",
        "description": "Obtiene las últimas métricas de rendimiento del modelo de forecasting de precios (MAE, RMSE, MAPE, R²).",
        "input_schema": {
            "type": "object",
            "properties": {
                "last_n_runs": {
                    "type": "integer",
                    "description": "Número de ejecuciones recientes a consultar (default: 5).",
                    "default": 5,
                }
            },
        },
    },
    {
        "name": "get_price_history",
        "description": "Obtiene la serie histórica de precios del aceite de palma y las predicciones del modelo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "months": {
                    "type": "integer",
                    "description": "Meses hacia atrás a consultar (default: 12).",
                    "default": 12,
                }
            },
        },
    },
    {
        "name": "detect_price_anomalies",
        "description": "Analiza las predicciones recientes y detecta anomalías o desviaciones inusuales.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_model_drift_status",
        "description": "Evalúa si hay drift en los datos de entrada o en la distribución de predicciones del modelo.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_price_forecast",
        "description": (
            "Obtiene el forecast más reciente del precio del aceite de palma para los 4 horizontes: "
            "mañana (tomorrow), semana siguiente (next_week), mes 1 (month_1) y mes 2 (month_2). "
            "Incluye precio predicho, intervalo de confianza 80% y MAPE del modelo."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# Handlers de herramientas
def _get_latest_price_metrics(last_n_runs: int = 5) -> dict:
    """Consulta métricas del modelo en Supabase."""
    try:
        db = get_supabase()
        result = (
            db.table("ml_model_runs")
            .select("run_date, metrics, status, notes")
            .eq("model_id", _get_price_model_id())
            .order("run_date", desc=True)
            .limit(last_n_runs)
            .execute()
        )
        if result.data:
            return {"runs": result.data, "count": len(result.data)}
        return {"message": "No hay ejecuciones registradas aún. El modelo no ha sido entrenado todavía."}
    except Exception as e:
        logger.error(f"[price_monitor] error consultando métricas: {e}")
        return {"error": str(e), "message": "No se pudieron obtener las métricas. Verifica la conexión a Supabase."}


def _get_price_history(months: int = 12) -> dict:
    """Consulta histórico de precios en Supabase."""
    try:
        db = get_supabase()
        from datetime import datetime, timedelta
        since = (datetime.utcnow() - timedelta(days=months * 30)).isoformat()
        result = (
            db.table("price_data")
            .select("date, actual_price, predicted_price, source")
            .gte("date", since)
            .order("date", desc=False)
            .execute()
        )
        if result.data:
            return {"records": result.data, "count": len(result.data), "months": months}
        return {"message": f"No hay datos de precios para los últimos {months} meses. Pendiente de cargar datos de FRED/World Bank."}
    except Exception as e:
        return {"error": str(e)}


def _detect_price_anomalies() -> dict:
    """Detección simple de anomalías en predicciones recientes."""
    try:
        db = get_supabase()
        result = (
            db.table("price_data")
            .select("date, actual_price, predicted_price")
            .order("date", desc=True)
            .limit(30)
            .execute()
        )
        if not result.data:
            return {"message": "No hay datos suficientes para detectar anomalías."}

        anomalies = []
        for row in result.data:
            if row.get("actual_price") and row.get("predicted_price"):
                error_pct = abs(row["actual_price"] - row["predicted_price"]) / row["actual_price"] * 100
                if error_pct > 15:  # Umbral: 15% de error
                    anomalies.append({
                        "date": row["date"],
                        "actual": row["actual_price"],
                        "predicted": row["predicted_price"],
                        "error_pct": round(error_pct, 2),
                    })
        return {"anomalies": anomalies, "total_checked": len(result.data)}
    except Exception as e:
        return {"error": str(e)}


def _get_price_forecast() -> dict:
    """Obtiene el forecast más reciente desde Supabase (generado por train_price_model.py)."""
    try:
        db = get_supabase()

        # Obtener el run más reciente
        latest_run = (
            db.table("price_forecasts")
            .select("forecast_run_at")
            .order("forecast_run_at", desc=True)
            .limit(1)
            .execute()
        )
        if not latest_run.data:
            return {
                "message": (
                    "No hay forecasts disponibles. El modelo aún no ha sido entrenado. "
                    "Ejecuta: python -m scripts.train_price_model"
                )
            }

        run_at = latest_run.data[0]["forecast_run_at"]

        # Obtener todos los horizontes de ese run
        result = (
            db.table("price_forecasts")
            .select("horizon, target_date, predicted_price, lower_bound, upper_bound, mape, data_points")
            .eq("forecast_run_at", run_at)
            .order("target_date", desc=False)
            .execute()
        )

        if not result.data:
            return {"message": "Forecast vacío."}

        horizon_order = {"tomorrow": 1, "next_week": 2, "month_1": 3, "month_2": 4}
        forecasts = sorted(result.data, key=lambda x: horizon_order.get(x["horizon"], 9))

        return {
            "forecast_generated_at": run_at,
            "model_mape": forecasts[0].get("mape"),
            "data_points_used": forecasts[0].get("data_points"),
            "forecasts": forecasts,
        }
    except Exception as e:
        logger.error(f"[price_monitor] error consultando forecast: {e}")
        return {"error": str(e)}


def _get_model_drift_status() -> dict:
    """Estado de drift del modelo (placeholder hasta tener modelo real)."""
    try:
        db = get_supabase()
        result = (
            db.table("ml_models")
            .select("name, metrics, last_evaluated")
            .eq("type", "price_forecast")
            .execute()
        )
        if result.data:
            return {"model": result.data[0]}
        return {"message": "Modelo de forecasting de precios aún no registrado. Fase 2 del proyecto."}
    except Exception as e:
        return {"error": str(e)}


def _get_price_model_id() -> str:
    """Obtiene el ID del modelo de precios en Supabase."""
    try:
        db = get_supabase()
        result = db.table("ml_models").select("id").eq("type", "price_forecast").execute()
        if result.data:
            return result.data[0]["id"]
    except Exception:
        pass
    return "00000000-0000-0000-0000-000000000000"  # Placeholder


TOOL_HANDLERS = {
    "get_latest_price_metrics": _get_latest_price_metrics,
    "get_price_history": _get_price_history,
    "detect_price_anomalies": _detect_price_anomalies,
    "get_model_drift_status": _get_model_drift_status,
    "get_price_forecast": _get_price_forecast,
}


class PriceMonitorAgent:
    """Agente de monitoreo del modelo de forecasting de precios."""

    def run(self, message: str, session_history: list | None = None) -> tuple[str, list]:
        return run_agent(
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
            tools=TOOLS,
            tool_handlers=TOOL_HANDLERS,
            model=MODEL_SMART,
            conversation_history=session_history,
        )
