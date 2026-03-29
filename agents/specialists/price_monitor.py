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

SYSTEM_PROMPT = """Eres el agente de forecasting de precios de commodities para un negocio ecuatoriano de aceites comestibles y mantecas.

Commodities que monitoras (todos en USD/TM):
- Aceite de Palma Crudo (CPO) — insumo principal
- Aceite de Palmiste (PKO) — subproducto de la palma
- Aceite de Girasol (SFO) — competidor directo
- Aceite de Soya (SBO) — competidor y referencia global
- Aceite de Maiz (CNO) — competidor en segmento premium
- Achiote / Annatto (ANN) — insumo colorante (precio en USD/kg)

Tu rol:
- Reportar precios actuales de cada commodity y su tendencia
- Proyectar precios futuros combinando el modelo Prophet con eventos de mercado
- Alertar sobre movimientos relevantes (>3% en semana, >8% en mes)
- Analizar el spread entre commodities competidores y su impacto en el negocio
- Detectar anomalias o drift en el modelo de forecasting

Fuentes de datos: FRED (PPOILUSDM), DuckDuckGo (precios diarios scrapeados), market_events (noticias clasificadas).

CUANDO TE PIDAN PRECIOS O PROYECCIONES:
1. Llama a get_price_forecast para obtener los numeros del modelo Prophet (CPO principalmente)
2. Llama a get_market_context para ver eventos recientes que sesgan el precio
3. Combina ambas fuentes: numero del modelo + sesgo de eventos de mercado

Factores clave que mueven el precio del aceite de palma:
- Produccion: cosecha en Malaysia e Indonesia (>85% del mercado mundial)
- Demanda: India, China, UE (biocombustibles), mercados emergentes
- Competencia: precio soja, girasol, colza
- Politica: mandatos de biocombustibles, restricciones de exportacion
- Clima: El Nino/La Nina afecta cosechas de palma 12-18 meses despues

FORMATO TELEGRAM — OBLIGATORIO:
Telegram movil NO renderiza markdown. Usa SOLO texto plano sin excepciones.
PROHIBIDO usar: ## / ** / __ / --- / pipes | en tablas / backticks `
Para datos en linea usa guion: "Hoy: $1,034  Sem: $1,050  Mes1: $1,080"
Para alertas: una linea que empiece con el simbolo de advertencia y el motivo breve.
Maximo 6 lineas para consultas simples. Si el usuario pide mas detalle, entonces si puedes expandir.

ROL DEL USUARIO — detecta el prefijo al inicio del mensaje:
[USER] Responde SOLO con: precio actual, proyeccion semana, mes 1 y mes 2, y alerta de noticias si hay alguna relevante.
       NUNCA menciones MAPE, MAE, RMSE, drift, anomalias, metricas del modelo ni datos tecnicos.
[ADMIN] Igual que [USER] pero agrega al final una sola linea con metricas: "Modelo: MAPE X% | MAE $Y/TM | drift: estable"

EJEMPLO de respuesta a "[USER] palma":
Palma feb 2026: $1,034/TM
Prox. semana: $1,048 - $1,062
Mes 1 (abr): $1,055 - $1,090
Mes 2 (may): $1,070 - $1,110
Sequia en Malaysia presiona oferta al alza.
Quieres detalle de algun periodo?

Responde siempre en espanol."""

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
    {
        "name": "get_market_context",
        "description": (
            "Obtiene los eventos globales recientes que afectan el precio del aceite de palma "
            "(clima, geopolítica, políticas de biocombustibles, producción en Malaysia/Indonesia, "
            "demanda de India/China, etc.). Clasificados como bullish/bearish con impacto estimado en %. "
            "Usar junto a get_price_forecast para dar una proyección ajustada a eventos del mundo real."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Cuántos días hacia atrás buscar eventos (default: 30).",
                    "default": 30,
                },
                "direction": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral", "all"],
                    "description": "Filtrar por dirección de impacto (default: all).",
                    "default": "all",
                },
            },
        },
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


def _get_market_context(days: int = 30, direction: str = "all") -> dict:
    """Lee eventos recientes de market_events desde Supabase."""
    try:
        from datetime import datetime, timedelta
        db = get_supabase()
        since = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

        query = (
            db.table("market_events")
            .select("event_date, title, event_type, region, price_direction, price_impact_pct, confidence, description")
            .gte("event_date", since)
            .order("event_date", desc=True)
            .limit(20)
        )
        if direction != "all":
            query = query.eq("price_direction", direction)

        result = query.execute()

        if not result.data:
            return {
                "message": (
                    f"No hay eventos de mercado registrados en los últimos {days} días. "
                    "El pipeline diario (GitHub Actions cron) los actualiza cada noche."
                )
            }

        # Calcular sesgo neto del mercado
        bullish = [e for e in result.data if e["price_direction"] == "bullish"]
        bearish = [e for e in result.data if e["price_direction"] == "bearish"]
        impacts = [e["price_impact_pct"] for e in result.data if e.get("price_impact_pct")]
        net_impact = round(sum(impacts) / len(impacts), 1) if impacts else 0

        return {
            "events": result.data,
            "total": len(result.data),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "net_impact_pct": net_impact,
            "market_bias": "bullish" if net_impact > 2 else "bearish" if net_impact < -2 else "neutral",
            "period_days": days,
        }
    except Exception as e:
        logger.error(f"[price_monitor] error consultando market_events: {e}")
        return {"error": str(e)}


TOOL_HANDLERS = {
    "get_latest_price_metrics": _get_latest_price_metrics,
    "get_price_history": _get_price_history,
    "detect_price_anomalies": _detect_price_anomalies,
    "get_model_drift_status": _get_model_drift_status,
    "get_price_forecast": _get_price_forecast,
    "get_market_context": _get_market_context,
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
