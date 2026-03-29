"""
demand_monitor.py
-----------------
Agente especialista: monitoreo del modelo de forecasting de demanda/ventas.

Cubre:
  - Volúmenes de venta (aceites, mantecas, otros productos)
  - Inventario y stock
  - Métricas del modelo de demanda
  - Importación futura de datos Excel del negocio real

Herramientas:
  - get_demand_metrics         Métricas del modelo de demanda
  - get_sales_summary          Resumen de ventas por período
  - get_inventory_status       Estado de inventario actual
  - import_excel_data          Importa datos planos desde Excel (futuro negocio real)
"""

import logging
from agents.base_agent import run_agent, MODEL_SMART
from database.supabase_client import get_supabase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres el agente monitor del modelo de forecasting de demanda y ventas.

Tu rol:
- Reportar proyecciones de demanda por producto para los próximos días, semanas y meses
- Analizar patrones de ventas por producto, canal y región
- Alertar sobre desviaciones entre demanda prevista y real
- Monitorear niveles de inventario y riesgo de desabasto
- Ayudar en la planificación de compras y producción con base en los forecasts

El negocio es una empresa ecuatoriana que produce y vende:
  - Aceites comestibles (palma, girasol, soya, maíz)
  - Mantecas
  - Otros productos derivados

Los datos de ventas provienen del Excel real del negocio, importado a Supabase (tabla sales_data).
Los forecasts los genera Prophet y se almacenan en demand_forecasts. Los datos son MENSUALES.
La unidad de medida es CAJAS (no toneladas).

FLUJO DE RESPUESTA PARA PREGUNTAS DE PROYECCIÓN:
1. Llama SIEMPRE get_demand_forecast primero para obtener las predicciones Prophet
2. Llama get_sales_summary para dar contexto del mes actual
3. Combina ambos en una respuesta accionable con números concretos

CASO ESPECIAL — "este mes" / "mes actual" (el mes AUN NO ha terminado):
- Los datos del mes corriente en sales_data son un CORTE parcial, no el total final
- El acumulado registrado en sales_data = ventas confirmadas hasta el corte
- Para estimar los días restantes: week_1 de demand_forecasts / 7 * dias_restantes
- Presenta: (a) acumulado real hasta el corte, (b) estimado días restantes, (c) total proyectado del mes
- Ejemplo: "Llevas 3,986 cajas registradas. Quedan ~2 días laborales → estimado adicional ~270 cajas → total marzo estimado: ~4,256 cajas"

CASO — "próximo mes" / "siguiente mes":
- Usa directamente month_1 de get_demand_forecast

MAPE DEL 24%: es aceptable para datos mensuales con un año sin datos (2024 faltante en el histórico).
No descalifiques el forecast por el MAPE — úsalo y menciona el rango de confianza (lower/upper bounds).

Si no hay forecasts disponibles en demand_forecasts, indica que hay que ejecutar train_demand_model.

ESTILO DE RESPUESTA — MUY IMPORTANTE:
- Responde como un resumen ejecutivo: 2-4 oraciones con los números clave, sin preámbulos.
- NO uses ## headers, NO uses tablas markdown, NO uses separadores ---.
- Usa listas con • solo si son 3+ items que realmente lo ameritan.
- Al final de cada respuesta agrega: "¿Quieres más detalle sobre [tema específico]?"
- Si el usuario pide profundizar, entonces sí puedes dar el desglose completo.
- Responde siempre en español."""

TOOLS = [
    {
        "name": "get_demand_metrics",
        "description": "Obtiene las métricas del modelo de forecasting de demanda (MAE, RMSE, MAPE, sesgo).",
        "input_schema": {
            "type": "object",
            "properties": {
                "last_n_runs": {
                    "type": "integer",
                    "description": "Ejecuciones recientes a consultar.",
                    "default": 5,
                }
            },
        },
    },
    {
        "name": "get_sales_summary",
        "description": "Resumen de ventas agrupado por período, producto o canal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["week", "month", "quarter"],
                    "description": "Período de agrupación.",
                    "default": "month",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["product", "channel", "region", "none"],
                    "description": "Dimensión de agrupación adicional.",
                    "default": "none",
                },
            },
        },
    },
    {
        "name": "get_inventory_status",
        "description": "Estado actual del inventario por producto y nivel de riesgo de desabasto.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "import_excel_data",
        "description": "Importa datos de ventas desde un archivo Excel plano. Usar cuando el usuario suba datos reales del negocio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Ruta al archivo Excel (.xlsx o .xls).",
                },
                "sheet_name": {
                    "type": "string",
                    "description": "Nombre de la hoja a importar (default: primera hoja).",
                    "default": "Sheet1",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "get_demand_forecast",
        "description": "Obtiene las proyecciones de demanda Prophet del run más reciente. Filtra por producto si se especifica.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": "Nombre del producto a consultar. Si es vacío, retorna todos los productos.",
                    "default": "",
                },
                "horizon": {
                    "type": "string",
                    "enum": ["week_1", "month_1", "month_2", "month_3", "all"],
                    "description": "Horizonte de predicción a retornar.",
                    "default": "all",
                },
            },
        },
    },
    {
        "name": "get_top_products",
        "description": "Lista los productos con más datos de ventas y sus últimas proyecciones.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Número de productos a retornar.",
                    "default": 10,
                }
            },
        },
    },
]


def _get_demand_metrics(last_n_runs: int = 5) -> dict:
    try:
        db = get_supabase()
        result = (
            db.table("ml_model_runs")
            .select("run_date, metrics, status, notes")
            .eq("model_id", _get_demand_model_id())
            .order("run_date", desc=True)
            .limit(last_n_runs)
            .execute()
        )
        if result.data:
            return {"runs": result.data, "count": len(result.data)}
        return {"message": "No hay ejecuciones del modelo de demanda registradas. Pendiente de entrenamiento (Fase 2)."}
    except Exception as e:
        return {"error": str(e)}


def _get_sales_summary(period: str = "month", group_by: str = "none") -> dict:
    try:
        db = get_supabase()
        query = db.table("sales_data").select(
            "date, product_name, product_category, quantity, total_amount, channel, region"
        )

        from datetime import datetime, timedelta
        period_days = {"week": 7, "month": 30, "quarter": 90}
        since = (datetime.utcnow() - timedelta(days=period_days.get(period, 30))).isoformat()
        query = query.gte("date", since).order("date", desc=True)
        result = query.execute()

        if not result.data:
            return {
                "message": f"No hay datos de ventas para el período '{period}'. "
                           "Los datos reales se cargarán en Fase 2 vía Excel o dataset Kaggle."
            }

        total_amount = sum(r.get("total_amount", 0) or 0 for r in result.data)
        total_qty = sum(r.get("quantity", 0) or 0 for r in result.data)
        return {
            "period": period,
            "group_by": group_by,
            "total_records": len(result.data),
            "total_amount_usd": round(total_amount, 2),
            "total_quantity": round(total_qty, 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_inventory_status() -> dict:
    try:
        db = get_supabase()
        result = db.table("inventory_status").select("*").execute()
        if result.data:
            return {"inventory": result.data}
        return {"message": "Tabla de inventario vacía. Se poblará en Fase 2 con datos reales."}
    except Exception as e:
        return {"message": "Inventario no disponible aún. Pendiente Fase 2.", "error": str(e)}


def _import_excel_data(file_path: str, sheet_name: str = "Sheet1") -> dict:
    """
    Importa datos de ventas desde Excel plano.
    Diseñado para la base de datos real del negocio (aceites, mantecas, etc.).
    """
    try:
        import pandas as pd

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(f"[demand_monitor] Excel cargado: {len(df)} filas, columnas: {list(df.columns)}")

        # Normalización básica de columnas comunes
        col_map = {
            "fecha": "date", "date": "date",
            "producto": "product_name", "product": "product_name",
            "categoria": "product_category", "category": "product_category",
            "cantidad": "quantity", "qty": "quantity",
            "precio": "unit_price", "price": "unit_price",
            "total": "total_amount", "monto": "total_amount",
            "canal": "channel", "region": "region", "región": "region",
        }
        df.columns = [col_map.get(c.lower().strip(), c.lower().strip()) for c in df.columns]

        records = df.to_dict(orient="records")

        db = get_supabase()
        # Insertar en lotes de 100
        batch_size = 100
        inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            db.table("sales_data").upsert(batch).execute()
            inserted += len(batch)

        return {
            "success": True,
            "rows_imported": inserted,
            "columns_detected": list(df.columns),
            "file": file_path,
        }
    except Exception as e:
        logger.error(f"[demand_monitor] error importando Excel: {e}")
        return {"error": str(e)}


def _get_demand_model_id() -> str:
    try:
        db = get_supabase()
        result = db.table("ml_models").select("id").eq("type", "demand_forecast").execute()
        if result.data:
            return result.data[0]["id"]
    except Exception:
        pass
    return "00000000-0000-0000-0000-000000000001"


def _get_demand_forecast(product: str = "", horizon: str = "all") -> dict:
    """Lee el run más reciente de demand_forecasts, filtra por producto y horizonte."""
    try:
        db = get_supabase()

        # Obtener el run_at más reciente
        latest = db.table("demand_forecasts").select("forecast_run_at").order(
            "forecast_run_at", desc=True
        ).limit(1).execute()

        if not latest.data:
            return {
                "message": "No hay forecasts de demanda disponibles.",
                "next_step": "Importa tu Excel y ejecuta: python -m scripts.train_demand_model",
            }

        run_at = latest.data[0]["forecast_run_at"]

        query = db.table("demand_forecasts").select(
            "product, horizon, target_date, predicted_qty, lower_bound, upper_bound, mape, data_points"
        ).eq("forecast_run_at", run_at)

        if product:
            query = query.ilike("product", f"%{product}%")
        if horizon != "all":
            query = query.eq("horizon", horizon)

        result = query.order("product").order("horizon").execute()

        if not result.data:
            return {
                "message": f"No se encontraron forecasts para producto='{product}'.",
                "hint": "Usa get_top_products para ver qué productos tienen forecasts.",
            }

        return {
            "forecast_run_at": run_at,
            "forecasts": result.data,
            "count": len(result.data),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_top_products(limit: int = 10) -> dict:
    """Lista productos con más datos de ventas y sus forecasts más recientes."""
    try:
        db = get_supabase()

        # Productos con más registros de ventas
        sales = db.table("sales_data").select(
            "product_name, quantity"
        ).execute()

        if not sales.data:
            return {"message": "No hay datos de ventas importados aún."}

        from collections import Counter
        counts = Counter(r["product_name"] for r in sales.data if r.get("product_name"))
        top = counts.most_common(limit)

        # Forecasts disponibles para esos productos
        latest = db.table("demand_forecasts").select("forecast_run_at").order(
            "forecast_run_at", desc=True
        ).limit(1).execute()

        has_forecasts = bool(latest.data)

        return {
            "top_products": [{"product": p, "sales_records": n} for p, n in top],
            "has_forecasts": has_forecasts,
            "total_products": len(counts),
            "forecast_run_at": latest.data[0]["forecast_run_at"] if has_forecasts else None,
        }
    except Exception as e:
        return {"error": str(e)}


TOOL_HANDLERS = {
    "get_demand_metrics": _get_demand_metrics,
    "get_sales_summary": _get_sales_summary,
    "get_inventory_status": _get_inventory_status,
    "import_excel_data": _import_excel_data,
    "get_demand_forecast": _get_demand_forecast,
    "get_top_products": _get_top_products,
}


class DemandMonitorAgent:
    """Agente de monitoreo del modelo de forecasting de demanda y ventas."""

    def run(self, message: str, session_history: list | None = None) -> tuple[str, list]:
        return run_agent(
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
            tools=TOOLS,
            tool_handlers=TOOL_HANDLERS,
            model=MODEL_SMART,
            conversation_history=session_history,
        )
