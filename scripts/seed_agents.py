"""
seed_agents.py
--------------
Pobla la tabla `agents` en Supabase con las configuraciones iniciales
de los 4 agentes del sistema.

Ejecutar una sola vez (o cuando se resetee la DB):
    python -m scripts.seed_agents
"""

import hashlib
import json
import logging
from dotenv import load_dotenv
load_dotenv()

from database.supabase_client import get_supabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_hash(data: dict) -> str:
    """Genera un version_hash del estado actual del agente."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


AGENTS = [
    {
        "name": "orchestrator",
        "description": "Clasifica intents del usuario y enruta al agente especialista correcto. Usa keyword matching (~0ms) con LLM Haiku como fallback.",
        "model": "claude-haiku-4-5-20251001",
        "system_prompt": "Clasificador de intents del sistema monitor-ml-aceites. Enruta mensajes a price_monitor, demand_monitor o agent_lab según el contenido.",
        "tools": [],
        "skills": ["keyword_matching", "llm_fallback"],
    },
    {
        "name": "price_monitor",
        "description": "Monitorea el modelo de forecasting de precios del aceite de palma. Analiza tendencias, anomalías, drift y métricas del modelo.",
        "model": "claude-sonnet-4-6",
        "system_prompt": (
            "Eres el agente monitor del modelo de forecasting de precios del aceite de palma. "
            "Reportas métricas del modelo, detectas anomalías y analizas tendencias de precio "
            "usando datos de FRED, World Bank y fuentes ecuatorianas (CFN, ANCUPA)."
        ),
        "tools": [
            "get_latest_price_metrics",
            "get_price_history",
            "detect_price_anomalies",
            "get_model_drift_status",
        ],
        "skills": [],
    },
    {
        "name": "demand_monitor",
        "description": "Monitorea el modelo de forecasting de demanda y ventas de aceites, mantecas y otros productos.",
        "model": "claude-sonnet-4-6",
        "system_prompt": (
            "Eres el agente monitor del modelo de forecasting de demanda y ventas. "
            "Analizas volúmenes de venta, inventario, canales y métricas del modelo. "
            "Soportas importación de datos reales desde Excel (aceites, mantecas)."
        ),
        "tools": [
            "get_demand_metrics",
            "get_sales_summary",
            "get_inventory_status",
            "import_excel_data",
        ],
        "skills": [],
    },
    {
        "name": "agent_lab",
        "description": "Meta-agente de mejora continua. Lee configs de todos los agentes, aplica fixes reactivos automáticamente y encola mejoras proactivas para aprobación humana.",
        "model": "claude-sonnet-4-6",
        "system_prompt": (
            "Eres Agent Lab, el meta-agente de mejora continua del sistema monitor-ml-aceites. "
            "Cada noche: scrapeás research de AI, evaluás relevancia, leés configs de agentes, "
            "aplicás fixes reactivos y encolás cambios proactivos para aprobación vía Telegram."
        ),
        "tools": [
            "list_agents",
            "get_agent_config",
            "search_research",
            "create_recommendation",
            "apply_reactive_fix",
            "check_skill_bloat",
        ],
        "skills": ["reactive_autonomy", "proactive_recommendations", "skill_bloat_monitor"],
    },
]


def seed():
    db = get_supabase()
    seeded = 0
    updated = 0

    for agent in AGENTS:
        version_hash = make_hash(agent)
        record = {**agent, "version_hash": version_hash}

        # Verificar si ya existe
        existing = db.table("agents").select("id, version_hash").eq("name", agent["name"]).execute()

        if existing.data:
            if existing.data[0]["version_hash"] != version_hash:
                db.table("agents").update(record).eq("name", agent["name"]).execute()
                logger.info(f"  ↻ actualizado: {agent['name']}")
                updated += 1
            else:
                logger.info(f"  ✓ sin cambios: {agent['name']}")
        else:
            db.table("agents").insert(record).execute()
            logger.info(f"  + creado: {agent['name']}")
            seeded += 1

    logger.info(f"\nSeed completado: {seeded} creados, {updated} actualizados.")


if __name__ == "__main__":
    logger.info("Iniciando seed de agentes...")
    seed()
