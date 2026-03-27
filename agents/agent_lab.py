"""
agent_lab.py
------------
Meta-agente de auto-mejora. Cada noche (via GitHub Actions cron):
  1. Lee los findings de research recientes
  2. Revisa las configs de todos los agentes
  3. Aplica mejoras REACTIVAS automáticamente
  4. Encola mejoras PROACTIVAS para aprobación humana vía Telegram

Autonomía:
  REACTIVA (auto-aplica):
    - Corrección de errores recurrentes en tool calls
    - Anti-patrones en system prompts
    - Memorias de feedback basadas en errores claros

  PROACTIVA (requiere aprobación):
    - Agregar nuevas herramientas a agentes
    - Restructurar system prompts significativamente
    - Cambiar modelo (haiku ↔ sonnet)
    - Agregar nuevos sub-agentes
"""

import logging
from agents.base_agent import run_agent, MODEL_SMART
from database.supabase_client import get_supabase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres Agent Lab, el meta-agente de mejora continua del sistema monitor-ml-aceites.

Tu misión diaria:
1. Consultar los findings de investigación recientes (relevancia > 0.5)
2. Leer la configuración de cada agente del sistema
3. Generar recomendaciones específicas y accionables
4. Auto-aplicar las REACTIVAS, encolar las PROACTIVAS para aprobación humana

## Autonomía

### REACTIVA (aplica inmediatamente, sin pedir permiso):
- Corrección de patrones de error en tool calls
- Mejora de instrucciones poco claras en system prompts
- Adición de memorias de feedback basadas en errores documentados

### PROACTIVA (encola para aprobación humana):
- Agregar nuevas herramientas a agentes existentes
- Restructuración significativa de system prompts
- Cambio de modelo tier (haiku → sonnet o viceversa)
- Nuevos agentes especialistas
- Adopción de técnicas de research con score > 0.8

## Principios
- Toda recomendación debe tener un finding de research como respaldo (trazabilidad)
- Cada config de agente tiene un version_hash; no re-apliques si no cambió
- Monitorea skill bloat: alerta si un agente tiene 5+ herramientas, recomienda split en 8+
- Sé conservador: menos cambios con más impacto es mejor que muchos cambios pequeños

Responde en español."""

TOOLS = [
    {
        "name": "list_agents",
        "description": "Lista todos los agentes registrados en el sistema con su nombre y descripción.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_agent_config",
        "description": "Lee la configuración completa de un agente: system prompt, herramientas, skills, version hash.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Nombre del agente."}
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "search_research",
        "description": "Busca findings de investigación recientes filtrados por score mínimo de relevancia.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_score": {
                    "type": "number",
                    "description": "Score mínimo de relevancia (0.0 a 1.0).",
                    "default": 0.5,
                },
                "limit": {
                    "type": "integer",
                    "description": "Máximo de resultados.",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "create_recommendation",
        "description": "Crea una recomendación de mejora para un agente. Tipo 'reactive' se auto-aplica; 'proactive' espera aprobación humana.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["reactive", "proactive"]},
                "category": {"type": "string", "description": "Categoría: 'prompt', 'tool', 'model', 'skill', 'memory'"},
                "target_agent": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "rationale": {"type": "string", "description": "Justificación basada en research o evidencia."},
                "priority": {"type": "integer", "default": 5, "description": "1 (baja) a 10 (crítica)"},
            },
            "required": ["type", "category", "target_agent", "title", "description", "rationale"],
        },
    },
    {
        "name": "apply_reactive_fix",
        "description": "Aplica inmediatamente una mejora reactiva a un agente (corrección de prompt, memoria de feedback, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation_id": {"type": "string"},
                "agent_name": {"type": "string"},
                "change_type": {"type": "string", "enum": ["prompt_fix", "memory_update", "skill_update"]},
                "before_state": {"type": "string"},
                "after_state": {"type": "string"},
            },
            "required": ["recommendation_id", "agent_name", "change_type", "after_state"],
        },
    },
    {
        "name": "check_skill_bloat",
        "description": "Verifica si algún agente tiene demasiadas herramientas (alerta en 5+, split recomendado en 8+).",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _list_agents() -> dict:
    try:
        db = get_supabase()
        result = db.table("agents").select("name, description, updated_at").execute()
        if result.data:
            return {"agents": result.data}
        # Fallback: agentes hardcoded si aún no hay DB
        return {
            "agents": [
                {"name": "price_monitor", "description": "Monitoreo forecasting de precios"},
                {"name": "demand_monitor", "description": "Monitoreo forecasting de demanda"},
                {"name": "agent_lab", "description": "Meta-agente de mejora"},
                {"name": "orchestrator", "description": "Clasificación de intents y routing"},
            ]
        }
    except Exception as e:
        return {"error": str(e)}


def _get_agent_config(agent_name: str) -> dict:
    try:
        db = get_supabase()
        result = (
            db.table("agents")
            .select("*")
            .eq("name", agent_name)
            .execute()
        )
        if result.data:
            return result.data[0]
        return {"message": f"Agente '{agent_name}' no encontrado en la base de datos."}
    except Exception as e:
        return {"error": str(e)}


def _search_research(min_score: float = 0.5, limit: int = 20) -> dict:
    try:
        db = get_supabase()
        result = (
            db.table("agent_lab_research")
            .select("title, summary, relevance_score, source_url, tags, created_at")
            .gte("relevance_score", min_score)
            .order("relevance_score", desc=True)
            .limit(limit)
            .execute()
        )
        if result.data:
            return {"findings": result.data, "count": len(result.data)}
        return {"message": "No hay findings de research aún. Ejecutar el cron de investigación primero."}
    except Exception as e:
        return {"error": str(e)}


def _create_recommendation(
    type: str,
    category: str,
    target_agent: str,
    title: str,
    description: str,
    rationale: str,
    priority: int = 5,
) -> dict:
    try:
        db = get_supabase()
        result = db.table("agent_lab_recommendations").insert({
            "type": type,
            "category": category,
            "target_agent": target_agent,
            "title": title,
            "description": description,
            "rationale": rationale,
            "priority": priority,
            "status": "pending",
        }).execute()
        return {"success": True, "id": result.data[0]["id"] if result.data else None}
    except Exception as e:
        return {"error": str(e)}


def _apply_reactive_fix(
    recommendation_id: str,
    agent_name: str,
    change_type: str,
    after_state: str,
    before_state: str = "",
) -> dict:
    try:
        db = get_supabase()
        # Registrar en audit log
        db.table("agent_lab_changes").insert({
            "recommendation_id": recommendation_id,
            "change_type": "auto",
            "target_agent": agent_name,
            "description": f"Fix reactivo: {change_type}",
            "before_state": {"content": before_state},
            "after_state": {"content": after_state},
        }).execute()

        # Marcar recomendación como aplicada
        db.table("agent_lab_recommendations").update(
            {"status": "applied"}
        ).eq("id", recommendation_id).execute()

        return {"success": True, "applied": change_type, "agent": agent_name}
    except Exception as e:
        return {"error": str(e)}


def _check_skill_bloat() -> dict:
    try:
        db = get_supabase()
        result = db.table("agents").select("name, tools").execute()
        if not result.data:
            return {"message": "No hay agentes registrados en DB aún."}

        report = []
        for agent in result.data:
            tools = agent.get("tools") or []
            count = len(tools) if isinstance(tools, list) else 0
            status = "ok"
            if count >= 8:
                status = "split_recommended"
            elif count >= 5:
                status = "flag"
            report.append({"agent": agent["name"], "tool_count": count, "status": status})

        return {"bloat_report": report}
    except Exception as e:
        return {"error": str(e)}


TOOL_HANDLERS = {
    "list_agents": _list_agents,
    "get_agent_config": _get_agent_config,
    "search_research": _search_research,
    "create_recommendation": _create_recommendation,
    "apply_reactive_fix": _apply_reactive_fix,
    "check_skill_bloat": _check_skill_bloat,
}


class AgentLabAgent:
    """Meta-agente de mejora continua del sistema."""

    def run(self, message: str, session_history: list | None = None) -> tuple[str, list]:
        return run_agent(
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
            tools=TOOLS,
            tool_handlers=TOOL_HANDLERS,
            model=MODEL_SMART,
            conversation_history=session_history,
        )

    def run_daily_cycle(self) -> str:
        """
        Ejecuta el ciclo completo de mejora diaria.
        Llamado por el cron de GitHub Actions.
        """
        logger.info("[AgentLab] iniciando ciclo diario de mejora")
        response, _ = self.run(
            "Ejecuta el ciclo diario completo: "
            "1) Revisa findings de research con score > 0.5. "
            "2) Lista todos los agentes y revisa sus configs. "
            "3) Genera recomendaciones específicas. "
            "4) Aplica las reactivas. "
            "5) Verifica skill bloat. "
            "6) Resume qué cambió y qué queda pendiente de aprobación."
        )
        logger.info("[AgentLab] ciclo diario completado")
        return response
