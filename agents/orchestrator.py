"""
orchestrator.py
---------------
Punto de entrada único del sistema. Clasifica el intent del mensaje entrante
y lo enruta al agente especialista correcto.

Clasificación en dos niveles (según el PDF):
  1. Fast path: keyword matching (gratis, ~0ms). Cubre ~70% de los mensajes.
  2. Slow path: LLM fallback con Haiku si el score de keywords es insuficiente.
"""

import logging
import re
from dataclasses import dataclass, field
from agents.base_agent import run_agent, MODEL_FAST

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.6  # Score mínimo para rutear por keywords


@dataclass
class AgentRoute:
    """Definición de un agente disponible para routing."""
    name: str
    description: str
    keywords: list[str]
    weight: float = 1.0


# Registro de agentes y sus palabras clave
AGENT_REGISTRY: list[AgentRoute] = [
    AgentRoute(
        name="price_monitor",
        description="Monitorea el modelo de forecasting de precios del aceite de palma. Analiza tendencias, alertas de precio, drift del modelo y métricas de forecasting.",
        keywords=[
            "precio", "price", "palma", "palm", "cotizaci", "commodity",
            "tendencia", "trend", "forecast", "pron\u00f3stico", "pronostico",
            "fred", "world bank", "imf", "mercado", "mercados", "coste", "costo",
            "barrel", "tonel", "usd", "d\u00f3lar", "dolar",
            # Commodities adicionales
            "girasol", "sunflower", "sfo",
            "soya", "soja", "soybean", "sbo",
            "ma\u00edz", "maiz", "corn", "cno",
            "achiote", "annatto", "ann",
            "palmiste", "palmiste", "pko",
            "cpo", "crudo",
        ],
        weight=1.2,
    ),
    AgentRoute(
        name="demand_monitor",
        description="Monitorea el modelo de forecasting de demanda y ventas. Analiza volúmenes, inventario, canales, regiones y métricas del modelo.",
        keywords=[
            "demanda", "demand", "venta", "sales", "inventario", "inventory",
            "stock", "pedido", "order", "canal", "channel", "regi\u00f3n", "region",
            "producci\u00f3n", "produccion", "production", "distribuci\u00f3n", "distribucion",
            "aceite", "manteca", "liter", "litro", "kilo", "kg", "unidad",
            "fmcg", "sku", "producto", "product",
        ],
        weight=1.2,
    ),
    AgentRoute(
        name="agent_lab",
        description="Meta-agente que gestiona mejoras del sistema. Revisa recomendaciones pendientes, aplica fixes y reporta el estado del sistema.",
        keywords=[
            "mejora", "improve", "update", "actualizar", "recomendar", "recomendaci\u00f3n",
            "recomendacion", "lab", "agente", "agent", "config", "configuraci\u00f3n",
            "configuracion", "sistema", "system", "fix", "arreglar", "optimizar",
            "pendiente", "pending", "aprobar", "approve", "rechazar", "reject",
        ],
        weight=1.0,
    ),
]

ORCHESTRATOR_SYSTEM_PROMPT = """Eres el clasificador de intents del sistema monitor-ml-aceites.
Tu única tarea es determinar a qué agente debe ir el mensaje del usuario.

Agentes disponibles:
- price_monitor: preguntas sobre precios de commodities (palma, girasol, soya, maíz, achiote, palmiste), forecasting de precios, tendencias de mercado
- demand_monitor: preguntas sobre ventas, demanda, inventario, producción, canales de distribución
- agent_lab: gestión del sistema de agentes, recomendaciones, mejoras, estado del sistema

Responde ÚNICAMENTE con el nombre exacto del agente (price_monitor, demand_monitor, o agent_lab).
Sin explicación. Sin texto adicional."""


class Orchestrator:
    """
    Enruta mensajes al agente especialista correcto.
    Usa keyword matching primero; si el score es bajo, usa Haiku como fallback.
    """

    def __init__(self):
        self._invocation_counts: dict[str, int] = {r.name: 0 for r in AGENT_REGISTRY}

    def classify(self, message: str) -> str:
        """
        Clasifica el intent del mensaje y devuelve el nombre del agente destino.
        """
        # Fast path: keyword matching
        agent_name, score = self._keyword_match(message)

        if score >= SCORE_THRESHOLD:
            logger.info(f"[Orchestrator] fast path → {agent_name} (score={score:.2f})")
            self._invocation_counts[agent_name] += 1
            return agent_name

        # Slow path: LLM fallback
        logger.info(f"[Orchestrator] score bajo ({score:.2f}), usando LLM fallback")
        agent_name = self._llm_classify(message)
        self._invocation_counts[agent_name] += 1
        return agent_name

    def _keyword_match(self, message: str) -> tuple[str, float]:
        """
        Calcula score por palabras clave para cada agente.
        Devuelve (mejor_agente, score).
        """
        text = message.lower()
        best_agent = AGENT_REGISTRY[0].name
        best_score = 0.0

        for route in AGENT_REGISTRY:
            matches = sum(1 for kw in route.keywords if kw in text)
            if route.keywords:
                score = (matches / len(route.keywords)) * route.weight * 10
                # Normalizar a [0, 1]
                score = min(score, 1.0)
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_agent = route.name

        return best_agent, best_score

    def _llm_classify(self, message: str) -> str:
        """
        Usa claude-haiku para clasificar el intent cuando keywords no alcanzan el threshold.
        """
        try:
            response, _ = run_agent(
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
                user_message=message,
                tools=[],
                tool_handlers={},
                model=MODEL_FAST,
                max_iterations=1,
            )
            agent_name = response.strip().lower()

            valid_names = {r.name for r in AGENT_REGISTRY}
            if agent_name in valid_names:
                logger.info(f"[Orchestrator] LLM fallback → {agent_name}")
                return agent_name

            logger.warning(f"[Orchestrator] LLM devolvió nombre inválido: '{agent_name}', usando demand_monitor")
        except Exception as e:
            logger.error(f"[Orchestrator] error en LLM fallback: {e}")

        return "demand_monitor"  # Default seguro

    def get_stats(self) -> dict:
        """Devuelve estadísticas de invocaciones por agente."""
        return dict(self._invocation_counts)
