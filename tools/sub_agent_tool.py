"""
sub_agent_tool.py
-----------------
Patrón subAgentTool(): permite que cualquier agente delegue tareas a otro agente
como si fuera una llamada a herramienta. Es la base de la jerarquía de agentes.

Uso:
    # Definir la herramienta para el agente padre
    tool_def = create_sub_agent_tool(
        agent_name="price_monitor",
        description="Monitorea el modelo de forecasting de precios de aceite de palma."
    )

    # Crear el handler que ejecuta el sub-agente
    runner = SubAgentRunner(agent_name="price_monitor", system_prompt=PRICE_MONITOR_PROMPT)
    handler = runner.as_handler()
"""

import logging
from typing import Callable
from agents.base_agent import run_agent, MODEL_SMART

logger = logging.getLogger(__name__)


def create_sub_agent_tool(agent_name: str, description: str) -> dict:
    """
    Crea la definición de herramienta Anthropic para invocar un sub-agente.

    Args:
        agent_name:   Nombre del agente (usado como nombre de la tool).
        description:  Descripción de lo que hace el agente (el modelo usará esto para decidir cuándo llamarlo).

    Returns:
        Definición de herramienta en formato Anthropic.
    """
    return {
        "name": f"call_{agent_name}",
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Descripción detallada de la tarea a delegar al sub-agente.",
                }
            },
            "required": ["task"],
        },
    }


class SubAgentRunner:
    """
    Encapsula un sub-agente listo para ser llamado como handler de herramienta.
    """

    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        tools: list[dict] | None = None,
        tool_handlers: dict[str, Callable] | None = None,
        model: str = MODEL_SMART,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.model = model

    def as_handler(self) -> Callable[[str], str]:
        """
        Devuelve una función que puede usarse como handler en tool_handlers del agente padre.
        """
        def handler(task: str) -> str:
            logger.info(f"[SubAgentRunner] llamando a '{self.agent_name}' con tarea: {task[:80]}...")
            response, _ = run_agent(
                system_prompt=self.system_prompt,
                user_message=task,
                tools=self.tools,
                tool_handlers=self.tool_handlers,
                model=self.model,
            )
            return response

        handler.__name__ = f"call_{self.agent_name}"
        return handler
