"""
base_agent.py
-------------
Core del framework de agentes. Implementa runAgent() con loop de tool-calling
contra la API de Anthropic, memoria de conversación por sesión, y soporte para
delegación jerárquica vía subAgentTool().
"""

import logging
from typing import Callable, Any
import anthropic

logger = logging.getLogger(__name__)

# Modelos disponibles
MODEL_FAST = "claude-haiku-4-5-20251001"    # Clasificación, tareas simples
MODEL_SMART = "claude-sonnet-4-6"           # Agentes especialistas, Agent Lab


def run_agent(
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    tool_handlers: dict[str, Callable[..., Any]],
    model: str = MODEL_SMART,
    conversation_history: list[dict] | None = None,
    max_iterations: int = 15,
) -> tuple[str, list[dict]]:
    """
    Ejecuta un agente con loop de tool-calling hasta obtener respuesta final.

    Args:
        system_prompt:        Instrucciones del agente.
        user_message:         Mensaje del usuario para esta llamada.
        tools:                Lista de definiciones de herramientas (formato Anthropic).
        tool_handlers:        Dict {nombre_tool: función_handler}.
        model:                Modelo a usar (default: sonnet-4-6).
        conversation_history: Historial previo de la sesión (se modifica in-place).
        max_iterations:       Límite de ciclos para evitar loops infinitos.

    Returns:
        (respuesta_final: str, historial_actualizado: list)
    """
    client = anthropic.Anthropic()

    messages = conversation_history if conversation_history is not None else []
    messages.append({"role": "user", "content": user_message})

    for iteration in range(max_iterations):
        logger.debug(f"[run_agent] iteración {iteration + 1}, modelo={model}")

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=tools if tools else [],
            messages=messages,
        )

        # Agregar respuesta del asistente al historial
        messages.append({"role": "assistant", "content": response.content})

        # Sin tool calls → respuesta final
        if response.stop_reason == "end_turn":
            final_text = _extract_text(response.content)
            logger.debug(f"[run_agent] respuesta final obtenida en iteración {iteration + 1}")
            return final_text, messages

        # Procesar tool calls
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.debug(f"[run_agent] ejecutando tool: {block.name}")
                handler = tool_handlers.get(block.name)

                if handler:
                    try:
                        result = handler(**block.input)
                    except Exception as e:
                        result = f"Error ejecutando {block.name}: {str(e)}"
                        logger.error(f"[run_agent] error en tool {block.name}: {e}")
                else:
                    result = f"Herramienta '{block.name}' no encontrada."
                    logger.warning(f"[run_agent] tool no encontrada: {block.name}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

            messages.append({"role": "user", "content": tool_results})

    logger.warning(f"[run_agent] límite de iteraciones alcanzado ({max_iterations})")
    return "Se alcanzó el límite de iteraciones sin obtener respuesta final.", messages


def _extract_text(content_blocks: list) -> str:
    """Extrae el texto de los bloques de contenido de la respuesta."""
    parts = []
    for block in content_blocks:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()
