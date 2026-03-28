"""
bot.py
------
Integración con Telegram via webhook.
El Space de HuggingFace expone el endpoint POST /telegram-webhook.
Telegram envía cada mensaje a esa URL.

Setup inicial (una sola vez):
    python telegram/bot.py --setup

Esto registra el webhook en Telegram apuntando a tu Space URL.
"""

import os
import logging
import asyncio
import argparse
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import httpx
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logger = logging.getLogger(__name__)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_URL = os.environ.get("TELEGRAM_WEBHOOK_URL", "")


class TelegramBot:
    """
    Maneja mensajes de Telegram y los enruta al orquestador de agentes.
    """

    def __init__(self, orchestrator=None, agent_map: dict | None = None):
        """
        Args:
            orchestrator: instancia de Orchestrator para clasificar intents
            agent_map:    dict {nombre_agente: instancia_agente}
        """
        self.orchestrator = orchestrator
        self.agent_map = agent_map or {}
        self.sessions: dict[int, list] = {}  # user_id → historial de conversación

    async def process_update(self, update_data: dict) -> str | None:
        """
        Procesa un update recibido desde el webhook de Telegram.
        Devuelve la respuesta generada o None si no hay nada que responder.
        """
        try:
            bot = Bot(token=TOKEN)
            update = Update.de_json(update_data, bot)

            if not update.message or not update.message.text:
                return None

            user_id = update.message.from_user.id
            user_name = update.message.from_user.first_name or "usuario"
            text = update.message.text.strip()

            logger.info(f"[TelegramBot] mensaje de {user_name} ({user_id}): {text[:60]}")

            # Comandos especiales
            if text.startswith("/"):
                response = await self._handle_command(text, user_id, user_name)
            else:
                response = await self._handle_message(text, user_id)

            if response:
                try:
                    await bot.send_message(
                        chat_id=update.message.chat_id,
                        text=response,
                        parse_mode="Markdown",
                    )
                except Exception:
                    # Si falla el Markdown (caracteres especiales), enviar sin formato
                    await bot.send_message(
                        chat_id=update.message.chat_id,
                        text=response,
                    )

            return response

        except Exception as e:
            logger.error(f"[TelegramBot] error procesando update: {e}")
            return None

    async def _handle_message(self, text: str, user_id: int) -> str:
        """Enruta el mensaje al agente correcto y devuelve la respuesta."""
        if not self.orchestrator:
            return "Sistema no inicializado. Contacta al administrador."

        # Clasificar intent (sync → thread para no bloquear el event loop)
        try:
            agent_name = await asyncio.to_thread(self.orchestrator.classify, text)
        except Exception as e:
            logger.error(f"[TelegramBot] error clasificando mensaje: {e}", exc_info=True)
            return "No pude clasificar tu consulta. Por favor intenta de nuevo."

        agent = self.agent_map.get(agent_name)
        if not agent:
            return f"Agente '{agent_name}' no disponible en este momento."

        logger.info(f"[TelegramBot] enrutando a '{agent_name}'")

        # Obtener/crear historial de sesión
        session_history = self.sessions.get(user_id, [])

        # Ejecutar agente (sync → thread para no bloquear el event loop)
        try:
            response, updated_history = await asyncio.to_thread(
                agent.run, text, session_history
            )
            # Guardar historial (últimos 20 mensajes para no crecer infinito)
            self.sessions[user_id] = updated_history[-20:]
            return response
        except Exception as e:
            logger.error(f"[TelegramBot] error en agente '{agent_name}': {e}", exc_info=True)
            return (
                f"⚠️ Error al procesar tu consulta con el agente `{agent_name}`.\n"
                f"Detalle: `{type(e).__name__}: {str(e)[:120]}`"
            )

    async def _handle_command(self, command: str, user_id: int, user_name: str) -> str:
        """Maneja comandos especiales de Telegram."""
        cmd = command.split()[0].lower()

        if cmd == "/start":
            return (
                f"Hola {user_name}! 👋\n\n"
                "Soy el monitor de ML para aceites comestibles.\n\n"
                "Puedo ayudarte con:\n"
                "• Precios del aceite de palma\n"
                "• Forecasting de demanda y ventas\n"
                "• Estado del sistema de agentes\n\n"
                "¿Qué quieres consultar?"
            )

        elif cmd == "/help":
            return (
                "*Comandos disponibles:*\n\n"
                "/start — Bienvenida\n"
                "/help — Esta ayuda\n"
                "/status — Estado del sistema\n"
                "/pending — Recomendaciones pendientes de aprobación\n"
                "/reset — Limpiar historial de conversación\n\n"
                "*Consultas en lenguaje natural:*\n"
                "• ¿Cuál es el precio actual del aceite de palma?\n"
                "• ¿Cómo están las ventas del último mes?\n"
                "• ¿Hay alguna anomalía en el modelo?"
            )

        elif cmd == "/status":
            stats = self.orchestrator.get_stats() if self.orchestrator else {}
            lines = ["*Estado del sistema:* ✅\n"]
            lines.append("*Agentes activos:*")
            for name, count in stats.items():
                lines.append(f"  • {name}: {count} consultas")
            return "\n".join(lines)

        elif cmd == "/pending":
            return await self._get_pending_recommendations()

        elif cmd == "/reset":
            self.sessions.pop(user_id, None)
            return "Historial de conversación limpiado. ¿En qué puedo ayudarte?"

        elif cmd.startswith("/approve_"):
            rec_id = cmd.replace("/approve_", "")
            return await self._approve_recommendation(rec_id)

        elif cmd.startswith("/reject_"):
            rec_id = cmd.replace("/reject_", "")
            return await self._reject_recommendation(rec_id)

        return f"Comando '{cmd}' no reconocido. Usa /help para ver los disponibles."

    async def _get_pending_recommendations(self) -> str:
        """Muestra las recomendaciones proactivas pendientes de aprobación."""
        try:
            from database.supabase_client import get_supabase
            db = get_supabase()
            result = (
                db.table("agent_lab_recommendations")
                .select("id, title, target_agent, priority, created_at")
                .eq("status", "pending")
                .eq("type", "proactive")
                .order("priority", desc=True)
                .limit(10)
                .execute()
            )

            if not result.data:
                return "No hay recomendaciones pendientes de aprobación."

            lines = [f"*Recomendaciones pendientes ({len(result.data)}):*\n"]
            for rec in result.data:
                short_id = rec["id"][:8]
                lines.append(
                    f"📋 *{rec['title']}*\n"
                    f"   Agente: {rec['target_agent']} | Prioridad: {rec['priority']}\n"
                    f"   /approve_{short_id} | /reject_{short_id}\n"
                )
            return "\n".join(lines)

        except Exception as e:
            return f"Error consultando recomendaciones: {e}"

    async def _approve_recommendation(self, rec_id_prefix: str) -> str:
        try:
            from database.supabase_client import get_supabase
            db = get_supabase()
            result = (
                db.table("agent_lab_recommendations")
                .select("id, title")
                .ilike("id", f"{rec_id_prefix}%")
                .execute()
            )
            if not result.data:
                return f"Recomendación {rec_id_prefix} no encontrada."

            rec = result.data[0]
            db.table("agent_lab_recommendations").update(
                {"status": "approved"}
            ).eq("id", rec["id"]).execute()

            return f"✅ Recomendación aprobada: *{rec['title']}*\nSe aplicará en el próximo ciclo de Agent Lab."
        except Exception as e:
            return f"Error aprobando recomendación: {e}"

    async def _reject_recommendation(self, rec_id_prefix: str) -> str:
        try:
            from database.supabase_client import get_supabase
            db = get_supabase()
            result = (
                db.table("agent_lab_recommendations")
                .select("id, title")
                .ilike("id", f"{rec_id_prefix}%")
                .execute()
            )
            if not result.data:
                return f"Recomendación {rec_id_prefix} no encontrada."

            rec = result.data[0]
            db.table("agent_lab_recommendations").update(
                {"status": "rejected"}
            ).eq("id", rec["id"]).execute()

            return f"❌ Recomendación rechazada: *{rec['title']}*"
        except Exception as e:
            return f"Error rechazando recomendación: {e}"


async def setup_webhook():
    """Registra el webhook de Telegram apuntando al Space de HuggingFace."""
    if not TOKEN or not WEBHOOK_URL:
        print("ERROR: Configura TELEGRAM_BOT_TOKEN y TELEGRAM_WEBHOOK_URL en .env")
        return

    webhook_endpoint = f"{WEBHOOK_URL.rstrip('/')}/telegram-webhook"
    url = f"https://api.telegram.org/bot{TOKEN}/setWebhook"

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"url": webhook_endpoint})
        data = resp.json()

    if data.get("ok"):
        print(f"✅ Webhook registrado: {webhook_endpoint}")
    else:
        print(f"❌ Error: {data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Registrar webhook en Telegram")
    args = parser.parse_args()

    if args.setup:
        asyncio.run(setup_webhook())
