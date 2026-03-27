"""
app.py
------
Entry point del HuggingFace Space.
Combina:
  - FastAPI: webhook de Telegram (/telegram-webhook)
  - Gradio: dashboard de monitoreo (montado en /)

Para ejecutar localmente:
    python app.py

Para configurar el webhook de Telegram (una sola vez):
    python telegram/bot.py --setup
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from fastapi import FastAPI, Request, Response
import uvicorn

# ── Agentes ───────────────────────────────────────────────
from agents.orchestrator import Orchestrator
from agents.specialists.price_monitor import PriceMonitorAgent
from agents.specialists.demand_monitor import DemandMonitorAgent
from agents.agent_lab import AgentLabAgent
from telegram.bot import TelegramBot

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Inicialización de agentes ─────────────────────────────
orchestrator = Orchestrator()
agent_map = {
    "price_monitor": PriceMonitorAgent(),
    "demand_monitor": DemandMonitorAgent(),
    "agent_lab": AgentLabAgent(),
}
telegram_bot = TelegramBot(orchestrator=orchestrator, agent_map=agent_map)

# ── FastAPI ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("monitor-ml-aceitess iniciado")
    yield
    logger.info("monitor-ml-aceitess detenido")

fastapi_app = FastAPI(
    title="monitor-ml-aceitess",
    description="Sistema multiagente de monitoreo ML para aceites comestibles",
    lifespan=lifespan,
)


@fastapi_app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Recibe updates de Telegram vía webhook."""
    try:
        data = await request.json()
        asyncio.create_task(telegram_bot.process_update(data))
        return Response(content="ok", status_code=200)
    except Exception as e:
        logger.error(f"[webhook] error: {e}")
        return Response(content="error", status_code=500)


@fastapi_app.get("/health")
def health():
    """Endpoint de salud para UptimeRobot."""
    return {"status": "ok", "service": "monitor-ml-aceitess"}


# ── Gradio Dashboard ──────────────────────────────────────
def build_dashboard() -> gr.Blocks:
    with gr.Blocks(
        title="Monitor ML — Aceites",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 900px; margin: auto; }",
    ) as demo:

        gr.Markdown("# Monitor ML — Aceites Comestibles")
        gr.Markdown(
            "Sistema multiagente para monitoreo de modelos de forecasting de "
            "precios y demanda de aceites comestibles (Ecuador)."
        )

        with gr.Tabs():

            # ── Tab: Chat con agentes ──────────────────────
            with gr.TabItem("Chat con Agentes"):
                gr.Markdown("Consulta directamente a los agentes especialistas.")

                chatbot = gr.Chatbot(height=400, label="Conversación")
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ej: ¿Cuál es el estado del modelo de precios?",
                        label="Mensaje",
                        scale=4,
                    )
                    send_btn = gr.Button("Enviar", variant="primary", scale=1)

                session_state = gr.State([])

                def chat(message: str, history: list, session: list):
                    if not message.strip():
                        return history, session, ""

                    agent_name = orchestrator.classify(message)
                    agent = agent_map.get(agent_name)

                    if agent:
                        response, updated_session = agent.run(message, session[-20:])
                    else:
                        response = f"Agente '{agent_name}' no disponible."
                        updated_session = session

                    history.append((message, f"[{agent_name}] {response}"))
                    return history, updated_session, ""

                send_btn.click(
                    chat,
                    inputs=[msg_input, chatbot, session_state],
                    outputs=[chatbot, session_state, msg_input],
                )
                msg_input.submit(
                    chat,
                    inputs=[msg_input, chatbot, session_state],
                    outputs=[chatbot, session_state, msg_input],
                )

            # ── Tab: Estado del sistema ────────────────────
            with gr.TabItem("Estado del Sistema"):
                gr.Markdown("### Agentes registrados")

                def get_system_status():
                    stats = orchestrator.get_stats()
                    rows = [[name, count, "✅ activo"] for name, count in stats.items()]
                    return rows

                status_table = gr.Dataframe(
                    headers=["Agente", "Consultas", "Estado"],
                    value=get_system_status(),
                    interactive=False,
                )
                refresh_btn = gr.Button("Actualizar")
                refresh_btn.click(get_system_status, outputs=status_table)

            # ── Tab: Recomendaciones Agent Lab ─────────────
            with gr.TabItem("Recomendaciones Agent Lab"):
                gr.Markdown(
                    "Recomendaciones proactivas generadas por Agent Lab "
                    "que requieren aprobación humana."
                )

                def get_recommendations():
                    try:
                        from database.supabase_client import get_supabase
                        db = get_supabase()
                        result = (
                            db.table("agent_lab_recommendations")
                            .select("title, target_agent, type, status, priority, created_at")
                            .order("created_at", desc=True)
                            .limit(20)
                            .execute()
                        )
                        if result.data:
                            return [[
                                r["title"], r["target_agent"], r["type"],
                                r["status"], r["priority"], r["created_at"][:10]
                            ] for r in result.data]
                    except Exception as e:
                        logger.warning(f"No se pudieron cargar recomendaciones: {e}")
                    return [["Sin datos", "—", "—", "—", "—", "—"]]

                rec_table = gr.Dataframe(
                    headers=["Título", "Agente", "Tipo", "Estado", "Prioridad", "Fecha"],
                    value=get_recommendations(),
                    interactive=False,
                )
                refresh_rec_btn = gr.Button("Actualizar")
                refresh_rec_btn.click(get_recommendations, outputs=rec_table)

            # ── Tab: Info ──────────────────────────────────
            with gr.TabItem("Acerca de"):
                gr.Markdown("""
## monitor-ml-aceitess

Sistema multiagente basado en Claude (Anthropic) para monitoreo de modelos ML
aplicados a la producción y venta de aceites comestibles en Ecuador.

### Agentes
| Agente | Rol |
|--------|-----|
| `price_monitor` | Forecasting de precios del aceite de palma (FRED/World Bank) |
| `demand_monitor` | Forecasting de demanda y ventas (aceites, mantecas) |
| `agent_lab` | Meta-agente de auto-mejora continua |
| `orchestrator` | Clasificación de intents y routing |

### Fuentes de datos
- **Precio palma**: FRED (PPOILUSDM), World Bank Pink Sheet, CFN Ecuador
- **Ventas**: FMCG Kaggle (proxy) → datos reales del negocio en Excel (Fase 2)
- **Ecuador**: ANCUPA, USDA FAS PSD Online

### Arquitectura
Desplegado en HuggingFace Spaces (Gradio + FastAPI).
Interfaz principal vía Telegram bot (webhook).
Base de datos: Supabase.
Cron diario de research: GitHub Actions.
                """)

    return demo


# ── Montar Gradio sobre FastAPI ───────────────────────────
dashboard = build_dashboard()
app = gr.mount_gradio_app(fastapi_app, dashboard, path="/")


# ── Punto de entrada local ────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("APP_ENV") == "development",
    )
