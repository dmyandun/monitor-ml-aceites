"""
app.py
------
Entry point del HuggingFace Space (Docker).
FastAPI puro — sin Gradio, más liviano y siempre activo.

Endpoints:
  POST /telegram-webhook  → recibe updates de Telegram
  GET  /                  → dashboard HTML de estado
  GET  /health            → health check (UptimeRobot, etc.)

Para ejecutar localmente:
    python app.py

Para configurar el webhook de Telegram (una sola vez):
    python telegram/bot.py --setup
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
import uvicorn

# ── Agentes ───────────────────────────────────────────────
from agents.orchestrator import Orchestrator
from agents.specialists.price_monitor import PriceMonitorAgent
from agents.specialists.demand_monitor import DemandMonitorAgent
from agents.agent_lab import AgentLabAgent
from tg_bot.bot import TelegramBot

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
    logger.info("monitor-ml-aceites iniciado")
    yield
    logger.info("monitor-ml-aceites detenido")

app = FastAPI(
    title="monitor-ml-aceites",
    description="Sistema multiagente de monitoreo ML para aceites comestibles",
    lifespan=lifespan,
)


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Recibe updates de Telegram vía webhook."""
    try:
        data = await request.json()
        asyncio.create_task(telegram_bot.process_update(data))
        return Response(content="ok", status_code=200)
    except Exception as e:
        logger.error(f"[webhook] error: {e}")
        return Response(content="error", status_code=500)


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "service": "monitor-ml-aceites"}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Dashboard HTML de estado del sistema."""
    stats = orchestrator.get_stats()
    rows = "".join(
        f"<tr><td>{name}</td><td>{count}</td><td>✅ activo</td></tr>"
        for name, count in stats.items()
    )
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Monitor ML — Aceites</title>
  <style>
    body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }}
    h1 {{ color: #2d6a4f; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #2d6a4f; color: white; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    .badge {{ background: #d8f3dc; color: #1b4332; padding: 3px 8px; border-radius: 12px; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>🫒 Monitor ML — Aceites Comestibles</h1>
  <p>Sistema multiagente de monitoreo ML para producción y venta de aceites en Ecuador.</p>
  <p>Interfaz principal: <strong>Telegram bot</strong></p>

  <h2>Agentes activos</h2>
  <table>
    <tr><th>Agente</th><th>Consultas</th><th>Estado</th></tr>
    {rows}
  </table>

  <h2>Fuentes de datos</h2>
  <ul>
    <li><strong>Precio palma</strong>: FRED (PPOILUSDM), World Bank Pink Sheet, CFN Ecuador</li>
    <li><strong>Ventas</strong>: FMCG Kaggle → datos reales Excel (Fase 2)</li>
    <li><strong>Ecuador</strong>: ANCUPA, USDA FAS PSD Online</li>
  </ul>

  <p style="margin-top:40px; color:#999; font-size:0.85em;">
    <a href="/docs">API docs</a> · <a href="/health">health</a>
  </p>
</body>
</html>"""


# ── Punto de entrada local ────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("APP_ENV") == "development",
    )
