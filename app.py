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
from database.supabase_client import get_supabase
from agents.orchestrator import Orchestrator
from agents.specialists.price_monitor import PriceMonitorAgent
from agents.specialists.demand_monitor import DemandMonitorAgent
from agents.agent_lab import AgentLabAgent
from tg_bot.bot import TelegramBot

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suprimir logs de httpx para no exponer tokens de Telegram en URLs
logging.getLogger("httpx").setLevel(logging.WARNING)
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


@app.get("/api/status")
def api_status():
    """API JSON con estado completo del sistema para el dashboard."""
    from datetime import datetime, timezone
    stats = orchestrator.get_stats()

    agents_data = []
    try:
        db = get_supabase()
        db_agents = db.table("agents").select("name, model, updated_at, tools, skills").execute()
        db_map = {a["name"]: a for a in (db_agents.data or [])}
    except Exception:
        db_map = {}

    agent_meta = {
        "orchestrator": {"icon": "🔀", "desc": "Clasifica intents y enruta mensajes"},
        "price_monitor": {"icon": "📈", "desc": "Forecasting de precios del aceite de palma"},
        "demand_monitor": {"icon": "📦", "desc": "Forecasting de demanda y ventas"},
        "agent_lab": {"icon": "🔬", "desc": "Meta-agente de mejora continua"},
    }

    for name, calls in stats.items():
        meta = agent_meta.get(name, {"icon": "🤖", "desc": ""})
        db_info = db_map.get(name, {})
        agents_data.append({
            "name": name,
            "icon": meta["icon"],
            "desc": meta["desc"],
            "calls": calls,
            "model": db_info.get("model", "—"),
            "tools": len(db_info.get("tools") or []),
            "in_db": name in db_map,
            "updated_at": db_info.get("updated_at", "—"),
        })

    # Recomendaciones pendientes
    pending_recs = []
    try:
        db = get_supabase()
        recs = db.table("agent_lab_recommendations").select(
            "title, target_agent, priority, created_at"
        ).eq("status", "pending").order("priority", desc=True).limit(5).execute()
        pending_recs = recs.data or []
    except Exception:
        pass

    # Último precio cargado
    last_price = None
    try:
        db = get_supabase()
        p = db.table("price_data").select("date, actual_price, source").order("date", desc=True).limit(1).execute()
        last_price = p.data[0] if p.data else None
    except Exception:
        pass

    # Findings de research
    research_count = 0
    try:
        db = get_supabase()
        r = db.table("agent_lab_research").select("id", count="exact").execute()
        research_count = r.count or 0
    except Exception:
        pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents": agents_data,
        "pending_recommendations": pending_recs,
        "last_price": last_price,
        "research_count": research_count,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Dashboard visual tipo n8n — auto-refresh cada 30s."""
    return """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Monitor ML — Aceites</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f1117; color: #e2e8f0; min-height: 100vh; }
    header { background: #1a1d2e; border-bottom: 1px solid #2d3748; padding: 16px 24px;
             display: flex; align-items: center; justify-content: space-between; }
    header h1 { font-size: 1.2rem; font-weight: 600; color: #68d391; }
    header .subtitle { font-size: 0.8rem; color: #718096; margin-top: 2px; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #68d391;
                  display: inline-block; margin-right: 6px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
    main { padding: 24px; max-width: 1200px; margin: 0 auto; }
    .section-title { font-size: 0.75rem; font-weight: 600; color: #718096;
                     text-transform: uppercase; letter-spacing: .08em; margin: 24px 0 12px; }
    .grid { display: grid; gap: 16px; }
    .grid-4 { grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
    .grid-2 { grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); }
    .card { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 12px; padding: 18px; }
    .card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
    .card-icon { font-size: 1.5rem; }
    .card-title { font-weight: 600; font-size: 0.95rem; }
    .card-subtitle { font-size: 0.75rem; color: #718096; margin-top: 2px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 20px;
             font-size: 0.7rem; font-weight: 600; }
    .badge-green { background: #1c4532; color: #68d391; }
    .badge-yellow { background: #3d2e00; color: #f6c90e; }
    .badge-red { background: #3d1515; color: #fc8181; }
    .badge-gray { background: #2d3748; color: #a0aec0; }
    .metric { display: flex; justify-content: space-between; align-items: center;
              padding: 8px 0; border-bottom: 1px solid #2d3748; font-size: 0.85rem; }
    .metric:last-child { border-bottom: none; }
    .metric-val { font-weight: 600; color: #90cdf4; }
    .rec-item { padding: 10px 0; border-bottom: 1px solid #2d3748; font-size: 0.82rem; }
    .rec-item:last-child { border-bottom: none; }
    .rec-agent { color: #90cdf4; font-weight: 600; }
    .empty { color: #4a5568; font-size: 0.85rem; padding: 12px 0; text-align: center; }
    .refresh-info { font-size: 0.72rem; color: #4a5568; }
    .pipeline { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .pipeline-step { background: #2d3748; border-radius: 8px; padding: 6px 12px;
                     font-size: 0.78rem; position: relative; }
    .pipeline-arrow { color: #4a5568; font-size: 0.9rem; }
    .pipeline-step.active { background: #1c4532; color: #68d391; border: 1px solid #276749; }
    .pipeline-step.pending { background: #3d2e00; color: #f6c90e; }
  </style>
</head>
<body>
<header>
  <div>
    <h1>🫒 Monitor ML — Aceites Comestibles</h1>
    <div class="subtitle">Sistema multiagente · Ecuador · Interfaz principal: Telegram bot</div>
  </div>
  <div style="text-align:right">
    <span class="status-dot"></span><span style="font-size:.8rem;color:#68d391">EN LÍNEA</span>
    <div class="refresh-info" id="last-refresh">Cargando...</div>
  </div>
</header>

<main>
  <!-- Agentes -->
  <div class="section-title">Agentes</div>
  <div class="grid grid-4" id="agents-grid">
    <div class="card"><div class="empty">Cargando...</div></div>
  </div>

  <!-- Métricas + Recomendaciones -->
  <div class="grid grid-2" style="margin-top:0">
    <div>
      <div class="section-title">Estado de datos</div>
      <div class="card" id="data-status">
        <div class="empty">Cargando...</div>
      </div>
    </div>
    <div>
      <div class="section-title">Recomendaciones Agent Lab pendientes</div>
      <div class="card" id="recs-card">
        <div class="empty">Cargando...</div>
      </div>
    </div>
  </div>

  <!-- Pipeline -->
  <div class="section-title">Pipeline del sistema</div>
  <div class="card">
    <div class="pipeline">
      <div class="pipeline-step active">Telegram Webhook</div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step active">Orchestrator</div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step active">Agente Especialista</div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step active">Supabase</div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step active">Anthropic API</div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step active">Respuesta Telegram</div>
    </div>
    <div class="pipeline" style="margin-top:12px">
      <div class="pipeline-step active">GitHub Actions Cron</div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step active">Research Pipeline</div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step active">Agent Lab</div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step pending">Auto-mejora</div>
    </div>
  </div>

  <div style="margin-top:24px;color:#4a5568;font-size:0.75rem">
    <a href="/docs" style="color:#4a5568">API docs</a> ·
    <a href="/health" style="color:#4a5568">health</a> ·
    <a href="/api/status" style="color:#4a5568">JSON status</a>
  </div>
</main>

<script>
async function refresh() {
  try {
    const res = await fetch('/api/status');
    const d = await res.json();

    // Agentes
    const agentModels = { 'claude-haiku-4-5-20251001': 'Haiku 4.5', 'claude-sonnet-4-6': 'Sonnet 4.6' };
    document.getElementById('agents-grid').innerHTML = d.agents.map(a => `
      <div class="card">
        <div class="card-header">
          <span class="card-icon">${a.icon}</span>
          <div>
            <div class="card-title">${a.name}</div>
            <div class="card-subtitle">${a.desc}</div>
          </div>
        </div>
        <div class="metric"><span>Estado</span>
          <span class="badge badge-green">✓ activo</span></div>
        <div class="metric"><span>Consultas</span>
          <span class="metric-val">${a.calls}</span></div>
        <div class="metric"><span>Modelo</span>
          <span class="metric-val">${agentModels[a.model] || a.model}</span></div>
        <div class="metric"><span>Herramientas</span>
          <span class="metric-val">${a.tools}</span></div>
        <div class="metric"><span>Config en DB</span>
          <span class="badge ${a.in_db ? 'badge-green' : 'badge-yellow'}">${a.in_db ? '✓ persistida' : '⚠ pendiente'}</span></div>
      </div>`).join('');

    // Estado de datos
    document.getElementById('data-status').innerHTML = `
      <div class="metric"><span>Precio palma (último)</span>
        <span class="metric-val">${d.last_price ? '$' + d.last_price.actual_price + ' — ' + d.last_price.date : '⚠ Sin datos'}</span></div>
      <div class="metric"><span>Fuente precio</span>
        <span class="metric-val">${d.last_price ? d.last_price.source : '—'}</span></div>
      <div class="metric"><span>Findings de research</span>
        <span class="metric-val">${d.research_count} registros</span></div>
      <div class="metric"><span>Ventas (Excel)</span>
        <span class="badge badge-yellow">Fase 2 — pendiente</span></div>`;

    // Recomendaciones
    document.getElementById('recs-card').innerHTML = d.pending_recommendations.length
      ? d.pending_recommendations.map(r => `
          <div class="rec-item">
            <span class="rec-agent">${r.target_agent}</span> — ${r.title}
            <span class="badge badge-yellow" style="margin-left:6px">P${r.priority}</span>
          </div>`).join('')
      : '<div class="empty">Sin recomendaciones pendientes</div>';

    // Timestamp
    const ts = new Date(d.timestamp);
    document.getElementById('last-refresh').textContent =
      'Actualizado: ' + ts.toLocaleTimeString('es-EC');

  } catch(e) {
    document.getElementById('last-refresh').textContent = 'Error al cargar datos';
  }
}

refresh();
setInterval(refresh, 30000);
</script>
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
