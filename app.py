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
    """Dashboard estilo N8N con árbol jerárquico de agentes."""
    return """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Monitor ML — Aceites</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    /* ── Fondo N8N: oscuro con grid de puntos ── */
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background-color: #0d0d12;
      background-image: radial-gradient(circle, #1e1e2e 1px, transparent 1px);
      background-size: 28px 28px;
      color: #e2e8f0;
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      background: rgba(16,16,24,0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid #1e1e30;
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky; top: 0; z-index: 10;
    }
    .header-brand { display: flex; align-items: center; gap: 10px; }
    .header-brand h1 { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; letter-spacing: -.01em; }
    .header-brand .tag { font-size: 0.68rem; background: #1a3326; color: #68d391;
                         border: 1px solid #276749; border-radius: 20px; padding: 2px 8px; }
    .status-pill { display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: #68d391; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #68d391; animation: blink 2s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
    .refresh-ts { font-size: 0.68rem; color: #3a3a55; margin-top: 2px; }

    /* ── Layout ── */
    main { padding: 32px 24px; max-width: 960px; margin: 0 auto; }
    .section-label {
      font-size: 0.68rem; font-weight: 700; color: #3a3a6a;
      text-transform: uppercase; letter-spacing: .12em; margin-bottom: 20px;
    }

    /* ── Nodo base ── */
    .node {
      background: #13131e;
      border: 1px solid #22223a;
      border-radius: 14px;
      padding: 18px 20px;
      text-align: center;
      width: 200px;
      box-shadow: 0 4px 32px rgba(0,0,0,.5);
      transition: border-color .2s, box-shadow .2s;
    }
    .node:hover { border-color: #35355a; box-shadow: 0 8px 40px rgba(0,0,0,.7); }
    .node-icon { font-size: 1.8rem; margin-bottom: 8px; }
    .node-name { font-size: 0.82rem; font-weight: 700; font-family: 'SF Mono', monospace;
                 color: #c8c8e8; margin-bottom: 4px; }
    .node-desc { font-size: 0.68rem; color: #44445a; line-height: 1.5; margin-bottom: 10px; }
    .node-pills { display: flex; gap: 4px; justify-content: center; flex-wrap: wrap; }
    .pill { font-size: 0.62rem; font-weight: 600; padding: 2px 7px;
            border-radius: 20px; }
    .pill-green { background: #0e2a1a; color: #68d391; border: 1px solid #1a4a2e; }
    .pill-blue  { background: #0e1a2a; color: #90cdf4; border: 1px solid #1a2e4a; }

    /* ── Nodo raíz especial ── */
    .node-root {
      width: 230px;
      border-color: #276749;
      box-shadow: 0 0 0 1px #276749, 0 0 40px rgba(104,211,145,.12), 0 4px 32px rgba(0,0,0,.5);
    }
    .node-root .node-name { color: #68d391; }
    .node-root:hover { box-shadow: 0 0 0 1px #38a169, 0 0 50px rgba(104,211,145,.2), 0 8px 40px rgba(0,0,0,.7); }

    /* ── Árbol jerárquico ── */
    .tree { display: flex; flex-direction: column; align-items: center; }

    .tree-root-row { display: flex; justify-content: center; }

    /* Conector: línea vertical + horizontal */
    .tree-connector {
      display: flex; flex-direction: column; align-items: center;
      width: 100%; position: relative; height: 52px;
    }
    .conn-trunk {
      width: 2px; height: 22px;
      background: linear-gradient(to bottom, #276749, #1e3a2a);
    }
    .conn-branch {
      position: absolute; bottom: 0;
      left: calc(16.67% + 100px); right: calc(16.67% + 100px);
      height: 2px;
      background: linear-gradient(to right, #1e3a2a, #276749 50%, #1e3a2a);
    }

    /* Columnas hijas */
    .tree-children-row {
      display: flex; justify-content: center; gap: 20px; width: 100%;
    }
    .tree-col { display: flex; flex-direction: column; align-items: center; }
    .conn-stem {
      width: 2px; height: 30px;
      background: linear-gradient(to bottom, #276749, #1e3a2a);
    }

    /* ── Cards de datos ── */
    .cards-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 32px; }
    @media(max-width:600px){ .cards-row { grid-template-columns: 1fr; } }
    .card {
      background: #13131e; border: 1px solid #1e1e30;
      border-radius: 14px; padding: 20px;
      box-shadow: 0 4px 24px rgba(0,0,0,.4);
    }
    .card-title { font-size: 0.7rem; font-weight: 700; color: #3a3a6a;
                  text-transform: uppercase; letter-spacing: .1em; margin-bottom: 14px; }
    .metric { display: flex; justify-content: space-between; align-items: center;
              padding: 7px 0; border-bottom: 1px solid #1a1a2a; font-size: 0.82rem; color: #6060a0; }
    .metric:last-child { border-bottom: none; }
    .metric-val { font-weight: 600; color: #90cdf4; font-size: 0.8rem; }
    .badge { display: inline-block; padding: 2px 7px; border-radius: 20px;
             font-size: 0.65rem; font-weight: 600; }
    .badge-green { background: #0e2a1a; color: #68d391; border: 1px solid #1a4a2e; }
    .badge-yellow { background: #2a1e00; color: #f6c90e; border: 1px solid #4a3800; }
    .rec-item { padding: 8px 0; border-bottom: 1px solid #1a1a2a; font-size: 0.8rem; color: #5050a0; }
    .rec-item:last-child { border-bottom: none; }
    .rec-agent { color: #90cdf4; font-weight: 600; }
    .empty { color: #2a2a4a; font-size: 0.82rem; padding: 10px 0; text-align: center; }

    /* ── Pipeline ── */
    .pipeline-wrap { margin-top: 32px; }
    .pipeline { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
    .pipe-step { background: #13131e; border: 1px solid #1e1e30; border-radius: 8px;
                 padding: 5px 11px; font-size: 0.72rem; color: #3a3a6a; }
    .pipe-step.on { border-color: #1a4a2e; color: #68d391; background: #0a1a10; }
    .pipe-step.soon { border-color: #4a3800; color: #f6c90e; background: #1a1400; }
    .pipe-arrow { color: #1e1e30; font-size: 0.8rem; }

    /* ── Footer ── */
    .footer { margin-top: 36px; font-size: 0.68rem; color: #22223a; }
    .footer a { color: #2a2a4a; text-decoration: none; }
    .footer a:hover { color: #68d391; }
  </style>
</head>
<body>

<header>
  <div class="header-brand">
    <span style="font-size:1.4rem">🫒</span>
    <div>
      <h1>Monitor ML — Aceites Comestibles</h1>
      <div style="font-size:.68rem;color:#3a3a6a;margin-top:2px">Ecuador · Interfaz principal: Telegram bot</div>
    </div>
    <span class="tag">Fase 1</span>
  </div>
  <div style="text-align:right">
    <div class="status-pill"><span class="dot"></span>EN LÍNEA</div>
    <div class="refresh-ts" id="last-refresh">Cargando...</div>
  </div>
</header>

<main>

  <!-- Árbol de agentes -->
  <div class="section-label">Arquitectura multiagente</div>
  <div class="tree">

    <!-- Raíz -->
    <div class="tree-root-row">
      <div class="node node-root">
        <div class="node-icon">🔀</div>
        <div class="node-name">orchestrator</div>
        <div class="node-desc">Clasifica intents y<br>enruta al especialista</div>
        <div class="node-pills">
          <span class="pill pill-green">Haiku 4.5</span>
          <span class="pill pill-blue" id="calls-orchestrator">0 calls</span>
        </div>
      </div>
    </div>

    <!-- Conector raíz → hijos -->
    <div class="tree-connector">
      <div class="conn-trunk"></div>
      <div class="conn-branch"></div>
    </div>

    <!-- Hijos -->
    <div class="tree-children-row">
      <div class="tree-col">
        <div class="conn-stem"></div>
        <div class="node">
          <div class="node-icon">📈</div>
          <div class="node-name">price_monitor</div>
          <div class="node-desc">Forecasting de precios<br>aceite de palma</div>
          <div class="node-pills">
            <span class="pill pill-green">Sonnet 4.6</span>
            <span class="pill pill-blue" id="calls-price_monitor">0 calls</span>
          </div>
        </div>
      </div>
      <div class="tree-col">
        <div class="conn-stem"></div>
        <div class="node">
          <div class="node-icon">📦</div>
          <div class="node-name">demand_monitor</div>
          <div class="node-desc">Forecasting de demanda<br>y ventas</div>
          <div class="node-pills">
            <span class="pill pill-green">Sonnet 4.6</span>
            <span class="pill pill-blue" id="calls-demand_monitor">0 calls</span>
          </div>
        </div>
      </div>
      <div class="tree-col">
        <div class="conn-stem"></div>
        <div class="node">
          <div class="node-icon">🔬</div>
          <div class="node-name">agent_lab</div>
          <div class="node-desc">Meta-agente de mejora<br>continua del sistema</div>
          <div class="node-pills">
            <span class="pill pill-green">Sonnet 4.6</span>
            <span class="pill pill-blue" id="calls-agent_lab">0 calls</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Cards de datos -->
  <div class="cards-row">
    <div class="card">
      <div class="card-title">Estado de datos</div>
      <div id="data-status"><div class="empty">Cargando...</div></div>
    </div>
    <div class="card">
      <div class="card-title">Recomendaciones pendientes</div>
      <div id="recs-card"><div class="empty">Cargando...</div></div>
    </div>
  </div>

  <!-- Pipeline -->
  <div class="pipeline-wrap">
    <div class="section-label" style="margin-top:0">Pipeline</div>
    <div class="pipeline">
      <div class="pipe-step on">Telegram Webhook</div><div class="pipe-arrow">›</div>
      <div class="pipe-step on">Orchestrator</div><div class="pipe-arrow">›</div>
      <div class="pipe-step on">Especialista</div><div class="pipe-arrow">›</div>
      <div class="pipe-step on">Supabase</div><div class="pipe-arrow">›</div>
      <div class="pipe-step on">Anthropic API</div><div class="pipe-arrow">›</div>
      <div class="pipe-step on">Respuesta Telegram</div>
    </div>
    <div class="pipeline" style="margin-top:8px">
      <div class="pipe-step on">GitHub Actions Cron</div><div class="pipe-arrow">›</div>
      <div class="pipe-step on">Research Pipeline</div><div class="pipe-arrow">›</div>
      <div class="pipe-step on">Agent Lab</div><div class="pipe-arrow">›</div>
      <div class="pipe-step soon">Auto-mejora</div>
    </div>
  </div>

  <div class="footer">
    <a href="/docs">API docs</a> · <a href="/health">health</a> · <a href="/api/status">JSON status</a>
  </div>

</main>

<script>
async function refresh() {
  try {
    const res = await fetch('/api/status');
    const d = await res.json();

    // Actualizar call counts en el árbol
    d.agents.forEach(a => {
      const el = document.getElementById('calls-' + a.name);
      if (el) el.textContent = a.calls + ' calls';
    });

    // Estado de datos
    document.getElementById('data-status').innerHTML = `
      <div class="metric"><span>Precio palma (último)</span>
        <span class="metric-val">${d.last_price ? '$'+d.last_price.actual_price+' &mdash; '+d.last_price.date : '— sin datos'}</span></div>
      <div class="metric"><span>Fuente</span>
        <span class="metric-val">${d.last_price ? d.last_price.source : '—'}</span></div>
      <div class="metric"><span>Findings research</span>
        <span class="metric-val">${d.research_count} registros</span></div>
      <div class="metric"><span>Ventas (Excel)</span>
        <span class="badge badge-yellow">Fase 2</span></div>`;

    // Recomendaciones
    document.getElementById('recs-card').innerHTML = d.pending_recommendations.length
      ? d.pending_recommendations.map(r => `
          <div class="rec-item">
            <span class="rec-agent">${r.target_agent}</span> — ${r.title}
            <span class="badge badge-yellow" style="margin-left:5px">P${r.priority}</span>
          </div>`).join('')
      : '<div class="empty">Sin recomendaciones pendientes</div>';

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
