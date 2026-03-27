# monitor-ml-aceites

Sistema multiagente de monitoreo ML para producción y venta de aceites comestibles.
Desplegado en HuggingFace Spaces. Interfaz vía Telegram bot.

## Stack

- **Runtime**: Python 3.11+
- **Framework agentes**: Anthropic SDK (claude-haiku-4-5 para clasificación, claude-sonnet-4-6 para agentes)
- **Interfaz**: Telegram bot (webhook) + Gradio dashboard
- **Base de datos**: Supabase (PostgreSQL)
- **Despliegue**: HuggingFace Spaces (Gradio + FastAPI)
- **Cron**: GitHub Actions

## Estructura

```
agents/
  base_agent.py        # runAgent() — loop de tool-calling
  orchestrator.py      # Clasificación de intents + routing
  agent_lab.py         # Meta-agente de auto-mejora
  specialists/
    price_monitor.py   # Monitoreo modelo forecasting de precios (aceite de palma)
    demand_monitor.py  # Monitoreo modelo forecasting de demanda/ventas
tools/
  sub_agent_tool.py    # subAgentTool() — delegación jerárquica
telegram/
  bot.py               # Webhook handler + comandos
database/
  supabase_client.py   # Cliente Supabase
  schema.sql           # Schema completo
data/
  raw/                 # Datasets crudos (FRED, World Bank, Kaggle)
  processed/           # Datasets procesados
models/                # Modelos ML entrenados (.pkl, etc.)
cron/
  research_pipeline.py # Pipeline diario de investigación (GitHub Actions)
app.py                 # Entry point HuggingFace Space
```

## Fuentes de datos

- **Precio mundial aceite de palma**: FRED (PPOILUSDM), World Bank Pink Sheet
- **Ecuador producción/precios**: CFN Ficha Sectorial, ANCUPA, USDA FAS PSD Online
- **Ventas FMCG**: Kaggle FMCG Daily Sales 2022–2024
- **Futuro**: base de datos plana Excel del negocio real (aceites, mantecas, otros productos)

## Agentes especialistas

| Agente | Modelo | Función |
|--------|--------|---------|
| price_monitor | sonnet-4-6 | Monitorea modelo de forecasting de precios de palma |
| demand_monitor | sonnet-4-6 | Monitorea modelo de forecasting de demanda/ventas |
| agent_lab | sonnet-4-6 | Meta-agente: lee configs, aplica mejoras reactivas, encola proactivas |
| orchestrator | haiku-4-5 | Clasifica intents y enruta al especialista correcto |

## Autonomía de Agent Lab

- **REACTIVA** (auto-aplica): corrección de errores, anti-patrones en prompts, memorias de feedback
- **PROACTIVA** (requiere aprobación vía Telegram): nuevas herramientas, restructuración, cambio de modelo

## Comandos de desarrollo

```bash
# Instalar dependencias
pip install -r requirements.txt

# Copiar y configurar variables de entorno
cp .env.example .env

# Ejecutar localmente
python app.py

# Aplicar schema a Supabase
# Ejecutar database/schema.sql en el SQL Editor de Supabase
```

## Control de versiones

Después de cada cambio relevante: commit + push a GitHub inmediatamente.
Formato: `<tipo>: <descripción en imperativo>`
