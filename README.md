---
title: Monitor ML Aceitess
emoji: 🫒
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Monitor ML — Aceites Comestibles

Sistema multiagente de monitoreo ML para producción y venta de aceites comestibles en Ecuador.

## Agentes
- **price_monitor**: Forecasting de precios del aceite de palma (FRED / World Bank)
- **demand_monitor**: Forecasting de demanda y ventas (aceites, mantecas)
- **agent_lab**: Meta-agente de auto-mejora continua
- **orchestrator**: Clasificación de intents y routing

## Interfaz
Telegram bot (webhook) + Gradio dashboard.
