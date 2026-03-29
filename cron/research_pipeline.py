"""
research_pipeline.py
--------------------
Pipeline diario de investigación. Ejecutado por GitHub Actions cada noche.

Pasos:
  0. PRICE   — Fetch precio diario CPO desde Commodities-API (si hay key configurada)
  1. SCRAPE  — Fetch RSS/HTML de 6 fuentes de AI research
  2. DEDUP   — Filtrar URLs ya procesadas en DB
  3. SCORE   — Haiku evalúa relevancia para el dominio (0.0 a 1.0)
  4. SAVE    — Guardar findings con score > 0.3
  5. BLOAT   — Verificar skill bloat en agentes
  6. IMPROVE — Agent Lab ejecuta ciclo completo de mejora
  7. EVENTS  — Scraping noticias de palma (DuckDuckGo) + clasificación Claude
  8. REPORT  — Enviar resumen vía Telegram

Uso:
    python -m cron.research_pipeline
    python -m cron.research_pipeline --dry-run   (sin escribir a DB)
"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Fuentes de research
RESEARCH_SOURCES = [
    {"name": "Anthropic Blog",    "url": "https://www.anthropic.com/news", "tier": "anthropic"},
    {"name": "Anthropic Cookbook","url": "https://github.com/anthropics/anthropic-cookbook/commits/main.atom", "tier": "anthropic"},
    {"name": "Simon Willison",    "url": "https://simonwillison.net/atom/entries/", "tier": "community"},
    {"name": "Lilian Weng",       "url": "https://lilianweng.github.io/index.xml", "tier": "research"},
    {"name": "LangChain Blog",    "url": "https://blog.langchain.dev/rss/", "tier": "ecosystem"},
    {"name": "HuggingFace Blog",  "url": "https://huggingface.co/blog/feed.xml", "tier": "ecosystem"},
]

EVENT_CLASSIFICATION_PROMPT = """Eres un analista de commodities especializado en aceite de palma.

Clasifica esta noticia de mercado y responde SOLO con un JSON válido (sin markdown, sin texto extra):

{{
  "event_type": "weather|geopolitical|policy|supply|demand|trade|macro",
  "region": "Malaysia|Indonesia|Ecuador|India|EU|China|global|other",
  "price_direction": "bullish|bearish|neutral",
  "price_impact_pct": <número entre -20 y +20, estimación del impacto % en precio>,
  "confidence": <número entre 0.0 y 1.0>,
  "event_date": "<YYYY-MM-DD o fecha aproximada del evento, no de hoy>",
  "tags": ["tag1", "tag2"]
}}

Guía de clasificación:
- bullish: sube precio (sequía, restricción exportaciones, mayor demanda biocombustibles, reducción cosecha)
- bearish: baja precio (cosecha récord, caída demanda, competencia soja/girasol, recesión)
- neutral: impacto ambiguo o mínimo

Noticia:
Título: {title}
Descripción: {description}
Fuente: {source}"""

RELEVANCE_PROMPT = """Eres un evaluador de relevancia para un sistema de agentes de monitoreo ML
aplicado a la industria de aceites comestibles en Ecuador.

El sistema usa:
- Claude (Anthropic) como LLM
- Agentes especialistas con tool-calling
- Forecasting de precios de aceite de palma y demanda de ventas
- Python, Supabase, HuggingFace Spaces, Telegram

Evalúa este artículo con un score de 0.0 a 1.0 de relevancia para MEJORAR el sistema:
- >0.8: Específico y directamente aplicable (nueva técnica de agentes, mejora de tool-calling, etc.)
- >0.5: Relevante pero genérico (mejores prácticas de LLM, patrones de agentes)
- <0.3: Irrelevante para este sistema

Responde SOLO con el número (ej: 0.75). Sin texto adicional.

Artículo:
Título: {title}
Resumen: {summary}"""


async def scrape_rss(url: str) -> list[dict]:
    """Obtiene entradas de un feed RSS/Atom."""
    try:
        import xml.etree.ElementTree as ET

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entries = []

        # Atom feed
        for entry in root.findall(".//atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            link_el = entry.find("atom:link", ns)
            summary_el = entry.find("atom:summary", ns) or entry.find("atom:content", ns)

            if title_el is not None and link_el is not None:
                entries.append({
                    "title": title_el.text or "",
                    "url": link_el.get("href", ""),
                    "summary": (summary_el.text or "")[:500] if summary_el is not None else "",
                })

        # RSS 2.0 feed
        if not entries:
            for item in root.findall(".//item"):
                title_el = item.find("title")
                link_el = item.find("link")
                desc_el = item.find("description")

                if title_el is not None and link_el is not None:
                    entries.append({
                        "title": title_el.text or "",
                        "url": link_el.text or "",
                        "summary": (desc_el.text or "")[:500] if desc_el is not None else "",
                    })

        return entries[:10]  # Máximo 10 por fuente

    except Exception as e:
        logger.warning(f"Error scrapeando {url}: {e}")
        return []


async def score_relevance(title: str, summary: str) -> float:
    """Usa Haiku para evaluar relevancia del artículo."""
    import anthropic
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": RELEVANCE_PROMPT.format(title=title, summary=summary[:300]),
            }],
        )
        score_text = response.content[0].text.strip()
        return min(max(float(score_text), 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Error evaluando relevancia: {e}")
        return 0.0


COMMODITIES = [
    # USD/MT — rango esperado 400-2500
    {"name": "palm_oil",     "symbol": "CPO",  "unit": "USD/MT", "range": (400, 2500),
     "queries": ["crude palm oil CPO price USD metric ton today", "palm oil futures price USD today"]},
    {"name": "pko",          "symbol": "PKO",  "unit": "USD/MT", "range": (500, 3000),
     "queries": ["palm kernel oil PKO price USD metric ton today", "palm kernel oil price USD ton"]},
    {"name": "sunflower_oil","symbol": "SFO",  "unit": "USD/MT", "range": (800, 3000),
     "queries": ["sunflower oil price USD metric ton today", "Ukraine sunflower oil FOB USD ton"]},
    {"name": "soybean_oil",  "symbol": "SBO",  "unit": "USD/MT", "range": (800, 2500),
     "queries": ["soybean oil price USD metric ton today", "soy oil Chicago futures USD ton"]},
    {"name": "corn_oil",     "symbol": "CNO",  "unit": "USD/MT", "range": (800, 3000),
     "queries": ["corn oil price USD metric ton today", "maize oil price USD ton"]},
    # USD/kg — rango esperado 5-80
    {"name": "achiote",      "symbol": "ANN",  "unit": "USD/kg", "range": (5, 80),
     "queries": ["annatto extract bixin price USD kg", "annatto oleoresin price USD kilogram"]},
]

import re as _re

def _extract_price_from_text(text: str, expected_range: tuple) -> float | None:
    """Extrae el primer precio numerico dentro del rango esperado."""
    # Patrones de mayor a menor especificidad
    patterns = [
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+)?(?:metric\s+)?(?:ton|MT|tonne|kg)',
        r'([\d,]+(?:\.\d+)?)\s*(?:USD|US\$)\s*(?:per\s+)?(?:metric\s+)?(?:ton|MT|kg)',
        r'price[^$\d]{0,20}\$([\d,]+(?:\.\d+)?)',
        r'([\d,]+\.\d{1,2})',   # numeros con decimales (e.g. 1,330.00 o 1138.5)
        r'(?<!\d)(1[,.]?\d{3})(?!\d)',  # numeros de 4 digitos tipo 1330 o 1,330
    ]
    for pattern in patterns:
        matches = _re.findall(pattern, text, _re.IGNORECASE)
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if expected_range[0] <= val <= expected_range[1]:
                    return round(val, 2)
            except ValueError:
                continue
    return None


async def fetch_commodity_prices(dry_run: bool = False) -> str:
    """Paso 0: fetch precios diarios de commodities relevantes via DuckDuckGo."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "ddgs no instalado — omitiendo fetch de precios"

    from datetime import date
    results_log = []

    for commodity in COMMODITIES:
        price = None
        source_snippet = ""

        try:
            with DDGS() as ddgs:
                for query in commodity["queries"]:
                    if price:
                        break
                    snippets = list(ddgs.text(query, max_results=4))
                    for s in snippets:
                        text = s.get("title", "") + " " + s.get("body", "")
                        price = _extract_price_from_text(text, commodity["range"])
                        if price:
                            source_snippet = s.get("title", "")[:60]
                            break
        except Exception as e:
            logger.warning(f"  Error buscando {commodity['name']}: {e}")
            continue

        if price:
            unit_short = commodity["unit"].replace("USD/", "")
            results_log.append(f"{commodity['symbol']}: ${price}/{unit_short}")
            if not dry_run:
                try:
                    from database.supabase_client import get_supabase
                    db = get_supabase()
                    db.table("price_data").upsert({
                        "date": date.today().isoformat(),
                        "actual_price": price,
                        "commodity": commodity["name"],
                        "source": "ddgs",
                        "currency": "USD",
                        "unit": commodity["unit"],
                    }, on_conflict="date,commodity").execute()
                except Exception as e:
                    logger.warning(f"  Error guardando {commodity['name']}: {e}")
        else:
            results_log.append(f"{commodity['symbol']}: sin precio")
            logger.warning(f"  No se pudo extraer precio para {commodity['name']}")

    return "Precios commodities: " + " | ".join(results_log)


async def fetch_commodity_news() -> list[dict]:
    """
    Busca noticias recientes sobre commodities relevantes (palma, PKO, girasol, soya, maiz, achiote).
    Sin API key. Busqueda en ingles para maxima cobertura.
    """
    try:
        from ddgs import DDGS

        queries = [
            "crude palm oil CPO price market news",
            "Malaysia Indonesia palm oil production export 2026",
            "palm oil biofuel policy demand",
            "palm kernel oil PKO market",
            "soybean oil sunflower oil price 2026",
            "vegetable oil commodities market outlook",
        ]

        all_results = []
        seen_urls = set()

        with DDGS() as ddgs:
            for query in queries:
                try:
                    results = list(ddgs.news(keywords=query, max_results=4, timelimit="m"))
                    for r in results:
                        url = r.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append({
                                "title": r.get("title", ""),
                                "description": r.get("body", "")[:400],
                                "source_url": url,
                                "source_name": r.get("source", ""),
                                "published": r.get("date", ""),
                            })
                except Exception as e:
                    logger.warning(f"Error buscando '{query}': {e}")

        logger.info(f"Noticias commodities encontradas: {len(all_results)}")
        return all_results

    except ImportError:
        logger.warning("ddgs no instalado. Omitiendo paso de noticias.")
        return []
    except Exception as e:
        logger.warning(f"Error en fetch_commodity_news: {e}")
        return []


async def classify_market_event(article: dict) -> dict | None:
    """Usa Claude Haiku para clasificar el impacto en precio del artículo."""
    import anthropic
    import json
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": EVENT_CLASSIFICATION_PROMPT.format(
                    title=article["title"],
                    description=article["description"][:300],
                    source=article["source_name"],
                ),
            }],
        )
        raw = response.content[0].text.strip()
        classification = json.loads(raw)
        return classification
    except Exception as e:
        logger.warning(f"Error clasificando evento '{article['title'][:50]}': {e}")
        return None


async def run_palm_oil_events_step(dry_run: bool = False) -> int:
    """
    Paso 7: Scraping de noticias de palma + clasificación + guardado en market_events.
    Retorna número de eventos guardados.
    """
    articles = await fetch_commodity_news()
    if not articles:
        return 0

    # Dedup contra URLs ya guardadas
    saved_count = 0
    if not dry_run:
        from database.supabase_client import get_supabase
        db = get_supabase()
        existing = db.table("market_events").select("source_url").execute()
        existing_urls = {r["source_url"] for r in (existing.data or []) if r.get("source_url")}
        articles = [a for a in articles if a["source_url"] not in existing_urls]

    logger.info(f"Artículos nuevos a clasificar: {len(articles)}")

    for article in articles:
        classification = await classify_market_event(article)
        if not classification:
            continue

        # Parsear fecha del evento
        from datetime import date
        event_date_str = classification.get("event_date") or date.today().isoformat()
        try:
            event_date = date.fromisoformat(event_date_str[:10]).isoformat()
        except ValueError:
            event_date = date.today().isoformat()

        record = {
            "event_date": event_date,
            "title": article["title"],
            "description": article["description"],
            "source_url": article["source_url"],
            "source_name": article["source_name"],
            "event_type": classification.get("event_type", "macro"),
            "region": classification.get("region", "global"),
            "price_direction": classification.get("price_direction", "neutral"),
            "price_impact_pct": classification.get("price_impact_pct"),
            "confidence": classification.get("confidence"),
            "tags": classification.get("tags", []),
        }

        logger.info(
            f"  [{record['price_direction'].upper():7s}] "
            f"{record['event_type']:12s} | {record['region']:10s} | "
            f"{record['price_impact_pct']:+.1f}% | {article['title'][:60]}"
        )

        if not dry_run:
            try:
                db.table("market_events").insert(record).execute()
                saved_count += 1
            except Exception as e:
                logger.warning(f"Error guardando evento: {e}")

    return saved_count


async def run_pipeline(dry_run: bool = False):
    """Ejecuta el pipeline completo de investigación."""
    logger.info("=== Iniciando pipeline de research ===")
    start_time = datetime.utcnow()

    from database.supabase_client import get_supabase
    db = get_supabase() if not dry_run else None

    # 0. PRECIOS DIARIOS COMMODITIES
    logger.info("PASO 0: Fetch precios commodities (DuckDuckGo)...")
    price_result = await fetch_commodity_prices(dry_run=dry_run)
    logger.info(f"  {price_result}")

    # 1. SCRAPE
    logger.info("PASO 1: Scraping de fuentes...")
    all_entries = []
    for source in RESEARCH_SOURCES:
        entries = await scrape_rss(source["url"])
        for e in entries:
            e["source_tier"] = source["tier"]
            e["source_name"] = source["name"]
        all_entries.extend(entries)
        logger.info(f"  {source['name']}: {len(entries)} entradas")

    logger.info(f"Total entradas scraped: {len(all_entries)}")

    # 2. DEDUP
    logger.info("PASO 2: Deduplicación...")
    if db:
        existing = db.table("agent_lab_research").select("source_url").execute()
        existing_urls = {r["source_url"] for r in (existing.data or [])}
        all_entries = [e for e in all_entries if e["url"] not in existing_urls]
    logger.info(f"Entradas nuevas después de dedup: {len(all_entries)}")

    # 3. SCORE + 4. SAVE
    logger.info("PASO 3-4: Scoring y guardado...")
    saved = 0
    for entry in all_entries:
        score = await score_relevance(entry["title"], entry["summary"])
        logger.debug(f"  Score {score:.2f}: {entry['title'][:60]}")

        if score > 0.3:
            if not dry_run and db:
                db.table("agent_lab_research").insert({
                    "source_url": entry["url"],
                    "source_tier": entry["source_tier"],
                    "title": entry["title"],
                    "summary": entry["summary"],
                    "relevance_score": round(score, 2),
                    "tags": [entry["source_tier"]],
                }).execute()
            saved += 1

    logger.info(f"Findings guardados (score > 0.3): {saved}")

    # 5. BLOAT CHECK
    logger.info("PASO 5: Verificando skill bloat...")
    if db:
        agents = db.table("agents").select("name, tools").execute()
        for agent in (agents.data or []):
            tool_count = len(agent.get("tools") or [])
            if tool_count >= 5:
                logger.warning(f"  BLOAT: {agent['name']} tiene {tool_count} herramientas")

    # 6. AGENT LAB
    logger.info("PASO 6: Ejecutando ciclo de Agent Lab...")
    if not dry_run:
        from agents.agent_lab import AgentLabAgent
        lab = AgentLabAgent()
        lab_report = lab.run_daily_cycle()
        logger.info(f"Agent Lab completado: {lab_report[:200]}...")
    else:
        lab_report = "[dry-run: Agent Lab omitido]"

    # 7. EVENTOS MERCADO COMMODITIES
    logger.info("PASO 7: Scraping noticias commodities (DuckDuckGo + clasificacion Claude)...")
    events_saved = await run_palm_oil_events_step(dry_run=dry_run)
    logger.info(f"Eventos de mercado guardados: {events_saved}")

    # 8. REPORT
    elapsed = (datetime.utcnow() - start_time).seconds
    report = (
        f"📊 *Reporte diario Agent Lab*\n\n"
        f"• Precio CPO hoy: {price_result}\n"
        f"• Fuentes AI scrapeadas: {len(RESEARCH_SOURCES)}\n"
        f"• Findings AI guardados: {saved}\n"
        f"• Eventos mercado palma: {events_saved} nuevos\n"
        f"• Duración: {elapsed}s\n\n"
        f"*Resumen Agent Lab:*\n{lab_report[:500]}"
    )

    await send_telegram_report(report)
    logger.info("=== Pipeline completado ===")
    return report


async def send_telegram_report(message: str):
    """Envía el reporte al bot de Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ADMIN_CHAT_ID")

    if not token or not chat_id:
        logger.info(f"[Report] (Telegram no configurado)\n{message}")
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            )
        logger.info("Reporte enviado por Telegram")
    except Exception as e:
        logger.error(f"Error enviando reporte Telegram: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline diario de research")
    parser.add_argument("--dry-run", action="store_true", help="Ejecutar sin escribir a DB")
    args = parser.parse_args()

    asyncio.run(run_pipeline(dry_run=args.dry_run))
