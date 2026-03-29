"""
train_demand_model.py
---------------------
Entrena modelos Prophet de forecasting de demanda para cada producto
en sales_data y guarda las predicciones en demand_forecasts (Supabase).

Uso:
    python -m scripts.train_demand_model
    python -m scripts.train_demand_model --min-points 20   # mínimo de datos por producto
    python -m scripts.train_demand_model --dry-run         # solo muestra qué haría

Requisitos:
    pip install -r requirements-ml.txt
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

# Silenciar logs de cmdstanpy/prophet/numpy
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)
logging.getLogger("numexpr").setLevel(logging.CRITICAL)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_sales_data(db) -> list[dict]:
    """Carga todos los registros de sales_data desde Supabase."""
    result = db.table("sales_data").select(
        "date, product_name, quantity"
    ).order("date", desc=False).execute()
    return result.data or []


def group_by_product(records: list[dict]) -> dict[str, list[dict]]:
    """Agrupa registros por producto."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        product = (r.get("product_name") or "desconocido").strip()
        if not product:
            continue
        groups.setdefault(product, []).append({
            "date": r["date"],
            "quantity": r["quantity"],
        })
    return groups


def save_forecasts(db, product: str, forecasts: dict, metrics: dict, n_records: int, dry_run: bool):
    """Guarda las predicciones de un producto en demand_forecasts."""
    run_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for horizon, fc in forecasts.items():
        rows.append({
            "forecast_run_at": run_at,
            "product": product,
            "target_date": fc["target_date"],
            "horizon": horizon,
            "predicted_qty": fc["predicted_qty"],
            "lower_bound": fc["lower_bound"],
            "upper_bound": fc["upper_bound"],
            "model_name": "sales_demand_forecast",
            "data_points": n_records,
            "mape": metrics.get("mape"),
        })

    if dry_run:
        logger.info(f"  [DRY RUN] guardaría {len(rows)} filas para '{product}'")
        return

    db.table("demand_forecasts").insert(rows).execute()
    logger.info(f"  ✓ {len(rows)} forecasts guardados en Supabase para '{product}'")


def get_demand_model_id(db) -> str | None:
    try:
        r = db.table("ml_models").select("id").eq("type", "demand_forecast").execute()
        return r.data[0]["id"] if r.data else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de demanda por producto")
    parser.add_argument("--min-points", type=int, default=30,
                        help="Mínimo de fechas únicas por producto para entrenar (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra qué haría sin guardar nada")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  ENTRENAMIENTO — Modelo de Forecasting de Demanda")
    print(f"  Mínimo de puntos por producto: {args.min_points}")
    print(f"  Modo: {'DRY RUN' if args.dry_run else 'PRODUCCIÓN'}")
    print(f"{'='*60}\n")

    from database.supabase_client import get_supabase
    from models.demand_forecast import DemandForecastModel

    db = get_supabase()

    # 1. Cargar datos
    logger.info("Cargando datos de sales_data desde Supabase...")
    records = load_sales_data(db)

    if not records:
        logger.error("No hay datos en sales_data. Importa tu Excel primero:")
        logger.error("  python -m scripts.import_excel --file tu_archivo.xlsx")
        sys.exit(1)

    logger.info(f"Total registros cargados: {len(records):,}")

    # 2. Agrupar por producto
    groups = group_by_product(records)
    logger.info(f"Productos encontrados: {len(groups)}")

    # 3. Filtrar por mínimo de puntos
    trainable = {p: recs for p, recs in groups.items() if len(recs) >= args.min_points}
    skipped = {p: len(recs) for p, recs in groups.items() if len(recs) < args.min_points}

    print(f"Productos con suficientes datos (≥{args.min_points}): {len(trainable)}")
    if skipped:
        print(f"Productos omitidos por pocos datos: {len(skipped)}")
        for p, n in list(skipped.items())[:5]:
            print(f"  — {p}: {n} registros")
        if len(skipped) > 5:
            print(f"  ... y {len(skipped)-5} más")
    print()

    if not trainable:
        logger.error(f"Ningún producto tiene suficientes datos (mínimo {args.min_points}).")
        logger.error("Reduce --min-points o importa más datos.")
        sys.exit(1)

    # 4. Entrenar modelo por producto
    results_summary = []
    total_success = 0
    total_failed = 0

    for product, product_records in trainable.items():
        print(f"  Entrenando: {product[:50]} ({len(product_records)} registros)...")

        model = DemandForecastModel(product_name=product)
        metrics = model.train(product_records)

        if "error" in metrics:
            logger.warning(f"  ✗ Error en '{product}': {metrics['error']}")
            total_failed += 1
            continue

        try:
            forecasts = model.predict()
            save_forecasts(db, product, forecasts, metrics, model.n_records, args.dry_run)

            mape_str = f"{metrics['mape']:.1f}%" if metrics.get("mape") else "N/A"
            print(f"    MAPE: {mape_str} | "
                  f"sem1={forecasts['week_1']['predicted_qty']:.1f} | "
                  f"mes1={forecasts['month_1']['predicted_qty']:.1f} | "
                  f"mes2={forecasts['month_2']['predicted_qty']:.1f}")

            results_summary.append({
                "product": product,
                "mape": metrics.get("mape"),
                "week_1": forecasts["week_1"]["predicted_qty"],
                "month_1": forecasts["month_1"]["predicted_qty"],
            })
            total_success += 1

        except Exception as e:
            logger.error(f"  ✗ Predicción falló para '{product}': {e}")
            total_failed += 1

    # 5. Actualizar ml_models y ml_model_runs
    if not args.dry_run and total_success > 0:
        try:
            avg_mape = None
            mapes = [r["mape"] for r in results_summary if r.get("mape")]
            if mapes:
                avg_mape = round(sum(mapes) / len(mapes), 2)

            model_id = get_demand_model_id(db)
            if model_id:
                db.table("ml_models").update({
                    "metrics": {
                        "mape": avg_mape,
                        "products_trained": total_success,
                        "last_run": datetime.now(timezone.utc).isoformat(),
                    },
                    "last_evaluated": datetime.now(timezone.utc).isoformat(),
                }).eq("id", model_id).execute()

                db.table("ml_model_runs").insert({
                    "model_id": model_id,
                    "metrics": {
                        "avg_mape": avg_mape,
                        "products_trained": total_success,
                        "products_failed": total_failed,
                    },
                    "data_source": "excel_import",
                    "rows_processed": len(records),
                    "status": "success",
                    "notes": f"{total_success} productos entrenados",
                }).execute()
        except Exception as e:
            logger.warning(f"No se pudo actualizar ml_models: {e}")

    # 6. Resumen final
    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"  Productos entrenados: {total_success}")
    print(f"  Productos con error:  {total_failed}")
    if results_summary:
        mapes = [r["mape"] for r in results_summary if r.get("mape")]
        if mapes:
            print(f"  MAPE promedio: {sum(mapes)/len(mapes):.1f}%")
    print()
    print("  Siguiente paso: pregunta al bot en Telegram")
    print('  "¿Cuánto venderemos de [producto] el próximo mes?"')
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
