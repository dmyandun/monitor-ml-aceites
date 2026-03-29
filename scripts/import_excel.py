"""
import_excel.py
--------------
Importa datos de ventas desde cualquier Excel o CSV.
Claude detecta automáticamente qué columna es fecha, producto, cantidad, etc.
No requiere configuración previa — funciona con cualquier estructura.

Uso:
    python -m scripts.import_excel --file ventas_2024.xlsx
    python -m scripts.import_excel --file ventas.csv --dry-run

Requisitos:
    pip install -r requirements-ml.txt   (incluye openpyxl, pandas)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Silenciar logs ruidosos
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── Detección de columnas con Claude ─────────────────────────────────────────

DETECTION_PROMPT = """Analiza los encabezados y datos de muestra de esta hoja de cálculo e identifica qué columna corresponde a cada campo de ventas.

ENCABEZADOS:
{headers}

FILAS DE MUESTRA (primeras {n} filas):
{sample_rows}

Identifica qué columna (usando el nombre exacto del encabezado) corresponde a:
- date: columna con fechas completas (formato YYYY-MM-DD, DD/MM/YYYY, etc.). Si no existe una sola columna de fecha pero hay columnas separadas de año y mes, usa null aquí.
- year: columna con el año (ANIO, AÑO, YEAR, etc.). null si no aplica.
- month: columna con el mes numérico (MES, MONTH, etc.). null si no aplica.
- product: columna con nombre del producto o ítem (obligatoria)
- quantity: columna con cantidad vendida — unidades, cajas, litros, kg, toneladas, etc. (obligatoria). Si hay varias columnas de cantidad, elige la más granular (cajas sobre toneladas).
- unit_price: columna con precio unitario (opcional, null si no existe)
- total_amount: columna con monto total de la venta (opcional, null si no existe)
- channel: columna con canal de venta — retail, mayorista, distribuidor, etc. (opcional, null si no existe)
- region: columna con región geográfica, zona o ciudad (opcional, null si no existe)

IMPORTANTE sobre números: algunas hojas usan coma como separador de miles (ej: "2,522" = dos mil quinientos veintidós). Indica si este es el caso en _thousands_sep.

Devuelve ÚNICAMENTE un objeto JSON con este formato exacto:
{{
  "date": "<nombre_columna_o_null>",
  "year": "<nombre_columna_o_null>",
  "month": "<nombre_columna_o_null>",
  "product": "<nombre_columna_o_null>",
  "quantity": "<nombre_columna_o_null>",
  "unit_price": "<nombre_columna_o_null>",
  "total_amount": "<nombre_columna_o_null>",
  "channel": "<nombre_columna_o_null>",
  "region": "<nombre_columna_o_null>",
  "_unit": "<unidad de la cantidad: units/kg/liters/boxes/tons/other>",
  "_thousands_sep": true,
  "_reasoning": "<explicación breve de tus decisiones>"
}}

Si no puedes identificar product o quantity, devuelve null para ese campo.
No inventes columnas — usa solo los nombres exactos que aparecen en los encabezados."""


def detect_columns(headers: list[str], sample_rows: list[list], filename: str) -> dict:
    """
    Usa Claude para identificar qué columna tiene qué información.
    Retorna un dict con el mapeo {campo_estándar: nombre_columna_original}.
    """
    import anthropic

    client = anthropic.Anthropic()

    sample_text = "\n".join(
        f"  Fila {i+1}: {dict(zip(headers, row))}"
        for i, row in enumerate(sample_rows[:8])
    )

    prompt = DETECTION_PROMPT.format(
        headers=headers,
        n=min(8, len(sample_rows)),
        sample_rows=sample_text,
    )

    logger.info(f"[import_excel] enviando {len(headers)} columnas a Claude para detección...")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku es suficiente para esta tarea
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Extraer JSON limpio (Claude puede añadir texto antes/después)
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Claude no devolvió JSON válido: {raw[:200]}")

    mapping = json.loads(raw[start:end])
    logger.info(f"[import_excel] columnas detectadas: {mapping}")
    return mapping


# ── Lectura del archivo ───────────────────────────────────────────────────────

def read_file(file_path: str) -> tuple[list[str], list[list]]:
    """Lee Excel o CSV y retorna (headers, rows)."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls", ".xlsm"):
        try:
            import openpyxl
        except ImportError:
            raise ImportError("Instala openpyxl: pip install openpyxl")

        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()

        if not rows:
            raise ValueError("El archivo Excel está vacío")

        headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
        data_rows = [list(r) for r in rows[1:] if any(v is not None for v in r)]

    elif suffix == ".csv":
        import csv
        with open(file_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("El archivo CSV está vacío")

        headers = [h.strip() for h in rows[0]]
        data_rows = [r for r in rows[1:] if any(v.strip() for v in r)]

    else:
        raise ValueError(f"Formato no soportado: {suffix}. Usa .xlsx, .xls o .csv")

    logger.info(f"[import_excel] archivo leído: {len(data_rows)} filas, {len(headers)} columnas")
    return headers, data_rows


# ── Normalización de datos ────────────────────────────────────────────────────

def _parse_date(value) -> str | None:
    """Convierte cualquier formato de fecha a ISO (YYYY-MM-DD)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):  # date object
        return value.isoformat()

    s = str(value).strip()
    if not s or s.lower() in ("none", "null", "nan"):
        return None

    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue

    logger.warning(f"[import_excel] no se pudo parsear fecha: {value!r}")
    return None


def _build_date_from_year_month(year_val, month_val) -> str | None:
    """Construye una fecha ISO desde columnas separadas de año y mes."""
    try:
        year = int(str(year_val).strip().split(".")[0])
        month = int(str(month_val).strip().split(".")[0])
        if 1 <= month <= 12 and 1990 <= year <= 2100:
            return f"{year:04d}-{month:02d}-01"
    except (ValueError, TypeError):
        pass
    return None


def _parse_number(value, thousands_sep: bool = False) -> float | None:
    """
    Convierte a float.
    thousands_sep=True: la coma es separador de miles (2,522 -> 2522).
    thousands_sep=False: la coma es separador decimal (2,52 -> 2.52).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value == value else None  # NaN check
    s = str(value).strip().replace(" ", "")
    if not s or s.lower() in ("none", "null", "nan", "-"):
        return None
    if thousands_sep:
        s = s.replace(",", "")   # 2,522 -> 2522
    else:
        s = s.replace(",", ".")  # 2,52 -> 2.52
    try:
        return float(s)
    except ValueError:
        return None


def normalize_rows(
    headers: list[str],
    data_rows: list[list],
    mapping: dict,
    filename: str,
) -> list[dict]:
    """
    Convierte las filas crudas al esquema estándar de sales_data.
    Maneja fechas únicas, fechas separadas año/mes, y miles con coma.
    """
    col_idx = {col: headers.index(col) for col in headers}
    thousands_sep = bool(mapping.get("_thousands_sep", False))

    def get(row, field_name):
        col = mapping.get(field_name)
        if not col or col not in col_idx:
            return None
        idx = col_idx[col]
        return row[idx] if idx < len(row) else None

    records = []
    skipped = 0

    for i, row in enumerate(data_rows):
        # ── Fecha ────────────────────────────────────────────
        if mapping.get("date"):
            date_val = _parse_date(get(row, "date"))
        elif mapping.get("year") and mapping.get("month"):
            date_val = _build_date_from_year_month(get(row, "year"), get(row, "month"))
        else:
            date_val = None

        product_val = str(get(row, "product") or "").strip()
        qty_val = _parse_number(get(row, "quantity"), thousands_sep=thousands_sep)

        if not date_val or not product_val or qty_val is None:
            skipped += 1
            continue

        record = {
            "date": date_val,
            "product_name": product_val,
            "quantity": qty_val,
            "unit_price": _parse_number(get(row, "unit_price"), thousands_sep=thousands_sep),
            "total_amount": _parse_number(get(row, "total_amount"), thousands_sep=thousands_sep),
            "channel": str(get(row, "channel") or "").strip() or None,
            "region": str(get(row, "region") or "").strip() or None,
            "metadata": {
                "source_file": filename,
                "row_index": i + 2,
                "unit": mapping.get("_unit", "units"),
            },
        }

        if record["total_amount"] is None and record["unit_price"] and record["quantity"]:
            record["total_amount"] = round(record["unit_price"] * record["quantity"], 4)

        records.append(record)

    if skipped:
        logger.warning(f"[import_excel] {skipped} filas omitidas (sin fecha, producto o cantidad)")

    return records


# ── Guardado en Supabase ──────────────────────────────────────────────────────

def save_to_supabase(
    records: list[dict],
    mapping: dict,
    filename: str,
    dry_run: bool = False,
) -> dict:
    """Inserta los registros normalizados en sales_data y registra en excel_imports."""
    if dry_run:
        products = list({r["product_name"] for r in records})
        logger.info(f"[import_excel] DRY RUN — {len(records)} filas no guardadas")
        return {
            "dry_run": True,
            "rows_would_import": len(records),
            "products_found": products,
            "mapping": mapping,
        }

    from database.supabase_client import get_supabase
    db = get_supabase()

    # Insertar en lotes de 200
    batch_size = 200
    inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        db.table("sales_data").insert(batch).execute()
        inserted += len(batch)
        logger.info(f"[import_excel] insertado lote {i // batch_size + 1} ({inserted}/{len(records)})")

    # Registrar importación
    products_found = list({r["product_name"] for r in records})
    db.table("excel_imports").insert({
        "filename": filename,
        "rows_imported": inserted,
        "column_mapping": mapping,
        "products_found": products_found,
        "status": "success",
    }).execute()

    return {
        "rows_imported": inserted,
        "products_found": products_found,
        "mapping": mapping,
    }


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Importa Excel/CSV de ventas a Supabase con detección automática de columnas."
    )
    parser.add_argument("--file", required=True, help="Ruta al archivo .xlsx, .xls o .csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra qué importaría, sin guardar nada")
    args = parser.parse_args()

    file_path = args.file
    filename = Path(file_path).name

    if not Path(file_path).exists():
        logger.error(f"Archivo no encontrado: {file_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Importando: {filename}")
    print(f"  Modo: {'DRY RUN (no guarda)' if args.dry_run else 'PRODUCCIÓN (guarda en Supabase)'}")
    print(f"{'='*60}\n")

    # 1. Leer archivo
    headers, data_rows = read_file(file_path)
    print(f"OK Archivo leído: {len(data_rows):,} filas, {len(headers)} columnas")
    print(f"  Columnas encontradas: {headers}\n")

    # 2. Detectar columnas con Claude
    mapping = detect_columns(headers, data_rows[:10], filename)
    print("OK Columnas detectadas por Claude:")
    for field, col in mapping.items():
        if not field.startswith("_"):
            status = "OK" if col else "—"
            print(f"  {status} {field:15} -> {col or '(no encontrado)'}")
    print(f"\n  Unidad detectada: {mapping.get('_unit', 'desconocida')}")
    print(f"  Razonamiento: {mapping.get('_reasoning', '')}\n")

    # Verificar campos obligatorios
    # Para fecha: puede ser columna única (date) o año+mes (year+month)
    has_date = mapping.get("date") or (mapping.get("year") and mapping.get("month"))
    missing = []
    if not has_date:
        missing.append("date (o year+month)")
    if not mapping.get("product"):
        missing.append("product")
    if not mapping.get("quantity"):
        missing.append("quantity")
    if missing:
        logger.error(f"No se detectaron columnas obligatorias: {missing}")
        logger.error("Verifica que el archivo tenga columnas de fecha, producto y cantidad.")
        sys.exit(1)

    # 3. Normalizar filas
    records = normalize_rows(headers, data_rows, mapping, filename)
    print(f"OK Filas normalizadas: {len(records):,}")

    # Mostrar muestra
    if records:
        print("\n  Muestra de 3 filas procesadas:")
        for r in records[:3]:
            print(f"    {r['date']} | {r['product_name'][:30]} | qty: {r['quantity']} | "
                  f"total: {r.get('total_amount')}")

    # 4. Guardar (o simular)
    print()
    result = save_to_supabase(records, mapping, filename, dry_run=args.dry_run)

    if args.dry_run:
        print(f"OK DRY RUN completado — se importarían {result['rows_would_import']:,} filas")
    else:
        print(f"OK Importación completada: {result['rows_imported']:,} filas en Supabase")

    print(f"\n  Productos encontrados ({len(result.get('products_found', []))}):")
    for p in sorted(result.get("products_found", []))[:20]:
        print(f"    • {p}")

    print(f"\n{'='*60}")
    print("  Siguiente paso: entrenar el modelo de demanda")
    print("  python -m scripts.train_demand_model")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
