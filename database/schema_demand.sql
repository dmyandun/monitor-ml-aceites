-- ============================================================
-- schema_demand.sql — monitor-ml-aceites
-- Tablas para el pipeline de demanda (Fase 2)
-- Ejecutar en el SQL Editor de Supabase (una sola vez)
-- ============================================================

-- ─── Forecasts de demanda ─────────────────────────────────
-- Almacena predicciones Prophet por producto.
-- Generadas localmente con train_demand_model.py
-- El agente demand_monitor las lee desde Supabase.
CREATE TABLE IF NOT EXISTS demand_forecasts (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    forecast_run_at TIMESTAMPTZ DEFAULT NOW(),
    product         TEXT NOT NULL,              -- nombre del producto
    target_date     DATE NOT NULL,              -- fecha central de la predicción
    horizon         VARCHAR(20) NOT NULL,       -- 'week_1' | 'month_1' | 'month_2' | 'month_3'
    predicted_qty   DECIMAL(14,2) NOT NULL,     -- cantidad predicha (unidades/kg/litros según datos)
    lower_bound     DECIMAL(14,2),              -- intervalo de confianza 80% inferior
    upper_bound     DECIMAL(14,2),              -- intervalo de confianza 80% superior
    unit            TEXT DEFAULT 'units',       -- unidad de la cantidad ('units', 'kg', 'liters')
    model_name      TEXT DEFAULT 'sales_demand_forecast',
    data_points     INTEGER,                    -- puntos de datos usados para entrenar
    mape            DECIMAL(6,2),               -- error del modelo (%)
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_demand_forecasts_run
    ON demand_forecasts (forecast_run_at DESC);
CREATE INDEX IF NOT EXISTS idx_demand_forecasts_product
    ON demand_forecasts (product, horizon, forecast_run_at DESC);

-- ─── Log de importaciones Excel ───────────────────────────
-- Registra cada archivo importado con el mapeo de columnas detectado.
CREATE TABLE IF NOT EXISTS excel_imports (
    id             UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    filename       TEXT NOT NULL,
    imported_at    TIMESTAMPTZ DEFAULT NOW(),
    rows_imported  INTEGER,
    column_mapping JSONB,        -- mapeo detectado por Claude {date, product, quantity, ...}
    products_found JSONB,        -- lista de productos únicos encontrados
    status         TEXT DEFAULT 'success',
    notes          TEXT
);
