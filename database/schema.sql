-- ============================================================
-- schema.sql — monitor-ml-aceites
-- Ejecutar en el SQL Editor de Supabase (una sola vez)
-- ============================================================

-- ─── Extensiones ──────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─── Agentes ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agents (
    id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name          VARCHAR(100) UNIQUE NOT NULL,
    description   TEXT,
    system_prompt TEXT NOT NULL DEFAULT '',
    model         VARCHAR(100) DEFAULT 'claude-sonnet-4-6',
    tools         JSONB DEFAULT '[]',
    skills        JSONB DEFAULT '[]',
    memory        JSONB DEFAULT '{}',
    version_hash  VARCHAR(64),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Sesiones de conversación ─────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL,
    agent_name       VARCHAR(100),
    history          JSONB DEFAULT '[]',
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (telegram_user_id);

-- ─── Modelos ML ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml_models (
    id             UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name           VARCHAR(100) UNIQUE NOT NULL,
    type           VARCHAR(50) NOT NULL,   -- 'price_forecast' | 'demand_forecast'
    description    TEXT,
    metrics        JSONB DEFAULT '{}',     -- MAE, RMSE, MAPE, R², etc.
    last_evaluated TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Insertar modelos base
INSERT INTO ml_models (name, type, description) VALUES
    ('palm_price_forecast', 'price_forecast', 'Forecasting de precio mundial aceite de palma (FRED/World Bank)'),
    ('sales_demand_forecast', 'demand_forecast', 'Forecasting de demanda y ventas de aceites y mantecas')
ON CONFLICT (name) DO NOTHING;

-- ─── Ejecuciones de modelos ML ────────────────────────────
CREATE TABLE IF NOT EXISTS ml_model_runs (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_id        UUID REFERENCES ml_models(id) ON DELETE CASCADE,
    run_date        TIMESTAMPTZ DEFAULT NOW(),
    metrics         JSONB NOT NULL DEFAULT '{}',
    data_source     VARCHAR(100),
    rows_processed  INTEGER,
    status          VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'failed', 'warning')),
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_runs_model ON ml_model_runs (model_id, run_date DESC);

-- ─── Serie de precios (aceite de palma) ───────────────────
CREATE TABLE IF NOT EXISTS price_data (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    date            DATE NOT NULL,
    actual_price    DECIMAL(10,2),         -- USD/tonelada métrica
    predicted_price DECIMAL(10,2),
    source          VARCHAR(50),           -- 'FRED' | 'WorldBank' | 'CFN' | 'ANCUPA'
    currency        VARCHAR(3) DEFAULT 'USD',
    unit            VARCHAR(20) DEFAULT 'USD/MT',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (date, source)
);
CREATE INDEX IF NOT EXISTS idx_price_data_date ON price_data (date DESC);

-- ─── Datos de ventas ──────────────────────────────────────
-- Diseñado para recibir datos reales del negocio (Excel plano en el futuro)
CREATE TABLE IF NOT EXISTS sales_data (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    source_id        UUID,                  -- referencia a data_sources
    date             DATE NOT NULL,
    product_name     VARCHAR(200),
    product_category VARCHAR(100),          -- 'aceite', 'manteca', 'otros'
    quantity         DECIMAL(10,2),
    unit_price       DECIMAL(10,2),
    total_amount     DECIMAL(10,2),
    currency         VARCHAR(3) DEFAULT 'USD',
    region           VARCHAR(100),
    channel          VARCHAR(50),           -- 'retail', 'wholesale', 'ecommerce'
    metadata         JSONB DEFAULT '{}',   -- columnas extra del Excel original
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_data (date DESC);
CREATE INDEX IF NOT EXISTS idx_sales_category ON sales_data (product_category);

-- ─── Inventario ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS inventory_status (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    product_name     VARCHAR(200) NOT NULL,
    product_category VARCHAR(100),
    current_stock    DECIMAL(10,2),
    unit             VARCHAR(20) DEFAULT 'kg',
    reorder_point    DECIMAL(10,2),
    risk_level       VARCHAR(20) DEFAULT 'ok' CHECK (risk_level IN ('ok', 'low', 'critical')),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Fuentes de datos ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_sources (
    id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name         VARCHAR(100) NOT NULL,
    type         VARCHAR(50),               -- 'kaggle' | 'fred' | 'world_bank' | 'excel_import' | 'ancupa'
    description  TEXT,
    url          TEXT,
    last_updated TIMESTAMPTZ,
    rows_count   INTEGER,
    columns      JSONB DEFAULT '[]',
    metadata     JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Insertar fuentes conocidas
INSERT INTO data_sources (name, type, description, url) VALUES
    ('FRED PPOILUSDM', 'fred', 'Precio mundial aceite de palma mensual (IMF vía FRED)', 'https://fred.stlouisfed.org/series/PPOILUSDM'),
    ('World Bank Pink Sheet', 'world_bank', 'Commodities price data mensual desde 1960', 'https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/world-bank-commodities-price-data-the-pink-sheet'),
    ('FMCG Daily Sales 2022-2024', 'kaggle', 'Ventas diarias FMCG sintéticas (proxy hasta tener datos reales)', 'https://www.kaggle.com/datasets/beatafaron/fmcg-daily-sales-data-to-2022-2024'),
    ('CFN Ficha Sectorial Palma 2024', 'cfn_ecuador', 'Precios productor y estadísticas palma africana Ecuador 2024', 'https://www.cfn.fin.ec/wp-content/uploads/2024/07/Ficha-Sectorial-Palma-Africana.pdf'),
    ('ANCUPA Ecuador', 'ancupa', 'Estadísticas nacionales palma aceitera Ecuador desde 1994', 'http://ancupa.com/estadisticas/')
ON CONFLICT DO NOTHING;

-- ─── Agent Lab: Research ──────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_lab_research (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    source_url      TEXT NOT NULL UNIQUE,
    source_tier     VARCHAR(50),            -- 'anthropic' | 'huggingface' | 'langchain' | etc.
    title           TEXT NOT NULL,
    summary         TEXT,
    relevance_score DECIMAL(3,2) DEFAULT 0.00 CHECK (relevance_score BETWEEN 0 AND 1),
    tags            JSONB DEFAULT '[]',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_research_score ON agent_lab_research (relevance_score DESC);

-- ─── Agent Lab: Recomendaciones ───────────────────────────
CREATE TABLE IF NOT EXISTS agent_lab_recommendations (
    id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    type         VARCHAR(20) NOT NULL CHECK (type IN ('reactive', 'proactive')),
    category     VARCHAR(100),              -- 'prompt' | 'tool' | 'model' | 'skill' | 'memory'
    target_agent VARCHAR(100) NOT NULL,
    title        TEXT NOT NULL,
    description  TEXT,
    rationale    TEXT,
    status       VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'applied')),
    priority     INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON agent_lab_recommendations (status, type);

-- ─── Agent Lab: Audit Log ─────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_lab_changes (
    id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    recommendation_id UUID REFERENCES agent_lab_recommendations(id),
    change_type       VARCHAR(20) CHECK (change_type IN ('auto', 'approved')),
    target_agent      VARCHAR(100),
    description       TEXT,
    before_state      JSONB,
    after_state       JSONB,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Forecasts de precios ─────────────────────────────────
-- Almacena las predicciones generadas por el modelo Prophet
-- El modelo se reentrena mensualmente via GitHub Actions cron
CREATE TABLE IF NOT EXISTS price_forecasts (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    forecast_run_at TIMESTAMPTZ DEFAULT NOW(),  -- cuándo se generó el forecast
    target_date     DATE NOT NULL,              -- fecha que se predice
    horizon         VARCHAR(20) NOT NULL,       -- 'tomorrow' | 'next_week' | 'month_1' | 'month_2'
    predicted_price DECIMAL(10,2) NOT NULL,     -- USD/MT
    lower_bound     DECIMAL(10,2),              -- intervalo 80%
    upper_bound     DECIMAL(10,2),              -- intervalo 80%
    model_name      VARCHAR(100) DEFAULT 'palm_price_forecast',
    data_points     INTEGER,                    -- registros usados para entrenar
    mape            DECIMAL(6,2),               -- error del modelo al momento del entrenamiento (%)
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_price_forecasts_run ON price_forecasts (forecast_run_at DESC);
CREATE INDEX IF NOT EXISTS idx_price_forecasts_horizon ON price_forecasts (horizon, forecast_run_at DESC);

-- ─── Triggers: updated_at automático ─────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_recommendations_updated_at
    BEFORE UPDATE ON agent_lab_recommendations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
