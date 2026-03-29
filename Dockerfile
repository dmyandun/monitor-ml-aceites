FROM python:3.11-slim

WORKDIR /app

# curl para HEALTHCHECK — no instalar gcc ni herramientas de compilación.
# Sin gcc, pip no puede compilar desde fuente: si un wheel no existe,
# falla inmediatamente con error claro en lugar de quedar colgado.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Capa 1: paquetes con extensiones Rust (los que pueden colgar si compilan)
# --only-binary=:all: rechaza cualquier sdist — si no hay wheel, falla rapido
# --timeout 60 evita esperas indefinidas si PyPI no responde
RUN pip install --no-cache-dir \
    --only-binary=:all: \
    --timeout 60 \
    "pydantic==2.11.3" \
    "pydantic-core==2.33.1" \
    "jiter==0.13.0"

# Capa 2: resto de dependencias (pure Python, nunca compilan)
RUN pip install --no-cache-dir \
    --only-binary=:all: \
    --timeout 60 \
    -r requirements.txt

COPY . .

EXPOSE 7860

# Si /health no responde en 3 intentos, HF marca el Space como unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
