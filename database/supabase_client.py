"""
supabase_client.py
------------------
Cliente Supabase singleton. Se inicializa una sola vez con las variables de entorno.
"""

import os
import logging
from functools import lru_cache
from supabase import create_client, Client

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """
    Devuelve el cliente Supabase (singleton).
    Llama a esta función en cualquier lugar del proyecto para acceder a la DB.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Faltan variables de entorno SUPABASE_URL y/o SUPABASE_KEY. "
            "Copia .env.example a .env y completa los valores."
        )

    client = create_client(url, key)
    logger.info("[Supabase] cliente inicializado correctamente")
    return client
