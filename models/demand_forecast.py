"""
demand_forecast.py
------------------
Modelo de forecasting de demanda usando Prophet.
Entrena un modelo por producto con datos reales del negocio.

Horizontes de predicción (orientados a planificación de compras):
  - week_1  : demanda estimada próximos 7 días
  - month_1 : demanda estimada próximo mes (días 1-30)
  - month_2 : demanda estimada en 2 meses (días 31-60)
  - month_3 : demanda estimada en 3 meses (días 61-90)

Uso:
    from models.demand_forecast import DemandForecastModel
    model = DemandForecastModel()
    metrics = model.train(records)   # records: list[{date, quantity}]
    forecasts = model.predict()      # dict con los 4 horizontes
"""

import logging
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)


def _check_deps():
    try:
        import pandas   # noqa
        import prophet  # noqa
        return True
    except ImportError as e:
        logger.error(f"[DemandForecast] dependencia faltante: {e}")
        return False


class DemandForecastModel:
    """Prophet para forecasting de demanda de un producto específico."""

    def __init__(self, product_name: str = "total"):
        self.product_name = product_name
        self.model = None
        self.last_trained: datetime | None = None
        self.train_mape: float | None = None
        self.n_records: int = 0
        self.unit: str = "units"

    def train(self, records: list[dict]) -> dict:
        """
        Entrena Prophet con registros históricos de venta de un producto.

        Args:
            records: lista de dicts con 'date' (str ISO) y 'quantity' (float).
                     Si hay múltiples filas por fecha, se suman (upsert por día).

        Returns:
            dict con métricas: mape, mae, rmse, n_records
        """
        if not _check_deps():
            return {"error": "prophet/pandas no disponibles"}

        import pandas as pd
        import numpy as np
        from prophet import Prophet

        # Agregar por fecha (suma de cantidades del día)
        df = pd.DataFrame([
            {"ds": pd.to_datetime(r["date"]), "y": float(r["quantity"])}
            for r in records
            if r.get("quantity") is not None and float(r["quantity"]) >= 0
        ])

        if df.empty:
            return {"error": "Sin datos válidos para entrenar"}

        df = df.groupby("ds")["y"].sum().reset_index()
        df = df.sort_values("ds").reset_index(drop=True)

        if len(df) < 12:
            return {"error": f"Datos insuficientes: {len(df)} fechas únicas (mínimo 12)"}

        logger.info(
            f"[DemandForecast:{self.product_name}] entrenando con {len(df)} días "
            f"({df['ds'].min().date()} → {df['ds'].max().date()})"
        )

        # Detectar frecuencia dominante (diaria o mensual)
        median_gap = (df["ds"].diff().median()).days if len(df) > 1 else 1
        is_monthly = median_gap > 20

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=not is_monthly,  # solo si hay datos diarios
            daily_seasonality=False,
            changepoint_prior_scale=0.1,   # algo más flexible que precios (demanda más volátil)
            seasonality_prior_scale=10.0,
            interval_width=0.80,
        )

        # Agregar estacionalidad mensual explícita si datos son diarios
        if not is_monthly:
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        model.fit(df)

        metrics = self._evaluate(model, df, is_monthly)

        self.model = model
        self.last_trained = datetime.utcnow()
        self.train_mape = metrics.get("mape")
        self.n_records = len(df)

        logger.info(
            f"[DemandForecast:{self.product_name}] listo — "
            f"MAPE: {self.train_mape}% | {self.n_records} puntos"
        )
        return metrics

    def _evaluate(self, model, df, is_monthly: bool) -> dict:
        """Evaluación hold-out: reserva los últimos N puntos para test."""
        try:
            import numpy as np
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            from prophet import Prophet

            n_eval = min(12 if is_monthly else 30, len(df) // 4)
            if n_eval < 4:
                return {"mape": None, "mae": None, "rmse": None}

            train_df = df.iloc[:-n_eval]
            test_df = df.iloc[-n_eval:]

            freq = "MS" if is_monthly else "D"
            eval_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=not is_monthly,
                daily_seasonality=False,
                changepoint_prior_scale=0.1,
                interval_width=0.80,
            )
            if not is_monthly:
                eval_model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
            eval_model.fit(train_df)

            future = eval_model.make_future_dataframe(periods=n_eval, freq=freq)
            forecast = eval_model.predict(future)
            preds = forecast.tail(n_eval)["yhat"].clip(lower=0).values
            actuals = test_df["y"].values

            mae = mean_absolute_error(actuals, preds)
            rmse = float(np.sqrt(mean_squared_error(actuals, preds)))

            # MAPE robusto (evita división por cero)
            mask = actuals > 0
            mape = float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100) if mask.any() else None

            return {
                "mape": round(mape, 2) if mape is not None else None,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "n_eval": n_eval,
            }
        except Exception as e:
            logger.warning(f"[DemandForecast:{self.product_name}] evaluación falló: {e}")
            return {"mape": None, "mae": None, "rmse": None}

    def predict(self) -> dict:
        """
        Genera predicciones para los 4 horizontes de planificación.

        Returns:
            dict con claves: week_1, month_1, month_2, month_3
        """
        if not self.model:
            raise RuntimeError("Modelo no entrenado. Llama a train() primero.")

        import pandas as pd
        import numpy as np

        today = date.today()

        # Generar 95 días hacia el futuro
        future = self.model.make_future_dataframe(periods=95, freq="D")
        forecast = self.model.predict(future)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
        forecast["ds_date"] = forecast["ds"].dt.date
        future_fc = forecast[forecast["ds_date"] > today].copy()

        def sum_range(start: date, end: date) -> dict:
            mask = (future_fc["ds_date"] >= start) & (future_fc["ds_date"] <= end)
            subset = future_fc[mask]
            if subset.empty:
                return {"predicted_qty": 0.0, "lower_bound": 0.0, "upper_bound": 0.0,
                        "target_date": start.isoformat()}
            mid = start + (end - start) // 2
            return {
                "target_date": mid.isoformat(),
                "predicted_qty": round(float(subset["yhat"].sum()), 2),
                "lower_bound": round(float(subset["yhat_lower"].sum()), 2),
                "upper_bound": round(float(subset["yhat_upper"].sum()), 2),
            }

        w1_end = today + timedelta(days=7)
        m1_end = today + timedelta(days=30)
        m2_start = today + timedelta(days=31)
        m2_end = today + timedelta(days=60)
        m3_start = today + timedelta(days=61)
        m3_end = today + timedelta(days=90)

        results = {
            "week_1":  {**sum_range(today + timedelta(1), w1_end),  "horizon": "week_1"},
            "month_1": {**sum_range(today + timedelta(1), m1_end),  "horizon": "month_1"},
            "month_2": {**sum_range(m2_start, m2_end),              "horizon": "month_2"},
            "month_3": {**sum_range(m3_start, m3_end),              "horizon": "month_3"},
        }

        logger.info(
            f"[DemandForecast:{self.product_name}] "
            f"sem1={results['week_1']['predicted_qty']} | "
            f"mes1={results['month_1']['predicted_qty']} | "
            f"mes2={results['month_2']['predicted_qty']}"
        )
        return results
