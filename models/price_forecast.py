"""
price_forecast.py
-----------------
Modelo de forecasting de precios del aceite de palma usando Prophet.

Entrenado con datos mensuales de FRED (PPOILUSDM) + datos diarios de Commodities-API.
Genera predicciones para 4 horizontes:
  - tomorrow    : precio estimado mañana
  - next_week   : precio medio estimado la próxima semana
  - month_1     : precio medio estimado el próximo mes
  - month_2     : precio medio estimado en 2 meses

Prophet maneja datos mensuales/diarios mezclados, tendencias y estacionalidad.

Uso:
    from models.price_forecast import PriceForecastModel
    model = PriceForecastModel()
    model.train(records)         # records: list[{date, actual_price}]
    forecasts = model.predict()  # dict con los 4 horizontes
"""

import logging
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)


def _check_deps():
    """Valida que prophet y pandas estén disponibles."""
    try:
        import pandas  # noqa: F401
        import prophet  # noqa: F401
        return True
    except ImportError as e:
        logger.error(f"[PriceForecast] dependencia faltante: {e}. Instala prophet y pandas.")
        return False


class PriceForecastModel:
    """
    Encapsula entrenamiento y predicción con Prophet para precios de palma.
    """

    def __init__(self):
        self.model = None
        self.last_trained: datetime | None = None
        self.train_mape: float | None = None
        self.n_records: int = 0

    def train(self, records: list[dict]) -> dict:
        """
        Entrena el modelo Prophet con registros históricos de precio.

        Args:
            records: lista de dicts con 'date' (str ISO) y 'actual_price' (float)

        Returns:
            dict con métricas: mape, mae, rmse, n_records
        """
        if not _check_deps():
            return {"error": "prophet/pandas no disponibles"}

        import pandas as pd
        import numpy as np
        from prophet import Prophet
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Preparar DataFrame para Prophet (requiere columnas 'ds' y 'y')
        df = pd.DataFrame([
            {"ds": pd.to_datetime(r["date"]), "y": float(r["actual_price"])}
            for r in records
            if r.get("actual_price") and float(r["actual_price"]) > 0
        ]).sort_values("ds").drop_duplicates("ds")

        if len(df) < 12:
            return {"error": f"Datos insuficientes: {len(df)} registros (mínimo 12)"}

        logger.info(f"[PriceForecast] entrenando con {len(df)} registros "
                    f"({df['ds'].min().date()} → {df['ds'].max().date()})")

        # Configurar Prophet
        # - weekly_seasonality=False (datos mensuales/diarios sin patrón semanal claro)
        # - yearly_seasonality=True  (palma tiene ciclos anuales)
        # - changepoint_prior_scale=0.05 (moderado, evita overfit en series commodities)
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.80,  # intervalos de confianza al 80%
        )
        model.fit(df)

        # Evaluar con cross-validation sobre últimos 24 puntos (2 años)
        metrics = self._evaluate(model, df)

        self.model = model
        self.last_trained = datetime.utcnow()
        self.train_mape = metrics.get("mape")
        self.n_records = len(df)

        logger.info(f"[PriceForecast] entrenamiento completado — MAPE: {self.train_mape:.1f}%")
        return metrics

    def _evaluate(self, model, df) -> dict:
        """Evaluación simple: re-predice los últimos 24 puntos y calcula errores."""
        try:
            import numpy as np
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            n_eval = min(24, len(df) // 4)
            if n_eval < 6:
                return {"mape": None, "mae": None, "rmse": None}

            train_df = df.iloc[:-n_eval]
            test_df = df.iloc[-n_eval:]

            from prophet import Prophet
            eval_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.80,
            )
            eval_model.fit(train_df)

            future = eval_model.make_future_dataframe(periods=n_eval, freq="MS")
            forecast = eval_model.predict(future)
            preds = forecast.tail(n_eval)["yhat"].values
            actuals = test_df["y"].values

            mae = mean_absolute_error(actuals, preds)
            rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
            mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)

            return {
                "mape": round(mape, 2),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "n_eval": n_eval,
            }
        except Exception as e:
            logger.warning(f"[PriceForecast] evaluación falló: {e}")
            return {"mape": None, "mae": None, "rmse": None}

    def predict(self) -> dict:
        """
        Genera predicciones para los 4 horizontes.

        Returns:
            dict con claves: tomorrow, next_week, month_1, month_2
            Cada valor: {target_date, predicted_price, lower_bound, upper_bound, horizon}
        """
        if not self.model:
            raise RuntimeError("Modelo no entrenado. Llama a train() primero.")

        import pandas as pd

        today = date.today()

        # Generar futuro diario para los próximos 65 días (cubre 2 meses completos)
        future = self.model.make_future_dataframe(periods=65, freq="D")
        forecast = self.model.predict(future)

        # Filtrar solo predicciones futuras
        forecast["ds_date"] = forecast["ds"].dt.date
        future_fc = forecast[forecast["ds_date"] > today].copy()

        def get_forecast_for_date(target: date) -> dict:
            row = future_fc[future_fc["ds_date"] == target]
            if row.empty:
                # Buscar el más cercano
                row = future_fc.iloc[(future_fc["ds_date"] - target).abs().argsort()[:1]]
            return {
                "target_date": target.isoformat(),
                "predicted_price": round(float(row["yhat"].iloc[0]), 2),
                "lower_bound": round(float(row["yhat_lower"].iloc[0]), 2),
                "upper_bound": round(float(row["yhat_upper"].iloc[0]), 2),
            }

        def get_mean_forecast_range(start: date, end: date) -> dict:
            """Precio medio esperado en un rango de fechas."""
            mask = (future_fc["ds_date"] >= start) & (future_fc["ds_date"] <= end)
            subset = future_fc[mask]
            if subset.empty:
                return get_forecast_for_date(start)
            mid_date = start + (end - start) // 2
            return {
                "target_date": mid_date.isoformat(),
                "predicted_price": round(float(subset["yhat"].mean()), 2),
                "lower_bound": round(float(subset["yhat_lower"].mean()), 2),
                "upper_bound": round(float(subset["yhat_upper"].mean()), 2),
            }

        tomorrow = today + timedelta(days=1)
        next_week_start = today + timedelta(days=7)
        next_week_end = today + timedelta(days=13)
        month_1_start = today + timedelta(days=28)
        month_1_end = today + timedelta(days=35)
        month_2_start = today + timedelta(days=56)
        month_2_end = today + timedelta(days=63)

        results = {
            "tomorrow": {**get_forecast_for_date(tomorrow), "horizon": "tomorrow"},
            "next_week": {**get_mean_forecast_range(next_week_start, next_week_end), "horizon": "next_week"},
            "month_1": {**get_mean_forecast_range(month_1_start, month_1_end), "horizon": "month_1"},
            "month_2": {**get_mean_forecast_range(month_2_start, month_2_end), "horizon": "month_2"},
        }

        logger.info(
            f"[PriceForecast] predicciones generadas — "
            f"mañana: ${results['tomorrow']['predicted_price']}, "
            f"mes 1: ${results['month_1']['predicted_price']}, "
            f"mes 2: ${results['month_2']['predicted_price']}"
        )
        return results
