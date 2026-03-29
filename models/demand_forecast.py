"""
demand_forecast.py
------------------
Modelo de forecasting de demanda usando Prophet.
Entrena un modelo por producto con datos reales del negocio.

Detecta automáticamente si los datos son diarios o mensuales
y ajusta las predicciones en consecuencia.

Horizontes de predicción (orientados a planificación de compras):
  - week_1  : demanda estimada próxima semana (o proporción mensual)
  - month_1 : demanda estimada próximo mes
  - month_2 : demanda estimada en 2 meses
  - month_3 : demanda estimada en 3 meses
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
        self.is_monthly = False
        self.last_trained: datetime | None = None
        self.train_mape: float | None = None
        self.n_records: int = 0

    def train(self, records: list[dict]) -> dict:
        """
        Entrena Prophet con registros históricos de venta de un producto.

        Args:
            records: lista de dicts con 'date' (str ISO) y 'quantity' (float).
                     Si hay múltiples filas por fecha, se suman.

        Returns:
            dict con métricas: mape, mae, rmse, n_records
        """
        if not _check_deps():
            return {"error": "prophet/pandas no disponibles"}

        import pandas as pd
        from prophet import Prophet

        df = pd.DataFrame([
            {"ds": pd.to_datetime(r["date"]), "y": float(r["quantity"])}
            for r in records
            if r.get("quantity") is not None and float(r["quantity"]) >= 0
        ])

        if df.empty:
            return {"error": "Sin datos validos para entrenar"}

        df = df.groupby("ds")["y"].sum().reset_index()
        df = df.sort_values("ds").reset_index(drop=True)

        if len(df) < 12:
            return {"error": f"Datos insuficientes: {len(df)} fechas unicas (minimo 12)"}

        # Detectar frecuencia: si el gap promedio es > 20 días, son datos mensuales
        median_gap = int(df["ds"].diff().dt.days.median()) if len(df) > 1 else 1
        self.is_monthly = median_gap > 20
        freq_label = "mensual" if self.is_monthly else "diaria"

        logger.info(
            f"[DemandForecast:{self.product_name}] entrenando -- "
            f"{len(df)} puntos {freq_label}s "
            f"({df['ds'].min().date()} -> {df['ds'].max().date()})"
        )

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            interval_width=0.80,
        )

        if not self.is_monthly:
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        model.fit(df)

        metrics = self._evaluate(model, df)

        self.model = model
        self.last_trained = datetime.utcnow()
        self.train_mape = metrics.get("mape")
        self.n_records = len(df)

        logger.info(
            f"[DemandForecast:{self.product_name}] listo -- "
            f"MAPE: {self.train_mape}% | {self.n_records} puntos"
        )
        return metrics

    def _evaluate(self, model, df) -> dict:
        """Evaluación hold-out: reserva los últimos N puntos para test."""
        try:
            import numpy as np
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            from prophet import Prophet

            n_eval = min(12, len(df) // 4)
            if n_eval < 4:
                return {"mape": None, "mae": None, "rmse": None}

            train_df = df.iloc[:-n_eval]
            test_df = df.iloc[-n_eval:]

            freq = "MS" if self.is_monthly else "D"
            eval_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.1,
                interval_width=0.80,
            )
            eval_model.fit(train_df)

            future = eval_model.make_future_dataframe(periods=n_eval, freq=freq)
            forecast = eval_model.predict(future)
            preds = forecast.tail(n_eval)["yhat"].clip(lower=0).values
            actuals = test_df["y"].values

            mae = mean_absolute_error(actuals, preds)
            rmse = float(np.sqrt(mean_squared_error(actuals, preds)))

            mask = actuals > 0
            mape = float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100) if mask.any() else None

            return {
                "mape": round(mape, 2) if mape is not None else None,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "n_eval": n_eval,
            }
        except Exception as e:
            logger.warning(f"[DemandForecast:{self.product_name}] evaluacion fallo: {e}")
            return {"mape": None, "mae": None, "rmse": None}

    def predict(self) -> dict:
        """
        Genera predicciones para los 4 horizontes de planificación.

        Para datos mensuales: predice meses completos directamente.
        Para datos diarios: suma días en rangos.

        Returns:
            dict con claves: week_1, month_1, month_2, month_3
        """
        if not self.model:
            raise RuntimeError("Modelo no entrenado. Llama a train() primero.")

        import pandas as pd

        today = date.today()

        if self.is_monthly:
            return self._predict_monthly(today)
        else:
            return self._predict_daily(today)

    def _predict_monthly(self, today: date) -> dict:
        """Para datos mensuales: genera 4 predicciones mensuales hacia adelante."""
        import pandas as pd

        # Generar 4 meses hacia el futuro con frecuencia mensual
        future = self.model.make_future_dataframe(periods=4, freq="MS")
        forecast = self.model.predict(future)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
        forecast["ds_date"] = forecast["ds"].dt.date

        # Filtrar solo predicciones futuras (ds > hoy)
        future_fc = forecast[forecast["ds_date"] > today].head(4).reset_index(drop=True)

        if future_fc.empty:
            return self._empty_forecasts(today)

        def row_to_forecast(idx: int, horizon: str) -> dict:
            if idx >= len(future_fc):
                return {"target_date": today.isoformat(), "predicted_qty": 0.0,
                        "lower_bound": 0.0, "upper_bound": 0.0, "horizon": horizon}
            row = future_fc.iloc[idx]
            return {
                "target_date": str(row["ds_date"]),
                "predicted_qty": round(float(row["yhat"]), 1),
                "lower_bound": round(float(row["yhat_lower"]), 1),
                "upper_bound": round(float(row["yhat_upper"]), 1),
                "horizon": horizon,
            }

        # week_1: proporción de los primeros 7 días del mes siguiente (7/30 del mes)
        m1 = row_to_forecast(0, "week_1")
        m1["predicted_qty"] = round(m1["predicted_qty"] * 7 / 30, 1)
        m1["lower_bound"] = round(m1["lower_bound"] * 7 / 30, 1)
        m1["upper_bound"] = round(m1["upper_bound"] * 7 / 30, 1)

        results = {
            "week_1":  m1,
            "month_1": row_to_forecast(0, "month_1"),
            "month_2": row_to_forecast(1, "month_2"),
            "month_3": row_to_forecast(2, "month_3"),
        }

        logger.info(
            f"[DemandForecast:{self.product_name}] "
            f"mes1={results['month_1']['predicted_qty']} | "
            f"mes2={results['month_2']['predicted_qty']} | "
            f"mes3={results['month_3']['predicted_qty']}"
        )
        return results

    def _predict_daily(self, today: date) -> dict:
        """Para datos diarios: suma predicciones en rangos de días."""
        import pandas as pd

        future = self.model.make_future_dataframe(periods=95, freq="D")
        forecast = self.model.predict(future)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
        forecast["ds_date"] = forecast["ds"].dt.date
        future_fc = forecast[forecast["ds_date"] > today].copy()

        def sum_range(start: date, end: date, horizon: str) -> dict:
            mask = (future_fc["ds_date"] >= start) & (future_fc["ds_date"] <= end)
            subset = future_fc[mask]
            if subset.empty:
                return {"target_date": start.isoformat(), "predicted_qty": 0.0,
                        "lower_bound": 0.0, "upper_bound": 0.0, "horizon": horizon}
            mid = start + (end - start) // 2
            return {
                "target_date": mid.isoformat(),
                "predicted_qty": round(float(subset["yhat"].sum()), 1),
                "lower_bound": round(float(subset["yhat_lower"].sum()), 1),
                "upper_bound": round(float(subset["yhat_upper"].sum()), 1),
                "horizon": horizon,
            }

        return {
            "week_1":  sum_range(today + timedelta(1), today + timedelta(7), "week_1"),
            "month_1": sum_range(today + timedelta(1), today + timedelta(30), "month_1"),
            "month_2": sum_range(today + timedelta(31), today + timedelta(60), "month_2"),
            "month_3": sum_range(today + timedelta(61), today + timedelta(90), "month_3"),
        }

    def _empty_forecasts(self, today: date) -> dict:
        return {h: {"target_date": today.isoformat(), "predicted_qty": 0.0,
                    "lower_bound": 0.0, "upper_bound": 0.0, "horizon": h}
                for h in ("week_1", "month_1", "month_2", "month_3")}
