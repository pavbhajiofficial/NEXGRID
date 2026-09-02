"""
Forecasting layer: predicts P10 / P50 / P90 demand per zone per hour.
Uses GradientBoostingRegressor with quantile loss (fast, no GPU needed,
trains in seconds on synthetic data — good enough for a live demo).

Swap in an LSTM later if you have time; the API/optimizer code below
doesn't care what's inside this box as long as it returns the same dict shape.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

QUANTILES = [0.1, 0.5, 0.9]
FEATURES = ["hour", "is_festival", "temp_c", "cloud_cover", "zone_enc", "dow"]


class DemandForecaster:
    def __init__(self):
        self.models = {}  # quantile -> trained model
        self.zone_encoder = LabelEncoder()

    def _prep(self, df: pd.DataFrame, fit_encoder=False) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["dow"] = df["timestamp"].dt.dayofweek
        if fit_encoder:
            df["zone_enc"] = self.zone_encoder.fit_transform(df["zone"])
        else:
            df["zone_enc"] = self.zone_encoder.transform(df["zone"])
        return df

    def train(self, df: pd.DataFrame):
        df = self._prep(df, fit_encoder=True)
        X = df[FEATURES]
        y = df["demand_mw"]
        for q in QUANTILES:
            model = GradientBoostingRegressor(
                loss="quantile", alpha=q,
                n_estimators=150, max_depth=3, learning_rate=0.08,
            )
            model.fit(X, y)
            self.models[q] = model
        print(f"Trained quantile models for {QUANTILES} on {len(df)} rows.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._prep(df, fit_encoder=False)
        X = df[FEATURES]
        out = df[["timestamp", "zone", "hour"]].copy()
        for q in QUANTILES:
            out[f"p{int(q*100)}"] = self.models[q].predict(X)
        # enforce monotonicity p10 <= p50 <= p90 (quantile crossing can happen)
        out[["p10", "p50", "p90"]] = np.sort(out[["p10", "p50", "p90"]].values, axis=1)
        return out

    def save(self, path="src/forecast/forecaster.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"models": self.models, "encoder": self.zone_encoder}, path)

    def load(self, path="src/forecast/forecaster.joblib"):
        obj = joblib.load(path)
        self.models = obj["models"]
        self.zone_encoder = obj["encoder"]


if __name__ == "__main__":
    df = pd.read_csv("data/delhi_synthetic_load.csv")
    fc = DemandForecaster()
    fc.train(df)
    fc.save()
    preds = fc.predict(df.sample(5, random_state=1))
    print(preds)
