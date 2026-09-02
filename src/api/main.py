"""
Thin API layer. Only needed if your frontend is a separate React app calling
over REST. If you go the Streamlit route (recommended for hackathon speed),
you can import forecaster/allocator directly in app.py and skip this file.
Kept here so you have the option either way.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.forecast.model import DemandForecaster
from src.optimize.allocator import allocate, DEFAULT_PRIORITY

app = FastAPI(title="Delhi Grid AI - Forecast & Allocation API")

forecaster = DemandForecaster()
forecaster.load("src/forecast/forecaster.joblib")


class ScenarioRequest(BaseModel):
    timestamp: str          # ISO string
    hour: int
    is_festival: bool
    temp_c: float
    cloud_cover: float
    total_available_mw: float
    scenario: str = "p50"   # "p10" (best case) | "p50" (expected) | "p90" (worst case)


@app.get("/")
def root():
    return {"status": "ok", "message": "Delhi Grid AI API running"}


@app.post("/forecast_and_allocate")
def forecast_and_allocate(req: ScenarioRequest):
    from data.generate_synthetic_data import ZONES, ZONE_TX_CAPACITY_MW

    rows = [{
        "timestamp": req.timestamp, "zone": z, "hour": req.hour,
        "is_festival": int(req.is_festival), "temp_c": req.temp_c,
        "cloud_cover": req.cloud_cover,
    } for z in ZONES]
    df = pd.DataFrame(rows)

    preds = forecaster.predict(df)
    demand_by_zone = dict(zip(preds["zone"], preds[req.scenario]))

    result = allocate(
        demand_by_zone=demand_by_zone,
        total_available_mw=req.total_available_mw,
        tx_capacity=ZONE_TX_CAPACITY_MW,
        is_festival=req.is_festival,
        priority=DEFAULT_PRIORITY,
    )
    result["forecast_all_quantiles"] = preds.to_dict(orient="records")
    return result
