"""
Generates synthetic hourly load + solar + weather data for N Delhi zones.
Replace with real DERC/BSES/discom data if you get access during the hackathon —
this exists so the pipeline runs end-to-end from hour 1 without waiting on real data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

ZONES = ["Rohini", "Dwarka", "Connaught_Place", "Karol_Bagh", "Saket", "Shahdara"]
ZONE_BASE_LOAD_MW = {  # rough relative scale, not real figures
    "Rohini": 180, "Dwarka": 160, "Connaught_Place": 220,
    "Karol_Bagh": 140, "Saket": 150, "Shahdara": 170,
}
ZONE_TX_CAPACITY_MW = {z: v * 1.35 for z, v in ZONE_BASE_LOAD_MW.items()}  # feeder headroom
ZONE_SOLAR_CAPACITY_MW = {  # rooftop solar penetration varies by zone
    "Rohini": 25, "Dwarka": 30, "Connaught_Place": 10,
    "Karol_Bagh": 8, "Saket": 22, "Shahdara": 15,
}
# festival dates you can toggle to test "soft priority" logic (temples, markets)
FESTIVAL_DATES = {"2025-10-20", "2025-11-01"}  # example: Diwali-ish window

# ----------------------------------------------------------------------------
# Hyperlocal microclimate. Previously every zone got the exact same temp_c /
# cloud_cover each hour, and the heat-response formula was identical across
# zones -- so the trained model had literally no way to learn that geography
# matters. These offsets + sensitivities are applied per zone below, so
# zone identity and local weather now genuinely interact in the training
# data (and the model in src/forecast/model.py can learn that interaction,
# rather than the app faking it after the fact at inference time).
# ----------------------------------------------------------------------------
ZONE_TEMP_OFFSET_C = {   # urban heat island in dense zones, cooling in green zones
    "Connaught_Place": 1.6,
    "Karol_Bagh": 1.3,
    "Shahdara": 1.1,
    "Rohini": 0.2,
    "Dwarka": -0.6,
    "Saket": -0.9,
}
ZONE_CLOUD_DELTA = {     # small local sky-cover variation
    "Connaught_Place": -0.03,
    "Karol_Bagh": -0.02,
    "Shahdara": 0.01,
    "Rohini": 0.00,
    "Dwarka": 0.02,
    "Saket": 0.03,
}
ZONE_HEAT_SENSITIVITY = {  # how strongly AC load responds to heat above 30C
    "Connaught_Place": 1.3,  # dense commercial AC load
    "Karol_Bagh": 1.2,       # market density, older buildings
    "Shahdara": 1.0,
    "Rohini": 1.0,
    "Dwarka": 0.9,           # newer, better-insulated stock
    "Saket": 0.6,            # hospital hub: backup/critical-load HVAC, less heat-elastic
}


def hourly_load_multiplier(hour, is_festival):
    """Double daily peak shape: morning + evening, AC-driven summer evening spike."""
    base = (
        0.55
        + 0.35 * np.exp(-((hour - 9) ** 2) / 8)   # morning peak
        + 0.55 * np.exp(-((hour - 20) ** 2) / 10)  # evening peak (AC + lighting)
    )
    if is_festival:
        base *= 1.15  # festival lighting/markets bump
    return base


def solar_multiplier(hour):
    if hour < 6 or hour > 18:
        return 0.0
    # bell curve peaking at noon
    return max(0.0, np.exp(-((hour - 13) ** 2) / 12))


def generate(days=30, start_date="2025-10-01"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []
    for d in range(days):
        date = start + timedelta(days=d)
        is_festival = date.strftime("%Y-%m-%d") in FESTIVAL_DATES
        temp_c = 28 + 6 * np.sin(2 * np.pi * d / 30) + np.random.normal(0, 1.5)
        cloud_cover = np.clip(np.random.beta(2, 5), 0, 1)  # mostly clear, some cloudy days

        for hour in range(24):
            for zone in ZONES:
                # hyperlocal weather: city-wide temp_c/cloud_cover perturbed by
                # this zone's microclimate, so the model can learn that (e.g.)
                # Connaught_Place runs hotter and more heat-elastic than Saket.
                local_temp_c = temp_c + ZONE_TEMP_OFFSET_C[zone]
                local_cloud_cover = float(np.clip(cloud_cover + ZONE_CLOUD_DELTA[zone], 0, 1))

                load_mult = hourly_load_multiplier(hour, is_festival)
                noise = np.random.normal(1.0, 0.06)
                temp_effect = 1 + max(0, local_temp_c - 30) * 0.015 * ZONE_HEAT_SENSITIVITY[zone]
                demand = ZONE_BASE_LOAD_MW[zone] * load_mult * temp_effect * noise

                solar_gen = (
                    ZONE_SOLAR_CAPACITY_MW[zone]
                    * solar_multiplier(hour)
                    * (1 - 0.7 * local_cloud_cover)
                )

                rows.append({
                    "timestamp": date + timedelta(hours=hour),
                    "zone": zone,
                    "hour": hour,
                    "is_festival": int(is_festival),
                    "temp_c": round(local_temp_c, 1),
                    "cloud_cover": round(local_cloud_cover, 2),
                    "demand_mw": round(demand, 2),
                    "solar_gen_mw": round(solar_gen, 2),
                    "tx_capacity_mw": ZONE_TX_CAPACITY_MW[zone],
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate(days=45)
    df.to_csv("data/delhi_synthetic_load.csv", index=False)
    print(f"Generated {len(df)} rows -> data/delhi_synthetic_load.csv")
    print(df.head())