"""
Builds data/custom_grid_dataset.csv: a zone-level electricity grid dataset where
the DEMAND is grounded in real observed Delhi grid data, and the zone split,
solar/storage/local-generation layers, network topology, and forecast columns
are a controlled simulation built on top of that real base.

Ground truth demand source: data/raw/load_data.csv
  - Real Delhi 5-min demand, 2023-04-01 to 2026-01-12 (this is the longer,
    ground-truth series -- used as-is, not re-derived).

Weather source: data/raw/powerdemand_5min_2021_to_2024_with_weather.csv
  - Real observed weather (temp, dew point, humidity, wind, pressure), 2021-2024.
  - Merged onto the demand series by timestamp for the overlapping window
    (2023-04 to 2024-12). For the remaining ~13 months (2025-01 to 2026-01,
    where this file has no coverage), weather is filled from a monthly x hourly
    climatology computed from the real weather data itself -- i.e. "what does
    a typical hour at this time of year look like", not invented numbers.

Everything from here down (5-zone split, solar, battery, local generation,
priority, transmission limits, forecast quantiles, scenario tags) is simulation
layered on top of that real, grounded demand -- and is documented as such.
Nothing here is presented as more "real" than it is.
"""
import numpy as np
import pandas as pd
import json
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_DIR = os.path.dirname(__file__)

ZONES = ["North", "South", "East", "West", "Central"]

# hour-of-day share CURVES per zone (relative weights, renormalized to sum to 1
# across zones at every hour so zones always sum back to the real citywide total).
# Shapes chosen to be distinct, not just fixed percentages of the total:
#   Central = commercial hub, strong daytime peak, quiet at night
#   North/South = residential-leaning, strong evening peak
#   East = flatter, industrial-leaning profile
#   West = mixed commercial/residential (airport/business-district-like)
def zone_hour_weight(zone, hour):
    if zone == "Central":
        return 0.9 + 1.3 * np.exp(-((hour - 13) ** 2) / 18)
    if zone == "North":
        return 1.0 + 1.1 * np.exp(-((hour - 20) ** 2) / 10)
    if zone == "South":
        return 1.0 + 1.0 * np.exp(-((hour - 21) ** 2) / 12)
    if zone == "East":
        return 1.0 + 0.5 * np.exp(-((hour - 11) ** 2) / 30) + 0.3 * np.exp(-((hour - 19) ** 2) / 15)
    if zone == "West":
        return 0.95 + 0.8 * np.exp(-((hour - 10) ** 2) / 20) + 0.6 * np.exp(-((hour - 19) ** 2) / 12)
    raise ValueError(zone)


ZONE_SOLAR_CAPACITY_MW = {"North": 35, "South": 30, "East": 18, "West": 28, "Central": 12}
ZONE_BATTERY_CAPACITY_MWH = {"North": 12, "South": 10, "East": 6, "West": 9, "Central": 5}
ZONE_BATTERY_MAX_RATE_MW = {z: round(cap / 3, 2) for z, cap in ZONE_BATTERY_CAPACITY_MWH.items()}
ZONE_LOCAL_GEN_CAPACITY_MW = {"North": 8, "South": 6, "East": 10, "West": 7, "Central": 4}
ZONE_BASE_PRIORITY = {"North": 1.0, "South": 1.0, "East": 1.0, "West": 1.0, "Central": 1.2}
FESTIVAL_BOOST_ZONES = {"Central", "West"}   # markets/commercial hubs
FESTIVAL_BOOST_FACTOR = 1.3

# Major Indian festival dates falling in the dataset's real date range (2023-04 to
# 2026-01) -- used for the FESTIVAL_SURGE scenario tag and soft priority boost.
FESTIVAL_DATES = {
    "2023-11-12",  # Diwali 2023
    "2024-03-25",  # Holi 2024
    "2024-11-01",  # Diwali 2024
    "2025-03-14",  # Holi 2025
    "2025-10-20",  # Diwali 2025 (approx)
}

PEAK_HOURS = {8, 9, 10, 18, 19, 20, 21}
FAIRNESS_PCT = 0.6


def load_and_prepare_base_demand():
    print("Loading real demand series (load_data.csv)...")
    df = pd.read_csv(os.path.join(RAW_DIR, "load_data.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df["load_MW"] = df["load_MW"].interpolate(limit=6).ffill().bfill()
    hourly = df["load_MW"].resample("h").mean().rename("citywide_demand_MW")
    print(f"  -> {len(hourly)} hourly points, {hourly.index.min()} to {hourly.index.max()}")
    return hourly.to_frame()


def load_and_prepare_weather():
    print("Loading real weather series (powerdemand file)...")
    df = pd.read_csv(os.path.join(RAW_DIR, "powerdemand_5min_2021_to_2024_with_weather.csv"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    weather_cols = ["temp", "dwpt", "rhum", "wspd", "pres"]
    hourly = df[weather_cols].resample("h").mean()
    print(f"  -> {len(hourly)} hourly points, {hourly.index.min()} to {hourly.index.max()}")
    return hourly


def fill_weather_gaps_with_climatology(base_index, weather_hourly):
    """For timestamps beyond the weather file's coverage, fill using a monthxhour
    climatology built from the real weather data (not invented values)."""
    clim = weather_hourly.copy()
    clim["month"] = clim.index.month
    clim["hour"] = clim.index.hour
    climatology = clim.groupby(["month", "hour"]).mean(numeric_only=True)

    full = pd.DataFrame(index=base_index)
    full = full.join(weather_hourly)
    missing_mask = full["temp"].isna()
    n_missing = missing_mask.sum()
    if n_missing > 0:
        print(f"  Filling {n_missing} hours of weather via monthxhour climatology "
              f"(no direct observation available for this period)...")
        for ts in full.index[missing_mask]:
            key = (ts.month, ts.hour)
            if key in climatology.index:
                full.loc[ts, ["temp", "dwpt", "rhum", "wspd", "pres"]] = \
                    climatology.loc[key, ["temp", "dwpt", "rhum", "wspd", "pres"]].values
    full = full.ffill().bfill()  # catch any climatology gaps (rare hour/month combos)
    return full


def solar_multiplier(hour):
    if hour < 6 or hour > 18:
        return 0.0
    return max(0.0, np.exp(-((hour - 13) ** 2) / 12))


def build_zone_frame(base_df):
    """Split citywide demand into 5 zones, add all engineered features."""
    rows_per_hour = []
    base_df = base_df.copy()
    base_df["hour"] = base_df.index.hour
    base_df["dow"] = base_df.index.dayofweek
    base_df["date_str"] = base_df.index.strftime("%Y-%m-%d")

    print("Splitting into 5 zones and computing zone-level features...")
    weights_by_hour = {
        h: {z: zone_hour_weight(z, h) for z in ZONES} for h in range(24)
    }
    for h in range(24):
        total_w = sum(weights_by_hour[h].values())
        weights_by_hour[h] = {z: w / total_w for z, w in weights_by_hour[h].items()}

    rng = np.random.default_rng(11)

    for ts, row in base_df.iterrows():
        hour = row["hour"]
        dow = row["dow"]
        is_weekend = int(dow >= 5)
        is_peak_hour = int(hour in PEAK_HOURS)
        is_festival = int(row["date_str"] in FESTIVAL_DATES)
        total_demand = row["citywide_demand_MW"]
        temp_c = row["temp"]
        rhum = row["rhum"]
        cloud_proxy = np.clip(rhum / 100.0, 0, 1)  # humidity as a real-data cloud proxy

        base_weights = weights_by_hour[hour]
        noise = rng.normal(1.0, 0.03, size=len(ZONES))
        raw_shares = np.array([base_weights[z] for z in ZONES]) * noise
        raw_shares = raw_shares / raw_shares.sum()   # renormalize -> zones sum exactly to total

        for z, share in zip(ZONES, raw_shares):
            demand_mw = total_demand * share
            solar_gen = ZONE_SOLAR_CAPACITY_MW[z] * solar_multiplier(hour) * (1 - 0.7 * cloud_proxy)

            priority = ZONE_BASE_PRIORITY[z]
            if is_festival and z in FESTIVAL_BOOST_ZONES:
                priority *= FESTIVAL_BOOST_FACTOR

            rows_per_hour.append({
                "timestamp": ts, "zone": z,
                "actual_demand_MW": round(demand_mw, 2),
                "temperature": round(temp_c, 1) if pd.notna(temp_c) else None,
                "humidity": round(rhum, 1) if pd.notna(rhum) else None,
                "hour": hour, "day_of_week": int(dow),
                "is_weekend": is_weekend, "is_peak_hour": is_peak_hour,
                "is_festival": is_festival,
                "solar_generation_MW": round(solar_gen, 2),
                "local_generation_MW": 0.0,   # filled in next pass (needs zone's own percentile)
                "priority": round(priority, 3),
            })

    zdf = pd.DataFrame(rows_per_hour)
    print(f"  -> {len(zdf)} zone-hour rows across {len(ZONES)} zones")
    return zdf


def add_capacity_and_storage_columns(zdf):
    print("Computing per-zone capacity limits, battery simulation, local generation...")
    zdf = zdf.sort_values(["zone", "timestamp"]).reset_index(drop=True)

    for z in ZONES:
        mask = zdf["zone"] == z
        zone_demand = zdf.loc[mask, "actual_demand_MW"]
        p95 = zone_demand.quantile(0.95)
        p90 = zone_demand.quantile(0.90)

        tx_capacity = round(1.35 * p95, 2)
        zdf.loc[mask, "maximum_supply_MW"] = tx_capacity
        zdf.loc[mask, "minimum_required_MW"] = round(FAIRNESS_PCT, 2) * zdf.loc[mask, "actual_demand_MW"]

        # local generation: kicks in only when demand exceeds this zone's own p90
        over_p90 = (zone_demand > p90).astype(float)
        utilization = np.clip((zone_demand - p90) / (zone_demand.max() - p90 + 1e-6), 0, 1)
        zdf.loc[mask, "local_generation_MW"] = round(ZONE_LOCAL_GEN_CAPACITY_MW[z], 3) * (over_p90 * utilization)

        # simple battery simulation: charge from surplus solar in daytime, discharge in evening peak
        battery_capacity = ZONE_BATTERY_CAPACITY_MWH[z]
        max_rate = ZONE_BATTERY_MAX_RATE_MW[z]
        soc = battery_capacity * 0.5   # start half-charged
        soc_series = []
        idx = zdf.loc[mask].index
        for i in idx:
            hour = zdf.at[i, "hour"]
            solar = zdf.at[i, "solar_generation_MW"]
            demand = zdf.at[i, "actual_demand_MW"]
            if 10 <= hour <= 15 and solar > 0.3 * demand and soc < battery_capacity:
                soc = min(battery_capacity, soc + max_rate)
            elif hour in PEAK_HOURS and soc > 0:
                soc = max(0.0, soc - max_rate)
            soc_series.append(round(soc, 3))
        zdf.loc[idx, "battery_soc_MWh"] = soc_series
        zdf.loc[mask, "battery_capacity_MWh"] = battery_capacity
        zdf.loc[mask, "max_charge_MW"] = max_rate
        zdf.loc[mask, "max_discharge_MW"] = max_rate

    return zdf


def add_transmission_links_and_scenarios(zdf):
    print("Adding hub-and-spoke transmission link limits and scenario tags...")
    # hub-and-spoke topology: each ring zone connects to Central. Link capacity is
    # a static base value derived from that zone's own transmission capacity.
    link_cols = {
        "North": "north_central_limit_MW", "South": "south_central_limit_MW",
        "East": "east_central_limit_MW", "West": "west_central_limit_MW",
    }
    base_link_capacity = {}
    for z, col in link_cols.items():
        base_link_capacity[z] = zdf.loc[zdf["zone"] == z, "maximum_supply_MW"].iloc[0] * 0.9

    timestamps = zdf["timestamp"].unique()
    rng = np.random.default_rng(23)
    # inject N-1 transmission failure events on a small % of timestamps (demo/test hook)
    n1_event_timestamps = set(
        pd.Series(timestamps).sample(frac=0.003, random_state=23).tolist()
    )
    failed_link_by_ts = {ts: rng.choice(list(link_cols.keys())) for ts in n1_event_timestamps}

    for z, col in link_cols.items():
        zdf[col] = base_link_capacity[z]

    for ts, failed_zone in failed_link_by_ts.items():
        col = link_cols[failed_zone]
        zdf.loc[zdf["timestamp"] == ts, col] = round(base_link_capacity[failed_zone] * 0.05, 2)

    # scenario tagging, priority order = first match wins
    def tag_scenario(row):
        ts = row["timestamp"]
        if ts in failed_link_by_ts:
            return "TRANSMISSION_FAILURE_N-1"
        if row["is_festival"]:
            return "FESTIVAL_SURGE"
        return None  # filled in second pass once we have per-zone p90 thresholds

    zdf["scenario"] = zdf.apply(tag_scenario, axis=1)

    for z in ZONES:
        mask = zdf["zone"] == z
        p90 = zdf.loc[mask, "actual_demand_MW"].quantile(0.90)
        remaining = zdf.loc[mask & zdf["scenario"].isna()]
        idx = remaining.index

        is_extreme = zdf.loc[idx, "actual_demand_MW"] > p90
        is_low_solar = (zdf.loc[idx, "solar_generation_MW"] < 0.2 * ZONE_SOLAR_CAPACITY_MW[z]) & \
                        (zdf.loc[idx, "hour"].between(8, 17))
        is_summer_peak = zdf.loc[idx, "timestamp"].dt.month.isin([4, 5, 6]) & \
                          (zdf.loc[idx, "temperature"] > 38) & zdf.loc[idx, "is_peak_hour"].astype(bool)
        is_evening_peak = zdf.loc[idx, "hour"].between(18, 21)

        scenario_vals = np.select(
            [is_extreme, is_summer_peak, is_low_solar, is_evening_peak],
            ["EXTREME_DEMAND_P90", "SUMMER_PEAK", "LOW_SOLAR", "EVENING_PEAK"],
            default="NORMAL",
        )
        zdf.loc[idx, "scenario"] = scenario_vals

    return zdf


def train_quantile_forecaster_and_predict(zdf):
    print("Training quantile regression forecaster on the assembled dataset...")
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder

    zdf = zdf.copy()
    le = LabelEncoder()
    zdf["zone_enc"] = le.fit_transform(zdf["zone"])
    features = ["hour", "day_of_week", "is_weekend", "is_festival",
                "temperature", "humidity", "zone_enc"]
    zdf[features] = zdf[features].fillna(zdf[features].median(numeric_only=True))

    X = zdf[features]
    y = zdf["actual_demand_MW"]

    preds = {}
    for q, name in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        model = GradientBoostingRegressor(loss="quantile", alpha=q,
                                           n_estimators=120, max_depth=3, learning_rate=0.08)
        model.fit(X, y)
        preds[name] = model.predict(X)
        print(f"  trained quantile={q}")

    preds_df = pd.DataFrame(preds)
    preds_df[["p10", "p50", "p90"]] = np.sort(preds_df[["p10", "p50", "p90"]].values, axis=1)
    zdf["p10_demand_MW"] = preds_df["p10"].round(2)
    zdf["p50_demand_MW"] = preds_df["p50"].round(2)
    zdf["p90_demand_MW"] = preds_df["p90"].round(2)
    zdf["predicted_demand_MW"] = zdf["p50_demand_MW"]
    return zdf


def finalize_columns(zdf):
    col_order = [
        "timestamp", "zone", "actual_demand_MW", "predicted_demand_MW",
        "p10_demand_MW", "p50_demand_MW", "p90_demand_MW",
        "temperature", "humidity", "hour", "day_of_week", "is_weekend", "is_peak_hour",
        "solar_generation_MW",
        "battery_soc_MWh", "battery_capacity_MWh", "max_charge_MW", "max_discharge_MW",
        "local_generation_MW",
        "priority", "minimum_required_MW", "maximum_supply_MW",
        "north_central_limit_MW", "south_central_limit_MW",
        "east_central_limit_MW", "west_central_limit_MW",
        "scenario",
    ]
    return zdf[col_order]


def main():
    base_demand = load_and_prepare_base_demand()
    weather = load_and_prepare_weather()
    weather_filled = fill_weather_gaps_with_climatology(base_demand.index, weather)
    base = base_demand.join(weather_filled)

    zdf = build_zone_frame(base)
    zdf = add_capacity_and_storage_columns(zdf)
    zdf = add_transmission_links_and_scenarios(zdf)
    zdf = train_quantile_forecaster_and_predict(zdf)
    zdf = finalize_columns(zdf)

    out_path = os.path.join(OUT_DIR, "custom_grid_dataset.csv")
    zdf.to_csv(out_path, index=False)
    print(f"\nSaved final dataset -> {out_path}  ({len(zdf)} rows)")

    config = {
        "zones": ZONES,
        "zone_solar_capacity_mw": ZONE_SOLAR_CAPACITY_MW,
        "zone_battery_capacity_mwh": ZONE_BATTERY_CAPACITY_MWH,
        "zone_battery_max_rate_mw": ZONE_BATTERY_MAX_RATE_MW,
        "zone_local_gen_capacity_mw": ZONE_LOCAL_GEN_CAPACITY_MW,
        "zone_base_priority": ZONE_BASE_PRIORITY,
        "festival_boost_zones": list(FESTIVAL_BOOST_ZONES),
        "festival_dates": sorted(FESTIVAL_DATES),
        "fairness_pct": FAIRNESS_PCT,
        "scenario_categories": sorted(zdf["scenario"].unique().tolist()),
    }
    config_path = os.path.join(OUT_DIR, "custom_grid_dataset_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config -> {config_path}")

    print("\nScenario distribution:")
    print(zdf["scenario"].value_counts())
    print(f"\nDate range: {zdf['timestamp'].min()} to {zdf['timestamp'].max()}")
    print(f"Zones: {zdf['zone'].unique().tolist()}")
    print(f"\nSum check (zones should sum back to real citywide demand at a sample hour):")
    sample_ts = zdf["timestamp"].iloc[1000]
    sample = zdf[zdf["timestamp"] == sample_ts]
    print(f"  {sample_ts}: zone sum = {sample['actual_demand_MW'].sum():.2f} MW")


if __name__ == "__main__":
    main()
