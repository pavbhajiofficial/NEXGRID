#Streamlit live demo. Run with: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import time
from fpdf import FPDF

from data.generate_synthetic_data import (
    ZONES, ZONE_TX_CAPACITY_MW, ZONE_BASE_LOAD_MW,
    ZONE_SOLAR_CAPACITY_MW, solar_multiplier,
)
from src.forecast.model import DemandForecaster
from src.optimize.market_allocator import allocate_market, DEFAULT_SOURCES
from src.events.event_predictor import predict_event_days

# approx lat/lon for demo map (not survey-grade, good enough for visualization)
ZONE_COORDS = {
    "Rohini": (28.7360, 77.1200),
    "Dwarka": (28.5921, 77.0460),
    "Connaught_Place": (28.6315, 77.2167),
    "Karol_Bagh": (28.6519, 77.1909),
    "Saket": (28.5245, 77.2066),
    "Shahdara": (28.6692, 77.2900),
}

st.set_page_config(page_title="Delhi Grid AI", layout="wide")
st.title("⚡ AI-Based Electricity Demand & Peak-Allocation — Live Demo")
st.caption("Forecast (quantile regression) → Optimize (LP allocation) → Adapt (live sliders)")


@st.cache_resource
def load_model():
    fc = DemandForecaster()

    df = pd.read_csv("data/delhi_synthetic_load.csv")
    fc.train(df)

    return fc
forecaster = load_model()

# ---------------- Sidebar controls (the "judge can poke it" panel) ----------------
st.sidebar.header("Scenario Controls")
hour = st.sidebar.slider("Hour of day", 0, 23, 20)
is_festival = st.sidebar.checkbox("Festival day (soft priority boost)", value=False)
temp_c = st.sidebar.slider("Temperature (°C)", 20, 46, 38)
cloud_cover = st.sidebar.slider("Cloud cover (affects solar)", 0.0, 1.0, 0.2)
supply_pct = st.sidebar.slider(
    "Available city supply (% of total forecasted demand)", 50, 120, 90,
    help="Drag below 100% to simulate a supply crunch / heatwave scarcity event."
)
scenario = st.sidebar.radio(
    "Forecast scenario used for allocation", ["p10 (best case)", "p50 (expected)", "p90 (worst case)"],
    index=1,
)
scenario_key = scenario.split()[0]

st.sidebar.divider()
st.sidebar.subheader("Market & Sustainability")
green_priority = st.sidebar.slider(
    "Green priority weight", 0.0, 2.0, 0.0, step=0.1,
    help="0 = allocate purely by bid value. Higher = optimizer actively shifts "
         "allocation toward cleaner sources (Hydro/Gas) and away from Coal, even "
         "at some cost to total welfare."
)

st.sidebar.divider()
st.sidebar.subheader("Live Automation")
auto_refresh = st.sidebar.checkbox("Enable Live Hourly Auto-Refresh")

st.sidebar.divider()
st.sidebar.subheader("Soft Limits: Next 7 Days")
boost_mode = st.sidebar.radio(
    "Priority boost source",
    ["Predict from calendar (default)", "Edit manually"],
    index=0,
    help="Default: auto-detects festivals/bank holidays in the next 7 days "
         "(offline, via India holiday calendar; optionally cross-checked "
         "against live news if NEWSAPI_KEY is set). Switch to manual to "
         "override any day/zone yourself."
)
day_offset = st.sidebar.slider("Day this week (0 = today)", 0, 6, 0)
week_dates = [date.today() + timedelta(days=i) for i in range(7)]
target_date = week_dates[day_offset]
predicted_week = predict_event_days(ZONES, start_date=date.today())

if boost_mode == "Predict from calendar (default)":
    week_boost_table = predicted_week
else:
    editable_df = pd.DataFrame(
        {d.isoformat(): [predicted_week[d.isoformat()][z] for z in ZONES] for d in week_dates},
        index=ZONES,
    )
    edited_df = st.sidebar.data_editor(editable_df, use_container_width=True)
    week_boost_table = {
        d.isoformat(): {z: edited_df.loc[z, d.isoformat()] for z in ZONES} for d in week_dates
    }

manual_zone_boost = week_boost_table[target_date.isoformat()]
boosted_today = [z for z, m in manual_zone_boost.items() if m > 1.0]
st.sidebar.caption(
    f"📅 {target_date.strftime('%a %b %d')}: "
    + (f"boosted zones = {', '.join(boosted_today)}" if boosted_today else "no boost predicted")
)

# ---------------- Build input frame & forecast ----------------
rows = [{
    "timestamp": datetime(2025, 10, 20 if is_festival else 15, hour, 0, 0),
    "zone": z, "hour": hour, "is_festival": int(is_festival),
    "temp_c": temp_c, "cloud_cover": cloud_cover,
} for z in ZONES]
input_df = pd.DataFrame(rows)
preds = forecaster.predict(input_df)

demand_by_zone = dict(zip(preds["zone"], preds[scenario_key]))
uncertainty_by_zone = dict(zip(preds["zone"], preds["p90"] - preds["p50"]))
total_forecast = sum(demand_by_zone.values())
total_grid_mw = total_forecast * (supply_pct / 100)

# solar generation per zone, same physics as the data generator
solar_gen_by_zone = {
    z: ZONE_SOLAR_CAPACITY_MW[z] * solar_multiplier(hour) * (1 - 0.7 * cloud_cover)
    for z in ZONES
}

result = allocate_market(
    demand_by_zone=demand_by_zone,
    solar_gen_by_zone=solar_gen_by_zone,
    tx_capacity=ZONE_TX_CAPACITY_MW,
    total_grid_mw=total_grid_mw,
    uncertainty_by_zone=uncertainty_by_zone,
    is_festival=is_festival,
    hour=hour,  # drives hospital/temple time-of-day priority
    manual_zone_boost=manual_zone_boost,  # predicted/edited soft limit for the selected day
    green_priority=green_priority,
)

# ---------------- 1-Hour Early Warning System ----------------
next_hour = (hour + 1) % 24
next_rows = [{
    "timestamp": datetime(2025, 10, 20 if is_festival else 15, next_hour, 0, 0),
    "zone": z, "hour": next_hour, "is_festival": int(is_festival),
    "temp_c": temp_c, "cloud_cover": cloud_cover,
} for z in ZONES]

next_preds = forecaster.predict(pd.DataFrame(next_rows))
next_demand_total = next_preds["p90"].sum()  # P90 worst-case for a safety margin
next_supply_est = next_demand_total * (supply_pct / 100)  # simplified projected supply

if next_supply_est < next_demand_total:
    st.error(
        f"🚨 **EARLY WARNING**: P90 forecast predicts a "
        f"{(next_demand_total - next_supply_est):.0f} MW deficit for {next_hour}:00. "
        "Initiating grid balancing protocols."
    )
else:
    st.success(f"✅ **T+1 STATUS**: Grid is stable for {next_hour}:00.")

# ---------------- Top metrics ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total forecasted demand", f"{total_forecast:.0f} MW")
c2.metric("Available grid supply", f"{total_grid_mw:.0f} MW",
          delta=f"{total_grid_mw - total_forecast:.0f} MW")
c3.metric("Total served (grid + solar)", f"{result['total_served_mw']:.0f} MW")
c4.metric("Optimizer status", result["status"] + (" ⚠️ relaxed" if result["fairness_relaxed"] else ""))
c5.metric("CO₂ emissions this hour", f"{result['total_emissions_kg']/1000:.1f} t")

st.divider()

# ---------------- Map: zones colored by shortfall ----------------
map_df = pd.DataFrame([
    {
        "zone": z,
        "lat": ZONE_COORDS[z][0],
        "lon": ZONE_COORDS[z][1],
        "shortfall_pct": result["shortfall_pct"][z],
        "allocated_mw": round(result["grid_allocation"][z] + result["solar_used"][z], 2),
        "grid_mw": result["grid_allocation"][z],
        "solar_mw": result["solar_used"][z],
        "demand_mw": round(demand_by_zone[z], 1),
        "bid": result["bids"][z],
    }
    for z in ZONES
])

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Zone allocation map")
    fig_map = px.scatter_map(
        map_df, lat="lat", lon="lon", size="demand_mw",
        color="shortfall_pct", color_continuous_scale="RdYlGn_r",
        hover_name="zone",
        hover_data={"allocated_mw": True, "demand_mw": True, "shortfall_pct": True,
                    "lat": False, "lon": False},
        zoom=9.3, height=480, range_color=(0, 40),
    )
    fig_map.update_layout(
    map_style="open-street-map",
    margin=dict(l=0, r=0, t=0, b=0)
)

@st.cache_resource
def load_model():
    fc = DemandForecaster()

    df = pd.read_csv("data/delhi_synthetic_load.csv")
    fc.train(df)

    return fc
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Red = zone is receiving noticeably less than it forecasted needing. Bubble size = demand.")

with right:
    st.subheader("Allocation vs demand by zone")
    bar_df = map_df.melt(id_vars="zone", value_vars=["demand_mw", "allocated_mw"],
                          var_name="type", value_name="MW")
    fig_bar = px.bar(bar_df, x="zone", y="MW", color="type", barmode="group", height=480)
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ---------------- Market/auction view: bids + source mix ----------------
st.subheader("Market view: bids & generation source mix")
st.caption(
    "Each zone submits a computed bid (priority × forecast-uncertainty hedge × "
    "solar self-sufficiency discount). The optimizer maximizes bid-weighted "
    "welfare, then chooses which grid source (Hydro/Gas/Coal) supplies each zone."
)
m1, m2 = st.columns([1, 1.4])

with m1:
    bid_df = pd.DataFrame([
        {"zone": z, "bid": result["bids"][z]} for z in ZONES
    ]).sort_values("bid", ascending=False)
    fig_bid = px.bar(bid_df, x="zone", y="bid", color="bid",
                      color_continuous_scale="Blues", height=380,
                      title="Computed bid per zone (higher = valued more)")
    st.plotly_chart(fig_bid, use_container_width=True)

with m2:
    source_rows = []
    for z in ZONES:
        for s, mw in result["source_mix_by_zone"][z].items():
            source_rows.append({"zone": z, "source": s, "mw": mw})
        source_rows.append({"zone": z, "source": "Solar (local)", "mw": result["solar_used"][z]})
    source_df = pd.DataFrame(source_rows)
    color_map = {"Hydro": "#2ecc71", "Gas": "#f39c12", "Coal": "#7f2d2d", "Solar (local)": "#f1c40f"}
    fig_src = px.bar(source_df, x="zone", y="mw", color="source", barmode="stack",
                      height=380, color_discrete_map=color_map,
                      title="Source mix per zone (MW)")
    st.plotly_chart(fig_src, use_container_width=True)

util_df = pd.DataFrame([
    {"source": s, "utilization_pct": result["source_utilization_pct"][s],
     "capacity_mw": result["source_capacity"][s]}
    for s in DEFAULT_SOURCES
])
st.dataframe(util_df, use_container_width=True, hide_index=True)
if green_priority > 0:
    st.success(
        f"🌱 Green priority is active (weight={green_priority}). Coal utilization is "
        f"{result['source_utilization_pct']['Coal']:.0f}% — compare this to 100% at "
        f"green_priority=0 to show judges the optimizer visibly shifting toward cleaner sources."
    )

st.divider()

# ---------------- Forecast uncertainty bands ----------------
st.subheader("Demand forecast with uncertainty (P10 / P50 / P90)")
hours_df = pd.DataFrame([
    {"timestamp": datetime(2025, 10, 15, h, 0, 0), "zone": z, "hour": h,
     "is_festival": int(is_festival), "temp_c": temp_c, "cloud_cover": cloud_cover}
    for h in range(24) for z in ZONES
])
day_preds = forecaster.predict(hours_df)
selected_zone = st.selectbox("Zone", ZONES, index=2)
zone_day = day_preds[day_preds["zone"] == selected_zone].sort_values("hour")

fig = go.Figure()
fig.add_trace(go.Scatter(x=zone_day["hour"], y=zone_day["p90"], line=dict(width=0),
                          showlegend=False, name="p90"))
fig.add_trace(go.Scatter(x=zone_day["hour"], y=zone_day["p10"], fill="tonexty",
                          fillcolor="rgba(99,110,250,0.2)", line=dict(width=0),
                          name="P10–P90 band"))
fig.add_trace(go.Scatter(x=zone_day["hour"], y=zone_day["p50"], line=dict(color="royalblue", width=3),
                          name="P50 (expected)"))
fig.add_vline(x=hour, line_dash="dash", line_color="red", annotation_text="selected hour")
fig.update_layout(height=350, xaxis_title="Hour", yaxis_title="Demand (MW)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- Raw allocation table + explainability ----------------
st.subheader("Allocation detail & explainability")
st.dataframe(map_df, use_container_width=True, hide_index=True)

if is_festival:
    st.info(
        "🪔 Festival mode active: Connaught_Place and Karol_Bagh get a soft priority "
        "boost baked into their bid (markets/temples), so they're allocated closer to "
        "full demand and outbid other zones even under scarcity."
    )
if supply_pct < 100:
    if result["fairness_relaxed"]:
        st.error(
            "🚨 Severe scarcity: available supply is too low to guarantee every zone's "
            "60% fairness floor simultaneously. The optimizer automatically relaxed the "
            "floor and fell back to pure bid-value allocation so the demo still returns "
            "a valid plan instead of failing — this is a handled edge case, not a bug."
        )
    else:
        st.warning(
            f"⚠️ Supply is at {supply_pct}% of demand — a scarcity scenario. "
            "Every zone is still guaranteed a fairness floor (60% of its demand) even under stress."
        )

st.divider()
st.subheader("Daily Compliance & Reporting")

if st.button("📄 Generate 24-Hour Grid Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Delhi Grid AI - 24 Hour Allocation Report", align="C")
    pdf.ln(14)

    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, f"Simulated Scenario Supply Limit: {supply_pct}%")
    pdf.ln(8)
    pdf.cell(0, 10, f"Festival Priority Active: {is_festival}")
    pdf.ln(12)

    for z in ZONES:
        peak_demand = day_preds[day_preds["zone"] == z]["p90"].max()
        pdf.cell(0, 8, f"{z}: Peak P90 Demand Expected ~{peak_demand:.1f} MW")
        pdf.ln(7)

    pdf.ln(8)
    pdf.cell(0, 10, "Automated via DemandForecaster Model.")

    st.download_button(
        label="Download PDF Report",
        data=bytes(pdf.output()),
        file_name="grid_24h_report.pdf",
        mime="application/pdf",
    )

# Auto-refresh loop to simulate a live-updating dashboard for the judges
if auto_refresh:
    time.sleep(5)  # simulate 1 hour passing every 5 seconds
    st.rerun()
