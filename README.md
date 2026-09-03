# NEXGRID - AI-Based Electricity Demand & Peak-Demand Forecasting (Delhi)
**WINS-AID AI Hackathon 2026 — Problem 5, Innovator Track**

## Problem
Delhi's grid reacts to overload/anomalies instead of anticipating them. As rooftop
solar and renewables grow, forecasting demand, storing energy, and distributing
power intelligently during peak hours is critical to preventing outages and waste.

## Our approach: Forecast → Optimize → Adapt (closed loop)
Existing literature (see `docs/lit_review.md`) shows most AI-for-grid work treats
forecasting, dispatch, and demand response as **loosely coupled** — forecast errors
aren't corrected by scheduling. We instead build one pipeline where the optimizer
directly consumes forecast uncertainty, not just a point estimate.

```
Historical load + weather + solar
        ↓
[Quantile Regression Forecaster] → P10 / P50 / P90 demand per zone per hour
        ↓
[LP Allocation Engine] → maximizes priority-weighted served demand
        subject to: transmission capacity, fairness floor (min 60% per zone),
        soft time-based priority (festivals boost temple/market zones)
        ↓
Live dashboard: map + charts + allocation table + shadow price (scarcity signal)
```

## What's novel here (vs. existing commercial systems like AutoGrid/Uplight, already
deployed on Tata Power Delhi Distribution)
1. **Uncertainty-aware forecasting** — P10/P90 bands feed directly into the optimizer,
   so allocation can be run for best-case, expected, or worst-case demand — not a
   single fragile point forecast.
2. **Market/auction framing, not hardcoded priority rules** — each zone's bid for
   grid power is *computed*, not looked up in a table:
   `bid = priority_weight × (1 + forecast_uncertainty_hedge) × (1 − solar_self_sufficiency_discount)`.
   Zones with more local solar bid less for grid power (they need less of it); zones
   with volatile forecasts bid slightly more to hedge against being caught short.
   The LP then maximizes bid-weighted welfare — this is the actual OR/optimization
   story, not an if-else priority queue.
3. **Multi-source, carbon-aware allocation ("green priority" mode)** — grid supply
   is modeled as multiple sources (Hydro/Gas/Coal, each with its own capacity and
   carbon intensity). A `green_priority` slider trades off welfare against total
   emissions: turn it up and the optimizer visibly shifts allocation toward
   cleaner sources first, leaving dirtier capacity underutilized even though it's
   "free" from a pure-welfare standpoint.
4. **Soft, time-varying priority** — priority weights shift with context (festival
   days boost market/temple zones) rather than a fixed hard-coded priority list.
5. **Fairness floor as a hard constraint, with graceful degradation** — no zone is
   starved below 60% of its forecasted demand even under scarcity. If supply is so
   low that the floor is mathematically infeasible for every zone at once, the
   optimizer automatically relaxes it and falls back to pure bid-value allocation
   rather than failing — a handled edge case, not a bug, and a good talking point
   for judges on "what happens in a real crisis."
6. **Shadow price as market signal** — each grid source's dual value gives an
   implicit "clearing price" for that source's capacity, letting us tell the
   auction/market story without needing to build actual bidding logic.

## Repo structure
```
data/                  synthetic Delhi zone data generator (swap for real DERC/BSES data if available)
src/forecast/model.py  quantile regression forecaster
src/optimize/allocator.py         LP allocation engine v1 (single-source, kept for reference)
src/optimize/market_allocator.py  LP allocation engine v2 -- bids + multi-source + green priority
                                   (this is what app.py actually uses)
src/api/main.py         optional FastAPI wrapper (only needed for a separate React frontend)
app.py                  Streamlit live demo — THIS IS WHAT YOU RUN FOR THE JUDGES
notebooks/              Colab-exported EDA / model training notebook
docs/lit_review.md      papers + patents + gap analysis backing our novelty claims
```

## Real-data mode (recommended over synthetic data if you have it)
`data/build_custom_dataset.py` builds `data/custom_grid_dataset.csv`: real Delhi
demand (2023-04 to 2026-01, from a Delhi SLDC-style 5-min dataset) split into 5
zones (North/South/East/West/Central), with real observed weather merged in
(monthxhour climatology fills the ~13 months the weather source doesn't cover).
Solar/battery/local-generation/priority/transmission-limit layers are simulation
built on top of that real base -- documented as such, not passed off as more real
than it is. Includes forecast quantiles (trained on the real+simulated data) and
scenario tags (NORMAL, SUMMER_PEAK, EVENING_PEAK, FESTIVAL_SURGE, LOW_SOLAR,
EXTREME_DEMAND_P90, TRANSMISSION_FAILURE_N-1 -- the last one randomly zeroes one
zone's link-to-Central capacity on ~0.3% of hours, a ready-made "what if this
link fails during peak" demo hook).

To rebuild it yourself: put your two raw CSVs in `data/raw/` (same filenames as
in the script), then `python data/build_custom_dataset.py`. Tested: output flows
directly into `src/optimize/market_allocator.py` with no changes needed --
verified on an EVENING_PEAK row and a TRANSMISSION_FAILURE_N-1 row.

**Known simplification:** zone split is a calibrated simulation (distinct hourly
share curves per zone, renormalized to sum exactly to the real citywide total at
every hour) -- there's no real zone-level Delhi meter data behind the 5-way split,
only the citywide aggregate is real. Say this plainly if asked; it's still a much
stronger methodology than presenting fully synthetic data as real.

## Last-mile consumer layer (the AutoGrid/Uplight gap)
`src/lastmile/consumer_layer.py` -- a no-smart-meter, SMS/WhatsApp-style DR layer
for small/informal consumers (rooftop solar households, shopkeepers, RWA common
areas) that enterprise DERMS platforms don't economically reach. Consumers are
ranked by (reliability x flexible load) and messaged in that order; incentive
rate is set directly from the LP's shadow price, so scarcity signals from the
optimizer flow straight into what the SMS offers. Reliability is a running score
that nudges up on response, down on non-response -- the targeting improves over
time, cheaply, without hardware.

## Multi-agent, non-stationary RL (research-gap layer)
`src/rl/multi_agent_dr.py` -- addresses the specific literature gap flagged
earlier: most RL-for-DR papers use a single agent in a stationary environment.
Here, each zone is an independent Q-learning agent that shares a resource (total
grid supply) with the others -- genuinely multi-agent, not N parallel single-agent
problems. The environment is non-stationary: a seasonal drift plus a hard regime
shift (a heatwave that permanently cuts available supply) partway through
training. Run `python src/rl/multi_agent_dr.py` -- it trains, measures how fast
agents find their NEW stable equilibrium after the shock (not a return to the old
one, which is structurally impossible once supply is permanently reduced), and
saves a learning-curve plot. This is a stretch/research add-on, not required for
the core live demo -- cite the adaptation numbers in your pitch as evidence you
went beyond the single-agent/stationary baseline the literature review flagged.
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python data/generate_synthetic_data.py   # generates data/delhi_synthetic_load.csv
python src/forecast/model.py             # trains + saves the forecaster
streamlit run app.py                     
```

## Known simplifications (be upfront about these to judges — it reads as maturity,
not weakness)
- Transmission constraints are capacity limits, not a full AC/DC power-flow model.
- No battery storage charge/discharge scheduling yet.
- No ramp-rate or N-1 contingency check yet.
- Synthetic data, not real DERC/BSES feeds (structure is ready to swap in real data).
- Source capacities (Hydro/Gas/Coal split) are illustrative fixed fractions of
  total grid supply, not real-time generator dispatch data.

These are explicitly listed as "Phase 2" in our pitch, not hidden.

## Team
Bhavyasri Kurapati - 25BCE0298
Nallapaneni Keerthi Sri - 25BCE0422
