"""
Market/auction-framed allocation engine, v2.

Two upgrades over src/optimize/allocator.py:

1. BID-BASED FRAMING (not hardcoded priority rules)
   Each zone's "bid" for grid power is computed as:
       bid[z] = f(priority_weight, forecast_uncertainty, solar_self_sufficiency)
   - higher priority weight -> higher bid
   - higher forecast uncertainty (P90-P50 gap) -> zone bids a bit more to hedge
     against being caught short
   - more local solar covering its own demand -> zone bids a bit less for grid
     power, since it needs less of it
   This replaces "if priority == high: allocate first" with a computed valuation,
   which is what actually earns the "auction" framing in your pitch.

2. CARBON / RENEWABLE PRIORITY MODE
   Grid supply is modeled as multiple sources (Hydro, Gas, Coal by default) with
   different capacities and carbon intensities (kg CO2/MWh). The optimizer chooses
   not just HOW MUCH to serve each zone, but WHICH source mix to draw from. A
   `green_priority` weight lets you trade off welfare (bid satisfaction) against
   total emissions -- turn it up and the LP visibly shifts allocation toward
   cleaner sources first, even if that means slightly lower total welfare.

Still uses PuLP underneath -- same solver, richer decision variables.
"""
import pulp

MIN_FAIRNESS_PCT_DEFAULT = 0.6

# Default grid-source mix: name -> (capacity_fraction_of_total, carbon_intensity_kg_per_mwh)
# Fractions are of whatever `total_grid_mw` you pass in. Real intensities are in the
# right ballpark for India's grid mix (hydro very clean, gas moderate, coal dirty).
DEFAULT_SOURCES = {
    "Hydro": {"capacity_fraction": 0.15, "carbon_intensity": 20},
    "Gas":   {"capacity_fraction": 0.35, "carbon_intensity": 400},
    "Coal":  {"capacity_fraction": 0.50, "carbon_intensity": 900},
}

DEFAULT_PRIORITY = {
    "Rohini": 1.0, "Dwarka": 1.0, "Connaught_Place": 1.2,
    "Karol_Bagh": 1.0, "Saket": 1.0, "Shahdara": 1.0,
}
FESTIVAL_BOOST_ZONES = {"Connaught_Place", "Karol_Bagh"}
FESTIVAL_BOOST_FACTOR = 1.3


def compute_bids(
    demand_by_zone: dict,
    solar_gen_by_zone: dict,
    priority: dict = None,
    uncertainty_by_zone: dict = None,   # {zone: (p90 - p50)} in MW, optional
    is_festival: bool = False,
    solar_discount: float = 0.3,        # how much local solar reduces a zone's bid
    uncertainty_weight: float = 0.5,    # how much forecast uncertainty raises a zone's bid
):
    """bid[z] = priority_weight * (1 + uncertainty_bonus) * (1 - solar_discount_applied)"""
    priority = priority or DEFAULT_PRIORITY
    uncertainty_by_zone = uncertainty_by_zone or {}
    bids = {}
    for z, demand in demand_by_zone.items():
        w = priority.get(z, 1.0)
        if is_festival and z in FESTIVAL_BOOST_ZONES:
            w *= FESTIVAL_BOOST_FACTOR

        # uncertainty bonus: zones with volatile forecasts bid a bit more to hedge
        unc = uncertainty_by_zone.get(z, 0.0)
        uncertainty_bonus = uncertainty_weight * (unc / demand) if demand > 0 else 0.0

        # solar discount: zones already covering more of their own demand bid less
        # for grid power (they need less of it)
        solar_ratio = min(1.0, solar_gen_by_zone.get(z, 0.0) / demand) if demand > 0 else 0.0
        solar_factor = 1 - solar_discount * solar_ratio

        bids[z] = round(w * (1 + uncertainty_bonus) * solar_factor, 4)
    return bids


def allocate_market(
    demand_by_zone: dict,
    solar_gen_by_zone: dict,
    tx_capacity: dict,
    total_grid_mw: float,          # total non-solar grid supply available this hour
    priority: dict = None,
    uncertainty_by_zone: dict = None,
    is_festival: bool = False,
    fairness_pct: float = MIN_FAIRNESS_PCT_DEFAULT,
    green_priority: float = 0.0,   # 0 = ignore carbon, higher = prioritize clean sources more
    sources: dict = None,
):
    """
    Multi-source, bid-weighted LP allocation.

    Zone demand is first netted against its own local solar. The remainder is
    served from a shared pool of grid sources (Hydro/Gas/Coal by default), each
    with its own capacity and carbon intensity. The LP picks both how much each
    zone gets AND which source mix supplies it.

    Objective: maximize (bid-weighted served demand) - green_priority * (emissions),
    both terms scaled to comparable magnitude so the green_priority slider has an
    intuitive 0-2ish range.
    """
    sources = sources or DEFAULT_SOURCES
    priority = priority or DEFAULT_PRIORITY
    zones = list(demand_by_zone.keys())
    source_names = list(sources.keys())

    bids = compute_bids(
        demand_by_zone, solar_gen_by_zone, priority,
        uncertainty_by_zone, is_festival,
    )

    # local solar self-consumption (capped at demand -- can't "use" more than you need)
    solar_used = {
        z: min(solar_gen_by_zone.get(z, 0.0), demand_by_zone[z]) for z in zones
    }
    net_grid_demand = {z: demand_by_zone[z] - solar_used[z] for z in zones}

    source_capacity = {
        s: sources[s]["capacity_fraction"] * total_grid_mw for s in source_names
    }
    carbon_intensity = {s: sources[s]["carbon_intensity"] for s in source_names}
    max_intensity = max(carbon_intensity.values())

    prob = pulp.LpProblem("market_allocation", pulp.LpMaximize)

    # served[z][s] = MW served to zone z from source s
    served = {
        (z, s): pulp.LpVariable(f"served_{z}_{s}", lowBound=0)
        for z in zones for s in source_names
    }

    def zone_total(z):
        return pulp.lpSum(served[(z, s)] for s in source_names)

    welfare = pulp.lpSum(bids[z] * zone_total(z) for z in zones)
    emissions = pulp.lpSum(
        (carbon_intensity[s] / max_intensity) * served[(z, s)]
        for z in zones for s in source_names
    )
    avg_bid = sum(bids.values()) / len(bids)
    prob += welfare - green_priority * avg_bid * emissions

    # per-zone: can't serve more than net grid demand or feeder capacity
    for z in zones:
        cap = min(net_grid_demand[z], max(0.0, tx_capacity.get(z, net_grid_demand[z]) - solar_used[z]))
        prob += zone_total(z) <= cap, f"cap_{z}"
        # fairness floor applies to TOTAL supply (solar + grid), not just grid portion
        floor = fairness_pct * demand_by_zone[z] - solar_used[z]
        if floor > 0:
            prob += zone_total(z) >= floor, f"fair_{z}"

    # per-source: can't exceed that source's capacity across all zones
    for s in source_names:
        prob += pulp.lpSum(served[(z, s)] for z in zones) <= source_capacity[s], f"source_cap_{s}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    fairness_relaxed = False

    if status != "Optimal":
        # Extreme scarcity: total supply can't even meet every zone's fairness
        # floor simultaneously. Rather than return garbage/stale solver output,
        # re-solve WITHOUT the fairness floor so the demo always shows a valid
        # (if harsher) allocation, and flag it clearly so this reads as a
        # handled edge case, not a bug.
        prob = pulp.LpProblem("market_allocation_relaxed", pulp.LpMaximize)
        served = {
            (z, s): pulp.LpVariable(f"served_{z}_{s}", lowBound=0)
            for z in zones for s in source_names
        }

        def zone_total(z):
            return pulp.lpSum(served[(z, s)] for s in source_names)

        welfare = pulp.lpSum(bids[z] * zone_total(z) for z in zones)
        emissions = pulp.lpSum(
            (carbon_intensity[s] / max_intensity) * served[(z, s)]
            for z in zones for s in source_names
        )
        prob += welfare - green_priority * avg_bid * emissions

        for z in zones:
            cap = min(net_grid_demand[z], max(0.0, tx_capacity.get(z, net_grid_demand[z]) - solar_used[z]))
            prob += zone_total(z) <= cap, f"cap_{z}"
        for s in source_names:
            prob += pulp.lpSum(served[(z, s)] for z in zones) <= source_capacity[s], f"source_cap_{s}"

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]
        fairness_relaxed = True

    source_constraints = {
        s: prob.constraints[f"source_cap_{s}"] for s in source_names
    }

    allocation_by_zone = {z: round(zone_total(z).value(), 2) for z in zones}
    source_mix = {
        z: {s: round(served[(z, s)].value(), 2) for s in source_names} for z in zones
    }
    total_served_mw = sum(allocation_by_zone.values()) + sum(solar_used.values())
    total_emissions_kg = sum(
        served[(z, s)].value() * carbon_intensity[s] for z in zones for s in source_names
    )

    return {
        "status": status,
        "fairness_relaxed": fairness_relaxed,
        "bids": bids,
        "solar_used": {z: round(v, 2) for z, v in solar_used.items()},
        "grid_allocation": allocation_by_zone,
        "total_served_mw": round(total_served_mw, 2),
        "source_mix_by_zone": source_mix,
        "source_capacity": {s: round(v, 2) for s, v in source_capacity.items()},
        "source_utilization_pct": {
            s: round(100 * sum(served[(z, s)].value() for z in zones) / source_capacity[s], 1)
            if source_capacity[s] > 0 else 0
            for s in source_names
        },
        "total_emissions_kg": round(total_emissions_kg, 1),
        "shortfall_pct": {
            z: round(100 * (1 - (allocation_by_zone[z] + solar_used[z]) / demand_by_zone[z]), 1)
            if demand_by_zone[z] > 0 else 0
            for z in zones
        },
        "shadow_price_by_source": {
            s: source_constraints[s].pi for s in source_names
        },
    }


if __name__ == "__main__":
    import json

    demo_demand = {"Rohini": 210, "Dwarka": 195, "Connaught_Place": 260,
                    "Karol_Bagh": 165, "Saket": 175, "Shahdara": 200}
    demo_solar = {"Rohini": 15, "Dwarka": 20, "Connaught_Place": 4,
                  "Karol_Bagh": 3, "Saket": 12, "Shahdara": 8}
    demo_tx = {"Rohini": 243, "Dwarka": 216, "Connaught_Place": 297,
               "Karol_Bagh": 189, "Saket": 202, "Shahdara": 229}
    demo_uncertainty = {"Rohini": 18, "Dwarka": 14, "Connaught_Place": 22,
                        "Karol_Bagh": 12, "Saket": 16, "Shahdara": 15}

    print("--- green_priority = 0 (welfare only) ---")
    r1 = allocate_market(demo_demand, demo_solar, demo_tx, total_grid_mw=800,
                          uncertainty_by_zone=demo_uncertainty, is_festival=True,
                          green_priority=0.0)
    print(json.dumps({k: r1[k] for k in ["bids", "total_emissions_kg", "source_utilization_pct"]}, indent=2))

    print("\n--- green_priority = 1.5 (prioritize clean sources) ---")
    r2 = allocate_market(demo_demand, demo_solar, demo_tx, total_grid_mw=800,
                          uncertainty_by_zone=demo_uncertainty, is_festival=True,
                          green_priority=1.5)
    print(json.dumps({k: r2[k] for k in ["total_emissions_kg", "source_utilization_pct"]}, indent=2))
    print(f"\nEmissions dropped from {r1['total_emissions_kg']} kg to {r2['total_emissions_kg']} kg "
          f"by turning up green_priority.")
