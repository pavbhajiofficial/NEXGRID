"""
Optimization layer: given forecasted demand (P50, or a chosen scenario)
and total available supply, decide how much power each zone gets.

Framing: this is a constrained welfare-maximization LP. The dual value
(shadow price) on each zone's demand-satisfaction constraint doubles as
an implicit "market clearing price" -- this is what lets you tell the
auction/market story in your pitch without building real bidding logic.

Constraints modeled:
  - per-zone transmission capacity (feeder headroom)
  - total available supply (citywide generation constraint, incl. deficit scenarios)
  - fairness floor: every zone must receive >= MIN_FAIRNESS_PCT of its demand
  - soft priority weight: festival days boost priority zones (e.g. temples/markets)

NOT modeled (documented as explicit future work, don't pretend otherwise):
  - full AC/DC power flow, voltage drop, ramp rates, N-1 contingency,
    battery charge/discharge scheduling
"""
import pulp
import pandas as pd

MIN_FAIRNESS_PCT = 0.6  # no zone drops below 60% of its forecasted demand, even in scarcity

# Optional: static priority weights per zone (tune to your narrative).
# 1.0 = normal, >1.0 = higher priority (e.g. hospital-dense zone).
DEFAULT_PRIORITY = {
    "Rohini": 1.0, "Dwarka": 1.0, "Connaught_Place": 1.2,  # govt/commercial hub
    "Karol_Bagh": 1.0, "Saket": 1.0, "Shahdara": 1.0,
}

FESTIVAL_BOOST_ZONES = {"Connaught_Place", "Karol_Bagh"}  # markets/temples -> demo talking point
FESTIVAL_BOOST_FACTOR = 1.3


def allocate(
    demand_by_zone: dict,       # {zone: forecast_mw} -- pass P50, or P90 for "worst case" run
    total_available_mw: float,  # citywide supply for this hour (generation + solar + storage)
    tx_capacity: dict,          # {zone: feeder_capacity_mw}
    is_festival: bool = False,
    priority: dict = None,
):
    priority = priority or DEFAULT_PRIORITY
    zones = list(demand_by_zone.keys())

    weight = {}
    for z in zones:
        w = priority.get(z, 1.0)
        if is_festival and z in FESTIVAL_BOOST_ZONES:
            w *= FESTIVAL_BOOST_FACTOR
        weight[z] = w

    prob = pulp.LpProblem("peak_allocation", pulp.LpMaximize)

    # decision variable: MW actually served to each zone
    served = {z: pulp.LpVariable(f"served_{z}", lowBound=0) for z in zones}

    # objective: maximize priority-weighted served demand
    prob += pulp.lpSum(weight[z] * served[z] for z in zones)

    # constraint: can't serve more than forecasted demand or feeder capacity
    fairness_constraints = {}
    for z in zones:
        cap = min(demand_by_zone[z], tx_capacity.get(z, demand_by_zone[z]))
        prob += served[z] <= cap, f"cap_{z}"
        fairness_floor = MIN_FAIRNESS_PCT * demand_by_zone[z]
        c = served[z] >= fairness_floor
        prob += c, f"fair_{z}"
        fairness_constraints[z] = c

    # constraint: total served can't exceed available citywide supply
    supply_constraint = pulp.lpSum(served[z] for z in zones) <= total_available_mw
    prob += supply_constraint, "supply_limit"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    result = {
        "status": pulp.LpStatus[prob.status],
        "allocation": {z: round(served[z].value(), 2) for z in zones},
        "demand": demand_by_zone,
        "shortfall_pct": {
            z: round(100 * (1 - served[z].value() / demand_by_zone[z]), 1)
            if demand_by_zone[z] > 0 else 0
            for z in zones
        },
        # shadow price on supply constraint ~ "clearing price" of scarce power this hour
        "shadow_price_supply": prob.constraints["supply_limit"].pi,
        "weights_used": weight,
    }
    return result


if __name__ == "__main__":
    demo_demand = {"Rohini": 210, "Dwarka": 195, "Connaught_Place": 260,
                    "Karol_Bagh": 165, "Saket": 175, "Shahdara": 200}
    demo_tx = {"Rohini": 243, "Dwarka": 216, "Connaught_Place": 297,
               "Karol_Bagh": 189, "Saket": 202, "Shahdara": 229}
    res = allocate(demo_demand, total_available_mw=950, tx_capacity=demo_tx, is_festival=True)
    import json
    print(json.dumps(res, indent=2))
