"""
Last-mile consumer layer: the piece enterprise DERMS platforms (AutoGrid/Uplight)
admit they don't reach cheaply -- small/informal consumers (rooftop solar
households, shopkeepers, RWAs) who don't have smart meters or the scale to be
worth enrolling via a heavyweight enterprise integration.

Design principle: NO smart meter, NO app install required. Just SMS/WhatsApp-style
text signals + self-reported curtailment + a simple trust/reliability score that
improves the targeting over time. This is intentionally low-tech -- that's the
point, it's what makes it cheap enough to reach the last mile.

Key link to the rest of the system: the incentive rate offered to consumers is
set directly from the LP's shadow price (src/optimize/market_allocator.py). When
a source is scarce, its shadow price rises, incentives rise, and the model sends
more/higher-value signals -- this connects the "enterprise-grade optimizer" story
to the "cheap last-mile mobilization" story as ONE pipeline, not two bolted-together
ideas.
"""
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List

random.seed(7)
np.random.seed(7)

CONSUMER_TYPES = ["rooftop_solar_household", "shopkeeper", "rwa_common_area"]

# Base flexible load each consumer type can plausibly shed (kW), and a rough
# reliability prior (how often they actually respond when messaged) before we've
# observed anything -- refined per-consumer as we simulate responses over time.
TYPE_PROFILE = {
    "rooftop_solar_household": {"flexible_load_kw": (0.5, 2.0), "base_reliability": 0.55},
    "shopkeeper":              {"flexible_load_kw": (0.3, 1.5), "base_reliability": 0.45},
    "rwa_common_area":         {"flexible_load_kw": (1.0, 4.0), "base_reliability": 0.65},
}


@dataclass
class Consumer:
    consumer_id: str
    zone: str
    consumer_type: str
    flexible_load_kw: float
    reliability: float          # P(responds to a signal), updated over time
    has_solar: bool
    times_messaged: int = 0
    times_responded: int = 0
    total_incentive_earned: float = 0.0


def generate_lastmile_consumers(zone: str, n: int = 40, seed_offset: int = 0) -> List[Consumer]:
    """Synthetic directory of small/informal consumers for one zone. Swap this for
    a real opt-in registry (even a Google Form + phone number list is enough to
    start -- that's the whole point of "low-cost")."""
    rng = random.Random(hash(zone) + seed_offset)
    consumers = []
    for i in range(n):
        ctype = rng.choice(CONSUMER_TYPES)
        prof = TYPE_PROFILE[ctype]
        load = rng.uniform(*prof["flexible_load_kw"])
        reliability = np.clip(rng.gauss(prof["base_reliability"], 0.1), 0.05, 0.95)
        consumers.append(Consumer(
            consumer_id=f"{zone[:3].upper()}-{i:03d}",
            zone=zone,
            consumer_type=ctype,
            flexible_load_kw=round(load, 2),
            reliability=round(reliability, 3),
            has_solar=(ctype == "rooftop_solar_household"),
        ))
    return consumers


def send_dr_signal(
    consumers: List[Consumer],
    mw_needed: float,
    incentive_rate_per_kwh: float,   # derived from LP shadow price -- see app.py
    max_messages: int = None,
):
    """
    Simulates sending a low-cost text signal to the consumers most likely to help,
    ranked by (reliability x flexible load) -- i.e. message the people most likely
    to actually respond with meaningful load first, don't spam everyone.

    Returns: mw curtailed (simulated), total incentive payout, and an SMS log you
    can display directly in the demo.
    """
    kw_needed = mw_needed * 1000
    ranked = sorted(consumers, key=lambda c: c.reliability * c.flexible_load_kw, reverse=True)
    if max_messages:
        ranked = ranked[:max_messages]

    sms_log = []
    kw_secured = 0.0
    total_payout = 0.0

    for c in ranked:
        if kw_secured >= kw_needed:
            break
        c.times_messaged += 1
        responded = random.random() < c.reliability
        payout = 0.0

        if responded:
            c.times_responded += 1
            kw_secured += c.flexible_load_kw
            payout = c.flexible_load_kw * incentive_rate_per_kwh
            c.total_incentive_earned += payout
            total_payout += payout
            # reliability nudges up slightly on a successful response (trust builds)
            c.reliability = min(0.95, c.reliability + 0.01)
            msg = (f"[SMS to {c.consumer_id}] Peak alert: reduce load now, "
                   f"earn ₹{payout:.1f}. -> RESPONDED (+{c.flexible_load_kw:.2f} kW)")
        else:
            # reliability nudges down slightly on a miss
            c.reliability = max(0.05, c.reliability - 0.005)
            msg = (f"[SMS to {c.consumer_id}] Peak alert: reduce load now, "
                   f"earn up to ₹{c.flexible_load_kw * incentive_rate_per_kwh:.1f}. -> no response")

        sms_log.append(msg)

    return {
        "mw_curtailed": round(kw_secured / 1000, 3),
        "total_incentive_payout": round(total_payout, 2),
        "messages_sent": len(sms_log),
        "sms_log": sms_log,
        "fulfillment_pct": round(100 * min(1.0, kw_secured / kw_needed), 1) if kw_needed > 0 else 100.0,
    }


if __name__ == "__main__":
    consumers = generate_lastmile_consumers("Karol_Bagh", n=30)
    result = send_dr_signal(consumers, mw_needed=0.05, incentive_rate_per_kwh=6.0, max_messages=15)
    print(f"Curtailed: {result['mw_curtailed']} MW | Payout: Rs.{result['total_incentive_payout']} "
          f"| Fulfillment: {result['fulfillment_pct']}%")
    for line in result["sms_log"][:8]:
        print(" ", line)
