"""
Predicts soft priority-boost multipliers per zone for the next 7 days.

DEFAULT (offline, always works): uses the `holidays` package's India calendar
to detect festivals/bank holidays landing in the next 7 days. This is the "bank
calendar" data source -- it needs zero network access and zero API keys, so the
demo never breaks if you're offline during judging.

OPTIONAL (online, best-effort): if a NEWSAPI_KEY environment variable is set,
also queries NewsAPI for Delhi event/festival/VIP-visit headlines to catch
things a fixed calendar can't know about (a cricket final, a state visit, a
protest). Wrapped in try/except -- if there's no key, no internet, or the
request fails for any reason, this step is silently skipped and you fall back
to the calendar-only prediction. Never a hard dependency.

Either source only decides WHICH days over the next week look like event days.
The actual multiplier applied to a zone's bid still lives in ZONE_PROFILES /
compute_bids() in market_allocator.py -- this module is a prediction layer that
feeds a boost table into that existing logic, not a replacement for it.
"""
import os
import datetime as dt

try:
    import holidays as holidays_lib
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

EVENT_DAY_MULTIPLIER = 1.5   # applied uniformly to all zones on a predicted event day;
                              # the user can narrow this down to specific zones in the
                              # Streamlit editor if only some zones should be boosted


def predict_event_days(zones, start_date=None, days_ahead=7):
    """
    Returns: {date_iso_string: {zone: multiplier}} for the next `days_ahead` days.
    multiplier = EVENT_DAY_MULTIPLIER on a predicted event day, 1.0 otherwise.
    This dict is a DEFAULT/starting point -- the app lets the user override it
    day-by-day and zone-by-zone before it's used in allocation.
    """
    start_date = start_date or dt.date.today()
    dates = [start_date + dt.timedelta(days=i) for i in range(days_ahead)]

    event_dates = set()
    if HOLIDAYS_AVAILABLE:
        years_needed = {d.year for d in dates}
        india_holidays = holidays_lib.India(years=years_needed)
        for d in dates:
            if d in india_holidays:
                event_dates.add(d)

    event_dates |= _fetch_news_event_days(dates)

    result = {}
    for d in dates:
        is_event_day = d in event_dates
        result[d.isoformat()] = {
            z: (EVENT_DAY_MULTIPLIER if is_event_day else 1.0) for z in zones
        }
    return result


def _fetch_news_event_days(dates):
    """
    Best-effort live news check. Requires NEWSAPI_KEY env var + outbound
    internet access to newsapi.org (not available in every deployment
    environment -- e.g. sandboxed dev containers often block this). Silently
    returns an empty set on any failure so this is always a bonus signal on
    top of the offline holiday calendar, never a hard dependency.

    To enable: set NEWSAPI_KEY in your environment before running the app,
    e.g. `export NEWSAPI_KEY=your_key_here` (get a free key at newsapi.org).
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return set()
    try:
        import requests
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "Delhi (festival OR event OR \"VIP visit\" OR mela OR protest)",
                "from": dates[0].isoformat(), "to": dates[-1].isoformat(),
                "language": "en", "sortBy": "relevancy", "apiKey": api_key,
            },
            timeout=5,
        )
        articles = resp.json().get("articles", [])
        found = set()
        for a in articles:
            published = a.get("publishedAt", "")[:10]
            try:
                d = dt.date.fromisoformat(published)
                if d in dates:
                    found.add(d)
            except ValueError:
                continue
        return found
    except Exception:
        # No key, no network, API down, rate-limited, malformed response --
        # any of these just means "no news signal today", not a crash.
        return set()


if __name__ == "__main__":
    zones = ["Rohini", "Dwarka", "Connaught_Place", "Karol_Bagh", "Saket", "Shahdara"]
    preds = predict_event_days(zones, start_date=dt.date(2025, 10, 18))
    print(f"holidays package available: {HOLIDAYS_AVAILABLE}")
    for date_str, boosts in preds.items():
        boosted = [z for z, m in boosts.items() if m > 1.0]
        print(date_str, "-> boosted zones:", boosted if boosted else "none")
