"""Project-local test exposure guard shared across run directories and label modes."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

from .labels import group_key, group_intervals, iso_date
from .profiles import instrument_for
from .runner import run_lock, write_json

REGISTRY = Path(__file__).resolve().parents[2] / "output/tuning/holdout-use"


def _intervals(reference):
    groups = {}
    for label in reference["labels"]:
        key = group_key(label)
        start, end = iso_date(label["information_start"]), iso_date(label["information_end"])
        if not start <= label["as_of"] <= end:
            raise ValueError("holdout information bounds do not contain label")
        groups.setdefault(key, []).append(label)
    components = []
    for start, end in sorted((min(start for start, _ in group_intervals(values)),
                             max(end for _, end in group_intervals(values))) for values in groups.values()):
        if components and start <= components[-1][1]:
            components[-1][1] = max(components[-1][1], end)
        else:
            components.append([start, end])
    if not components:
        raise ValueError("cannot claim an empty holdout")
    return components


def _load(path, market, symbol):
    if not path.exists():
        return {"schema_version": 1, "market": market, "symbol": symbol, "timeframe": "1d", "uses": []}
    payload = json.loads(path.read_text())
    if (payload.get("schema_version") != 1 or payload.get("market") != market
            or payload.get("symbol") != symbol or payload.get("timeframe") != "1d"
            or not isinstance(payload.get("uses"), list)):
        raise ValueError("invalid shared holdout exposure record")
    return payload


def _overlaps(left, right):
    return any(a <= d and c <= b for a, b in left for c, d in right)


def check_unused(reference):
    """Fail fast before search. Claim at evaluation remains authoritative."""
    market, symbol = reference["market"], reference["symbol"]
    instrument_for(market, symbol)
    intervals = _intervals(reference)
    directory = REGISTRY / market / symbol
    with run_lock(directory):
        payload = _load(directory / "uses.json", market, symbol)
        if any(_overlaps(intervals, use["information_intervals"]) for use in payload["uses"]):
            raise ValueError("holdout information was already exposed; use new unexposed groups, not another search")


def check_owner(reference, identity):
    market, symbol = reference["market"], reference["symbol"]
    directory = REGISTRY / market / symbol
    with run_lock(directory):
        payload = _load(directory / "uses.json", market, symbol)
        owned = [use for use in payload["uses"] if use["identity"] == identity]
        if len(owned) != 1 or owned[0]["information_intervals"] != _intervals(reference):
            raise ValueError("cached validation lacks matching shared holdout ownership")


@contextmanager
def claim_holdout(reference, identity):
    """Keep the reservation on failure; identical frozen candidates can recover."""
    market, symbol = reference["market"], reference["symbol"]
    instrument_for(market, symbol)
    intervals = _intervals(reference)
    directory = REGISTRY / market / symbol
    with run_lock(directory):
        path = directory / "uses.json"
        payload = _load(path, market, symbol)
        owned = [use for use in payload["uses"] if use["identity"] == identity]
        if owned and (len(owned) != 1 or owned[0]["information_intervals"] != intervals):
            raise ValueError("frozen candidate holdout ownership changed")
        if any(use["identity"] != identity and _overlaps(intervals, use["information_intervals"])
               for use in payload["uses"]):
            raise ValueError("holdout already exposed by another frozen candidate, across runs or label modes")
        if not owned:
            payload["uses"].append({"identity": identity, "information_intervals": intervals,
                                    "claimed_at": datetime.now(timezone.utc).isoformat(),
                                    "label_mode": reference["label_mode"]})
            write_json(path, payload)
        yield
