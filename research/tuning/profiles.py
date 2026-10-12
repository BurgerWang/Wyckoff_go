"""Explicit, time-aware per-security configuration resolution."""
import json
from pathlib import Path

import pandas as pd

from research.universe import PROFILES
from wyckoff_core.models import CoreConfig


def instrument_for(market, symbol):
    if market not in PROFILES:
        raise ValueError("unknown research market")
    found = next((i for i in PROFILES[market].instruments if i.symbol == symbol), None)
    if found is None:
        raise ValueError(f"{market}/{symbol} is outside the declared universe")
    return found


def resolve_config(market, symbol, *, explicit=None, profiles=None, as_of=None, mode="historical", label_mode="asof"):
    instrument_for(market, symbol)
    if mode not in ("historical", "reconstruction", "live"):
        raise ValueError("invalid config application mode")
    if explicit is not None:
        if not isinstance(explicit, CoreConfig):
            raise ValueError("explicit config must be CoreConfig")
        return explicit
    if profiles is None:
        return PROFILES[market].config
    payload = json.loads(Path(profiles).read_text()) if isinstance(profiles, (str, Path)) else profiles
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported symbol profile schema")
    records = payload.get("profiles")
    if not isinstance(records, list):
        raise ValueError("profiles must be a list")
    found = [p for p in records if p.get("market") == market and p.get("symbol") == symbol]
    if len(found) > 1:
        raise ValueError("duplicate per-symbol profiles")
    if not found:
        return PROFILES[market].config
    item = found[0]
    if (item.get("timeframe") != "1d" or item.get("status") != "VALIDATED_ON_LABELED_HOLDOUT"
            or item.get("label_mode") != label_mode):
        raise ValueError("only validated matching daily symbol profiles are applicable")
    def date_value(key):
        value = pd.Timestamp(item[key])
        if pd.isna(value) or value.tz is not None or value != value.normalize():
            raise ValueError(f"{key} must be a valid naive date")
        return value
    trained, effective = date_value("training_end"), date_value("effective_from")
    information_end = date_value("label_information_end")
    created = pd.Timestamp(item["created_at"])
    if pd.isna(created) or created.tz is None:
        raise ValueError("profile created_at must include a timezone")
    created_date = created.tz_convert("UTC").tz_localize(None).normalize()
    if trained >= effective or information_end >= effective or created_date > effective:
        raise ValueError("profile training end must precede effective date")
    if mode != "reconstruction":
        if as_of is None:
            raise ValueError("as_of is required for calibrated symbol profiles")
        date = pd.Timestamp(as_of)
        if pd.isna(date) or date.tz is not None or date < effective or date <= max(trained, information_end):
            raise ValueError("later-calibrated parameters cannot be applied to this historical date")
        if item.get("effective_until") and date >= date_value("effective_until"):
            raise ValueError("symbol profile is outside its application period")
    return CoreConfig(**item["params"])
