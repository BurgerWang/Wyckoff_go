"""Independent human reference labels; permissive missing provenance is diagnostic only."""
import json
from datetime import date, datetime
from pathlib import Path

MARKETS = {"us_equities": "US equity", "binance_spot": "Binance spot USDT"}
LABEL_MODES = ("asof", "retrospective")
PHASES = tuple("ABCDE")


def iso_date(value, field="date"):
    """Labels require exact dates; prediction timestamps may be datetime objects."""
    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must be YYYY-MM-DD")
    return value


def directional_side(regime):
    if regime in ("accumulation", "reaccumulation", "markup"):
        return "accumulation"
    if regime in ("distribution", "redistribution", "markdown"):
        return "distribution"
    return "undetermined"


def joint_classes(label_mode):
    if label_mode not in LABEL_MODES:
        raise ValueError("label_mode must be asof or retrospective")
    states = tuple(f"{p}/{side}" for side in ("accumulation", "distribution") for p in PHASES)
    return states + (("A/undetermined", "B/undetermined", "no_observable_phase")
                     if label_mode == "asof" else ())


def reference_state(label, label_mode):
    phase, regime = label.get("phase"), label.get("regime")
    if phase == "no_observable_phase":
        if label_mode != "asof" or regime not in ("none", "unknown", "undetermined", "no_observable_phase"):
            raise ValueError("no_observable_phase is only an explicit asof negative class")
        return "no_observable_phase"
    if phase not in PHASES:
        raise ValueError("reference phase must be A-E or explicit asof no_observable_phase")
    if regime not in ("accumulation", "reaccumulation", "markup", "distribution",
                      "redistribution", "markdown", "unknown", "undetermined", "trading range"):
        raise ValueError("invalid reference regime")
    side = directional_side(regime)
    state = f"{phase}/{side}"
    if state not in joint_classes(label_mode):
        raise ValueError("reference requires a unique phase/direction in this label mode")
    return state


def group_key(label):
    panel = label.get("panel")
    ident = label.get("cycle_id" if panel == "cycle" else "block_id")
    if panel not in ("cycle", "window") or not isinstance(ident, str) or not ident.strip():
        raise ValueError("label requires a cycle_id or a predeclared window block_id")
    return panel, ident


def group_information_bounds(label):
    """Keep full cycle selection exposure separate from each visible case."""
    start = iso_date(label.get("information_start"), "information_start")
    end = iso_date(label.get("information_end"), "information_end")
    day = iso_date(label.get("as_of"), "as_of")
    if not start <= day <= end or (label.get("label_mode") == "asof" and end > day):
        raise ValueError("invalid case information bounds")
    a, b = label.get("group_information_start"), label.get("group_information_end")
    if a is None and b is None:
        return start, end
    if a is None or b is None or label.get("panel") != "cycle":
        raise ValueError("complete group information bounds require a cycle")
    a, b = iso_date(a, "group_information_start"), iso_date(b, "group_information_end")
    if not a <= start <= end <= b:
        raise ValueError("group information bounds must contain actual case exposure")
    return a, b


def group_intervals(labels):
    """Validate one group's declarations; never hide inconsistent full bounds."""
    declarations = {(x.get("group_information_start"), x.get("group_information_end")) for x in labels}
    if len(declarations) > 1 and any(pair != (None, None) for pair in declarations):
        raise ValueError("inconsistent complete group information bounds")
    return [group_information_bounds(x) for x in labels]


def load_labels(path, *, market, symbol, label_mode):
    """Load a possibly multi-instrument JSON reference without transplanting sources.

    Missing interval, panel and role fields are recorded, never guessed into
    supervised eligibility. Legacy digitized cases can still be diagnosed.
    """
    if market not in MARKETS or not isinstance(symbol, str) or not symbol:
        raise ValueError("invalid target market/symbol")
    joint_classes(label_mode)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("labels"), list) or not isinstance(raw.get("sources"), list):
        raise ValueError("reference requires labels and sources lists")
    declared_mode = raw.get("label_mode")
    if declared_mode is not None and declared_mode != label_mode:
        raise ValueError("reference label_mode mismatch")
    if raw.get("market", market) != market or raw.get("symbol", symbol) != symbol:
        raise ValueError("reference target market/symbol mismatch")
    sources = {}
    for source in raw["sources"]:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not source["id"]:
            raise ValueError("source requires an id")
        if source["id"] in sources:
            raise ValueError("duplicate source id")
        sources[source["id"]] = source
    labels, seen, ids, local_manifests = [], set(), set(), {}
    for original in raw["labels"]:
        if not isinstance(original, dict):
            raise ValueError("each label must be an object")
        # Multi-symbol documents are supported; mismatched source identity for
        # a selected target is always an error rather than a dropped sample.
        if original.get("market") != market or original.get("symbol") != symbol:
            continue
        label = dict(original)
        day = iso_date(label.get("as_of"), "as_of")
        if day in seen:
            raise ValueError("duplicate/conflicting target session")
        seen.add(day)
        if label.get("id") is not None:
            if not isinstance(label["id"], str) or label["id"] in ids:
                raise ValueError("invalid/duplicate label id")
            ids.add(label["id"])
        if label.get("timeframe") != "1d":
            raise ValueError("reference timeframe must be 1d")
        mode = label.get("label_mode", declared_mode)
        if mode is None and label.get("retrospective") is True:
            mode = "retrospective"
        if mode != label_mode:
            raise ValueError("missing or mismatched label_mode")
        if label.get("retrospective") is True and mode == "asof":
            raise ValueError("retrospective context cannot be an asof label")
        if label.get("independence") != "external_human":
            raise ValueError("reference must be independent external human, not synthetic/rule/LLM truth")
        if label.get("eligible", True) is not True or label.get("adjudication") in ("ambiguous", "excluded"):
            raise ValueError("excluded or ambiguous references cannot become scoreable truth")
        ambiguous = label.get("ambiguity") not in (None, False, "", "none")
        if ambiguous and (label.get("ambiguity_resolved") is not True
                or not isinstance(label.get("resolver"), str) or not label["resolver"].strip()
                or not isinstance(label.get("resolution_rationale"), str) or not label["resolution_rationale"].strip()):
            raise ValueError("reference ambiguity requires explicit named independent resolution and rationale")
        source = sources.get(label.get("source_id"))
        if not source or (source.get("original_symbol") != symbol
                          or source.get("original_market") not in (market, MARKETS[market])
                          or source.get("original_timeframe") != "1d"):
            raise ValueError("source symbol/market/timeframe does not support target")
        if not isinstance(source.get("author"), str) or not source["author"].strip():
            raise ValueError("reference requires attributable human author and HTTPS source or verified local protocol")
        if ambiguous and label["resolver"].strip() == source["author"].strip():
            raise ValueError("ambiguity resolver must be independent of initial source author")
        if source.get("type") == "local_annotation_protocol":
            from .annotation import validate_annotation_label, validate_annotation_source
            if source["id"] not in local_manifests:
                local_manifests[source["id"]] = validate_annotation_source(source, path)
            validate_annotation_label(label, source, path, local_manifests[source["id"]])
        elif not str(source.get("source_url", "")).startswith("https://"):
            raise ValueError("reference requires HTTPS source or verified local annotation protocol")
        if source.get("independence", "external_human") != "external_human":
            raise ValueError("source is not independent human evidence")
        if mode == "asof" and source.get("hindsight") is True:
            raise ValueError("hindsight source cannot support asof labels")
        issues = []
        for field in ("information_start", "information_end"):
            if label.get(field) is None:
                issues.append(f"missing_{field}")
                label[field] = None
            else:
                label[field] = iso_date(label[field], field)
        start, end = label["information_start"], label["information_end"]
        if (start is not None and start > day) or (end is not None and end < day):
            raise ValueError("information interval must contain as_of")
        if mode == "asof" and end is not None and end > day:
            raise ValueError("asof information_end cannot include future prices")
        for field in ("start", "end"):
            if field in label and iso_date(label[field], field) != day:
                raise ValueError("labels must identify exact single sessions, not interpolated spans")
        if not label.get("panel"):
            issues.append("missing_panel")
            label["panel"] = "cycle" if label.get("cycle_id") else "window"
        if label.get("window_id") and not label.get("block_id"):
            label["block_id"] = label["window_id"]
        group_key(label)
        if label.get("role") is None:
            issues.append("missing_role")
            label["role"] = "diagnostic"
        if label["role"] not in ("development", "holdout", "diagnostic"):
            raise ValueError("role must be development, holdout or diagnostic")
        label.update(as_of=day, label_mode=mode, source=dict(source),
                     provenance_issues=issues, diagnostic_only=bool(issues) or label["role"] == "diagnostic")
        label["joint_state"] = reference_state(label, label_mode)
        if label.get("group_information_start") is not None or label.get("group_information_end") is not None:
            group_information_bounds(label)
        labels.append(label)
    grouped = {}
    for label in labels:
        grouped.setdefault(group_key(label), []).append(label)
    for values in grouped.values():
        if all(x["information_start"] is not None and x["information_end"] is not None for x in values):
            group_intervals(values)
    for ident, source in sources.items():
        if (source.get("type") == "local_annotation_protocol" and "consensus_reviews" in source
                and source.get("original_symbol") == symbol and source.get("original_market") in (market, MARKETS[market])):
            from .annotation import validate_annotation_source
            if ident not in local_manifests:
                local_manifests[ident] = validate_annotation_source(source, path)
            manifest = local_manifests[ident]
            from .annotation import _verified_review
            merged = _verified_review(source, path, manifest)
            expected = {x["id"] for x in merged["labels"] if x["status"] == "completed"}
            actual = {x["id"] for x in labels if x["source_id"] == ident}
            if expected != actual:
                raise ValueError("consensus reference must preserve the complete completed case set and coverage")
    return {"labels": sorted(labels, key=lambda x: x["as_of"]), "label_mode": label_mode,
            "market": market, "symbol": symbol, "sources": list(sources.values()),
            "sampling": raw.get("sampling"), "label_path": str(Path(path).resolve()),
            "diagnostic_only": any(x["diagnostic_only"] for x in labels),
            "schema_version": 1}
