"""Offline AI reference contract. No network, credentials, or human-label import.

Schema compliance and exact citations do not establish phase correctness.
"""
from copy import deepcopy
from datetime import date
import json
import math
import re

import pandas as pd

from market_data import validate_ohlcv

VERSION = "wyckoff-ai-v1"
VERSION_V2 = "wyckoff-ai-v2"
VERSION_V3 = "wyckoff-ai-v3"
MODES = ("blind_label", "core_review")


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _enum(*values):
    return {"enum": list(values)}


TEXT = {"type": "string", "minLength": 1, "maxLength": 800}
REF = _object({"pointer": TEXT, "value": {"type": ["string", "number", "boolean", "null"]}})
CLAIM = _object({
    "kind": _enum("support", "counterevidence", "core_issue"), "statement": TEXT,
    "happened_at": TEXT, "known_at": TEXT,
    "observations": {"type": "array", "items": REF, "minItems": 1, "maxItems": 8},
})
RESPONSE_SCHEMA = _object({
    "version": _enum(VERSION), "origin": _enum("ai_reference"),
    "case_ref": TEXT, "mode": _enum(*MODES), "as_of": TEXT,
    "status": _enum("judged", "unresolved", "no_observable_phase"),
    "phase": _enum(None, "A", "B", "C", "D", "E"),
    "direction": _enum(None, "accumulation", "distribution", "undetermined"),
    "reason": _enum("supported", "missing_context", "missing_volume",
                    "competing_hypotheses", "no_observable_phase"),
    "range_start": _enum(None),  # extended to session strings below
    "bc_terminal_verified": _enum(False),
    "e_assessment": _object({
        "historical_structure": _enum("supported", "unsupported", "unassessed"),
        "current_structure": _enum("holding", "at_pivot", "challenged", "unavailable"),
        "support_lifecycle": _enum("valid", "invalidated", "unavailable"),
    }),
    "claims": {"type": "array", "items": CLAIM, "maxItems": 12},
    "needs": {"type": "array", "items": TEXT, "maxItems": 8},
})
RESPONSE_SCHEMA["properties"]["range_start"] = {"type": ["string", "null"]}
RESPONSE_SCHEMA_V2 = deepcopy(RESPONSE_SCHEMA)
RESPONSE_SCHEMA_V2["properties"]["version"] = _enum(VERSION_V2)
_v2_claim = RESPONSE_SCHEMA_V2["properties"]["claims"]["items"]
_v2_claim["properties"]["e_component"] = _enum(None, "historical_structure", "current_structure", "support_lifecycle")
_v2_claim["required"].append("e_component")


def _nullable(schema):
    return {**schema, "type": ["object", "null"]}


SESSION = {"type": "string", "minLength": 5, "maxLength": 12}
OPTIONAL_SESSION = {"type": ["string", "null"]}
# Sessions only: every price, extreme and relation is recomputed from the bars.
E_FACTS = _object({
    "side": _enum(None, "up", "down"),
    "range": _nullable(_object({"start": SESSION, "end": SESSION, "upper": SESSION, "lower": SESSION})),
    "leave": OPTIONAL_SESSION,
    "pivot": _nullable(_object({"comparison": SESSION, "reaction": SESSION, "action": SESSION,
                                "confirmed": SESSION})),
    "support": _nullable(_object({"reaction": SESSION, "test": SESSION, "confirmed": SESSION,
                                  "invalidated": OPTIONAL_SESSION})),
})
RESPONSE_SCHEMA_V3 = deepcopy(RESPONSE_SCHEMA_V2)
RESPONSE_SCHEMA_V3["properties"]["version"] = _enum(VERSION_V3)
RESPONSE_SCHEMA_V3["properties"]["e_facts"] = E_FACTS
RESPONSE_SCHEMA_V3["required"].append("e_facts")
_SCHEMAS = {VERSION: RESPONSE_SCHEMA, VERSION_V2: RESPONSE_SCHEMA_V2, VERSION_V3: RESPONSE_SCHEMA_V3}


def response_schema(version, mode, *, packet=None):
    """V1 stays frozen; v2 gives each E assessment its own cited claims; v3 adds checkable E facts."""
    if version not in _SCHEMAS or mode not in MODES:
        raise ValueError("unknown AI contract version/mode")
    schema = deepcopy(_SCHEMAS[version])
    if version != VERSION and mode == "blind_label":
        schema["properties"]["claims"]["items"]["properties"]["kind"] = _enum("support", "counterevidence")
    scope = packet.get("claim_time_scope") if packet is not None else None
    if version == VERSION_V3 and packet is not None and scope != "asof_assessment":
        raise ValueError("v3 facts require the as-of claim scope")
    if scope is not None:
        if version == VERSION or scope != "asof_assessment":
            raise ValueError("unsupported claim time scope")
        for field in ("case_ref", "mode", "as_of"):
            schema["properties"][field] = _enum(packet[field])
        schema["properties"]["claims"]["items"]["properties"]["known_at"] = _enum(packet["as_of"])
    return schema


def _schema_check(value, schema, path="response"):
    """Validate only the explicit schema vocabulary used by RESPONSE_SCHEMA."""
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        raise ValueError(f"{path}: invalid enum")
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    predicates = {"object": lambda: isinstance(value, dict),
                  "array": lambda: isinstance(value, list),
                  "string": lambda: isinstance(value, str),
                  "number": lambda: type(value) is int or (type(value) is float and math.isfinite(value)),
                  "boolean": lambda: type(value) is bool, "null": lambda: value is None}
    if kinds and not any(predicates[k]() for k in kinds):
        raise ValueError(f"{path}: invalid type")
    if isinstance(value, dict) and "properties" in schema:
        if set(value) != set(schema["properties"]):
            raise ValueError(f"{path}: missing or extra fields")
        for key, item in value.items():
            _schema_check(item, schema["properties"][key], f"{path}.{key}")
    if isinstance(value, list) and schema.get("type") == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: invalid item count")
        for i, item in enumerate(value):
            _schema_check(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            raise ValueError(f"{path}: invalid text length")
        if schema.get("minLength", 0) and not value.strip():
            raise ValueError(f"{path}: blank text")


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("explicit YYYY-MM-DD required")
    return value


def _anonymize_core(value, sessions):
    if isinstance(value, dict):
        return {k: _anonymize_core(v, sessions) for k, v in value.items() if k != "symbol"}
    if isinstance(value, list):
        return [_anonymize_core(v, sessions) for v in value]
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}(?:T.*)?$", value):
        if value[:10] not in sessions:
            raise ValueError("core date outside registered information window")
        return sessions[value[:10]]
    return value


def make_packet(df, *, as_of, information_start, analysis_scale, case_ref,
                mode="blind_label", core_snapshot=None):
    """Physically truncate before validation, scaling, or model-visible serialization.

    This daily offline entrypoint assumes supplied sessions are completed. It
    is not a live bar-closure/calendar resolver. Extra columns/attrs are ignored.
    Real dates remain in local metadata only. Review keeps raw prices to match
    core calculations; blind labels use ratios and no core-derived features.
    """
    as_of, information_start = _day(as_of), _day(information_start)
    if mode not in MODES or (mode == "blind_label" and core_snapshot is not None):
        raise ValueError("blind input must not contain a core answer")
    if analysis_scale not in ("registered_daily_window", "registered_daily_cycle"):
        raise ValueError("use an explicit supported scale; do not infer one from the answer")
    if not isinstance(case_ref, str) or not re.fullmatch(r"case-[0-9]{3,6}", case_ref):
        raise ValueError("case_ref must be an anonymous ordinal")
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is not None:
        raise ValueError("naive daily session dates required")
    if information_start > as_of:
        raise ValueError("information_start after as_of")
    session_days = df.index.normalize()
    visible = df.loc[(session_days >= pd.Timestamp(information_start)) & (session_days <= pd.Timestamp(as_of))]
    if not visible.index.equals(visible.index.normalize()):
        raise ValueError("daily session dates required")
    visible = validate_ohlcv(visible, require_volume=False)
    # Arbitrary attrs can contain provider paths, future exclusions, or notes.
    # Supply only neutral current-window metadata to the optional core call.
    visible.attrs = {"source": "registered_offline_frame", "timeframe": "1d"}
    if (visible.index[0].date().isoformat() != information_start
            or visible.index[-1].date().isoformat() != as_of):
        raise ValueError("registered endpoints must be observed sessions")
    dates = [t.date().isoformat() for t in visible.index]
    sessions = {d: f"s{i:04d}" for i, d in enumerate(dates)}
    price_base = float(visible.Close.iloc[0]) / 100 if mode == "blind_label" else 1.
    positive = visible.Volume[visible.Volume > 0] if "Volume" in visible else []
    volume_base = float(positive.iloc[0]) if len(positive) else None
    bars = []
    for day, row in visible.iterrows():
        bars.append({"session": sessions[day.date().isoformat()],
                     **{k.lower(): float(row[k]) / price_base for k in ("Open", "High", "Low", "Close")},
                     "volume": float(row.Volume) / volume_base if volume_base else None})
    packet = {"version": VERSION, "case_ref": case_ref, "mode": mode,
              "as_of": bars[-1]["session"], "timeframe": "1d",
              "analysis_scale": analysis_scale, "bars": bars,
              "context_complete": "not_established_by_window_length",
              "price_units": "first_close_100" if mode == "blind_label" else "source_price",
              "volume_units": "first_positive_volume_1" if volume_base else "unavailable",
              "volume_available": volume_base is not None}
    if mode == "core_review":
        if core_snapshot is not None:
            raise ValueError("external core snapshot is unbound; generate from the registered frame")
        # A fresh pure snapshot binds every derived observation to this exact
        # visible frame. A same-date/bar-count external JSON is not sufficient.
        from analysis import analyze_frame, to_json
        core_snapshot = json.loads(to_json(analyze_frame(visible)))
        if (not isinstance(core_snapshot, dict) or core_snapshot.get("schema_version") != 2
                or str(core_snapshot.get("as_of", ""))[:10] != as_of
                or core_snapshot.get("data", {}).get("bars") != len(visible)):
            raise ValueError("review requires a matching schema-2 snapshot/window")
        packet["core"] = _anonymize_core(deepcopy(core_snapshot), sessions)
        packet["core_units"] = "source_prices_and_source_volume; bars_volume_is_normalized"
    # Local provenance is deliberately excluded by build_prompt().
    return {"packet": packet, "local": {"as_of": as_of, "information_start": information_start,
            "information_end": as_of, "session_dates": dates, "origin": "ai_reference",
            "eligible_human_truth": False}}


def _resolve(packet, pointer):
    if not isinstance(pointer, str) or not pointer.startswith(("/bars/", "/core/")):
        raise ValueError("citation must address supplied bars or review core")
    value = packet
    try:
        for part in pointer.split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", part):
                    raise ValueError("invalid index")
                value = value[int(part)]
            else:
                value = value[part]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("citation not present in supplied evidence") from exc
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("cite a scalar observation, not a whole object")
    return value


def _no_duplicates(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate JSON key")
        result[k] = v
    return result


def validate_response(text, bundle, *, finish_reason):
    """Reject malformed, truncated, unsupported, or misattributed AI outputs.

    No repair, provider switching, or promotion occurs here. Free-text meaning
    and inferred cycle ownership still require separate review.
    """
    if (finish_reason != "stop" or not isinstance(text, str) or not text.strip()
            or len(text.encode("utf-8")) > 64_000):
        raise ValueError("response incomplete, refused, empty, or not normalized")
    try:
        value = json.loads(text, object_pairs_hook=_no_duplicates,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid JSON response") from exc
    packet = bundle["packet"]
    _schema_check(value, response_schema(packet["version"], packet["mode"], packet=packet))
    for field in ("version", "case_ref", "mode", "as_of"):
        if value[field] != packet[field]:
            raise ValueError("response identity mismatch")
    sessions = {b["session"]: i for i, b in enumerate(packet["bars"])}
    if value["range_start"] is not None and value["range_start"] not in sessions:
        raise ValueError("range start outside input")
    status = value["status"]
    default_e = {"historical_structure": "unassessed", "current_structure": "unavailable",
                 "support_lifecycle": "unavailable"}
    if value["e_assessment"] != default_e and not value["claims"]:
        raise ValueError("E assertions require cited observations")
    if packet["version"] != VERSION:
        for component, assessment in value["e_assessment"].items():
            if assessment != default_e[component]:
                expected_kind = "counterevidence" if assessment in ("unsupported", "challenged", "invalidated") else "support"
                if not any(c["e_component"] == component and c["kind"] == expected_kind for c in value["claims"]):
                    raise ValueError(f"E {component} requires its own {expected_kind} observations")
    if status == "judged":
        if value["phase"] not in tuple("ABCDE") or value["direction"] is None or value["reason"] != "supported":
            raise ValueError("judged response requires a unique supported phase/direction")
        if value["phase"] in "CDE" and value["direction"] == "undetermined":
            raise ValueError("C/D/E direction is unresolved")
        if value["needs"] or not any(c["kind"] == "support" for c in value["claims"]):
            raise ValueError("judged response requires support and no missing context")
        if not packet["volume_available"]:
            raise ValueError("volume unavailable: preserve unresolved")
        if value["phase"] == "E" and value["e_assessment"]["historical_structure"] != "supported":
            raise ValueError("judged E requires an assessed historical structure")
    else:
        if value["phase"] is not None or value["direction"] is not None:
            raise ValueError("unresolved/negative is separate from a phase class")
        if status == "unresolved":
            if value["reason"] not in ("missing_context", "missing_volume", "competing_hypotheses") or not value["needs"]:
                raise ValueError("unresolved response requires a reason and needed observation")
        elif (value["reason"] != "no_observable_phase" or value["needs"]
              or not packet["volume_available"] or not value["claims"]):
            raise ValueError("negative class requires an explicit evidence-backed assessment")
    for claim in value["claims"]:
        if claim["kind"] == "core_issue" and packet["mode"] != "core_review":
            raise ValueError("blind response cannot cite core issues")
        if (claim["happened_at"] not in sessions or claim["known_at"] not in sessions
                or sessions[claim["happened_at"]] > sessions[claim["known_at"]]):
            raise ValueError("unavailable or reversed claim dates")
        for observation in claim["observations"]:
            actual = _resolve(packet, observation["pointer"])
            quoted = observation["value"]
            numbers = type(actual) in (int, float) and type(quoted) in (int, float)
            if not ((numbers and actual == quoted) or (type(actual) is type(quoted) and actual == quoted)):
                raise ValueError("citation value differs from input")
            if observation["pointer"].startswith("/bars/"):
                i = int(observation["pointer"].split("/")[2])
                if i > sessions[claim["known_at"]]:
                    raise ValueError("claim predates its cited observation")
            elif claim["known_at"] != packet["as_of"]:
                raise ValueError("core snapshot is supplied as-of, not an earlier historical record")
    if packet["version"] == VERSION_V3:
        review = review_e_facts(value, packet)
        if review["findings"]:
            raise FactConstraintError(value, review)
    return value


class FactConstraintError(ValueError):
    """A well-formed v3 answer whose declared E facts fail the local bar check."""

    def __init__(self, answer, review):
        super().__init__("; ".join(f"{f['class']}: {f['component']}: {f['message']}" for f in review["findings"]))
        self.answer, self.review = answer, review


def review_e_facts(value, packet):
    """Check the declared E facts of a contract-valid v3 answer against the supplied bars.

    fact_error: a declaration contradicts the bars. evidence_gap: an assessment
    is asserted without the declaration that would make it traceable. Leaving a
    component null with its assessment unassessed/unavailable is neither.

    Only the published geometric conditions are applied; no core threshold or
    answer is consulted. Passing shows the stated geometry is true. It does not
    show that the swing scale, the chosen pivots or the phase are right.
    """
    bars = packet["bars"]
    index = {b["session"]: i for i, b in enumerate(bars)}
    last = len(bars) - 1
    facts, assessment = value["e_facts"], value["e_assessment"]
    findings, derived = [], {}

    def add(kind, component, message):
        findings.append({"class": kind, "component": component, "message": message})

    def located(component, declared):
        missing = [name for name, session in declared.items() if session not in index]
        if missing:
            raise ValueError(f"e_facts.{component} names a session outside input")
        return [index[session] for session in declared.values()]

    side = facts["side"]
    expected = {"accumulation": "up", "distribution": "down"}.get(value["direction"])
    if side is not None and expected is not None and side != expected:
        raise ValueError("e_facts.side contradicts the judged direction")
    up = side == "up"
    # Mirror the down side onto the up-side formulas: favourable extreme,
    # adverse extreme and close, each signed so that larger means further along.
    fav = [b["high"] if up else -b["low"] for b in bars]
    adv = [b["low"] if up else -b["high"] for b in bars]
    close = [b["close"] if up else -b["close"] for b in bars]
    adverse_field = "low" if up else "high"

    def swing(component, reaction, action, confirmed):
        """A pullback from `reaction` to `action`, confirmed by the first close beyond the reaction."""
        problems = []
        if fav[reaction] != max(fav[reaction:action + 1]):
            problems.append("reaction is not the extreme before the pullback")
        if adv[action] != min(adv[reaction + 1:confirmed + 1]):
            problems.append("action is not the pullback extreme between reaction and confirmation")
        first = next((i for i in range(action + 1, last + 1) if close[i] > fav[reaction]), None)
        if first != confirmed:
            problems.append("confirmed is not the first close beyond the reaction after the action")
        for problem in problems:
            add("fact_error", component, problem)
        return not problems

    start = end = None
    declared_range = facts["range"]
    if declared_range is not None:
        if value["range_start"] != declared_range["start"]:
            add("fact_error", "range", "range_start differs from e_facts.range.start")
        a, b, upper, lower = located("range", declared_range)
        if not (a < b and a <= upper <= b and a <= lower <= b):
            add("fact_error", "range", "start, end and extreme sessions are not ordered inside the range")
        else:
            start, end = a, b
            highest = max(x["high"] for x in bars[a:b + 1])
            lowest = min(x["low"] for x in bars[a:b + 1])
            derived["range"] = {
                "upper": highest, "lower": lowest,
                "upper_sessions": [x["session"] for x in bars[a:b + 1] if x["high"] == highest],
                "lower_sessions": [x["session"] for x in bars[a:b + 1] if x["low"] == lowest]}
            if bars[upper]["high"] != highest:
                add("fact_error", "range", "upper is not the highest high of the declared range")
            if bars[lower]["low"] != lowest:
                add("fact_error", "range", "lower is not the lowest low of the declared range")

    if facts["leave"] is not None:
        if side is None or declared_range is None:
            add("evidence_gap", "leave", "a leave needs the declared side and range")
        elif start is not None:
            (leave,) = located("leave", {"leave": facts["leave"]})
            # The boundary is the locally computed extreme, whichever session was named.
            boundary = derived["range"]["upper"] if up else -derived["range"]["lower"]
            first = next((i for i in range(end + 1, last + 1) if close[i] > boundary), None)
            derived["leave"] = {"boundary": abs(boundary),
                                "first_close_beyond": bars[first]["session"] if first is not None else None}
            if first != leave:
                add("fact_error", "leave", "leave is not the first close beyond the range boundary after its end")
            else:
                # Observable on its own; it is not a confirmed support object.
                derived["boundary_hold"] = {"from": bars[leave]["session"], "through": packet["as_of"],
                                            "unbroken": all(c > boundary for c in close[leave:])}

    pivot_ok = False
    if facts["pivot"] is not None:
        if side is None:
            add("evidence_gap", "pivot", "a pivot needs the declared side")
        else:
            comparison, reaction, action, confirmed = located("pivot", facts["pivot"])
            if not comparison < reaction < action < confirmed <= last:
                add("fact_error", "pivot", "comparison, reaction, action and confirmed are not in order")
            else:
                pivot_ok = swing("pivot", reaction, action, confirmed)
                if fav[reaction] != max(fav[comparison:action + 1]):
                    add("fact_error", "pivot", "reaction is not the extreme between comparison and action")
                    pivot_ok = False
                if adv[comparison] != min(adv[comparison:action + 1]) or not adv[action] > adv[comparison]:
                    add("fact_error", "pivot", "action does not hold beyond a comparison pivot that bounds the swing")
                    pivot_ok = False
                if pivot_ok:
                    relation = ("holding" if close[last] > adv[action] else
                                "at_pivot" if close[last] == adv[action] else "challenged")
                    derived["pivot"] = {
                        "level": bars[action][adverse_field], "asof_close": bars[last]["close"],
                        "relation": relation,
                        "pierced_after_confirmation": any(x < adv[action] for x in adv[confirmed + 1:])}
    current = assessment["current_structure"]
    if current != "unavailable":
        if facts["pivot"] is None:
            add("evidence_gap", "current_structure", "asserted without a declared confirmed pivot")
        elif pivot_ok and current != derived["pivot"]["relation"]:
            add("fact_error", "current_structure",
                f"as_of close is {derived['pivot']['relation']} against the declared pivot")

    support_ok = False
    if facts["support"] is not None:
        if side is None or declared_range is None:
            add("evidence_gap", "support", "a support needs the declared side and its target range")
        elif start is not None:
            declared = dict(facts["support"])
            invalidated = declared.pop("invalidated")
            reaction, test, confirmed = located("support", declared)
            if not start <= reaction < test < confirmed <= last:
                add("fact_error", "support", "reaction, test and confirmed are not in order inside the target range's history")
            elif swing("support", reaction, test, confirmed):
                first = next((i for i in range(confirmed + 1, last + 1) if close[i] < adv[test]), None)
                derived["support"] = {"level": bars[test][adverse_field],
                                      "first_close_through": bars[first]["session"] if first is not None else None}
                support_ok = invalidated == derived["support"]["first_close_through"]
                if not support_ok:
                    add("fact_error", "support", "invalidated is not the first close through the support after confirmation")
    lifecycle = assessment["support_lifecycle"]
    if lifecycle != "unavailable":
        if facts["support"] is None:
            add("evidence_gap", "support_lifecycle", "asserted without a declared confirmed support; an unbroken boundary is a separate observation")
        elif support_ok and lifecycle != ("valid" if facts["support"]["invalidated"] is None else "invalidated"):
            add("fact_error", "support_lifecycle", "contradicts the declared support's close history")

    if assessment["historical_structure"] == "supported" and (
            side is None or declared_range is None or facts["leave"] is None):
        add("evidence_gap", "historical_structure", "supported needs the declared side, range and leave")
    return {"scope": "declared geometry against supplied bars; not swing scale, pivot choice or phase truth",
            "findings": findings, "derived": derived}
