"""Human supplied cycle registration and blind charts; never infer a cycle."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from .labels import iso_date, joint_classes
from .label_quality import _identity
from .space import common_warmup

VERSION = "blind-ohclv-cycle-v1"
REGISTRATION_VERSION = "human-cycle-registration-v1"


def validate_registration(raw, dates, segments, exclusions):
    if not isinstance(raw, dict) or raw.get("version") != REGISTRATION_VERSION:
        raise ValueError("human cycle registration requires its declared version")
    mode = raw.get("label_mode")
    joint_classes(mode)
    person = raw.get("registered_by")
    if (not isinstance(person, dict) or not isinstance(person.get("name"), str) or not person["name"].strip()
            or person.get("independent_human") is not True or person.get("no_strategy_outputs") is not True
            or person.get("type", "human") != "human" or person.get("independence", "external_human") != "external_human"):
        raise ValueError("cycle registration requires a named independent human and no strategy outputs")
    try:
        stamp = datetime.fromisoformat(person.get("registered_at", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("registered_at requires a timezone timestamp") from exc
    if stamp.utcoffset() is None:
        raise ValueError("registered_at requires a timezone timestamp")
    if "identity_id" in person and (not isinstance(person["identity_id"], str) or not person["identity_id"].strip()):
        raise ValueError("registration identity_id must be nonempty")
    partition = raw.get("partition")
    if not isinstance(partition, dict):
        raise ValueError("cycle registration requires frozen partition")
    a, b, c = [iso_date(partition.get(key), key) for key in ("development_start", "holdout_start", "holdout_end")]
    if not a < b < c:
        raise ValueError("cycle registration partitions must be ordered and disjoint")
    cycles = raw.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise ValueError("no human cycles registered; blank templates are not cycles")
    lookup, seen, targets, result = {day: i for i, day in enumerate(dates)}, set(), set(), []
    for cycle in cycles:
        if not isinstance(cycle, dict):
            raise ValueError("cycle registration requires complete metadata objects")
        ident, role = cycle.get("cycle_id"), cycle.get("role")
        if not isinstance(ident, str) or not ident.strip() or ident in seen or role not in ("development", "holdout"):
            raise ValueError("cycle requires unique human cycle_id and registered role")
        seen.add(ident)
        if not isinstance(cycle.get("analysis_scale"), str) or not cycle["analysis_scale"].strip():
            raise ValueError("cycle requires human analysis_scale")
        parent = cycle.get("parent_cycle")
        if parent is not None and (not isinstance(parent, str) or not parent.strip() or parent == ident):
            raise ValueError("parent_cycle must be an independent opaque id or null")
        start, end = iso_date(cycle.get("information_start")), iso_date(cycle.get("information_end"))
        if start not in lookup or end not in lookup or start > end:
            raise ValueError("cycle requires ordered actual source information bounds")
        if not any(s["start"] <= start <= end <= s["end"] for s in segments):
            raise ValueError("cycle information crosses a continuous segment")
        left, right = (a, b) if role == "development" else (b, c)
        if not left <= start <= end < right:
            raise ValueError("complete cycle information crosses its registered partition")
        if any(start <= y and x <= end for x, y in exclusions):
            raise ValueError("complete cycle information overlaps diagnostic or consumed exposure")
        days = cycle.get("case_dates")
        if not isinstance(days, list) or not days:
            raise ValueError("cycle requires exact human selected case_dates")
        for day in days:
            iso_date(day, "case_date")
            if day not in lookup or not start <= day <= end or day in targets:
                raise ValueError("cycle target must be a unique actual session inside complete bounds")
            targets.add(day)
        result.append({**deepcopy(cycle), "parent_cycle": parent, "case_dates": sorted(days)})
    return {**deepcopy(raw), "cycles": sorted(result, key=lambda x: x["cycle_id"])}


def validate_cycle_reviewer(manifest, review):
    """Known registrar future exposure cannot be attested away by that person."""
    if manifest.get("version") != VERSION:
        return
    if not isinstance(review, dict) or not isinstance(review.get("reviewer"), dict):
        raise ValueError("cycle review requires named human reviewer metadata")
    registrar = manifest["cycle_registration"]["registered_by"]
    person = review.get("reviewer", {})
    if not isinstance(person.get("name"), str) or not person["name"].strip():
        raise ValueError("cycle review requires named human reviewer metadata")
    rows = review.get("labels")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("cycle review requires complete case objects")
    same = _identity(person.get("name", "")) == _identity(registrar["name"])
    if person.get("identity_id") and registrar.get("identity_id"):
        same |= _identity(person["identity_id"]) == _identity(registrar["identity_id"])
    latest = max(cycle["information_end"] for cycle in manifest["cycle_registration"]["cycles"])
    completed = [row for row in rows if row.get("status") == "completed"]
    if manifest["settings"]["label_mode"] == "asof" and same and any(iso_date(row.get("as_of")) < latest for row in completed):
        raise ValueError("cycle registrar already saw later prices; cannot supply these asof labels")
    if completed:
        try:
            reviewed = datetime.fromisoformat(person.get("reviewed_at", ""))
            registered = datetime.fromisoformat(registrar["registered_at"])
            if reviewed.utcoffset() is None or reviewed < registered:
                raise ValueError("cycle review precedes its frozen human registration")
        except (TypeError, ValueError) as exc:
            raise ValueError("cycle review requires timestamp after registration") from exc


def reference_case(manifest, case):
    if manifest.get("version") != VERSION:
        return dict(case)
    cycle = next(x for x in manifest["cycle_registration"]["cycles"] if x["cycle_id"] == case["cycle_id"])
    return {**case, "group_information_start": cycle["information_start"],
            "group_information_end": cycle["information_end"]}


def _build(history, raw, exclusion_snapshot):
    from . import annotation as ann
    market, symbol = raw.get("market"), raw.get("symbol")
    if raw.get("timeframe") != "1d":
        raise ValueError("cycle source timeframe must be 1d")
    frame, item = ann.load_history(history, market, symbol)
    source = ann._source(frame, item, history, market, symbol)
    excluded = [tuple(x) for x in exclusion_snapshot["diagnostic_intervals"] + exclusion_snapshot["consumed_intervals"]]
    registration = validate_registration(raw, frame.index.strftime("%Y-%m-%d").tolist(), item["continuous_segments"], excluded)
    mode, cases, charts = registration["label_mode"], [], {}
    for cycle in registration["cycles"]:
        for day in cycle["case_dates"]:
            end = day if mode == "asof" else cycle["information_end"]
            case = {"id": f"{market}-{symbol}-{mode}-{day}", "market": market, "symbol": symbol,
                    "timeframe": "1d", "as_of": day, "label_mode": mode, "role": cycle["role"],
                    "panel": "cycle", "block_id": None, "cycle_id": cycle["cycle_id"],
                    "analysis_scale": cycle["analysis_scale"], "parent_cycle": cycle["parent_cycle"],
                    "information_start": cycle["information_start"], "information_end": end,
                    "chart": f"charts/{market}-{symbol}-{mode}-{day}.html"}
            cases.append(case)
            charts[case["id"]] = ann._bars(frame.loc[case["information_start"]:end])
    return frame, source, registration, sorted(cases, key=lambda x: x["as_of"]), charts


def prepare_cycle_annotation_pack(history, registration_path, output, *, rubric_path):
    from . import annotation as ann
    from .holdout import REGISTRY, _load
    from .splits import KNOWN_DIAGNOSTIC_INTERVALS
    contents = Path(registration_path).read_bytes()
    raw = json.loads(contents)
    if not isinstance(raw, dict):
        raise ValueError("cycle registration must be an object")
    market, symbol = raw.get("market"), raw.get("symbol")
    uses = _load(REGISTRY / str(market) / str(symbol) / "uses.json", market, symbol)["uses"]
    snapshot = {"diagnostic_intervals": [list(x) for x in KNOWN_DIAGNOSTIC_INTERVALS.get((market, symbol), ())],
                "consumed_intervals": [list(x) for use in uses for x in use["information_intervals"]]}
    _, source, registration, cases, charts = _build(history, raw, snapshot)
    rubric = Path(rubric_path).read_bytes()
    if not rubric.strip():
        raise ValueError("cycle pack requires nonempty frozen rubric")
    import hashlib
    identity = {"version": VERSION, "source": source, "cycle_registration": registration,
                "exclusion_snapshot": snapshot, "registration_sha256": hashlib.sha256(contents).hexdigest(),
                "protocol_sha256": hashlib.sha256(ann.PROTOCOL.encode()).hexdigest(),
                "rubric_identity": {"version": "phase-rubric-v1", "sha256": hashlib.sha256(rubric).hexdigest()},
                "review_policy": "independent-double-blind-v1"}
    output = Path(output).resolve()
    if output.exists() and any(p.name != ".lock" for p in output.iterdir()):
        previous = load_cycle_manifest(output)
        if previous["pack_id"] != ann._digest(identity):
            raise ValueError("refusing overwrite of different registered cycle pack")
        return previous
    output.mkdir(parents=True, exist_ok=True)
    (output / "registration.json").write_bytes(contents)
    (output / "rubric.md").write_bytes(rubric)
    (output / "protocol.txt").write_text(ann.PROTOCOL)
    fingerprints = {}
    for case in cases:
        path = output / case["chart"]
        path.parent.mkdir(exist_ok=True)
        path.write_text(ann._chart(case, charts[case["id"]]))
        fingerprints[case["id"]] = {"sha256": ann._bytes_digest(path), "bars_sha256": ann._digest(charts[case["id"]]),
                                    "bars": len(charts[case["id"]])}
    sampling = {"method": REGISTRATION_VERSION, "panel": "cycle", "contexts_may_overlap": True,
                "holdout_end_exclusive": True, "human_cycle_bounds": [
                    {key: x[key] for key in ("cycle_id", "role", "information_start", "information_end")}
                    for x in registration["cycles"]]}
    manifest = {**identity, "pack_id": ann._digest(identity), "schema_version": 1,
                "settings": {"label_mode": registration["label_mode"], **registration["partition"]},
                "status": "AWAITING_HUMAN_REVIEW", "common_warmup_bars": common_warmup(market),
                "cases": cases, "charts": fingerprints, "sampling": sampling,
                "diagnostic_intervals": snapshot["diagnostic_intervals"]}
    ann._write_json(output / "manifest.json", manifest)
    template = {"schema_version": 1, "pack_id": manifest["pack_id"], "source_identity": source,
                "label_mode": registration["label_mode"], "rubric_identity": identity["rubric_identity"],
                "reviewer": {"name": "", "reviewed_at": "", "independent_human": False,
                             "no_strategy_outputs": False, "no_future_prices_for_asof": False,
                             "blind_review": False, "no_other_review_seen": False, "independent_of_other_review": False},
                "labels": [{**case, "status": "blank", "phase": None, "regime": None,
                            "request_context": False, "additional_exposure": [], "ambiguity": None,
                            "ambiguity_resolved": False, "resolver": "", "resolution_rationale": "", "notes": ""} for case in cases]}
    for name in ("annotation.json", "review-a.json", "review-b.json"):
        ann._write_json(output / name, template)
    ann._write_json(output / "cycle-bounds.json", {"market": market, "symbol": symbol, "timeframe": "1d",
                                                "human_cycle_bounds": sampling["human_cycle_bounds"]})
    return manifest


def load_cycle_manifest(pack):
    from . import annotation as ann
    pack = Path(pack)
    manifest = json.loads((pack / "manifest.json").read_text())
    keys = ("version", "source", "cycle_registration", "exclusion_snapshot", "registration_sha256",
            "protocol_sha256", "rubric_identity", "review_policy")
    identity = {key: manifest[key] for key in keys}
    if (manifest["version"] != VERSION or manifest["pack_id"] != ann._digest(identity)
            or manifest["review_policy"] != "independent-double-blind-v1"
            or ann._bytes_digest(pack / "registration.json") != manifest["registration_sha256"]
            or ann._bytes_digest(pack / "rubric.md") != manifest["rubric_identity"]["sha256"]
            or ann._bytes_digest(pack / "protocol.txt") != manifest["protocol_sha256"]):
        raise ValueError("registered cycle pack identity changed")
    raw = json.loads((pack / "registration.json").read_text())
    _, source, registration, cases, charts = _build(manifest["source"]["history"], raw, manifest["exclusion_snapshot"])
    expected_settings = {"label_mode": registration["label_mode"], **registration["partition"]}
    if (source != manifest["source"] or registration != manifest["cycle_registration"]
            or cases != manifest["cases"] or expected_settings != manifest["settings"]):
        raise ValueError("registered cycle source/case/complete bounds changed")
    expected_sampling = {"method": REGISTRATION_VERSION, "panel": "cycle", "contexts_may_overlap": True,
                "holdout_end_exclusive": True, "human_cycle_bounds": [
                    {key: x[key] for key in ("cycle_id", "role", "information_start", "information_end")}
                    for x in registration["cycles"]]}
    if manifest["sampling"] != expected_sampling:
        raise ValueError("registered cycle sampling changed")
    expected_bounds = {"market": source["market"], "symbol": source["symbol"], "timeframe": "1d",
                       "human_cycle_bounds": expected_sampling["human_cycle_bounds"]}
    if (manifest.get("status") != "AWAITING_HUMAN_REVIEW" or manifest.get("common_warmup_bars") != common_warmup(source["market"])
            or manifest.get("diagnostic_intervals") != manifest["exclusion_snapshot"]["diagnostic_intervals"]
            or manifest["rubric_identity"].get("version") != "phase-rubric-v1"
            or json.loads((pack / "cycle-bounds.json").read_text()) != expected_bounds
            or set(manifest["charts"]) != {case["id"] for case in cases}):
        raise ValueError("registered cycle metadata or full exported bounds changed")
    for case in cases:
        fingerprint = manifest["charts"][case["id"]]
        expected = {"sha256": ann._bytes_digest(pack / case["chart"]),
                    "bars_sha256": ann._digest(charts[case["id"]]), "bars": len(charts[case["id"]])}
        if fingerprint != expected or (pack / case["chart"]).read_text() != ann._chart(case, charts[case["id"]]):
            raise ValueError("registered cycle chart or actual prefix changed")
    return manifest
