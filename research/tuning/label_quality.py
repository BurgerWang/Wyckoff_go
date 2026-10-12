"""Descriptive agreement of two independent human reviews, never market truth.

This module is deliberately pure: callers read JSON and verify the pack's files
with annotation._load_manifest before passing its manifest. No trial, reference
import, adjudication, exposure registration or file write occurs here.
"""
from collections import Counter
from datetime import datetime
import json
import unicodedata

from .labels import MARKETS, group_key, iso_date, joint_classes, reference_state

STATUSES = ("blank", "completed", "unresolved", "excluded")
CASE_FIELDS = ("id", "market", "symbol", "timeframe", "as_of", "label_mode", "role",
               "panel", "block_id", "cycle_id", "analysis_scale", "parent_cycle", "information_start", "information_end", "chart")
FROZEN_FIELDS = ("rubric_id", "rubric_sha256", "protocol_sha256", "rubric_identity")

# Documentation/integration schema; semantic cross-file and mode checks below
# remain authoritative. No jsonschema dependency is required by the CLI.
REVIEW_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Independent blind human phase review",
    "type": "object",
    "required": ["pack_id", "source_identity", "label_mode", "reviewer", "labels"],
    "properties": {
        "pack_id": {"type": "string", "minLength": 1},
        "label_mode": {"enum": ["asof", "retrospective"]},
        "source_identity": {"type": "object", "required": ["market", "symbol", "timeframe"], "properties": {
            "market": {"enum": list(MARKETS)}, "symbol": {"type": "string", "minLength": 1}, "timeframe": {"const": "1d"}}},
        "sampling": {"type": "object", "minProperties": 1},
        "rubric_identity": {"type": "object", "required": ["version", "sha256"], "properties": {
            "version": {"type": "string", "minLength": 1}, "sha256": {"type": "string", "minLength": 1}}},
        "reviewer": {"type": "object", "required": ["name", "reviewed_at", "independent_human", "no_strategy_outputs",
                                                      "blind_review", "no_other_review_seen", "independent_of_other_review"],
                     "properties": {
                         "name": {"type": "string", "minLength": 1}, "identity_id": {"type": "string", "minLength": 1},
                         "reviewed_at": {"type": "string", "format": "date-time"}, "type": {"const": "human"},
                         "independence": {"const": "external_human"},
                         **{field: {"const": True} for field in ("independent_human", "no_strategy_outputs", "blind_review",
                                                                "no_other_review_seen", "independent_of_other_review")},
                         "no_future_prices_for_asof": {"type": "boolean"}}},
        "labels": {"type": "array", "items": {"type": "object", "required": [
            "id", "market", "symbol", "timeframe", "as_of", "label_mode", "role", "panel",
            "information_start", "information_end", "status"], "properties": {
                "id": {"type": "string", "minLength": 1}, "market": {"enum": list(MARKETS)},
                "symbol": {"type": "string", "minLength": 1}, "timeframe": {"const": "1d"},
                "label_mode": {"enum": ["asof", "retrospective"]}, "role": {"enum": ["development", "holdout", "diagnostic"]},
                "panel": {"enum": ["cycle", "window"]}, "block_id": {"type": "string", "minLength": 1},
                "cycle_id": {"type": "string", "minLength": 1},
                **{field: {"type": "string", "format": "date"} for field in ("as_of", "information_start", "information_end")},
                "status": {"enum": list(STATUSES)}, "phase": {"type": ["string", "null"]},
                "regime": {"type": ["string", "null"]}, "request_context": {"type": "boolean"},
                "additional_exposure": {"type": "array", "items": {"type": "object", "required": ["information_start", "information_end"],
                    "properties": {field: {"type": "string", "format": "date"} for field in ("information_start", "information_end")}}},
                "notes": {"type": "string"}}, "allOf": [
                    {"if": {"properties": {"panel": {"const": "window"}}}, "then": {"required": ["block_id"]}},
                    {"if": {"properties": {"panel": {"const": "cycle"}}}, "then": {"required": ["cycle_id"]}},
                    {"if": {"properties": {"status": {"const": "completed"}}}, "then": {"required": ["phase", "regime"]}}]}},
    },
    "allOf": [{"if": {"properties": {"label_mode": {"const": "asof"}}}, "then": {"properties": {
        "reviewer": {"required": ["no_future_prices_for_asof"], "properties": {"no_future_prices_for_asof": {"const": True}}}}}}],
    "description": "Two reviews and their verified manifest must match identities, frozen rubric, sampling and every case. "
                   "Completed states must satisfy labels.reference_state; actual exposure and independence require semantic checks and human process.",
}


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("review must contain finite JSON values") from exc


def _identity(value):
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def validate_review_exposure(row):
    """Check actual extra exposure and pending context in every review entry."""
    start = iso_date(row.get("information_start"), "information_start")
    end = iso_date(row.get("information_end"), "information_end")
    if row.get("request_context") is not None and type(row["request_context"]) is not bool:
        raise ValueError("request_context must be boolean")
    if row.get("request_context") is True and row.get("status") == "completed":
        raise ValueError("pending request_context cannot be completed")
    exposures = row.get("additional_exposure", [])
    if not isinstance(exposures, list):
        raise ValueError("additional_exposure must be a list of actual intervals")
    for exposure in exposures:
        if not isinstance(exposure, dict):
            raise ValueError("additional exposure requires recorded bounds")
        exposed_start = iso_date(exposure.get("information_start"), "additional information_start")
        exposed_end = iso_date(exposure.get("information_end"), "additional information_end")
        if not start <= exposed_start <= exposed_end <= end:
            raise ValueError("additional exposure lies outside registered actual information bounds; re-register context")


def _reviewer(review, mode):
    person = review.get("reviewer")
    if not isinstance(person, dict) or not isinstance(person.get("name"), str) or not person["name"].strip():
        raise ValueError("agreement requires named independent human reviewers")
    for field in ("independent_human", "no_strategy_outputs", "blind_review",
                  "no_other_review_seen", "independent_of_other_review"):
        if person.get(field) is not True:
            raise ValueError(f"reviewer must explicitly attest {field}")
    if mode == "asof" and person.get("no_future_prices_for_asof") is not True:
        raise ValueError("asof reviewer must attest no_future_prices_for_asof")
    if person.get("type", "human") != "human" or person.get("independence", "external_human") != "external_human":
        raise ValueError("LLM, rule or nonhuman reviews are not independent human evidence")
    try:
        timestamp = datetime.fromisoformat(person.get("reviewed_at", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("reviewed_at must be an ISO timestamp") from exc
    if timestamp.utcoffset() is None:
        raise ValueError("reviewed_at must include timezone")
    ident = person.get("identity_id")
    if ident is not None and (not isinstance(ident, str) or not ident.strip()):
        raise ValueError("identity_id must be a nonempty traceable human identity")
    return person


def _records(review, mode, source, people):
    rows = review.get("labels")
    if not isinstance(rows, list):
        raise ValueError("review requires a labels list")
    result, dates = {}, set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("review case requires a nonempty id")
        ident = row["id"]
        day = iso_date(row.get("as_of"), "as_of")
        if ident in result or day in dates:
            raise ValueError("duplicate case id or target date")
        if any(row.get(key) != source[key] for key in ("market", "symbol", "timeframe")) or row.get("label_mode") != mode:
            raise ValueError("case source or label_mode mismatch")
        if row.get("role") not in ("development", "holdout", "diagnostic"):
            raise ValueError("case requires its registered role")
        group_key(row)
        start = iso_date(row.get("information_start"), "information_start")
        end = iso_date(row.get("information_end"), "information_end")
        if not start <= day <= end or (mode == "asof" and end != day):
            raise ValueError("invalid actual information bounds or asof future exposure")
        if row.get("status") not in STATUSES:
            raise ValueError("review status must be blank, completed, unresolved or excluded")
        # Actual supplementary exposure cannot be omitted from registered bounds.
        # Requests alone are permitted, but remain unresolved until re-packaged.
        validate_review_exposure(row)
        if row["status"] == "completed":
            reference_state(row, mode)
            if row.get("ambiguity") not in (None, False, "", "none"):
                resolver = row.get("resolver")
                if (row.get("ambiguity_resolved") is not True or not isinstance(resolver, str)
                        or not resolver.strip() or not isinstance(row.get("resolution_rationale"), str)
                        or not row["resolution_rationale"].strip()):
                    raise ValueError("completed ambiguity requires independent resolution and rationale")
                if _identity(resolver) in people:
                    raise ValueError("adjudicator must be independent of both reviewers")
        result[ident] = row
        dates.add(day)
    return result


def _coverage(rows, mode):
    counts = Counter(row["status"] for row in rows.values())
    completed = [row for row in rows.values() if row["status"] == "completed"]
    return {"total": len(rows), "statuses": {status: counts[status] for status in STATUSES},
            "completed_fraction": len(completed) / len(rows) if rows else None,
            "unknown_or_unresolved": sum(row["status"] == "unresolved" or row.get("phase") in ("?", "unknown")
                                         for row in rows.values()),
            "direction_undetermined_completed": sum(reference_state(row, mode).endswith("/undetermined")
                                                       for row in completed),
            "no_observable_phase_completed": sum(row.get("phase") == "no_observable_phase" for row in completed),
            "request_context": sum(row.get("request_context") is True for row in rows.values())}


def _measure(pairs, classes, include_kappa):
    matrix = {left: {right: 0 for right in classes} for left in classes}
    for left, right in pairs:
        matrix[left][right] += 1
    n = len(pairs)
    matches = sum(left == right for left, right in pairs)
    a, b = Counter(left for left, _ in pairs), Counter(right for _, right in pairs)
    observed = matches / n if n else None
    result = {"support": n, "agreements": matches, "rate": observed, "confusion": matrix,
              "marginals": {"review_a": {state: a[state] for state in classes},
                            "review_b": {state: b[state] for state in classes}}}
    if include_kappa:
        expected = sum(a[state] * b[state] for state in classes) / n ** 2 if n else None
        result["cohens_kappa"] = ((observed - expected) / (1 - expected)
                                   if expected is not None and expected < 1 else None)
        result["chance_agreement"] = expected
        result["kappa_status"] = "unavailable_no_common_completed" if not n else (
            "undefined_constant_marginals" if expected == 1 else "descriptive_only")
    return result


def agreement_report(review_a, review_b, *, manifest=None, include_kappa=True):
    """Validate two raw review dicts and return JSON-compatible descriptive counts.

    Both reviews must cover the SAME registered case set (unreviewed cases stay
    blank). Require named, distinct humans with explicit blind_review,
    no_other_review_seen and independent_of_other_review attestations. The
    caller supplies a file-verified manifest, or both reviews must carry identical
    nonempty sampling metadata. A pack ID alone does not establish authenticity.
    Rates use only dates completed by BOTH reviewers; all other statuses are
    counted separately, never converted into a negative class. Rows are equally
    weighted descriptive dates, not statistically independent cycles. Inputs are
    never changed. ValueError means the comparison is invalid, not low agreement.
    """
    if not isinstance(review_a, dict) or not isinstance(review_b, dict):
        raise ValueError("reviews must be JSON objects")
    if type(include_kappa) is not bool:
        raise ValueError("include_kappa must be boolean")
    _canonical([review_a, review_b, manifest])
    mode = review_a.get("label_mode")
    classes = joint_classes(mode)
    if review_b.get("label_mode") != mode:
        raise ValueError("review label_mode mismatch; never mix asof and retrospective")
    source = review_a.get("source_identity")
    if (not isinstance(source, dict) or source.get("market") not in MARKETS or
            not isinstance(source.get("symbol"), str) or not source["symbol"].strip() or source.get("timeframe") != "1d"):
        raise ValueError("review requires exact market/symbol/timeframe source identity")
    pack_id = review_a.get("pack_id")
    if not isinstance(pack_id, str) or not pack_id.strip() or pack_id != review_b.get("pack_id"):
        raise ValueError("review pack identity mismatch")
    if source != review_b.get("source_identity"):
        raise ValueError("review source identity mismatch")
    for field in FROZEN_FIELDS:
        if review_a.get(field) != review_b.get(field):
            raise ValueError(f"review frozen {field} mismatch")
    sampling = manifest.get("sampling") if isinstance(manifest, dict) else review_a.get("sampling")
    if not isinstance(sampling, dict) or not sampling:
        raise ValueError("verified manifest or explicit sampling metadata is required")
    for review in (review_a, review_b):
        if "sampling" in review and review["sampling"] != sampling:
            raise ValueError("review sampling mismatch")
    if manifest is None and review_b.get("sampling") != sampling:
        raise ValueError("review sampling mismatch")
    if manifest is not None:
        if (not isinstance(manifest, dict) or not isinstance(manifest.get("settings"), dict)
                or manifest.get("pack_id") != pack_id or manifest.get("source") != source
                or manifest.get("settings", {}).get("label_mode") != mode):
            raise ValueError("manifest pack/source/label_mode mismatch")
        for field in FROZEN_FIELDS:
            required = field == "rubric_identity" or field in review_a or field in review_b
            if required and field in manifest and any(review.get(field) != manifest[field] for review in (review_a, review_b)):
                raise ValueError(f"review frozen {field} mismatch with manifest")
    a_person, b_person = _reviewer(review_a, mode), _reviewer(review_b, mode)
    people = {_identity(a_person["name"]), _identity(b_person["name"])}
    if len(people) != 2 or (a_person.get("identity_id") is not None and b_person.get("identity_id") is not None
                           and _identity(a_person["identity_id"]) == _identity(b_person["identity_id"])):
        raise ValueError("same author cannot independently review their own labels")
    a_rows = _records(review_a, mode, source, people)
    b_rows = _records(review_b, mode, source, people)
    if manifest is not None:
        from .cycles import validate_cycle_reviewer
        validate_cycle_reviewer(manifest, review_a)
        validate_cycle_reviewer(manifest, review_b)
    if a_rows.keys() != b_rows.keys():
        raise ValueError("reviews must contain the same complete registered case set")
    registered = None
    if manifest is not None:
        if not isinstance(manifest.get("cases"), list):
            raise ValueError("manifest requires registered cases")
        if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in manifest["cases"]):
            raise ValueError("manifest requires valid registered case ids")
        registered = {row["id"]: row for row in manifest["cases"]}
        if len(registered) != len(manifest["cases"]) or registered.keys() != a_rows.keys():
            raise ValueError("reviews differ from registered manifest cases")
    for ident, row in a_rows.items():
        for other in (b_rows[ident], *([registered[ident]] if registered is not None else [])):
            if any(row.get(key) != other.get(key) for key in CASE_FIELDS):
                raise ValueError("review changed registered case/source/date/information bounds")
    common = sorted((ident for ident in a_rows if a_rows[ident]["status"] == b_rows[ident]["status"] == "completed"),
                    key=lambda ident: (a_rows[ident]["as_of"], ident))
    joint_pairs, letter_pairs, direction_pairs, disagreements = [], [], [], []
    for ident in common:
        left, right = reference_state(a_rows[ident], mode), reference_state(b_rows[ident], mode)
        joint_pairs.append((left, right))
        # Explicit negative remains its own category in both projections.
        left_letter, left_direction = left.split("/") if "/" in left else (left, left)
        right_letter, right_direction = right.split("/") if "/" in right else (right, right)
        letter_pairs.append((left_letter, right_letter))
        direction_pairs.append((left_direction, right_direction))
        if left != right:
            disagreements.append({"id": ident, "as_of": a_rows[ident]["as_of"], "role": a_rows[ident]["role"],
                                  "panel": a_rows[ident]["panel"], "review_a": left, "review_b": right,
                                  "letter_disagreement": left_letter != right_letter,
                                  "direction_disagreement": left_direction != right_direction})
    negatives = ["no_observable_phase"] if mode == "asof" else []
    direction_classes = ["accumulation", "distribution"] + (["undetermined"] if mode == "asof" else []) + negatives
    status_confusion = {left: {right: 0 for right in STATUSES} for left in STATUSES}
    for ident in a_rows:
        status_confusion[a_rows[ident]["status"]][b_rows[ident]["status"]] += 1
    return {"schema_version": 1, "status": "DESCRIPTIVE_AGREEMENT" if common else "NO_COMMON_COMPLETED",
            "pack_id": pack_id, "label_mode": mode, "source_identity": source,
            "frozen_identity": {field: review_a[field] for field in FROZEN_FIELDS if field in review_a},
            "manifest_supplied": manifest is not None, "reviewers": {"review_a": a_person["name"], "review_b": b_person["name"]},
            "sampling": sampling, "registered_cases": len(a_rows), "common_completed": len(common),
            "common_completed_by_role": {role: sum(a_rows[ident]["role"] == role for ident in common)
                                         for role in ("development", "holdout", "diagnostic")},
            "common_completed_by_panel": {panel: sum(a_rows[ident]["panel"] == panel for ident in common)
                                          for panel in ("cycle", "window")},
            "common_completed_dates": [a_rows[ident]["as_of"] for ident in common],
            "coverage": {"review_a": _coverage(a_rows, mode), "review_b": _coverage(b_rows, mode)},
            "status_confusion": status_confusion,
            "letter": _measure(letter_pairs, list("ABCDE") + negatives, include_kappa),
            "direction": _measure(direction_pairs, direction_classes, include_kappa),
            "joint": _measure(joint_pairs, list(classes), include_kappa), "disagreements": disagreements,
            "not_common_completed": [{"id": ident, "as_of": a_rows[ident]["as_of"],
                                       "review_a_status": a_rows[ident]["status"], "review_b_status": b_rows[ident]["status"]}
                                      for ident in sorted(a_rows) if ident not in common],
            "inference_limit": "Nominal human agreement is not truth or classifier accuracy. Equal date counts do not imply independent cycles. "
                               "Kappa is descriptive and unstable with low support, class imbalance or prevalence. Identity and blindness are human attestations; "
                               "this pure function does not verify chart files, actual human independence or register exposure. Preserve both original blind reviews before adjudication."}
