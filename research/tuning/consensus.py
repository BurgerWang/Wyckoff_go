"""Pure reconciliation of registered human reviews, without supplying labels.

The caller verifies files and preserves both original blind reviews. This module
only carries their existing answers or an explicitly supplied human adjudication
into a complete merged review; it does not certify humans or reference truth.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime

from .label_quality import (CASE_FIELDS, FROZEN_FIELDS, STATUSES, _canonical,
                            _identity, _records, agreement_report,
                            validate_review_exposure)
from .labels import reference_state

VERSION = "independent-review-consensus-v1"


def _adjudicator(review, original_people, mode):
    person = review.get("reviewer")
    if not isinstance(person, dict) or not isinstance(person.get("name"), str) or not person["name"].strip():
        raise ValueError("adjudication requires a named independent human")
    if person.get("type", "human") != "human" or person.get("independence", "external_human") != "external_human":
        raise ValueError("adjudicator must be an independent human")
    for field in ("independent_human", "no_strategy_outputs"):
        if person.get(field) is not True:
            raise ValueError(f"adjudicator must explicitly attest {field}")
    if mode == "asof" and person.get("no_future_prices_for_asof") is not True:
        raise ValueError("asof adjudicator must attest no_future_prices_for_asof")
    try:
        timestamp = datetime.fromisoformat(person.get("reviewed_at", ""))
    except (ValueError, TypeError) as exc:
        raise ValueError("adjudicator reviewed_at must be an ISO timestamp") from exc
    if timestamp.utcoffset() is None:
        raise ValueError("adjudicator reviewed_at must include timezone")
    ident = person.get("identity_id")
    if ident is not None and (not isinstance(ident, str) or not ident.strip()):
        raise ValueError("adjudicator identity_id must be a nonempty human identity")
    for original in original_people:
        if (_identity(person["name"]) == _identity(original["name"])
                or (ident is not None and original.get("identity_id") is not None
                    and _identity(ident) == _identity(original["identity_id"]))):
            raise ValueError("adjudicator must be independent of both original reviewers")
    return person


def _adjudication_rows(manifest, review_a, review_b, adjudication):
    if not isinstance(adjudication, dict):
        raise ValueError("adjudication must be a complete review object")
    _canonical(adjudication)
    for field in ("pack_id", "source_identity", "label_mode", *FROZEN_FIELDS):
        if adjudication.get(field) != review_a.get(field):
            raise ValueError(f"adjudication registered {field} mismatch")
    if "sampling" in adjudication and adjudication["sampling"] != manifest["sampling"]:
        raise ValueError("adjudication sampling mismatch")
    person = _adjudicator(adjudication, (review_a["reviewer"], review_b["reviewer"]), review_a["label_mode"])
    if datetime.fromisoformat(person["reviewed_at"]) < max(datetime.fromisoformat(review["reviewer"]["reviewed_at"]) for review in (review_a, review_b)):
        raise ValueError("adjudication cannot precede either original review")
    people = {_identity(review_a["reviewer"]["name"]), _identity(review_b["reviewer"]["name"])}
    rows = _records(adjudication, review_a["label_mode"], review_a["source_identity"], people)
    registered = {row["id"]: row for row in manifest["cases"]}
    if rows.keys() != registered.keys():
        raise ValueError("adjudication must contain the complete registered case set")
    for ident, row in rows.items():
        if any(row.get(key) != registered[ident].get(key) for key in CASE_FIELDS):
            raise ValueError("adjudication changed registered case or information bounds")
        if row["status"] == "completed":
            if (row.get("ambiguity_resolved") is not True or row.get("resolver") != person["name"]
                    or not isinstance(row.get("resolution_rationale"), str)
                    or not row["resolution_rationale"].strip()):
                raise ValueError("completed adjudication requires its named resolver and resolution rationale")
    return rows, person


def _preserve_context(row, originals):
    # Requests cannot be silently withdrawn by a merge or adjudication. A new
    # registered context and new reviews are needed to resolve missing data.
    row["request_context"] = any(original.get("request_context") is True for original in originals)
    exposures, seen = [], set()
    for original in originals:
        for exposure in original.get("additional_exposure", []):
            key = _canonical(exposure)
            if key not in seen:
                exposures.append(deepcopy(exposure))
                seen.add(key)
    row["additional_exposure"] = exposures
    validate_review_exposure(row)


def build_consensus(manifest, review_a, review_b, adjudication=None):
    """Return all registered cases, preserving A's actual reviewer identity.

    Only joint agreement between two completed answers is adopted automatically.
    Other cases remain unavailable unless the supplied independent third human
    explicitly adjudicates them. Blank adjudication entries mean no intervention.
    No input mutation, file access, strategy evaluation or human answer generation
    occurs here. The merged review is derived evidence, not a third blind review.
    """
    if not isinstance(manifest, dict):
        raise ValueError("consensus requires a registered manifest object")
    if any(isinstance(review, dict) and "consensus" in review for review in (review_a, review_b)):
        raise ValueError("derived consensus cannot replace an original blind review")
    report = agreement_report(review_a, review_b, manifest=manifest)
    a_rows = {row["id"]: row for row in review_a["labels"]}
    b_rows = {row["id"]: row for row in review_b["labels"]}
    judge_rows, judge = ({}, None) if adjudication is None else _adjudication_rows(manifest, review_a, review_b, adjudication)
    mode = review_a["label_mode"]
    labels, decisions = [], []
    for case in manifest["cases"]:
        ident = case["id"]
        left, right, judged = a_rows[ident], b_rows[ident], judge_rows.get(ident)
        agreed = (left["status"] == right["status"] == "completed"
                  and reference_state(left, mode) == reference_state(right, mode))
        row = deepcopy(left)
        route = "joint_agreement" if agreed else "not_common_completed"
        if not agreed:
            status = ("excluded" if "excluded" in (left["status"], right["status"]) else
                      "blank" if left["status"] == right["status"] == "blank" else "unresolved")
            row.update(status=status, phase=None, regime=None)
            if left["status"] == right["status"] == "completed":
                route = "joint_disagreement"
        if judged is not None and judged["status"] != "blank":
            if judged["status"] == "completed" and left["status"] == right["status"] == "blank":
                raise ValueError("adjudication cannot complete a case without original human review")
            row = deepcopy(judged)
            if row["status"] != "completed":
                row.update(phase=None, regime=None)
            else:
                # An adjudication is a resolved intervention even when the
                # supplied human did not add an optional ambiguity description.
                if judged.get("ambiguity") in (None, False, "", "none"):
                    row["ambiguity"] = "independent_review_adjudication"
            route = "human_adjudication"
        originals = [left, right] + ([judged] if judged is not None else [])
        _preserve_context(row, originals)
        labels.append(row)
        decisions.append({"id": ident, "route": route, "review_a_status": left["status"],
                          "review_b_status": right["status"], "adjudication_status": judged["status"] if judged else None,
                          "original_notes": {"review_a": left.get("notes", ""), "review_b": right.get("notes", ""),
                                             **({"adjudication": judged.get("notes", "")} if judged else {})}})
    result = deepcopy(review_a)
    result["labels"] = labels
    counts = Counter(row["status"] for row in labels)
    result["consensus"] = {"version": VERSION,
        "reviewers": {"review_a": deepcopy(review_a["reviewer"]), "review_b": deepcopy(review_b["reviewer"]),
                      "adjudicator": deepcopy(judge)},
        "original_coverage": report["coverage"], "original_agreement_status": report["status"],
        "original_common_completed": report["common_completed"],
        "merged_status_counts": {status: counts[status] for status in STATUSES}, "case_decisions": decisions,
        "limitation": "Derived reconciliation of supplied human records; not a third blind review, certified human independence, truth or classifier accuracy. Caller must preserve and verify original artifacts."}
    return result
