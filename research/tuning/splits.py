"""Conservative chronological whole-group folds with label-information purging."""
from collections import defaultdict

from .labels import group_key, group_intervals, iso_date, joint_classes, reference_state

# Previously inspected case spans cannot become a blind holdout through a new
# label author, source or cycle identifier. Bounds cover the diagnosed range
# through retirement, not only the exact event day. This registry is an
# exclusion policy, never an independent human cycle definition.
KNOWN_DIAGNOSTIC_INTERVALS = {
    ("us_equities", "AAPL"): (("2008-09-08", "2009-04-17"),
                               ("2017-11-13", "2018-02-02")),
    ("binance_spot", "DOGEUSDT"): (("2020-07-27", "2020-10-06"),
                                    ("2025-02-07", "2025-06-15"),
                                    ("2026-02-20", "2026-07-22")),
    ("binance_spot", "BTCUSDT"): (("2018-08-14", "2018-12-26"),
                                   ("2018-12-08", "2019-04-17"),
                                   ("2026-02-08", "2026-04-29")),
    ("binance_spot", "ETHUSDT"): (("2025-05-13", "2025-08-05"),
                                   ("2026-02-06", "2026-04-29")),
}


def _overlaps(left, right):
    return left[0] <= right[1] and right[0] <= left[1]


def _dates(groups):
    return sorted(label["as_of"] for group in groups for label in group["labels"])


def _support(groups, mode):
    labels = [label for group in groups for label in group["labels"]]
    states = {reference_state(label, mode) for label in labels}
    panels = {label["panel"] for label in labels}
    return {"missing_classes": [state for state in joint_classes(mode) if state not in states],
            "missing_panels": [panel for panel in ("cycle", "window") if panel not in panels],
            "cycles": sum(group["key"][0] == "cycle" for group in groups),
            "blocks": sum(group["key"][0] == "window" for group in groups)}


def _purge(train, test):
    # Entire training groups are removed whenever any of their price/label
    # information extends into the next evaluation information interval.
    boundary = min(group["information_start"] for group in test)
    kept = [group for group in train if group["information_end"] < boundary
            and group["end"] < min(item["start"] for item in test)]
    return kept, [group["key"] for group in train if group not in kept]


def _folds(groups, count):
    """Deterministic expanding prefixes and disjoint later test blocks."""
    if len(groups) < count + 1:
        return []
    boundaries = [len(groups) * i // (count + 1) for i in range(count + 2)]
    result = []
    for index in range(1, count + 1):
        test = groups[boundaries[index]:boundaries[index + 1]]
        train, purged = _purge(groups[:boundaries[index]], test)
        result.append({"train": train, "test": test, "purged": purged})
    return result


def build_split_plan(reference, *, holdout_start="2024-10-01", holdout_end="2026-09-30",
                     outer_folds=2, inner_folds=2, min_development_cycles=12, min_holdout_cycles=4):
    """Return READY or LABELS_INSUFFICIENT; never silently relax support rules.

    Holdout is half-open. Groups crossing either cutoff are excluded rather
    than split. Missing provenance, diagnostic groups and groups whose label
    information crosses cutoffs cannot qualify. Per-fold zero supports remain
    visible; the full fixed-class support checks apply to pooled validation.
    """
    start, end = iso_date(holdout_start), iso_date(holdout_end)
    if start >= end:
        raise ValueError("holdout_start must precede holdout_end")
    for name, value, minimum in (("outer_folds", outer_folds, 2), ("inner_folds", inner_folds, 2),
                                 ("min_development_cycles", min_development_cycles, 1),
                                 ("min_holdout_cycles", min_holdout_cycles, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    mode = reference["label_mode"]
    joint_classes(mode)
    reasons, excluded, grouped, seen = [], [], defaultdict(list), set()
    for label in reference["labels"]:
        day = iso_date(label["as_of"])
        if day in seen:
            raise ValueError("duplicate reference session")
        seen.add(day)
        reference_state(label, mode)
        grouped[group_key(label)].append(label)
    forbidden = list(KNOWN_DIAGNOSTIC_INTERVALS.get((reference.get("market"), reference.get("symbol")), ()))
    # Preserve contamination even when a diagnostic group is subsequently
    # excluded for missing provenance or shared data qualification.
    for labels in grouped.values():
        if any(label.get("role") == "diagnostic" or label.get("diagnostic_only") for label in labels):
            intervals = [(iso_date(label["information_start"]), iso_date(label["information_end"]))
                         for label in labels if label.get("information_start") and label.get("information_end")]
            if intervals:
                if len(intervals) == len(labels):
                    intervals = group_intervals(labels)
                forbidden.append((min(a for a, _ in intervals), max(b for _, b in intervals)))
    groups = []
    for key, labels in grouped.items():
        intervals = [(label.get("information_start"), label.get("information_end")) for label in labels]
        if any(label.get("provenance_issues") or None in interval
               for label, interval in zip(labels, intervals)):
            excluded.append({"group": list(key), "reason": "missing_information_or_role_provenance"})
            continue
        intervals = [(iso_date(a), iso_date(b)) for a, b in intervals]
        if any(not a <= label["as_of"] <= b or (mode == "asof" and b > label["as_of"])
               for label, (a, b) in zip(labels, intervals)):
            raise ValueError("invalid label information interval")
        intervals = group_intervals(labels)
        if any(label.get("role") not in ("development", "holdout", "diagnostic") for label in labels):
            raise ValueError("missing or invalid reference role")
        if any(label.get("qualification_issue") for label in labels):
            excluded.append({"group": list(key), "reason": "common_data_qualification_failed",
                             "qualification_issues": sorted({label["qualification_issue"] for label in labels
                                                             if label.get("qualification_issue")}),
                             "dates": [min(label["as_of"] for label in labels), max(label["as_of"] for label in labels)],
                             "information": [min(a for a, _ in intervals), max(b for _, b in intervals)]})
            continue
        groups.append({"key": key, "labels": labels,
                       "start": min(label["as_of"] for label in labels),
                       "end": max(label["as_of"] for label in labels),
                       "information_start": min(a for a, _ in intervals),
                       "information_end": max(b for _, b in intervals)})
    groups.sort(key=lambda group: (group["start"], group["end"], group["key"]))
    development, holdout = [], []
    for group in groups:
        info = group["information_start"], group["information_end"]
        roles = {label["role"] for label in group["labels"]}
        if "diagnostic" in roles or any(label.get("diagnostic_only") for label in group["labels"]):
            reason = "diagnostic_group"
        elif group["end"] < start and group["information_end"] < start:
            reason = None if roles == {"development"} else "role_does_not_match_development_dates"
            if reason is None:
                development.append(group)
        elif (start <= group["start"] <= group["end"] < end
              and start <= group["information_start"] <= group["information_end"] < end):
            if any(_overlaps(info, interval) for interval in forbidden):
                reason = "known_diagnostic_information_overlap"
            else:
                reason = None if roles == {"holdout"} else "role_does_not_match_holdout_dates"
            if reason is None:
                holdout.append(group)
        else:
            reason = "group_or_information_crosses_cutoff_or_outside_scope"
        if reason:
            excluded.append({"group": list(group["key"]), "reason": reason,
                             "dates": [group["start"], group["end"]], "information": list(info)})
    if holdout:
        development, purged = _purge(development, holdout)
        excluded.extend({"group": list(key), "reason": "holdout_information_purge"} for key in purged)
    supports = {"development": _support(development, mode), "holdout": _support(holdout, mode)}
    for partition, minimum in (("development", min_development_cycles), ("holdout", min_holdout_cycles)):
        support = supports[partition]
        if support["cycles"] < minimum:
            reasons.append(f"{partition}: requires >= {minimum} independent cycles; found {support['cycles']}")
        if support["missing_classes"]:
            reasons.append(f"{partition}: missing fixed-class support: {', '.join(support['missing_classes'])}")
        if support["missing_panels"]:
            reasons.append(f"{partition}: missing panels: {', '.join(support['missing_panels'])}")
    outer = []
    raw_outer = _folds(development, outer_folds)
    if len(raw_outer) != outer_folds:
        reasons.append("cannot construct requested chronological outer folds")
    pooled_outer = []
    for index, fold in enumerate(raw_outer):
        inner = _folds(fold["train"], inner_folds)
        if not fold["train"] or len(inner) != inner_folds or any(not item["train"] for item in inner):
            reasons.append(f"outer {index}: insufficient non-overlapping history for requested inner folds")
        pooled_inner = [group for item in inner for group in item["test"]]
        pooled_support = _support(pooled_inner, mode)
        if pooled_support["missing_classes"] or pooled_support["missing_panels"]:
            reasons.append(f"outer {index}: pooled inner validation lacks full fixed-class/panel support")
        pooled_outer.extend(fold["test"])
        outer.append({"train_dates": _dates(fold["train"]), "test_dates": _dates(fold["test"]),
                      "purged_groups": [list(key) for key in fold["purged"]],
                      "test_support": _support(fold["test"], mode), "pooled_inner_support": pooled_support,
                      "inner": [{"train_dates": _dates(item["train"]), "test_dates": _dates(item["test"]),
                                 "purged_groups": [list(key) for key in item["purged"]],
                                 "test_support": _support(item["test"], mode)} for item in inner]})
    supports["pooled_outer"] = _support(pooled_outer, mode)
    if supports["pooled_outer"]["missing_classes"] or supports["pooled_outer"]["missing_panels"]:
        reasons.append("pooled outer validation lacks full fixed-class/panel support")
    raw_final = _folds(development, inner_folds)
    final_support = _support([group for item in raw_final for group in item["test"]], mode)
    if len(raw_final) != inner_folds or any(not item["train"] for item in raw_final):
        reasons.append("insufficient non-overlapping history for final development inner folds")
    if final_support["missing_classes"] or final_support["missing_panels"]:
        reasons.append("pooled final inner validation lacks full fixed-class/panel support")
    supports["pooled_final_inner"] = final_support
    final_inner = [{"train_dates": _dates(item["train"]), "test_dates": _dates(item["test"]),
                    "purged_groups": [list(key) for key in item["purged"]],
                    "test_support": _support(item["test"], mode)} for item in raw_final]
    if excluded:
        # Exclusions are reported rather than automatically invalidating an
        # otherwise sufficient remaining independently labelled cohort.
        supports["excluded_groups"] = len(excluded)
    return {"status": "LABELS_INSUFFICIENT" if reasons else "READY", "reasons": reasons,
            "label_mode": mode, "market": reference.get("market"), "symbol": reference.get("symbol"),
            "holdout_start": start, "holdout_end": end, "holdout_end_exclusive": True,
            "development_dates": _dates(development), "holdout_dates": _dates(holdout),
            "outer": outer, "final_inner": final_inner, "support": supports, "excluded": excluded,
            "policy": {"crossing_groups": "exclude", "purge": "whole_group_information_overlap",
                       "panel_weights": {"cycle": 0.5, "window": 0.5},
                       "min_development_cycles": min_development_cycles,
                       "min_holdout_cycles": min_holdout_cycles}}
