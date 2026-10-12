"""Prediction-free feasibility of the frozen labelled evaluation protocol.

These helpers inspect only reference metadata. They never load prices, create
labels, evaluate a strategy, reserve a holdout or authorize an accuracy claim.
Passing this check still leaves sampling assumptions and score-dependent
degeneracy/stability checks to validation.
"""
from collections import defaultdict

from .labels import group_key, group_intervals, iso_date, joint_classes, reference_state
from .policy import VALIDATION_POLICY
from .splits import KNOWN_DIAGNOSTIC_INTERVALS

PANELS = ("cycle", "window")
MINIMUM = VALIDATION_POLICY["minimum_effective_clusters_per_panel"]


def _minimum(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < MINIMUM:
        raise ValueError(f"minimum must be an integer >= frozen policy minimum {MINIMUM}")
    return value


def reference_interval_clusters(reference):
    """Return deterministic full-information components, without reading states.

    A human-defined group is one dependency unit, including any gaps between
    its observations. Closed information intervals are merged transitively
    within each panel. Missing information never implies independence: raw
    groups are returned as a diagnostic fallback with ``complete=False``.
    """
    grouped, seen = defaultdict(list), set()
    mode = reference["label_mode"]
    joint_classes(mode)
    for label in reference["labels"]:
        day = iso_date(label["as_of"])
        if day in seen:
            raise ValueError("duplicate reference session")
        seen.add(day)
        grouped[group_key(label)].append(label)
    bounds, complete = {}, True
    for key, labels in grouped.items():
        intervals = []
        for label in labels:
            start, end = label.get("information_start"), label.get("information_end")
            if start is None or end is None:
                complete = False
                continue
            start, end, day = iso_date(start), iso_date(end), iso_date(label["as_of"])
            if not start <= day <= end or (mode == "asof" and end > day):
                raise ValueError("invalid reference information interval")
            intervals.append((start, end))
        if len(intervals) == len(labels):
            intervals = group_intervals(labels)
            bounds[key] = (min(a for a, _ in intervals), max(b for _, b in intervals))
    components = {panel: [] for panel in PANELS}
    if complete:
        for panel in PANELS:
            entries = sorted((a, b, key[1]) for key, (a, b) in bounds.items() if key[0] == panel)
            for start, end, ident in entries:
                if components[panel] and start <= components[panel][-1]["end"]:
                    component = components[panel][-1]
                    component["end"] = max(component["end"], end)
                    component["members"].append(ident)
                else:
                    components[panel].append({"start": start, "end": end, "members": [ident]})
        latest = {panel: "" for panel in PANELS}
        cross_panel = False
        for start, end, panel in sorted((a, b, key[0]) for key, (a, b) in bounds.items()):
            other = "window" if panel == "cycle" else "cycle"
            if latest[other] >= start:
                cross_panel = True
            latest[panel] = max(latest[panel], end)
    else:
        cross_panel = None
        for panel, ident in sorted(grouped):
            start, end = bounds.get((panel, ident), (None, None))
            components[panel].append({"start": start, "end": end, "members": [ident]})
    return {"groups": dict(grouped), "bounds": bounds, "components": components,
            "complete": complete, "cross_panel_overlap": cross_panel}


def _contaminated(reference, geometry):
    forbidden = [(iso_date(a), iso_date(b)) for a, b in reference.get("diagnostic_intervals", ())]
    forbidden.extend(KNOWN_DIAGNOSTIC_INTERVALS.get((reference.get("market"), reference.get("symbol")), ()))
    for key, labels in geometry["groups"].items():
        if any(label.get("role") == "diagnostic" or label.get("diagnostic_only") for label in labels):
            # Missing information still taints the group's known observed dates.
            forbidden.append(geometry["bounds"].get(key, (
                min(iso_date(label["as_of"]) for label in labels),
                max(iso_date(label["as_of"]) for label in labels))))
    result = []
    for key, labels in sorted(geometry["groups"].items()):
        reasons = []
        if any(label.get("role") == "diagnostic" or label.get("diagnostic_only") for label in labels):
            reasons.append("diagnostic_group")
        if any(label.get("qualification_issue") for label in labels):
            reasons.append("common_data_qualification_failed")
        if any(label.get("provenance_issues") for label in labels):
            reasons.append("missing_information_or_role_provenance")
        roles = {label.get("role") for label in labels}
        if not roles <= {"development", "holdout", "diagnostic"}:
            reasons.append("missing_or_invalid_reference_role")
        if len(roles) > 1:
            reasons.append("mixed_group_roles")
        interval = geometry["bounds"].get(key)
        if "holdout" in roles and interval and any(interval[0] <= b and a <= interval[1] for a, b in forbidden):
            reasons.append("known_diagnostic_information_overlap")
        if reasons:
            result.append({"group": list(key), "reasons": reasons})
    return result


def _capacity(reference, *, minimum, states_known):
    minimum = _minimum(minimum)
    classes = tuple(joint_classes(reference["label_mode"]))
    geometry = reference_interval_clusters(reference)
    groups, components = geometry["groups"], geometry["components"]
    raw_counts = {panel: sum(key[0] == panel for key in groups) for panel in PANELS}
    counts = {panel: len(components[panel]) for panel in PANELS}
    raw_support = {state: {panel: set() for panel in PANELS} for state in classes}
    support = {state: {panel: 0 for panel in PANELS} for state in classes}
    if states_known:
        for key, labels in groups.items():
            for label in labels:
                raw_support[reference_state(label, reference["label_mode"])][key[0]].add(key[1])
        for panel in PANELS:
            for component in components[panel]:
                members = set(component["members"])
                for state in classes:
                    support[state][panel] += bool(members & raw_support[state][panel])
    missing = [state for state in classes if not sum(support[state].values())]
    missing_panels = [panel for panel in PANELS if not raw_counts[panel]]
    contaminated = _contaminated(reference, geometry)
    reasons = []
    if not geometry["complete"]:
        reasons.append("missing_reference_information_intervals")
    if any(counts[panel] < minimum for panel in PANELS):
        reasons.append(f"fewer_than_{minimum}_effective_clusters_per_panel")
    if missing_panels:
        reasons.append("missing_sampling_panels")
    if geometry["cross_panel_overlap"]:
        reasons.append("cross_panel_information_overlap")
    if not states_known:
        reasons.append("unlabelled_metadata_only")
    if missing:
        reasons.append("missing_fixed_class_support")
    if contaminated or reference.get("diagnostic_only"):
        reasons.append("contaminated_or_unqualified_reference_groups")
    unit_eligible = geometry["complete"] and all(counts[panel] >= minimum for panel in PANELS)
    geometry_eligible = unit_eligible and not geometry["cross_panel_overlap"] and not contaminated
    eligible = not reasons
    return {"status": "REFERENCE_CAPACITY_READY" if eligible else "INSUFFICIENT_REFERENCE_CAPACITY",
            "eligible": eligible, "geometry_eligible": geometry_eligible,
            "metadata_only": not states_known, "class_support_known": states_known,
            "label_mode": reference["label_mode"], "market": reference.get("market"),
            "symbol": reference.get("symbol"), "policy_version": VALIDATION_POLICY["version"],
            "minimum_units_per_panel": minimum, "unit_count_eligible": unit_eligible,
            "raw_groups": raw_counts, "effective_clusters": counts,
            "independent_units": counts,
            "overlap_clusters": {panel: [item["members"] for item in components[panel]] for panel in PANELS},
            "cluster_information_intervals": components,
            "reference_intervals_complete": geometry["complete"],
            "cross_panel_information_overlap": geometry["cross_panel_overlap"],
            "missing_panels": missing_panels, "classes": list(classes), "missing_classes": missing,
            "class_unit_support": support,
            "raw_class_unit_support": {state: {panel: len(ids) for panel, ids in panels.items()}
                                       for state, panels in raw_support.items()},
            "contaminated_groups": contaminated, "diagnostic_reasons": reasons,
            "limitations": "Metadata eligibility only; no predictions, scores, power calculation or inference. "
                "Full information overlap is merged transitively; v1 rejects cross-panel overlap. "
                "Nonoverlap does not prove independence or representative sampling."}


def evaluate_reference_capacity(reference, *, minimum=MINIMUM):
    """Check one caller-selected labelled partition before any score or claim.

    The caller owns chronological cutoffs and passes exactly that partition's
    labels. No missing dates, intervals, class states or panels are invented.
    ``minimum`` may strengthen, but cannot relax, the frozen v1 threshold.
    """
    return _capacity(reference, minimum=minimum, states_known=True)


def information_capacity_upper_bound(available_bars, minimum_information_bars, *, minimum=MINIMUM):
    """Pigeonhole bound for disjoint exposures inside an actual bar partition.

    Both counts must be verified bar counts, never elapsed calendar dates. An
    upper bound below the minimum proves impossibility; exceeding it does not
    establish feasible sampling or class support.
    """
    minimum = _minimum(minimum)
    for name, value, lower in (("available_bars", available_bars, 0),
                               ("minimum_information_bars", minimum_information_bars, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    maximum = available_bars // minimum_information_bars
    return {"available_bars": available_bars, "minimum_information_bars": minimum_information_bars,
            "maximum_disjoint_information_clusters": maximum, "minimum_units_per_panel": minimum,
            "provably_insufficient": maximum < minimum,
            "scope": "Upper bound for exposures wholly inside this bar partition; no class/panel support assumption."}


def evaluate_annotation_capacity(manifest, *, role="holdout", minimum=MINIMUM, holdout_bar_count=None):
    """Inspect a blank annotation manifest without opening charts or prices.

    Geometry uses the registered window or human cycle panel. Blank phases never become genuine
    fixed-class support; even sufficient geometry cannot qualify this pack.
    The optional holdout count must come from separately verified metadata.
    """
    if role not in ("development", "holdout"):
        raise ValueError("role must be development or holdout")
    settings, source = manifest["settings"], manifest["source"]
    panel = manifest.get("sampling", {}).get("panel", "window")
    from .cycles import VERSION as CYCLE_VERSION, reference_case
    if panel not in PANELS or (panel == "cycle" and manifest.get("version") != CYCLE_VERSION):
        raise ValueError("cycle samples require a registered cycle pack")
    cases = [reference_case(manifest, case) for case in manifest["cases"] if case["role"] == role]
    if any(case.get("panel") != panel or (panel == "window" and case.get("cycle_id") is not None) for case in cases):
        raise ValueError("annotation manifest requires its registered sampling panel")
    reference = {"label_mode": settings["label_mode"], "labels": cases,
                 "market": source["market"], "symbol": source["symbol"],
                 "diagnostic_intervals": manifest.get("diagnostic_intervals", [])}
    result = _capacity(reference, minimum=minimum, states_known=False)
    result.update(window_only=panel == "window", registered_panel=panel, role=role, pack_id=manifest.get("pack_id"))
    cutoff_start, cutoff_end = iso_date(settings["holdout_start"]), iso_date(settings["holdout_end"])
    if cutoff_start >= cutoff_end:
        raise ValueError("holdout_start must precede holdout_end")
    crossing = [list(key) for key, (start, end) in reference_interval_clusters(reference)["bounds"].items()
                if (role == "holdout" and not cutoff_start <= start <= end < cutoff_end)
                or (role == "development" and (end >= cutoff_start or
                    (settings.get("development_start") is not None and start < iso_date(settings["development_start"]))))]
    if crossing:
        result["diagnostic_reasons"].append("information_crosses_partition_cutoff")
        result["geometry_eligible"] = False
    result["cutoff_crossing_groups"] = crossing
    if holdout_bar_count is not None and panel == "window":
        if role != "holdout":
            raise ValueError("holdout_bar_count applies only to the holdout partition")
        result["information_capacity_upper_bound"] = information_capacity_upper_bound(
            holdout_bar_count, settings["context_bars"], minimum=minimum)
    elif holdout_bar_count is not None:
        result["information_capacity_upper_bound"] = {"status": "not_applicable_variable_cycle_information"}
    return result


def evaluate_annotation_joint_capacity(manifests, *, role="holdout", minimum=MINIMUM,
                                       consumed_intervals=()):
    """Combine two caller-verified blank packs, without scores or file writes.

    The caller verifies the frozen charts/source and supplies current shared
    consumption read-only. Geometry can pass; unknown human classes cannot.
    """
    if role not in ("development", "holdout") or len(manifests) != 2:
        raise ValueError("joint capacity requires two packs and a registered role")
    from .cycles import reference_case
    first = manifests[0]
    source, settings = first["source"], first["settings"]
    partition = {key: iso_date(settings.get(key), key) for key in
                 ("development_start", "holdout_start", "holdout_end")}
    if not partition["development_start"] < partition["holdout_start"] < partition["holdout_end"]:
        raise ValueError("joint partition must be ordered and disjoint")
    if not first.get("rubric_identity") or not first.get("protocol_sha256"):
        raise ValueError("joint packs require a frozen rubric and protocol")
    cases, packs, panels, group_owners, forbidden, consumed = [], set(), set(), set(), [], []
    for manifest in manifests:
        if (manifest["source"] != source or manifest["settings"].get("label_mode") != settings["label_mode"]):
            raise ValueError("joint packs require the same exact source and label mode")
        if (manifest.get("rubric_identity") != first["rubric_identity"] or
                manifest.get("protocol_sha256") != first["protocol_sha256"]):
            raise ValueError("joint packs require the same frozen rubric and protocol")
        if any(manifest["settings"].get(key) != value for key, value in partition.items()):
            raise ValueError("joint packs require the same explicit partition")
        ident = manifest.get("pack_id")
        if not isinstance(ident, str) or not ident or ident in packs:
            raise ValueError("joint packs require distinct registered pack ids")
        packs.add(ident)
        panel = manifest.get("sampling", {}).get("panel")
        # Reuse single-panel/version checks before combining anything.
        evaluate_annotation_capacity(manifest, role=role, minimum=minimum)
        owned = set()
        for case in manifest["cases"]:
            if (case.get("panel") != panel or case.get("role") not in ("development", "holdout")
                    or any(case.get(key) != source[key] for key in ("market", "symbol", "timeframe"))
                    or case.get("label_mode") != settings["label_mode"]):
                raise ValueError("joint case differs from registered identity or role")
            owned.add(group_key(case))
            cases.append(reference_case(manifest, case))
        if owned & group_owners:
            raise ValueError("duplicate group id across joint packs")
        group_owners.update(owned)
        panels.add(panel)
        forbidden.extend(manifest.get("diagnostic_intervals", ()))
        snapshot = manifest.get("sampling_registration", manifest.get("exclusion_snapshot", {}))
        forbidden.extend(snapshot.get("diagnostic_intervals", ()))
        consumed.extend(snapshot.get("consumed_intervals", ()))
    if panels != set(PANELS):
        raise ValueError("joint packs require one human cycle panel and one window panel")
    reference = {"label_mode": settings["label_mode"], "market": source["market"],
                 "symbol": source["symbol"], "labels": cases}
    # Check all roles first: filtering cannot hide duplicate dates or mixed groups.
    geometry = reference_interval_clusters(reference)
    crossing = []
    for key, labels in sorted(geometry["groups"].items()):
        left, right = ((partition["development_start"], partition["holdout_start"])
                       if labels[0]["role"] == "development" else
                       (partition["holdout_start"], partition["holdout_end"]))
        start, end = geometry["bounds"][key]
        if len({label["role"] for label in labels}) != 1 or not left <= start <= end < right:
            crossing.append(list(key))
    selected = [case for case in cases if case["role"] == role]
    result = _capacity({**reference, "labels": selected, "diagnostic_intervals": forbidden},
                       minimum=minimum, states_known=False)
    consumed.extend(consumed_intervals)
    consumed = sorted(set((iso_date(a), iso_date(b)) for a, b in consumed))
    if any(a > b for a, b in consumed):
        raise ValueError("consumed information interval is reversed")
    selected_geometry = reference_interval_clusters({**reference, "labels": selected})
    overlaps = [list(key) for key, (a, b) in sorted(selected_geometry["bounds"].items())
                if any(a <= d and c <= b for c, d in consumed)]
    if overlaps:
        result["diagnostic_reasons"].append("known_consumed_information_overlap")
    if crossing:
        result["diagnostic_reasons"].append("information_crosses_partition_cutoff")
    result["geometry_eligible"] &= not overlaps and not crossing
    result.update(role=role, pack_ids=sorted(packs), partition=partition, cutoff_crossing_groups=crossing,
                  consumed_overlap_groups=overlaps, consumed_information_intervals=[list(x) for x in consumed],
                  rubric_identity=first["rubric_identity"], predictions_evaluated=False, holdout_consumed=False)
    return result
