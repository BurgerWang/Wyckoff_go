"""Pure validation statistics and preregistered engineering stability checks.

The percentile intervals are conditional on the labelled sampling panels and
assumed independent human-defined clusters. Twelve units in *each* panel is a
conservative engineering eligibility rule, not a power or coverage theorem.
No helper creates labels, changes the fixed class set, or selects a neighbor.
"""
from collections import defaultdict
from dataclasses import asdict
import math
import random
import statistics

from .labels import group_key, iso_date, joint_classes, reference_state
from .metrics import ABSTENTION
from .policy import VALIDATION_POLICY
from .preflight import evaluate_reference_capacity, reference_interval_clusters
from .space import search_space, validate_candidate

PANELS = ("cycle", "window")
MIN_INFERENCE_UNITS = VALIDATION_POLICY["minimum_effective_clusters_per_panel"]
MAX_STABILITY_DROP = VALIDATION_POLICY["maximum_development_regression"]
MIN_STABLE_NEIGHBOR_FRACTION = VALIDATION_POLICY["neighbor_acceptable_fraction"]


def _cases(metric):
    classes = tuple(joint_classes(metric["label_mode"]))
    if tuple(metric.get("classes", ())) != classes:
        raise ValueError("metric must retain the fixed label-mode classes")
    rows = {}
    for case in metric["cases"]:
        day = iso_date(case["as_of"])
        if day in rows:
            raise ValueError("duplicate metric session")
        if (case["panel"] not in PANELS or not isinstance(case["group_id"], str)
                or not case["group_id"].strip() or case["expected"] not in classes
                or case["predicted"] not in (*classes, ABSTENTION)):
            raise ValueError("invalid weighted metric case")
        weight = case.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("metric weights must be positive and finite")
        rows[day] = case
    groups = defaultdict(list)
    for day in sorted(rows):
        case = rows[day]
        groups[(case["panel"], case["group_id"])].append(case)
    counts = {panel: sum(key[0] == panel for key in groups) for panel in PANELS}
    for (panel, _), group in groups.items():
        weight = 0.5 / counts[panel] / len(group)
        if any(not math.isclose(case["weight"], weight, rel_tol=1e-9, abs_tol=1e-12) for case in group):
            raise ValueError("metric cases must use fixed 50/50 panel and equal-cluster weights")
    return classes, rows, groups, counts


def _paired(default_metric, candidate_metric):
    classes, defaults, groups, counts = _cases(default_metric)
    candidate_classes, candidates, candidate_groups, _ = _cases(candidate_metric)
    if classes != candidate_classes or defaults.keys() != candidates.keys():
        raise ValueError("paired metrics require identical label mode and dates")
    for day, case in defaults.items():
        other = candidates[day]
        if any(case[field] != other[field] for field in ("expected", "panel", "group_id")):
            raise ValueError("paired metrics require identical references, panels and groups")
    return classes, groups, candidate_groups, counts


def _group_confusions(groups, classes):
    """Each table has unit mass: a repeated draw keeps its own unit mass."""
    result = {panel: [] for panel in PANELS}
    indices = {state: i for i, state in enumerate((*classes, ABSTENTION))}
    for (panel, _), cases in sorted(groups.items()):
        table = [[0.0] * (len(classes) + 1) for _ in classes]
        for case in cases:
            table[indices[case["expected"]]][indices[case["predicted"]]] += 1 / len(cases)
        result[panel].append(table)
    return result


def _overlap_clusters(groups, classes, reference):
    """Merge overlapping full information intervals transitively within panels."""
    tables = _group_confusions(groups, classes)
    raw_keys = {panel: [key for key in sorted(groups) if key[0] == panel] for panel in PANELS}
    if reference is None:
        return tables, {panel: [[key[1]] for key in keys] for panel, keys in raw_keys.items()}, False
    if tuple(joint_classes(reference["label_mode"])) != tuple(classes):
        raise ValueError("bootstrap reference label mode mismatch")
    labels = {iso_date(label["as_of"]): label for label in reference["labels"]}
    expected_rows = {case["as_of"]: case for cases in groups.values() for case in cases}
    if len(labels) != len(reference["labels"]) or labels.keys() != expected_rows.keys():
        raise ValueError("bootstrap reference requires exactly the scored dates")
    for day, label in labels.items():
        key = group_key(label)
        case = expected_rows[day]
        if key != (case["panel"], case["group_id"]) or reference_state(label, reference["label_mode"]) != case["expected"]:
            raise ValueError("bootstrap reference groups or states mismatch")
    geometry = reference_interval_clusters(reference)
    merged_tables = {panel: [] for panel in PANELS}
    membership = {panel: [] for panel in PANELS}
    for panel, keys in raw_keys.items():
        indices = {key[1]: i for i, key in enumerate(keys)}
        for component in geometry["components"][panel]:
            table = [[0.0] * (len(classes) + 1) for _ in classes]
            for ident in component["members"]:
                for i, row in enumerate(tables[panel][indices[ident]]):
                    for j, value in enumerate(row):
                        table[i][j] += value
            merged_tables[panel].append(table)
            membership[panel].append(component["members"])
    return merged_tables, membership, geometry["complete"]


def _cross_panel_overlap(reference):
    """Use the same metadata dependency graph as the prediction-free gate."""
    return None if reference is None else reference_interval_clusters(reference)["cross_panel_overlap"]


def _sample_score(tables, draws, class_count):
    confusion = [[0.0] * (class_count + 1) for _ in range(class_count)]
    for panel in PANELS:
        panel_draws = draws[panel]
        if not panel_draws:
            continue
        # Index repetitions are deliberately not deduplicated or reweighted
        # according to the size of the originating cluster.
        # A merged cluster retains each constituent original-group mass. Use
        # a panel ratio to preserve those weights when cluster sizes differ.
        drawn_mass = sum(sum(sum(row) for row in tables[panel][index]) for index in panel_draws)
        weight = 0.5 / drawn_mass
        for index in panel_draws:
            table = tables[panel][index]
            for i, row in enumerate(table):
                for j, value in enumerate(row):
                    confusion[i][j] += weight * value
    macro = 0.0
    correct = mass = 0.0
    for i, row in enumerate(confusion):
        support = sum(row)
        predicted = sum(other[i] for other in confusion)
        true_positive = row[i]
        macro += 2 * true_positive / (support + predicted) if support + predicted else 0.0
        correct += true_positive
        mass += support
    return {"joint_macro_f1": macro / class_count if mass else None,
            "joint_agreement": correct / mass if mass else None}


def _percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def score_summaries(metric):
    """Weighted descriptive phase and direction errors, with explicit denominators.

    Phase-letter agreement includes abstentions as failures and the explicit
    no-observable class as its own letter. Direction errors only count opposite
    *known* directional sides; unknown and abstaining predictions are separately
    reported, never silently excluded from the reference-side denominator.
    """
    classes, rows, _, counts = _cases(metric)
    mass = sum(case["weight"] for case in rows.values())
    phase_correct = direction_mass = direction_wrong = direction_unknown = 0.0
    support = {state: {panel: set() for panel in PANELS} for state in classes}
    for case in rows.values():
        expected, predicted, weight = case["expected"], case["predicted"], case["weight"]
        phase_correct += weight * (expected.split("/")[0] == predicted.split("/")[0])
        support[expected][case["panel"]].add(case["group_id"])
        side = expected.split("/")[-1]
        if side in ("accumulation", "distribution"):
            direction_mass += weight
            predicted_side = predicted.split("/")[-1]
            if predicted_side in ("accumulation", "distribution"):
                direction_wrong += weight * (side != predicted_side)
            else:
                direction_unknown += weight
    return {"phase_letter_agreement": phase_correct / mass if mass else None,
            "phase_letter_denominator_weight": mass,
            "direction_error_rate": direction_wrong / direction_mass if direction_mass else None,
            "direction_unknown_or_abstention_rate": direction_unknown / direction_mass if direction_mass else None,
            "direction_denominator_weight": direction_mass,
            "independent_units": counts,
            "class_unit_support": {state: {panel: len(units) for panel, units in panels.items()}
                                   for state, panels in support.items()},
            "scope": "Descriptive only for the labelled samples and predeclared 50/50 panels."}


def paired_cluster_bootstrap(default_metric, candidate_metric, *, reference=None, seed=20260930, repeats=2000):
    """Resample independent cycle/window clusters within panels, paired by date.

    Every replicate retains the fixed 10/13-class macro denominator, including
    absent classes, and exactly half its mass in each represented panel. Both
    models receive identical draws; cycle and window draws are independent.
    Sparse data may receive a diagnostic interval, never inferential clearance.
    The original default is required; do not substitute a previous candidate.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("bootstrap repeats must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    classes, default_groups, candidate_groups, counts = _paired(default_metric, candidate_metric)
    default_tables, membership, intervals_complete = _overlap_clusters(default_groups, classes, reference)
    candidate_tables, _, _ = _overlap_clusters(candidate_groups, classes, reference)
    effective_counts = {panel: len(default_tables[panel]) for panel in PANELS}
    complete_draws = {panel: list(range(effective_counts[panel])) for panel in PANELS}
    default_score = _sample_score(default_tables, complete_draws, len(classes))
    candidate_score = _sample_score(candidate_tables, complete_draws, len(classes))
    differences = {key: candidate_score[key] - value if value is not None else None
                   for key, value in default_score.items()}
    missing = [state for state, panels in score_summaries(default_metric)["class_unit_support"].items()
               if not sum(panels.values())]
    units_eligible = intervals_complete and all(effective_counts[panel] >= MIN_INFERENCE_UNITS for panel in PANELS)
    cross_panel_overlap = _cross_panel_overlap(reference)
    diagnostic_reasons = []
    if not intervals_complete:
        diagnostic_reasons.append("missing_reference_information_intervals")
    if any(effective_counts[panel] < MIN_INFERENCE_UNITS for panel in PANELS):
        diagnostic_reasons.append("fewer_than_12_effective_clusters_per_panel")
    if cross_panel_overlap:
        diagnostic_reasons.append("cross_panel_information_overlap")
    if missing:
        diagnostic_reasons.append("missing_fixed_class_support")
    if default_metric["status"] != "complete_scope" or candidate_metric["status"] != "complete_scope":
        diagnostic_reasons.append("partial_or_diagnostic_reference_scope")
    if reference is not None:
        capacity = evaluate_reference_capacity(reference)
        diagnostic_reasons.extend(reason for reason in capacity["diagnostic_reasons"]
                                  if reason not in diagnostic_reasons)
    inferential = not diagnostic_reasons
    intervals = {key: None for key in differences}
    distribution = {key: None for key in differences}
    if all(counts.values()):
        rng = random.Random(seed)
        samples = {key: [] for key in differences}
        for _ in range(repeats):
            draws = {panel: [rng.randrange(effective_counts[panel]) for _ in range(effective_counts[panel])]
                     for panel in PANELS}
            baseline = _sample_score(default_tables, draws, len(classes))
            candidate = _sample_score(candidate_tables, draws, len(classes))
            for key in samples:
                samples[key].append(candidate[key] - baseline[key])
        distribution = {key: {"min": min(values), "max": max(values),
                              "std": statistics.pstdev(values), "unique_count": len(set(values)),
                              "degenerate": max(values) - min(values) <= 1e-12}
                        for key, values in samples.items()}
        if any(item["degenerate"] for item in distribution.values()):
            diagnostic_reasons.append("degenerate_resampling_distribution")
            inferential = False
        intervals = {key: {"lower": _percentile(values, 0.025), "upper": _percentile(values, 0.975),
                           "nominal_level": 0.95, "inferential": inferential,
                           "scope": "conditional_cluster_percentile" if inferential else "noninferential_diagnostic"}
                     for key, values in samples.items()}
    return {"status": "INSUFFICIENT_TEST_POWER" if not units_eligible or cross_panel_overlap else
                       "INFERENCE_ELIGIBLE" if inferential else "PARTIAL_SCOPE_DIAGNOSTIC",
            "seed": seed, "repeats": repeats, "classes": list(classes),
            "method": "paired_panel_stratified_overlap_cluster_ratio_percentile_bootstrap",
            "independent_units": effective_counts, "raw_groups": counts, "effective_clusters": effective_counts,
            "overlap_clusters": membership, "reference_intervals_complete": intervals_complete,
            "cross_panel_information_overlap": cross_panel_overlap, "diagnostic_reasons": diagnostic_reasons,
            "minimum_units_per_panel": MIN_INFERENCE_UNITS,
            "unit_count_eligible": units_eligible, "inference_eligible": inferential,
            "missing_classes": missing, "observed_differences": differences, "intervals": intervals,
            "resampling_distribution": distribution,
            "panel_weights": {panel: 0.5 for panel in PANELS},
            "descriptive_scope": "The labelled samples and predeclared equal-mass sampling panels only.",
            "inference_scope": "Conditional on representative sampling and exchangeable independent overlap-clusters within each panel.",
            "limitations": "Overlapping full information intervals are transitively merged within each panel; cross-panel overlap, missing intervals or degenerate resampling prohibit inference. Nonoverlap alone does not guarantee independence. Twelve effective clusters per panel is an engineering threshold, not proof of power or 95% coverage."}


def legal_neighbors(config, market):
    """All legal adjacent values in the frozen grid, changing exactly one field.

    Off-grid configurations are rejected; no nearest-value rounding is used.
    Only neighbors satisfying all cross-parameter constraints are returned.
    """
    params = asdict(config) if not isinstance(config, dict) else dict(config)
    params = asdict(validate_candidate(params, market))
    result = []
    for name, dimension in search_space(market).items():
        values = dimension.values()
        index = next(i for i, value in enumerate(values) if abs(params[name] - value) <= 1e-10)
        for direction in (-1, 1):
            adjacent = index + direction
            if not 0 <= adjacent < len(values):
                continue
            neighbor = {**params, name: values[adjacent]}
            try:
                valid = validate_candidate(neighbor, market)
            except ValueError:
                continue
            result.append({"parameter": name, "direction": direction, "config": asdict(valid)})
    return result


def _metric(item):
    return item["metric"] if "metric" in item else item


def _ce_regressions(default_metric, candidate_metric):
    _paired(default_metric, candidate_metric)
    regressions = []
    for state in default_metric["classes"]:
        baseline = default_metric["per_class"][state]
        if state.split("/")[0] not in ("C", "E") or not baseline["support"]:
            continue
        recall = candidate_metric["per_class"][state]["recall"]
        delta = recall - baseline["recall"] if recall is not None else None
        if delta is None or delta < -MAX_STABILITY_DROP - 1e-12:
            regressions.append({"class": state, "recall_difference": delta})
    return regressions


def summarize_stability(default_metric, candidate_metric, neighbor_metrics, *, folds=()):
    """Development-only gate; caller must evaluate *all* legal neighbors.

    The 90%/.05 thresholds are preregistered project engineering choices. For
    each supplied inner fold compare candidate and every neighbor to its
    ORIGINAL default. Unsupported fold classes are explicitly not constraints.
    Neighbor identities/completeness belong to the orchestrator, which knows
    the configuration and must reject omitted/duplicate neighbors.
    """
    _paired(default_metric, candidate_metric)
    neighbors = [_metric(item) for item in neighbor_metrics]
    details = []
    for index, neighbor in enumerate(neighbors):
        _paired(candidate_metric, neighbor)
        delta = neighbor["joint_macro_f1"] - candidate_metric["joint_macro_f1"]
        details.append({"index": index, "joint_macro_f1_difference": delta,
                        "stable": delta >= -MAX_STABILITY_DROP - 1e-12,
                        "ce_recall_regressions": _ce_regressions(default_metric, neighbor)})
    stable_fraction = sum(item["stable"] for item in details) / len(details) if details else None
    candidate_ce = _ce_regressions(default_metric, candidate_metric)
    fold_details = []
    for index, fold in enumerate(folds):
        original = fold["default_metric"]
        evaluations = [("candidate", fold["candidate_metric"])] + [
            (f"neighbor_{i}", _metric(metric)) for i, metric in enumerate(fold.get("neighbor_metrics", []))]
        checks = []
        for name, metric in evaluations:
            _paired(original, metric)
            delta = metric["joint_macro_f1"] - original["joint_macro_f1"]
            checks.append({"evaluation": name, "joint_macro_f1_difference": delta,
                           "f1_passed": delta >= -MAX_STABILITY_DROP - 1e-12,
                           "ce_recall_regressions": _ce_regressions(original, metric)})
        fold_details.append({"index": index, "checks": checks,
                             "unsupported_ce_classes": [state for state in original["classes"]
                                 if state.split("/")[0] in ("C", "E") and not original["per_class"][state]["support"]]})
    recall_passed = not candidate_ce and not any(item["ce_recall_regressions"] for item in details)
    folds_passed = all(check["f1_passed"] and not check["ce_recall_regressions"]
                       for fold in fold_details for check in fold["checks"])
    passed = bool(details) and stable_fraction >= MIN_STABLE_NEIGHBOR_FRACTION and recall_passed and folds_passed
    return {"status": "STABLE" if passed else "UNSTABLE" if details else "INSUFFICIENT_NEIGHBOR_EVIDENCE",
            "passed": passed, "scope": "development_only", "neighbors": details,
            "neighbor_count": len(details), "stable_fraction": stable_fraction,
            "minimum_stable_fraction": MIN_STABLE_NEIGHBOR_FRACTION, "maximum_drop": MAX_STABILITY_DROP,
            "candidate_ce_recall_regressions": candidate_ce, "folds": fold_details,
            "ce_recall_passed": recall_passed, "folds_passed": folds_passed,
            "limitations": "Engineering gate, not inferential evidence; caller must supply all legal neighbors and original-default fold scores."}
