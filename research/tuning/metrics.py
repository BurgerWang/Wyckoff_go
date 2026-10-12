"""Fixed-class, panel/group-balanced joint scoring without optimizer dependencies."""
from collections import Counter

from .labels import PHASES, directional_side, group_key, iso_date, joint_classes, reference_state

ABSTENTION = "abstention"
ABSTAIN_STATUSES = {"insufficient_data", "insufficient_evidence", "ambiguous", "missing"}


def prediction_state(prediction, label_mode):
    if not prediction:
        return ABSTENTION
    if (label_mode == "asof" and prediction.get("normal_no_observable") is True
            and prediction.get("status") in {"candidate", "insufficient_evidence", "normal", "no_observable_phase"}):
        return "no_observable_phase"
    if prediction.get("status") in ABSTAIN_STATUSES:
        return ABSTENTION
    phase = prediction.get("phase")
    if phase == "no_observable_phase":
        return "no_observable_phase" if label_mode == "asof" else ABSTENTION
    if phase not in PHASES:
        return ABSTENTION
    state = f"{phase}/{directional_side(prediction.get('regime'))}"
    return state if state in joint_classes(label_mode) else ABSTENTION


def _prediction_rows(rows):
    if isinstance(rows, dict) and "rows" in rows:
        rows = rows["rows"]
    predictions = {}
    for row in rows:
        day = row.get("as_of", row.get("Date"))
        # Production rows carry pandas timestamps; JSON replay carries ISO timestamps.
        if isinstance(day, str) and "T" in day:
            from datetime import datetime
            day = datetime.fromisoformat(day)
        day = iso_date(day, "prediction as_of")
        if day in predictions:
            raise ValueError("duplicate prediction session")
        prediction = dict(row["phase"] if isinstance(row.get("phase"), dict) else row)
        predictions[day] = prediction
    return predictions


def score(reference, rows):
    """All labelled dates remain in the denominator, including absent predictions.

    Both panels have predeclared 0.5 mass. A missing panel is reported as a
    partial diagnostic and its absent mass is not silently redistributed.
    Missing reference classes contribute zero to the fixed-class macro F1;
    their recall/F1 estimates are separately marked unavailable.
    """
    mode = reference["label_mode"]
    classes = joint_classes(mode)
    labels = reference["labels"]
    predictions = _prediction_rows(rows)
    dates = [x["as_of"] for x in labels]
    if len(set(dates)) != len(dates):
        raise ValueError("duplicate reference session")
    counts = Counter(group_key(label) for label in labels)
    panel_groups = Counter(panel for panel, _ in counts)
    confusion = {actual: {predicted: 0.0 for predicted in (*classes, ABSTENTION)} for actual in classes}
    cases = []
    for label in labels:
        key = group_key(label)
        weight = 0.5 / panel_groups[key[0]] / counts[key]
        actual = reference_state(label, mode)
        prediction = predictions.get(label["as_of"])
        predicted = prediction_state(prediction, mode)
        confusion[actual][predicted] += weight
        cases.append({"as_of": label["as_of"], "expected": actual, "predicted": predicted,
                      "weight": weight, "panel": key[0], "group_id": key[1]})
    per_class = {}
    for state in classes:
        tp = confusion[state][state]
        support = sum(confusion[state].values())
        predicted = sum(confusion[actual][state] for actual in classes)
        fixed_f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[state] = {"support": support, "predicted": predicted,
                            "precision": tp / predicted if predicted else 0.0,
                            "recall": tp / support if support else None,
                            "f1": fixed_f1 if support else None,
                            "fixed_macro_contribution": fixed_f1}
    missing_classes = [state for state in classes if not per_class[state]["support"]]
    missing_panels = [panel for panel in ("cycle", "window") if not panel_groups[panel]]
    total_weight = sum(case["weight"] for case in cases)
    correct = sum(case["weight"] for case in cases if case["expected"] == case["predicted"])
    covered = sum(case["weight"] for case in cases if case["predicted"] != ABSTENTION)
    def summary(subset):
        mass = sum(case["weight"] for case in subset)
        return {"points": len(subset), "weight": mass,
                "joint_agreement": sum(case["weight"] for case in subset
                                       if case["expected"] == case["predicted"]) / mass if mass else None,
                "coverage": sum(case["weight"] for case in subset
                                if case["predicted"] != ABSTENTION) / mass if mass else None}
    return {"status": "partial_scope_diagnostic" if (missing_classes or missing_panels
                                                      or any(label.get("diagnostic_only") or label.get("role") == "diagnostic"
                                                             for label in labels)) else "complete_scope",
            "label_mode": mode, "classes": list(classes),
            "joint_macro_f1": sum(x["fixed_macro_contribution"] for x in per_class.values()) / len(classes) if labels else None,
            "joint_agreement": correct / total_weight if total_weight else None,
            "coverage": covered / total_weight if total_weight else None,
            "abstention_rate": 1 - covered / total_weight if total_weight else None,
            "selective_agreement": correct / covered if covered else None,
            "per_class": per_class, "confusion": confusion, "missing_classes": missing_classes,
            "missing_panels": missing_panels, "represented_panel_weight": total_weight,
            "panel_weights": {"cycle": 0.5, "window": 0.5},
            "panels": {panel: summary([case for case in cases if case["panel"] == panel])
                       for panel in ("cycle", "window")},
            "groups": [{"panel": panel, "group_id": ident,
                        **summary([case for case in cases if (case["panel"], case["group_id"]) == (panel, ident)])}
                       for panel, ident in counts],
            "points": len(labels), "missing_prediction_dates": sorted(set(dates) - predictions.keys()),
            "cases": cases,
            "inference_limit": "Scores cover labelled samples only; synthetic fixtures are software checks, not market accuracy."}
