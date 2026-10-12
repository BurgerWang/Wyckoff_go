"""Registered sparse windows from date metadata, never prices or predictions.

This creates window-panel geometry only. Human cycle bounds are supplied by the
caller and kept as exclusions, not inferred from machine intervals. A complete
schedule does not establish cycle/class support, statistical power or promotion.
The labelled-holdout-v1 validation policy and old annotation packs are untouched.
"""
import hashlib
import json
from bisect import bisect_left
from collections import Counter

from .labels import iso_date

VERSION = "sparse-window-v2"
ROLES = ("development", "holdout")
ORDERS = ("seeded_hash_year_round_robin", "chronological")


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _settings(settings):
    required = {"label_mode", "seed", "context_bars", "block_bars", "dates_per_block",
                "development_blocks", "holdout_blocks", "development_start",
                "development_end", "holdout_start", "holdout_end"}
    optional = {"version", "order", "warmup_bars"}
    if not isinstance(settings, dict) or required - settings.keys() or settings.keys() - required - optional:
        raise ValueError("sparse settings require exact registered metadata fields")
    result = {"version": VERSION, "order": ORDERS[0], "warmup_bars": 1, **settings}
    if result["version"] != VERSION or result["order"] not in ORDERS:
        raise ValueError("unsupported sparse sampling version or order")
    if result["label_mode"] not in ("asof", "retrospective"):
        raise ValueError("label_mode must be asof or retrospective")
    for key in ("seed", "context_bars", "block_bars", "dates_per_block",
                "development_blocks", "holdout_blocks", "warmup_bars"):
        lower = 0 if key in ("seed", "development_blocks", "holdout_blocks") else 1
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < lower:
            raise ValueError(f"{key} must be an integer >= {lower}")
    if result["dates_per_block"] > result["block_bars"]:
        raise ValueError("dates_per_block exceeds block_bars")
    for role in ROLES:
        for bound in ("start", "end"):
            key = f"{role}_{bound}"
            result[key] = iso_date(result[key], key)
        if result[f"{role}_start"] >= result[f"{role}_end"]:
            raise ValueError(f"{role}_start must precede {role}_end")
    if result["development_end"] > result["holdout_start"]:
        raise ValueError("development and holdout partitions overlap or reverse chronology")
    return result


def _segments(dates, continuous_segments):
    lookup = {day: index for index, day in enumerate(dates)}
    normalized, memberships = [], [None] * len(dates)
    for segment in continuous_segments:
        start, end = iso_date(segment["start"]), iso_date(segment["end"])
        if start not in lookup or end not in lookup or start > end:
            raise ValueError("continuous segment bounds must be actual ordered source dates")
        left, right = lookup[start], lookup[end]
        if normalized and left != normalized[-1]["end_index"] + 1:
            raise ValueError("continuous segments must partition source dates in order")
        if not normalized and left != 0:
            raise ValueError("continuous segments must cover all source dates")
        bars = right - left + 1
        if "bars" in segment and (isinstance(segment["bars"], bool)
                                  or not isinstance(segment["bars"], int) or segment["bars"] != bars):
            raise ValueError("continuous segment bar count mismatch")
        index = len(normalized)
        memberships[left:right + 1] = [index] * bars
        normalized.append({"start": start, "end": end, "bars": bars,
                           "start_index": left, "end_index": right})
    if any(index is None for index in memberships):
        raise ValueError("continuous segments must cover all source dates")
    return normalized, memberships, lookup


def _intervals(intervals, kind):
    result = []
    for interval in intervals:
        if isinstance(interval, dict):
            start, end = interval["information_start"], interval["information_end"]
        else:
            start, end = interval
        start, end = iso_date(start), iso_date(end)
        if start > end:
            raise ValueError(f"{kind} interval is reversed")
        result.append({"information_start": start, "information_end": end})
    return sorted(result, key=lambda item: (item["information_start"], item["information_end"]))


def _cycles(bounds, settings, lookup, memberships):
    if not isinstance(bounds, (list, tuple)):
        raise ValueError("human cycle bounds must be a list of complete metadata objects")
    result, seen = [], set()
    for bound in bounds:
        if not isinstance(bound, dict):
            raise ValueError("human cycle requires complete metadata object")
        ident, role = bound.get("cycle_id"), bound.get("role")
        if not isinstance(ident, str) or not ident.strip() or ident in seen or role not in ROLES:
            raise ValueError("human cycle requires unique cycle_id and registered role")
        seen.add(ident)
        try:
            start, end = iso_date(bound["information_start"]), iso_date(bound["information_end"])
        except KeyError as exc:
            raise ValueError("human cycle requires complete information bounds") from exc
        if start not in lookup or end not in lookup or start > end:
            raise ValueError("human cycle bounds must be actual ordered source dates")
        if memberships[lookup[start]] != memberships[lookup[end]]:
            raise ValueError("human cycle information crosses a continuous segment")
        if not settings[f"{role}_start"] <= start <= end < settings[f"{role}_end"]:
            raise ValueError("human cycle information crosses its registered partition")
        result.append({"cycle_id": ident, "role": role,
                       "information_start": start, "information_end": end})
    return sorted(result, key=lambda item: (item["role"], item["cycle_id"]))


def _overlap(start, end, interval):
    return start <= interval["information_end"] and interval["information_start"] <= end


def _targets(width, count, seed, role, day):
    # Endpoints keep the full registered background exposed for multi-date
    # windows. Interior dates use disjoint bins; no target date is duplicated.
    if count == 1:
        return [width - 1]
    if count == 2:
        return [0, width - 1]
    interior = []
    for index in range(count - 2):
        lo = 1 + (width - 2) * index // (count - 2)
        hi = 1 + (width - 2) * (index + 1) // (count - 2)
        offset = int(_digest([VERSION, "target", seed, role, day, index]), 16) % (hi - lo)
        interior.append(lo + offset)
    return [0, *interior, width - 1]


def _maximum_schedule(candidates, seed, role):
    """Exact maximum cardinality; seeded hash sum breaks only cardinality ties."""
    ordered = sorted(candidates, key=lambda item: (item["information_end"], item["information_start"], item["start"]))
    ends = [item["information_end"] for item in ordered]
    scores, previous, taken = [(0, 0)], [], []
    for index, block in enumerate(ordered):
        predecessor = bisect_left(ends, block["information_start"], 0, index)
        count, weight = scores[predecessor]
        rank = int(_digest([VERSION, "schedule", seed, role, block["start"], block["end"]]), 16)
        include = (count + 1, weight + rank)
        take = include > scores[-1]
        scores.append(include if take else scores[-1])
        previous.append(predecessor)
        taken.append(take)
    schedule, cursor = [], len(ordered)
    while cursor:
        if taken[cursor - 1]:
            schedule.append(ordered[cursor - 1])
            cursor = previous[cursor - 1]
        else:
            cursor -= 1
    return list(reversed(schedule))


def _order(schedule, settings, role):
    if settings["order"] == "chronological":
        return sorted(schedule, key=lambda block: block["start"])
    grouped = {}
    for block in schedule:
        grouped.setdefault(block["start"][:4], []).append(block)
    seed = settings["seed"]
    years = sorted(grouped, key=lambda year: _digest([VERSION, "year", seed, role, year]))
    for year in years:
        grouped[year].sort(key=lambda block: _digest([VERSION, "order", seed, role, block["start"], block["end"]]))
    return [grouped[year][index] for index in range(max((len(group) for group in grouped.values()), default=0))
            for year in years if index < len(grouped[year])]


def sample_sparse_windows(dates, continuous_segments, settings, *, diagnostic_intervals=(),
                          consumed_intervals=(), human_cycle_bounds=()):
    """Return a deterministic, unlabelled window schedule and exact capacity.

    ``dates`` is the verified, strictly increasing source Date column. Segments
    partition that column with inclusive start/end (and optional bar count).
    Settings freeze both half-open development/holdout partitions, requests,
    label mode, seed, order, background context, target block width and dates per
    block. ``warmup_bars`` (default 1) is metadata supplied by the caller.
    Diagnostic/consumed bounds are inclusive pairs or information-bound dicts;
    all are excluded from both roles. Human cycle bounds require cycle_id, role
    and full information_start/end on actual source dates within one segment and
    one declared partition. Invalid bounds fail; they are never silently clipped.

    Each ASOF case exposes exactly context_bars ending on its target date.
    Retrospective cases expose the registered block plus earlier context. Blocks
    contain exact per-case bounds and the union of all actually exposed charts.
    Capacity is exact for this registered candidate family, not a guarantee for
    a future protocol with larger human-required context. Count changes preserve
    a schedule prefix; source dates outside the partitions cannot select targets.
    No label, strategy, score, holdout reservation or filesystem operation occurs.
    """
    settings = _settings(settings)
    dates = [iso_date(day) for day in dates]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("source dates must be unique and strictly increasing")
    segments, memberships, lookup = _segments(dates, continuous_segments)
    exclusions = {"known_diagnostic_information_overlap": _intervals(diagnostic_intervals, "diagnostic"),
                  "consumed_holdout_information_overlap": _intervals(consumed_intervals, "consumed")}
    cycles = _cycles(human_cycle_bounds, settings, lookup, memberships)
    exclusions["human_cycle_information_overlap"] = cycles
    identity = {"version": VERSION, "settings": settings,
                "source_metadata": {"dates_sha256": _digest(dates), "bars": len(dates),
                    "first": dates[0] if dates else None, "last": dates[-1] if dates else None,
                    "continuous_segments": [{key: value for key, value in segment.items() if not key.endswith("_index")}
                                            for segment in segments]},
                "exclusions": exclusions}
    chosen, support = [], {}
    width, context = settings["block_bars"], settings["context_bars"]
    for role in ROLES:
        candidates, rejected = [], Counter()
        partition_start, partition_end = settings[f"{role}_start"], settings[f"{role}_end"]
        last_partition_index = bisect_left(dates, partition_end) - 1
        for start, day in enumerate(dates):
            if not partition_start <= day < partition_end:
                continue
            end, earliest_context = start + width - 1, start - context + 1
            if end > last_partition_index:
                rejected["insufficient_target_bars_in_partition"] += 1
                continue
            segment = segments[memberships[start]]
            if end > segment["end_index"]:
                rejected["target_crosses_continuous_segment"] += 1
                continue
            offsets = _targets(width, settings["dates_per_block"], settings["seed"], role, day)
            if settings["label_mode"] == "asof":
                cases = [{"as_of": dates[start + offset],
                          "information_start": dates[start + offset - context + 1]
                              if start + offset - context + 1 >= 0 else None,
                          "information_end": dates[start + offset]} for offset in offsets]
                first = start + offsets[0] - context + 1
            else:
                first = earliest_context
                cases = [{"as_of": dates[start + offset],
                          "information_start": dates[first] if first >= 0 else None,
                          "information_end": dates[end]} for offset in offsets]
            if first < segment["start_index"] or start + offsets[0] < segment["start_index"] + settings["warmup_bars"] - 1:
                rejected["insufficient_context_or_segment_warmup"] += 1
                continue
            information_start, information_end = cases[0]["information_start"], cases[-1]["information_end"]
            if not partition_start <= information_start <= information_end < partition_end:
                rejected["information_crosses_partition_cutoff"] += 1
                continue
            reasons = [reason for reason, intervals in exclusions.items()
                       if any(_overlap(information_start, information_end, interval) for interval in intervals)]
            if reasons:
                rejected.update(reasons)
                continue
            block = {"role": role, "panel": "window", "start": day, "end": dates[end],
                     "dates": [case["as_of"] for case in cases], "exposure_start": information_start,
                     "information_start": information_start, "information_end": information_end,
                     "exposure_bars": lookup[information_end] - first + 1,
                     "cases": cases, "block_id": f"sparse-window-{role}-{day}-{dates[end]}"}
            candidates.append(block)
        ordered = _order(_maximum_schedule(candidates, settings["seed"], role), settings, role)
        requested = settings[f"{role}_blocks"]
        selected = ordered[:requested]
        status = "COMPLETE" if len(selected) == requested else "PARTIAL" if selected else "INSUFFICIENT"
        support[role] = {"requested_blocks": requested, "selected_blocks": len(selected),
                         "max_disjoint_capacity": len(ordered), "available_blocks": len(candidates),
                         "status": status, "rejection_counts": dict(sorted(rejected.items())),
                         "reasons": [] if status == "COMPLETE" else ["fewer_than_requested_disjoint_windows"],
                         "selected_years": sorted({block["start"][:4] for block in selected})}
        chosen.extend(selected)
    incomplete = any(item["status"] != "COMPLETE" for item in support.values())
    return {"version": VERSION, "identity": identity, "sampling_id": _digest(identity),
            "status": "PARTIAL" if incomplete and chosen else "INSUFFICIENT" if incomplete else "COMPLETE",
            "metadata_only": True, "panel": "window", "contexts_may_overlap": False,
            "blocks_are_not_independent_cycles": True, "class_support_known": False,
            "cycle_support_qualified": False, "promotion_eligible": False,
            "partition_end_exclusive": True, "support": support, "blocks": chosen,
            "diagnostic_reasons": (["incomplete_window_request"] if incomplete else []) +
                ["window_only_metadata_no_human_cycle_or_class_qualification"],
            "limitations": "Exact disjoint capacity for registered window exposures only. "
                "Nonoverlap does not prove independence or class/cycle support. "
                "Human-required context extensions need a new identity and capacity check; "
                "no labelled-holdout-v1 threshold or eligibility is changed."}
