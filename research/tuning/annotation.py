"""Predeclared OHLCV-only review packs. This module never supplies human truth.

Random blocks belong to the window panel, not the independent cycle panel.
Independent cycles require separately acquired human descriptions and exact
phase dates, manually assigned cycle_id, role, market/source identity and the
full information_start/end used to identify the cycle. Do not rename sampled
blocks to cycle_id or infer cycles from strategy ranges. Human cycle labels
must satisfy labels.load_labels and the same chronological/purge policy.
"""
import hashlib
import html
import json
from datetime import datetime, timedelta
from pathlib import Path

from .data import load_history
from .labels import MARKETS, iso_date, joint_classes, reference_state
from .space import common_warmup
from .splits import KNOWN_DIAGNOSTIC_INTERVALS

VERSION = "blind-ohclv-window-v1"
SPARSE_PACK_VERSION = "blind-ohclv-window-v2"
IMMUTABLE = ("id", "market", "symbol", "timeframe", "as_of", "label_mode", "role",
             "panel", "block_id", "information_start", "information_end", "chart")
CASE_IDENTITY = (*IMMUTABLE, "cycle_id", "analysis_scale", "parent_cycle")
PROTOCOL = """Independent human OHLCV annotation protocol v1
Review only the assigned standalone chart, without strategy output or automated
phase suggestions. ASOF charts contain no later prices. Complete earlier dates
before later charts; prior exposure to later prices makes an ASOF case unresolved
or excluded. Record actual prior exposure rather than attesting falsely. The
reviewer must be a human independent of this classifier; rules and LLM answers
are not reference truth. Write status completed only after personally reviewing
that case, with an exact phase and direction allowed by the mode. No observable
phase is an explicit ASOF negative; blank and unknown are not negative labels.
Ambiguity or disagreement remains unresolved unless independently adjudicated;
record resolver and resolution rationale. Retrospective charts explicitly show
the later endpoint: these labels cannot be used as ASOF truth. Information bounds
are the actual chart bounds and must not be shortened in the returned review.
Random blocks are window-panel samples, never independently identified cycles.
Acquire the cycle panel separately with a human-assigned cycle_id, full exposed
information_start/end, exact phase dates, reviewer and genuine source provenance.
A sampler cannot guarantee phase coverage, independent cycles or eligibility.
"""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _bytes_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _bars(frame):
    return [{"date": index.date().isoformat(), **{name.lower(): float(row[name])
             for name in ("Open", "High", "Low", "Close", "Volume")}}
            for index, row in frame.iterrows()]


def _source(frame, item, history, market, symbol):
    if market not in MARKETS or frame.attrs.get("symbol") != symbol or frame.attrs.get("timeframe") != "1d":
        raise ValueError("source identity mismatch")
    return {"market": market, "symbol": symbol, "timeframe": "1d",
            "history": str(Path(history).resolve()), "attrs": dict(frame.attrs),
            "coverage": {key: item.get(key) for key in ("status", "first", "last", "bars", "span_years",
                          "continuous_segments", "missing_dates", "short_history_reason", "security_boundary")},
            "ohlcv_sha256": _digest(_bars(frame))}


def _overlap(left, right):
    return left[0] <= right[1] and right[0] <= left[1]


def _order_blocks(blocks, seed, role):
    # Count-independent order: each year gets one block per round. This is
    # stratified sampling, not an estimate of natural phase prevalence.
    by_year = {}
    for block in blocks:
        by_year.setdefault(block["dates"][0][:4], []).append(block)
    years = sorted(by_year, key=lambda year: _digest([VERSION, seed, role, year]))
    for year in years:
        by_year[year].sort(key=lambda block: _digest([VERSION, seed, role, block["start"], block["end"]]))
    return [by_year[year][rank] for rank in range(max((len(v) for v in by_year.values()), default=0))
            for year in years if rank < len(by_year[year])]


def _sample(frame, item, market, symbol, settings, registration=None):
    if settings.get("sampling_version") == "sparse-window-v2":
        from .sampling import sample_sparse_windows
        if not isinstance(registration, dict):
            raise ValueError("sparse pack requires frozen exclusions")
        sparse_settings = {key: settings[key] for key in (
            "label_mode", "seed", "context_bars", "block_bars", "dates_per_block",
            "development_blocks", "holdout_blocks", "development_start", "holdout_start", "holdout_end")}
        sparse_settings.update(development_end=settings["holdout_start"], warmup_bars=common_warmup(market))
        result = sample_sparse_windows(frame.index.strftime("%Y-%m-%d").tolist(),
            item["continuous_segments"], sparse_settings,
            diagnostic_intervals=registration["diagnostic_intervals"],
            consumed_intervals=registration["consumed_intervals"],
            human_cycle_bounds=registration["human_cycle_bounds"])
        return result["blocks"], result["support"], common_warmup(market)
    dates = [index.date().isoformat() for index in frame.index]
    lookup = {day: pos for pos, day in enumerate(dates)}
    warmup, context, width = common_warmup(market), settings["context_bars"], settings["block_bars"]
    forbidden = KNOWN_DIAGNOSTIC_INTERVALS.get((market, symbol), ())
    chosen, support = [], {}
    for role, requested in (("development", settings["development_blocks"]), ("holdout", settings["holdout_blocks"])):
        candidates, rejected = [], 0
        for segment in item["continuous_segments"]:
            left, right = lookup[segment["start"]], lookup[segment["end"]]
            eligible = [pos for pos in range(left, right + 1) if
                        (dates[pos] < settings["holdout_start"] if role == "development" else
                         settings["holdout_start"] <= dates[pos] < settings["holdout_end"])]
            if not eligible:
                continue
            # Holdout context itself starts inside the holdout. Development
            # context may reach its historical origin, never another segment.
            begin = max(left + warmup - 1, eligible[0] + context - 1)
            for start in range(begin, eligible[-1] - width + 2, width):
                end = start + width - 1
                exposure_start = start - context + 1
                interval = dates[exposure_start], dates[end]
                if role == "holdout" and any(_overlap(interval, other) for other in forbidden):
                    rejected += 1
                    continue
                # Target dates are separated across the fixed block; their
                # positions depend on the seed, not requested sample counts.
                targets = []
                count = settings["dates_per_block"]
                for bin_index in range(count):
                    lo, hi = width * bin_index // count, width * (bin_index + 1) // count
                    # Keep dates away from adjacent bin edges: with defaults
                    # each target is separated by at least six daily bars.
                    inset = (hi - lo) // 4
                    lo, hi = lo + inset, hi - inset
                    offset = int(_digest([VERSION, settings["seed"], role, dates[start], bin_index]), 16) % (hi - lo)
                    targets.append(dates[start + lo + offset])
                candidates.append({"role": role, "start": dates[start], "end": dates[end], "dates": targets,
                                   "exposure_start": dates[exposure_start]})
        ordered = _order_blocks(candidates, settings["seed"], role)
        selected = ordered[:requested]
        support[role] = {"requested_blocks": requested, "available_blocks": len(candidates),
                         "selected_blocks": len(selected), "diagnostic_context_rejected_blocks": rejected,
                         "selected_years": sorted({block["start"][:4] for block in selected})}
        chosen.extend(selected)
    return chosen, support, warmup


def _sampling_description(blocks, support, settings):
    sparse = settings.get("sampling_version") == "sparse-window-v2"
    return {"method": "sparse-window-v2" if sparse else "seeded_hash_year_round_robin_fixed_blocks",
            "panel": "window", "contexts_may_overlap": not sparse,
            "blocks_are_not_independent_cycles": True, "holdout_end_exclusive": True,
            "support": support, "blocks": [{**block, "block_id": block.get("block_id",
                f"window-{block['role']}-{block['start']}-{block['end']}")} for block in blocks]}


def _chart(case, bars):
    # A single-case payload is intentionally the only market data in the file.
    payload = _json({"case": case, "bars": bars}).replace("<", "\\u003c")
    label = "ASOF: no later prices" if case["label_mode"] == "asof" else "RETROSPECTIVE: later prices shown"
    scale = (f"<p>登记尺度：{html.escape(case['analysis_scale'])}；"
             f"父周期：{html.escape(case.get('parent_cycle') or '未登记')}。</p>"
             if case.get("panel") == "cycle" else "")
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(case['id'])}</title>
<style>body{{font:16px system-ui;margin:24px;background:#f8fafc;color:#172033}}canvas{{width:100%;height:560px;background:white}}.warning{{color:#a32519;font-weight:bold}}pre{{white-space:pre-wrap}}</style>
<h1>{html.escape(case['symbol'])} · {case['as_of']}</h1><p class="warning">{label}</p>
<p>Exact source: {html.escape(case['market'])} / {html.escape(case['symbol'])} / 1d.
Exposed: {case['information_start']} through {case['information_end']}. Target: {case['as_of']}.</p>{scale}
<p>OHLC candles and source volume only. Hover for values. Complete this case before opening later dates. Strategy outputs are unavailable.</p>
<canvas id="chart" width="1400" height="560"></canvas><pre id="detail"></pre>
<script id="annotation-data" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('annotation-data').textContent),b=data.bars,c=document.getElementById('chart'),x=c.getContext('2d');
const pad=65,w=c.width-pad-25,h=350,low=Math.min(...b.map(v=>v.low)),high=Math.max(...b.map(v=>v.high)),range=high-low||1;
const step=w/b.length,y=v=>25+h-(v-low)/range*h,maxVolume=Math.max(...b.map(v=>v.volume))||1;
x.font='14px sans-serif';x.fillStyle='#172033';x.fillText('Price',5,20);x.fillText('Volume',5,435);
for(let k=0;k<=4;k++){{const value=low+range*k/4;x.fillStyle='#526078';x.fillText(value.toFixed(2),2,y(value));x.strokeStyle='#e4e8ee';x.beginPath();x.moveTo(pad,y(value));x.lineTo(c.width-25,y(value));x.stroke();}}
b.forEach((v,i)=>{{const px=pad+step*(i+.5),color=v.close>=v.open?'#087c6c':'#c53c43';x.strokeStyle=color;x.fillStyle=color;x.beginPath();x.moveTo(px,y(v.high));x.lineTo(px,y(v.low));x.stroke();x.fillRect(px-step*.3,Math.min(y(v.open),y(v.close)),Math.max(1,step*.6),Math.max(1,Math.abs(y(v.open)-y(v.close))));const vh=90*v.volume/maxVolume;x.fillRect(px-step*.3,525-vh,Math.max(1,step*.6),vh);if(v.date===data.case.as_of){{x.strokeStyle='#3854bd';x.beginPath();x.moveTo(px,15);x.lineTo(px,530);x.stroke();}}if(i===0||i===b.length-1||i%Math.max(1,Math.floor(b.length/6))===0){{x.fillStyle='#526078';x.fillText(v.date,Math.max(pad,px-35),550);}}}});
c.onmousemove=e=>{{const i=Math.max(0,Math.min(b.length-1,Math.floor(((e.clientX-c.getBoundingClientRect().left)*c.width/c.getBoundingClientRect().width-pad)/step)));document.getElementById('detail').textContent=JSON.stringify(b[i],null,2);}};
</script></html>'''


def prepare_annotation_pack(history, market, symbol, output, *, label_mode="asof", seed=20260930,
                            development_blocks=24, holdout_blocks=12, dates_per_block=3,
                            context_bars=100, block_bars=30, holdout_start="2024-10-01", holdout_end="2026-09-30",
                            rubric_path=None, sampling_version="window-v1", development_start=None,
                            human_cycle_bounds=()):
    """Create immutable sampling manifest, blank human review and offline charts.

    Return manifest; an existing identical pack is verified and reused without
    replacing human review. Different settings/history fail rather than overwrite.
    """
    joint_classes(label_mode)
    start, end = iso_date(holdout_start), iso_date(holdout_end)
    if start >= end:
        raise ValueError("holdout_start must precede holdout_end")
    settings = {"label_mode": label_mode, "seed": seed, "development_blocks": development_blocks,
                "holdout_blocks": holdout_blocks, "dates_per_block": dates_per_block,
                "context_bars": context_bars, "block_bars": block_bars,
                "holdout_start": start, "holdout_end": end}
    for key in ("seed", "development_blocks", "holdout_blocks", "dates_per_block", "context_bars", "block_bars"):
        minimum = 0 if key in ("seed", "development_blocks", "holdout_blocks") else 1
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if dates_per_block > block_bars or context_bars > 5000:
        raise ValueError("dates_per_block exceeds block_bars or context_bars exceeds 5000")
    if sampling_version not in ("window-v1", "sparse-window-v2"):
        raise ValueError("unsupported annotation sampling version")
    if sampling_version == "window-v1" and (development_start is not None or human_cycle_bounds):
        raise ValueError("development_start and cycle bounds require sparse-window-v2")
    frame, item = load_history(history, market, symbol)
    source = _source(frame, item, history, market, symbol)
    registration = None
    if sampling_version == "sparse-window-v2":
        from .holdout import REGISTRY, _load
        exposure = _load(REGISTRY / market / symbol / "uses.json", market, symbol)
        settings.update(sampling_version=sampling_version,
                        development_start=iso_date(development_start or min(item["first"],
                            (datetime.fromisoformat(start) - timedelta(days=1)).date().isoformat())))
        registration = {"diagnostic_intervals": [list(x) for x in KNOWN_DIAGNOSTIC_INTERVALS.get((market, symbol), ())],
                        "consumed_intervals": [list(x) for use in exposure["uses"] for x in use["information_intervals"]],
                        "human_cycle_bounds": list(human_cycle_bounds)}
    blocks, support, warmup = _sample(frame, item, market, symbol, settings, registration)
    identity = {"version": SPARSE_PACK_VERSION if registration is not None else VERSION, "source": source, "settings": settings,
                "protocol_sha256": hashlib.sha256(PROTOCOL.encode()).hexdigest(),
                "diagnostic_intervals": [list(interval) for interval in KNOWN_DIAGNOSTIC_INTERVALS.get((market, symbol), ())]}
    if registration is not None:
        identity["sampling_registration"] = registration
    rubric = Path(rubric_path).read_bytes() if rubric_path is not None else None
    if rubric is not None:
        if not rubric.strip():
            raise ValueError("annotation rubric cannot be empty")
        identity["rubric_identity"] = {"version": "phase-rubric-v1", "sha256": hashlib.sha256(rubric).hexdigest()}
    pack_id = _digest(identity)
    output = Path(output).resolve()
    if output.exists() and any(path.name != ".lock" for path in output.iterdir()):
        manifest = _load_manifest(output)
        if manifest["pack_id"] != pack_id:
            raise ValueError("existing annotation pack has different sampling/source identity")
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    if rubric is not None:
        (output / "rubric.md").write_bytes(rubric)
    cases, charts = [], {}
    for block in blocks:
        block_id = block.get("block_id", f"window-{block['role']}-{block['start']}-{block['end']}")
        for day in block["dates"]:
            observed = frame.loc[:day].tail(context_bars) if label_mode == "asof" else frame.loc[block["exposure_start"]:block["end"]]
            bars = _bars(observed)
            if registration is not None:
                registered_case = next(case for case in block["cases"] if case["as_of"] == day)
                if (bars[0]["date"], bars[-1]["date"]) != (registered_case["information_start"], registered_case["information_end"]):
                    raise ValueError("actual chart differs from sparse registered exposure")
            case_id = f"{market}-{symbol}-{label_mode}-{day}"
            case = {"id": case_id, "market": market, "symbol": symbol, "timeframe": "1d",
                    "as_of": day, "label_mode": label_mode, "role": block["role"], "panel": "window",
                    "block_id": block_id, "information_start": bars[0]["date"],
                    "information_end": bars[-1]["date"], "chart": f"charts/{case_id}.html"}
            chart = _chart(case, bars)
            path = output / case["chart"]
            path.parent.mkdir(exist_ok=True)
            path.write_text(chart, encoding="utf-8")
            charts[case_id] = {"sha256": _bytes_digest(path), "bars_sha256": _digest(bars), "bars": len(bars)}
            cases.append(case)
    insufficient = any(value["selected_blocks"] < value["requested_blocks"] for value in support.values())
    manifest = {**identity, "pack_id": pack_id, "schema_version": 1,
                "status": "SHORT_HISTORY" if symbol == "SPCX" or item["status"] == "short_history" else
                          "INSUFFICIENT_WINDOWS" if insufficient else "AWAITING_HUMAN_REVIEW",
                "common_warmup_bars": warmup, "sampling": _sampling_description(blocks, support, settings),
                "cases": sorted(cases, key=lambda case: case["as_of"]), "charts": charts}
    _write_json(output / "manifest.json", manifest)
    (output / "protocol.txt").write_text(PROTOCOL, encoding="utf-8")
    template = {"schema_version": 1, "pack_id": pack_id, "source_identity": source,
                "label_mode": label_mode, "reviewer": {"name": "", "reviewed_at": "",
                    "independent_human": False, "no_strategy_outputs": False,
                    "no_future_prices_for_asof": False, "blind_review": False,
                    "no_other_review_seen": False, "independent_of_other_review": False},
                "labels": [{**case, "status": "blank", "phase": None, "regime": None,
                            "request_context": False, "additional_exposure": [],
                            "ambiguity": None, "ambiguity_resolved": False,
                            "resolver": "", "resolution_rationale": "", "notes": ""}
                           for case in manifest["cases"]]}
    if rubric is not None:
        template["rubric_identity"] = identity["rubric_identity"]
    _write_json(output / "annotation.json", template)
    return manifest


def _load_manifest(pack):
    pack = Path(pack).resolve()
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    from .cycles import VERSION as CYCLE_VERSION, load_cycle_manifest
    if manifest.get("version") == CYCLE_VERSION:
        return load_cycle_manifest(pack)
    identity = {key: manifest[key] for key in ("version", "source", "settings", "protocol_sha256", "diagnostic_intervals")}
    if "rubric_identity" in manifest:
        rubric_identity = manifest["rubric_identity"]
        if (rubric_identity.get("version") != "phase-rubric-v1"
                or _bytes_digest(pack / "rubric.md") != rubric_identity.get("sha256")):
            raise ValueError("registered annotation rubric changed")
        identity["rubric_identity"] = rubric_identity
    if manifest.get("version") == SPARSE_PACK_VERSION:
        identity["sampling_registration"] = manifest["sampling_registration"]
    expected_version = SPARSE_PACK_VERSION if manifest["settings"].get("sampling_version") == "sparse-window-v2" else VERSION
    if manifest["version"] != expected_version or _digest(identity) != manifest["pack_id"]:
        raise ValueError("annotation pack identity mismatch")
    if _bytes_digest(pack / "protocol.txt") != manifest["protocol_sha256"]:
        raise ValueError("annotation protocol changed")
    source = manifest["source"]
    frame, item = load_history(source["history"], source["market"], source["symbol"])
    if _source(frame, item, source["history"], source["market"], source["symbol"]) != source:
        raise ValueError("annotation history source identity changed")
    blocks, support, warmup = _sample(frame, item, source["market"], source["symbol"], manifest["settings"],
                                     manifest.get("sampling_registration"))
    expected = []
    for block in blocks:
        for day in block["dates"]:
            observed = (frame.loc[:day].tail(manifest["settings"]["context_bars"])
                        if manifest["settings"]["label_mode"] == "asof" else
                        frame.loc[block["exposure_start"]:block["end"]])
            bars = _bars(observed)
            case_id = f"{source['market']}-{source['symbol']}-{manifest['settings']['label_mode']}-{day}"
            case = {"id": case_id, "market": source["market"], "symbol": source["symbol"], "timeframe": "1d",
                    "as_of": day, "label_mode": manifest["settings"]["label_mode"], "role": block["role"],
                    "panel": "window", "block_id": block.get("block_id", f"window-{block['role']}-{block['start']}-{block['end']}"),
                    "information_start": bars[0]["date"], "information_end": bars[-1]["date"],
                    "chart": f"charts/{case_id}.html"}
            expected.append(case)
            path = pack / case["chart"]
            expected_chart = _chart(case, bars).encode()
            chart_meta = manifest["charts"].get(case_id)
            if (path.read_bytes() != expected_chart or chart_meta !=
                    {"sha256": hashlib.sha256(expected_chart).hexdigest(), "bars_sha256": _digest(bars), "bars": len(bars)}):
                raise ValueError("annotation chart changed or contains unregistered exposure")
    if sorted(expected, key=lambda case: case["as_of"]) != manifest["cases"] or warmup != manifest["common_warmup_bars"]:
        raise ValueError("annotation cases changed")
    expected_sampling = _sampling_description(blocks, support, manifest["settings"])
    if expected_sampling != manifest["sampling"]:
        raise ValueError("annotation sampling support changed")
    insufficient = any(value["selected_blocks"] < value["requested_blocks"] for value in support.values())
    expected_status = ("SHORT_HISTORY" if source["symbol"] == "SPCX" or item["status"] == "short_history"
                       else "INSUFFICIENT_WINDOWS" if insufficient else "AWAITING_HUMAN_REVIEW")
    if manifest.get("status") != expected_status:
        raise ValueError("annotation coverage status changed")
    return manifest


def _reviewer(reviewer, mode):
    if not isinstance(reviewer, dict) or not isinstance(reviewer.get("name"), str) or not reviewer["name"].strip():
        raise ValueError("completed labels require named human reviewer")
    for field in ("independent_human", "no_strategy_outputs"):
        if reviewer.get(field) is not True:
            raise ValueError(f"human reviewer must explicitly attest {field}")
    if mode == "asof" and reviewer.get("no_future_prices_for_asof") is not True:
        raise ValueError("asof reviewer must attest no future price exposure")
    try:
        stamp = datetime.fromisoformat(reviewer.get("reviewed_at", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("reviewed_at must be an ISO timestamp") from exc
    if stamp.utcoffset() is None:
        raise ValueError("reviewed_at must include timezone")


def validate_annotation_source(source, reference_path):
    """Resolve genuine local protocol provenance; return verified pack manifest.

    load_labels must additionally call validate_annotation_label for each label.
    This is a local artifact source, not a fabricated published HTTPS page.
    """
    path = Path(source.get("pack_manifest", ""))
    if not path.is_absolute():
        path = Path(reference_path).resolve().parent / path
    if path.name != "manifest.json":
        raise ValueError("local annotation source requires pack manifest")
    manifest = _load_manifest(path.parent)
    identity = manifest["source"]
    if (source.get("type") != "local_annotation_protocol" or source.get("pack_id") != manifest["pack_id"]
            or source.get("source_identity") != identity
            or source.get("protocol_sha256") != manifest["protocol_sha256"]
            or source.get("source_url") != f"annotation-pack://{manifest['pack_id']}"
            or source.get("original_market") != identity["market"]
            or source.get("original_symbol") != identity["symbol"] or source.get("original_timeframe") != "1d"
            or source.get("independence") != "external_human"
            or source.get("hindsight") != (manifest["settings"]["label_mode"] == "retrospective")):
        raise ValueError("local annotation source identity mismatch")
    _reviewer(source.get("reviewer"), manifest["settings"]["label_mode"])
    if source.get("author") != source["reviewer"]["name"]:
        raise ValueError("local source author must identify reviewer")
    if manifest.get("review_policy") == "independent-double-blind-v1" and "consensus_reviews" not in source:
        raise ValueError("registered cycle source requires both original blind reviews")
    _verified_review(source, reference_path, manifest)
    return manifest


def _verified_review(source, reference_path, manifest):
    if "consensus_reviews" in source:
        from .consensus import build_consensus
        from .cycles import validate_cycle_reviewer
        pins = source["consensus_reviews"]
        if not isinstance(pins, dict) or set(pins) not in ({"review_a", "review_b"}, {"review_a", "review_b", "adjudication"}):
            raise ValueError("consensus requires both original reviews and registered adjudication")
        originals = {}
        for key, pin in pins.items():
            if not isinstance(pin, dict):
                raise ValueError("consensus review pin requires actual file")
            path = Path(pin.get("path", ""))
            if not path.is_absolute():
                path = Path(reference_path).resolve().parent / path
            if not path.is_file() or _bytes_digest(path) != pin.get("sha256"):
                raise ValueError("original consensus review artifact changed or missing")
            originals[key] = json.loads(path.read_text())
            validate_cycle_reviewer(manifest, originals[key])
        review = build_consensus(manifest, originals["review_a"], originals["review_b"], originals.get("adjudication"))
        if review["consensus"] != source.get("consensus_summary") or review["reviewer"] != source.get("reviewer"):
            raise ValueError("consensus provenance differs from original reviews")
        return review
    path = Path(source.get("review_file", ""))
    if not path.is_absolute():
        path = Path(reference_path).resolve().parent / path
    if not path.is_file() or _bytes_digest(path) != source.get("review_sha256"):
        raise ValueError("human review artifact changed or missing")
    review = json.loads(path.read_text(encoding="utf-8"))
    if (review.get("pack_id") != manifest["pack_id"] or review.get("source_identity") != manifest["source"]
            or review.get("reviewer") != source.get("reviewer")
            or review.get("label_mode") != manifest["settings"]["label_mode"]):
        raise ValueError("human review artifact source identity mismatch")
    if review.get("rubric_identity") != manifest.get("rubric_identity"):
        raise ValueError("human review rubric identity mismatch")
    return review


def validate_annotation_label(label, source, reference_path, manifest=None):
    manifest = manifest or validate_annotation_source(source, reference_path)
    case = next((case for case in manifest["cases"] if case["id"] == label.get("id")), None)
    if case is None or any(label.get(key) != case.get(key) for key in CASE_IDENTITY):
        raise ValueError("human label changed registered source/date/information bounds")
    if label.get("review_status") != "completed" or label.get("independence") != "external_human":
        raise ValueError("local reference is not explicitly human reviewed")
    if case["panel"] == "window" and label.get("cycle_id") is not None:
        raise ValueError("sampled blocks cannot become independent cycles")
    from .cycles import reference_case
    expected = reference_case(manifest, case)
    if any(label.get(key) != expected.get(key) for key in ("group_information_start", "group_information_end")):
        raise ValueError("reference shortened registered complete cycle information")
    review = _verified_review(source, reference_path, manifest)
    records = [record for record in review.get("labels", []) if record.get("id") == label.get("id")]
    if len(records) != 1 or records[0].get("status") != "completed":
        raise ValueError("reference lacks original completed human review")
    original = records[0]
    from .label_quality import validate_review_exposure
    validate_review_exposure(original)
    for key in (*CASE_IDENTITY, "phase", "regime", "ambiguity", "ambiguity_resolved", "resolver", "resolution_rationale", "notes",
                "request_context", "additional_exposure"):
        if original.get(key) != label.get(key):
            raise ValueError("reference differs from original human review")
    return case


def _import_review(pack, manifest, raw, provenance, output=None):
    pack = Path(pack).resolve()
    if raw.get("pack_id") != manifest["pack_id"] or raw.get("source_identity") != manifest["source"]:
        raise ValueError("review source identity mismatch")
    if raw.get("rubric_identity") != manifest.get("rubric_identity"):
        raise ValueError("review rubric identity mismatch")
    mode = manifest["settings"]["label_mode"]
    if raw.get("label_mode") != mode or not isinstance(raw.get("labels"), list):
        raise ValueError("review label_mode mismatch or missing labels")
    cases = {case["id"]: case for case in manifest["cases"]}
    seen, labels, excluded = set(), [], []
    for record in raw["labels"]:
        if not isinstance(record, dict) or record.get("id") not in cases or record["id"] in seen:
            raise ValueError("unknown or duplicate review case")
        seen.add(record["id"])
        case = cases[record["id"]]
        if any(record.get(key) != case.get(key) for key in CASE_IDENTITY) or (case["panel"] == "window" and record.get("cycle_id") is not None):
            raise ValueError("review changed registered source/date/information bounds")
        from .label_quality import validate_review_exposure
        validate_review_exposure(record)
        status = record.get("status")
        if status in ("blank", "unresolved", "excluded"):
            excluded.append({"id": record["id"], "status": status, "reason": record.get("notes", "")})
            continue
        if status != "completed":
            raise ValueError("review status must be blank, unresolved, excluded or completed")
        reference_state(record, mode)
        ambiguous = record.get("ambiguity") not in (None, False, "", "none")
        if ambiguous and (record.get("ambiguity_resolved") is not True or
                          not str(record.get("resolver", "")).strip() or
                          not str(record.get("resolution_rationale", "")).strip()):
            raise ValueError("completed ambiguity requires named independent resolver and rationale")
        from .cycles import reference_case
        labels.append({**reference_case(manifest, case), **{key: record[key] for key in ("request_context", "additional_exposure") if key in record},
                       "phase": record["phase"], "regime": record["regime"],
                       "ambiguity": record.get("ambiguity"), "ambiguity_resolved": record.get("ambiguity_resolved", False),
                       "resolver": record.get("resolver", ""), "resolution_rationale": record.get("resolution_rationale", ""),
                       "notes": record.get("notes", ""), "review_status": "completed", "eligible": True,
                       "adjudication": "resolved" if ambiguous else "explicit", "independence": "external_human",
                       "source_id": f"human-{manifest['pack_id']}"})
    excluded.extend({"id": ident, "status": "missing", "reason": "not returned for review"} for ident in cases if ident not in seen)
    if not labels:
        raise ValueError("no completed human-reviewed labels; blanks/unresolved/excluded are not scoreable")
    _reviewer(raw.get("reviewer"), mode)
    if any(label["adjudication"] == "resolved" and label["resolver"].strip() == raw["reviewer"]["name"].strip()
           for label in labels):
        raise ValueError("ambiguity resolver must be independent of initial reviewer")
    source = {"id": f"human-{manifest['pack_id']}", "type": "local_annotation_protocol",
              "original_market": manifest["source"]["market"], "original_symbol": manifest["source"]["symbol"],
              "original_timeframe": "1d", "author": raw["reviewer"]["name"], "reviewer": raw["reviewer"],
              "independence": "external_human", "hindsight": mode == "retrospective",
              "source_url": f"annotation-pack://{manifest['pack_id']}", "pack_id": manifest["pack_id"],
              "pack_manifest": str(pack / "manifest.json"), "protocol_sha256": manifest["protocol_sha256"],
              "source_identity": manifest["source"], **provenance}
    result = {"schema_version": 1, "market": manifest["source"]["market"], "symbol": manifest["source"]["symbol"],
              "label_mode": mode, "sources": [source], "labels": sorted(labels, key=lambda label: label["as_of"]),
              "sampling": manifest["sampling"], "import_summary": {"completed": len(labels), "unscoreable": excluded,
                  "cycle_panel": "registered_human_cycles" if manifest["sampling"].get("panel") == "cycle" else
                                 "not_acquired; random window blocks are not human cycles"}}
    if output is not None:
        output = Path(output)
        if output.exists() and json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("refusing overwrite of different imported labels")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output, result)
    return result


def import_annotations(pack, reviewed, output=None):
    """Legacy single-review window import; new cycle packs need both originals."""
    manifest = _load_manifest(pack)
    if manifest.get("review_policy") == "independent-double-blind-v1":
        raise ValueError("registered cycle pack requires paired consensus import")
    raw = json.loads(Path(reviewed).read_text(encoding="utf-8"))
    return _import_review(pack, manifest, raw, {"review_file": str(Path(reviewed).resolve()),
                          "review_sha256": _bytes_digest(reviewed)}, output)


def import_consensus_annotations(pack, review_a, review_b, output=None, *, adjudication=None):
    from .consensus import build_consensus
    from .cycles import validate_cycle_reviewer
    manifest = _load_manifest(pack)
    paths = {"review_a": review_a, "review_b": review_b}
    if adjudication is not None:
        paths["adjudication"] = adjudication
    originals = {key: json.loads(Path(path).read_text()) for key, path in paths.items()}
    for review in originals.values():
        validate_cycle_reviewer(manifest, review)
    merged = build_consensus(manifest, originals["review_a"], originals["review_b"], originals.get("adjudication"))
    provenance = {"consensus_reviews": {key: {"path": str(Path(path).resolve()), "sha256": _bytes_digest(path)}
                                         for key, path in paths.items()}, "consensus_summary": merged["consensus"]}
    return _import_review(pack, manifest, merged, provenance, output)


def annotation_overlap_support(manifest):
    """Count actual selected block exposure overlap; counts are not independence."""
    result = {}
    for role in ("development", "holdout"):
        grouped = {}
        for case in manifest["cases"]:
            if case["role"] == role:
                ident = case["cycle_id"] if case.get("panel", "window") == "cycle" else case["block_id"]
                from .cycles import reference_case
                exposure = reference_case(manifest, case)
                a, b = exposure.get("group_information_start", case["information_start"]), exposure.get("group_information_end", case["information_end"])
                interval = grouped.setdefault(ident, [a, b])
                interval[0] = min(interval[0], a)
                interval[1] = max(interval[1], b)
        pairs, involved = 0, set()
        entries = sorted(grouped.items())
        for index, (left_id, left) in enumerate(entries):
            for right_id, right in entries[index + 1:]:
                if _overlap(left, right):
                    pairs += 1
                    involved.update((left_id, right_id))
        result[role] = {"blocks": len(entries), "overlapping_block_pairs": pairs,
                        "blocks_with_context_overlap": len(involved), "independent_cycle_count": None}
    return result


def write_annotation_index(pack):
    """Write an offline manual editor; charts load only on current-case selection.

    The blank master and registered pack/source identities remain unchanged.
    Downloaded drafts are resumable through the editor's local file picker.
    """
    pack = Path(pack).resolve()
    manifest = _load_manifest(pack)
    template = {"schema_version": 1, "pack_id": manifest["pack_id"], "source_identity": manifest["source"],
                "label_mode": manifest["settings"]["label_mode"], "reviewer": {
                    "name": "", "reviewed_at": "", "independent_human": False,
                    "no_strategy_outputs": False, "no_future_prices_for_asof": False},
                "labels": [{**case, "status": "blank", "phase": None, "regime": None, "ambiguity": None,
                            "ambiguity_resolved": False, "resolver": "", "resolution_rationale": "", "notes": ""}
                           for case in manifest["cases"]]}
    if "rubric_identity" in manifest:
        template["rubric_identity"] = manifest["rubric_identity"]
    payload = _json({"template": template, "immutable": CASE_IDENTITY, "context_overlap": annotation_overlap_support(manifest),
                     "classes": joint_classes(template["label_mode"])}).replace("<", "\\u003c")
    options = "".join(f'<option value="{i}">{case["as_of"]} · {case["role"]} · {html.escape(case.get("cycle_id") or case["block_id"])}</option>'
                      for i, case in enumerate(manifest["cases"]))
    classes = '<option value="">空白：尚未给出参考标签</option>' + "".join(
        f'<option value="{value}">{value}</option>' for value in joint_classes(template["label_mode"]))
    mode = "ASOF：仅看截至目标日行情" if template["label_mode"] == "asof" else "RETROSPECTIVE：允许后续行情，不能用于 ASOF 真值"
    document = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>人工盲标 · __SYMBOL__</title>
<style>body{font:16px system-ui;max-width:1450px;margin:24px auto;padding:0 18px;background:#f8fafc;color:#172033}label{display:block;margin:10px 0}input[type=text],textarea,select{font:inherit;padding:7px;max-width:95%;box-sizing:border-box}textarea{width:100%;height:70px}iframe{width:100%;height:740px;border:1px solid #ccd3de;background:white}fieldset{margin:16px 0;border:1px solid #ccd3de}button{font:inherit;padding:8px 14px;margin:5px 5px 5px 0}.warning{color:#9e2520}details{margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>__SYMBOL__ · 人工独立标注</h1><p><strong>__MODE__</strong> · <a href="protocol.txt" target="_blank">完整标注协议</a></p>
<p class="warning">按日期从早到晚完成，再看下一张图。曾提前看过目标日之后行情的 ASOF 案例应标为未解决或排除。随机窗口不是独立周期；周期 panel 需另行人工识别。当前页没有模型阶段或自动真值。</p>
<p>本页不自动保存。离开前下载草稿；载入草稿可以继续。未完成记录保留空白，下载不会自动完成标注。</p>
<label>日期案例 <select id="picker">__OPTIONS__</select></label>
<button id="previous" type="button">上一日期</button><button id="next" type="button">下一日期</button><span id="position"></span>
<p id="bounds"></p><iframe id="case-chart" title="当前案例行情图" loading="eager"></iframe>
<fieldset id="case-editor"><legend>当前日期的人工判断</legend>
<label>阶段 / 方向 <select id="state">__CLASSES__</select></label>
<label>状态 <select id="status"><option value="blank">空白，尚未完成</option><option value="completed">completed，已亲自审阅且有明确标签</option><option value="unresolved">unresolved，分歧或信息不足</option><option value="excluded">excluded，明确排除</option></select></label>
<label>说明 / 排除原因 <textarea id="notes"></textarea></label>
<label><input type="checkbox" id="context-request">背景不足，需要更多历史；保持未解决，不作为正常无阶段</label>
<label><input type="checkbox" id="ambiguity">有作者分歧或歧义</label>
<label><input type="checkbox" id="resolved">歧义已由独立人员仲裁</label>
<label>独立仲裁人 <input type="text" id="resolver"></label><label>仲裁理由 <textarea id="rationale"></textarea></label>
</fieldset>
<fieldset><legend>人工审阅者及明确声明（默认均不勾选）</legend>
<label>姓名 / 可追溯身份 <input type="text" id="reviewer-name"></label>
<label>可追溯身份编号（可选） <input type="text" id="reviewer-id"></label>
<label><input type="checkbox" id="human">本人为独立人类审阅者，没有使用规则或 LLM 生成参考真值</label>
<label><input type="checkbox" id="strategy">标注期间没有查看本分类器的阶段、事件或参数输出</label>
<label><input type="checkbox" id="future">ASOF 已完成案例的判断没有使用目标日之后行情；既有暴露已明确排除</label>
<label><input type="checkbox" id="blind">本人独立盲标，没有查看另一审阅者的答案</label>
<label><input type="checkbox" id="independent-review">本人独立于另一审阅者，未协商阶段或方向答案</label>
</fieldset>
<button id="download" type="button">下载审阅 JSON / 空白草稿</button>
<label>载入本包已下载草稿 <input type="file" id="draft" accept=".json,application/json"></label>
<p id="message" role="status"></p>
<details><summary>精确行情来源、复权版本和窗口上下文重叠</summary><pre id="source-details"></pre></details>
<script id="review-data" type="application/json">__PAYLOAD__</script>
<script>
const setup=JSON.parse(document.getElementById('review-data').textContent);
const original=setup.template,review=JSON.parse(JSON.stringify(original)),get=id=>document.getElementById(id);
let current=-1;
const canonical=value=>JSON.stringify(value,(_k,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
function saveCurrent(){
 if(current<0)return;const record=review.labels[current],state=get('state').value;
 record.phase=state?(state==='no_observable_phase'?state:state.split('/')[0]):null;
 record.regime=state?(state==='no_observable_phase'?'undetermined':state.split('/')[1]):null;
 record.status=get('status').value;record.notes=get('notes').value;
 record.request_context=get('context-request').checked;
 if(record.request_context&&record.status==='completed'){record.status='unresolved';get('status').value='unresolved';get('message').textContent='背景请求尚未解决，已保留为 unresolved；需重新登记更完整背景后再审阅。';}
 record.ambiguity=get('ambiguity').checked?true:null;record.ambiguity_resolved=get('resolved').checked;
 record.resolver=get('resolver').value;record.resolution_rationale=get('rationale').value;
}
function selectCase(index){
 if(index<0||index>=review.labels.length)return;saveCurrent();current=index;
 const record=review.labels[index];get('picker').value=String(index);
 get('state').value=record.phase?(record.phase==='no_observable_phase'?record.phase:record.phase+'/'+record.regime):'';
 get('status').value=record.status;get('notes').value=record.notes||'';get('ambiguity').checked=Boolean(record.ambiguity);
 get('context-request').checked=record.request_context===true;
 get('resolved').checked=record.ambiguity_resolved===true;get('resolver').value=record.resolver||'';get('rationale').value=record.resolution_rationale||'';
 get('bounds').textContent='目标 '+record.as_of+'；实际可见 '+record.information_start+' 至 '+record.information_end+'；'+record.role+' / '+record.panel+(record.panel==='cycle'?'；登记尺度：'+record.analysis_scale+'；父周期：'+(record.parent_cycle||'未登记'):'');
 get('position').textContent=(index+1)+' / '+review.labels.length;get('case-chart').src=record.chart;
}
function snapshot(){
 saveCurrent();review.reviewer={...(review.reviewer||{}),name:get('reviewer-name').value,reviewed_at:new Date().toISOString(),
 independent_human:get('human').checked,no_strategy_outputs:get('strategy').checked,no_future_prices_for_asof:get('future').checked};
 review.reviewer.blind_review=get('blind').checked;review.reviewer.no_other_review_seen=get('blind').checked;
 review.reviewer.independent_of_other_review=get('independent-review').checked;
 if(get('reviewer-id').value.trim())review.reviewer.identity_id=get('reviewer-id').value;else delete review.reviewer.identity_id;
 return JSON.parse(JSON.stringify(review));
}
function importDraft(raw){
 const who=raw.reviewer||{};if(who.type!=null&&who.type!=='human'||who.independence!=null&&who.independence!=='external_human')throw Error('草稿声明不属于独立人类标注');if(who.identity_id!=null&&(typeof who.identity_id!=='string'||!who.identity_id.trim()))throw Error('草稿身份编号无效');
 if(raw.pack_id!==original.pack_id||raw.label_mode!==original.label_mode||canonical(raw.source_identity)!==canonical(original.source_identity)||canonical(raw.rubric_identity)!==canonical(original.rubric_identity)||!Array.isArray(raw.labels)||raw.labels.length!==original.labels.length)throw Error('草稿与本包来源、规范或样本不匹配');
 const allowed=['blank','completed','unresolved','excluded'],records=new Map(raw.labels.map(v=>[v.id,v]));
 if(records.size!==raw.labels.length)throw Error('草稿含重复案例');
 const fields=['status','phase','regime','ambiguity','ambiguity_resolved','resolver','resolution_rationale','notes','request_context','additional_exposure'];
 const restored=original.labels.map(base=>{const row=records.get(base.id);if(!row||setup.immutable.some(k=>canonical(row[k])!==canonical(base[k]))||(base.panel==='window'&&row.cycle_id!=null))throw Error('草稿改变已登记日期、来源或信息边界');if(!allowed.includes(row.status))throw Error('草稿状态无效');if('request_context' in row&&typeof row.request_context!=='boolean')throw Error('草稿背景请求必须是布尔值');if('additional_exposure' in row&&!Array.isArray(row.additional_exposure))throw Error('草稿新增暴露必须是数组');const state=row.phase?(row.phase==='no_observable_phase'?row.phase:row.phase+'/'+row.regime):'';if(state&&!setup.classes.includes(state))throw Error('草稿阶段与方向不属于此模式');const result=JSON.parse(JSON.stringify(base));fields.forEach(k=>{if(k in row)result[k]=row[k];});return result;});
 review.labels=restored;current=-1;const person=raw.reviewer||{};
 get('reviewer-name').value=person.name||'';get('human').checked=person.independent_human===true;get('strategy').checked=person.no_strategy_outputs===true;get('future').checked=person.no_future_prices_for_asof===true;
 get('reviewer-id').value=person.identity_id||'';review.reviewer=JSON.parse(JSON.stringify(person));
 get('blind').checked=person.blind_review===true&&person.no_other_review_seen===true;get('independent-review').checked=person.independent_of_other_review===true;
 selectCase(0);get('message').textContent='已载入草稿；状态保持原样，没有自动补全。';
}
get('picker').onchange=()=>selectCase(Number(get('picker').value));
get('previous').onclick=()=>selectCase(current-1);get('next').onclick=()=>selectCase(current+1);
get('download').onclick=()=>{const value=snapshot(),blob=new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=original.source_identity.symbol+'-'+original.label_mode+'-review.json';link.click();URL.revokeObjectURL(url);get('message').textContent='已下载草稿，完成 '+value.labels.filter(v=>v.status==='completed').length+' / '+value.labels.length+'。下载时间记录为 reviewed_at，未完成项不会变成真值。';};
get('draft').onchange=async()=>{try{if(get('draft').files.length)importDraft(JSON.parse(await get('draft').files[0].text()));}catch(e){get('message').textContent=e.message;}};
get('source-details').textContent=JSON.stringify({source:original.source_identity,context_overlap:setup.context_overlap},null,2);
window.annotationReview={snapshot,selectCase,importDraft};
if(review.labels.length)selectCase(0);else{get('case-editor').disabled=true;get('position').textContent='0 / 0';get('bounds').textContent='历史或资格不足：没有可标注日期，不生成虚假标签。';}
</script></html>'''
    document = (document.replace("__SYMBOL__", html.escape(manifest["source"]["symbol"]))
                .replace("__MODE__", mode).replace("__OPTIONS__", options).replace("__CLASSES__", classes)
                .replace("__PAYLOAD__", payload))
    if "rubric_identity" in manifest:
        document = document.replace('<a href="protocol.txt" target="_blank">完整标注协议</a>',
                                    '<a href="protocol.txt" target="_blank">完整标注协议</a> · <a href="rubric.md" target="_blank">已冻结阶段标注规范</a>')
    path = pack / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def write_annotation_overview(root, summaries):
    """Write linked multi-symbol entry page, with measured context overlap counts.

    summaries entries require market/symbol. All counts are read from verified
    manifests under root/<market>/<symbol>, never promoted to independent cycles.
    """
    root = Path(root).resolve()
    rows = []
    for summary in summaries:
        market, symbol = summary["market"], summary["symbol"]
        pack = root / market / symbol
        manifest = _load_manifest(pack)
        if manifest["source"]["market"] != market or manifest["source"]["symbol"] != symbol:
            raise ValueError("overview source identity mismatch")
        support = manifest["sampling"]["support"]
        overlap = annotation_overlap_support(manifest)
        link = f"{market}/{symbol}"
        counts = []
        for role in ("development", "holdout"):
            counts.append(f'{support[role]["selected_blocks"]}/{support[role]["requested_blocks"]} 块；'
                          f'{overlap[role]["overlapping_block_pairs"]} 对上下文重叠；'
                          f'{overlap[role]["blocks_with_context_overlap"]} 块涉及重叠')
        rows.append(f'<tr><td>{html.escape(market)}</td><td><a href="{html.escape(link)}/index.html">{html.escape(symbol)}</a></td>'
                    f'<td>{html.escape(manifest["status"])}</td><td>{len(manifest["cases"])}</td>'
                    f'<td>{counts[0]}</td><td>{counts[1]}</td><td><a href="{html.escape(link)}/protocol.txt">协议</a></td></tr>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>独立人工标注入口</title>
<style>body{font:16px system-ui;max-width:1450px;margin:28px auto;padding:0 18px;color:#172033;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border:1px solid #ccd3de;text-align:left}a{color:#204b9b}.warning{color:#9e2520}</style>
<h1>独立人工盲标 · 标的入口</h1>
<p>选择标的后按日期从早到晚审阅，仅显示当前日期图。人工填写阶段、方向和状态，下载 JSON 后再导入；空白、未解决及排除项不能评分。</p>
<p class="warning">这里没有真实参考标签或模型建议。随机窗口不是独立结构周期，日期数和窗口块数不等于独立样本数。周期 panel 必须另行人工识别；当前包不能保证全类别覆盖或监督资格。</p>
<p>上下文重叠按各块所有已选案例实际 information_start/end 的联合区间计算，重叠对包含任意共同日期；仅报告潜在依赖，不能据此宣称未重叠块统计独立。行情使用来源记录的复权版本，非原时点行情时明确保留该限制。SPCX 短历史不填补。</p>
<table><thead><tr><th>市场</th><th>标的 / 开始审阅</th><th>状态</th><th>待审日期</th><th>开发段：实际/请求</th><th>留出段：实际/请求</th><th>协议</th></tr></thead><tbody>__ROWS__</tbody></table>
</html>'''.replace("__ROWS__", "\n".join(rows))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
