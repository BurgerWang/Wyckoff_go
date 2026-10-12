"""Blind human-review safety checks; fixtures are not market reference truth."""
import copy
import json
import re

import pandas as pd
import pytest

from research.tuning import annotation
from research.tuning.labels import load_labels
from research.tuning.runner import run_lock


@pytest.fixture
def history(monkeypatch, tmp_path):
    index = pd.date_range("2019-01-01", "2026-09-29", freq="D", name="Date")
    frame = pd.DataFrame({"Open": [10000.25 + i for i in range(len(index))],
                          "High": [10001.5 + i for i in range(len(index))],
                          "Low": [9999.75 + i for i in range(len(index))],
                          "Close": [10000.5 + i for i in range(len(index))],
                          "Volume": [100 + i for i in range(len(index))]}, index=index)
    frame.attrs = {"symbol": "BTCUSDT", "timeframe": "1d", "market_type": "spot",
                   "source": "binance_spot", "price_currency": "USDT", "volume_unit": "BTC",
                   "source_urls": ["https://data.binance.vision/"]}
    item = {"status": "complete", "first": "2019-01-01", "last": "2026-09-29", "bars": len(frame),
            "continuous_segments": [{"start": "2019-01-01", "end": "2026-09-29", "bars": len(frame)}],
            "missing_dates": [], "attrs": frame.attrs}
    def load(path, market, symbol):
        if market != "binance_spot" or symbol != "BTCUSDT":
            raise ValueError("source identity mismatch")
        return frame.copy(), copy.deepcopy(item)
    monkeypatch.setattr(annotation, "load_history", load)
    return tmp_path / "history", frame, item


def prepare(history, target, **kwargs):
    return annotation.prepare_annotation_pack(history[0], "binance_spot", "BTCUSDT", target,
                                               development_blocks=2, holdout_blocks=2, **kwargs)


def payload(path):
    raw = path.read_text()
    data = re.search(r'<script id="annotation-data" type="application/json">(.*?)</script>', raw, re.S).group(1)
    return json.loads(data)


def reviewed(pack, tmp_path):
    raw = json.loads((pack / "annotation.json").read_text())
    raw["reviewer"] = {"name": "Independent fixture human", "reviewed_at": "2026-09-30T11:00:00+07:00",
                       "independent_human": True, "no_strategy_outputs": True, "no_future_prices_for_asof": True}
    raw["labels"][0].update(status="completed", phase="A", regime="undetermined")
    path = tmp_path / "review.json"
    path.write_text(json.dumps(raw))
    return raw, path


def test_resolved_string_requires_named_independent_resolution(history, tmp_path):
    pack = tmp_path / "pack"
    prepare(history, pack)
    raw, path = reviewed(pack, tmp_path)
    record = raw["labels"][0]
    record["ambiguity"] = "resolved"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="resolver"):
        annotation.import_annotations(pack, path)
    record.update(ambiguity_resolved=True, resolver="Independent adjudicator", resolution_rationale="Reviewed evidence")
    path.write_text(json.dumps(raw))
    result = annotation.import_annotations(pack, path)
    assert result["labels"][0]["adjudication"] == "resolved"


def test_registered_rubric_preserves_old_pack_and_binds_review(history, tmp_path):
    old = tmp_path / "old"
    old_manifest = prepare(history, old)
    before = {name: (old / name).read_bytes() for name in ("manifest.json", "annotation.json", "protocol.txt")}
    rubric = tmp_path / "rubric.md"
    rubric.write_text("Frozen human phase definitions, fixture only.\n")
    new = tmp_path / "new"
    manifest = prepare(history, new, rubric_path=rubric)
    assert manifest["pack_id"] != old_manifest["pack_id"]
    assert annotation._load_manifest(old) == old_manifest
    assert annotation._load_manifest(new) == manifest
    assert (new / "rubric.md").read_bytes() == rubric.read_bytes()
    annotation.write_annotation_index(new)
    assert 'href="rubric.md"' in (new / "index.html").read_text()
    raw, path = reviewed(new, tmp_path)
    assert raw["rubric_identity"] == manifest["rubric_identity"]
    assert annotation.import_annotations(new, path)["labels"]
    raw["rubric_identity"]["sha256"] = "changed"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="rubric"):
        annotation.import_annotations(new, path)
    (new / "rubric.md").write_text("Changed after freezing")
    with pytest.raises(ValueError, match="rubric"):
        annotation._load_manifest(new)
    assert before == {name: (old / name).read_bytes() for name in before}


def test_asof_payload_physically_contains_only_source_prefix(history, tmp_path):
    pack = tmp_path / "pack"
    result = prepare(history, pack)
    assert len(result["cases"]) == 12
    for case in result["cases"]:
        chart = payload(pack / case["chart"])
        assert chart["case"] == case
        assert len(chart["bars"]) == 100
        assert max(bar["date"] for bar in chart["bars"]) == case["as_of"] == case["information_end"]
        assert chart["bars"] == annotation._bars(history[1].loc[:case["as_of"]].tail(100))
        assert all(set(bar) == {"date", "open", "high", "low", "close", "volume"} for bar in chart["bars"])
        next_day = history[1].loc[case["as_of"]:].iloc[1]
        assert str(float(next_day["High"])) not in (pack / case["chart"]).read_text()
        assert "cycle_id" not in case and case["panel"] == "window"


def test_sampling_prefix_years_boundaries_and_full_context_diagnostic_exclusion(history, tmp_path):
    small = prepare(history, tmp_path / "small")
    large = annotation.prepare_annotation_pack(history[0], "binance_spot", "BTCUSDT", tmp_path / "large",
                                               development_blocks=12, holdout_blocks=12)
    for role in ("development", "holdout"):
        first = [b["block_id"] for b in small["sampling"]["blocks"] if b["role"] == role]
        second = [b["block_id"] for b in large["sampling"]["blocks"] if b["role"] == role]
        assert first == second[:len(first)]
    assert len(large["sampling"]["support"]["development"]["selected_years"]) >= 5
    for block in large["sampling"]["blocks"]:
        dates = [pd.Timestamp(day) for day in block["dates"]]
        assert min((right - left).days for left, right in zip(dates, dates[1:])) >= 5
    for case in large["cases"]:
        if case["role"] == "development":
            assert case["information_end"] < "2024-10-01"
        else:
            assert "2024-10-01" <= case["information_start"] <= case["as_of"] < "2026-09-30"
            assert not any(annotation._overlap((case["information_start"], case["information_end"]), interval)
                           for interval in annotation.KNOWN_DIAGNOSTIC_INTERVALS[("binance_spot", "BTCUSDT")])
    assert large["sampling"]["support"]["holdout"]["diagnostic_context_rejected_blocks"] > 0
    assert prepare(history, tmp_path / "small") == small
    with pytest.raises(ValueError, match="different sampling"):
        prepare(history, tmp_path / "small", seed=41)


def test_retrospective_separate_mode_and_actual_future_bound(history, tmp_path):
    pack = tmp_path / "retro"
    result = prepare(history, pack, label_mode="retrospective")
    for case in result["cases"]:
        chart = payload(pack / case["chart"])
        assert chart["bars"][-1]["date"] == case["information_end"] >= case["as_of"]
        assert "RETROSPECTIVE: later prices shown" in (pack / case["chart"]).read_text()
    assert any(case["information_end"] > case["as_of"] for case in result["cases"])
    with pytest.raises(ValueError, match="different sampling"):
        prepare(history, pack)


def test_short_history_no_fake_labels(history, monkeypatch, tmp_path):
    frame = history[1].tail(75).copy()
    item = {**history[2], "status": "short_history", "bars": 75, "first": "2026-07-17",
            "short_history_reason": "newly listed real security", "continuous_segments": [
                {"start": frame.index[0].date().isoformat(), "end": frame.index[-1].date().isoformat(), "bars": 75}]}
    monkeypatch.setattr(annotation, "load_history", lambda *args: (frame, item))
    result = prepare(history, tmp_path / "short")
    assert result["status"] == "SHORT_HISTORY"
    assert result["cases"] == []
    assert all(s["selected_blocks"] == 0 for s in result["sampling"]["support"].values())


def test_import_completed_only_and_resolvable_local_human_provenance(history, tmp_path):
    pack = tmp_path / "pack"
    prepare(history, pack)
    raw, path = reviewed(pack, tmp_path)
    raw["labels"][1].update(status="unresolved", phase="C", regime="accumulation", notes="direction disputed")
    raw["labels"][2].update(status="excluded", notes="prior future exposure")
    path.write_text(json.dumps(raw))
    output = tmp_path / "labels.json"
    result = annotation.import_annotations(pack, path, output)
    assert len(result["labels"]) == 1
    assert result["labels"][0]["phase"] == "A"
    assert len(result["import_summary"]["unscoreable"]) == 11
    source = result["sources"][0]
    manifest = annotation.validate_annotation_source(source, output)
    annotation.validate_annotation_label(result["labels"][0], source, output, manifest)
    assert source["author"] == raw["reviewer"]["name"]
    assert source["source_url"].startswith("annotation-pack://")
    assert source["source_identity"]["attrs"] == history[1].attrs
    assert result["sampling"]["blocks_are_not_independent_cycles"]
    loaded = load_labels(output, market="binance_spot", symbol="BTCUSDT", label_mode="asof")
    assert loaded["labels"][0]["joint_state"] == "A/undetermined"
    assert not loaded["diagnostic_only"]
    with pytest.raises(ValueError, match="no completed"):
        annotation.import_annotations(pack, pack / "annotation.json")
    raw["labels"][0]["regime"] = "accumulation"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="artifact changed"):
        annotation.validate_annotation_source(source, output)
    with pytest.raises(ValueError, match="overwrite"):
        annotation.import_annotations(pack, path, output)


@pytest.mark.parametrize("mutation", ["source", "future", "asof", "role", "cycle", "phase_blank", "no_attestation", "ambiguous"])
def test_import_rejects_identity_exposure_and_incomplete_truth(history, tmp_path, mutation):
    pack = tmp_path / "pack"
    prepare(history, pack)
    raw, path = reviewed(pack, tmp_path)
    if mutation == "source":
        raw["source_identity"]["symbol"] = "ETHUSDT"
    elif mutation == "future":
        raw["labels"][0]["information_end"] = "2026-09-29"
    elif mutation == "asof":
        raw["labels"][0]["as_of"] = "2020-01-01"
    elif mutation == "role":
        raw["labels"][0]["role"] = "holdout"
    elif mutation == "cycle":
        raw["labels"][0]["cycle_id"] = "fake-independent-cycle"
    elif mutation == "phase_blank":
        raw["labels"][0]["phase"] = None
    elif mutation == "no_attestation":
        raw["reviewer"]["no_future_prices_for_asof"] = False
    else:
        raw["labels"][0].update(ambiguity=True, ambiguity_resolved=True)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        annotation.import_annotations(pack, path)


def test_manifest_chart_and_source_tampering_rejected(history, tmp_path):
    pack = tmp_path / "pack"
    result = prepare(history, pack)
    raw, path = reviewed(pack, tmp_path)
    case = result["cases"][0]
    chart = pack / case["chart"]
    original = chart.read_text()
    chart.write_text(original.replace('"bars":[', '"bars":[{"date":"2099-01-01","close":1},'))
    with pytest.raises(ValueError, match="unregistered exposure"):
        annotation.import_annotations(pack, path)
    chart.write_text(original)
    result["cases"][0]["information_end"] = "2099-01-01"
    (pack / "manifest.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="cases changed"):
        annotation.import_annotations(pack, path)
    result["cases"][0] = raw["labels"][0]  # restore only immutable values
    result["cases"][0] = {key: raw["labels"][0][key] for key in annotation.IMMUTABLE}
    (pack / "manifest.json").write_text(json.dumps(result))
    history[1].iloc[-1, 0] += 1
    with pytest.raises(ValueError, match="source identity changed"):
        annotation.import_annotations(pack, path)


def test_local_reference_cannot_change_human_truth_or_invent_cycle(history, tmp_path):
    pack = tmp_path / "pack"
    prepare(history, pack)
    _, path = reviewed(pack, tmp_path)
    result = annotation.import_annotations(pack, path)
    source, label = result["sources"][0], result["labels"][0]
    for field, value in (("regime", "accumulation"), ("information_end", "2099-01-01"),
                         ("cycle_id", "renamed-random-block"), ("review_status", "blank")):
        tampered = {**label, field: value}
        with pytest.raises(ValueError):
            annotation.validate_annotation_label(tampered, source, tmp_path / "labels.json")


def test_pack_cooperates_with_cli_run_lock(history, tmp_path):
    pack = tmp_path / "locked"
    with run_lock(pack):
        manifest = prepare(history, pack)
    with run_lock(pack):
        assert prepare(history, pack) == manifest


@pytest.mark.parametrize("with_rubric", [False, True])
def test_annotation_editor_blank_export_manual_completion_and_single_frame(history, tmp_path, with_rubric):
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("Node is needed for offline editor behavior verification")
    pack = tmp_path / "pack"
    kwargs = {}
    if with_rubric:
        rubric = tmp_path / "rubric.md"
        rubric.write_text("Human rubric fixture\n")
        kwargs["rubric_path"] = rubric
    manifest = prepare(history, pack, **kwargs)
    master = (pack / "annotation.json").read_bytes()
    index = annotation.write_annotation_index(pack)
    page = index.read_text()
    data = json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', page, re.S).group(1))
    script = re.findall(r'<script>(.*?)</script>', page, re.S)[0]
    assert page.count("<iframe") == 1
    assert 'src="charts/' not in page and 'rel="preload"' not in page
    assert (pack / "annotation.json").read_bytes() == master
    assert data["template"]["source_identity"] == manifest["source"]
    assert data["template"].get("rubric_identity") == manifest.get("rubric_identity")
    assert [case["as_of"] for case in data["template"]["labels"]] == sorted(case["as_of"] for case in manifest["cases"])
    assert all(not {"open", "high", "low", "close", "volume"}.intersection(record) for record in data["template"]["labels"])
    first = manifest["cases"][0]
    future_price = str(float(history[1].loc[first["as_of"]:].iloc[1]["High"]))
    assert future_price not in page
    bundle = tmp_path / "editor-bundle.json"
    bundle.write_text(json.dumps({"data": data, "script": script}))
    runner = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const bundle=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),elements={},loads=[];
for(const id of ['review-data','picker','state','status','notes','context-request','ambiguity','resolved','resolver','rationale','reviewer-name','reviewer-id','human','strategy','future','blind','independent-review','bounds','position','case-chart','previous','next','download','draft','message','source-details','case-editor'])elements[id]={value:'',checked:false,textContent:'',files:[]};
elements['review-data'].textContent=JSON.stringify(bundle.data);
Object.defineProperty(elements['case-chart'],'src',{set(value){loads.push(value);},get(){return loads.at(-1);}});
let downloaded=null;
const sandbox={document:{getElementById:id=>elements[id],createElement:()=>({click(){}})},window:{},Date,Blob:class{constructor(values){downloaded=JSON.parse(values[0]);}},URL:{createObjectURL:()=> 'blob:local',revokeObjectURL(){}},console};
vm.createContext(sandbox);vm.runInContext(bundle.script,sandbox);
const api=sandbox.window.annotationReview,original=bundle.data.template;
assert.deepStrictEqual(loads,[original.labels[0].chart]);
let blank=api.snapshot();assert(blank.labels.every(v=>v.status==='blank'&&v.phase===null&&v.regime===null));
assert.strictEqual(blank.reviewer.independent_human,false);assert.strictEqual(blank.reviewer.no_strategy_outputs,false);assert.strictEqual(blank.reviewer.no_future_prices_for_asof,false);assert.strictEqual(blank.reviewer.blind_review,false);assert.strictEqual(blank.reviewer.independent_of_other_review,false);
elements.download.onclick();assert(downloaded.labels.every(v=>v.status==='blank'));assert(downloaded.reviewer.reviewed_at.includes('T'));
elements.state.value='A/undetermined';api.selectCase(1);
assert.deepStrictEqual(loads,[original.labels[0].chart,original.labels[1].chart]);
let picked=api.snapshot();assert.strictEqual(picked.labels[0].phase,'A');assert.strictEqual(picked.labels[0].status,'blank');assert.strictEqual(picked.labels[1].phase,null);
api.selectCase(0);elements.status.value='completed';elements.notes.value='personally reviewed fixture';
elements['reviewer-name'].value='Independent fixture human';elements['reviewer-id'].value='fixture-ui-person';elements.human.checked=true;elements.strategy.checked=true;elements.future.checked=true;elements.blind.checked=true;elements['independent-review'].checked=true;
elements.download.onclick();assert.strictEqual(downloaded.labels[0].status,'completed');assert.strictEqual(downloaded.labels[1].status,'blank');
assert.deepStrictEqual(downloaded.source_identity,original.source_identity);assert.strictEqual(downloaded.pack_id,original.pack_id);
const completed=downloaded;assert.strictEqual(completed.reviewer.identity_id,'fixture-ui-person');api.importDraft(completed);assert.strictEqual(api.snapshot().reviewer.identity_id,'fixture-ui-person');assert.strictEqual(api.snapshot().reviewer.blind_review,true);assert.strictEqual(api.snapshot().reviewer.independent_of_other_review,true);assert.strictEqual(api.snapshot().labels[0].status,'completed');
elements['context-request'].checked=true;assert.strictEqual(api.snapshot().labels[0].status,'unresolved');assert.strictEqual(api.snapshot().labels[0].request_context,true);assert(elements.message.textContent.includes('unresolved'));api.importDraft(completed);assert.strictEqual(api.snapshot().labels[0].request_context,false);
for(const [key,value] of [['request_context','yes'],['additional_exposure',{}]]){const malformed=JSON.parse(JSON.stringify(completed));malformed.labels[0][key]=value;assert.throws(()=>api.importDraft(malformed));}
for(const [key,value] of [['type','llm'],['independence','strategy_internal']]){const incompatible=JSON.parse(JSON.stringify(completed));incompatible.reviewer[key]=value;assert.throws(()=>api.importDraft(incompatible));}
const tampered=JSON.parse(JSON.stringify(completed));tampered.labels[0].information_end='2099-01-01';assert.throws(()=>api.importDraft(tampered));
assert.strictEqual(api.snapshot().labels[0].information_end,original.labels[0].information_end);
console.log(JSON.stringify(completed));
'''
    runner_path = tmp_path / "editor-test.js"
    runner_path.write_text(runner)
    check = subprocess.run(["node", str(runner_path), str(bundle)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    downloaded = tmp_path / "downloaded-review.json"
    downloaded.write_text(check.stdout)
    imported = annotation.import_annotations(pack, downloaded)
    assert len(imported["labels"]) == 1
    assert imported["labels"][0]["phase"] == "A"


def test_overview_reports_actual_cross_block_context_overlap(history, tmp_path):
    pack = tmp_path / "binance_spot" / "BTCUSDT"
    manifest = prepare(history, pack)
    annotation.write_annotation_index(pack)
    overview = annotation.write_annotation_overview(tmp_path, [{"market": "binance_spot", "symbol": "BTCUSDT"}])
    page = overview.read_text()
    assert 'href="binance_spot/BTCUSDT/index.html"' in page
    assert 'href="binance_spot/BTCUSDT/protocol.txt"' in page
    assert "不等于独立样本数" in page
    support = annotation.annotation_overlap_support(manifest)
    assert all(part["independent_cycle_count"] is None for part in support.values())
    fixture = {"cases": [
        {"role": "holdout", "block_id": "first", "information_start": "2025-01-01", "information_end": "2025-02-01"},
        {"role": "holdout", "block_id": "first", "information_start": "2025-01-10", "information_end": "2025-02-10"},
        {"role": "holdout", "block_id": "second", "information_start": "2025-02-05", "information_end": "2025-03-01"},
        {"role": "holdout", "block_id": "third", "information_start": "2025-04-01", "information_end": "2025-05-01"}]}
    measured = annotation.annotation_overlap_support(fixture)["holdout"]
    assert measured == {"blocks": 3, "overlapping_block_pairs": 1, "blocks_with_context_overlap": 2, "independent_cycle_count": None}
