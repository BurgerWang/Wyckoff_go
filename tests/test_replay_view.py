"""Offline output safety and report visibility checks."""
import json
import re
import shutil
import subprocess

import pytest

from research.replay_view import write_replay_html


def test_embedded_report_cannot_end_script_or_inject_markup(tmp_path):
    report = {"profile": {"label": "</script><script>alert('x')</script>\u2028&__PACKED_ITEMS__"}, "instruments": []}
    path = tmp_path / "report.html"
    write_replay_html(report, path)
    html = path.read_text()
    payload = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert "<" not in payload
    assert "&" not in payload
    assert "\u2028" not in payload
    assert json.loads(payload) == report
    assert "textContent" in html


def test_failed_instruments_retained_and_output_is_offline(tmp_path):
    path = tmp_path / "nested" / "report.html"
    write_replay_html({"instruments": [{"symbol": "FAILED", "status": "failed", "error": "HTTP 404"}]}, path)
    html = path.read_text()
    assert '"symbol":"FAILED"' in html
    assert '"error":"HTTP 404"' in html
    assert "[获取失败]" in html
    assert 'id="error"' in html
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=', html)
    assert "https://" not in html
    assert "all.slice(first,end+1)" in html
    assert "date(event.available_date)<=date(snapshot.as_of)" in html


def test_cases_do_not_disclose_future_dates_when_advancing_or_rewinding(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is only needed for this optional JS behavior check")
    path = tmp_path / "report.html"
    write_replay_html({"instruments": []}, path)
    html = path.read_text()
    render_cases = re.search(r"function renderCases\(\).*?\n", html).group(0)
    harness = r"""
const assert = require('node:assert/strict');
const select = {entries: [], replaceChildren(value){this.entries=[value]}, append(value){this.entries.push(value)}};
const byId = () => select;
const option = (value,label) => ({value,label});
const date = value => value;
const item = () => ({snapshots: [{},{},{}], summary: {cases: [
  {snapshot_index: 0, date: '2026-01-01', label: 'early'},
  {snapshot_index: 2, date: '2026-01-03', label: 'future'}
]}});
let cursor = 0;
""" + render_cases + r"""
renderCases();
assert.equal(select.entries.length, 2);
assert.equal(select.entries[1].label, '2026-01-01 · early');
assert.ok(!JSON.stringify(select.entries).includes('2026-01-03'));
cursor = 2; renderCases();
assert.equal(select.entries.length, 3);
cursor = 1; renderCases();
assert.equal(select.entries.length, 2);
assert.ok(!JSON.stringify(select.entries).includes('future'));
assert.ok(!JSON.stringify(select.entries).includes('2026-01-03'));
"""
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_instrument_payloads_roundtrip_all_evidence_and_reduce_size(tmp_path):
    import base64
    import gzip

    instrument = {
        "symbol": "SAFE", "name": "</script>研究", "group": "stocks", "status": "ok",
        "bars": [{"date": "2026-01-01", "close": 42}],
        "snapshots": [{"context": {"evidence": "重复历史证据" * 300}, "laws": {"notes": "x" * 2000}}
                      for _ in range(100)],
    }
    report = {"schema_version": 1, "instruments": [instrument]}
    path = tmp_path / "report.html"
    write_replay_html(report, path)
    html = path.read_text()
    manifest = json.loads(re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert set(manifest["instruments"][0]) == {"symbol", "name", "group", "status", "payload_id"}
    encoded = re.search(r'<script id="replay-item-0" type="application/octet-stream">(.*?)</script>', html, re.S).group(1)
    assert json.loads(gzip.decompress(base64.b64decode(encoded))) == instrument
    assert len(html.encode()) < len(json.dumps(report, ensure_ascii=False).encode()) / 10
    assert "</script>研究" not in html


def test_lazy_unpack_selection_race_failure_and_unsupported_browser(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is only needed for this optional JS behavior check")
    report = {"instruments": [
        {"symbol": "FIRST", "status": "ok", "snapshots": [{"as_of": "2026-01-01", "context": {"note": "完整证据"}}]},
        {"symbol": "SECOND", "status": "ok", "snapshots": [{"as_of": "2026-01-02"}]},
        {"symbol": "FAILED", "status": "failed", "error": "HTTP 404"},
    ]}
    path = tmp_path / "report.html"
    write_replay_html(report, path)
    html = path.read_text()
    manifest = json.loads(re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    payloads = dict(re.findall(r'<script id="(replay-item-\d+)" type="application/octet-stream">(.*?)</script>', html, re.S))
    functions = html.split("async function unpackInstrument", 1)[1].split("function snap()", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const nodes = {};
function byId(id){return nodes[id] ||= {textContent:payloads[id] || '',hidden:true,disabled:false,
classList:{add(value){this.hidden=true}, remove(value){this.hidden=false}}, replaceChildren(){}};}
const show=(id,value)=>byId(id).textContent=value;
const pretty=JSON.stringify;
const option=()=>({});
let selected=0,cursor=0,chartPoints=[],currentItem=null,generation=0,rendered=[];
const render=()=>rendered.push(currentItem.symbol);
"""
    harness = ("const instruments=" + json.dumps(manifest["instruments"]) + ";const payloads=" + json.dumps(payloads) + ";\n"
               + harness + "async function unpackInstrument" + functions + r"""
(async()=>{
const decode=unpackInstrument;
assert.deepEqual(await decode(instruments[0]), EXPECTED);
assert.deepEqual(await decode(instruments[2]), instruments[2]);
const supported=globalThis.DecompressionStream;
globalThis.DecompressionStream=undefined;
await selectInstrument(0);
assert.ok(byId('error').textContent.includes('现代浏览器'));
assert.ok(byId('error').textContent.includes('JSON'));
assert.equal(currentItem,null);
globalThis.DecompressionStream=supported;
let release;
unpackInstrument=entry=>entry.symbol==='FIRST'?new Promise(resolve=>{release=resolve}):decode(entry);
const old=selectInstrument(0);
assert.equal(currentItem,null);
const newest=selectInstrument(1);
await newest;
assert.equal(currentItem.symbol,'SECOND');
release(EXPECTED);await old;
assert.equal(currentItem.symbol,'SECOND');
assert.deepEqual(rendered,['SECOND']);
assert.equal(byId('loading').classList.hidden,true);
await selectInstrument(2);
assert.equal(currentItem.symbol,'FAILED');
assert.deepEqual(rendered,['SECOND','FAILED']);
unpackInstrument=async()=>{throw new Error('corrupt gzip')};
await selectInstrument(1);
assert.equal(currentItem,null);
assert.ok(byId('error').textContent.includes('corrupt gzip'));
assert.equal(byId('loading').classList.hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1});
""".replace("EXPECTED", json.dumps(report["instruments"][0], ensure_ascii=False)))
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
