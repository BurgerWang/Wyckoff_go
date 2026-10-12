"""Dependency-free, offline viewer for causal daily-bar research snapshots."""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path


def write_replay_html(report: dict, path: Path) -> None:
    """Write a single HTML file; analysis comes exclusively from each snapshot.

    Input follows replay report schema version 1. Instruments are separately
    gzip-packed and loaded on selection. No final phase/event summary
    is used to annotate earlier bars. Embedded data is escaped for HTML raw text.
    """
    manifest = {key: value for key, value in report.items() if key != "instruments"}
    manifest["instruments"] = []
    packed = []
    for index, instrument in enumerate(report.get("instruments", [])):
        if instrument.get("status") == "failed":
            manifest["instruments"].append(instrument)
            continue
        payload_id = f"replay-item-{index}"
        entry = {key: instrument.get(key) for key in ("symbol", "name", "group", "status")}
        entry["payload_id"] = payload_id
        manifest["instruments"].append(entry)
        raw = json.dumps(instrument, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
        packed.append(f'<script id="{payload_id}" type="application/octet-stream">{encoded}</script>')
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = (payload.replace("&", "\\u0026").replace("<", "\\u003c")
               .replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = _HTML.replace("__PACKED_ITEMS__", "\n".join(packed)).replace("__REPORT_JSON__", payload)
    path.write_text(html, encoding="utf-8")


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wyckoff 逐根量价回放</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0c1422;color:#e4edf9}body{max-width:1440px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 8px}h2{font-size:17px;margin:0 0 12px}.muted{color:#a4b3c9;font-size:13px;line-height:1.6}.notice{background:#1b2940;border-left:3px solid #e8bf69;padding:12px;margin:14px 0;line-height:1.6}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}select,button,input{font:inherit;color:inherit;background:#18263c;border:1px solid #435572;border-radius:6px;padding:8px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}input[type=range]{flex:1;min-width:170px;padding:0}label{font-size:13px}.panel{background:#121e30;border:1px solid #2d3c53;border-radius:10px;padding:16px;margin-bottom:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.hypotheses{display:grid;grid-template-columns:1fr 1fr;gap:12px}.hypothesis{border:1px solid #35475f;border-radius:8px;padding:12px}canvas{display:block;width:100%;height:480px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #2d3c53}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.65;margin:0}details{margin-top:10px}summary{cursor:pointer}#error{color:#ffb8b2;white-space:pre-wrap}.hidden{display:none}#chart-readout{min-height:23px;color:#a4b3c9;font-size:13px}.scroll{overflow:auto;max-height:320px}@media(max-width:800px){body{padding:12px}.grid{grid-template-columns:1fr}.hypotheses{grid-template-columns:1fr}canvas{height:390px}}
</style></head><body>
<h1 id="report-title">Wyckoff 逐根量价回放</h1><div id="report-meta" class="muted"></div>
<div class="notice">阶段、事件及案例均为<strong>规则候选</strong>，缺少独立专家标签；confidence / strength 表示证据等级，<strong>不是统计概率</strong>。每个回放点仅呈现截至当日已确认的信息，不构成交易建议。向后切换时间后，未来 K 线与未来确认事件不会显示。</div>
<div class="controls"><label for="symbol-select">研究标的</label><select id="symbol-select"></select><label for="case-select">案例定位</label><select id="case-select"></select></div>
<div id="loading" class="panel hidden" role="status" aria-live="polite"></div>
<div id="error" class="panel hidden" role="alert"></div>
<div id="replay-body"><div class="controls"><button id="previous">← 上一步</button><button id="next">下一步 →</button><input id="replay-slider" type="range" min="0" value="0" aria-label="逐根回放进度"><strong id="as-of"></strong><span id="replay-position" class="muted"></span></div>
<div class="panel"><h2>价格与成交量 · 最近 160 根已可见 K 线</h2><div id="range-meta" class="muted"></div><canvas id="price-chart" aria-label="截至当前回放日的量价图"></canvas><div id="chart-readout"></div></div>
<div class="grid"><section class="panel"><h2 id="phase-title">阶段规则候选</h2><div id="phase-notes" class="muted"></div><div id="hypotheses" class="hypotheses"></div><details><summary>阶段完整证据</summary><pre id="phase-details"></pre></details></section>
<section class="panel"><h2>当前已确认事件</h2><div class="muted">按事件日期及可用日期双重过滤；反向运动、支点等确认可晚于发生日。</div><div class="scroll"><table><thead><tr><th>发生日</th><th>可用日</th><th>规则事件</th><th>价格</th></tr></thead><tbody id="event-table"></tbody></table></div></section>
<section class="panel"><h2>准备度 · 九测与五步</h2><div id="readiness-summary" class="muted"></div><details><summary>逐项依据与缺失资料</summary><pre id="readiness"></pre></details></section><section class="panel"><h2>上下文、三大定律与计数</h2><div id="context-summary" class="muted"></div><details><summary>上下文证据</summary><pre id="context"></pre></details><details><summary>三大定律证据</summary><pre id="laws"></pre></details><details><summary>P&amp;F 计数证据</summary><pre id="count"></pre></details></section></div></div>
<section class="panel"><h2>数据来源与覆盖</h2><div id="data-summary" class="muted"></div><details><summary>完整来源元数据</summary><pre id="data-source"></pre></details><div id="warnings" class="muted"></div></section><section class="panel"><h2>研究限制</h2><div id="limitations" class="muted"></div><details><summary>本市场配置</summary><pre id="configuration"></pre></details></section>
__PACKED_ITEMS__
<script id="replay-data" type="application/json">__REPORT_JSON__</script>
<script>
'use strict';
const report=JSON.parse(document.getElementById('replay-data').textContent);
const byId=id=>document.getElementById(id), instruments=report.instruments||[];
let selected=0, cursor=0, chartPoints=[], currentItem=null, generation=0;
const show=(id,value)=>{byId(id).textContent=value??'未提供';};
const pretty=value=>value==null?'不可用 / 未提供':JSON.stringify(value,null,2);
const textLines=(id,lines)=>show(id,(lines||[]).join('\n'));
const date=value=>String(value??'').slice(0,10);
const number=value=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('zh-CN',{maximumFractionDigits:4}):'—';
function option(value,label){const node=document.createElement('option');node.value=value;node.textContent=label;return node;}
const profile=report.profile||{};
show('report-title',`${profile.label||profile.id||'市场'} · Wyckoff 逐根量价回放`);
show('report-meta',`${profile.asset_class||''} / ${profile.market_type||''} · ${profile.timeframe||'1d'} · 价格币种 ${profile.price_currency||'未提供'} · 成功 ${report.summary?.success??0} / 失败 ${report.summary?.failed??0}。股票与数字货币在独立报告中研究。`);
textLines('limitations',report.limitations);show('configuration',pretty(report.config));
instruments.forEach((item,index)=>byId('symbol-select').append(option(index,`${item.symbol} · ${item.name||''}${item.status==='failed'?' [获取失败]':''}`)));
function item(){return currentItem||{};}
async function unpackInstrument(entry){
if(!entry.payload_id)return entry;
if(typeof DecompressionStream==='undefined')throw new Error('当前浏览器不支持 DecompressionStream。请使用支持 gzip 解压的现代浏览器，或读取同目录完整研究 JSON 文件。');
const encoded=byId(entry.payload_id).textContent;
const bytes=Uint8Array.from(atob(encoded),character=>character.charCodeAt(0));
const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
return JSON.parse(await new Response(stream).text());
}
async function selectInstrument(index){
const ticket=++generation;selected=index;cursor=0;currentItem=null;chartPoints=[];
const entry=instruments[index]||{symbol:'无标的',status:'failed',error:'报告没有标的'};
byId('replay-body').classList.add('hidden');byId('error').classList.add('hidden');byId('loading').classList.remove('hidden');
show('loading',`正在解压 ${entry.symbol} 的逐根证据…`);byId('case-select').replaceChildren(option('', '加载中…'));byId('case-select').disabled=true;
show('data-source',pretty({symbol:entry.symbol,name:entry.name,group:entry.group,status:entry.status}));show('warnings','');
show('data-summary',`${entry.symbol} · ${entry.name||''} · 正在加载`);
try{const decoded=await unpackInstrument(entry);if(ticket!==generation)return;currentItem=decoded;render();}
catch(error){if(ticket!==generation)return;show('error',`${entry.symbol}：加载失败。${error.message}`);byId('error').classList.remove('hidden');}
finally{if(ticket===generation){byId('loading').classList.add('hidden');byId('case-select').disabled=false;}}
}
function snap(){return (item().snapshots||[])[cursor];}
function visibleEvents(snapshot){return (snapshot.events||[]).filter(event=>event.date&&event.available_date&&date(event.date)<=date(snapshot.as_of)&&date(event.available_date)<=date(snapshot.as_of));}
function renderCases(){const select=byId('case-select');select.replaceChildren(option('', '选择规则候选案例日期…'));(item().summary?.cases||[]).forEach(c=>{if(!Number.isInteger(c.snapshot_index)||c.snapshot_index<0||c.snapshot_index>cursor||!item().snapshots?.[c.snapshot_index])return;select.append(option(c.snapshot_index,`${date(c.date)} · ${c.label||'规则候选'}`));});}
function render(){const current=item(), snapshots=current.snapshots||[], snapshot=snap();const failed=current.status==='failed'||!snapshot;
byId('error').classList.toggle('hidden',!failed);byId('replay-body').classList.toggle('hidden',failed);
show('error',`${current.symbol||'无标的'}：${current.error||(snapshots.length?'回放快照不可用':'没有可回放快照')}`);
show('data-source',pretty({symbol:current.symbol,name:current.name,group:current.group,status:current.status,...(current.data||{}),error:current.error}));
const data=current.data||{};show('data-summary',`${current.symbol} · ${data.source||'来源不可用'} · ${data.bars??0} 根日线 · ${date(data.start)} 至 ${date(data.end)} · 价格 ${data.price_currency||profile.price_currency||''} / 成交量 ${data.volume_unit||'未提供'} · ${data.adjusted===true?'下载时复权':data.adjusted===false?'未复权':'复权状态未提供'}`);
textLines('warnings',[...(current.data?.warnings||[]),...(snapshot?.warnings||[])]);renderCases();if(failed)return;
const slider=byId('replay-slider');slider.max=snapshots.length-1;slider.value=cursor;byId('previous').disabled=cursor===0;byId('next').disabled=cursor===snapshots.length-1;
show('as-of',`截至 ${date(snapshot.as_of)}`);show('replay-position',`${cursor+1} / ${snapshots.length} · 原始 bar ${snapshot.bar_index+1}`);
const phase=snapshot.phase||{};show('phase-title',`阶段规则候选：${phase.regime||'unknown'} / ${phase.phase||'?'} · ${phase.status||'不可用'}`);
show('phase-notes',`${(phase.notes||[]).join('；')}\n证据等级 confidence：${phase.confidence||'未提供'}（不是概率）`);show('phase-details',pretty(phase));
byId('hypotheses').replaceChildren();for(const [key,label] of [['accumulation','吸筹假设'],['distribution','派发假设']]){const box=document.createElement('div');box.className='hypothesis';const title=document.createElement('strong');title.textContent=label;const content=document.createElement('pre');content.textContent=pretty(phase.hypotheses?.[key]);box.append(title,content);byId('hypotheses').append(box);}
const table=byId('event-table');table.replaceChildren();const events=visibleEvents(snapshot);for(const e of events){const row=document.createElement('tr');for(const v of [date(e.date),date(e.available_date),e.event,number(e.price)]){const td=document.createElement('td');td.textContent=v;row.append(td);}table.append(row);}if(!events.length){const row=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent='截至当日无已确认事件';row.append(td);table.append(row);}
for(const key of ['readiness','context','laws','count'])show(key,pretty(snapshot[key]));
textLines('readiness-summary',[['nine_buying','买方九测'],['nine_selling','卖方九测'],['five_steps','五步检查']].map(([key,label])=>{const checks=snapshot.readiness?.[key]||[];return `${label}：满足 ${checks.filter(c=>c.status==='met').length} / 未满足 ${checks.filter(c=>c.status==='not_met').length} / 缺资料 ${checks.filter(c=>c.status==='unavailable').length}`;}));
const direction=value=>({up:'向上',down:'向下',flat:'横向',unknown:'未知',mixed:'混合',stronger:'较强',weaker:'较弱'})[value]||value||'不可用';
textLines('context-summary',[`此前趋势：${direction(snapshot.context?.prior_trend?.direction)}；确认价格结构：${direction(snapshot.context?.price_structure?.direction)}`,`相对基准：${snapshot.context?.relative_strength?.status==='unavailable'?'缺资料':direction(snapshot.context?.relative_strength?.direction)}`,`供需量价代理：${direction(snapshot.laws?.supply_demand?.direction)}；水平计数：${snapshot.count?.status==='available'?'有条件情景':'不可用（展开查看原因）'}`]);
const tr=snapshot.tr;show('range-meta',tr?`当前区间 ${tr.range_id||''} · ${tr.lifecycle||''} · 上界 ${number(tr.top)} / 下界 ${number(tr.bottom)} · 确认日 ${date(tr.confirmed_date)||'未提供'}`:'截至当日没有已确认交易区间');
draw(snapshot,events);}
function draw(snapshot,events){const canvas=byId('price-chart'),width=canvas.clientWidth||900,height=canvas.clientHeight||480,dpr=window.devicePixelRatio||1;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
// Read only the selected prefix. A date guard also rejects inconsistent input.
const all=item().bars||[],end=Math.min(snapshot.bar_index,all.length-1),first=Math.max(0,end-159);
const bars=all.slice(first,end+1).filter(b=>date(b.date)<=date(snapshot.as_of));chartPoints=[];show('chart-readout','移动指针查看已可见 K 线的 OHLCV');if(!bars.length)return;
const left=14,right=84,top=27,bottom=height-30,priceBottom=height*.70,volumeTop=height*.77,plotWidth=width-left-right,step=plotWidth/bars.length;
const tr=snapshot.tr,confirmed=tr&&tr.confirmed_date&&date(tr.confirmed_date)<=date(snapshot.as_of);
let lows=bars.map(b=>b.low),highs=bars.map(b=>b.high);if(confirmed){lows.push(tr.bottom);highs.push(tr.top);}const min=Math.min(...lows),max=Math.max(...highs),padding=(max-min)*.06||1,lo=min-padding,hi=max+padding;
const y=p=>top+(hi-p)/(hi-lo)*(priceBottom-top),volumeMax=Math.max(...bars.map(b=>b.volume),1);
ctx.font='11px system-ui';ctx.textAlign='left';for(let i=0;i<5;i++){const p=lo+(hi-lo)*i/4,py=y(p);ctx.strokeStyle='#29374d';ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(width-right,py);ctx.stroke();ctx.fillStyle='#a4b3c9';ctx.fillText(number(p),width-right+8,py+4);}
if(confirmed){for(const [p,label] of [[tr.top,'上界'],[tr.bottom,'下界']]){ctx.strokeStyle='#dfbd73';ctx.setLineDash([5,5]);ctx.beginPath();const start=bars.findIndex(b=>date(b.date)>=date(tr.start_date));ctx.moveTo(left+Math.max(0,start)*step,y(p));ctx.lineTo(width-right,y(p));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#dfbd73';ctx.fillText(label,width-right+8,y(p)-6);}}
bars.forEach((bar,index)=>{const x=left+(index+.5)*step,color=bar.close>=bar.open?'#46c5a0':'#ee7b87';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(x,y(bar.high));ctx.lineTo(x,y(bar.low));ctx.stroke();const body=Math.max(1,Math.abs(y(bar.open)-y(bar.close)));ctx.fillRect(x-Math.max(1,step*.3),Math.min(y(bar.open),y(bar.close)),Math.max(1,step*.6),body);ctx.globalAlpha=.65;const vh=Math.max(0,bar.volume)/volumeMax*(bottom-volumeTop);ctx.fillRect(x-Math.max(1,step*.3),bottom-vh,Math.max(1,step*.6),vh);ctx.globalAlpha=1;chartPoints.push({x,bar});});
const used=new Map();for(const e of events){const index=bars.findIndex(b=>date(b.date)===date(e.date));if(index<0)continue;const x=left+(index+.5)*step,slot=used.get(index)||0;used.set(index,slot+1);const py=Math.max(top+10,Math.min(priceBottom-10,y(e.price??bars[index].high)-12-slot*13));ctx.fillStyle='#e7c677';ctx.beginPath();ctx.arc(x,py+5,2.5,0,Math.PI*2);ctx.fill();ctx.save();ctx.textAlign='center';ctx.fillText(e.event,x,py);ctx.restore();}
ctx.fillStyle='#a4b3c9';ctx.fillText('成交量 '+number(volumeMax),left,volumeTop-7);const n=Math.min(5,bars.length);for(let i=0;i<n;i++){const idx=Math.round(i*(bars.length-1)/Math.max(1,n-1)),x=left+(idx+.5)*step;ctx.textAlign=i===0?'left':i===n-1?'right':'center';ctx.fillText(date(bars[idx].date),x,height-9);}}
byId('symbol-select').addEventListener('change',event=>{void selectInstrument(Number(event.target.value));});
byId('case-select').addEventListener('change',event=>{if(event.target.value==='')return;cursor=Number(event.target.value);render();});
byId('previous').addEventListener('click',()=>{cursor=Math.max(0,cursor-1);render();});byId('next').addEventListener('click',()=>{cursor=Math.min((item().snapshots||[]).length-1,cursor+1);render();});
byId('replay-slider').addEventListener('input',event=>{cursor=Number(event.target.value);render();});
byId('price-chart').addEventListener('mousemove',event=>{if(!chartPoints.length)return;const x=event.clientX-event.currentTarget.getBoundingClientRect().left;const point=chartPoints.reduce((a,b)=>Math.abs(a.x-x)<Math.abs(b.x-x)?a:b);const b=point.bar;show('chart-readout',`${date(b.date)} · 开 ${number(b.open)} / 高 ${number(b.high)} / 低 ${number(b.low)} / 收 ${number(b.close)} / 量 ${number(b.volume)}`);});
window.addEventListener('resize',()=>{if(snap())draw(snap(),visibleEvents(snap()));});void selectInstrument(0);
</script></body></html>'''
