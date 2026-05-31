"""Live TensorBoard-style dashboard for the autoresearch loop. Zero-dependency
(stdlib http.server only): serves artifacts/metrics.jsonl as a page that POLLS every
2s, so you watch each iteration's metrics + cost appear LIVE as the loop runs.

  - start_server(port) launches it in a background thread; the loop auto-starts it.
  - Open http://localhost:<port> and leave it open while the loop runs.
  - /data returns the latest metrics.jsonl as JSON; the page refetches it on a timer.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# metrics to chart (line per metric); cost shown both per-iter and cumulative.
CHART_METRICS = [
    "mean_return", "best_profit", "total_pnl", "win_rate",
    "avg_trades", "avg_deployed", "max_drawdown",
    "codex_cost_usd", "total_cost_usd",
    "train_secs", "iter_secs",
]


def _load_rows() -> list[dict]:
    f = ARTIFACTS / "metrics.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def _status() -> dict:
    f = ARTIFACTS / "status.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def _data_payload() -> dict:
    return {"metrics": CHART_METRICS, "rows": _load_rows(), "status": _status()}


def start_server(port: int = 6060) -> str:
    """Launch the live dashboard server in a daemon thread. Returns the URL."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_GET(self):
            if self.path.startswith("/data"):
                body = json.dumps(_data_payload()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = _PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Autoresearch — live</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{font-family:-apple-system,Segoe UI,sans-serif;margin:18px;background:#0d1117;color:#e6edf3}
 h1{font-size:19px;margin:0 0 2px} .sub{color:#9da7b3;font-size:12px;margin-bottom:14px}
 .dot{height:8px;width:8px;border-radius:50%;background:#3fb950;display:inline-block;margin-right:5px;animation:p 1.4s infinite}
 @keyframes p{50%{opacity:.3}}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px}
 .card h3{margin:0 0 6px;font-size:13px;color:#9da7b3;font-weight:600}
 table{width:100%;border-collapse:collapse;margin-top:18px;font-size:12px}
 th,td{border:1px solid #30363d;padding:5px 7px;text-align:right}
 th{background:#161b22;color:#9da7b3} td.l,th.l{text-align:left;max-width:340px}
 .kept{color:#3fb950;font-weight:600} .rev{color:#f85149}
</style></head><body>
<h1>Autoresearch — live experiment run</h1>
<div class="sub"><span class="dot"></span><span id="status">connecting…</span></div>
<div class="card" style="margin-bottom:14px">
  <h3>Learning curves — latest iteration (train reward, train PnL, val PnL over PPO steps)</h3>
  <canvas id="lc" style="max-height:260px"></canvas></div>
<div class="grid" id="charts"></div>
<table><thead><tr><th>iter</th><th>verdict</th><th>try</th><th>PnL/game</th><th>total PnL</th>
<th>win%</th><th>trades/g</th><th>train s</th><th>cost $</th><th>cum $</th><th class="l">commit</th>
<th class="l">hypothesis</th></tr></thead><tbody id="tbody"></tbody></table>
<script>
let charts={};
const f=(x,d=4)=>x==null?'':(typeof x==='number'?x.toFixed(d):x);
function makeChart(m){
  const grid=document.getElementById('charts');
  const c=document.createElement('div');c.className='card';
  c.innerHTML=`<h3>${m}</h3><canvas></canvas>`;grid.appendChild(c);
  charts[m]=new Chart(c.querySelector('canvas'),{type:'line',
    data:{labels:[],datasets:[{data:[],borderColor:'#58a6ff',
      backgroundColor:'rgba(88,166,255,.15)',tension:.2,pointRadius:4,
      borderWidth:2,fill:true,spanGaps:true}]},
    options:{animation:false,plugins:{legend:{display:false}},scales:{
      x:{ticks:{color:'#9da7b3'},grid:{color:'#21262d'}},
      y:{ticks:{color:'#9da7b3'},grid:{color:'#21262d'}}}}});  // y auto-scales
}
let lcChart=null;
function drawLearningCurves(row){
  if(!row) return;
  // train_reward = list per PPO iter; train_pnl/val_pnl = [[iter,val],...] checkpoints
  const tr=row.train_reward_curve||[];
  const trLabels=tr.map((_,i)=>i);
  const mkPts=(arr)=>(arr||[]).map(p=>({x:p[0],y:p[1]}));
  const datasets=[
    {label:'train reward (per PPO iter)',data:tr.map((y,i)=>({x:i,y})),
     borderColor:'#58a6ff',backgroundColor:'transparent',tension:.2,pointRadius:0,borderWidth:2},
    {label:'train PnL (greedy)',data:mkPts(row.train_pnl_curve),
     borderColor:'#3fb950',backgroundColor:'transparent',tension:.2,pointRadius:4,borderWidth:2},
    {label:'val PnL (greedy)',data:mkPts(row.val_pnl_curve),
     borderColor:'#f0883e',backgroundColor:'transparent',tension:.2,pointRadius:4,borderWidth:2},
  ];
  if(!lcChart){
    lcChart=new Chart(document.getElementById('lc'),{type:'line',
      data:{datasets},options:{animation:false,parsing:false,
        plugins:{legend:{labels:{color:'#e6edf3'}}},
        scales:{x:{type:'linear',title:{display:true,text:'PPO training iteration',color:'#9da7b3'},
          ticks:{color:'#9da7b3'},grid:{color:'#21262d'}},
          y:{title:{display:true,text:'reward / mean PnL',color:'#9da7b3'},
          ticks:{color:'#9da7b3'},grid:{color:'#21262d'}}}}});
  }else{ lcChart.data.datasets=datasets; lcChart.update(); }
}
async function tick(){
  try{
    const d=await (await fetch('/data')).json();
    const its=d.rows.map(r=>r.iter);
    for(const m of d.metrics){
      const vals=d.rows.map(r=>r[m]);
      const hasData=vals.some(v=>v!=null);
      if(!hasData) continue;               // skip all-empty metrics (no chart)
      if(!charts[m]) makeChart(m);
      charts[m].data.labels=its;
      charts[m].data.datasets[0].data=vals;
      charts[m].update();
    }
    drawLearningCurves(d.rows[d.rows.length-1]);  // latest iteration's PPO curves
    const tb=document.getElementById('tbody');tb.innerHTML='';
    for(const r of d.rows){
      const tr=document.createElement('tr');
      const verdict = r.iter===0?'base':(r.kept?'KEPT':'rev');
      tr.innerHTML=`<td>${r.iter}</td>
       <td class="${r.kept?'kept':'rev'}">${verdict}</td>
       <td>${r.attempts??''}</td>
       <td>${f(r.mean_return)}</td><td>${f(r.total_pnl,3)}</td><td>${f(r.win_rate,2)}</td>
       <td>${f(r.avg_trades,1)}</td><td>${f(r.train_secs,1)}</td><td>${f(r.codex_cost_usd)}</td><td>${f(r.total_cost_usd,2)}</td>
       <td class="l">${r.commit||''}</td><td class="l">${(r.hypothesis||'').slice(0,140)}</td>`;
      tb.appendChild(tr);
    }
    const last=d.rows[d.rows.length-1];
    const s=d.status||{};
    // heartbeat: if status.ts is >180s old and not DONE, the run is likely stuck/ended
    const ageS = s.ts ? (Date.now()/1000 - s.ts) : null;
    // a gpt-5.5 codex_proposing call legitimately takes several minutes, so only the
    // proposing phase gets a long stuck-threshold; other phases should be quick.
    const stuckAfter = (s.phase==='codex_proposing') ? 900 : 240;
    let live;
    if(s.phase==='DONE') live=`✅ DONE (best ${f(s.best_profit)}, cost $${f(s.total_cost_usd,2)})`;
    else if(ageS!=null && ageS>stuckAfter) live=`⚠️ no heartbeat ${Math.round(ageS)}s — possibly STUCK/ENDED (last: ${s.phase} iter ${s.iter})`;
    else if(s.phase==='codex_proposing') live=`▶ iter ${s.iter}/${s.total_iters} · Codex reasoning (takes a few min) · ${Math.round(ageS||0)}s`;
    else if(s.phase) live=`▶ iter ${s.iter}/${s.total_iters} · ${s.phase} · upd ${s.updated_at}`;
    else live='live (refreshes 2s)';
    document.getElementById('status').textContent=
      `${d.rows.length} rows · best PnL/game ${last?f(last.best_profit):'-'} · cost $${last?f(last.total_cost_usd,2):'0'} · ${live}`;
  }catch(e){document.getElementById('status').textContent='waiting for run…';}
}
tick(); setInterval(tick,2000);
</script></body></html>"""


# NOTE: do not run this file directly (python3 src/dashboard.py) — that puts src/ on
# sys.path and src/types.py shadows the stdlib `types` module. Use ../run_dashboard.py.
