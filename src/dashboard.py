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
    "headline", "best_headline", "mean_return", "sharpe", "win_rate",
    "avg_trades", "avg_deployed", "max_drawdown", "total_pnl",
    "codex_cost_usd", "total_cost_usd",
]


def _load_rows() -> list[dict]:
    f = ARTIFACTS / "metrics.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def _data_payload() -> dict:
    return {"metrics": CHART_METRICS, "rows": _load_rows()}


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
<div class="grid" id="charts"></div>
<table><thead><tr><th>iter</th><th>verdict</th><th>headline</th><th>mean_ret</th>
<th>win%</th><th>trades/g</th><th>cost $</th><th>cum $</th><th class="l">commit</th>
<th class="l">hypothesis</th></tr></thead><tbody id="tbody"></tbody></table>
<script>
let charts={}, metrics=[];
const f=(x,d=4)=>x==null?'':(typeof x==='number'?x.toFixed(d):x);
function ensureCharts(ms){
  if(charts._init) return; charts._init=true; metrics=ms;
  const grid=document.getElementById('charts');
  for(const m of ms){
    const c=document.createElement('div');c.className='card';
    c.innerHTML=`<h3>${m}</h3><canvas></canvas>`;grid.appendChild(c);
    charts[m]=new Chart(c.querySelector('canvas'),{type:'line',
      data:{labels:[],datasets:[{data:[],borderColor:'#58a6ff',
        backgroundColor:'rgba(88,166,255,.15)',tension:.2,pointRadius:3,fill:true}]},
      options:{animation:false,plugins:{legend:{display:false}},scales:{
        x:{ticks:{color:'#9da7b3'},grid:{color:'#21262d'}},
        y:{ticks:{color:'#9da7b3'},grid:{color:'#21262d'}}}}});
  }
}
async function tick(){
  try{
    const d=await (await fetch('/data')).json();
    ensureCharts(d.metrics);
    const its=d.rows.map(r=>r.iter);
    for(const m of d.metrics){
      charts[m].data.labels=its;
      charts[m].data.datasets[0].data=d.rows.map(r=>r[m]);
      charts[m].update();
    }
    const tb=document.getElementById('tbody');tb.innerHTML='';
    for(const r of d.rows){
      const tr=document.createElement('tr');
      tr.innerHTML=`<td>${r.iter}</td>
       <td class="${r.kept?'kept':'rev'}">${r.iter===0?'base':(r.kept?'KEPT':'rev')}</td>
       <td>${f(r.headline)}</td><td>${f(r.mean_return)}</td><td>${f(r.win_rate,2)}</td>
       <td>${f(r.avg_trades,1)}</td><td>${f(r.codex_cost_usd)}</td><td>${f(r.total_cost_usd,2)}</td>
       <td class="l">${r.commit||''}</td><td class="l">${(r.hypothesis||'').slice(0,140)}</td>`;
      tb.appendChild(tr);
    }
    const last=d.rows[d.rows.length-1];
    document.getElementById('status').textContent=
      `${d.rows.length} iterations · best headline ${last?f(last.best_headline):'-'} · total cost $${last?f(last.total_cost_usd,2):'0'} · live (refreshes 2s)`;
  }catch(e){document.getElementById('status').textContent='waiting for run…';}
}
tick(); setInterval(tick,2000);
</script></body></html>"""


# NOTE: do not run this file directly (python3 src/dashboard.py) — that puts src/ on
# sys.path and src/types.py shadows the stdlib `types` module. Use ../run_dashboard.py.
