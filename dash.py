#!/usr/bin/env python3
"""dash.py - the war-drive dashboard: locate.py on a phone screen.

Serves one dark page on the LAN (default 0.0.0.0:8088) that starts a stop,
watches it live, and shows what the receiver sees: capture progress, each
bird as its ephemeris decodes (PRN, C/N0, word parity, anchors, clock rate),
every epoch solution as it lands, and the final fix.

It deliberately does NOT touch the proven capture->solve pipeline: locate.py
runs unmodified as a subprocess and this file parses the same progress lines
a terminal shows. The dashboard cannot break the fix.

PRIVACY: the standing law keeps coordinates out of the repo, logs and
reports; your own screen is where they are ALLOWED to appear. This page
shows the fix to anyone on the same network segment, so run it on your home
LAN or the car hotspot, not a network you share with strangers.

  python3 dash.py                # then open http://<pi>:8090 on the phone
  python3 dash.py --port 9000 --bind 127.0.0.1
"""
import argparse
import json
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "lab_local"
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_UTC_LEAP = 18                     # GPS-UTC offset since 2017-01-01

# ---------------------------------------------------------------- state --
LOCK = threading.Lock()
PROC = None                           # running locate.py, or None


def _fresh_state():
    return {
        "phase": "idle",              # idle/capturing/acquiring/decoding/
                                      # solving/solved/nofix/no-sky/
                                      # incomplete/error/aborted
        "mode": None,                 # "live" | "replay"
        "started_at": None,
        "secs_wanted": None,
        "bias_t": None,
        "capture": None,              # {got, want, gb}
        "capture_file": None,
        "disk": None,                 # {free_gb, more}
        "sky": None,                  # {strong, weak}
        "sats": {},                   # prn -> {...}
        "epochs": [],                 # [{t, rms, alt}]
        "summary": None,              # the [fix] SOLVED / NO FIX line
        "fix": None,                  # fix_result.json when fresh
        "rc": None,
        "log": [],
    }


STATE = _fresh_state()
LOGTAIL = deque(maxlen=200)

# The exact strings locate.py and fix.py print, matched exactly.  If a
# format changes upstream the row just stops updating -- the run itself is
# never affected.
RX = [
    ("bias", re.compile(r"\[locate\] bias-T ON via (\w+)")),
    ("bias_warn", re.compile(r"\[locate\] WARNING: could not confirm bias-T")),
    ("cap_prog", re.compile(
        r"\[locate\] captured\s+(\d+) s of (\d+) s \(([\d.]+) GB\)")),
    ("cap_done", re.compile(
        r"\[locate\] captured (\d+) s -> (\S+) \(([\d.]+) GB\)")),
    ("disk", re.compile(
        r"\[locate\] disk: ([\d.]+) GB free -- about (\d+) more")),
    ("sky", re.compile(r"SKY VIEW: (\d+) strong \+ (\d+) weak")),
    ("acq", re.compile(r"^\s+PRN\s?(\d+)\s+metric ([\d.]+)\s+STRONG")),
    ("decoding", re.compile(r">= 4 strong satellites - decoding")),
    ("no_sky", re.compile(r"Need >= 4 strong satellites .*have (\d+)")),
    ("parity", re.compile(r"\[rel\] word parity: (\d+)/(\d+) clean")),
    ("eph", re.compile(
        r"^\s+PRN(\d+): ephemeris (\w+), (\d+) anchors, C/N0 (\d+) dB-Hz")),
    ("cutover", re.compile(r"^\s+PRN(\d+): toe \d+ off the 2-hour grid.*KEPT")),
    ("anchor", re.compile(
        r"^\s+PRN(\d+): anchor fit residual ([\d.]+) us "
        r"\(clock rate ([+\-0-9.eE]+)\)")),
    ("epoch", re.compile(
        r"^\s+T=\s*([\d.]+)s: rms ([\d.,]+) m, alt (-?[\d,]+) m")),
    ("solved", re.compile(r"\[fix\] SOLVED with .*")),
    ("nofix", re.compile(r"\[fix\] NO FIX .*")),
    ("incomplete", re.compile(r"\[locate\] not enough complete orbits")),
]


def _num(s):
    return float(s.replace(",", ""))


def _sat(st, prn):
    return st["sats"].setdefault(int(prn), {"prn": int(prn)})


def _parse_line(st, line, pending):
    for name, rx in RX:
        m = rx.search(line)
        if not m:
            continue
        if name == "bias":
            st["bias_t"] = True
        elif name == "bias_warn":
            st["bias_t"] = False
        elif name == "cap_prog":
            st["phase"] = "capturing"
            st["capture"] = {"got": int(m[1]), "want": int(m[2]),
                             "gb": float(m[3])}
        elif name == "cap_done":
            st["capture"] = {"got": int(m[1]), "want": int(m[1]),
                             "gb": float(m[3])}
            st["capture_file"] = m[2].rsplit("/", 1)[-1]
            st["phase"] = "acquiring"
        elif name == "disk":
            st["disk"] = {"free_gb": float(m[1]), "more": int(m[2])}
        elif name == "sky":
            st["sky"] = {"strong": int(m[1]), "weak": int(m[2])}
        elif name == "acq":
            _sat(st, m[1])["metric"] = float(m[2])
        elif name == "decoding":
            st["phase"] = "decoding"
        elif name == "no_sky":
            st["phase"] = "no-sky"
            st["sky"] = st["sky"] or {"strong": int(m[1]), "weak": 0}
        elif name == "parity":
            pending["parity"] = f"{m[1]}/{m[2]}"    # attaches to next eph line
        elif name == "eph":
            s = _sat(st, m[1])
            s.update(eph=m[2], anchors=int(m[3]), cn0=int(m[4]),
                     parity=pending.pop("parity", None))
        elif name == "cutover":
            _sat(st, m[1])["cutover"] = True
        elif name == "anchor":
            s = _sat(st, m[1])
            s.update(anchor_res_us=float(m[2]), clock_rate=float(m[3]))
        elif name == "epoch":
            st["phase"] = "solving"
            st["epochs"].append({"t": float(m[1]), "rms": _num(m[2]),
                                 "alt": _num(m[3])})
        elif name == "solved":
            st["phase"] = "solved"
            st["summary"] = line.strip()
        elif name == "nofix":
            st["phase"] = "nofix"
            st["summary"] = line.strip()
        elif name == "incomplete":
            st["phase"] = "incomplete"
        return


def _sv_time(eph, tow):
    """GPS TOW + week -> ISO UTC, so a bird's clock reads as a real time."""
    wn = eph.get("WN")
    if wn is None:
        return None
    # 10-bit broadcast week rolls over every 1024 weeks; pin to the rollover
    # era that contains the present (2019-04-07 started era 2).
    wn = int(wn) % 1024 + 2048
    t = GPS_EPOCH + timedelta(weeks=wn, seconds=float(tow) - GPS_UTC_LEAP)
    return t.strftime("%Y-%m-%d %H:%M:%S") + "Z"


def _enrich_after_run(st, t0):
    """Fold in what the pipeline wrote to lab_local: the per-bird transmit
    clocks from prs_cache.json and the fix itself -- but only when the file
    is from THIS run, never a stale one from a previous stop."""
    try:
        pc_path = LOCAL / "prs_cache.json"
        if pc_path.stat().st_mtime >= t0:
            for c in json.loads(pc_path.read_text()):
                s = _sat(st, c["prn"])
                eph = c.get("eph", {})
                s["af0_us"] = round(eph.get("af0", 0.0) * 1e6, 3)
                s["tx_tow"] = round(c["t_gps_tx_coarse"], 3)
                s["sv_utc"] = _sv_time(eph, c["t_gps_tx_coarse"])
    except Exception:                                            # noqa: BLE001
        pass
    try:
        fr_path = LOCAL / "fix_result.json"
        if fr_path.stat().st_mtime >= t0:
            fix = json.loads(fr_path.read_text())
            st["fix"] = fix
            for prn, el in zip(fix.get("birds", []), fix.get("el_deg", [])):
                _sat(st, prn)["el_deg"] = el
    except Exception:                                            # noqa: BLE001
        pass


def _latest_fix_on_disk():
    """The most recent fix for the idle screen, marked stale=True so the UI
    labels it 'last stop' rather than pretending it is current."""
    try:
        fix = json.loads((LOCAL / "fix_result.json").read_text())
        fix["stale"] = True
        return fix
    except Exception:                                            # noqa: BLE001
        return None


def _run(cmd, mode, secs):
    global PROC, STATE
    t0 = time.time()
    with LOCK:
        STATE = _fresh_state()
        STATE.update(phase="capturing" if mode == "live" else "acquiring",
                     mode=mode, secs_wanted=secs, started_at=t0)
    LOGTAIL.clear()
    proc = subprocess.Popen(cmd, cwd=HERE, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with LOCK:
        PROC = proc
    pending = {}
    noise = re.compile(r"RtApi|ALSA lib|snd_pcm|Jack server|^\s*$")
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not noise.search(line):       # audio-driver probe spam, not ours
            LOGTAIL.append(line)
        with LOCK:
            _parse_line(STATE, line, pending)
    rc = proc.wait()
    with LOCK:
        PROC = None
        STATE["rc"] = rc
        if STATE["phase"] in ("capturing", "acquiring", "decoding", "solving"):
            # 130 = locate.py's clean signal exit; negative = killed by signal
            STATE["phase"] = "aborted" if (rc < 0 or rc == 130) else "error"
        _enrich_after_run(STATE, t0)


def start_run(secs=None, replay=False):
    with LOCK:
        if PROC is not None:
            return False, "a run is already in progress"
    if replay:
        caps = sorted(LOCAL.glob("sky_capture_*.cs16"),
                      key=lambda p: p.stat().st_mtime)
        if not caps:
            return False, "no sky_capture_*.cs16 in lab_local to replay"
        cmd = [sys.executable, "-u", "locate.py", "--iq", str(caps[-1])]
        mode = "replay"
    else:
        secs = int(secs or 180)
        cmd = [sys.executable, "-u", "locate.py", "--secs", str(secs),
               "--antenna", "Antenna B"]
        mode = "live"
    threading.Thread(target=_run, args=(cmd, mode, secs), daemon=True).start()
    return True, "started"


def abort_run():
    with LOCK:
        proc = PROC
    if proc is None:
        return False, "nothing running"

    # SIGINT first: locate.py turns it into a clean stop that de-powers the
    # bias-T. But the SDRplay library has been seen swallowing signals
    # (measured 8/23: an abort mid-capture was simply ignored), so escalate
    # to SIGTERM and, as a last resort, SIGKILL rather than leave the user a
    # dead Abort button in the car.
    def _escalate(p):
        for sig, grace in ((signal.SIGINT, 6), (signal.SIGTERM, 6)):
            p.send_signal(sig)
            try:
                p.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                pass
        p.kill()

    threading.Thread(target=_escalate, args=(proc,), daemon=True).start()
    return True, "aborting..."


# ------------------------------------------------------------------ http --
PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>numpy-gps dash</title>
<style>
 :root { color-scheme: dark; }
 body { background:#0b0f14; color:#c9d4e0; font:15px/1.45 ui-monospace,
        SFMono-Regular,Menlo,Consolas,monospace; margin:0; padding:12px; }
 h1 { font-size:18px; margin:0; }
 .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
 .pill { padding:2px 10px; border-radius:99px; font-weight:bold;
         background:#22303f; }
 .pill.capturing,.pill.acquiring,.pill.decoding,.pill.solving
   { background:#7a5c00; color:#fff; }
 .pill.solved { background:#0f6f3f; color:#fff; }
 .pill.nofix,.pill.no-sky,.pill.error,.pill.incomplete,.pill.aborted
   { background:#8b1e2d; color:#fff; }
 button,select { background:#1a2530; color:#c9d4e0; border:1px solid #33475c;
        border-radius:6px; padding:8px 14px; font:inherit; }
 button:active { background:#33475c; }
 .card { background:#111820; border:1px solid #223042; border-radius:10px;
         padding:10px 12px; margin-top:10px; }
 .bar { height:14px; background:#1a2530; border-radius:7px; overflow:hidden; }
 .bar>div { height:100%; background:#2f81f7; width:0; transition:width .5s; }
 table { border-collapse:collapse; width:100%; font-size:13px; }
 th,td { text-align:right; padding:3px 6px; border-bottom:1px solid #1d2a38;
         white-space:nowrap; }
 th:first-child,td:first-child { text-align:left; }
 .fix-big { font-size:22px; font-weight:bold; }
 .ok { color:#4dd88a; } .bad { color:#ff6b7f; } .dim { color:#5d6f82; }
 #log { font-size:12px; color:#8fa3b8; white-space:pre-wrap; }
 .grid { display:grid; grid-template-columns:auto 1fr; gap:2px 14px; }
</style></head><body>
<div class="row">
 <h1>&#128752; numpy-gps</h1>
 <span id="phase" class="pill">idle</span>
 <span id="clock" class="dim"></span>
</div>
<div class="card row">
 <select id="secs"><option>90</option><option selected>180</option>
  <option>300</option></select>
 <button onclick="start(false)">Start stop</button>
 <button onclick="start(true)">Replay last capture</button>
 <button onclick="post('/abort')">Abort</button>
 <span id="msg" class="dim"></span>
</div>
<div class="card" id="capcard" hidden>
 <div class="row" style="justify-content:space-between">
  <span>capture</span><span id="captext" class="dim"></span></div>
 <div class="bar"><div id="capbar"></div></div>
</div>
<div class="card" id="skycard" hidden><span id="skytext"></span></div>
<div class="card" id="satcard" hidden>
 <table><thead><tr><th>bird</th><th>acq</th><th>C/N0</th><th>eph</th>
  <th>parity</th><th>anch</th><th>clk rate</th><th>af0 &micro;s</th>
  <th>el&deg;</th><th>SV clock (UTC)</th></tr></thead>
 <tbody id="sats"></tbody></table>
</div>
<div class="card" id="fixcard" hidden></div>
<div class="card" id="epochcard" hidden>
 <div>epoch solves <span id="epochn" class="dim"></span></div>
 <div id="epochs" style="font-size:12px"></div>
</div>
<div class="card dim" id="footer"></div>
<details class="card"><summary class="dim">pipeline log</summary>
 <div id="log"></div></details>
<script>
const $ = id => document.getElementById(id);
function post(u,b){ fetch(u,{method:'POST',body:JSON.stringify(b||{})})
  .then(r=>r.json()).then(j=>$('msg').textContent=j.msg); }
function start(replay){ post('/start',{secs:+$('secs').value,replay}); }
function fmt(x,d){ return x==null?'—':(+x).toFixed(d); }
setInterval(()=>{ $('clock').textContent =
  new Date().toISOString().slice(0,19).replace('T',' ')+'Z'; },1000);
async function tick(){
 let s; try { s = await (await fetch('/status.json')).json(); }
 catch(e){ $('phase').textContent='dash offline'; return; }
 const ph=$('phase'); ph.textContent=s.phase; ph.className='pill '+s.phase;
 if(s.capture){ $('capcard').hidden=false;
   $('capbar').style.width=(100*s.capture.got/s.capture.want)+'%';
   $('captext').textContent=`${s.capture.got}/${s.capture.want} s  `+
     `${s.capture.gb.toFixed(2)} GB`+
     (s.capture_file?`  → ${s.capture_file}`:'');
 } else if(s.phase=='capturing' && s.elapsed!=null && s.secs_wanted){
   $('capcard').hidden=false;
   const el=Math.min(s.elapsed,s.secs_wanted);
   $('capbar').style.width=(100*el/s.secs_wanted)+'%';
   $('captext').textContent=`~${el.toFixed(0)}/${s.secs_wanted} s`;
 } else $('capcard').hidden=true;
 if(s.sky){ $('skycard').hidden=false;
   $('skytext').innerHTML=`SKY VIEW: <b>${s.sky.strong} strong</b> + `+
     `${s.sky.weak} weak satellites`+
     (s.bias_t===false?' &mdash; <span class="bad">bias-T UNCONFIRMED</span>'
      :s.bias_t?' &mdash; <span class="ok">bias-T on</span>':'');
 } else $('skycard').hidden=true;
 const prns=Object.keys(s.sats).sort((a,b)=>a-b);
 $('satcard').hidden=!prns.length;
 $('sats').innerHTML=prns.map(p=>{const b=s.sats[p];return `<tr>
  <td>PRN ${p}${b.cutover?' <span class="dim">(cutover)</span>':''}</td>
  <td>${fmt(b.metric,1)}</td><td>${b.cn0??'—'}</td>
  <td>${b.eph=='COMPLETE'?'<span class="ok">✓</span>':(b.eph??'…')}</td>
  <td>${b.parity??'—'}</td><td>${b.anchors??'—'}</td>
  <td>${b.clock_rate!=null?b.clock_rate.toExponential(2):'—'}</td>
  <td>${fmt(b.af0_us,2)}</td><td>${fmt(b.el_deg,0)}</td>
  <td>${b.sv_utc??'—'}</td></tr>`;}).join('');
 if(s.epochs.length){ $('epochcard').hidden=false;
   $('epochn').textContent=`(${s.epochs.length})`;
   $('epochs').textContent=s.epochs.map(e=>
     `T=${e.t.toFixed(0)}s rms ${e.rms.toFixed(0)}m alt ${e.alt.toFixed(0)}m`)
     .join('   ');
 } else $('epochcard').hidden=true;
 const f=s.fix, fc=$('fixcard');
 if(f){ fc.hidden=false;
  fc.innerHTML=(f.valid
    ?`<span class="ok fix-big">FIX${f.stale?' (last stop)':''}</span>`
    :`<span class="bad fix-big">NOT A FIX (valid=false)</span>`)+
   `<div class="grid" style="margin-top:6px">`+
   `<span class="dim">lat</span><b>${f.lat.toFixed(6)}&deg;</b>`+
   `<span class="dim">lon</span><b>${f.lon.toFixed(6)}&deg;</b>`+
   `<span class="dim">alt</span><span>${f.alt_m.toFixed(0)} m</span>`+
   `<span class="dim">birds</span><span>${f.birds.join(', ')}</span>`+
   `<span class="dim">scatter</span><span>${fmt(f.scatter_m,1)} m `+
     `(the honest number)</span>`+
   `<span class="dim">rms</span><span>${fmt(f.resid_rms_m,1)} m</span>`+
   `<span class="dim">capture</span><span>${(f.capture||'')
     .split('/').pop()}</span></div>`;
 } else fc.hidden=true;
 $('footer').textContent=(s.disk?
   `disk ${s.disk.free_gb.toFixed(1)} GB free — ~${s.disk.more} more `+
   `stops this length. `:'')+(s.summary||'');
 $('log').textContent=s.log.join('\\n');
}
setInterval(tick,1000); tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/status.json":
            with LOCK:
                st = json.loads(json.dumps(STATE))   # deep copy under lock
            st["log"] = list(LOGTAIL)[-40:]
            if st["started_at"]:
                # locate.py only prints capture progress once a minute; the
                # bar between prints runs on elapsed wall time, which for a
                # real-time capture is the same thing.
                st["elapsed"] = round(time.time() - st["started_at"], 1)
            if st["fix"] is None and st["phase"] == "idle":
                st["fix"] = _latest_fix_on_disk()
            return self._send(200, st)
        self._send(404, {"msg": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:                                        # noqa: BLE001
            body = {}
        if self.path == "/start":
            ok, msg = start_run(body.get("secs"), bool(body.get("replay")))
        elif self.path == "/abort":
            ok, msg = abort_run()
        else:
            return self._send(404, {"msg": "not found"})
        self._send(200 if ok else 409, {"msg": msg})

    def log_message(self, *a):                       # quiet: no request spam
        pass


def main():
    ap = argparse.ArgumentParser()
    # not 8088/8086: both are taken on radiopi2 (memory-server, influx)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="0.0.0.0 shows the fix to the whole LAN; use "
                         "127.0.0.1 to keep it on this box")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f"[dash] serving on http://{a.bind}:{a.port}  (ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        abort_run()


if __name__ == "__main__":
    main()
