#!/usr/bin/env python3
"""drive_map.py -- draw a war drive's fixes on a map so the eye can judge them.

Reads the stamped fix results locate.py keeps in lab_local/ (one per solved
stop, each naming the capture that made it) plus drive.py's drive_log.jsonl,
and writes ONE self-contained HTML file back into lab_local/: numbered stops
in capture order, a circle per stop whose radius is that stop's epoch scatter
(the honest accuracy figure -- with 4 birds the residual rms is meaningless),
the route line between them, OpenStreetMap and satellite-imagery layers so a
fix can be checked against the car park it was actually taken in, and a table.

Coordinates go into the HTML only, and the HTML lives in lab_local/, which is
gitignored: nothing this prints or commits contains a position. The map tiles
and the Leaflet library load from the web when the page is opened, like any
map page; the fixes themselves stay in the file on your disk.

    python3 drive_map.py                      # every stamped fix
    python3 drive_map.py --drive 20260825     # captures named sky_capture_20260825_*
    python3 drive_map.py --out my_drive.html  # inside lab_local/ unless absolute
"""
import argparse
import glob
import html
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "lab_local"
STAMP_RE = re.compile(r"sky_capture_(\d{8})_(\d{6})Z")


def capture_stamp(path):
    """'sky_capture_20260825_002140Z.cs16' -> ('20260825', '002140') or None."""
    m = STAMP_RE.search(os.path.basename(str(path)))
    return (m.group(1), m.group(2)) if m else None


def load_fixes(drive):
    """Latest valid stamped fix per capture, in capture order."""
    # A drive that crosses 00:00 UTC has captures under two dates, so
    # --drive takes a comma list: 20260829,20260830.
    drives = set(drive.split(",")) if drive else None
    by_capture = {}
    for f in sorted(glob.glob(str(LOCAL / "fix_result_*Z.json"))):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not d.get("valid") or "lat" not in d or "lon" not in d:
            continue
        st = capture_stamp(d.get("capture", ""))
        if st is None or (drives and st[0] not in drives):
            continue
        d["_stamp"] = st
        d["_solved"] = os.path.basename(f)
        by_capture[st] = d                      # sorted() => last solve wins
    return [by_capture[k] for k in sorted(by_capture)]


def load_drive_log():
    """drive.py's per-cycle log, keyed by capture stamp (may be absent)."""
    out = {}
    p = LOCAL / "drive_log.jsonl"
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        st = capture_stamp(d.get("file", ""))
        if st:
            out[st] = d
    return out


MIN_EPOCHS = 3          # matches fix.MIN_EPOCHS
MOVING_MPS = 3.5        # ~8 mph: above this the epochs are a track, not a stop
TELEPORT_MPS = 60.0     # 134 mph between consecutive stops = not a car


def accuracy_m(d):
    """The number that actually describes this stop's accuracy.

    Drive #3 (8/29) was recorded at 25-50 mph with 90 s captures, so the
    epoch scatter was measuring the road: 2.0 km of it per capture at 50 mph.
    Grading on that painted 19 of 23 stops red for driving. When fix.py has
    fitted a track, grade on the spread ABOUT the track instead, which is
    what is left when the vehicle is removed."""
    spd = d.get("speed_mps")
    trk = d.get("scatter_about_track_m")
    if trk is not None and spd is not None and spd > MOVING_MPS:
        return float(trk), True
    sc = d.get("scatter_m")
    if sc is None:
        sc = float(trk) if trk is not None else float("nan")
    return float(sc), False


def quality(d):
    """(colour, grade) for one stamped fix result."""
    if int(d.get("epochs_used", 0)) < MIN_EPOCHS:
        # One epoch makes the scatter zero by construction. Grey, not green.
        return "#8a8a8a", "ungraded"
    acc, _moving = accuracy_m(d)
    n_birds = len(d.get("birds", []))
    if acc != acc:                                   # NaN: nothing to grade on
        return "#8a8a8a", "ungraded"
    if acc < 40 and n_birds >= 5:
        return "#2e9e44", "good"
    if acc < 120:
        return "#d9a400", "fair"
    return "#d13b2e", "poor"


def stamp_epoch(when):
    """'2026-08-29 23:02:21Z' -> unix seconds."""
    import calendar
    import time as _t
    return calendar.timegm(_t.strptime(when, "%Y-%m-%d %H:%M:%SZ"))


def haversine_m(la1, lo1, la2, lo2):
    import math
    R = 6371000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    return R * math.hypot(math.radians(lo2 - lo1) * math.cos((p1 + p2) / 2),
                          p2 - p1)


def build_stops(fixes, dlog):
    stops = []
    for i, d in enumerate(fixes, 1):
        day, hms = d["_stamp"]
        when = f"{day[:4]}-{day[4:6]}-{day[6:]} {hms[:2]}:{hms[2:4]}:{hms[4:]}Z"
        birds = d.get("birds", [])
        sc = d.get("scatter_m")
        scatter = float(sc) if sc is not None else float("nan")
        acc, moving = accuracy_m(d)
        rms = float(d.get("resid_rms_m", 0.0))
        dof = max(0, len(birds) - 4)
        color, grade = quality(d)
        log = dlog.get(d["_stamp"], {})
        stops.append({
            "n": i, "when": when, "lat": d["lat"], "lon": d["lon"],
            "alt": float(d.get("alt_m", float("nan"))),
            "birds": birds, "el": d.get("el_deg", []),
            "epochs": int(d.get("epochs_used", 0)),
            "scatter": scatter, "acc": acc, "moving": bool(moving),
            "speed": d.get("speed_mps"),
            "track": [[t["lat"], t["lon"]] for t in d.get("track") or []],
            "rms": rms, "dof": dof,
            "color": color, "grade": grade, "flag": None,
            "cycle": log.get("cycle"), "strong_at_capture": log.get("strong"),
            "capture": os.path.basename(str(d.get("capture", ""))),
            "solved": d["_solved"],
        })
    # Cross-stop sanity: the capture stamps say how long it was between two
    # fixes, and the fixes say how far apart they are. A pair implying 300 mph
    # is one bad fix, whatever its altitude looked like -- this catches the
    # blow-ups without a magic number for how deep the mantle is.
    prev = None
    for b in stops:
        if b["grade"] == "ungraded":
            continue                       # already disbelieved; not an anchor
        if prev is not None:
            dt = stamp_epoch(b["when"]) - stamp_epoch(prev["when"])
            v = (haversine_m(prev["lat"], prev["lon"], b["lat"], b["lon"]) / dt
                 if dt > 0 else 0.0)
            if v > TELEPORT_MPS:
                b["flag"] = (f"implies {v:,.0f} m/s ({v*2.23694:,.0f} mph) "
                             f"from stop {prev['n']} - suspect")
                b["color"], b["grade"] = "#8a8a8a", "suspect"
                continue                   # a suspect stop is not an anchor
        prev = b
    return stops


def build_html(stops, title):
    rows = []
    for s in stops:
        rms_txt = "&mdash;" if s["dof"] == 0 else f"{s['rms']:.1f} m"
        birds_txt = ", ".join(str(b) for b in s["birds"])
        spd = s["speed"]
        spd_txt = ("&mdash;" if spd is None else
                   f"{spd*2.23694:.0f} mph" if spd > MOVING_MPS else "stopped")
        acc_txt = ("&mdash;" if s["acc"] != s["acc"] else
                   f"{s['acc']:.0f} m" + (" <small>(track)</small>"
                                          if s["moving"] else ""))
        scat_txt = ("&mdash;" if s["scatter"] != s["scatter"]
                    else f"{s['scatter']:.0f} m")
        note = f" <small>{html.escape(s['flag'])}</small>" if s["flag"] else ""
        rows.append(
            f"<tr style='border-left:6px solid {s['color']}'><td>{s['n']}</td>"
            f"<td>{s['when']}</td><td>{len(s['birds'])} <small>({birds_txt})</small></td>"
            f"<td>{s['epochs']}</td><td>{spd_txt}</td><td>{acc_txt}</td>"
            f"<td>{scat_txt}</td>"
            f"<td>{rms_txt}</td>"
            f"<td>{s['alt']:.0f} m</td><td>{s['grade']}{note}</td></tr>")
    rows = "\n".join(rows)
    data = json.dumps(stops)
    t = html.escape(title)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{t}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{{margin:0;height:100%;font:14px/1.4 system-ui,sans-serif;background:#111;color:#ddd}}
 #map{{height:68vh}}
 #panel{{padding:10px 14px;overflow:auto;height:calc(32vh - 20px)}}
 table{{border-collapse:collapse;width:100%}} th,td{{padding:3px 8px;text-align:left;border-bottom:1px solid #333}}
 th{{color:#9ab}} small{{color:#889}}
 .stop{{background:#fff;border:2px solid #333;border-radius:50%;width:22px;height:22px;
        line-height:20px;text-align:center;font-weight:700;color:#000;font-size:12px}}
 .leaflet-popup-content{{font:13px/1.35 system-ui,sans-serif}}
 .legend{{position:absolute;right:10px;top:10px;z-index:1000;background:rgba(17,17,17,.85);
          padding:8px 10px;border-radius:6px;font-size:12px}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}
</style></head><body>
<div id="map"></div>
<div class="legend"><b>{t}</b><br>
 <span class="sw" style="background:#2e9e44"></span>good: &lt; 40 m, &ge; 5 birds<br>
 <span class="sw" style="background:#d9a400"></span>fair: &lt; 120 m<br>
 <span class="sw" style="background:#d13b2e"></span>poor<br>
 <span class="sw" style="background:#8a8a8a"></span>ungraded / suspect<br>
 circle = accuracy; while moving that is the spread about the
 fitted track, and the orange line is the epoch track itself</div>
<div id="panel">
<table><thead><tr><th>#</th><th>capture (UTC)</th><th>birds</th><th>epochs</th>
<th>speed</th><th>accuracy</th><th>epoch spread</th><th>rms</th><th>alt</th>
<th>grade</th></tr></thead>
<tbody>{rows}</tbody></table>
<p><small>rms is shown only when the solve had degrees of freedom (more than 4 birds);
with exactly 4 the residual is zero by construction and only the epoch spread says anything.
<b>Accuracy vs epoch spread:</b> a 90 s capture taken at 50 mph covers 2 km of road, so the
raw epoch spread measures the car. Where fix.py fitted a track, accuracy is the spread about
that track and the stop is graded on it. A stop solved on fewer than three epochs is
<i>ungraded</i>: with one epoch the spread is zero by construction, not by accuracy. Stops are
numbered in capture order; ungraded and suspect stops are drawn but left out of the route line
and the map bounds. Open the satellite layer (top-right) to check a fix against the ground you
were actually parked on.</small></p>
</div>
<script>
const stops = {data};
const map = L.map('map');
const osm = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{maxZoom: 19, attribution: 'Imagery &copy; Esri'}});
osm.addTo(map);
L.control.layers({{'Map': osm, 'Satellite': sat}}).addTo(map);
const pts = [];
for (const s of stops) {{
  const ll = [s.lat, s.lon];
  const trusted = s.grade !== 'suspect' && s.grade !== 'ungraded';
  if (trusted) pts.push(ll);
  // The epoch track: where the receiver actually went DURING the capture.
  // Drawn instead of a big meaningless circle whenever the car was moving.
  if (s.moving && s.track.length > 1)
    L.polyline(s.track, {{color: '#e08a2e', weight: 3, opacity: 0.85}}).addTo(map);
  if (isFinite(s.acc))
    L.circle(ll, {{radius: s.acc, color: s.color, weight: 1.5, fillOpacity: 0.12}}).addTo(map);
  const rms = s.dof === 0 ? '&mdash; (0 DOF)' : s.rms.toFixed(1) + ' m';
  const spd = s.speed === null ? '&mdash;'
            : (s.speed > {MOVING_MPS} ? (s.speed * 2.23694).toFixed(0) + ' mph' : 'stopped');
  const acc = isFinite(s.acc) ? s.acc.toFixed(0) + ' m' + (s.moving ? ' about the track' : '')
                              : '&mdash;';
  L.marker(ll, {{icon: L.divIcon({{className: '', html: `<div class="stop" style="border-color:${{s.color}}">${{s.n}}</div>`,
                                   iconSize: [22, 22], iconAnchor: [11, 11]}})}})
   .bindPopup(`<b>Stop ${{s.n}}</b> &middot; ${{s.when}} &middot; <b>${{s.grade}}</b><br>` +
              `${{s.lat.toFixed(6)}}, ${{s.lon.toFixed(6)}} &middot; alt ${{s.alt.toFixed(0)}} m<br>` +
              `birds ${{s.birds.join(', ')}} (el ${{s.el.map(e => e.toFixed(0)).join('/')}}&deg;)<br>` +
              `epochs ${{s.epochs}} &middot; speed ${{spd}} &middot; accuracy ${{acc}}<br>` +
              `epoch spread ${{isFinite(s.scatter) ? s.scatter.toFixed(0) + ' m' : '&mdash;'}} ` +
              `&middot; rms ${{rms}}<br>` +
              (s.flag ? `<b style="color:#e08a2e">${{s.flag}}</b><br>` : '') +
              `<small>${{s.capture}}</small>`).addTo(map);
}}
if (pts.length > 1) L.polyline(pts, {{color: '#5aa9ff', weight: 2, dashArray: '6 6'}}).addTo(map);
if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.25)); else map.setView([0, 0], 2);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drive", help="YYYYMMDD of the captures to draw, comma "
                    "list for a drive that crosses midnight UTC (default: all)")
    ap.add_argument("--out", help="output HTML (relative paths land in lab_local/)")
    a = ap.parse_args()
    fixes = load_fixes(a.drive)
    if not fixes:
        sys.exit("no stamped fix results in lab_local/ match -- run locate.py first")
    tag = (a.drive or "all").replace(",", "-")
    out = Path(a.out) if a.out else Path(f"drive_map_{tag}.html")
    if not out.is_absolute():
        out = LOCAL / out
    title = f"numpy-gps war drive {a.drive}" if a.drive else "numpy-gps fixes"
    stops = build_stops(fixes, load_drive_log())
    out.write_text(build_html(stops, title), encoding="utf-8")
    grades = [st["grade"] for st in stops]
    moving = sum(1 for f in fixes
                 if (f.get("speed_mps") or 0.0) > MOVING_MPS)
    print(f"{len(stops)} stops -> {out}  "
          f"(good {grades.count('good')}, fair {grades.count('fair')}, "
          f"poor {grades.count('poor')}, ungraded {grades.count('ungraded')}, "
          f"suspect {grades.count('suspect')})")
    if moving:
        print(f"{moving} of them were taken while MOVING - those are graded on "
              f"the spread about the fitted epoch track, not on the raw epoch "
              f"scatter, which at speed is mostly the road")
    stale = sum(1 for f in fixes if f.get("track") is None)
    if stale:
        print(f"{stale} stop(s) predate the epoch track (fix.py before 8/30); "
              f"re-run locate.py on those captures to grade them on motion")
    print("open it in a browser; coordinates are in that file only")


if __name__ == "__main__":
    main()
