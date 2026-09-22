#!/usr/bin/env python3
"""fix.py - a GPS position fix from raw L1 IQ (the full receiver).

Reuses relativity.py's proven nav decode (acquire -> track -> 1 ms prompts ->
carrier cleanup -> parity/vote -> ephemeris) and adds the two stages that turn
a satellite's broadcast into YOUR location:
  * sat_ecef(): IS-GPS-200 ephemeris -> satellite ECEF position (Kepler solve +
    all harmonic corrections + Earth-rotation of the orbit plane)
  * solve(): weighted least-squares for (x, y, z, clock) from >= 4 pseudoranges

PRIVACY: a computed fix is written ONLY to lab_local/ (gitignored) and never
printed to any shared log. The code is public; the coordinates never are.

  python fix.py --iq ../captures/gps_fix_20260725.cs16   # needs >=4 birds
  python fix.py --validate                               # sat-position sanity
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import acquire, load_seg, track_sv
from relativity import (bit_sums, clean_carrier, costas, find_grid,
                        harvest_words, parity_ok, parse_harvest, prompt_stream,
                        ubits)

MU = 3.986005e14
OMEGA_E = 7.2921151467e-5
C = 299792458.0
F_REL = -4.442807633e-10        # IS-GPS-200 relativistic clock constant
MIN_EPOCHS = 3                  # below this the epoch scatter is not a measurement
SPEED_MAX = 60.0                # m/s (134 mph): faster is a solve, not a car
MOVING_MPS = 3.5                # ~8 mph: above this, call the epochs a track


EPH_NEED = {"sqrtA", "e", "M0", "toe", "omega", "i0", "Omega0", "af0"}
EPH_VALID_S = 7200.0            # +-2 h of toe: the broadcast's own fit interval


def capture_time(path):
    """UTC of a capture, as unix seconds. drive.py names its files
    sky_capture_YYYYMMDD_HHMMSSZ.cs16, which is the only timestamp that
    survives copying the card; the file's mtime is the fallback."""
    import calendar
    import re as _re
    import time as _t
    m = _re.search(r"_(\d{8})_(\d{6})Z", os.path.basename(str(path)))
    if m:
        return float(calendar.timegm(
            _t.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")))
    try:
        return float(Path(path).stat().st_mtime)
    except OSError:
        return float(_t.time())


def _pool_file():
    return HERE / "lab_local" / "eph_pool.json"


def load_eph_pool():
    """Orbits harvested from earlier captures. A satellite's ephemeris is
    the same broadcast for every receiver for ~2 hours, so a capture too
    short to decode subframes 1-3 cleanly does not need to: it needs to
    ACQUIRE the bird and catch one parity-clean subframe for its clock.
    That is what makes a short capture affordable -- 3 of 28 captures on
    the 8/29 drive were thrown away for want of a 4th orbit that the
    capture two minutes earlier had already decoded."""
    import json as _json
    try:
        return _json.loads(_pool_file().read_text())
    except (OSError, ValueError):
        return {}


def pool_put(pool, eph, tow_now, cap_t):
    pool[str(int(eph["prn"]))] = {
        "eph": {k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in eph.items() if k != "tows"},
        "toe": float(eph["toe"]), "tow_seen": float(tow_now),
        "capture_utc": float(cap_t)}


def pool_get(pool, prn, tow_now, cap_t):
    """A pooled orbit, or (None, why-not). Two independent clocks must both
    agree it is fresh: seconds-of-week wraps every 7 days, so a week-old
    entry can alias to 'ten minutes ago' on toe alone."""
    e = pool.get(str(int(prn)))
    if not e:
        return None, "no pooled orbit"
    age = abs(float(cap_t) - float(e.get("capture_utc", 0.0)))
    if age > EPH_VALID_S:
        return None, f"pooled orbit is {age/3600:.1f} h old"
    dt = (float(tow_now) - float(e["toe"]) + 302400.0) % 604800.0 - 302400.0
    if abs(dt) > EPH_VALID_S:
        return None, f"pooled toe is {dt/60:+.0f} min away"
    return e["eph"], dt


def save_eph_pool(pool):
    import json as _json
    (HERE / "lab_local").mkdir(exist_ok=True)
    _pool_file().write_text(_json.dumps(pool, indent=1))


def write_fix_result(payload):
    """Write the fix to lab_local/fix_result.json (the latest, where every
    reader looks) AND to a UTC-stamped copy beside it. The single fixed file
    meant a multi-stop drive kept only the final stop's fix; the stamped
    copies keep every stop, still inside gitignored lab_local."""
    import json as _json
    import time as _time
    local = HERE / "lab_local"
    local.mkdir(exist_ok=True)
    text = _json.dumps(payload, indent=1)
    (local / "fix_result.json").write_text(text)
    stamp = _time.strftime("%Y%m%d_%H%M%SZ", _time.gmtime())
    kept = local / f"fix_result_{stamp}.json"
    kept.write_text(text)
    return kept


def decode_eph(path, fs, prn, dopp, dur, want_timing=False):
    """Full nav decode for one PRN -> ephemeris dict (mirrors relativity.main's
    union+vote harvest, standalone so it runs per satellite).
    want_timing=True also returns {tent, anchors:[(bit_idx, sf, tow)]} - the
    millisecond-accurate subframe clock the pseudorange assembly needs."""
    # Track only over the demod window (prompt_stream caps at 118 s): a
    # single fixed fd fitted across a LONG track is hundreds of Hz wrong at
    # the window and collapses parity to ~1% - the "long-file decode" bug.
    # Anchor-fit extrapolation to later epochs is unaffected (clock-rate
    # residuals ~1e-12 x hundreds of s = ns).
    tr = track_sv(path, fs, prn, dopp, min(dur, 120.0))
    from measure import bit_tent
    tent = int(np.argmax(bit_tent(path, fs, tr, 5.0, 10.0)))
    p = prompt_stream(path, fs, tr, 1.0, min(dur - 2.0, 118.0))
    s, bits, _ = bit_sums(costas(clean_carrier(p)), tent)
    pol, starts = find_grid(bits)
    harvest = harvest_words(p, tent, starts)
    sf_anchors = []
    b = bits ^ pol
    for i in starts:                                  # global-stitch union
        d29s, d30s = (b[i - 2], b[i - 1]) if i >= 2 else (0, 0)
        words = {}
        for w in range(10):
            word = b[i + w * 30:i + (w + 1) * 30]
            if len(word) < 30:
                break
            ok, d = parity_ok(word, d29s, d30s)
            if ok:
                words[w] = d
            d29s, d30s = word[28], word[29]
        if 1 in words:
            sf = ubits(words[1], 20, 22)
            if 1 <= sf <= 5:
                tow = ubits(words[1], 1, 17)
                harvest.append((sf, tow, words))
                sf_anchors.append((int(i), int(sf), int(tow)))
    # how many INDEPENDENT parity-clean subframe-2 copies carry each toe --
    # counted before the vote below adds its own (possibly bit-flipped) copy,
    # so the consumer can tell a broadcast off-grid toe from a voted-in one.
    toe_clean = {}
    for sf, _tow, W in harvest:
        if sf == 2 and 9 in W:
            t = ubits(W[9], 1, 16) * 16
            toe_clean[t] = toe_clean.get(t, 0) + 1
    # majority vote across repeats
    anchors = {}
    for k, i in enumerate(starts):
        w1 = b[i:i + 30]
        ok2, d2 = parity_ok(b[i + 30:i + 60], w1[28], w1[29])
        if ok2:
            sf = ubits(d2, 20, 22)
            if 1 <= sf <= 5:
                anchors[k] = sf
    if anchors:
        k0, s0 = next(iter(anchors.items()))
        groups = {}
        for k, i in enumerate(starts):
            sf = (s0 - 1 + (k - k0)) % 5 + 1
            if i + 300 <= len(b):
                groups.setdefault(sf, []).append(i)
        for sf, idxs in sorted(groups.items()):
            if sf > 3 or len(idxs) < 2:
                continue
            voted = (np.stack([b[i:i + 300] for i in idxs]).mean(0) > 0.5).astype(np.int8)
            hyb = voted.copy()
            hyb[:60] = b[idxs[0]:idxs[0] + 60]
            d29s, d30s = (b[idxs[0] - 2], b[idxs[0] - 1]) if idxs[0] >= 2 else (0, 0)
            words = {}
            for w in range(10):
                word = hyb[w * 30:(w + 1) * 30]
                ok, d = parity_ok(word, d29s, d30s)
                if ok:
                    words[w] = d
                d29s, d30s = word[28], word[29]
            harvest.append((sf, ubits(words[1], 1, 17) if 1 in words else 0, words))
    eph = parse_harvest(harvest)
    eph["prn"] = prn
    if want_timing:
        return eph, {"tent": tent, "anchors": sf_anchors, "tr": tr,
                     "toe_clean": toe_clean.get(eph.get("toe"), 0)}
    return eph


def sat_ecef(eph, t):
    """IS-GPS-200 Table 20-IV: ephemeris + GPS time t -> satellite ECEF (m).
    Works for Galileo too (same Kepler machinery, GTRF~WGS84 at our accuracy):
    put mu=3.986004418e14 (Galileo OS SIS ICD 5.1.1) in the eph dict."""
    A = eph["sqrtA"] ** 2
    n0 = np.sqrt(eph.get("mu", MU) / A ** 3)
    tk = t - eph["toe"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    n = n0 + eph.get("dn", 0.0)
    M = eph["M0"] + n * tk
    E = M
    for _ in range(15):
        E = M + eph["e"] * np.sin(E)
    e = eph["e"]
    nu = np.arctan2(np.sqrt(1 - e * e) * np.sin(E), np.cos(E) - e)
    # orientation fields (omega/i0/Omega0) come from subframe 3 and set only the
    # DIRECTION; the orbit RADIUS r depends only on A,e,E - so |r| validates the
    # Kepler math even from a partial ephemeris. Default the orientation to 0.
    phi = nu + eph.get("omega", 0.0)
    s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
    u = phi + eph.get("Cus", 0) * s2 + eph.get("Cuc", 0) * c2
    r = A * (1 - e * np.cos(E)) + eph.get("Crs", 0) * s2 + eph.get("Crc", 0) * c2
    i = eph.get("i0", 0.964) + eph.get("IDOT", 0) * tk + eph.get("Cis", 0) * s2 + eph.get("Cic", 0) * c2
    xo, yo = r * np.cos(u), r * np.sin(u)
    Om = eph.get("Omega0", 0.0) + (eph.get("OmegaDot", 0) - OMEGA_E) * tk - OMEGA_E * eph["toe"]
    x = xo * np.cos(Om) - yo * np.cos(i) * np.sin(Om)
    y = xo * np.sin(Om) + yo * np.cos(i) * np.cos(Om)
    z = yo * np.sin(i)
    return np.array([x, y, z])


def ecc_anomaly(eph, t):
    """Eccentric anomaly at GPS time-of-week t (same Kepler solve as sat_ecef)."""
    A = eph["sqrtA"] ** 2
    tk = t - eph["toe"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    M = eph["M0"] + (np.sqrt(eph.get("mu", MU) / A ** 3) + eph.get("dn", 0.0)) * tk
    E = M
    for _ in range(15):
        E = M + eph["e"] * np.sin(E)
    return E


def clock_corr(eph, t_sv):
    """Satellite clock offset dt_sv = af0 + af1*dt + af2*dt^2 + relativistic
    eccentricity term (IS-GPS-200 20.3.3.3.3.1). t_gps = t_sv - dt_sv."""
    dt = t_sv - eph.get("toc", t_sv)
    if dt > 302400:
        dt -= 604800
    elif dt < -302400:
        dt += 604800
    dtr = F_REL * eph["e"] * eph["sqrtA"] * np.sin(ecc_anomaly(eph, t_sv))
    return (eph.get("af0", 0.0) + eph.get("af1", 0.0) * dt
            + eph.get("af2", 0.0) * dt * dt + dtr)


def ecef_to_llh(p):
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    e2 = f * (2 - f)
    x, y, z = p
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1 - e2))
    for _ in range(8):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - e2 * N / (N + h)))
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = r / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h


def az_el(rx, sp):
    """Azimuth/elevation (radians) of satellite ECEF sp from receiver ECEF rx."""
    lat, lon, _ = ecef_to_llh(rx)
    la, lo = np.radians(lat), np.radians(lon)
    d = sp - rx
    e = np.array([-np.sin(lo), np.cos(lo), 0.0]) @ d
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo),
                  np.cos(la)]) @ d
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                  np.sin(la)]) @ d
    return np.arctan2(e, n) % (2 * np.pi), np.arcsin(u / np.linalg.norm(d))


def klobuchar(a, b, lat_deg, lon_deg, az, el, t_gps):
    """IS-GPS-200 20.3.3.5.2.5 broadcast ionosphere model: L1 delay in
    SECONDS from the subframe-4 page-18 alpha/beta terms. az/el radians,
    t_gps in GPS seconds (time-of-week is fine; only time-of-day matters)."""
    E = el / np.pi                                     # semicircles
    psi = 0.0137 / (E + 0.11) - 0.022                  # earth central angle
    phi_i = np.clip(lat_deg / 180.0 + psi * np.cos(az), -0.416, 0.416)
    lam_i = lon_deg / 180.0 + psi * np.sin(az) / np.cos(phi_i * np.pi)
    phi_m = phi_i + 0.064 * np.cos((lam_i - 1.617) * np.pi)  # geomagnetic
    t = (4.32e4 * lam_i + t_gps) % 86400.0             # local time at IPP
    amp = max(sum(a[k] * phi_m ** k for k in range(4)), 0.0)
    per = max(sum(b[k] * phi_m ** k for k in range(4)), 72000.0)
    x = 2.0 * np.pi * (t - 50400.0) / per
    F = 1.0 + 16.0 * (0.53 - E) ** 3                   # obliquity
    if abs(x) < 1.57:
        return F * (5e-9 + amp * (1.0 - x * x / 2.0 + x ** 4 / 24.0))
    return F * 5e-9


def tropo_delay(el):
    """Simple troposphere mapping (meters): ~2.4 m zenith, grows toward
    the horizon. Good to ~1 m above 15 deg elevation."""
    return 2.47 / (np.sin(el) + 0.0121)


def solve(sats, weights=None):
    """sats = [(ecef_xyz, pseudorange_m)]. LS for (x,y,z,c*dt).

    Vectorised over satellites (same Gauss-Newton, same 12-iteration /
    1 mm stop): this is called ~60,000 times in one fix and the per-bird
    Python loop with np.append was 2.8 M array builds. `weights` (optional,
    one per satellite) turns it into weighted LS -- sqrt(w) scales the
    rows, so w = 1 everywhere reproduces the unweighted solve exactly."""
    SP = np.asarray([sp for sp, _ in sats], dtype=float)      # (n, 3)
    PR = np.asarray([pr for _, pr in sats], dtype=float)      # (n,)
    n = len(PR)
    x = np.zeros(4)
    A = np.empty((n, 4))
    A[:, 3] = 1.0
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, dtype=float))[:, None]
    for _ in range(12):
        d = x[:3] - SP                                        # (n, 3)
        rng = np.sqrt((d * d).sum(axis=1))
        A[:, :3] = d / rng[:, None]
        res = PR - (rng + x[3])
        if weights is not None:
            dx, *_ = np.linalg.lstsq(A * sw, res * sw[:, 0], rcond=None)
        else:
            dx, *_ = np.linalg.lstsq(A, res, rcond=None)
        x = x + dx
        if np.linalg.norm(dx[:3]) < 1e-3:
            break
    return x


def solve_prs(prs, weights=None):
    """prs = [(prn, eph, t_gps_tx)] at a common receive epoch. Iterates the
    unknown receive time, Sagnac-rotates each satellite by its travel time,
    LS-solves. Returns (rms_m, lat, lon, h, x)."""
    rms, lat, lon, h, x, _res = solve_prs_full(prs, weights)
    return rms, lat, lon, h, x


def solve_prs_full(prs, weights=None):
    """solve_prs, also returning the per-satellite residuals (m, in prs
    order) -- the ambiguity search reads them. `weights` (per satellite)
    makes the LS weighted; the reported rms stays UNWEIGHTED so runs with
    and without weights are comparable."""
    t_rx = max(t for _, _, t in prs) + 0.075
    x = np.array([0.0, 0.0, 0.0, 0.0])
    sats = []
    for _ in range(8):
        sats = []
        for prn, eph, t_tx in prs:
            sp = sat_ecef(eph, t_tx)
            tau = max(t_rx - t_tx, 0.0)
            th = OMEGA_E * tau                    # Sagnac: rotate into the
            rot = np.array([[np.cos(th), np.sin(th), 0],   # rx-epoch frame
                            [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
            sats.append((rot @ sp, C * (t_rx - t_tx)))
        x = solve(sats, weights)
        t_rx -= x[3] / C                          # absorb clock into epoch
    res = np.array([pr - (np.linalg.norm(x[:3] - sp) + x[3])
                    for sp, pr in sats])
    rms = float(np.sqrt(np.mean(np.square(res))))
    lat, lon, h = ecef_to_llh(x[:3])
    return rms, lat, lon, h, x, res


def solve_snapshot(entries):
    """entries = [{prn, eph, t_sv_coarse, phi_ms}] with t_sv_coarse the
    SV-CLOCK transmit time from the anchor fit (ms-quantized) and phi_ms the
    snapshot code phase (offset of the next code epoch AFTER the epoch, ms).

    Assembly laws (each one was a bug once):
      * work in SV time: code epochs align to SV-clock ms boundaries, so the
        integer-ms + code-phase combination happens BEFORE the clock
        correction (af0 alone is up to +-0.5 ms = +-150 km if applied first);
      * the code epoch starts phi AFTER the snapshot epoch, so the SV transmit
        time at the epoch is t_sv_tx = N - phi (NOT N + phi);
      * the anchor-fit integer is only ms-accurate -> exhaustive search of
        per-bird offsets in {-1,0,+1}, scored by residual rms + altitude
        sanity (with >= 5 birds only the true set collapses the residuals).
    Returns (rms, lat, lon, h, offsets)."""
    import itertools
    n = len(entries)
    frac = [(-e["phi_ms"]) % 1.0 for e in entries]
    N0 = [np.round(e["t_sv_coarse"] * 1e3 - f)
          for e, f in zip(entries, frac)]

    def assemble(offs):
        prs = []
        for k, e in enumerate(entries):
            t_sv_tx = (N0[k] + offs[k] + frac[k]) * 1e-3
            t_gps_tx = t_sv_tx - clock_corr(e["eph"], t_sv_tx)
            prs.append((e["prn"], e["eph"], t_gps_tx))
        return prs

    # a COMMON integer shift across all birds is absorbed by the receiver
    # clock (unobservable) - pin bird 0 and search only RELATIVE offsets.
    total = [0] * n
    # STAGE 1 -- residual-guided (8/15). One millisecond of integer slip on
    # one bird is c*1 ms = 299.8 km on THAT bird's pseudorange, and the LS
    # residuals point straight at it: solve, round each residual to whole
    # milliseconds of range, apply, repeat until nothing moves. Typically 1-3
    # solves where the exhaustive search below did 3^(n-1) per round (729 for
    # 7 birds, ~70 s of a 390 s fix). Same answer -- gated below.
    KM_PER_MS = C * 1e-3
    converged = False
    for _it in range(3 * n):
        rms, lat, lon, h, x, res = solve_prs_full(assemble(total))
        step = np.round(res / KM_PER_MS).astype(int)
        # NOT pinned here: a slip on bird 0 has to be correctable too (the
        # 8/15 gate: every unsolved pattern had the pin slipped). The common
        # shift is removed once, below, before returning.
        if not step.any():
            converged = True
            break
        # one bird per pass -- the LARGEST residual. With two or more slips
        # the LS smears each error over its neighbours; correcting them all
        # at once from the smeared residuals can oscillate, correcting the
        # worst one and re-solving does not (the sequential-RAIM shape).
        k = int(np.argmax(np.abs(res) * (step != 0)))
        total[k] += int(step[k])
    total = [t - total[0] for t in total]     # pin bird 0 (convention)
    if converged and rms < 1000.0 and -3000 < h < 9000:
        return rms, lat, lon, h, total
    # STAGE 1b -- coordinate descent, +-2 ms per bird, sweeps until stable.
    # Reaches the patterns stage 1 cannot (a slip on the pinned bird makes
    # every other bird's RELATIVE offset +-1 or +-2, and residual rounding
    # smears that) at ~5 solves per bird per sweep. The 8/15 gate found two
    # 2-slip patterns where the exhaustive +-1 search below returned 69 km
    # and 127 km "solutions"; this stage solves both to 32 m.
    def score(offs):
        r, la, lo, hh, xx = solve_prs(assemble(offs))
        return r + (0 if -3000 < hh < 9000 else 1e6), r, la, lo, hh
    cur = score(total)
    for _sweep in range(6):
        moved = False
        for k in range(1, n):                 # bird 0 stays pinned
            best_o, best_s = total[k], cur
            for o in (-2, -1, 1, 2):
                trial = list(total)
                trial[k] += o
                sc = score(trial)
                if sc[0] < best_s[0] - 1e-9:
                    best_o, best_s = trial[k], sc
            if best_o != total[k]:
                total[k] = best_o
                cur = best_s
                moved = True
        if not moved:
            break
    _, rms, lat, lon, h = cur
    if rms < 1000.0 and -3000 < h < 9000:
        return rms, lat, lon, h, total
    # STAGE 2 -- the exhaustive +-1 relative search, as before, for the
    # cases the residuals cannot untangle (few birds, poor geometry, a
    # gross outlier). Starts from wherever stage 1 left off.
    # Window: +-2 per bird when that stays under ~4000 solves (n <= 6), else
    # +-1 -- this stage only runs when the alternative is NO FIX, and the
    # 8/15 gate's last six 2-slip patterns needed a relative +-2 together
    # with an opposite-sign move that a +-1 window walks into a local
    # minimum on.
    wide = 5 ** (n - 1) <= 4000
    win = (-2, -1, 0, 1, 2) if wide else (-1, 0, 1)
    for _round in range(2 if wide else 4):
        best = None
        for rel in itertools.product(win, repeat=n - 1):
            offs = (0,) + rel
            trial = [t + o for t, o in zip(total, offs)]
            rms, lat, lon, h, x = solve_prs(assemble(trial))
            score = rms + (0 if -3000 < h < 9000 else 1e6)
            if best is None or score < best[0]:
                best = (score, rms, lat, lon, h, offs)
                if score < 100.0:             # a real fix: stop searching
                    break
        _, rms, lat, lon, h, offs = best
        total = [t + o for t, o in zip(total, offs)]
        if all(o == 0 for o in offs) or rms < 100.0:
            break
    return rms, lat, lon, h, total


import os as _os_w
WEIGHTING = _os_w.environ.get("GPSTUNA_WEIGHT", "none")


def elevation_weight(el_rad):
    """Per-satellite LS weight from elevation. Low satellites carry more
    troposphere, ionosphere and multipath error than the models remove;
    the usual single-frequency variance model is sigma^2 = a^2 + b^2 /
    sin^2(el). GPSTUNA_WEIGHT=none reproduces the unweighted solve."""
    if WEIGHTING == "none":
        return 1.0
    a, b = 0.5, 1.0
    s = max(np.sin(el_rad), 0.05)
    return 1.0 / (a * a + (b * b) / (s * s))


def motion_fit(P, T):
    """Fit a constant-velocity track through per-epoch ECEF positions.

    Returns (speed_mps, scatter_about_track_m), or (None, None) when there
    are too few epochs to say anything. The residual about the fitted line
    is the accuracy figure for a receiver that was moving; the spread about
    the MEAN is not, because most of it is the vehicle."""
    P = np.asarray(P, float)
    T = np.asarray(T, float)
    if len(T) < MIN_EPOCHS or float(T.std()) <= 1e-6:
        return None, None
    A = np.vstack([T - T.mean(), np.ones(len(T))]).T
    coef = np.linalg.lstsq(A, P, rcond=None)[0]          # rows: [velocity, p0]
    speed = float(np.linalg.norm(coef[0]))               # m/s in ECEF
    return speed, float(np.linalg.norm((P - A @ coef).std(axis=0)))


def solve_final(entries):
    """solve_snapshot + atmospheric refinement. Troposphere always; the
    Klobuchar ionosphere correction only when some bird's decode caught the
    subframe-4 page-18 broadcast (it's one constellation-wide model, so any
    bird's copy serves all). Corrections enter as transmit-time shifts:
    pr - delay == C*(t_rx - (t_tx + delay/C)). Returns a result dict; the
    uncorrected solve is kept alongside for honest A/B."""
    rms0, lat0, lon0, h0, total = solve_snapshot(entries)
    frac = [(-e["phi_ms"]) % 1.0 for e in entries]
    N0 = [np.round(e["t_sv_coarse"] * 1e3 - f)
          for e, f in zip(entries, frac)]

    def assemble(extra):
        prs = []
        for k, e in enumerate(entries):
            t_sv_tx = (N0[k] + total[k] + frac[k]) * 1e-3
            prs.append((e["prn"], e["eph"],
                        t_sv_tx - clock_corr(e["eph"], t_sv_tx) + extra[k]))
        return prs

    prs0 = assemble([0.0] * len(entries))
    _, _, _, _, x0 = solve_prs(prs0)
    iono = next((e["eph"] for e in entries
                 if "iono_a" in e["eph"] and "iono_b" in e["eph"]), None)
    extra, els, wts = [], [], []
    for prn, eph, t_tx in prs0:
        az, el = az_el(x0[:3], sat_ecef(eph, t_tx))
        els.append(round(float(np.degrees(el)), 1))
        d = tropo_delay(el) / C
        if iono is not None:
            d += klobuchar(iono["iono_a"], iono["iono_b"], lat0, lon0,
                           az, el, t_tx)
        extra.append(d)
        wts.append(elevation_weight(el))
    rms, lat, lon, h, x = solve_prs(assemble(extra), wts)
    return {"rms": rms, "lat": lat, "lon": lon, "h": h, "x": x,
            "offsets": total, "raw_rms": rms0, "raw_h": h0,
            "iono": iono is not None, "el_deg": els}


# A real broadcast ephemeris (PRN 9, GPS week 384 mod 1024, toe 7200 s), as
# every receiver on Earth received it -- public by construction, and it
# reveals nothing about where it was received. --validate exercises the
# ephemeris -> ECEF -> clock chain against it with no capture at all.
VALIDATE_EPH = {
    "prn": 9, "WN": 384, "IODE2": 21, "toe": 7200.0, "toc": 7200.0,
    "sqrtA": 5153.677988052368, "e": 0.0037707004230469465,
    "M0": 0.4683617065393228, "dn": 4.7612697545679945e-09,
    "omega": 2.0456646482237035, "i0": 0.9666972893594765,
    "Omega0": 2.1252002053131904, "OmegaDot": -8.454995041623894e-09,
    "IDOT": 6.750281176305986e-11,
    "Cuc": -1.0579824447631836e-06, "Cus": 3.6582350730895996e-06,
    "Crc": 309.625, "Crs": -18.96875,
    "Cic": -7.450580596923828e-09, "Cis": -5.587935447692871e-08,
    "af0": 0.0007003778591752052, "af1": -7.958078640513122e-12, "af2": 0.0,
}


def validate(path=None):
    """Prove the ephemeris -> satellite-position -> clock chain WITHOUT a
    capture (with one, --iq, it also decodes and checks every bird in it).
    Physical invariants, each a number the code cannot fake:
      * orbit radius at toe in the GPS shell (26,560 +- 100 km, e = 0.004);
      * speed from finite differences ~3.87 km/s (a circular orbit at that
        radius: sqrt(mu / r));
      * one sidereal-ish period later (43,082 s) the INERTIAL position
        repeats: rotate the ECEF result back by Earth's spin and compare;
      * the clock correction is af0 (0.7 ms here) plus a relativistic term
        of order F*e*sqrtA ~ tens of ns.
    Before 8/15 this read a capture from the author's disk, so on any other
    machine `--validate` -- and the locate.py fallback that calls it --
    printed "No IQ capture" and proved nothing."""
    ok_all = True

    def check(name, cond, detail):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}: {detail}")

    eph = dict(VALIDATE_EPH)
    t = eph["toe"]
    p0 = sat_ecef(eph, t)
    r = np.linalg.norm(p0)
    check("orbit radius", 26_460e3 < r < 26_660e3,
          f"|r| = {r/1e3:,.1f} km at toe (GPS shell 26,560 km)")
    dt = 1.0
    v_ecef = (sat_ecef(eph, t + dt) - sat_ecef(eph, t - dt)) / (2 * dt)
    # ECEF velocity carries Earth's spin; the Kepler speed is INERTIAL:
    # v_i = v_ecef + omega x r
    v_i = v_ecef + np.cross([0.0, 0.0, OMEGA_E], p0)
    v = np.linalg.norm(v_i)
    v_circ = np.sqrt(MU / r)
    check("orbital speed", abs(v - v_circ) < 60.0,
          f"{v:,.1f} m/s inertial (ECEF {np.linalg.norm(v_ecef):,.1f}), "
          f"circular {v_circ:,.1f} m/s")
    T = 43_082.0
    p1 = sat_ecef(eph, t + T)
    th = OMEGA_E * T                              # undo Earth's rotation
    rot = np.array([[np.cos(th), -np.sin(th), 0],
                    [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    d = np.linalg.norm(rot @ p1 - p0)
    check("period repeat", d < 60e3,
          f"inertial position after one period differs by {d/1e3:,.1f} km "
          f"(perturbations only)")
    dclk = clock_corr(eph, t)
    check("clock correction", abs(dclk - eph["af0"]) < 1e-7,
          f"{dclk*1e3:.4f} ms (af0 {eph['af0']*1e3:.4f} ms, relativistic "
          f"{(dclk-eph['af0'])*1e9:+.1f} ns)")
    lat, lon, h = ecef_to_llh(p0)
    check("llh round trip", abs(h - (r - 6_371e3)) < 30e3 and abs(lat) < 60,
          f"sub-satellite point lat {lat:+.1f} deg, height {h/1e3:,.0f} km")
    print(f"[fix] validate: {'ALL CHECKS PASSED' if ok_all else 'FAILED'} "
          f"(ephemeris -> ECEF -> clock chain, no capture needed)")
    if not path:
        return 0 if ok_all else 1
    # with a capture: decode every acquirable bird and check its shell radius
    fs = 2.048e6
    from measure import require_capture
    require_capture(path)
    dur = Path(path).stat().st_size / 4 / fs
    x = load_seg(path, fs, 0.5, 0.310)
    acq = acquire(x, fs, list(range(1, 33)), np.arange(-7000, 7001, 250.0), 300)
    det = {p: r for p, r in acq.items() if r["metric"] > 2.5}
    print(f"[fix] validate on {Path(path).name}: {len(det)} birds {sorted(det)}")
    for prn, r in det.items():
        eph = decode_eph(path, fs, prn, r["dopp"], dur)
        if not {"sqrtA", "e", "M0"}.issubset(eph):
            print(f"  PRN{prn}: partial ephemeris {sorted(set(eph)-{'tows','prn'})}"
                  f" - need subframe 2 (e, sqrtA, M0)")
            continue
        pos = sat_ecef(eph, eph["toe"])
        rr = np.linalg.norm(pos) / 1e3
        full = {"omega", "i0", "Omega0"}.issubset(eph)
        ok = 26000 < rr < 27200               # GPS orbit radius (dir needs sf3)
        ok_all &= ok
        print(f"  PRN{prn}: orbit radius |r|={rr:,.1f} km  "
              f"{'VALID GPS shell' if ok else 'OUT OF RANGE'}  "
              f"(ephemeris {'COMPLETE - full 3D position ready' if full else 'radius-only, sf3 partial'})")
    return 0 if ok_all else 1


def resolve_from_cache():
    """Re-run only the assembly+solve from lab_local/prs_cache.json (seconds,
    vs 15 min for a full redecode). Reconstructs the SV-clock coarse time
    from the cached (already clock-corrected) t_gps_tx_coarse, then applies
    the proven SV-time / N-minus-phi assembly in solve_snapshot()."""
    import json as _json
    cache = _json.loads((HERE / "lab_local" / "prs_cache.json").read_text())
    entries = []
    for c in cache:
        eph = c["eph"]
        dt0 = eph.get("af0", 0.0) + eph.get("af1", 0.0) * (
            c["t_gps_tx_coarse"] - eph.get("toc", c["t_gps_tx_coarse"]))
        entries.append({"prn": c["prn"], "eph": eph,
                        "t_sv_coarse": c["t_gps_tx_coarse"] + dt0,
                        "phi_ms": c["phi_ms"]})
    res = solve_final(entries)
    rms, lat, lon, h, offs = (res["rms"], res["lat"], res["lon"], res["h"],
                              res["offsets"])
    sane = -500 < h < 5000 and rms < 1000.0
    write_fix_result({
        "valid": sane,
        "lat": lat, "lon": lon, "alt_m": h,
        "birds": [e["prn"] for e in entries],
        "resid_rms_m": rms, "ms_offsets": offs,
        "epochs_used": 1, "scatter_m": None,
        "iono_corrected": res["iono"], "el_deg": res["el_deg"]})
    print(f"[resolve] residual rms {rms:,.1f} m "
          f"(uncorrected {res['raw_rms']:,.1f} m), altitude "
          f"{'PLAUSIBLE' if sane else 'IMPLAUSIBLE'} ({h:,.0f} m), "
          f"ms offsets {offs}, "
          f"{'tropo+iono' if res['iono'] else 'tropo only'}")
    print("[resolve] coordinates in lab_local/fix_result.json (private)")
    return 0 if sane else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", default=None,
                    help="raw L1 IQ capture (2.048 Msps int16 IQ); required "
                         "unless --validate or --resolve")
    ap.add_argument("--fs", type=float, default=2.048e6)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--resolve", action="store_true",
                    help="re-run assembly+solve from lab_local cache")
    ap.add_argument("--no-eph-pool", action="store_true",
                    help="do not reuse orbits decoded from nearby captures "
                         "(lab_local/eph_pool.json), and do not add to them")
    ap.add_argument("--multi", type=int,
                    default=int(os.environ.get("GPSTUNA_EPOCHS", "15")),
                    help="solve at N snapshot epochs (default 15, or "
                         "$GPSTUNA_EPOCHS). This used to default to 1, which "
                         "locate.py never did -- and a one-epoch solve is now "
                         f"refused outright, because its scatter is zero by "
                         "construction. Same invariant, same consequence, "
                         "both entry points.")
    a = ap.parse_args()
    if a.validate:
        return validate(a.iq)
    if a.resolve:
        return resolve_from_cache()
    fs = a.fs
    if not a.iq:
        ap.error("--iq CAPTURE is required (or use --validate / --resolve)")
    from measure import check_sidecar, require_capture
    require_capture(a.iq)
    check_sidecar(a.iq, fs)
    dur = Path(a.iq).stat().st_size / 4 / fs
    x = load_seg(a.iq, fs, 0.5, 0.310)
    acq = acquire(x, fs, list(range(1, 33)), np.arange(-7000, 7001, 250.0), 300)
    det = {p: r for p, r in acq.items() if r["metric"] > 2.5}
    print(f"[fix] {len(det)} birds acquired: {sorted(det)}")
    if len(det) < 4:
        print(f"[fix] NEED >= 4 satellites for a position fix, have {len(det)}.")
        print("[fix] indoor capture sees too few - re-capture with the antenna at")
        print("      a window or outdoors (a $10 GPS patch antenna gets 8-12).")
        print("[fix] running --validate instead (proves the sat-position math):")
        return validate()
    # (4+ birds path) decode, pseudoranges from code phase + TOW, solve
    print("[fix] 4+ birds - decoding ephemerides + solving (fix -> lab_local/ only)")
    return full_fix(a.iq, fs, det, dur, multi=a.multi,
                    use_pool=not a.no_eph_pool)


def _decode_one(job):
    """Pool worker: decode one PRN, capture its prints. Top-level so it
    pickles under the spawn start method (Windows, macOS)."""
    import io as _io
    import contextlib as _ctx
    path, fs, prn, dopp, dur = job
    buf = _io.StringIO()
    try:
        with _ctx.redirect_stdout(buf):
            eph, tim = decode_eph(path, fs, prn, dopp, dur, want_timing=True)
        return (prn, eph, tim, None, buf.getvalue())
    except Exception as e:                                   # noqa: BLE001
        return (prn, None, None, f"{type(e).__name__}: {e}", buf.getvalue())


def _decode_all(jobs):
    """Run _decode_one over all birds, in parallel when there is more than
    one bird and more than one core; serial otherwise (and serial as the
    fallback if the pool cannot start -- a locked-down interpreter, a
    frozen build -- so a pool problem can never cost the fix)."""
    import os as _os
    from measure import _pool_allowed, pool_silence, drain
    n_workers = min(len(jobs), max(1, (_os.cpu_count() or 1)))
    if n_workers > 1 and _pool_allowed():
        try:
            import multiprocessing as _mp
            # Children inherit the environment and import numpy fresh: pin
            # their BLAS to one thread each. The matmuls here are 100x2048;
            # N workers x a 64-thread BLAS on ops that size is a thread
            # storm, not parallelism (measured elsewhere on this fleet:
            # 480 % CPU for 0.09x). The parent's own numpy is unaffected.
            for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                       "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                _os.environ.setdefault(_v, "1")
            ctx = _mp.get_context("spawn")
            with ctx.Pool(n_workers) as pool:
                # serial cost ~ 1.5 s per second of capture per bird on a
                # slow core; the pool must beat that by a lot or it is dead
                # one bird's decode is ~1-2 min on an idle Pi, longer on a
                # busy one (5 min was tripped by a healthy pool, 8/24). A
                # dead worker is caught in half a second by drain(); the
                # silence limit only backstops a pool that is truly stuck.
                ars = [pool.apply_async(_decode_one, (j,)) for j in jobs]
                return drain(ars, pool_silence(900), pool)
        except Exception as e:                               # noqa: BLE001
            print(f"  (process pool unavailable: {type(e).__name__}: {e}; "
                  f"decoding serially)")
    return [_decode_one(j) for j in jobs]


def full_fix(path, fs, det, dur, multi=1, use_pool=True):
    """The final stage: coarse-time from each bird's nav-bit grid
    (millisecond-accurate subframe clocks) + sub-millisecond from a
    common-epoch acquisition snapshot -> pseudoranges -> solve().
    The computed position goes ONLY to lab_local/ (gitignored)."""
    import json as _json
    # 1. per-bird nav decode with timing anchors -- one process per bird.
    # Every satellite's track/decode is independent, and before 8/15 they
    # ran one after another: 285 of a 390 s fix. The pool sizes itself to
    # the birds and the box (a Pi with 4 cores gets 4-wide, this desktop 7).
    # Each worker returns its own log so the output stays ordered by PRN.
    birds = {}
    borrowed, pool_dirty = [], []
    cap_t = capture_time(path)
    pool = load_eph_pool() if use_pool else None
    jobs = [(path, fs, prn, r["dopp"], dur) for prn, r in det.items()]
    results = _decode_all(jobs)
    for (prn, eph, tim, err, log_txt) in results:
        if log_txt:
            print(log_txt.rstrip("\n"))
        if err:
            print(f"  PRN{prn}: decode failed ({err})")
            continue
        missing = EPH_NEED - set(eph)
        if not tim["anchors"]:
            print(f"  PRN{prn}: no parity-clean subframe anchor")
            continue
        tow_now = min(tw for _i, _sf, tw in tim["anchors"]) * 6.0 - 6.0
        if missing:
            # The orbit is the same broadcast for hours; the CLOCK is not.
            # A bird with a clean subframe anchor but an incomplete orbit
            # can borrow the orbit from a nearby capture and still bring
            # its own timing -- which is what a short capture cannot decode
            # and does not have to.
            pooled, why = (pool_get(pool, prn, tow_now, cap_t) if pool is not None
                           else (None, "pool disabled"))
            if pooled is not None:
                birds[prn] = (pooled, tim)
                borrowed.append(int(prn))
                print(f"  PRN{prn}: ephemeris incomplete (missing "
                      f"{sorted(missing)}) - BORROWED the orbit decoded "
                      f"{abs(why)/60:.0f} min from its toe earlier in this "
                      f"drive; timing is this capture's own")
                continue
            print(f"  PRN{prn}: ephemeris incomplete (missing "
                  f"{sorted(missing)}) - {why}")
            continue
        birds[prn] = (eph, tim)
        if pool is not None:
            pool_put(pool, eph, tow_now, cap_t)
            pool_dirty.append(prn)
        print(f"  PRN{prn}: ephemeris COMPLETE, {len(tim['anchors'])} anchors, "
              f"C/N0 {tim['tr']['cn0']:.0f} dB-Hz")
    # toe consensus gate: ephemeris reference times normally sit on the
    # 2-hour grid (toe % 7200 == 0). A lone bird off-grid while its
    # siblings agree is almost always a voted-in bit error (one LSB of
    # toe = 16 s = ~60 km of satellite position - it poisons the whole
    # solve). Post-upload cutover sets CAN be legitimately off-grid, so
    # this only fires when the bird is BOTH off-grid and alone in it --
    # AND no two independent parity-clean copies of its subframe 2 agree
    # on that toe. Two clean copies agreeing is a broadcast, not a flip
    # (8/15: a user's strongest bird, 190/190 parity, toe = grid - 16 s,
    # was being thrown away by this gate for no reason).
    on_grid = [p for p, (e, _t) in birds.items() if e["toe"] % 7200 == 0]
    if len(on_grid) >= 3:
        for prn in [p for p in list(birds) if p not in on_grid]:
            toe_bad = birds[prn][0]["toe"]
            n_clean = birds[prn][1].get("toe_clean", 0)
            if n_clean >= 2:
                print(f"  PRN{prn}: toe {toe_bad:.0f} off the 2-hour grid "
                      f"but {n_clean} independent clean copies agree - "
                      f"KEPT (post-upload cutover ephemeris)")
                continue
            print(f"  PRN{prn}: toe {toe_bad:.0f} off the 2-hour grid while "
                  f"{len(on_grid)} siblings are on it - DROPPED "
                  f"(suspected voted-in bit error, {n_clean} clean copies)")
            del birds[prn]
    if pool is not None and pool_dirty:
        # Bank the orbits BEFORE the bird-count check: a capture that cannot
        # solve on its own still decoded real ephemerides, and the next one
        # along the road should get them.
        save_eph_pool(pool)
        print(f"[fix] {len(pool_dirty)} orbit(s) banked in "
              f"lab_local/eph_pool.json for the next capture")
    if len(birds) < 4:
        print(f"[fix] only {len(birds)} birds fully decoded - need 4"
              + (f" (the orbit pool supplied {len(borrowed)} of them; "
                 f"the rest never acquired or brought no clean subframe)"
                 if borrowed else
                 ""
                 if pool is None else
                 " - no pooled orbit was fresh enough to lend"))
        return 1

    # 2. coarse SV clock, SELF-CALIBRATED, once per bird: every parity-clean
    # subframe anchor is a (file-time, satellite-time) pair; a linear fit
    # across them derives the exact mapping with no Doppler-sign assumptions,
    # and the fit residual is a built-in truth check (microseconds = right,
    # anything more = broken)
    fits = {}
    n1 = int(round(fs * 1e-3))
    for prn, (eph, tim) in birds.items():
        # Receiver time of each anchor's bit edge. The bit begins at a CODE
        # EPOCH, and that epoch sits INSIDE the tent prompt at the code
        # phase the track model gives (slope*t + icpt samples -- the same
        # model prompt_stream rolls the replica by), wrapped to the nearest
        # prompt boundary. Before 8/15 this used the prompt boundary alone,
        # so every bird's coarse SV time carried a constant 0..1 ms error
        # -- measured on a live capture: t_sv*1e3 - frac landed at .46 .52
        # .23 .18 .35 .54 .59 across seven birds, i.e. round() was a coin
        # flip for three of them and the integer-ms search had to rescue
        # most epochs (1,400 solves each). With the epoch position in the
        # fit the coarse time is microsecond-accurate and the search
        # converges on its first solve.
        # The epoch DRIFTS through the prompt (~6 samples/s at 5 kHz of
        # Doppler, a whole period per ~5 min), so wrap ONCE -- at the tent
        # time, where the tent chose the nearest boundary -- and carry the
        # position forward continuously with the slope; a per-anchor wrap
        # would put a 1 ms step into ft where the epoch crosses the prompt
        # middle and break the fit (measured: later epochs at 40-120 km).
        _tr = tim["tr"]
        t_tent = 1.0 + tim["tent"] * 1e-3
        c0 = (_tr["slope"] * t_tent + _tr["icpt"]) % n1
        c0 = ((c0 + n1 / 2.0) % n1) - n1 / 2.0          # signed, nearest
        ft = []
        for i, _sf, _tw in tim["anchors"]:
            t_prompt = 1.0 + (tim["tent"] + 20.0 * i) * 1e-3
            c = c0 + _tr["slope"] * (t_prompt - t_tent)
            ft.append(t_prompt + c / fs)
        ft = np.array(ft, float)
        st = np.array([tw * 6.0 - 6.0 for _i, _sf, tw in tim["anchors"]],
                      float)
        aa, bb = np.polyfit(ft, st, 1)
        fitres = float(np.std(st - (aa * ft + bb)))
        print(f"  PRN{prn}: anchor fit residual {fitres*1e6:.1f} us "
              f"(clock rate {aa - 1.0:+.2e})")
        fits[prn] = (aa, bb)
    if any("iono_a" in eph for eph, _ in birds.values()):
        print("  ionosphere: Klobuchar terms decoded (subframe 4 page 18)")
    else:
        # short captures rarely span the 12.5-min page cycle - fall back to
        # the newest archived broadcast (constellation-wide, slowly varying)
        _if = HERE / "lab_local" / "iono_terms.json"
        if _if.exists():
            _io = _json.loads(_if.read_text())
            for eph, _t in birds.values():
                eph["iono_a"] = _io["iono_a"]
                eph["iono_b"] = _io["iono_b"]
            print(f"  ionosphere: archived Klobuchar terms applied "
                  f"({_io.get('src', 'unknown src')})")

    # 3. snapshot epochs: sub-ms code phase per bird at each T_RX, then
    # SV-time assembly (t_sv_tx = N - phi) + integer search + atmos + solve.
    # NOTE assembly happens in SV time inside solve_snapshot(); the clock
    # correction (af0/af1/af2 + relativistic) applies AFTER the integer-ms +
    # code-phase combination - code epochs align to SV-clock ms boundaries,
    # and af0 alone can be +-0.5 ms (+-150 km) if subtracted first.
    n1 = int(round(fs * 1e-3))
    epochs = (np.linspace(10.0, dur - 2.0, multi) if multi > 1
              else np.array([min(dur - 1.0, 45.0)]))
    import json as _json2
    results, cache = [], []
    SNAP_MS = 300
    for T_RX_start in epochs:
        xsnap = load_seg(path, fs, T_RX_start, SNAP_MS * 1e-3 + 0.010)
        # The snapshot code phase is a NON-COHERENT SUM over SNAP_MS one-ms blocks, and the
        # code drifts through them at (Doppler / 1540) chips/s - up to a whole chip across
        # the window at 5 kHz. The summed peak therefore sits at the phase of the window's
        # CENTRE, not its start, and the transmit time must be evaluated there too.
        # Measured 9/22 against gr-gpsrx's tracking channels on the same capture: every
        # bird's range was off by -0.15 s x its range rate (-114 m at +4.5 kHz, +20 m at
        # -1 kHz; slope 0.93 to this prediction, 6.8 m residual) - a ~150 m position error,
        # mostly in altitude, that two captures two hours apart both carried.
        T_RX = T_RX_start + 0.5 * SNAP_MS * 1e-3
        entries = []
        for prn, (eph, tim) in birds.items():
            # Evaluate the Doppler AT this epoch. tr["fd"] is referenced to
            # tr["tref"], the centre of the tracking window, and these epochs
            # walk the whole capture -- on a 600 s file the far one is ~540 s
            # away, which at ~0.8 Hz/s is hundreds of Hz. This acquire is
            # handed a SINGLE candidate frequency and does no search, so that
            # error lands straight on the correlation.
            _tr = tim["tr"]
            fd_at = (_tr["fd"]
                     + _tr.get("fdot", 0.0) * (T_RX - _tr.get("tref", 0.0)))
            r = acquire(xsnap, fs, [prn], np.array([fd_at]), SNAP_MS)[prn]
            cp = r.get("code_phase_f", r["code_phase"])     # sub-sample
            phi_ms = (cp % n1) / fs * 1e3                   # 0..1 ms
            aa, bb = fits[prn]
            t_sv_at_rx = aa * T_RX + bb       # SV-CLOCK time (TOW is SV time)
            entries.append({"prn": prn, "eph": eph,
                            "t_sv_coarse": t_sv_at_rx, "phi_ms": phi_ms})
        if not cache:                          # cache the first epoch for --resolve
            for e in entries:
                dt_sv = clock_corr(e["eph"], e["t_sv_coarse"])
                cache.append({"prn": int(e["prn"]),
                              "phi_ms": float(e["phi_ms"]),
                              "t_gps_tx_coarse": float(e["t_sv_coarse"] - dt_sv),
                              "eph": {k: (float(v) if isinstance(v, (int, float))
                                          else v)
                                      for k, v in e["eph"].items()
                                      if k != "tows"}})
        try:
            res = solve_final(entries)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
            # One bad epoch (a NaN pseudorange from a snapshot acquire that
            # found nothing) used to take the whole solve down with "SVD did
            # not converge" (drive #3, 8/29 23:58Z capture). Skip it; the
            # other 14 epochs are the fix.
            print(f"  T={T_RX:5.1f}s: epoch skipped ({e})")
            continue
        res["t_rx"] = float(T_RX)
        results.append(res)
        print(f"  T={T_RX:5.1f}s: rms {res['rms']:,.1f} m, "
              f"alt {res['h']:,.0f} m"
              f"{' (tropo+iono)' if res['iono'] else ' (tropo only)'}")

    (HERE / "lab_local").mkdir(exist_ok=True)
    (HERE / "lab_local" / "prs_cache.json").write_text(_json2.dumps(cache))
    # 4. average the sane epochs in ECEF; scatter = honest repeatability
    if not results:
        print("[fix] every epoch failed to solve - no fix")
        return 1
    good = [r for r in results if -500 < r["h"] < 5000]
    if not good:
        # nothing sane to average: fall back to everything ONLY so the
        # numbers below describe what happened -- never as a position.
        good = results
    P = np.array([r["x"][:3] for r in good])
    T = np.array([r.get("t_rx", 0.0) for r in good], float)
    scatter = float(np.linalg.norm(P.std(axis=0))) if len(good) > 1 else None
    lat, lon, h = ecef_to_llh(P.mean(axis=0))
    rms = float(np.mean([r["rms"] for r in good]))
    # 4b. the epochs are not repeat measurements of one point when the
    # receiver is MOVING -- they are a track. Drive #3 (8/29) recorded 90 s
    # captures at 50 mph: the car covered 2.0 km DURING each one, and the
    # epoch scatter dutifully reported ~500 m, which was the road, not the
    # error (median |scatter - v*T/sqrt(12)| across the drive: 33 m). So
    # keep every epoch's own position, fit a straight line through them,
    # and report BOTH numbers: the spread about the mean (what scatter has
    # always meant) and the spread about the fitted track, which is the
    # accuracy figure once the vehicle is taken out of it.
    track = []
    for r in good:
        _la, _lo, _h = ecef_to_llh(np.asarray(r["x"][:3], float))
        track.append({"t_rx_s": round(float(r.get("t_rx", 0.0)), 2),
                      "lat": _la, "lon": _lo, "alt_m": _h,
                      "rms_m": float(r["rms"])})
    speed, scatter_track = motion_fit(P, T)
    # 5. is this a FIX or a number? A real single-frequency solve sits at
    # tens of metres rms with epochs that agree to ~100 m. A wrong-integer
    # solve (the ms-ambiguity search landing on a garbage minimum) shows up
    # as km of residual, km of scatter, or an altitude in the mantle -- and
    # used to be printed as SOLVED with coordinates, which put a user in
    # Hungary from a capture made in the Netherlands (8/15). Every gate
    # below is loose enough that a rooftop with a bad sky still passes;
    # only a non-solution fails them all at once.
    reasons = []
    if len(good) < MIN_EPOCHS:
        # Drive #3 stop 18 (8/29 23:56Z) solved on ONE epoch: scatter 0.0 m,
        # altitude +4,595 m, valid=true, and the map drew it as the best
        # stop of the whole moving leg. With one epoch the scatter is zero
        # BY CONSTRUCTION -- the honesty metric reads perfect exactly where
        # there is no repeatability left to measure. Refuse the fix instead.
        if len(epochs) >= MIN_EPOCHS:
            # the solve is sick: most of its epochs died on the way
            reasons.append(f"only {len(good)} of {len(epochs)} epochs "
                           f"survived - too few to tell an accurate fix from "
                           f"a lucky one (with one epoch the scatter is zero "
                           f"by construction, not by accuracy)")
        else:
            # the RUN was configured too small to be gradeable
            reasons.append(f"asked for {len(epochs)} epoch(s); a fix needs at "
                           f"least {MIN_EPOCHS} before its scatter means "
                           f"anything - re-run without --multi (default 15)")
    if not (-500 < h < 5000):
        reasons.append(f"altitude {h:,.0f} m is not on this planet's surface")
    if rms > 1000.0:
        reasons.append(f"residual rms {rms:,.0f} m (a real fix is tens of m)")
    if scatter is not None and scatter > 5000.0:
        reasons.append(f"epoch scatter {scatter:,.0f} m (epochs disagree by km)")
    if speed is not None and speed > SPEED_MAX:
        # A gate that needs no magic altitude: the epochs imply how fast the
        # receiver was moving, and a car does not do 300 mph.
        reasons.append(f"epochs imply a receiver speed of {speed:,.0f} m/s "
                       f"({speed*2.23694:,.0f} mph) - not a vehicle")
    valid = not reasons
    kept = write_fix_result({
        "valid": valid,
        "lat": lat, "lon": lon, "alt_m": h,
        "birds": sorted(int(p) for p in birds),
        "resid_rms_m": rms, "epochs_used": len(good),
        "epochs_attempted": int(len(epochs)),
        "scatter_m": scatter,
        "scatter_about_track_m": scatter_track,
        "speed_mps": speed,
        "t_rx_ref_s": round(float(T.mean()), 2) if len(good) else None,
        "eph_borrowed": sorted(borrowed),
        "track": track,
        "iono_corrected": good[0]["iono"],
        "el_deg": good[0]["el_deg"],
        "ms_offsets": good[0]["offsets"],
        "capture": str(path)})
    if valid:
        print(f"[fix] SOLVED with {len(birds)} birds x {len(good)} epoch(s): "
              f"mean rms {rms:,.1f} m, epoch scatter {scatter:,.1f} m, "
              f"altitude PLAUSIBLE ({h:,.0f} m)")
        if borrowed:
            print(f"[fix] {len(borrowed)} of {len(birds)} orbits came from "
                  f"the pool (PRN {', '.join(str(b) for b in sorted(borrowed))}"
                  f") - decoded from a capture less than "
                  f"{EPH_VALID_S/3600:.0f} h away and still inside their "
                  f"fit interval. Timing came from THIS capture.")
        if speed is not None and speed > MOVING_MPS:
            span = speed * (float(T.max() - T.min()))
            print(f"[fix] the receiver was MOVING at {speed:,.1f} m/s "
                  f"({speed*2.23694:,.0f} mph): the epochs span {span:,.0f} m "
                  f"of road, so {scatter:,.0f} m of 'scatter' is mostly the "
                  f"car. Spread about the fitted track: "
                  f"{scatter_track:,.0f} m -- THAT is the accuracy figure. "
                  f"The position above is the mid-point of the run; "
                  f"lab_local/fix_result.json['track'] has all "
                  f"{len(track)} epoch positions.")
        elif speed is not None:
            print(f"[fix] receiver effectively stationary "
                  f"({speed:,.1f} m/s implied); spread about the fitted "
                  f"track {scatter_track:,.0f} m")
        dof = len(birds) - 4
        if dof <= 1:
            # 8/15 live: 5 birds printed "rms 2.8 m" beside a 41 m scatter.
            # With one degree of freedom the residual is small by
            # construction, not by accuracy; say which number to believe.
            print(f"[fix] note: with {len(birds)} satellites the solve has "
                  f"{dof} degree{'s' if dof != 1 else ''} of freedom, so the "
                  f"residual rms is NOT an accuracy figure -- the epoch "
                  f"scatter is"
                  + (f" ({scatter_track:,.0f} m about the track, once the "
                     f"vehicle's motion is removed)"
                     if speed is not None and speed > MOVING_MPS
                     else f" ({scatter:,.0f} m)")
                  + ". More sky = more birds = an honest rms.")
        print(f"[fix] coordinates written to lab_local/fix_result.json "
              f"(gitignored - yours alone)")
        print(f"[fix] this stop kept as lab_local/{kept.name} -- later stops "
              f"will not overwrite it")
        return 0
    print(f"[fix] NO FIX from {len(birds)} birds x {len(good)} epoch(s) -- "
          f"the solve converged, but not on a position:")
    for r in reasons:
        print(f"       - {r}")
    print("[fix] lab_local/fix_result.json holds the numbers with valid=false; "
          "do NOT read a location from it.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
