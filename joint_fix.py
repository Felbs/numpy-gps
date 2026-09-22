#!/usr/bin/env python3
"""joint_fix.py - JOINT GPS + Galileo position fix from one wideband capture.

The payoff of the dual-constellation work: gal_inav.py proved we can read
CRC-clean Galileo I/NAV off the air; gal_eph.py assembles full ephemerides
from word types 1-4; this driver measures Galileo E1-B/C pseudoranges at
the same snapshot epochs as GPS and solves them TOGETHER.

Model (5 unknowns): x, y, z, c*dt_gps, c*dt_isb
  * every pseudorange sees the same receiver position + GPS-timeline clock
  * Galileo rows get one extra bias column: the GPS-Galileo time offset
    (GGTO) plus receiver inter-signal bias (BPSK vs BOC group delay)
  * alternative 4-unknown mode applies the DECODED GGTO (I/NAV word type
    10, ICD 5.1.8 Eq 21) instead of estimating it - agreement between the
    two is a built-in self-check.

Assembly laws inherited from fix.py (each was a bug once): work in SV time,
t_sv_tx = N*T_code - phi (Galileo T_code = 4 ms, not GPS's 1 ms!), clock
correction only AFTER the integer+phase combination, pinned-reference
integer search scored by residual rms + altitude sanity.

Galileo specifics (Galileo OS SIS ICD Issue 2.0, Jan 2021):
  * anchor law (5.1.2): broadcast TOW marks the leading edge of the first
    chip of the first symbol of its page -> (file_time, TOW) pairs at every
    time-stamped page, linear fit = the file-time -> SV-time mapping
  * clock: af0/af1/af2 + relativistic F*e*sqrtA*sin(E) (Eq 13); E1
    single-frequency users subtract BGD(E1,E5b) from af0 (5.1.5 Eq 15)
  * mu = 3.986004418e14 (5.1.1); GTRF ~ WGS84 at our accuracy

Stages (each cached in lab_local/ so long steps run once):
  --stage gal  --prn 29        Galileo track + pages + eph + anchors
  --stage gps  [--prn 16]      GPS windowed decode (pristine zone) + anchors
  --stage snap [--epochs 0-3]  code phases at the snapshot epochs
  --stage solve                joint assembly + integer search + solve

PRIVACY: computed coordinates go ONLY to lab_local/ (gitignored). This
tool prints residuals, altitudes, bird counts and time offsets - never
latitude/longitude.
"""
import argparse
import os
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fix
from fix import C, OMEGA_E, clock_corr, ecef_to_llh, sat_ecef
from measure import acquire as gps_acquire, require_capture
from measure import bit_tent, load_seg, prompts_ms
from relativity import (bit_sums, clean_carrier, costas, find_grid,
                        harvest_words, parity_ok, parse_harvest,
                        prompt_stream, ubits)
import gal_e1
from gal_e1 import CODE_LEN as GAL_CODE_LEN
from gal_e1 import T_COH, boc_replica, load_codes
from gal_inav import boc_chips, harvest_pages, parse_word, track_prn
import gal_eph
from gal_eph import (anchors_from_pages, assemble_eph, extract_ggto,
                     fold_bgd, ggto_eval)

# The ONE capture every stage reads. Set with --iq (or GPSTUNA_IQ); it used
# to be the author's path hard-coded here, which no other machine has.
PATH = os.environ.get("GPSTUNA_IQ", "")
FS = 4.096e6                     # ONE file, ONE fs for both constellations:
                                 # cross-file window rounding would pollute
                                 # the inter-system bias by up to 244 ns
LAB = HERE / "lab_local"
GAL_PRNS = [29, 27, 14]
GAL_T0, GAL_T1 = 292.0, 778.0    # track window (pristine zone is t>~290 s)
GPS_TA, GPS_TB = 300.0, 424.0    # GPS decode window
EPOCHS = np.linspace(305.0, 765.0, 8)


# ------------------------------------------------------------ stage: galileo
def stage_gal(prn):
    codes_b, codes_c = load_codes()
    require_capture(PATH)
    mm = np.memmap(PATH, dtype=np.int16, mode="r")
    x = load_seg(PATH, FS, GAL_T0, 40 * T_COH + 0.01)
    rep = {prn: boc_replica(codes_c[prn], FS)}
    r = gal_e1.acquire(x, FS, rep, np.arange(-5000, 5001, 125.0), 40)[prn]
    print(f"[gal] PRN{prn}: acq metric {r['metric']:.2f} "
          f"dopp {r['dopp']:+.0f} Hz C/N0 ~{r['cn0']:.1f} dB-Hz", flush=True)
    if r["metric"] < 2.0:
        print(f"[gal] PRN{prn}: too weak, abort")
        return 1
    tr = track_prn(mm, FS, prn, r["dopp"],
                   int(GAL_T0 * FS) + r["code_phase"], GAL_T1 - GAL_T0,
                   boc_chips(codes_b[prn]), boc_chips(codes_c[prn]))
    pages, st = harvest_pages(tr["sym"])
    print(f"[gal] PRN{prn}: {st['n_pairs']} pairs -> {st['n_crc']} CRC-clean "
          f"pages ({100*st['n_crc']/max(st['n_pairs'],1):.1f}%)", flush=True)
    parsed = [parse_word(pg["page"]) for pg in pages]
    eph = assemble_eph(parsed, prn)
    if eph is None:
        print(f"[gal] PRN{prn}: no IODnav-matched WT1-4 set - abort")
        return 1
    ggto = extract_ggto(parsed)
    ft, stow = anchors_from_pages(pages, parse_word, tr["ptrs"], FS)
    aa, bb = np.polyfit(ft, stow, 1)
    fitres = float(np.std(stow - (aa * ft + bb)))
    # Kepler sanity: orbit radius at toe must sit on the Galileo shell
    rr = float(np.linalg.norm(sat_ecef(eph, eph["toe"]))) / 1e3
    per = eph["sqrtA"] ** 2 * (1 - eph["e"]), eph["sqrtA"] ** 2 * (1 + eph["e"])
    shell_ok = per[0] / 1e3 - 500 < rr < per[1] / 1e3 + 500
    # Doppler model for the snapshot stage (skip 4 s of pull-in)
    tb = tr["ptrs"][1000:] / FS
    fdp = np.polyfit(tb, tr["fds"][1000:], 2)
    hs = eph.get("E1BHS", -1)
    print(f"[gal] PRN{prn}: IODnav {eph['IODnav']}, toe {eph['toe']}, "
          f"|r(toe)|={rr:,.1f} km {'VALID shell' if shell_ok else 'OUT OF RANGE'}")
    print(f"[gal] PRN{prn}: anchors {len(ft)} pages, fit residual "
          f"{fitres*1e6:.1f} us, clock rate {aa-1.0:+.2e}")
    print(f"[gal] PRN{prn}: af0 {eph['af0']*1e6:+.3f} us, "
          f"BGD(E1,E5b) {eph.get('BGD_E1E5b', 0)*1e9:+.2f} ns, "
          f"BGD(E1,E5a) {eph.get('BGD_E1E5a', 0)*1e9:+.2f} ns, "
          f"E1BHS {hs} {'(HEALTHY)' if hs == 0 else '(NOT healthy)'}")
    if ggto:
        print(f"[gal] PRN{prn}: GGTO decoded: A0G {ggto['A0G']*1e9:+.3f} ns, "
              f"A1G {ggto['A1G']:+.3e}, t0G {ggto['t0G']}, WN0G {ggto['WN0G']}")
    else:
        print(f"[gal] PRN{prn}: no valid WT10 GGTO in this track")
    LAB.mkdir(exist_ok=True)
    (LAB / f"joint_gal_prn{prn}.json").write_text(json.dumps({
        "prn": prn, "eph": eph, "ggto": ggto,
        "anchors_ft": ft.tolist(), "anchors_st": stow.tolist(),
        "fd_poly": [float(v) for v in fdp],
        "cn0_mean": float(np.mean(tr["cn0"][2:])),
        "n_crc": int(st["n_crc"]), "fit_res_us": fitres * 1e6,
        "shell_km": rr}, indent=1))
    print(f"[gal] PRN{prn}: cached -> lab_local/joint_gal_prn{prn}.json")
    return 0 if shell_ok else 1


# ---------------------------------------------------------------- stage: gps
def track_sv_win(path, fs, prn, fd0, ta, tb):
    """measure.track_sv with an explicit [ta, tb] window (the stock version
    hardcodes 0.5..dur - useless when the file's first minutes are bad)."""
    from measure import CODE_RATE, FL1
    n1 = int(round(fs * 1e-3))
    times, phases, fds, metrics, pofs = [], [], [], [], []
    for t0 in np.arange(ta, tb - 0.2, 1.0):
        x = load_seg(path, fs, t0, 0.110)
        dops = fd0 + np.arange(-375, 376, 125.0)
        r = gps_acquire(x, fs, [prn], dops, n_noncoh=100)[prn]
        p = prompts_ms(x, fs, prn, r["dopp"], r["code_phase"], 100)
        F = np.fft.fftshift(np.fft.fft(p ** 2, 65536))
        fax = np.fft.fftshift(np.fft.fftfreq(65536, 1e-3))
        fds.append(r["dopp"] + fax[np.argmax(np.abs(F))] / 2.0)
        times.append(t0); phases.append(r["code_phase"])
        metrics.append(r["metric"]); pofs.append(r["peak_over_floor"])
    ph = np.array(phases, float)
    for i in range(1, len(ph)):
        while ph[i] - ph[i - 1] > n1 / 2: ph[i] -= n1
        while ph[i] - ph[i - 1] < -n1 / 2: ph[i] += n1
    t = np.array(times)
    slope, icpt = np.polyfit(t, ph, 1)
    # linear Doppler model centered on the window: satellites drift up to
    # ~0.8 Hz/s, so a fixed median fd leaves +-45 Hz of residual chirp over
    # a 118 s stream - fatal for the p^2 phase reference. prompt_stream
    # picks up fd/fdot/tref from this dict and wipes the chirp.
    fdot, f0 = np.polyfit(t, np.array(fds), 1)
    tref = float(np.mean(t))
    fd = float(fdot * tref + f0)
    spc = fs / CODE_RATE
    lam = np.mean(pofs) - 1.0
    return dict(prn=prn, times=t, phases=ph, fd=fd, fd_std=float(np.std(fds)),
                fdot=float(fdot), tref=tref,
                slope=float(slope), icpt=float(icpt),
                pred_slope=float(-fd / 1540.0 * spc),
                resid_rms=float(np.std(ph - (slope * t + icpt))),
                epoch_ms=float(1023.0 / (CODE_RATE - slope / spc) * 1000.0),
                epoch_pred_ms=float(1.0 / (1.0 + fd / FL1)),
                cn0=float(10 * np.log10(max(lam, 1e-9)) + 30.0),
                metric_mean=float(np.mean(metrics)))


def decode_eph_win(path, fs, prn, dopp, ta, tb):
    """fix.decode_eph with the decode window moved into [ta, tb] (the stock
    version starts at t=1 s). Same union+vote harvest, same anchor output;
    anchor file-times are referenced to the prompt-stream start ta+1."""
    tr = track_sv_win(path, fs, prn, dopp, ta + 0.5, tb - 0.7)
    # bit_tent has no chirp model - hand it the LOCAL fd for its 10 s window
    trt = dict(tr)
    trt["fd"] = tr["fd"] + tr["fdot"] * (ta + 10.0 - tr["tref"])
    tent = int(np.argmax(bit_tent(path, fs, trt, ta + 5.0, 10.0)))
    t0p = ta + 1.0
    p = prompt_stream(path, fs, tr, t0p, min(tb - ta - 3.0, 118.0))
    s, bits, _ = bit_sums(costas(clean_carrier(p)), tent)
    pol, starts = find_grid(bits)
    harvest = harvest_words(p, tent, starts)
    sf_anchors = []
    b = bits ^ pol
    for i in starts:
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
            harvest.append((sf, ubits(words[1], 1, 17) if 1 in words else 0,
                            words))
    eph = parse_harvest(harvest)
    eph["prn"] = prn
    return eph, {"tent": tent, "anchors": sf_anchors, "tr": tr, "t0p": t0p}


def stage_gps(only_prn=None):
    x = load_seg(PATH, FS, GPS_TA + 5.0, 0.310)
    acq = gps_acquire(x, FS, list(range(1, 33)),
                      np.arange(-7000, 7001, 250.0), 300)
    det = {p: r for p, r in acq.items() if r["metric"] > 2.5}
    print(f"[gps] {len(det)} birds acquired at t={GPS_TA+5:.0f} s: "
          f"{sorted(det)}", flush=True)
    todo = [only_prn] if only_prn else sorted(det)
    for prn in todo:
        if prn not in det:
            print(f"[gps] PRN{prn} not acquired, skip")
            continue
        t0 = time.time()
        try:
            eph, tim = decode_eph_win(PATH, FS, prn, det[prn]["dopp"],
                                      GPS_TA, GPS_TB)
        except Exception as e:
            print(f"[gps] PRN{prn}: decode failed ({e})")
            continue
        need = {"sqrtA", "e", "M0", "toe", "omega", "i0", "Omega0", "af0"}
        missing = need - set(eph)
        if missing:
            print(f"[gps] PRN{prn}: incomplete (missing {sorted(missing)}) "
                  f"[{time.time()-t0:.0f} s]")
            continue
        if not tim["anchors"]:
            print(f"[gps] PRN{prn}: no parity-clean anchors")
            continue
        ft = [tim["t0p"] + (tim["tent"] + 20.0 * i) * 1e-3
              for i, _sf, _tw in tim["anchors"]]
        st = [tw * 6.0 - 6.0 for _i, _sf, tw in tim["anchors"]]
        aa, bb = np.polyfit(ft, st, 1)
        fitres = float(np.std(np.array(st) - (aa * np.array(ft) + bb)))
        eph_j = {k: (float(v) if isinstance(v, (int, float, np.floating))
                     else v) for k, v in eph.items() if k != "tows"}
        (LAB / f"joint_gps_prn{prn}.json").write_text(json.dumps({
            "prn": prn, "eph": eph_j, "anchors_ft": ft, "anchors_st": st,
            "fd": tim["tr"]["fd"], "cn0": tim["tr"]["cn0"],
            "metric": det[prn]["metric"], "fit_res_us": fitres * 1e6},
            indent=1))
        print(f"[gps] PRN{prn}: COMPLETE, {len(ft)} anchors, fit residual "
              f"{fitres*1e6:.1f} us, C/N0 {tim['tr']['cn0']:.0f} dB-Hz, "
              f"iono={'iono_a' in eph} [{time.time()-t0:.0f} s]", flush=True)
    return 0


# --------------------------------------------------------------- stage: snap
def stage_snap(ep_sel=None):
    gps = _load_cache("joint_gps_prn*.json")
    gal = _load_cache("joint_gal_prn*.json")
    if not gps or not gal:
        print("[snap] need gps+gal caches first")
        return 1
    codes_b, codes_c = load_codes()
    reps_c = {b["prn"]: boc_replica(codes_c[b["prn"]], FS) for b in gal}
    n1 = int(round(FS * 1e-3))
    snapf = LAB / "joint_snap.json"
    snap = json.loads(snapf.read_text()) if snapf.exists() else {}
    idxs = range(len(EPOCHS)) if ep_sel is None else ep_sel
    for ei in idxs:
        T = float(EPOCHS[ei])
        ent = {"T": T, "gps": {}, "gal": {}}
        # A non-coherent snapshot measures the code phase at the CENTRE of its window (the
        # code drifts Doppler/1540 chips/s through it): centre the window on T so the phase
        # belongs to T. (fix.py's 9/22 bug, -0.15 s x range rate per bird, ~150 m.)
        xg = load_seg(PATH, FS, T - 0.150, 0.310)
        for b in gps:
            prn = b["prn"]
            r = gps_acquire(xg, FS, [prn],
                            np.arange(b["fd"] - 1000, b["fd"] + 1001, 125.0),
                            300)[prn]
            cp = r.get("code_phase_f", r["code_phase"])
            ent["gps"][str(prn)] = {"metric": r["metric"],
                                    "phi_ms": (cp % n1) / FS * 1e3}
        xa = load_seg(PATH, FS, T - 20 * T_COH, 40 * T_COH + 0.01)      # centred on T, as above
        for b in gal:
            prn = b["prn"]
            fd = float(np.polyval(b["fd_poly"], T))
            r = gal_e1.acquire(xa, FS, {prn: reps_c[prn]},
                               np.arange(fd - 250, fd + 251, 125.0), 40)[prn]
            cp = r.get("code_phase_f", r["code_phase"])
            ent["gal"][str(prn)] = {"metric": r["metric"],
                                    "phi_ms": cp / FS * 1e3}
        snap[str(ei)] = ent
        gm = " ".join(f"{p}:{v['metric']:.1f}" for p, v in ent["gps"].items())
        am = " ".join(f"{p}:{v['metric']:.1f}" for p, v in ent["gal"].items())
        print(f"[snap] T={T:5.1f}s  GPS {gm}  GAL {am}", flush=True)
    snapf.write_text(json.dumps(snap, indent=1))
    print(f"[snap] cached -> lab_local/joint_snap.json ({len(snap)} epochs)")
    return 0


def _load_cache(pattern):
    return [json.loads(p.read_text()) for p in sorted(LAB.glob(pattern))]


# -------------------------------------------------------------- stage: solve
# stage_solve(snap_name=..., out_name=...) lets alternate snapshot files (e.g.
# hatch.py's carrier-smoothed code phases) run through the identical assembly
# + integer search + LS machinery, so smoothed-vs-raw is a one-variable A/B.
def solve_lsq(rows, use_isb):
    """rows = [(sat_ecef, pr_m, is_gal)]. LS for (x,y,z,c*dt[,c*isb])."""
    x = np.zeros(5 if use_isb else 4)
    for _ in range(12):
        A, res = [], []
        for sp, pr, g in rows:
            d = x[:3] - sp
            rng = np.linalg.norm(d)
            row = list(d / rng) + [1.0]
            if use_isb:
                row.append(1.0 if g else 0.0)
            A.append(row)
            res.append(pr - (rng + x[3] + (x[4] if use_isb and g else 0.0)))
        dx, *_ = np.linalg.lstsq(np.array(A), np.array(res), rcond=None)
        x = x + dx
        if np.linalg.norm(dx[:3]) < 1e-3:
            break
    return x


def solve_prs_joint(prs, use_isb):
    """prs = [(eph, t_tx_orb, t_tx_rng, is_gal)]: t_tx_orb in the satellite's
    own system timescale (feeds sat_ecef), t_tx_rng on the GPS timeline for
    the pseudorange (identical for GPS; for Galileo minus decoded GGTO in
    4-unknown mode, or as-is with the bias absorbed by the 5th unknown).
    Iterates receive epoch + Sagnac, as fix.solve_prs."""
    t_rx = max(t for _, _, t, _ in prs) + 0.075
    x = np.zeros(5 if use_isb else 4)
    rows = []
    for _ in range(8):
        rows = []
        for eph, to, trg, g in prs:
            sp = sat_ecef(eph, to)
            tau = max(t_rx - trg, 0.0)
            th = OMEGA_E * tau
            rot = np.array([[np.cos(th), np.sin(th), 0],
                            [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
            rows.append((rot @ sp, C * (t_rx - trg), g))
        x = solve_lsq(rows, use_isb)
        t_rx -= x[3] / C
    res = [pr - (np.linalg.norm(x[:3] - sp) + x[3]
                 + (x[4] if use_isb and g else 0.0)) for sp, pr, g in rows]
    rms = float(np.sqrt(np.mean(np.square(res))))
    return rms, x


def assemble(entries, offs, ggto_dt=None, extra=None):
    """SV-time assembly (the fix.py laws, generalized per-system period):
    t_sv_tx = (N + off)*T_code - phi, clock correction AFTER."""
    prs = []
    for k, e in enumerate(entries):
        per = e["per_ms"]
        phi = e["phi_ms"] % per
        frac = (-phi) % per
        N0 = np.round((e["t_sv_coarse"] * 1e3 - frac) / per)
        t_sv = ((N0 + offs[k]) * per + frac) * 1e-3
        t_sys = t_sv - clock_corr(e["eph"], t_sv) + (extra[k] if extra else 0.0)
        t_rng = t_sys - (ggto_dt if (ggto_dt is not None and e["gal"]) else 0.0)
        prs.append((e["eph"], t_sys, t_rng, e["gal"]))
    return prs


def _score(entries, offs, use_isb, ggto_dt, extra=None):
    rms, x = solve_prs_joint(assemble(entries, offs, ggto_dt, extra), use_isb)
    h = ecef_to_llh(x[:3])[2]
    return rms + (0.0 if -3000 < h < 9000 else 1e6), rms, h, x


def _core_gps_search(sub):
    """Robust GPS integer search. The bit-tent integer-ms can be wrong by
    SEVERAL ms on individual birds (seen: +3.5 ms on the strongest bird -
    pinning the outlier makes a +-1 search hopeless). Leave-one-out: for
    each candidate outlier, exhaustively solve the other birds (+-1
    relative, they were all sub-ms), then scan the left-out bird +-6
    integers against that core. Best final score wins."""
    m = len(sub)
    plans = []
    for leftout in range(m):
        subset = [k for k in range(m) if k != leftout]
        sub_e = [sub[k] for k in subset]
        best = None
        for rel in itertools.product((-1, 0, 1), repeat=len(sub_e) - 1):
            offs = (0,) + rel
            sc = _score(sub_e, offs, False, None)
            if best is None or sc[0] < best[0]:
                best = (sc[0], offs)
        best2 = None
        for d in range(-6, 7):
            offs_full = [0] * m
            for k, o in zip(subset, best[1]):
                offs_full[k] = o
            offs_full[leftout] = d
            sc = _score(sub, offs_full, False, None)
            if best2 is None or sc[0] < best2[0]:
                best2 = (sc[0], list(offs_full))
        plans.append(best2)
        if best2[0] < 200.0:          # collapsed: sane altitude + tight rms
            break
    return min(plans)[1]


def search_offsets(entries, use_isb, ggto_dt=None, seed=None):
    """Pinned-reference integer search (a common shift is absorbed by the
    receiver clock). Robust leave-one-out over the GPS birds on GPS-only
    solves, exhaustive over the (few) Galileo birds jointly, then all-bird
    coordinate descent until stable. seed = known-good offsets (same entry
    order) to skip the GPS stage."""
    n = len(entries)
    idx_gal = [k for k, e in enumerate(entries) if e["gal"]]
    idx_gps = [k for k, e in enumerate(entries) if not e["gal"]]
    total = [0] * n
    if seed is not None:
        for k in range(min(len(seed), n)):
            total[k] = seed[k]
    else:
        sub = [entries[k] for k in idx_gps]
        for k, o in zip(idx_gps, _core_gps_search(sub)):
            total[k] = o
    # Galileo exhaustive with GPS frozen, full joint solve (their page
    # anchors are sample-stamped -> coarse times are us-accurate, so +-1
    # covers only the rounding edges)
    if idx_gal:
        best = None
        for rel in itertools.product((-1, 0, 1), repeat=len(idx_gal)):
            offs = list(total)
            for k, o in zip(idx_gal, rel):
                offs[k] = o
            sc = _score(entries, offs, use_isb, ggto_dt)
            if best is None or sc[0] < best[0]:
                best = (*sc, tuple(offs))
        total = list(best[4])
    # coordinate descent, all birds except the pinned reference
    for _pass in range(4):
        moved = False
        for k in range(1, n):
            cands = []
            for d in (-1, 0, 1):
                offs = list(total)
                offs[k] += d
                cands.append((_score(entries, offs, use_isb, ggto_dt)[0], d))
            d = min(cands)[1]
            if d:
                total[k] += d
                moved = True
        if not moved:
            break
    return total


def stage_solve(snap_name="joint_snap.json", out_name="fix_result_joint.json"):
    gps = _load_cache("joint_gps_prn*.json")
    gal = _load_cache("joint_gal_prn*.json")
    snap = json.loads((LAB / snap_name).read_text())
    print(f"[solve] {len(gps)} GPS + {len(gal)} Galileo birds, "
          f"{len(snap)} snapshot epochs")

    # fits: file time -> SV time per bird
    fits = {}
    for b in gps + gal:
        key = ("GAL" if "fd_poly" in b else "GPS", b["prn"])
        aa, bb = np.polyfit(b["anchors_ft"], b["anchors_st"], 1)
        fits[key] = (aa, bb)

    # Galileo TOW convention cross-check vs GPS (both TOWs count the same
    # week seconds; GST-GPS is ns). System transmit times at T seen from
    # each constellation must differ only by propagation-delay spread.
    T = float(EPOCHS[len(EPOCHS) // 2])
    tg = [fits[("GPS", b["prn"])][0] * T + fits[("GPS", b["prn"])][1]
          - clock_corr(b["eph"], fits[("GPS", b["prn"])][0] * T
                       + fits[("GPS", b["prn"])][1]) for b in gps]
    ta = [fits[("GAL", b["prn"])][0] * T + fits[("GAL", b["prn"])][1]
          - clock_corr(fold_bgd(b["eph"]),
                       fits[("GAL", b["prn"])][0] * T
                       + fits[("GAL", b["prn"])][1]) for b in gal]
    dconv = float(np.median(ta) - np.median(tg))
    print(f"[solve] GAL-vs-GPS coarse transmit-time offset {dconv*1e3:+.1f} ms "
          f"(propagation spread only = convention correct; integer seconds "
          f"= page-TOW convention error)")
    if abs(round(dconv)) >= 1:
        print(f"[solve] applying delta = {-round(dconv):+d} s to Galileo fits")
        for b in gal:
            key = ("GAL", b["prn"])
            fits[key] = (fits[key][0], fits[key][1] - round(dconv))

    # iono terms: any GPS bird's page-18 decode, else the newest archive
    iono = next((b["eph"] for b in gps if "iono_a" in b["eph"]), None)
    if iono is None and (LAB / "iono_terms.json").exists():
        iono = json.loads((LAB / "iono_terms.json").read_text())
        print(f"[solve] ionosphere: archived Klobuchar terms "
              f"({iono.get('src', '?')})")
    else:
        print(f"[solve] ionosphere: {'decoded page-18' if iono else 'NONE'}")

    # decoded GGTO (any bird's WT10; constellation-wide)
    ggto = next((b["ggto"] for b in gal if b.get("ggto")), None)
    wn_gst = next((b["eph"].get("WN") for b in gal if b["eph"].get("WN")), 0)

    def entries_at(ei, use14):
        ent = []
        for b in sorted(gps, key=lambda b: -snap[str(ei)]["gps"]
                        .get(str(b["prn"]), {"metric": 0})["metric"]):
            s = snap[str(ei)]["gps"].get(str(b["prn"]))
            if not s or s["metric"] < 2.0:
                continue
            aa, bb = fits[("GPS", b["prn"])]
            T = snap[str(ei)]["T"]
            eph = dict(b["eph"])
            if iono is not None:
                eph["iono_a"], eph["iono_b"] = iono["iono_a"], iono["iono_b"]
            ent.append({"prn": b["prn"], "gal": False, "per_ms": 1.0,
                        "eph": eph, "phi_ms": s["phi_ms"],
                        "t_sv_coarse": aa * T + bb})
        for b in gal:
            if not use14 and b["eph"].get("E1BHS", 0) != 0:
                continue
            s = snap[str(ei)]["gal"].get(str(b["prn"]))
            if not s or s["metric"] < 2.0:
                continue
            aa, bb = fits[("GAL", b["prn"])]
            T = snap[str(ei)]["T"]
            ent.append({"prn": b["prn"], "gal": True, "per_ms": 4.0,
                        "eph": fold_bgd(b["eph"]), "phi_ms": s["phi_ms"],
                        "t_sv_coarse": aa * T + bb})
        return ent

    def atmos_extra(entries, offs, use_isb, ggto_dt):
        _, _, _, x0 = _score(entries, offs, use_isb, ggto_dt)
        lat0, lon0, _ = ecef_to_llh(x0[:3])
        extra, els = [], []
        for k, e in enumerate(entries):
            prs = assemble(entries, offs, ggto_dt)
            az, el = fix.az_el(x0[:3], sat_ecef(e["eph"], prs[k][1]))
            els.append(round(float(np.degrees(el)), 1))
            d = fix.tropo_delay(el) / C
            if "iono_a" in e["eph"]:
                d += fix.klobuchar(e["eph"]["iono_a"], e["eph"]["iono_b"],
                                   lat0, lon0, az, el, prs[k][1])
            elif iono is not None:      # same iono at the same L1 frequency
                d += fix.klobuchar(iono["iono_a"], iono["iono_b"],
                                   lat0, lon0, az, el, prs[k][1])
            extra.append(d)
        return extra, els

    per_epoch = []
    for ei in sorted(snap, key=int):
        T = snap[ei]["T"]
        row = {"T": T}
        # decoded GGTO at this epoch (GST tow ~ any gal bird's coarse time)
        ggto_dt = None
        if ggto is not None:
            tow_now = fits[("GAL", gal[0]["prn"])][0] * T \
                + fits[("GAL", gal[0]["prn"])][1]
            ggto_dt = ggto_eval(ggto, wn_gst, tow_now)
        offs_g_healthy = None
        for tag, use14 in (("healthy", False), ("with14", True)):
            ent = entries_at(ei, use14)
            ng = sum(1 for e in ent if not e["gal"])
            na = sum(1 for e in ent if e["gal"])
            if ng < 4 or na < 1:
                continue
            r = {"n_gps": ng, "n_gal": na}
            if tag == "healthy":
                # GPS-only baseline at the same epoch
                sub = [e for e in ent if not e["gal"]]
                offs_g = search_offsets(sub, False)
                offs_g_healthy = list(offs_g)
                ex_g, _ = atmos_extra(sub, offs_g, False, None)
                _, rms_g, h_g, x_g = _score(sub, offs_g, False, None, ex_g)
                r.update({"gps_rms": rms_g, "gps_alt": h_g,
                          "x_gps": x_g[:3].tolist()})
                seed5 = offs_g + [0] * na
            else:
                seed5 = (offs_g_healthy + [0] * na
                         if offs_g_healthy is not None else None)
            # JOINT, 5 unknowns (estimated inter-system bias)
            offs5 = search_offsets(ent, True, seed=seed5)
            _, _, _, x5r = _score(ent, offs5, True, None)
            # REBASE: pinning makes the GPS integers only RELATIVELY right -
            # a common integer-ms error rides in the receiver clock and
            # reappears as ms in the isb (Galileo's page anchors are
            # absolute to us). Estimate it from the isb and move the whole
            # GPS block, so the isb collapses to the physical GGTO + rx
            # inter-signal delay and the decoded-GGTO mode has a chance.
            s_ms = int(np.round(x5r[4] / (C * 1e-3)))
            if s_ms != 0:
                for k, e in enumerate(ent):
                    if not e["gal"]:
                        offs5[k] -= s_ms
                offs5 = search_offsets(ent, True, seed=offs5)
            ex5, els = atmos_extra(ent, offs5, True, None)
            _, rms5, h5, x5 = _score(ent, offs5, True, None, ex5)
            isb_ns = -x5[4] / C * 1e9 if len(x5) > 4 else None
            if isb_ns is not None:      # defined mod the 4 ms code period
                isb_ns = (isb_ns + 2e6) % 4e6 - 2e6
            r.update({"joint5_rms": rms5, "joint5_alt": h5,
                      "x5": x5[:3].tolist(), "isb_ns": isb_ns,
                      "el_deg": els})
            # JOINT, 4 unknowns with the DECODED GGTO applied
            if ggto_dt is not None and tag == "healthy":
                offs4 = search_offsets(ent, False, ggto_dt, seed=offs5)
                ex4, _ = atmos_extra(ent, offs4, False, ggto_dt)
                _, rms4, h4, x4 = _score(ent, offs4, False, ggto_dt, ex4)
                r.update({"joint4_rms": rms4, "joint4_alt": h4,
                          "x4": x4[:3].tolist()})
            row[tag] = r
        if "healthy" in row:
            r = row["healthy"]
            j4 = (f" | j4 rms {row['healthy']['joint4_rms']:7.1f} alt "
                  f"{row['healthy']['joint4_alt']:6.0f}"
                  if "joint4_rms" in r else "")
            print(f"[solve] T={T:5.1f}s {r['n_gps']}G+{r['n_gal']}E | "
                  f"GPS-only rms {r['gps_rms']:7.1f} m alt {r['gps_alt']:8.0f}"
                  f" | JOINT5 rms {r['joint5_rms']:7.1f} m alt "
                  f"{r['joint5_alt']:6.0f} isb {r['isb_ns']:+8.1f} ns{j4}",
                  flush=True)
        per_epoch.append(row)

    # ---- aggregate (healthy set = the headline)
    def agg(key_x, key_rms, key_alt, tag="healthy"):
        rows = [r[tag] for r in per_epoch
                if tag in r and key_x in r[tag]
                and -500 < r[tag][key_alt] < 5000]
        if not rows:
            return None
        P = np.array([r[key_x] for r in rows])
        lat, lon, h = ecef_to_llh(P.mean(axis=0))
        return {"lat": lat, "lon": lon, "alt_m": h,
                "rms": float(np.mean([r[key_rms] for r in rows])),
                "scatter_m": float(np.linalg.norm(P.std(axis=0)))
                if len(rows) > 1 else 0.0,
                "epochs_used": len(rows)}
    res_gps = agg("x_gps", "gps_rms", "gps_alt")
    res_j5 = agg("x5", "joint5_rms", "joint5_alt")
    res_j4 = agg("x4", "joint4_rms", "joint4_alt")
    res_j5_14 = agg("x5", "joint5_rms", "joint5_alt", "with14")
    isbs = [r["healthy"]["isb_ns"] for r in per_epoch
            if "healthy" in r and r["healthy"].get("isb_ns") is not None
            and -500 < r["healthy"]["joint5_alt"] < 5000]
    dec_ns = None
    if ggto is not None:
        tow_mid = fits[("GAL", gal[0]["prn"])][0] * float(EPOCHS[3]) \
            + fits[("GAL", gal[0]["prn"])][1]
        dec_ns = ggto_eval(ggto, wn_gst, tow_mid) * 1e9

    out = {"capture": PATH, "fs": FS,
           "gps_birds": sorted(b["prn"] for b in gps),
           "gal_birds": sorted(b["prn"] for b in gal),
           "gal_healthy": sorted(b["prn"] for b in gal
                                 if b["eph"].get("E1BHS", 0) == 0),
           "per_epoch": per_epoch, "gps_only": res_gps,
           "joint5": res_j5, "joint4": res_j4, "joint5_with14": res_j5_14,
           "ggto_decoded_ns": dec_ns,
           "ggto_A0G_ns": ggto["A0G"] * 1e9 if ggto else None,
           "isb_est_ns_mean": float(np.mean(isbs)) if isbs else None,
           "isb_est_ns_std": float(np.std(isbs)) if isbs else None,
           "bgd_e1e5b_ns": {str(b["prn"]): b["eph"].get("BGD_E1E5b", 0) * 1e9
                            for b in gal},
           "tow_convention_offset_s": dconv}
    (LAB / out_name).write_text(json.dumps(out, indent=1))

    print("\n" + "=" * 72)
    for name, r in (("GPS-only", res_gps), ("JOINT 5-unk", res_j5),
                    ("JOINT 4-unk (decoded GGTO)", res_j4),
                    ("JOINT 5-unk incl PRN14", res_j5_14)):
        if r:
            sane = -500 < r["alt_m"] < 5000
            print(f"{name:27s}: mean rms {r['rms']:8.1f} m, scatter "
                  f"{r['scatter_m']:7.1f} m, alt {r['alt_m']:7.0f} m "
                  f"({'PLAUSIBLE' if sane else 'IMPLAUSIBLE'}), "
                  f"{r['epochs_used']} epochs")
        else:
            print(f"{name:27s}: no sane epochs")
    if isbs:
        print(f"inter-system bias (5th unknown): {np.mean(isbs):+.1f} ns "
              f"+- {np.std(isbs):.1f} ns across {len(isbs)} epochs")
    if dec_ns is not None:
        print(f"decoded GGTO (WT10, Eq 21):      {dec_ns:+.1f} ns "
              f"(t_Galileo - t_GPS)")
        if isbs:
            print(f"  estimated-minus-decoded = {np.mean(isbs)-dec_ns:+.1f} ns"
                  f" (residual = receiver BOC-vs-BPSK group delay + solve "
                  f"noise)")
    print(f"coordinates written ONLY to lab_local/{out_name}")
    print("=" * 72)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", default=None,
                    help="the capture every stage reads (4.096 Msps for the "
                         "joint GPS+Galileo chain); or set GPSTUNA_IQ")
    ap.add_argument("--stage", required=True,
                    choices=["gal", "gps", "snap", "solve"])
    ap.add_argument("--prn", type=int, default=None)
    ap.add_argument("--epochs", default=None,
                    help="snap only: e.g. 0-3 or 4,5,6")
    ap.add_argument("--snap", default="joint_snap.json",
                    help="solve only: alternate snapshot file in lab_local/")
    ap.add_argument("--out", default="fix_result_joint.json",
                    help="solve only: result file name in lab_local/")
    a = ap.parse_args()
    global PATH
    if a.iq:
        PATH = a.iq
    if a.stage != "solve" and not PATH:
        ap.error("--iq CAPTURE is required for this stage (or set GPSTUNA_IQ)")
    if a.stage == "gal":
        return stage_gal(a.prn)
    if a.stage == "gps":
        return stage_gps(a.prn)
    if a.stage == "snap":
        sel = None
        if a.epochs:
            if "-" in a.epochs:
                lo, hi = a.epochs.split("-")
                sel = list(range(int(lo), int(hi) + 1))
            else:
                sel = [int(v) for v in a.epochs.split(",")]
        return stage_snap(sel)
    return stage_solve(a.snap, a.out)


if __name__ == "__main__":
    sys.exit(main())
