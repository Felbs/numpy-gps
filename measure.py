#!/usr/bin/env python3
"""Dig GPS L1 C/A satellites out of thermal noise with a TV antenna.

GPS L1 (1575.42 MHz) arrives at about -130 dBm — roughly 20 dB BELOW
the thermal noise floor of a 2 MHz channel. You cannot see it on any
spectrum display. What rescues it is the grid: every satellite
transmits a known 1023-chip Gold code at exactly 1.023 Mchip/s
(one code epoch per millisecond), BPSK-spread, with 50 bps navigation
data on top. Correlating 1 ms of signal against a local code replica
concentrates the signal 30 dB (10*log10(1023)); noncoherent stacking
of a few hundred epochs buys the rest.

This script derives, from raw wideband IQ (no LNA, no GPS antenna):
  1. C/A code generator self-check vs published IS-GPS-200 octals
  2. parallel code-phase acquisition: all 32 PRNs x Doppler bins
     (FFT circular correlation, 1 ms coherent x N noncoherent)
  3. per-SV track over the capture: code-phase drift must equal
     -carrier_doppler/1540 (the code clock and carrier are locked to
     the same atomic standard 1540 half-cycles apart)
  4. measured C/A epoch length vs the value the carrier predicts
  5. the 50 bps navigation-data bit grid (20 ms alignment energy)
  6. optional: TLE cross-check (which SVs were overhead, predicted
     Doppler per SV, common receiver clock offset)

Usage:
  python measure.py --iq gps_l1_rabbit.cs16 --fs 2048000
        [--iq2 gps_l1_discone.cs16]   # second capture for the acq figure
        [--tle gps_ops.tle]           # celestrak gps-ops TLE file
        [--t0 "2026-07-20T12:14:27Z"] # capture start UTC (for TLE check)
        [--lat <deg> --lon <deg>]     # optional, only for --tle sky-check
        [--selftest]                  # synthetic -20 dB proof, then exit

Privacy note: the default lat/lon is a deliberately coarse central-
Virginia grid point. Elevation and Doppler predictions change by less
than the TLE error over ~100 km, so a vague point is all this needs.
This script never computes a position fix.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FL1 = 1575.42e6
CODE_RATE = 1.023e6
C_LIGHT = 299792458.0
# Peak / second-peak detection threshold of the SKY SEARCH (sky_search()).
# Measured 2026-11-13 on two attic/patch captures, ten snapshots, every PRN
# that was not in the sky (215 trials each): the metric of an absent
# satellite never passed 1.24 with 1 ms coherent or 1.29 with 4 ms. The old
# 2.5 therefore left ~4 dB unused - on the August capture it rejected PRN 3
# (38 dB-Hz, metric 2.0 at 1 ms / 3.0 at 4 ms), which gr-gpsrx tracks and
# decodes for the whole file. The idea of testing it came from a same-file
# comparison with OnAir's receiver (ideas only, no code).
THRESH = 1.5
ACQ_COH = 4           # ms added coherently in the sky search
ACQ_STEP = 125.0      # Hz: a 4 ms sum resolves 250 Hz, so half of that

# 4-series categorical palette (Okabe-Ito subset, CVD-validated)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

# ---------------------------------------------------------------- C/A codes
# IS-GPS-200 G2 phase-selector taps (1-indexed) for PRN 1..32
G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10),
    7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
    13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5),
    19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
    25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}
# first 10 chips as octal, from IS-GPS-200 (generator must reproduce these)
PUBLISHED_OCTAL = {1: "1440", 2: "1620", 3: "1710", 4: "1744", 5: "1133"}


_CA_CACHE = {}
_SAMPLED_CACHE = {}


def ca_code(prn):
    """1023-chip C/A code, +1/-1 floats. G1: x^10+x^3+1; G2: x^10+x^9+x^8+x^6+x^3+x^2+1.

    Cached: the LFSR is bit-exact and pure, and before the cache it was being
    re-run 10,693 times in one fix (once per 0.1 s prompt chunk) -- 57 s of a
    390 s run spent regenerating the same 1023 chips. Returns a READ-ONLY
    array so a caller cannot corrupt the shared copy."""
    c = _CA_CACHE.get(prn)
    if c is not None:
        return c
    s1, s2 = G2_TAPS[prn]
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    out = np.empty(1023, dtype=int)
    for i in range(1023):
        out[i] = g1[9] ^ g2[s1 - 1] ^ g2[s2 - 1]
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.concatenate(([fb1], g1[:9]))
        g2 = np.concatenate(([fb2], g2[:9]))
    c = 1.0 - 2.0 * out
    c.setflags(write=False)
    _CA_CACHE[prn] = c
    return c


def generator_selfcheck():
    ok = True
    for prn, want in PUBLISHED_OCTAL.items():
        chips = (1.0 - ca_code(prn)[:10]) / 2.0
        got = format(int("".join(str(int(b)) for b in chips), 2), "04o")
        ok &= got == want
        print(f"  PRN{prn:2d} first-10-chips octal: {got} (published {want})"
              f" {'OK' if got == want else 'FAIL'}")
    if not ok:
        raise SystemExit("C/A generator failed the published check values -- stop.")


def sampled_code(prn, fs, n_samp):
    """The C/A code sampled at fs over n_samp samples. Cached per
    (prn, fs, n_samp) and read-only, for the same reason as ca_code."""
    key = (prn, float(fs), int(n_samp))
    c = _SAMPLED_CACHE.get(key)
    if c is None:
        idx = (np.arange(n_samp) * CODE_RATE / fs).astype(np.int64) % 1023
        c = ca_code(prn)[idx]
        c.setflags(write=False)
        _SAMPLED_CACHE[key] = c
    return c


# ------------------------------------------------------------- acquisition
def _acquire_rows(blocks, fs, prns, fd, coh=1):
    """One Doppler: noncoherent power vs code phase for every PRN, in prns
    order. The unit of work for both the serial and the parallel path.
    coh > 1: that many 1 ms correlations are added COHERENTLY before the
    power is taken (see acquire())."""
    nb, n1 = blocks.shape
    t = np.arange(n1) / fs
    BF = np.fft.fft(blocks * np.exp(-2j * np.pi * fd * t)[None, :], axis=1)
    if coh > 1:
        # the wipe-off restarts its phase in every block; a coherent sum
        # needs the carrier to run on from one block to the next
        nb = (nb // coh) * coh
        BF = BF[:nb] * np.exp(-2j * np.pi * fd * 1e-3 * np.arange(nb))[:, None]
    rows = []
    for p in prns:
        code_f = np.conj(np.fft.fft(sampled_code(p, fs, n1)))
        corr = np.fft.ifft(BF * code_f[None, :], axis=1)
        if coh > 1:
            corr = corr.reshape(nb // coh, coh, n1).sum(axis=1)
        rows.append((np.abs(corr) ** 2).sum(axis=0))
    return rows


_ACQ = {}                                   # worker-side: the shared blocks


def _acq_init(blocks, fs, prns, coh=1):
    _ACQ["blocks"], _ACQ["fs"], _ACQ["prns"] = blocks, fs, prns
    _ACQ["coh"] = coh


def _acq_task(fd):
    return _acquire_rows(_ACQ["blocks"], _ACQ["fs"], _ACQ["prns"], fd,
                         _ACQ.get("coh", 1))


def _pool_allowed():
    """Only the main process may open a pool (no nesting), only when it
    would help, and only when the spawned children can re-import __main__:
    under `python -`, `python -c` or a REPL there is no main FILE, the
    children die at bootstrap and Pool.map would wait forever (found 8/15,
    the hard way). GPSTUNA_SERIAL=1 forces the serial path everywhere."""
    import multiprocessing as _mp
    import os as _os
    import sys as _sys
    main = _sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    return (_mp.current_process().name == "MainProcess"
            and (_os.cpu_count() or 1) > 1
            and _os.environ.get("GPSTUNA_SERIAL", "0") != "1"
            and bool(main_file) and _os.path.isfile(str(main_file)))


def pool_timeout(serial_estimate_s):
    """A pool that has not answered in ~10x the serial estimate (min 5 min)
    is not slow, it is dead -- respawning workers that die at import look
    exactly like a long job. Callers fall back to serial on expiry."""
    return max(300.0, 10.0 * float(serial_estimate_s))


def pool_silence(default_s):
    """How long a pool may go without delivering ANY result before it is
    declared dead. The total-job timeout above scales with the serial
    estimate and reached 61 MINUTES on a Pi 5 (measured 8/24, war-drive
    capture 4: one worker went quiet, the other three sat idle, the parent
    waited the hour out). Healthy workers hand back a result every few
    seconds, so silence -- not elapsed time -- is the honest signal.
    GPSTUNA_POOL_SILENCE overrides (seconds)."""
    import os as _os
    try:
        return float(_os.environ.get("GPSTUNA_POOL_SILENCE", default_s))
    except ValueError:
        return float(default_s)


def drain(async_results, silence_s, pool=None):
    """Collect apply_async results in order. Two ways to give up, both
    measured 8/24 on the Pi 5 war-drive batch:

    * a worker DIED (when `pool` is given): multiprocessing replaces it
      silently and the task it was holding is never re-queued -- capture 4
      sat 33 min at 0 % CPU on exactly that. The worker set is checked every
      half second; a replaced or exited worker raises at once, naming the
      pid and exit code (negative = the signal that killed it).
    * SILENCE: no result from ANY task for silence_s, counted from the last
      arrival. The first version timed each result alone and in order, so
      the first one carried a whole bird's decode: on a loaded Pi a HEALTHY
      5-bird pool tripped the 300 s (capture 6), was torn down, and the
      birds were decoded again serially. With death detection carrying the
      fast case, silence is only the backstop and can afford to be generous.

    Raises RuntimeError (death) or multiprocessing.TimeoutError (silence).
    (AsyncResult.get is the one pool API guaranteed to exist: Pool.imap
    hands back a bare generator on this interpreter.)"""
    import time as _time
    from multiprocessing import TimeoutError as _Timeout
    results = list(async_results)
    workers = list(getattr(pool, "_pool", None) or []) if pool is not None else []
    pids0 = {p.pid for p in workers}
    last, n_seen = _time.monotonic(), 0
    out = []
    for r in results:
        while not r.ready():
            r.wait(0.5)
            now = _time.monotonic()
            n_ready = sum(1 for x in results if x.ready())
            if n_ready > n_seen:
                last, n_seen = now, n_ready
            if pool is not None:
                cur = list(getattr(pool, "_pool", None) or [])
                if ({p.pid for p in cur} != pids0
                        or any(p.exitcode is not None for p in cur)):
                    dead = [(p.pid, p.exitcode) for p in workers + cur
                            if p.exitcode is not None]
                    raise RuntimeError("pool worker died and was replaced "
                                       f"(pid, exit code): {dead or 'unknown'}")
            if now - last > silence_s:
                raise _Timeout(f"no result for {silence_s:.0f} s "
                               f"({n_seen}/{len(results)} done)")
        out.append(r.get())
    return out


def _acq_tasks(fds):
    return [_acq_task(fd) for fd in fds]


def _acquire_rows_parallel(blocks, fs, prns, dopplers, coh=1):
    """Doppler-parallel _acquire_rows. Returns rows in dopplers order, or
    None if a pool cannot be started (caller falls back to serial)."""
    import multiprocessing as _mp
    import os as _os
    # measured 8/15 on a 64-core box: 6 workers 9.3 s, 10 -> 8.1 s, 16 -> 8.9,
    # 24 -> 10.1 (serial 27 s). Past ~10 the interpreter start-ups and memory
    # bandwidth eat the gain; a 4-core Pi is capped by its cores anyway.
    n_workers = min(len(dopplers), max(1, _os.cpu_count() or 1), 10)
    if n_workers < 2:
        return None
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _os.environ.setdefault(_v, "1")
    try:
        ctx = _mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_acq_init,
                      initargs=(np.ascontiguousarray(blocks), fs, prns,
                                coh)) as pool:
            # serial cost ~ 0.5 ms per (PRN, Doppler, block) on a slow core
            fds = list(dopplers)
            cs = max(1, len(fds) // (3 * n_workers))
            chunks = [fds[i:i + cs] for i in range(0, len(fds), cs)]
            ars = [pool.apply_async(_acq_tasks, (c,)) for c in chunks]
            # ~30 s of work per worker on a Pi; 2 min of silence is dead
            return [row for rows in drain(ars, pool_silence(120), pool)
                    for row in rows]
    except Exception as e:                                   # noqa: BLE001
        print(f"  (acquisition pool unavailable: {type(e).__name__}: {e}; "
              f"searching serially)")
        return None


def acquire(x, fs, prns, dopplers, n_noncoh, coh=1):
    """Parallel code-phase search. 1 ms coherent x n_noncoh noncoherent.
    Returns {prn: dict(metric, dopp, code_phase, peak_over_floor)} where
    metric = peak / second peak (2nd peak excludes +-1 chip at all Dopplers).

    coh: milliseconds added coherently before the power is taken (n_noncoh
    ms are used in all, so n_noncoh // coh powers are summed). Worth ~3 dB
    on a weak satellite per doubling; the Doppler grid must then be no
    coarser than 500 / coh Hz (a T-second sum resolves 1/T Hz), and a nav
    bit that flips inside a group spoils that group (one in ten at 4 ms)."""
    coh = max(1, int(coh))
    n1 = int(round(fs * 1e-3))
    blocks = x[: n1 * n_noncoh].reshape(n_noncoh, n1)
    excl = int(round(fs / CODE_RATE))
    prns = list(prns)
    dopplers = np.asarray(dopplers, dtype=float)
    # The sky-view search (32 PRNs x 57 Dopplers x 400 ms) is 30 s on one
    # core and embarrassingly parallel over Dopplers: each worker FFTs its
    # Doppler's blocks once and correlates every PRN -- no duplicated work,
    # and the per-Doppler power rows are stacked in the same order, so the
    # maps are IDENTICAL to the serial ones. Small searches (a track step:
    # 1 PRN x 7 Dopplers) and anything already inside a worker stay serial.
    rows = None
    if (len(prns) * len(dopplers) * n_noncoh >= 200_000
            and _pool_allowed()):
        rows = _acquire_rows_parallel(blocks, fs, prns, dopplers, coh)
    if rows is None:
        rows = [_acquire_rows(blocks, fs, prns, fd, coh) for fd in dopplers]
    out = {}
    for pi, p in enumerate(prns):
        m = np.vstack([r[pi] for r in rows])
        di, ci = np.unravel_index(np.argmax(m), m.shape)
        mask = np.ones(m.shape[1], dtype=bool)
        for off in range(-excl, excl + 1):
            mask[(ci + off) % m.shape[1]] = False
        # sub-sample peak (parabolic on the power triplet): one integer
        # sample is 146 m of pseudorange at 2.048 MS/s - the fractional
        # peak recovers most of it. code_phase stays int for all existing
        # consumers; code_phase_f is the refined value for solvers.
        row = m[di]
        y1, y2, y3 = (row[(ci - 1) % len(row)], row[ci],
                      row[(ci + 1) % len(row)])
        den = y1 - 2.0 * y2 + y3
        frac = float(np.clip(0.5 * (y1 - y3) / den, -0.5, 0.5)) if den else 0.0
        out[p] = dict(metric=float(m[di, ci] / m[:, mask].max()),
                      dopp=float(dopplers[di]), code_phase=int(ci),
                      code_phase_f=float(ci + frac),
                      peak_over_floor=float(m[di, ci] / np.median(m)))
    return out


def sky_search(x, fs, n_ms, prns=None, span_hz=7000.0):
    """The whole-sky search every entry point starts with: all 32 PRNs,
    +-span_hz, ACQ_COH ms coherent over n_ms ms of x. Returns acquire()'s
    dict; a satellite is present when its metric > THRESH."""
    prns = list(range(1, 33)) if prns is None else list(prns)
    dops = np.arange(-span_hz, span_hz + 1, ACQ_STEP)
    return acquire(x, fs, prns, dops, n_ms, coh=ACQ_COH)


def check_sidecar(path, fs, expect_dtype="int16"):
    """If a SigMF sidecar (<name>.sigmf-meta) sits beside the capture, READ
    it and refuse a mismatch loudly. Every tool here assumes interleaved
    int16 at `fs`; a recording at another rate or datatype does not fail --
    it acquires nothing and the advice that follows says "move the antenna",
    which is the wrong advice. The one external capture this project has
    seen so far arrived as SigMF (8/15), so this is the common case, not
    the edge case. Returns the sidecar's sample rate when it names one."""
    import json as _json
    import os as _os
    base, ext = _os.path.splitext(path)
    cands = [base + ".sigmf-meta", path + ".sigmf-meta"]
    if ext.lower() == ".sigmf-data":
        cands.insert(0, base + ".sigmf-meta")
    for c in cands:
        if not _os.path.isfile(c):
            continue
        try:
            with open(c, "r", encoding="utf-8", errors="replace") as fh:
                g = (_json.load(fh) or {}).get("global", {}) or {}
        except Exception:                                    # noqa: BLE001
            return None
        problems = []
        rate = g.get("core:sample_rate")
        dt = str(g.get("core:datatype", "")).lower()
        if rate is not None and abs(float(rate) - fs) > 1.0:
            problems.append(f"sample rate {float(rate):,.0f} Sps, but this "
                            f"run assumes {fs:,.0f} (pass --fs {float(rate):.0f}"
                            f" if the tool has it, or resample the file)")
        if dt and not dt.startswith(("ci16", "cs16")):
            problems.append(f"datatype '{g.get('core:datatype')}', but these "
                            f"tools read interleaved int16 (ci16_le); "
                            f"convert first")
        if problems:
            raise SystemExit(
                f"\nThe SigMF sidecar {c} says this capture is not what the "
                f"tools assume:\n  * " + "\n  * ".join(problems) +
                "\nRefusing to run: at the wrong rate acquisition finds "
                "nothing and the advice that follows\n(\"move the antenna\") "
                "would be wrong.\n")
        return float(rate) if rate is not None else None
    return None


def require_capture(path):
    """Fail with an explanation instead of a FileNotFoundError traceback.

    Every tool here needs raw L1 IQ, and no sample capture ships with the
    repo -- see the README: a GPS recording encodes where and when it was
    made, so publishing one would publish a position.
    """
    import os as _os
    if path and _os.path.exists(path):
        return path
    raise SystemExit(
        f"\nNo IQ capture at:\n    {path}\n\n"
        "These tools read raw GPS L1 baseband; none ships with the repo\n"
        "(a capture encodes the position and time it was made).\n\n"
        "  * record your own -- 1575.42 MHz, 2.048 Msps, interleaved int16,\n"
        "    an active GPS patch antenna with sky view, 60 s or more:\n"
        "        python locate.py                 # if you have SoapySDR\n"
        "  * then point any tool at it:\n"
        "        python measure.py --iq your_capture.cs16\n\n"
        "To check the install with no radio and no capture:\n"
        "        python measure.py --selftest\n")


def load_seg(path, fs, t0_s, dur_s):
    require_capture(path)
    n0 = int(t0_s * fs) * 2
    n = int(dur_s * fs) * 2
    raw = np.memmap(path, dtype=np.int16, mode="r")[n0:n0 + n].astype(np.float32)
    # interleaved I,Q float32 IS the memory layout of complex64: a view, not
    # a second pass building r[0::2] + 1j*r[1::2] (measured 12 -> 0 ms per
    # second of data; this function ran 9,799 times in one fix)
    x = raw.view(np.complex64)
    return x - x.mean()


def noise_stats(path, fs):
    x = load_seg(path, fs, 5.0, 1.0)
    i, q = x.real, x.imag
    clip = max(np.mean(np.abs(i) >= 32000), np.mean(np.abs(q) >= 32000))
    print(f"  RMS {np.sqrt(np.mean(np.abs(x)**2)):.0f} counts, clipped "
          f"{clip*100:.3f}% -- {'AGC left headroom' if clip < 1e-4 else 'CLIPPING'}")


# ----------------------------------------------------------------- tracking
def prompts_ms(x, fs, prn, fd, code_phase, n_ms):
    n1 = int(round(fs * 1e-3))
    code = np.roll(sampled_code(prn, fs, n1), code_phase)
    t = np.arange(n1 * n_ms) / fs
    xw = x[: n1 * n_ms] * np.exp(-2j * np.pi * fd * t)
    return (xw.reshape(n_ms, n1) * code[None, :]).sum(axis=1)


def track_sv(path, fs, prn, fd0, dur_s):
    """Re-acquire in 1 s steps; fine Doppler from FFT of squared prompts."""
    n1 = int(round(fs * 1e-3))
    times, phases, fds, metrics, pofs = [], [], [], [], []
    for t0 in np.arange(0.5, dur_s - 0.7, 1.0):
        x = load_seg(path, fs, t0, 0.110)
        dops = fd0 + np.arange(-375, 376, 125.0)
        r = acquire(x, fs, [prn], dops, n_noncoh=100)[prn]
        p = prompts_ms(x, fs, prn, r["dopp"], r["code_phase"], 100)
        F = np.fft.fftshift(np.fft.fft(p ** 2, 65536))       # squaring wipes data bits
        fax = np.fft.fftshift(np.fft.fftfreq(65536, 1e-3))
        fds.append(r["dopp"] + fax[np.argmax(np.abs(F))] / 2.0)
        times.append(t0); phases.append(r["code_phase"])
        metrics.append(r["metric"]); pofs.append(r["peak_over_floor"])
    ph = np.array(phases, float)
    for i in range(1, len(ph)):                              # unwrap mod 1 code period
        while ph[i] - ph[i - 1] > n1 / 2: ph[i] -= n1
        while ph[i] - ph[i - 1] < -n1 / 2: ph[i] += n1
    t = np.array(times)
    slope, icpt = np.polyfit(t, ph, 1)
    # LINEAR Doppler model, not a median. Satellites drift up to ~0.8 Hz/s, so
    # one fixed fd leaves +-45 Hz of residual chirp across a 118 s stream --
    # fatal for the squared-prompt phase reference, and the failure is not
    # graceful: parity collapses to ~1% while acquisition still reports a
    # healthy satellite, so it reads as bad reception. Which satellites die
    # depends on their Doppler RATE, not their strength, which is why a strong
    # bird can fail while a weak one decodes.
    #
    # joint_fix.py has carried this model for a while; track_sv kept the median
    # and nobody noticed the two had diverged. prompt_stream already wipes the
    # chirp when fd/fdot/tref are present -- it just never received them here.
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


def bit_tent(path, fs, tr, t_start=5.0, dur_s=10.0):
    """50 bps bit-edge search: energy of 20-prompt coherent sums vs alignment."""
    n1 = int(round(fs * 1e-3))
    ps = []
    # tr["fd"] is the Doppler AT tr["tref"] (the tracking window's centre), not
    # a constant for the whole capture. This search runs near the START of the
    # file, so evaluate the model here instead of importing ~40 Hz of error
    # from the other end of the window.
    fdot = tr.get("fdot", 0.0)
    tref = tr.get("tref", 0.0)
    for chunk in range(int(dur_s * 10)):
        t0 = t_start + chunk * 0.1
        x = load_seg(path, fs, t0, 0.1)
        ci = int(round(tr["slope"] * t0 + tr["icpt"])) % n1
        code = np.roll(sampled_code(tr["prn"], fs, n1), ci)
        t = np.arange(len(x)) / fs
        fd_here = tr["fd"] + fdot * (t0 - tref)
        xw = x * np.exp(-2j * np.pi * fd_here * t)
        ps.append((xw.reshape(100, n1) * code[None, :]).sum(axis=1))
    p = np.concatenate(ps)
    nb = len(p) // 20 - 1
    e = np.array([np.mean(np.abs(p[off:off + nb * 20].reshape(nb, 20)
                                 .sum(axis=1)) ** 2) for off in range(20)])
    return e / e.max()


# ---------------------------------------------------------------- TLE check
def sky_check(tle_path, when_utc, lat_deg, lon_deg):
    import re
    from datetime import datetime, timezone
    from sgp4.api import Satrec, jday
    lines = [ln for ln in open(tle_path).read().splitlines() if ln.strip()]
    dt = datetime.fromisoformat(when_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    jd0, fr0 = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                    dt.second + dt.microsecond / 1e6)
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    a, f = 6378.137, 1 / 298.257223563
    e2 = f * (2 - f)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    obs = np.array([N * np.cos(lat) * np.cos(lon), N * np.cos(lat) * np.sin(lon),
                    N * (1 - e2) * np.sin(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    def gmst(jd, fr):
        d = jd + fr - 2451545.0
        return np.deg2rad((280.46061837 + 360.98564736629 * d) % 360.0)

    sky = {}
    for i in range(0, len(lines), 3):
        m = re.search(r"PRN (\d+)", lines[i])
        if not m:
            continue
        sat = Satrec.twoline2rv(lines[i + 1], lines[i + 2])
        rr = []
        for ddt in (0.0, 1.0):
            jd, fr = jd0, fr0 + ddt / 86400.0
            e, r, v = sat.sgp4(jd, fr)
            if e:
                break
            th = gmst(jd, fr)
            R = np.array([[np.cos(th), np.sin(th), 0],
                          [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
            d = R @ np.asarray(r) - obs
            rr.append(np.linalg.norm(d))
            if ddt == 0.0:
                el = np.rad2deg(np.arcsin(np.dot(d, up) / rr[0]))
        if len(rr) == 2:
            sky[int(m.group(1))] = dict(el=float(el),
                                        fd=float(-(rr[1] - rr[0]) * 1000.0 / C_LIGHT * FL1))
    return sky


# ----------------------------------------------------------------- selftest
def selftest(fs):
    print("synthetic self-test: PRN7 at -20 dB SNR "
          f"(C/N0 {(-20 + 10*np.log10(fs)):.1f} dB-Hz), Doppler +1830 Hz")
    rng = np.random.default_rng(42)
    n = int(fs * 1e-3) * 40
    t = np.arange(n) / fs
    cr = CODE_RATE * (1 + 1830.0 / FL1)
    sig = ca_code(7)[(np.floor(411.25 + t * cr).astype(np.int64)) % 1023] \
        * np.exp(2j * np.pi * 1830.0 * t)
    x = sig * 10 ** (-20 / 20) + (rng.standard_normal(n)
                                  + 1j * rng.standard_normal(n)) / np.sqrt(2)
    res = acquire(x, fs, list(range(1, 33)), np.arange(-5000, 5001, 250.0), 20)
    best = max(res, key=lambda p: res[p]["metric"])
    r = res[best]
    print(f"  found PRN{best} metric {r['metric']:.2f} at {r['dopp']:+.0f} Hz "
          f"({'PASS' if best == 7 and r['metric'] > THRESH else 'FAIL'})")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", help="CS16 interleaved IQ at L1")
    ap.add_argument("--fs", type=float, default=2.048e6)
    ap.add_argument("--iq2", help="optional second capture (comparison panel)")
    ap.add_argument("--tle", help="celestrak gps-ops TLE file (optional)")
    ap.add_argument("--t0", default=None, help="capture start UTC ISO time")
    ap.add_argument("--lat", type=float, default=0.0,
                help="observer latitude - ONLY needed for the optional --tle "
                     "sky-check; left at 0 so no location ships in the repo")
    ap.add_argument("--lon", type=float, default=0.0,
                help="observer longitude - see --lat")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    print("C/A generator self-check vs IS-GPS-200:")
    generator_selfcheck()
    if a.selftest:
        selftest(a.fs)
        return
    if not a.iq:
        raise SystemExit("--iq required (or --selftest)")

    fs = a.fs
    require_capture(a.iq)
    dur_s = Path(a.iq).stat().st_size / 4 / fs
    dop = np.arange(-7000, 7001, 250.0)

    caps = [("capture", a.iq)] + ([("capture 2", a.iq2)] if a.iq2 else [])
    acq = {}
    for label, path in caps:
        print(f"\n{label}: {Path(path).name}")
        noise_stats(path, fs)
        x = load_seg(path, fs, 0.5, 0.310)
        acq[label] = sky_search(x, fs, 300)
        det = {p: r for p, r in acq[label].items() if r["metric"] > THRESH}
        print(f"  acquisition {ACQ_COH} ms coherent x {300 // ACQ_COH}, +-7 kHz / "
              f"{ACQ_STEP:.0f} Hz, threshold {THRESH}:")
        for p, r in sorted(det.items(), key=lambda kv: -kv[1]["metric"]):
            print(f"    PRN{p:2d}  metric {r['metric']:.2f}  Doppler {r['dopp']:+5.0f} Hz"
                  f"  code phase {r['code_phase']} samp   DETECTED")
        if not det:
            print("    no PRN above threshold "
                  f"(best {max(r['metric'] for r in acq[label].values()):.2f})")

    main_label, main_path = caps[0]
    det = {p: r for p, r in acq[main_label].items() if r["metric"] > THRESH}

    # ---- track every detected SV
    tracks = []
    for p, r in sorted(det.items()):
        tr = track_sv(main_path, fs, p, r["dopp"], dur_s)
        tracks.append(tr)
        print(f"\nPRN{p} track over {dur_s:.0f} s:")
        print(f"  carrier Doppler {tr['fd']:+.1f} Hz (std {tr['fd_std']:.1f})"
              f"   C/N0 ~{tr['cn0']:.1f} dB-Hz")
        print(f"  code-phase drift {tr['slope']:+.3f} samp/s, "
              f"carrier predicts {tr['pred_slope']:+.3f} (-fd/1540): "
              f"ratio {tr['slope']/tr['pred_slope']:.3f}, fit rms {tr['resid_rms']:.2f} samp")
        print(f"  C/A epoch {tr['epoch_ms']:.9f} ms, carrier predicts "
              f"{tr['epoch_pred_ms']:.9f} ms "
              f"(diff {abs(tr['epoch_ms']-tr['epoch_pred_ms'])*1e9:.1f} ps)")

    # ---- 50 bps bit grid on the two strongest
    tents = []
    for tr in sorted(tracks, key=lambda t: -t["cn0"])[:2]:
        e = bit_tent(main_path, fs, tr, t_start=5.0, dur_s=min(10.0, dur_s - 6))
        tents.append((tr["prn"], e))
        print(f"\nPRN{tr['prn']} nav-bit alignment (20 ms blocks over 10 s): "
              f"tent peak at {int(np.argmax(e))} ms, max/min {e.max()/e.min():.2f} "
              f"-- 50 bps bit grid {'FOUND' if e.max()/e.min() > 1.2 else 'not seen'}")

    # ---- TLE cross-check
    if a.tle and a.t0:
        sky = sky_check(a.tle, a.t0, a.lat, a.lon)
        vis = sorted(p for p, s in sky.items() if s["el"] > 0)
        print(f"\nTLE check ({len(vis)} SVs above horizon): {vis}")
        dps = [tr["fd"] - sky[tr["prn"]]["fd"] for tr in tracks if tr["prn"] in sky]
        off = float(np.mean(dps))
        print(f"  common receiver clock offset {off:+.0f} Hz "
              f"({off/FL1*1e6:+.3f} ppm of the LO)")
        for tr in tracks:
            if tr["prn"] in sky:
                s = sky[tr["prn"]]
                print(f"  PRN{tr['prn']:2d} el {s['el']:+5.1f}  meas {tr['fd']:+7.1f} Hz"
                      f"  TLE {s['fd']:+6.0f}  resid {tr['fd']-s['fd']-off:+5.0f} Hz")
        # sub-threshold corroboration: best cell within 150 Hz of prediction?
        print("  sub-threshold PRNs whose best acquisition cell lands on the "
              "TLE-predicted Doppler (chance: ~1 in 19 each):")
        for p, r in sorted(acq[main_label].items()):
            if p in det or p not in sky or sky[p]["el"] < 5:
                continue
            resid = r["dopp"] - sky[p]["fd"] - off
            if abs(resid) <= 150:
                print(f"    PRN{p:2d} el {sky[p]['el']:+5.1f}  cell {r['dopp']:+6.0f} Hz"
                      f"  predicted {sky[p]['fd']+off:+6.0f}  (metric {r['metric']:.2f})")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, len(caps), figsize=(11, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, path) in zip(axes, caps):
        prns = sorted(acq[label])
        met = [acq[label][p]["metric"] for p in prns]
        cols = [COLORS[0] if m > THRESH else "#b0b0b0" for m in met]
        ax.bar(prns, met, color=cols, width=0.8)
        ax.axhline(THRESH, color="#444444", lw=1, ls="--")
        ax.text(32.3, THRESH, f" threshold {THRESH}", va="center", fontsize=8,
                color="#444444")
        for p, m in zip(prns, met):
            if m > THRESH:
                ax.annotate(f"PRN{p}", (p, m), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=9, color="#222222")
        ax.set_title(f"{Path(path).stem}", fontsize=11)
        ax.set_xlabel("PRN")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("acquisition metric\n(peak / 2nd peak)")
    fig.suptitle("GPS L1 C/A acquisition -- 1 ms x 300 noncoherent, 32 PRNs x +-7 kHz",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "gps_acq.png", dpi=140)
    print(f"\nwrote {figdir/'gps_acq.png'}")

    if tracks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
        for k, tr in enumerate(sorted(tracks, key=lambda t: -t["cn0"])):
            c = COLORS[k % len(COLORS)]
            ax1.plot(tr["times"], tr["phases"] - tr["phases"][0], "o", ms=3.5,
                     color=c)
            tt = np.array([tr["times"][0], tr["times"][-1]])
            ax1.plot(tt, tr["pred_slope"] * (tt - tr["times"][0]), "--", lw=1.2,
                     color=c)
            ax1.annotate(f"PRN{tr['prn']}  {tr['fd']:+.0f} Hz",
                         (tr["times"][-1], tr["phases"][-1] - tr["phases"][0]),
                         textcoords="offset points", xytext=(6, 0), fontsize=9,
                         color=c, va="center")
        ax1.set_xlim(right=ax1.get_xlim()[1] + 6)
        ax1.set_xlabel("time into capture (s)")
        ax1.set_ylabel("code-phase drift (samples)")
        ax1.set_title("measured code phase (dots) vs carrier-Doppler\nprediction "
                      "-fd/1540 (dashed)", fontsize=10)
        ax1.grid(alpha=0.25, lw=0.5); ax1.set_axisbelow(True)
        for prn_tent, (p, e) in zip(range(len(tents)), tents):
            ax2.plot(range(20), e, "-o", ms=4, lw=1.5, color=COLORS[prn_tent])
            side = -1 if prn_tent == 0 else 1
            ax2.annotate(f"PRN{p}", (int(np.argmax(e)), e.max()),
                         textcoords="offset points", xytext=(8 * side, 6),
                         ha="right" if side < 0 else "left",
                         fontsize=9, color=COLORS[prn_tent])
        ax2.set_xlabel("alignment offset (ms mod 20)")
        ax2.set_ylabel("20 ms coherent-sum energy (norm.)")
        ax2.set_title("navigation-data bit grid: energy peaks when\n20 ms sums "
                      "align with the 50 bps bits", fontsize=10)
        ax2.set_xticks(range(0, 20, 2))
        ax2.grid(alpha=0.25, lw=0.5); ax2.set_axisbelow(True)
        for ax in (ax1, ax2):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(figdir / "gps_track.png", dpi=140)
        print(f"wrote {figdir/'gps_track.png'}")


if __name__ == "__main__":
    main()
