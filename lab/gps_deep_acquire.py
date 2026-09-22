"""Deep GPS acquisition: 5 ms coherent x 100 noncoherent, vs the 1 ms baseline.

THE QUESTION: the standard search is 1 ms coherent x
300 noncoherent. Coherent integration gains SNR linearly where noncoherent
summing pays a squaring loss, so 5 ms coherent should dig several dB below
the baseline's floor. Does it — and does it change the answer on today's
zero-bird captures?

epistemic:
  type: hypothesis
  tier: measured
  prediction: |
    P1 known-good attic capture: deep finds the same 7 birds with LARGER
       metrics (it cannot lose birds the baseline finds).
    P2 sensitivity ladder (known-good + calibrated synthetic noise): deep
       keeps birds >= 3 dB past the level where the baseline loses them.
    P3 today's dead captures: if the antenna truly cannot see sky, deep
       still finds 0 — a stronger elimination than the baseline's 0. If it
       finds 1-4 weak birds, the antenna is marginal-not-blind and the
       "needs hands" conclusion softens to "needs a better window".
  test: this file, offline, no radio
  result: |
    RAN 2026-07-31 ~20:30 (gps_deep_results.txt).
    P1 CONFIRMED+: all 7 baseline birds kept, metrics 2.11-4.60x larger,
       plus PRN 2 found deep-only (1.45 -> 5.16). 8 birds total.
    P2 CONFIRMED, measured: baseline collapses 7->2 birds at +6 dB added
       noise; deep still holds 7 there and 2 birds even at +15 dB. The
       sensitivity gain is ~+6 dB (deep at +12 dB == baseline at +6 dB),
       matching 10*log10(5) minus bit-edge/squaring losses.
    P3 NEGATIVE, decisive: BOTH dead captures give 0 birds at the deeper
       threshold too.
  conclusion: |
    The 7/27 attic capture was findable at baseline sensitivity; today's
    captures have nothing even 6 dB further down. This is not a marginal
    sky view — effectively NO GPS energy reaches the port. Blind or
    disconnected antenna element; software cannot reach further.
  next: |
    (1) physical: move the antenna to sky view, re-run gps_replicate.py;
    (2) ship deep_acquire as locate.py's fallback when the fast search
        finds <4 birds — it would have rescued marginal captures like a
        one-bar attic, and costs ~60 s;
    (3) the +6 dB also re-prices old failures: any past "0 birds indoors"
        verdict was measured at baseline sensitivity and is reopenable.

MECHANICS. Per Doppler bin: mix the whole segment once with a CONTINUOUS
carrier (absolute time across blocks, so inter-block phase stays coherent),
FFT all 1-ms blocks in one batch, multiply by the conjugate code spectrum,
IFFT in one batch, then sum K=5 adjacent complex rows BEFORE the magnitude —
that sum is the coherent gain — and only then square and accumulate the
noncoherent groups. Doppler bins tighten to 100 Hz because a residual offset
rotates the phase across a 5 ms window (1/(2*0.005) = 100 Hz).

NAV BIT EDGES, honestly: bits flip every 20 ms, so ~25% of 5 ms windows
straddle an edge and partially self-cancel. With 100 noncoherent groups the
other 75% carry the peak; the loss is real (<1 dB) and accepted rather than
hidden behind a 2x runtime for edge hypotheses. 10 ms coherent would need
them; 5 ms is the honest sweet spot for a first measurement.

No radio, no coordinates. Results print as metrics only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure import CODE_RATE, acquire, ca_code, generator_selfcheck, load_seg, sampled_code  # noqa: E402

FS = 2_048_000
K_COH = 5                    # ms per coherent sum
N_GROUPS = 100               # noncoherent groups (0.5 s of data)
DOPPLERS_DEEP = np.arange(-7000.0, 7001.0, 100.0)
BASE_NONCOH = 300            # the baseline exactly as locate.py runs it
DOPPLERS_BASE = np.arange(-7000.0, 7001.0, 250.0)
STRONG = 3.5                 # same threshold locate.py calls a bird


def deep_acquire(x, fs, prns, dopplers=DOPPLERS_DEEP,
                 k_coh=K_COH, n_groups=N_GROUPS):
    n1 = int(round(fs * 1e-3))
    n_blocks = k_coh * n_groups
    x = x[: n1 * n_blocks]
    if len(x) < n1 * n_blocks:
        raise SystemExit(f"capture too short: need {n1*n_blocks} samples")
    t = (np.arange(len(x)) / fs).astype(np.float64)   # ABSOLUTE time: phase
    blocks_shape = (n_blocks, n1)                     # continuity across blocks
    code_f = {p: np.conj(np.fft.fft(sampled_code(p, fs, n1))).astype(np.complex64)
              for p in prns}
    excl = int(round(fs / CODE_RATE))
    best = {p: (0.0, 0.0, 0, None) for p in prns}     # power, dopp, phase, row
    acc = {p: None for p in prns}
    # per-PRN best doppler row is kept for the metric; full maps would be
    # 141 x 2048 x 32 floats and we only ever read the argmax row.
    maps_peak = {p: (-1.0, -1, None) for p in prns}
    for fd in dopplers:
        mixed = (x * np.exp(-2j * np.pi * fd * t)).astype(np.complex64)
        BF = np.fft.fft(mixed.reshape(blocks_shape), axis=1)
        for p in prns:
            corr = np.fft.ifft(BF * code_f[p][None, :], axis=1)
            coh = corr.reshape(n_groups, k_coh, n1).sum(axis=1)   # THE gain
            row = (np.abs(coh) ** 2).sum(axis=0)
            pk = float(row.max())
            if pk > maps_peak[p][0]:
                maps_peak[p] = (pk, int(row.argmax()), row.copy(), float(fd))
    out = {}
    for p in prns:
        pk, ci, row, fd = maps_peak[p]
        mask = np.ones(len(row), dtype=bool)
        for off in range(-excl, excl + 1):
            mask[(ci + off) % len(row)] = False
        out[p] = dict(metric=float(pk / row[mask].max()),
                      dopp=fd, code_phase=ci,
                      peak_over_floor=float(pk / np.median(row)))
    return out


def birds(res, thresh=STRONG):
    return {p: r for p, r in res.items() if r["metric"] > thresh}


def compare(path, label, prns=range(1, 33)):
    print(f"\n===== {label} =====")
    x = load_seg(path, FS, 0.5, 0.6)
    t0 = time.time()
    base = acquire(x[: int(FS * 0.3)], FS, list(prns), DOPPLERS_BASE, BASE_NONCOH)
    tb = time.time() - t0
    t0 = time.time()
    deep = deep_acquire(load_seg(path, FS, 0.5, 0.6), FS, list(prns))
    td = time.time() - t0
    b_base, b_deep = birds(base), birds(deep)
    print(f"  baseline (1 ms x {BASE_NONCOH}): {len(b_base)} birds  [{tb:.0f} s]")
    print(f"  deep   ({K_COH} ms x {N_GROUPS}): {len(b_deep)} birds  [{td:.0f} s]")
    both = sorted(set(b_base) | set(b_deep))
    if both:
        print(f"  {'PRN':>5} {'base':>7} {'deep':>7}   gain")
        for p in both:
            mb, md = base[p]["metric"], deep[p]["metric"]
            print(f"  {p:>5} {mb:>7.2f} {md:>7.2f}   {md/mb:>5.2f}x"
                  f"{'   deep-only' if p not in b_base else ''}"
                  f"{'   LOST BY DEEP (P1 violated!)' if p not in b_deep else ''}")
    return base, deep


def sensitivity_ladder(path):
    """Bury the known-good capture in calibrated noise until each search
    loses the birds. The gap between the two break points IS the dB gain —
    measured, not derived from integration-time arithmetic."""
    print("\n===== sensitivity ladder (known-good + synthetic noise) =====")
    rng = np.random.default_rng(7)
    x0 = load_seg(path, FS, 0.5, 0.6)
    p0 = float(np.mean(np.abs(x0) ** 2))
    print(f"  {'added':>7} {'base birds':>11} {'deep birds':>11}")
    for extra_db in (0, 3, 6, 9, 12, 15):
        if extra_db == 0:
            x = x0
        else:
            npow = p0 * (10 ** (extra_db / 10) - 1)
            noise = (rng.standard_normal(len(x0)) + 1j * rng.standard_normal(len(x0)))
            x = x0 + (np.sqrt(npow / 2) * noise).astype(np.complex64)
        nb = len(birds(acquire(x[: int(FS * 0.3)], FS, list(range(1, 33)),
                               DOPPLERS_BASE, BASE_NONCOH)))
        nd = len(birds(deep_acquire(x, FS, list(range(1, 33)))))
        print(f"  +{extra_db:>4} dB {nb:>11} {nd:>11}")
        if nb == 0 and nd == 0:
            break


if __name__ == "__main__":
    print("C/A generator self-check (published octal values):")
    generator_selfcheck()

    KNOWN_GOOD = r"Z:\src\grid-atlas\captures\gps_attic.cs16"
    DEAD_ON = r"Z:\src\GPSTuna\lab_local\sky_biast_on.cs16"
    DEAD_REPL = r"Z:\src\GPSTuna\lab_local\sky_capture.cs16"

    compare(KNOWN_GOOD, "P1: known-good 7/27 attic capture")
    sensitivity_ladder(KNOWN_GOOD)
    compare(DEAD_ON, "P3a: today, bias-T verified ON (A/B capture)")
    compare(DEAD_REPL, "P3b: today, replicated 7/27 front-end config")
    print("\nDone. Interpret against the pre-registered predictions in the"
          " docstring — they were written before any of this ran.")
