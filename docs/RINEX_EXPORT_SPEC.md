# RINEX export — design spec

**Goal:** turn a GPSTuna capture into the two files a free PPP service eats
(`CSRS-PPP`, `APPS`, `RTKLIB`), so a static capture becomes a **decimetre or
better** position using IGS precise orbits and clocks — instead of the
~29 m broadcast-ephemeris fix `locate.py` produces today.

Nothing here needs new hardware. Everything the format wants, the receiver
already computes; it is currently thrown away after the least-squares.

---

## 1. Why this is worth building

| | broadcast LS (today) | PPP (this spec) |
|---|---|---|
| orbits / clocks | broadcast ephemeris, ~1–2 m | IGS final, ~2 cm |
| ionosphere | Klobuchar model (measured 9/10: worth **1.3 m**) | eliminated by L1/L5, or estimated |
| observable | code, ~300 m chip, smoothed | **carrier phase, 19 cm** |
| troposphere | fixed 2.4 m zenith mapping | estimated per epoch |
| result at Smith Point | 28.8 m scatter, sitting in a creek | decimetre, on land |

The single biggest gain is **carrier phase**, and `hatch.py` proves we already
have it: Hatch smoothing is *defined* on continuous accumulated carrier phase
in cycles, and the module docstring describes maintaining exactly that
("one phase-continuous 1 ms prompt stream across the whole clean section").
Writing it to RINEX is plumbing, not new DSP.

---

## 2. What we already have, and what is missing

Sourced from `joint_fix.assemble()`, `fix.py`, `hatch.py`, `gal_inav.py`.

| RINEX needs | GPSTuna has | Where |
|---|---|---|
| pseudorange C1C | ✅ `t_rx - t_sys`, × c | `joint_fix.assemble()` → `t_sys` |
| carrier phase L1C (cycles) | ✅ continuous, per-epoch | `hatch.py` phase model |
| Doppler D1C (Hz) | ✅ tracked | `measure.py` `carrier_doppler` |
| signal strength S1C | ✅ C/N0 dB-Hz | tracking `cn0`, `cn0_mean` |
| GPS ephemeris (LNAV) | ✅ all 16 orbital + 3 clock terms | `fix.py` `EPH_NEED`, `eph.py` |
| Galileo ephemeris (I/NAV) | ✅ incl. `BGD_E1E5a/E5b` | `gal_eph.py`, `gal_inav.py` |
| **GPS week number** | ✅ **decoded** — `WN` is in the eph record | `fix.py:585` `VALIDATE_EPH` carries `"WN": 384` |
| **absolute UTC epoch** | ⚠️ derive from WN+TOW | see §4.1 |
| **cycle-slip flags (LLI)** | ⚠️ must be produced | see §4.3 |
| approximate position | ✅ the LS fix | `fix_result*.json` |

**One real gap:** loss-of-lock flagging. The week number question is settled —
`WN` is already a field of the decoded ephemeris (the validation vector at
`fix.py:585` pins `WN: 384`), so §4.1's rollover check has something to check.
Everything else is a formatter over data the receiver already holds.

---

## 3. Deliverable

`rinex.py`, one module, two entry points:

```
python rinex.py --iq lab_local/sky_capture_...cs16 --out lab_local/rinex/
  ->  SMIT2510.26O      RINEX 3.05 observation
      SMIT2510.26N      RINEX 3.05 navigation (mixed GPS+GAL)
```

Filenames: `SSSSDDDF.YYt` — 4-char marker, day-of-year, session, 2-digit year,
type. Long RINEX 3 names are also accepted by every service and are less
ambiguous; either is fine, pick one and be consistent.

---

## 4. The three things that will be wrong on the first try

### 4.1 Absolute time — get this right or PPP silently fails

RINEX epochs are **calendar UTC**, not `t_rx_s = 30.57`. The chain is:

```
GPS week (WN, subframe 1) + TOW (subframe HOW, 6 s resolution)
  -> GPS seconds since 1980-01-06
  -> minus current leap seconds (18 as of 2026)   -> UTC
```

Three traps:
- **WN is 10-bit and rolls over every 1024 weeks** (last roll 2019-04-06,
  next 2038-11). Resolve it against the capture file's wall-clock date, and
  **assert** the result is within ±1 week of the file mtime — a rollover error
  puts you 19.6 years out and PPP will simply reject the file.
- **Leap seconds** go in the header (`LEAP SECONDS`) and are *not* applied to
  the observation timestamps: RINEX GPS-system time is not UTC-adjusted. Write
  epochs in **GPS time** and declare `GPS` in `TIME OF FIRST OBS`.
- Our epoch times carry the **receiver clock offset**. Either steer epochs to
  integer GPS seconds and apply the offset to the pseudoranges, or write the
  offset in the optional clock-offset field. **Do the former** — services
  handle it better.

### 4.2 Epoch rate — 12 epochs will not converge

Today `locate.py` produces **15 attempted / 12 kept epochs over 300 s** — that
is the *fix* rate, not the *data* rate. PPP wants a continuous series: **1 Hz
(or 30 s) for the whole capture**, ideally hours.

The tracker already runs at 1 ms prompts continuously, so 1 Hz observables are
free — the sparsity is an artefact of how the fix loop samples. **The exporter
must tap the tracking loop, not the fix loop.** A 300 s capture then yields 300
epochs instead of 12, which is the difference between PPP converging and not.

> Practical note: PPP convergence for a static float solution is ~20–40 min of
> continuous data. **A 5-minute capture will not reach decimetre no matter how
> good the exporter is.** Plan a 1–2 hour static capture; at 2.46 GB per 300 s
> that is ~30–60 GB, so capture to the USB drive, and consider decimating.

### 4.3 Cycle slips — the flag that protects the whole solution

A carrier-phase break that is not flagged corrupts every epoch after it.
Set the **LLI bit (1)** on the phase observation whenever:
- the tracking loop reported loss of lock or re-acquisition, **or**
- code-minus-carrier jumps — `hatch.py` already computes CMC and its docstring
  names "CMC jumps" explicitly. Reuse that detector; do not write a second one.

Conservative flagging is free; a missed slip is not.

---

## 5. File formats — exact

### 5.1 Observation header (RINEX 3.05)

```
     3.05           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE
GPSTuna 0.x         Felbs               20260910 051944 UTC PGM / RUN BY / DATE
SMIT                                                        MARKER NAME
GEODETIC                                                    MARKER TYPE
Felbs               -                                       OBSERVER / AGENCY
1                   SDRplay RSPdx       GPSTuna             REC # / TYPE / VERS
1                   GPS patch                               ANT # / TYPE
  1122334.4455 -4966554.3210  3885221.9876                  APPROX POSITION XYZ
        0.0000        0.0000        0.0000                  ANTENNA: DELTA H/E/N
G    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES
E    4 C1B L1B D1B S1B                                      SYS / # / OBS TYPES
     1.000                                                  INTERVAL
    18                                                      LEAP SECONDS
  2026     9     8     0    36   19.0000000     GPS         TIME OF FIRST OBS
                                                            END OF HEADER
```

Header labels sit in **columns 61–80**, exactly. Get this wrong and every
parser rejects the file with an unhelpful message — build the header from a
`(value, label)` list and pad programmatically, never by hand.

`APPROX POSITION XYZ` may be tens of metres off; that is its purpose.

### 5.2 Observation records

```
> 2026 09 08 00 36 19.0000000  0  5
G07  22345678.901   117456789.01206     -1234.567          45.100
G08  23901234.567   125612345.67806     +2345.678          48.300
E30  24567890.123   129034567.89006      -456.789          41.700
```

- `>` then epoch, flag `0` (OK), satellite count.
- Per satellite: **3-char SVID** (`G07`, `E30` — zero-padded), then each
  observation as **F14.3**, each optionally followed by **LLI (I1)** and
  **SSI (I1)**.
- Missing observation = **blank field**, never `0.0`.
- Phase in **cycles**, not metres. Pseudorange and Doppler in m and Hz.
- SSI is 1–9 mapped from C/N0 (`min(9, max(1, int(cn0/6)))`).

### 5.3 Navigation records

GPS LNAV: 8 lines per ephemeris, 4 × `D19.12` per line, in the canonical
A/331-independent RINEX order (`toc`, `af0..af2`, then `IODE, Crs, Δn, M0` …).
Galileo I/NAV: same shape, `E` prefix, data-source word and `BGD` in place of
`TGD`. Emit one record per distinct `(PRN, toe)` — `fix.py`'s pool already
dedupes by `toe`, so reuse `pool_get`/`pool_put` rather than re-deriving.

---

## 6. Validation — before uploading anything

The lab rule applies: a formatter that produces a plausible-looking file is
the dangerous case. Four gates, in order:

1. **`gfzrnx -finp X.O -check`** or **`teqc +qc`** — a third-party parser must
   accept the file. Never let our own reader be the only judge.
2. **RTKLIB `rnx2rtkp` in single-point mode** must reproduce our own LS fix to
   **< 1 m**. This proves the pseudoranges and time tags round-trip. If this
   disagrees, the exporter is wrong — not RTKLIB.
3. **Carrier-phase sanity:** L1C × 0.1903 m/cycle minus C1C should be smooth
   and slowly varying (the CMC curve). Any step that is not LLI-flagged is a
   missed cycle slip.
4. **Only then** upload to CSRS-PPP. Compare against the broadcast LS fix:
   expect the PPP position to move **metres, not hundreds of metres**. A wild
   disagreement means time tags, not geophysics.

Add these as `rinex.py --selftest` with a synthetic epoch set, in the style of
`m6_bicm.selftest` — controls that must FAIL included (a planted cycle slip
with no LLI must be caught by gate 3).

---

## 7. Build order

1. **NAV file first.** It is static, self-contained, and gate 1 validates it
   alone. Also flushes out the week-number question immediately.
2. **OBS with code only** (C1C, D1C, S1C) at 1 Hz from the tracking loop.
   Gate 2 (RTKLIB agreement) is then meaningful.
3. **Add carrier phase** L1C + LLI. Gate 3.
4. **Add Galileo** (E1 → C1B/L1B). We already decode I/NAV; the joint fix
   proved 26.8 m with 6 GPS + 3 Galileo.
5. **Then** the long static capture, and PPP.

Steps 1–3 are the whole value; 4 and 5 are gravy.

---

## 8. What this does *not* fix

PPP will not rescue the **7.3° satellite** or the **multipath off the water**
at Smith Point — a low-elevation bird over a reflector is biased no matter
whose orbits you use. PPP lets you *see* it: per-satellite residuals make the
offender obvious, and an elevation mask (10–15°) is then a defensible choice
rather than a guess. See `smithpoint_gps_resolve_0910` in the memory wing.
