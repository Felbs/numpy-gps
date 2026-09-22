"""Re-run GPS acquisition through the EXACT code path that worked on 7/27.

The A/B established that the LNA is real and powered (rms 386 vs 105 with the
bias-T verified either way) and that acquisition still returns zero satellites.
So LNA power is eliminated, properly, for the first time.

The spectrum then said what the level never could. Comparing all three captures
through identical analysis code:

    capture              rms    peak-median    spur bins
    KNOWN-GOOD 7/27    143.6       12.7 dB            0
    today bias-T ON    385.6       45.9 dB           56
    today bias-T OFF   105.2       29.9 dB           10

and the spur is at EXACTLY 0.0 kHz offset, symmetric, about 1 kHz wide -- the
signature of DC offset / LO leakage sitting on top of the tune frequency, not
an external interferer, which would not land dead on DC.

WHY MINE AND NOT THEIRS. The 7/27 capture went through cw._open_sdr:
IFGR 30, rfgain_sel 0, native CS16. My A/B script opened the device by hand
with IFGR 40, rfgain_sel 4, CF32. Rather than theorise about which of those
matters, replicate the working path exactly -- locate.py calls cw._open_sdr,
so running it live IS the 7/27 path, now with a bias-T that actually latches.

Hypothesis: matching the known-good front-end configuration collapses the DC
spike and satellites return. If it does not, the cause is upstream of the SDR
(antenna placement or the antenna itself), and the next move is physical.

Takes the warden lock so the observatory yields cleanly instead of colliding.
Any fix lands in lab_local/ only.
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "gps_replicate_log.txt"


def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


sys.path.insert(0, "Z:/src/gr-radiotuna/tools")
import radio_lock

log("=" * 66)
log("GPS replicate: live capture through the proven 7/27 path")
log("  cw._open_sdr defaults -> IFGR 30, rfgain_sel 0, CS16, 2.048 MHz")
log("  bias-T now writes 'true'/'false' and is restored in a finally")

holder = radio_lock.Holder("gps_replicate",
                           "GPS live capture via known-good path", 60,
                           wait_s=900)
holder.__enter__()
try:
    sys.argv = ["locate.py", "--secs", "120", "--antenna", "Antenna B"]
    sys.path.insert(0, str(HERE.parent))
    log("capturing 120 s on Antenna B ...")
    runpy.run_path(str(HERE.parent / "locate.py"), run_name="__main__")
except SystemExit:
    pass
except Exception as exc:
    log(f"FAILED: {type(exc).__name__}: {exc}")
finally:
    holder.__exit__(None, None, None)
    log("lock released")
