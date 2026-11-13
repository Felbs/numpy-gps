#!/usr/bin/env python3
"""locate.py - one command: capture the sky, count satellites, and (if enough
are visible) compute your position - written ONLY to lab_local/ (gitignored).

Run this with the antenna at a window or outdoors, or on a GPS patch antenna.
Indoors you'll see 1-3 weak satellites (not enough); with sky view, 6-12.

  python locate.py            # 180 s capture on Antenna B, then locate (--secs 300 for a sure first fix)
  python locate.py --iq f.cs16   # locate from an existing capture

A 3D fix needs >= 4 satellites strong enough to decode their orbit (C/N0 ~ 36+;
the sky search finds them a few dB below that). This prints the count first so you know immediately whether the sky
view is good; if it is, the full pseudorange solve runs and the result stays
private in lab_local/fix_result.json.
"""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# A 300 s capture followed by a long decode looks HUNG when stdout is a file
# rather than a terminal, because Python block-buffers it and nothing appears
# until the process exits. Anyone redirecting this to a log -- which is the
# natural thing to do with a run this long -- deserves to see it working.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:                                                # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import acquire, check_sidecar, load_seg, require_capture
from fix import full_fix

FS = 2.048e6
LOCAL = HERE / "lab_local"


def _open_sdr(antenna, fs):
    """Open the first SoapySDR device at `fs`, return (device, active stream).

    Self-contained on purpose: this used to import a helper from a sibling
    project by absolute path, which meant `locate.py` could only ever run on
    one machine.
    """
    try:
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    except ImportError:
        raise SystemExit(
            "\nCapturing from a radio needs SoapySDR, which is not importable "
            "from this Python.\n\n"
            "  * install SoapySDR plus a driver for your radio, or\n"
            "  * record 1575.42 MHz yourself with any SDR tool and pass the "
            "file:\n"
            "        python locate.py --iq your_capture.cs16\n\n"
            "See the README (\"Getting a capture\") for the recording "
            "settings.\n")

    # SoapySDR logs an ERROR for every driver module whose library is absent --
    # airspy, bladeRF, HackRF, Pluto, rtlsdr, uhd. None of them are ours and
    # none of them matter, but a first run prints ten red ERROR lines before
    # anything works, which reads exactly like a failure.
    try:
        SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    except Exception:                                            # noqa: BLE001
        pass

    devices = SoapySDR.Device.enumerate()
    if not devices:
        hint = ""
        if sys.platform.startswith("win"):
            # Measured 8/13: the RSPdx was plugged in and healthy and
            # enumerate() still returned nothing, because the SDRplay API
            # SERVICE had stopped. "No radio" sends the reader to the cable;
            # the cable was fine. Name the usual cause instead.
            hint = ("\nOn Windows the SDRplay API runs as a SERVICE, which can "
                    "be stopped\neven while the radio is plugged in and "
                    "healthy. Check it with:\n"
                    "        Start-Service SDRplayAPIService\n"
                    "and try again.\n")
        raise SystemExit(
            "\nSoapySDR is installed but found no radio.\n"
            "Check that the device is plugged in and not held open by another "
            "program.\n" + hint)
    sdr = SoapySDR.Device(devices[0])
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)
    if antenna:
        # Read it BACK. A silently-swallowed setAntenna means capturing on the
        # wrong port and blaming the sky.
        try:
            sdr.setAntenna(SOAPY_SDR_RX, 0, antenna)
        except Exception as e:
            print(f"[locate] could not select antenna {antenna!r}: {e}")
        got = sdr.getAntenna(SOAPY_SDR_RX, 0)
        if got != antenna:
            ports = list(sdr.listAntennas(SOAPY_SDR_RX, 0))
            print(f"[locate] WARNING: asked for antenna {antenna!r} but the "
                  f"radio reports {got!r}. Available: {ports}")
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(st)
    return sdr, st


def _grab_to_file(sdr, st, secs, fs, path, max_stall_s=60, stop=None):
    """Stream `secs` of baseband to `path` as interleaved int16.

    Written to disk as it arrives rather than accumulated in memory. The
    obvious version buffers the whole capture as complex64 first, which is
    8 bytes per sample: a 600 s capture at 2.048 Msps is 9.8 GB, fine on a
    workstation and fatal on a 8 GB Raspberry Pi, where it drives the machine
    into swap and then the OOM killer. Streaming makes the memory cost the
    chunk size instead of the capture length, so the only limit left is disk
    -- and the file was going to be written anyway, at half the size, because
    int16 is 4 bytes per sample.
    """
    import queue
    import threading
    try:
        from SoapySDR import SOAPY_SDR_OVERFLOW
    except Exception:                                            # noqa: BLE001
        SOAPY_SDR_OVERFLOW = -4

    want = int(secs * fs)
    chunk = np.empty(65536, np.complex64)

    # The conversion and the write happen on a SEPARATE thread. Doing them in
    # the read loop -- which the first version did -- means the radio is not
    # being serviced while the SD card is written, its ring buffer overruns,
    # and samples are lost. Acquisition survives that (correlating 1023 chips
    # buys 30 dB and does not care about a hole); the 50 bps navigation decode
    # does not, so the symptom is "plenty of satellites, no ephemeris" rather
    # than an obvious failure. On a Pi it cost every fix we tried.
    q: queue.Queue = queue.Queue(maxsize=64)
    err = []

    def _writer():
        try:
            with open(path, "wb") as fh:
                while True:
                    buf = q.get()
                    if buf is None:
                        return
                    inter = np.empty(2 * len(buf), np.int16)
                    inter[0::2] = np.clip(buf.real * 32767, -32768,
                                          32767).astype(np.int16)
                    inter[1::2] = np.clip(buf.imag * 32767, -32768,
                                          32767).astype(np.int16)
                    inter.tofile(fh)
        except Exception as e:                                   # noqa: BLE001
            err.append(e)

    th = threading.Thread(target=_writer, daemon=True)
    th.start()

    got = overflows = 0
    last_progress = time.time()
    next_note = 60.0
    t0 = time.time()
    try:
        while got < want and (stop is None or stop["sig"] is None):
            sr = sdr.readStream(st, [chunk], len(chunk), timeoutUs=1000000)
            n = getattr(sr, "ret", sr if isinstance(sr, int) else -1)
            if n > 0:
                n = min(n, want - got)
                q.put(chunk[:n].copy())      # copy: the buffer is reused
                got += n
                last_progress = time.time()
                el = time.time() - t0
                if el >= next_note:
                    print(f"[locate] captured {got/fs:6.0f} s of {secs:.0f} s "
                          f"({got*4/1e9:.2f} GB){'' if not overflows else f'  OVERFLOWS {overflows}'}",
                          flush=True)
                    next_note += 60.0
            elif n == SOAPY_SDR_OVERFLOW:
                # COUNT it. Swallowing this silently is how a capture full of
                # holes reaches the decoder looking like a clean one.
                overflows += 1
                last_progress = time.time()
            elif time.time() - last_progress > max_stall_s:
                raise SystemExit(
                    f"\nThe radio stopped delivering samples after "
                    f"{got/fs:.1f} s ({got}/{want}).\nThis is usually USB: try "
                    f"a short, direct USB 3.0 cable and no hub.\n")
            if err:
                raise SystemExit(f"\nWriting the capture failed: {err[0]}\n")
    finally:
        q.put(None)
        th.join(timeout=30)
    if err:
        raise SystemExit(f"\nWriting the capture failed: {err[0]}\n")
    if overflows:
        print(f"[locate] WARNING: {overflows} SDR overflow(s) during the "
              f"capture.\n"
              f"         Samples were dropped. Acquisition will still find "
              f"satellites, but the\n"
              f"         navigation decode may fail to complete any orbit -- "
              f"which looks like a\n"
              f"         weak-sky problem and is not one. Use faster storage, "
              f"or a shorter capture.", flush=True)
    return got


def capture(secs=180, antenna="Antenna A"):
    # Check the disk BEFORE touching the radio. A 300 s capture is 2.46 GB;
    # on a drive the card fills a stop at a time, and the failure you get
    # otherwise is a short write partway through a capture you parked and
    # waited five minutes for.
    import shutil
    LOCAL.mkdir(exist_ok=True)
    need = int(secs * FS) * 4
    free = shutil.disk_usage(LOCAL).free
    margin = 1_000_000_000            # leave the OS a GB to breathe
    if free < need + margin:
        fits = max(0.0, (free - margin) / (FS * 4))
        raise SystemExit(
            f"\nNot enough disk for a {secs:.0f} s capture: need "
            f"{need/1e9:.2f} GB + 1 GB margin, have {free/1e9:.2f} GB free.\n"
            f"About {fits:.0f} s would fit. Delete old lab_local/sky_capture_*"
            f".cs16 files\n(each fix records which capture made it) and run "
            f"again.")
    # _open_sdr carries the friendly ImportError, so open FIRST and only then
    # import the constants -- otherwise this line raises the raw traceback.
    sdr, st = _open_sdr(antenna, FS)
    from SoapySDR import SOAPY_SDR_RX
    sdr.setFrequency(SOAPY_SDR_RX, 0, 1575.42e6)
    # Power an ACTIVE GPS antenna. These are boolean settings: SoapySDRPlay3
    # reads only the exact string "false" as off and treats everything else --
    # including "0" -- as ON, so the literals here are load-bearing.
    powered = False
    for key in ("biasT_ctrl", "biasT", "bias_tee"):
        try:
            sdr.writeSetting(key, "true")
            # An unverified writeSetting is a HOPE. On this driver only
            # biasT_ctrl exists; the other two accept the write and read back
            # empty, which would look like success and quietly leave an active
            # antenna's LNA unpowered.
            if str(sdr.readSetting(key)).lower() == "true":
                powered = True
                print(f"[locate] bias-T ON via {key} (readback confirmed)")
        except Exception:
            pass
    if not powered:
        print("[locate] WARNING: could not confirm bias-T power. An ACTIVE GPS "
              "patch antenna needs it;\n"
              "         without it expect to acquire a few birds but fail to "
              "decode their orbits.")
    # Take the signal handlers BACK. Opening the device hands SIGINT to the
    # SDRplay API library, which swallows it -- measured 8/23: ctrl-C and the
    # dashboard's abort were simply ignored mid-capture. Re-registering after
    # the open wins, and a signal becomes a clean stop through the finally
    # block below (bias-T off, stream closed) instead of a kill that leaves
    # DC on the coax.
    stop = {"sig": None}
    prev = {s: signal.signal(s, lambda sig, _f: stop.update(sig=sig))
            for s in (signal.SIGINT, signal.SIGTERM)}
    LOCAL.mkdir(exist_ok=True)
    # One file PER capture, stamped in UTC. The fixed name meant every stop of
    # a drive overwrote the previous stop's capture -- come home from five
    # stops with one file. The stamp stays in gitignored lab_local, where the
    # capture (which already encodes its own time) was going anyway.
    stamp = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
    fn = LOCAL / f"sky_capture_{stamp}.cs16"
    try:
        time.sleep(0.5)
        got = _grab_to_file(sdr, st, secs, FS, fn, max_stall_s=60, stop=stop)
    finally:
        # Hand the coax back de-powered even if the grab throws. Otherwise the
        # next thing plugged into this port -- a passive antenna, a filter, a
        # borrowed radio -- meets DC that nothing turned off.
        for key in ("biasT_ctrl", "biasT", "bias_tee"):
            try:
                sdr.writeSetting(key, "false")
            except Exception:
                pass
        sdr.deactivateStream(st)
        sdr.closeStream(st)
        for s, h in prev.items():
            signal.signal(s, h)
    if stop["sig"] is not None:
        # A partial capture is junk that reads as a real stop's data; on a
        # drive it also eats 8 MB/s of card. Delete it and say so.
        fn.unlink(missing_ok=True)
        print(f"[locate] capture ABORTED by signal {stop['sig']} after "
              f"{got/FS:.0f} s -- partial file deleted, bias-T off",
              flush=True)
        raise SystemExit(130)
    print(f"[locate] captured {got/FS:.0f} s -> {fn} "
          f"({fn.stat().st_size/1e9:.2f} GB)", flush=True)
    left = shutil.disk_usage(LOCAL).free
    print(f"[locate] disk: {left/1e9:.1f} GB free -- about "
          f"{int(max(0, left - 1_000_000_000) // (secs * FS * 4))} more "
          f"capture(s) this length", flush=True)
    return str(fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", help="existing capture; else capture live")
    ap.add_argument("--secs", type=float, default=180)
    ap.add_argument("--antenna", default="Antenna B",
                    help="SDR antenna port (the one with bias-T if your GPS patch is active)")
    a = ap.parse_args()
    path = a.iq or capture(a.secs, a.antenna)
    require_capture(path)
    check_sidecar(path, FS)
    dur = Path(path).stat().st_size / 4 / FS
    x = load_seg(path, FS, 0.5, 0.400)
    from measure import sky_search, THRESH
    acq = sky_search(x, FS, 400)
    # Every detected satellite is tried: one too weak to decode its orbit
    # ("no parity-clean subframe anchor") is simply left out of the solve.
    # GPSTUNA_MIN_METRIC raises the bar again (it was 3.5 on the 1 ms search,
    # which kept a 36 dB-Hz satellite that decodes completely out of the fix).
    thr = float(os.environ.get("GPSTUNA_MIN_METRIC", str(THRESH)))
    strong = {p: r for p, r in acq.items() if r["metric"] > thr}
    weak = {p: r for p, r in acq.items() if THRESH < r["metric"] <= thr}
    print(f"SKY VIEW: {len(strong)} strong + {len(weak)} weak satellites")
    for p, r in sorted(strong.items(), key=lambda kv: -kv[1]["metric"]):
        print(f"  PRN{p:2d}  metric {r['metric']:.1f}  STRONG")
    if len(strong) < 4:
        print(f"\nNeed >= 4 strong satellites for a fix; have {len(strong)}.")
        print("Move the antenna to a window / outside (or use a GPS patch antenna)")
        print("and run again - sky view is the only thing missing.")
        return
    print("\n>= 4 strong satellites - decoding orbits + solving position...")
    # full pipeline: nav decode with timing anchors -> common-epoch snapshot
    # code phases -> SV-time pseudorange assembly -> least-squares solve.
    # The fix is written ONLY to lab_local/fix_result.json (gitignored).
    rc = full_fix(path, FS, strong, dur, multi=int(os.environ.get('GPSTUNA_EPOCHS', '15')))
    if rc == 1:
        print("[locate] not enough complete orbits - a longer capture (300 s)")
        print("         usually gets the remaining subframes; retry.")
    elif rc:
        print("[locate] the solver did not find a position (see the reasons "
              "above). This is a")
        print("         software problem, not your antenna or radio: the sky "
              "view and the")
        print("         orbits decoded. Please open an issue with the "
              "[rel]/[fix] lines above.")


if __name__ == "__main__":
    main()
