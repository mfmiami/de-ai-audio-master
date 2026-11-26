#!/usr/bin/env python3
"""
Automated 4-archetype regression & future-proofing verification suite for De-AI Audio Master.

Runs standalone with zero external audio files (synthesizes realistic multi-stem fixtures for
electronic, rock, hiphop, and acoustic), plus optional --real-root check against local reference
track folders containing original.wav + stems/.

Usage:
  python3 tests/test_regression.py
  python3 tests/test_regression.py --real-root /path/to/reference_tracks
"""

import argparse
import glob
import os
import struct
import sys
import tempfile
import numpy as np
import soundfile as sf
from scipy import signal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "ai-audio-clean-master", "scripts"))
import clean_analog_master as cam


def _band_noise(rng, n, sr, lo, hi, rms_db, stereo_width=0.3):
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    sos = signal.butter(4, [lo, min(hi, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    m = signal.sosfilt(sos, a)
    s = signal.sosfilt(sos, b) * stereo_width
    x = np.stack([m + s, m - s], axis=1)
    x *= (10.0 ** (rms_db / 20.0)) / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    return x


def _pulsed_envelope(n, sr, hz, duty=0.25):
    t = np.arange(n) / sr
    ph = (t * hz) % 1.0
    env = np.exp(-ph / max(duty, 0.05))
    env /= np.sqrt(np.mean(env ** 2)) + 1e-12
    return env[:, None]


def make_synthetic_fixture(archetype, sr=48000, dur_s=8.0, seed=42):
    """Build deterministic multi-stem + stereo mix fixtures for each of the 4 archetypes."""
    rng = np.random.default_rng(seed)
    n = int(round(sr * dur_s))
    stems = {}

    if archetype == "electronic":
        stems["0 Lead Vocals.wav"] = _band_noise(rng, n, sr, 300, 9000, -25.0, 0.15) * _pulsed_envelope(n, sr, 1.5, 0.45)
        stems["2 Drums.wav"] = _band_noise(rng, n, sr, 45, 11000, -22.0, 0.15) * _pulsed_envelope(n, sr, 2.3, 0.20)
        stems["3 Bass.wav"] = _band_noise(rng, n, sr, 40, 180, -22.0, 0.05)
        stems["7 Synth.wav"] = _band_noise(rng, n, sr, 250, 12000, -21.0, 0.65)
        stems["4 Keyboard.wav"] = _band_noise(rng, n, sr, 200, 6000, -28.0, 0.45)
    elif archetype == "rock":
        stems["0 Lead Vocals.wav"] = _band_noise(rng, n, sr, 250, 10000, -22.0, 0.12) * _pulsed_envelope(n, sr, 1.8, 0.40)
        stems["2 Drums.wav"] = _band_noise(rng, n, sr, 50, 12000, -20.5, 0.25) * _pulsed_envelope(n, sr, 2.2, 0.22)
        stems["5 Guitar.wav"] = _band_noise(rng, n, sr, 150, 6500, -21.5, 0.35)
        stems["3 Bass.wav"] = _band_noise(rng, n, sr, 45, 500, -25.0, 0.05)
        stems["7 Synth.wav"] = _band_noise(rng, n, sr, 200, 4000, -45.0, 0.30)
    elif archetype == "hiphop":
        stems["0 Lead Vocals.wav"] = _band_noise(rng, n, sr, 250, 11000, -22.5, 0.10) * _pulsed_envelope(n, sr, 2.5, 0.28)
        stems["2 Drums.wav"] = _band_noise(rng, n, sr, 40, 12000, -20.0, 0.12) * _pulsed_envelope(n, sr, 1.8, 0.10)
        stems["3 Bass.wav"] = _band_noise(rng, n, sr, 35, 150, -24.5, 0.03) * _pulsed_envelope(n, sr, 1.8, 0.16)
        stems["4 Keyboard.wav"] = _band_noise(rng, n, sr, 180, 1400, -31.0, 0.35)
        stems["7 Synth.wav"] = _band_noise(rng, n, sr, 200, 1200, -36.0, 0.30)
    elif archetype == "acoustic":
        stems["0 Lead Vocals.wav"] = _band_noise(rng, n, sr, 220, 8500, -23.0, 0.15) * _pulsed_envelope(n, sr, 1.1, 0.55)
        stems["4 Piano.wav"] = _band_noise(rng, n, sr, 160, 5500, -22.0, 0.35) * _pulsed_envelope(n, sr, 1.4, 0.50)
        stems["6 Strings.wav"] = _band_noise(rng, n, sr, 250, 6500, -24.5, 0.45)
        stems["3 Bass.wav"] = _band_noise(rng, n, sr, 80, 400, -32.0, 0.10)
        stems["2 Drums.wav"] = _band_noise(rng, n, sr, 200, 6000, -46.0, 0.20)
    else:
        raise ValueError(archetype)

    orig = sum(stems.values())
    pk = float(np.max(np.abs(orig)))
    if pk > 0.65:
        scale = 0.65 / pk
        orig *= scale
        stems = {k: v * scale for k, v in stems.items()}
    return orig, stems, sr


def verify_wav_header_44_bytes(wav_path):
    with open(wav_path, "rb") as f:
        hdr = f.read(44)
    assert hdr[:4] == b"RIFF" and hdr[8:12] == b"WAVE", f"Invalid RIFF/WAVE header in {wav_path}"
    assert hdr[36:40] == b"data", f"Non-44-byte header (extra chunks found) in {wav_path}: {hdr[36:40]!r}"
    bits = struct.unpack("<H", hdr[34:36])[0]
    assert bits == 24, f"Expected 24-bit WAV, got {bits}-bit"


def run_synthetic_suite():
    print("=== Running 4-Archetype Synthetic Regression Suite ===")
    with tempfile.TemporaryDirectory(prefix="deai_reg_") as tmp:
        for idx, arch in enumerate(("electronic", "rock", "hiphop", "acoustic")):
            orig, stems, sr = make_synthetic_fixture(arch, seed=100 + idx)
            got = cam.detect_profile_from_stems(stems, sr=sr, orig=orig)
            assert got == arch, f"Stem auto-detect failed for {arch}: got {got}"

            tdir = os.path.join(tmp, arch)
            sdir = os.path.join(tdir, "stems")
            odir = os.path.join(tdir, "out")
            os.makedirs(sdir, exist_ok=True)
            in_wav = os.path.join(tdir, "orig.wav")
            sf.write(in_wav, orig, sr, subtype="PCM_16")
            for k, v in stems.items():
                sf.write(os.path.join(sdir, k), v, sr, subtype="PCM_16")

            res = cam.process_track(
                input_path=in_wav,
                out_dir=odir,
                stems_dir=sdir,
                profile="auto",
                oversample=4,
                mode="full",
                force_python_lufs=True,
                seed=2026 + idx,
            )
            assert res["profile"] == arch, f"process_track profile mismatch: {res['profile']} != {arch}"
            assert not res["any_nan_or_inf"], f"NaN/Inf in {arch}"
            assert abs(res["proc_lufs"] - res["orig_lufs"]) <= 0.4, f"LUFS drift in {arch}: {res['orig_lufs']} -> {res['proc_lufs']}"
            assert res["proc_tp_dbtp"] <= -1.0, f"True peak > -1.0 dBTP in {arch}: {res['proc_tp_dbtp']}"
            verify_wav_header_44_bytes(res["output_wav"])

            # Verify stereo-only fallback runs cleanly without error
            st_prof = cam.detect_profile_from_stereo(orig, sr=sr)
            assert st_prof in ("electronic", "rock", "hiphop", "acoustic")
            print(f"[PASS] {arch:10s} | stem-auto={got:10s} | stereo-auto={st_prof:10s} | LUFS {res['orig_lufs']}->{res['proc_lufs']} | TP {res['proc_tp_dbtp']} dBTP | 44B 24-bit OK")

    print("ALL 4 SYNTHETIC ARCHETYPE TESTS PASSED!")


def run_real_tracks_check(real_root):
    print(f"\n=== Checking Real Reference Tracks in {real_root} ===")
    expected = {
        "every_ordinary_thing": "hiphop",
        "falling_into_you": "rock",
        "go_around_again": "electronic",
        "session": "electronic",
    }
    checked = 0
    for p in sorted(glob.glob(os.path.join(real_root, "*", "original.wav"))):
        track = os.path.basename(os.path.dirname(p))
        orig, sr = sf.read(p, dtype="float64", always_2d=True)
        sdir = os.path.join(os.path.dirname(p), "stems")
        stems = {}
        if os.path.isdir(sdir):
            for f in sorted(glob.glob(os.path.join(sdir, "*.wav"))):
                x, _ = sf.read(f, dtype="float64", always_2d=True)
                stems[os.path.basename(f)] = x
        got_stem = cam.detect_profile_from_stems(stems, sr=sr, orig=orig) if stems else None
        got_stereo = cam.detect_profile_from_stereo(orig, sr=sr)
        exp = expected.get(track)
        print(f"  {track:24s} | stem-auto={str(got_stem):10s} | stereo-auto={got_stereo:10s} | expected={exp}")
        if exp:
            if got_stem is not None:
                assert got_stem == exp, f"Real stem auto-detect mismatch on {track}: {got_stem} != {exp}"
            assert got_stereo == exp, f"Real stereo-only auto-detect mismatch on {track}: {got_stereo} != {exp}"
        checked += 1
    assert checked > 0, f"No reference tracks found under {real_root}"
    print(f"ALL {checked} REAL REFERENCE TRACKS MATCHED (BOTH STEM AND STEREO-ONLY AUTO-DETECT)!")


def main():
    ap = argparse.ArgumentParser(description="Run De-AI Audio Master 4-archetype regression tests")
    ap.add_argument("--real-root", default=None, help="Optional folder of real reference tracks (<track>/original.wav + stems/)")
    args = ap.parse_args()
    run_synthetic_suite()
    if args.real_root:
        run_real_tracks_check(args.real_root)


if __name__ == "__main__":
    main()
