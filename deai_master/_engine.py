#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26.0",
#   "scipy>=1.11.0",
#   "soundfile>=0.12.1",
# ]
# ///
"""
De-AI Audio Master: Standalone Zero-Artifact Time-Domain Cleanup & Linear-Phase Mastering CLI.

Runs 100% locally on CPU with only numpy, scipy, and soundfile (ffmpeg is optional).
Can be run directly via `uv run`, installed as `deai-master`, used via `deai-master-ui`,
or invoked by an AI agent skill (Antigravity, Claude Code, Codex).

Pipeline:
1. Optional 87.5%-overlap exact additive complex Wiener stem split (orig == xs + xb to ~1e-15)
   with correlated-bleed projection (r > 0.35), 1.4x vocal protection, and sub-90 Hz backing guard.
2. Pure time-domain isolated target-stem re-amp:
   - Sub-protected (>120 Hz) Mid/Side 4x-oversampled Class-A asymmetric saturation
   - 8-tap Lanczos-4 quadrature Juno-style BBD stereo chorus (>900 Hz, zero treble loss)
   - Subtle BPM-synced warm LP analog delay tail (4.2% wet)
3. Zero-STFT-framing contiguous full-mix mastering chain:
   - 4097-tap Blackman-windowed Linear-Phase FIR stationary comb flattening (<=2.2 dB cuts only)
   - Envelope-following 4x-oversampled harmonic air exciter above the detected HF cutoff
   - Pure time-domain 2nd-order analog allpass cascade (>5.8 kHz, flat 0.000 dB magnitude)
   - Slow time-domain biquad tonal breath (+-0.40 dB across 4 bands)
   - Stereo-linked Lanczos-8 (16-tap) organic micro-tempo breath (identical L/R read pointers)
   - Sub-protected (>120 Hz) Mid/Side 4x-oversampled Class-A bus warmth
   - 2049-tap Linear-Phase FIR reference-sparkle balance trim (+-0.85 dB max)
   - -82 dBFS 1/f analog floor + true 24-bit 1-LSB TPDF dither
4. ITU-R BS.1770-4 LUFS matching (uses ffmpeg ebur128 if available, or built-in pure-Python
   K-weighted ITU-R BS.1770-4 loudness + 4x oversampled true peak fallback), clean 44-byte
   WAV header export, transient crest verification, and 4 automatic spot-check clips.
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import time
import numpy as np
import soundfile as sf
from scipy import signal


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _bs1770_k_weight(x_stereo, sr):
    """ITU-R BS.1770-4 Stage 1 (pre-filter shelf) + Stage 2 (RLB high-pass) at arbitrary sample rate."""
    # Exact 48 kHz coefficients from ITU-R BS.1770-4; bilinear warp for other sample rates
    if sr == 48000:
        b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
        a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
        b2 = np.array([1.0, -2.0, 1.0])
        a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])
    else:
        # Derive analog prototypes matching BS.1770-4 shelf (+4 dB @ 1681.97 Hz, Q=0.707175) & HP (38.135 Hz, Q=0.500327)
        f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
        K = math.tan(math.pi * f0 / sr)
        Vh = 10.0 ** (G / 20.0)
        Vb = Vh ** 0.4996667741545416
        a0 = 1.0 + K / Q + K * K
        b1 = np.array([(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0])
        a1 = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
        f_hp, Q_hp = 38.13547087602444, 0.5003270373238773
        K2 = math.tan(math.pi * f_hp / sr)
        a0_2 = 1.0 + K2 / Q_hp + K2 * K2
        b2 = np.array([1.0, -2.0, 1.0]) / a0_2
        a2 = np.array([1.0, 2.0 * (K2 * K2 - 1.0) / a0_2, (1.0 - K2 / Q_hp + K2 * K2) / a0_2])
    y = signal.lfilter(b1, a1, x_stereo, axis=0)
    y = signal.lfilter(b2, a2, y, axis=0)
    return y


def measure_lufs_lra_tp_python(x_stereo, sr):
    """Pure-Python ITU-R BS.1770-4 integrated LUFS, EBU R128 LRA, and 4x oversampled True Peak."""
    y = _bs1770_k_weight(x_stereo, sr)
    blk_len = int(round(0.4 * sr))
    hop = int(round(0.1 * sr))
    n = len(y)
    if n < blk_len:
        ms = float(np.mean(y[:, 0] ** 2) + np.mean(y[:, 1] ** 2))
        lufs = -0.691 + 10.0 * math.log10(max(ms, 1e-15))
        lra = 0.0
    else:
        n_blocks = 1 + (n - blk_len) // hop
        cs0 = np.concatenate([[0.0], np.cumsum(y[:, 0] ** 2)])
        cs1 = np.concatenate([[0.0], np.cumsum(y[:, 1] ** 2)])
        idx = np.arange(n_blocks) * hop
        z = ((cs0[idx + blk_len] - cs0[idx]) + (cs1[idx + blk_len] - cs1[idx])) / blk_len
        l_blk = -0.691 + 10.0 * np.log10(np.maximum(z, 1e-15))
        above_abs = l_blk > -70.0
        if not np.any(above_abs):
            lufs = -70.0
        else:
            g_rel = -0.691 + 10.0 * math.log10(max(float(np.mean(z[above_abs])), 1e-15)) - 10.0
            above_rel = l_blk > g_rel
            lufs = -0.691 + 10.0 * math.log10(max(float(np.mean(z[above_rel])), 1e-15)) if np.any(above_rel) else -70.0

        # Short-term 3.0s blocks (hop 0.1s) for LRA (EBU Tech 3342)
        st_len = int(round(3.0 * sr))
        if n >= st_len:
            n_st = 1 + (n - st_len) // hop
            s_idx = np.arange(n_st) * hop
            z_st = ((cs0[s_idx + st_len] - cs0[s_idx]) + (cs1[s_idx + st_len] - cs1[s_idx])) / st_len
            l_st = -0.691 + 10.0 * np.log10(np.maximum(z_st, 1e-15))
            m_abs = l_st > -70.0
            if np.any(m_abs):
                g_st = -0.691 + 10.0 * math.log10(max(float(np.mean(z_st[m_abs])), 1e-15)) - 20.0
                valid = l_st[l_st > g_st]
                lra = float(np.percentile(valid, 95) - np.percentile(valid, 10)) if len(valid) >= 2 else 0.0
            else:
                lra = 0.0
        else:
            lra = 0.0

    # 4x oversampled True Peak (dBTP) in chunks to stay memory-light
    tp_lin = 1e-9
    chunk = 524288
    for s0 in range(0, n, chunk):
        seg = x_stereo[s0:min(n, s0 + chunk)]
        for ch in range(2):
            up = signal.resample_poly(seg[:, ch], 4, 1)
            pk = float(np.max(np.abs(up)))
            if pk > tp_lin:
                tp_lin = pk
    tp_db = 20.0 * math.log10(max(tp_lin, 1e-9))
    return round(lufs, 1), round(lra, 1), round(tp_db, 1)


def measure_lufs_lra_tp(path, force_python=False):
    """Measure LUFS, LRA, and True Peak via ffmpeg ebur128 if available, else pure-Python BS.1770-4."""
    if _HAS_FFMPEG and not force_python:
        cmd = ["ffmpeg", "-nostats", "-i", path, "-filter_complex", "ebur128=peak=true", "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stderr
        m_i = re.findall(r"I:\s+(-?\d+\.\d+)\s+LUFS", out)
        m_lra = re.findall(r"LRA:\s+(-?\d+\.\d+)\s+LU", out)
        m_p = re.findall(r"Peak:\s+(-?\d+\.\d+)\s+dBFS", out)
        if m_i and m_p:
            lra = float(m_lra[-1]) if m_lra else 0.0
            return float(m_i[-1]), lra, float(m_p[-1])
    x, sr = sf.read(path, dtype="float64")
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    return measure_lufs_lra_tp_python(x, sr)


def write_wav_24bit_tpdf(path, x_float64, sr, seed=9999):
    """Write clean 44-byte 24-bit PCM stereo WAV with 1-LSB triangular-PDF (TPDF) dither."""
    rng = np.random.default_rng(seed)
    lsb = 1.0 / float(1 << 23)
    dither = (rng.uniform(-1.0, 1.0, size=x_float64.shape) + rng.uniform(-1.0, 1.0, size=x_float64.shape)) * (0.5 * lsb)
    y = np.clip(x_float64 + dither, -1.0 + lsb, 1.0 - lsb)
    sf.write(path, y, sr, subtype="PCM_24")


def _dyadic_fft_bpm(seg, sr, rate=200.0, bpm_lo=85.0, bpm_hi=170.0):
    """Dyadic FFT tempogram (scores 0.5x + 1x + 2x + 4x beat harmonics to resist subharmonic traps)."""
    hop = max(1, int(round(sr / rate)))
    n = len(seg) // hop
    if n < 64:
        return 120.0
    env = np.sqrt(np.mean(seg[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    odf = np.maximum(0.0, np.diff(env))
    odf = odf - np.mean(odf)
    nfft = 1 << int(np.ceil(np.log2(max(len(odf) * 4, 256))))
    spec = np.abs(np.fft.rfft(odf * np.hanning(len(odf)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / rate)
    bpms = freqs * 60.0
    idx = np.where((bpms >= bpm_lo) & (bpms <= bpm_hi))[0]
    if len(idx) == 0:
        return 120.0
    s = (spec[idx]
         + 0.6 * spec[np.maximum(idx // 2, 1)]
         + 0.6 * spec[np.minimum(idx * 2, len(spec) - 1)]
         + 0.35 * spec[np.minimum(idx * 4, len(spec) - 1)])
    best = idx[int(np.argmax(s))]
    return round(float(bpms[best]), 1)


def estimate_bpm(x_stereo, sr, drums_stereo=None):
    """
    Estimate BPM using a dyadic FFT tempogram. Uses the kick band (40-140 Hz) of the Drums
    stem when available, or the full mix otherwise.
    """
    if drums_stereo is not None and np.std(drums_stereo) > 1e-5:
        M = 0.5 * (drums_stereo[:, 0] + drums_stereo[:, 1])
        M = signal.sosfiltfilt(signal.butter(4, [40.0, 140.0], btype="bandpass", fs=sr, output="sos"), M)
        return _dyadic_fft_bpm(M, sr, rate=200.0)
    M = 0.5 * (x_stereo[:, 0] + x_stereo[:, 1])
    return _dyadic_fft_bpm(M, sr, rate=200.0)


def select_auto_audition_ranges(orig, sr, stems=None, target_name=None, clip_s=14.0):
    """
    Automatically pick 2-3 maximally revealing audition ranges:
      1) Exposed intro / quiet opening
      2) Strongest target-stem / melodic hook section
      3) Densest vocal + drum climax chorus
    """
    N = len(orig)
    dur = N / sr
    if dur <= clip_s * 1.5:
        return [(0.0, round(dur, 1))]

    step_s = 5.0
    win = int(round(clip_s * sr))
    step = int(round(step_s * sr))
    starts = list(range(0, max(1, N - win), step))
    if not starts:
        return [(0.0, round(min(clip_s, dur), 1))]

    mix_db = np.array([20.0 * math.log10(float(np.sqrt(np.mean(orig[s:s + win] ** 2))) + 1e-12) for s in starts])
    ts = np.array([s / sr for s in starts])

    # 1. Exposed intro: earliest non-silent window in first 25% of track
    peak_mix = float(np.max(mix_db))
    early_mask = (ts >= 4.0) & (ts <= max(20.0, dur * 0.25)) & (mix_db > peak_mix - 18.0)
    if np.any(early_mask):
        idx_intro = int(np.where(early_mask)[0][0])
    else:
        idx_intro = min(1, len(starts) - 1)
    t_intro = float(ts[idx_intro])

    # 2. Exposed hook / target stem peak (at least clip_s + 6s away from intro)
    if stems and target_name and target_name in stems:
        xs = stems[target_name]
        syn_db = np.array([20.0 * math.log10(float(np.sqrt(np.mean(xs[s:s + win] ** 2))) + 1e-12) for s in starts])
        vox_keys = [k for k in stems if "lead" in k.lower() or "vocal" in k.lower()]
        if vox_keys:
            xv = stems[vox_keys[0]]
            vox_db = np.array([20.0 * math.log10(float(np.sqrt(np.mean(xv[s:s + win] ** 2))) + 1e-12) for s in starts])
            hook_score = syn_db - 0.35 * np.maximum(vox_db, -45.0)
        else:
            hook_score = syn_db
    else:
        # Mid-high band energy proxy for melodic hook
        hook_score = mix_db.copy()

    far_from_intro = np.abs(ts - t_intro) >= (clip_s + 6.0)
    if np.any(far_from_intro):
        cand = np.where(far_from_intro)[0]
        idx_hook = int(cand[int(np.argmax(hook_score[cand]))])
        t_hook = float(ts[idx_hook])
    else:
        t_hook = None

    # 3. Densest climax chorus (loudest overall mix away from intro and hook)
    far_both = np.abs(ts - t_intro) >= (clip_s + 6.0)
    if t_hook is not None:
        far_both = far_both & (np.abs(ts - t_hook) >= (clip_s + 6.0))
    if np.any(far_both):
        cand = np.where(far_both)[0]
        idx_chorus = int(cand[int(np.argmax(mix_db[cand]))])
        t_chorus = float(ts[idx_chorus])
    else:
        t_chorus = None

    chosen = sorted([t for t in [t_intro, t_hook, t_chorus] if t is not None])
    return [(round(t, 1), round(min(dur, t + clip_s), 1)) for t in chosen]


def read_stems_folder(orig, sr, stems_dir, target_keyword):
    """Load and validate all WAV stems matching orig shape/sample rate."""
    files = sorted(glob.glob(os.path.join(stems_dir, "*.wav")))
    if not files:
        raise FileNotFoundError(f"No .wav stems found in {stems_dir}")

    stems = {}
    for f in files:
        x, s = sf.read(f, dtype="float64")
        if x.ndim == 1:
            x = np.stack([x, x], axis=1)
        if s != sr or len(x) != len(orig):
            raise ValueError(f"Stem {os.path.basename(f)} shape/sr mismatch: {x.shape}@{s} vs {orig.shape}@{sr}")
        stems[os.path.basename(f)] = x

    matches = [k for k in stems if target_keyword.lower() in k.lower()]
    if not matches:
        # Fallback: pick loudest non-vocal, non-drum, non-bass stem
        non_core = [k for k in stems if not any(w in k.lower() for w in ("vocal", "vox", "drum", "bass", "perc"))]
        pool = non_core if non_core else list(stems.keys())
        target_name = max(pool, key=lambda k: float(np.mean(stems[k] ** 2)))
        print(f"  Note: no stem matched '{target_keyword}'; auto-selected loudest melodic stem: {target_name}")
    else:
        target_name = matches[0]

    drums_matches = [k for k in stems if "drum" in k.lower()]
    drums_stereo = stems[drums_matches[0]] if drums_matches else None
    return stems, target_name, drums_stereo


def split_stems_wiener(orig, sr, stems, target_name, power=2.5):
    """
    87.5%-overlap exact additive complex Wiener split (orig == xs + xb to ~1e-15).
    - Auto-detects correlated bleed stems (e.g. Keyboard duplicating Synth)
    - Applies 1.4x protection weighting to any Vocal stem so lead vocals stay untouched
    - Protects sub-90 Hz 100% in backing
    """
    x_syn = stems[target_name]
    leak_names, vocal_names, other_names = [], [], []
    for k, v in stems.items():
        if k == target_name:
            continue
        if "vocal" in k.lower() or "vox" in k.lower():
            vocal_names.append(k)
            continue
        r = float(np.corrcoef(x_syn[:, 0], v[:, 0])[0, 1]) if (np.std(x_syn[:, 0]) > 1e-7 and np.std(v[:, 0]) > 1e-7) else 0.0
        if r > 0.35:
            leak_names.append((k, r))
        else:
            other_names.append(k)

    print(f"  Target stem: {target_name}")
    if leak_names:
        print(f"  Correlated bleed stem(s) projected cleanly: {[(n, round(r, 3)) for n, r in leak_names]}")
    if vocal_names:
        print(f"  Vocal-protected stem(s) (1.4x guard): {vocal_names}")

    x_leak = sum(stems[n] for n, _ in leak_names) if leak_names else np.zeros_like(orig)
    x_voc = sum(stems[n] for n in vocal_names) if vocal_names else np.zeros_like(orig)
    x_other = sum(stems[n] for n in other_names) if other_names else np.zeros_like(orig)

    n_fft, hop = 2048, 256
    win = signal.windows.hann(n_fft, sym=False)
    out_s = np.zeros_like(orig)
    out_b = np.zeros_like(orig)
    for ch in range(2):
        _, _, X = signal.stft(orig[:, ch], sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, S = signal.stft(x_syn[:, ch], sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, K = signal.stft(x_leak[:, ch], sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, V = signal.stft(x_voc[:, ch], sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, O = signal.stft(x_other[:, ch], sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)

        num = np.conj(S) * K
        den = (np.abs(S) ** 2) + 1e-12
        ker = np.ones((1, 7)) / 7.0
        alpha = signal.convolve2d(num, ker, mode="same") / (signal.convolve2d(den, ker, mode="same") + 1e-12)
        proj = np.clip(np.real(alpha), 0.0, 1.45)
        S_tot = S + proj * S
        O_tot = O + (K - proj * S) + 1.4 * V

        pS = np.abs(S_tot) ** power
        pO = np.abs(O_tot) ** power
        M = pS / (pS + pO + 1e-12)
        M = signal.convolve2d(M, np.ones((3, 5)) / 15.0, mode="same")
        freqs = np.linspace(0.0, sr / 2.0, X.shape[0])
        hp = np.clip((freqs - 90.0) / 70.0, 0.0, 1.0)[:, None]
        M *= hp

        _, xs = signal.istft(M * X, sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, xb = signal.istft((1.0 - M) * X, sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        out_s[:, ch] = xs[:len(orig)]
        out_b[:, ch] = xb[:len(orig)]

    err = float(np.max(np.abs((out_s + out_b) - orig)))
    print(f"  Exact 87.5%-overlap Wiener sum error vs original: {err:.2e}")
    return out_s, out_b


def oversampled_class_a_sat(x, sr, drive=1.45, asym=0.035, wet=0.22, hp_protect_hz=120.0, ms_mode=False, os_factor=4):
    """4x oversampled Class-A asymmetric triode/tape saturation with sub-bass protection (>hp_protect_hz)."""
    N = len(x)
    if hp_protect_hz and hp_protect_hz > 25.0:
        sos_lp = signal.butter(4, hp_protect_hz, btype="lowpass", fs=sr, output="sos")
        x_lo = signal.sosfiltfilt(sos_lp, x, axis=0)
        x_hi = x - x_lo
    else:
        x_lo = None
        x_hi = x

    if ms_mode:
        work = np.stack([0.5 * (x_hi[:, 0] + x_hi[:, 1]), 0.5 * (x_hi[:, 0] - x_hi[:, 1])], axis=1)
        drives = (drive, 1.0 + (drive - 1.0) * 0.65)
        asyms = (asym, asym * 0.5)
        wets = (wet, wet * 0.75)
    else:
        work = x_hi
        drives = (drive, drive)
        asyms = (asym, asym)
        wets = (wet, wet)

    out_hi = np.empty_like(work)
    blk, pad = 262144, 1024
    for s0 in range(0, N, blk):
        s1 = min(N, s0 + blk)
        a, b = max(0, s0 - pad), min(N, s1 + pad)
        seg = work[a:b]
        o0 = s0 - a
        o1 = o0 + (s1 - s0)
        for ch in range(2):
            pk = max(float(np.max(np.abs(seg[:, ch]))), 1e-4)
            d, ay, wt = drives[ch], asyms[ch], wets[ch]
            norm = math.tanh(d)
            up = signal.resample_poly(seg[:, ch] / pk, os_factor, 1)
            sat = np.tanh(d * up + ay * (up ** 2)) / norm
            dn = signal.resample_poly(sat, 1, os_factor)[:len(seg)] * pk
            out_hi[s0:s1, ch] = (1.0 - wt) * seg[o0:o1, ch] + wt * dn[o0:o1]

    if ms_mode:
        out_hi = np.stack([out_hi[:, 0] + out_hi[:, 1], out_hi[:, 0] - out_hi[:, 1]], axis=1)

    sos_dc = signal.butter(2, 20.0, btype="highpass", fs=sr, output="sos")
    out_hi = signal.sosfiltfilt(sos_dc, out_hi, axis=0)
    return (x_lo + out_hi) if x_lo is not None else out_hi


def lanczos_resample_variable(x_ch, read_pos, a_lz=4):
    """Exact windowed-sinc Lanczos-a variable delay/resampling (zero treble loss)."""
    N = len(x_ch)
    ks = np.arange(-a_lz + 1, a_lz + 1, dtype=np.int64)
    out = np.empty(N, dtype=np.float64)
    blk = 250000
    for s0 in range(0, N, blk):
        s1 = min(N, s0 + blk)
        rp = read_pos[s0:s1]
        i0 = np.floor(rp).astype(np.int64)
        frac = rp - i0
        dx = frac[:, None] - ks[None, :]
        w = np.sinc(dx) * np.sinc(dx / a_lz)
        w /= (np.sum(w, axis=1, keepdims=True) + 1e-12)
        idxs = np.clip(i0[:, None] + ks[None, :], 0, N - 1)
        out[s0:s1] = np.sum(x_ch[idxs] * w, axis=1)
    return out


def time_domain_bbd_chorus_hq(x, sr, hp_hz=900.0, base_ms=6.2, depth_ms=1.8, rate_hz=0.34, wet=0.14):
    """Pure time-domain quadrature Juno-style BBD stereo chorus using 8-tap Lanczos-4 sinc interpolation."""
    N = len(x)
    sos_hp = signal.butter(4, hp_hz, btype="highpass", fs=sr, output="sos")
    x_hi = signal.sosfiltfilt(sos_hp, x, axis=0)
    t = np.arange(N, dtype=np.float64)
    d_L = (base_ms + depth_ms * np.sin(2.0 * np.pi * rate_hz * (t / sr))) * (sr / 1000.0)
    d_R = (base_ms + depth_ms * np.cos(2.0 * np.pi * rate_hz * (t / sr))) * (sr / 1000.0)
    y_L = lanczos_resample_variable(x_hi[:, 0], np.clip(t - d_L, 0.0, N - 1.001), a_lz=4)
    y_R = lanczos_resample_variable(x_hi[:, 1], np.clip(t - d_R, 0.0, N - 1.001), a_lz=4)
    y_ch = np.stack([y_L, y_R], axis=1)
    lp_hz = min(13500.0, sr * 0.44)
    sos_lp = signal.butter(2, lp_hz, btype="lowpass", fs=sr, output="sos")
    y_ch = signal.sosfiltfilt(sos_lp, y_ch, axis=0)
    return x + wet * y_ch


def reamp_stem_time_domain(xs, sr, bpm):
    s = oversampled_class_a_sat(xs, sr, drive=1.50, asym=0.036, wet=0.26, hp_protect_hz=120.0, ms_mode=True, os_factor=4)
    s = time_domain_bbd_chorus_hq(s, sr, hp_hz=900.0, base_ms=6.2, depth_ms=1.8, rate_hz=0.34, wet=0.13)
    d1 = int(round(sr * (60.0 / bpm) * 0.5))
    d2 = int(round(sr * (60.0 / bpm) * 0.75))
    if d2 < len(s):
        del_sig = np.zeros_like(s)
        del_sig[d1:, 0] += s[:-d1, 1]
        del_sig[d2:, 1] += s[:-d2, 0]
        sos_dlp = signal.butter(2, 2800.0, btype="lowpass", fs=sr, output="sos")
        del_sig = signal.sosfiltfilt(sos_dlp, del_sig, axis=0)
        s = s + 0.042 * del_sig
    s *= np.sqrt(np.mean(xs ** 2)) / (np.sqrt(np.mean(s ** 2)) + 1e-12)
    return s


def detect_hf_cliff_hz(x_full, sr):
    M = 0.5 * (x_full[:, 0] + x_full[:, 1])
    fq, P = signal.welch(M, sr, nperseg=8192)
    Pd = 10.0 * np.log10(P + 1e-20)
    nyq = sr / 2.0
    best_hz, best_drop = min(18000.0, nyq - 2500.0), 0.0
    for fc in np.arange(14500.0, nyq - 2200.0, 250.0):
        e_lo = float(np.mean(Pd[(fq >= fc - 1500.0) & (fq < fc)]))
        e_hi = float(np.mean(Pd[(fq >= fc + 500.0) & (fq < min(fc + 2500.0, nyq - 200.0))]))
        if e_lo - e_hi > best_drop:
            best_drop = e_lo - e_hi
            best_hz = float(fc)
    return best_hz, best_drop


def build_linear_phase_comb_fir(x_full, sr, max_cut_db=2.2, fir_taps=4097):
    """Build an exact 0.000-degree Linear-Phase Blackman-windowed FIR for conservative comb flattening."""
    n_fft, hop = 16384, 4096
    win = signal.windows.hann(n_fft, sym=False)
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    hi_cap = min(18200.0, sr * 0.42)
    h_kernels = []

    for ch in range(2):
        _, _, Z = signal.stft(x_full[:, ch], sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        mag_db = 20.0 * np.log10(np.abs(Z) + 1e-9)
        mean_db = np.mean(mag_db, axis=1)
        floor_db = np.percentile(mag_db, 25, axis=1)
        exc_m = mean_db - signal.medfilt(mean_db, kernel_size=61)
        exc_f = floor_db - signal.medfilt(floor_db, kernel_size=61)

        band = (freqs >= 2200.0) & (freqs <= hi_cap)
        stat_exc = np.zeros_like(freqs)
        stat_exc[band] = np.maximum(0.0, np.minimum(exc_m[band], exc_f[band]))
        cut_db = np.where(stat_exc > 0.50, np.clip(stat_exc, 0.0, max_cut_db), 0.0)
        cut_db = np.convolve(cut_db, [0.25, 0.5, 0.25], mode="same")
        g_raw = 10.0 ** (-cut_db / 20.0)

        k_sm = 401
        avg_loss_db = np.convolve(cut_db, np.ones(k_sm) / k_sm, mode="same")
        g_bal = g_raw * (10.0 ** (np.clip(avg_loss_db, 0.0, 0.65) / 20.0))

        if ch == 0:
            print(f"  [Comb-FIR] Linear-phase FIR ({fir_taps} taps): {int(np.sum(cut_db > 0.2))} narrow comb bins (max -{cut_db.max():.2f} dB, broadband restore max +{avg_loss_db.max():.2f} dB)")

        h_full = np.fft.irfft(g_bal, n=n_fft)
        h_shift = np.fft.fftshift(h_full)
        mid = n_fft // 2
        half = fir_taps // 2
        h_fir = h_shift[mid - half:mid + half + 1] * signal.windows.blackman(fir_taps)
        h_kernels.append(h_fir)

    return h_kernels


def apply_linear_phase_fir(x, h_kernels):
    y = np.empty_like(x)
    for ch in range(2):
        y[:, ch] = signal.fftconvolve(x[:, ch], h_kernels[ch], mode="same")
    return y


def apply_reference_sparkle_fir(y_proc, x_ref, sr, fir_taps=2049, max_db=0.85):
    """
    Gentle broad (~1-octave) linear-phase FIR balance trim (60 Hz - 18.2 kHz)
    clamped strictly within +-0.85 dB so every band from 30 Hz to 16 kHz stays close
    to the original reference (zero STFT framing, zero phase distortion).
    """
    n_fft = 8192
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    log_f = np.log2(np.maximum(freqs, 20.0))
    anchors = np.array([70.0, 160.0, 380.0, 900.0, 2000.0, 4500.0, 9500.0, 15500.0])
    f_w, Pr = signal.welch(0.5 * (x_ref[:, 0] + x_ref[:, 1]), sr, nperseg=4096)
    _, Pp = signal.welch(0.5 * (y_proc[:, 0] + y_proc[:, 1]), sr, nperseg=4096)
    d_anc = np.zeros(len(anchors), dtype=np.float64)
    for i, fc in enumerate(anchors):
        m = (f_w >= fc * 0.72) & (f_w <= fc * 1.38)
        if np.any(m):
            d_anc[i] = float(np.clip(10.0 * np.log10((np.sum(Pr[m]) + 1e-20) / (np.sum(Pp[m]) + 1e-20)), -max_db, max_db))
    w = np.exp(-0.5 * ((log_f[:, None] - np.log2(anchors)[None, :]) / 0.72) ** 2)
    w /= (np.sum(w, axis=1, keepdims=True) + 1e-8)
    fade = np.clip((freqs - 45.0) / 35.0, 0.0, 1.0) * np.clip((19500.0 - freqs) / 2500.0, 0.0, 1.0)
    H = 10.0 ** (((w @ d_anc) * fade) / 20.0)
    h_full = np.fft.fftshift(np.fft.irfft(H, n=n_fft))
    mid, half = n_fft // 2, fir_taps // 2
    h_fir = h_full[mid - half:mid + half + 1] * signal.windows.blackman(fir_taps)
    out = np.empty_like(y_proc)
    for ch in range(2):
        out[:, ch] = signal.fftconvolve(y_proc[:, ch], h_fir, mode="same")
    return out


def time_domain_envelope_air_exciter(x, sr, cliff_hz=18000.0, mix=0.46):
    """Envelope-following 4x-oversampled time-domain harmonic air exciter above the HF cliff."""
    nyq = sr / 2.0
    out_hp = min(max(cliff_hz - 400.0, 14500.0), nyq - 2500.0)
    out_lp = min(nyq - 1200.0, 22800.0)
    if out_lp <= out_hp + 500.0:
        return x
    src_lo = max(out_hp * 0.5, 7500.0)
    src_hi = min(out_hp * 0.85, nyq - 3000.0)
    sos_bp = signal.butter(4, [src_lo, src_hi], btype="bandpass", fs=sr, output="sos")
    src = signal.sosfiltfilt(sos_bp, x, axis=0)
    N = len(x)
    air = np.empty_like(x)
    blk, pad = 262144, 1024
    for s0 in range(0, N, blk):
        s1 = min(N, s0 + blk)
        a, b = max(0, s0 - pad), min(N, s1 + pad)
        seg = src[a:b]
        rms = max(float(np.sqrt(np.mean(seg ** 2))), 1e-6)
        o0 = s0 - a
        for ch in range(2):
            up = signal.resample_poly(seg[:, ch] / rms, 4, 1)
            exc = np.sqrt(up * up + 0.04) - 0.2 + 0.14 * np.tanh(2.4 * up)
            dn = signal.resample_poly(exc, 1, 4)[:len(seg)] * rms
            air[s0:s1, ch] = dn[o0:o0 + (s1 - s0)]
    sos_out = signal.butter(4, [out_hp, out_lp], btype="bandpass", fs=sr, output="sos")
    air = signal.sosfiltfilt(sos_out, air, axis=0)

    env_sq = signal.sosfiltfilt(signal.butter(2, 25.0, btype="lowpass", fs=sr, output="sos"),
                                np.mean(src ** 2, axis=1))
    env = np.sqrt(np.maximum(env_sq, 0.0))
    ref_e = float(np.percentile(env, 75)) + 1e-9
    gate = np.clip(env / (ref_e * 0.15), 0.0, 1.0)[:, None]
    return x + mix * (air * gate)


def analog_allpass_cascade(x, sr, freqs_L=(5800.0, 9400.0, 13800.0), freqs_R=(6250.0, 10100.0, 14600.0), q=0.72):
    y = x.copy()
    nyq = sr / 2.0
    for ch, flist in [(0, freqs_L), (1, freqs_R)]:
        sig = y[:, ch]
        for fc in flist:
            if fc >= nyq * 0.88:
                continue
            w0 = 2.0 * math.pi * fc / sr
            alpha = math.sin(w0) / (2.0 * q)
            cw = math.cos(w0)
            a0 = 1.0 + alpha
            b = np.array([1.0 - alpha, -2.0 * cw, 1.0 + alpha]) / a0
            a = np.array([1.0, -2.0 * cw / a0, (1.0 - alpha) / a0])
            sig = signal.lfilter(b, a, sig)
        y[:, ch] = sig
    return y


def time_domain_slow_mastering_breath(x, sr, max_db=0.40, seed=2026):
    rng = np.random.default_rng(seed)
    N = len(x)
    t = np.arange(N, dtype=np.float64) / sr
    y = x.copy()
    for fc in [300.0, 900.0, 2400.0, 5800.0]:
        sos_bp = signal.butter(2, [fc * 0.65, fc * 1.55], btype="bandpass", fs=sr, output="sos")
        band_sig = signal.sosfiltfilt(sos_bp, y, axis=0)
        gain_db = max_db * np.sin(2.0 * np.pi * rng.uniform(0.04, 0.11) * t + rng.uniform(0, 2 * np.pi))
        y += band_sig * ((10.0 ** (gain_db / 20.0) - 1.0)[:, None])
    return y


def stereo_linked_lanczos8_timing_breath(x, sr, max_rate=0.0011, seed=2026):
    rng = np.random.default_rng(seed)
    N = len(x)
    n_knots = max(12, int(round(N / sr * 0.22)) + 2)
    kr = rng.uniform(-max_rate, max_rate, size=n_knots)
    kr -= np.mean(kr)
    rate_t = np.interp(np.arange(N, dtype=np.float64), np.linspace(0, N - 1, n_knots), kr)
    k_ma = min(int(sr * 1.2), max(3, N // 4))
    rate_t = signal.convolve(rate_t, np.ones(k_ma) / k_ma, mode="same")
    rate_t -= np.mean(rate_t)
    rp = np.clip(np.cumsum(1.0 + rate_t) - 1.0, 0.0, N - 1.001)
    y_L = lanczos_resample_variable(x[:, 0], rp, a_lz=8)
    y_R = lanczos_resample_variable(x[:, 1], rp, a_lz=8)
    return np.stack([y_L, y_R], axis=1)


def apply_full_mix_cleanup_v2(x, x_ref, sr, comb_firs, cliff_hz, seed=2026):
    y = apply_linear_phase_fir(x, comb_firs)
    y = time_domain_envelope_air_exciter(y, sr, cliff_hz=cliff_hz, mix=0.46)
    y = analog_allpass_cascade(y, sr)
    y = time_domain_slow_mastering_breath(y, sr, max_db=0.40, seed=seed)
    y = stereo_linked_lanczos8_timing_breath(y, sr, max_rate=0.0011, seed=seed)
    y = oversampled_class_a_sat(y, sr, drive=1.22, asym=0.015, wet=0.11, hp_protect_hz=120.0, ms_mode=True, os_factor=4)
    y = apply_reference_sparkle_fir(y, x_ref, sr, fir_taps=2049, max_db=0.85)
    rng = np.random.default_rng(seed + 5555)
    N = len(y)
    for ch in range(2):
        w = rng.standard_normal(N)
        b_lp, a_lp = signal.butter(1, 3500.0, btype="lowpass", fs=sr)
        pink = signal.lfilter(b_lp, a_lp, w)
        pink *= (10.0 ** (-82.0 / 20.0)) / (np.sqrt(np.mean(pink ** 2)) + 1e-12)
        y[:, ch] += pink
    return y


def apply_edge_fade(x, sr, ms=10.0):
    y = x.copy()
    n = int(round(sr * ms / 1000.0))
    fade = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n)))[:, None]
    y[:n] *= fade
    y[-n:] *= fade[::-1]
    return y


def spectral_qa(x_orig, x_proc, sr, cliff_hz):
    def stats(x):
        M = 0.5 * (x[:, 0] + x[:, 1])
        fq, P = signal.welch(M, sr, nperseg=16384)
        Pd = 10.0 * np.log10(P + 1e-20)
        cm = (fq >= 2500.0) & (fq <= min(15000.0, sr * 0.4))
        rip_std = float(np.std(Pd[cm] - signal.medfilt(Pd[cm], 31)))
        e_lo = float(np.mean(Pd[(fq >= cliff_hz - 2000.0) & (fq < cliff_hz)]))
        e_hi = float(np.mean(Pd[(fq >= cliff_hz + 1000.0) & (fq < min(cliff_hz + 3500.0, sr / 2.0 - 200.0))]))
        lr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
        mono = float(10.0 * np.log10(np.mean((0.5 * (x[:, 0] + x[:, 1])) ** 2) / (np.mean(x ** 2) + 1e-12)))
        sos_hf = signal.butter(4, [6000.0, min(16000.0, sr * 0.44)], btype="bandpass", fs=sr, output="sos")
        hf = signal.sosfiltfilt(sos_hf, M)
        hf_crest = float(20.0 * np.log10((np.percentile(np.abs(hf), 99.9) + 1e-12) / (np.sqrt(np.mean(hf ** 2)) + 1e-12)))
        return rip_std, e_lo - e_hi, lr, mono, hf_crest, fq, P

    ro, co, lro, mo, hco, fq, Po = stats(x_orig)
    rp, cp, lrp, mp, hcp, _, Pp = stats(x_proc)
    bands = {}
    for lo, hi in [(30, 100), (100, 250), (250, 1000), (1000, 4000), (4000, 10000), (10000, 16000), (16000, 18500), (18500, 22000)]:
        if hi >= sr / 2.0:
            continue
        m = (fq >= lo) & (fq < hi)
        bands[f"{lo}-{hi}Hz"] = round(float(10.0 * np.log10((np.sum(Pp[m]) + 1e-20) / (np.sum(Po[m]) + 1e-20))), 2)
    return {
        "orig_ripple_std_db": round(ro, 2), "proc_ripple_std_db": round(rp, 2),
        "orig_cliff_drop_db": round(co, 1), "proc_cliff_drop_db": round(cp, 1),
        "orig_lr_corr": round(lro, 4), "proc_lr_corr": round(lrp, 4),
        "orig_mono_db": round(mo, 2), "proc_mono_db": round(mp, 2),
        "orig_hf_crest_db": round(hco, 2), "proc_hf_crest_db": round(hcp, 2),
        "band_deltas_db": bands,
    }


def parse_ranges(spec):
    return [(float(p.split("-")[0]), float(p.split("-")[1])) for p in spec.split(",")]


def process_track(
    input_path,
    out_dir,
    stems_dir=None,
    target_stem="Synth",
    bpm=None,
    audition_ranges="auto",
    mode="audition",
    skip_stem_reamp=False,
    skip_mix_cleanup=False,
    also_float32=False,
    force_python_lufs=False,
    seed=2026,
):
    """Core standalone processing entry point usable from CLI, batch mode, or local web UI."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    stem_base = os.path.splitext(os.path.basename(input_path))[0]
    lufs_backend = "python-bs1770" if (force_python_lufs or not _HAS_FFMPEG) else "ffmpeg-ebur128"

    print(f"1. Loading original: {input_path} (loudness engine: {lufs_backend})")
    orig, sr = sf.read(input_path, dtype="float64")
    if orig.ndim == 1:
        orig = np.stack([orig, orig], axis=1)
    N = len(orig)

    stems, target_name, drums_stereo = None, None, None
    if stems_dir and os.path.isdir(stems_dir):
        stems, target_name, drums_stereo = read_stems_folder(orig, sr, stems_dir, target_stem)

    bpm_val = float(bpm) if bpm else estimate_bpm(orig, sr, drums_stereo=drums_stereo)
    cliff_hz, cliff_drop = detect_hf_cliff_hz(orig, sr)
    print(f"   {N/sr:.3f}s, {sr} Hz, stereo, BPM={bpm_val:.1f}, detected HF cliff={cliff_hz:.0f} Hz (drop {cliff_drop:.1f} dB)")

    comb_firs = build_linear_phase_comb_fir(orig, sr) if not skip_mix_cleanup else None

    if stems is not None and not skip_stem_reamp:
        print("2. Running 87.5%-overlap exact Wiener split (vocal-guarded) + Lanczos-4 time-domain stem re-amp...")
        xs, xb = split_stems_wiener(orig, sr, stems, target_name)
        s_reamp = reamp_stem_time_domain(xs, sr, bpm_val)
        working = xb + s_reamp
        del xs, xb, s_reamp
    else:
        print("2. Stem re-amp skipped (processing full mix directly).")
        working = orig.copy()

    if mode == "audition":
        if not audition_ranges or audition_ranges.strip().lower() == "auto":
            ranges = select_auto_audition_ranges(orig, sr, stems=stems, target_name=target_name)
            print(f"   Auto-selected audition ranges (s): {ranges}")
        else:
            ranges = parse_ranges(audition_ranges)

        sil_1 = np.zeros((sr, 2), dtype=np.float64)
        sil_2 = np.zeros((2 * sr, 2), dtype=np.float64)
        pieces, report_pairs, cursor = [], [], 0.0

        for idx, (t_a, t_b) in enumerate(ranges):
            s0, s1 = int(round(t_a * sr)), min(N, int(round(t_b * sr)))
            o_seg = orig[s0:s1]
            w_seg = working[s0:s1]
            p_seg = apply_full_mix_cleanup_v2(w_seg, o_seg, sr, comb_firs, cliff_hz, seed=seed + idx) if not skip_mix_cleanup else w_seg.copy()

            tmp_o = os.path.join(out_dir, f"_tmp_o_{idx}.wav")
            tmp_p = os.path.join(out_dir, f"_tmp_p_{idx}.wav")
            write_wav_24bit_tpdf(tmp_o, o_seg, sr, seed=100 + idx)
            write_wav_24bit_tpdf(tmp_p, p_seg, sr, seed=200 + idx)
            lo, _, tpo = measure_lufs_lra_tp(tmp_o, force_python=force_python_lufs)
            lp, _, _ = measure_lufs_lra_tp(tmp_p, force_python=force_python_lufs)
            p_seg *= 10.0 ** ((lo - lp) / 20.0)
            write_wav_24bit_tpdf(tmp_p, p_seg, sr, seed=300 + idx)
            lpm, _, tpp = measure_lufs_lra_tp(tmp_p, force_python=force_python_lufs)
            os.remove(tmp_o)
            os.remove(tmp_p)

            dur = len(o_seg) / sr
            if idx > 0:
                pieces.append(sil_2)
                cursor += 2.0
            t_os = cursor
            pieces.append(apply_edge_fade(o_seg, sr))
            cursor += dur
            pieces.append(sil_1)
            cursor += 1.0
            t_ps = cursor
            pieces.append(apply_edge_fade(p_seg, sr))
            cursor += dur

            report_pairs.append({
                "source_range_s": [t_a, t_b],
                "orig_at_s": [round(t_os, 2), round(t_os + dur, 2)],
                "proc_at_s": [round(t_ps, 2), round(t_ps + dur, 2)],
                "orig_lufs": lo, "proc_lufs": lpm, "orig_tp": tpo, "proc_tp": tpp,
                "qa": spectral_qa(o_seg, p_seg, sr, cliff_hz),
            })

        out_wav = os.path.join(out_dir, f"{stem_base}-Reference-Master-Audition.wav")
        write_wav_24bit_tpdf(out_wav, np.concatenate(pieces, axis=0), sr, seed=seed)
        result = {
            "mode": "audition",
            "output_wav": out_wav,
            "bpm": bpm_val,
            "lufs_backend": lufs_backend,
            "elapsed_s": round(time.time() - t0, 1),
            "pairs": report_pairs,
        }
        print(f"\nWrote audition WAV ({cursor:.2f}s, 24-bit TPDF/{sr}Hz): {out_wav}")
        print(json.dumps(report_pairs, indent=2))
        return result

    print("3. Running zero-STFT-framing contiguous linear-phase mastering chain across full track...")
    proc = apply_full_mix_cleanup_v2(working, orig, sr, comb_firs, cliff_hz, seed=seed) if not skip_mix_cleanup else working.copy()
    del working

    tmp_o = os.path.join(out_dir, "_tmp_full_orig.wav")
    tmp_p = os.path.join(out_dir, "_tmp_full_proc.wav")
    write_wav_24bit_tpdf(tmp_o, orig, sr, seed=11)
    write_wav_24bit_tpdf(tmp_p, proc, sr, seed=22)
    lo, lrao, tpo = measure_lufs_lra_tp(tmp_o, force_python=force_python_lufs)
    lp, _, _ = measure_lufs_lra_tp(tmp_p, force_python=force_python_lufs)
    proc *= 10.0 ** ((lo - lp) / 20.0)
    pk_lin = float(np.max(np.abs(proc)))
    if pk_lin > 0.891:
        proc *= (0.891 / pk_lin)

    out_wav = os.path.join(out_dir, f"{stem_base}-Reference-Analog-Master-24bit-{sr//1000}kHz.wav")
    write_wav_24bit_tpdf(out_wav, proc, sr, seed=seed)
    f32_wav = None
    if also_float32:
        f32_wav = os.path.join(out_dir, f"{stem_base}-Reference-Analog-Master-32f-{sr//1000}kHz.wav")
        sf.write(f32_wav, proc.astype(np.float32), sr, subtype="FLOAT")

    lpm, lrap, tpp = measure_lufs_lra_tp(out_wav, force_python=force_python_lufs)
    os.remove(tmp_o)
    os.remove(tmp_p)

    spot_dir = os.path.join(out_dir, "spot_checks")
    os.makedirs(spot_dir, exist_ok=True)
    total_s = N / sr
    spots = [
        ("intro", min(5.0, total_s * 0.05), 14.0),
        ("hook", min(32.0, total_s * 0.20), 14.0),
        ("verse", min(65.0, total_s * 0.35), 14.0),
        ("bridge", min(145.0, total_s * 0.56), 14.0),
    ]
    spot_files = {}
    for nm, ts, dur in spots:
        a, b = int(round(ts * sr)), min(N, int(round((ts + dur) * sr)))
        sp = os.path.join(spot_dir, f"spot_{nm}.wav")
        sf.write(sp, proc[a:b], sr, subtype="PCM_16")
        spot_files[nm] = sp

    summary = {
        "mode": "full",
        "output_wav": out_wav,
        "float32_wav": f32_wav,
        "spot_checks": spot_files,
        "bpm": bpm_val,
        "lufs_backend": lufs_backend,
        "duration_s": round(N / sr, 3),
        "sample_rate": sr,
        "bit_depth": 24,
        "dither": "1-LSB TPDF",
        "orig_lufs": lo, "proc_lufs": lpm,
        "orig_lra": lrao, "proc_lra": lrap,
        "orig_tp_dbtp": tpo, "proc_tp_dbtp": tpp,
        "any_nan_or_inf": bool(np.isnan(proc).any() or np.isinf(proc).any()),
        "elapsed_s": round(time.time() - t0, 1),
        "qa": spectral_qa(orig, proc, sr, cliff_hz),
    }
    print(f"\nWrote full 24-bit TPDF reference master: {out_wav}")
    print(json.dumps(summary, indent=2))
    return summary


def find_matching_stems_dir(wav_path):
    """Auto-discover a sibling '<TrackName> Stems' folder for batch processing."""
    base = os.path.splitext(wav_path)[0]
    for suffix in (" Stems", "_Stems", " stems", "_stems"):
        cand = base + suffix
        if os.path.isdir(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(
        description="De-AI Audio Master: Standalone Zero-Artifact Audio Cleanup & Linear-Phase Mastering CLI"
    )
    ap.add_argument("--input", required=True, help="Path to original stereo WAV or a folder of WAVs (batch mode)")
    ap.add_argument("--stems-dir", default=None, help="Optional folder of WAV stems (auto-matched in batch mode)")
    ap.add_argument("--target-stem", default="Synth", help="Substring identifying primary melodic stem to re-amp")
    ap.add_argument("--bpm", type=float, default=None, help="Track BPM (auto-estimated from Drums/mix if omitted)")
    ap.add_argument("--audition-ranges", default="auto", help="'auto' (default) or comma-separated start-end seconds, e.g. 5-19,105-119")
    ap.add_argument("--out-dir", default="./deai_out", help="Output directory (default: ./deai_out)")
    ap.add_argument("--mode", choices=["audition", "full"], default="audition", help="audition (default) or full")
    ap.add_argument("--skip-stem-reamp", action="store_true", help="Ablation flag: skip stem re-amp pass")
    ap.add_argument("--skip-mix-cleanup", action="store_true", help="Ablation flag: skip full-mix cleanup pass")
    ap.add_argument("--also-float32", action="store_true", help="Also write an archival 32-bit float WAV in full mode")
    ap.add_argument("--python-lufs", action="store_true", help="Use built-in pure-Python BS.1770-4 loudness instead of ffmpeg")
    ap.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        wavs = sorted(glob.glob(os.path.join(args.input, "*.wav")))
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in folder: {args.input}")
        print(f"Batch mode: found {len(wavs)} WAV file(s) in {args.input}")
        for idx, wav_path in enumerate(wavs, 1):
            name = os.path.splitext(os.path.basename(wav_path))[0]
            stems_dir = args.stems_dir or find_matching_stems_dir(wav_path)
            track_out = os.path.join(args.out_dir, name)
            print(f"\n=== [{idx}/{len(wavs)}] {os.path.basename(wav_path)} (stems: {stems_dir or 'none'}) ===")
            process_track(
                input_path=wav_path,
                out_dir=track_out,
                stems_dir=stems_dir,
                target_stem=args.target_stem,
                bpm=args.bpm,
                audition_ranges=args.audition_ranges,
                mode=args.mode,
                skip_stem_reamp=args.skip_stem_reamp,
                skip_mix_cleanup=args.skip_mix_cleanup,
                also_float32=args.also_float32,
                force_python_lufs=args.python_lufs,
                seed=args.seed,
            )
    else:
        stems_dir = args.stems_dir
        process_track(
            input_path=args.input,
            out_dir=args.out_dir,
            stems_dir=stems_dir,
            target_stem=args.target_stem,
            bpm=args.bpm,
            audition_ranges=args.audition_ranges,
            mode=args.mode,
            skip_stem_reamp=args.skip_stem_reamp,
            skip_mix_cleanup=args.skip_mix_cleanup,
            also_float32=args.also_float32,
            force_python_lufs=args.python_lufs,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
