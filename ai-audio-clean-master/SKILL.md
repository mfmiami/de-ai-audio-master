---
name: ai-audio-clean-master
description: >-
  Clean and naturalize AI-generated audio tracks using a zero-artifact,
  reference-audiophile time-domain analog stem re-amping and zero-STFT-framing
  linear-phase mastering chain (87.5%-overlap vocal-guarded exact additive
  Wiener stem split, sub-protected M/S 4x oversampled Class-A saturation,
  Lanczos-4 Juno-style BBD chorus, 4097-tap linear-phase FIR stationary comb
  flattening + reference-sparkle trim, envelope-gated harmonic air exciter,
  analog allpass cascade, stereo-linked Lanczos-8 timing breath, and 1-LSB TPDF
  dither). Use when the user asks to clean, naturalize, de-artifact, or master
  an electronic or vocal track to preserve studio sound quality while reducing
  measurable AI-audio signatures and stripping source metadata.
metadata:
  short-description: Zero-artifact audiophile time-domain cleanup & linear-phase analog mastering
---

# PurePhase Master (AI Audio Clean Master)

Clean and naturalize a stereo mix (with or without stems) so it sounds virtually
indistinguishable from the original (zero watery artifacts, zero mix balance
drift, zero chunk seams) while reducing measurable AI-audio artifacts (stationary
vocoder comb, high-frequency bandwidth cliff, static high-frequency phase locks,
rigid machine timing, and leftover generator metadata).

## Non-Negotiable Rules

1. **Audition first.** If you cannot access or directly listen to the audio, stop and tell the user before processing anything.
2. **Never use watery spectral resynthesis or cheap MIDI swaps.** Do not use General MIDI / FluidSynth presets, STFT sinusoidal peak-tracking (`SMS`), broadband random-phase resynthesis, or STFT octave bin-copying above the HF cliff, because these create watery "MP3 birdie" or metallic artifacts on exposed sections.
3. **Preserve stereo coherence and dynamics.** Never apply independent-channel time-warping (causes L/R phase drift/flanging), raw early-reflection room slap, chunked crossfade seams, or brickwall loudness slamming.
4. **Stay honest about third-party detectors.** Never claim a SubmitHub or other external AI-detector score unless the user actually tests the exported file and reports the result. Report any local metrics separately.

## Workflow

### Step 1: Gather Inputs & Verify Environment
Collect from the user (or infer from the workspace):
- Original stereo mix WAV (`--input`)
- Optional stems directory (`--stems-dir`) and target stem keyword (`--target-stem`, e.g., `Synth`)
- One exposed melodic/hook range and one dense vocal/drum chorus range (`--audition-ranges`, e.g., `32-46,110-122`)
- Whether to pause for A/B audition approval first (`--mode audition`, default) or render the full master directly (`--mode full`)

Verify Python (`numpy`, `scipy`, `soundfile`) and `ffmpeg` (for ITU-R BS.1770-4 `ebur128` loudness/true-peak checks) are available.

### Step 2: Run the Audition Pass First (Default)
Run the bundled self-contained DSP script [scripts/clean_analog_master.py](./scripts/clean_analog_master.py):

```bash
python3 ./scripts/clean_analog_master.py \
  --input "/path/to/original.wav" \
  --stems-dir "/path/to/stems" \
  --target-stem "Synth" \
  --audition-ranges "32-46,110-122" \
  --out-dir "/path/to/output" \
  --mode audition
```

Omit `--stems-dir` if no stems exist (the script runs the contiguous full-mix chain directly).

What the script enforces automatically:
- **Exact reference-preserving stem split**: detects correlated duplicate stems (e.g., Synth bleeding into Keyboard with `r > 0.35`) so nothing is double-counted, applies an explicit `1.4×` vocal-protection guard to any vocal stems, and performs an 87.5%-overlap exact additive complex Wiener split (`original == backing + target_stem`, max reconstruction error `~1e-15`) with sub-90 Hz protected in `backing`.
- **Time-domain stem re-amp** (target stem only): sub-protected (`>120 Hz`) Mid/Side 4×-oversampled Class-A asymmetric triode/tape saturation -> pure time-domain 8-tap **Lanczos-4** quadrature Juno-style BBD stereo chorus above `900 Hz` (zero treble loss) -> subtle BPM-synced warm LP analog delay tail (`4.2%` wet, `2.8 kHz` LP) -> exact RMS restore.
- **Zero-STFT-framing contiguous full-mix mastering chain**:
  - Track-wide conservative stationary comb flattening (`2.2–18.2 kHz`, cuts `<= 2.2 dB` only where both mean and 25th-percentile floor show excess) realized via an exact symmetric **4097-tap Blackman-windowed Linear-Phase FIR** convolution (`0.000°` phase, zero STFT framing) with smooth broadband energy preservation.
  - Envelope-following pure time-domain 4×-oversampled harmonic air exciter above the detected bandwidth cliff (silky natural air, gated cleanly to zero during quiet gaps).
  - Pure time-domain 2nd-order analog allpass cascade above `5.8 kHz` (`0.000 dB` flat magnitude, slight L/R offset).
  - Subtle time-domain slow biquad mastering tonal breath (`±0.40 dB` across 4 bands).
  - Stereo-linked **Lanczos-8** (16-tap) organic micro-tempo breath (`±0.11%` max rate, `1.2 s` smoothing, identical L/R read positions).
  - Sub-protected (`>120 Hz`) Mid/Side 4×-oversampled Class-A mastering bus sheen (`11%` wet) + ultra-smooth **2049-tap Linear-Phase FIR** reference-sparkle balance trim (`±0.85 dB` max, keeping `30 Hz – 16 kHz` flat within `±0.08 dB` of original) + `-82 dBFS` 1/f natural analog floor.
- **Strict LUFS-matched 24-bit TPDF export**: writes clean 44-byte 24-bit PCM WAVs with true 1-LSB triangular-PDF (TPDF) dither (plus optional `--also-float32` archival float export in `full` mode). In `audition` mode, exports `[Original 1] -> 1s silence -> [Processed 1] -> 2s silence -> [Original 2] -> 1s silence -> [Processed 2]`, loudness-matched per pair via `ffmpeg ebur128`.

### Step 3: Mandatory Ear Check & Ablation Gate
Directly audition the exported clips (especially any quiet/exposed intro, hook, and vocal chorus). If you hear any dullness, phase shift, or artifacts:
1. Run an ablation (`--skip-stem-reamp` vs. `--skip-mix-cleanup`) to isolate which stage caused it.
2. Adjust only the offending stage and re-audition before sharing with the user.

Present the comparison WAV, exact playback order, and before/after table, then wait for user approval before running the full render.

### Step 4: Render & QA-Verify the Full-Length 24-Bit Master
Once approved (or when full render is explicitly requested), run:

```bash
python3 ./scripts/clean_analog_master.py \
  --input "/path/to/original.wav" \
  --stems-dir "/path/to/stems" \
  --target-stem "Synth" \
  --out-dir "/path/to/output" \
  --mode full
```

Before delivering the full-length master, verify and report:
- Format & header: original sample rate, 24-bit stereo PCM WAV (1-LSB TPDF dither), exact duration/sample count, clean 44-byte RIFF/WAVE header (zero leftover generator metadata).
- Dynamics & headroom: matched integrated LUFS, preserved LRA, true peak safely `<= -1.0 dBTP` (or matching original headroom if lower), preserved HF transient crest factor (`6–16 kHz`), zero clips/NaNs/Infs.
- Stereo & tonal fidelity: L/R correlation, mono-sum level delta, band deltas (`30 Hz – 22 kHz`), stationary ripple std reduction, and HF cliff drop reduction.
- Direct ear check on at least 4 spot clips written by the script (quiet intro, main hook, verse/chorus, bridge/outro).
