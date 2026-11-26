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

# De-AI Audio Master (`ai-audio-clean-master`)

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
- Original stereo mix WAV (`--input`, or a folder of WAVs for batch mode)
- Optional stems directory (`--stems-dir`)
- Genre profile (`--profile auto` by default, which picks `rock` when Guitar/Drums/Vocals dominate over Synth and `electronic` otherwise; or pass `--profile electronic` / `--profile rock` explicitly)
- Optional `--varispeed-pct` (defaults to `0.0` for `electronic` and `+1.15` for `rock`; pass `--varispeed-pct 0` if exact original pitch/duration is needed for video sync or DAW stem alignment)
- Optional audition ranges (`--audition-ranges auto` by default, which picks an exposed opening, strongest hook, and densest chorus; or pass explicit ranges like `32-46,110-122`)
- Whether to pause for A/B audition approval first (`--mode audition`, default) or render the full master directly (`--mode full`)

Verify Python (`numpy`, `scipy`, `soundfile`) is available (or use `uv run`). `ffmpeg` is optional: when installed, the script uses `ffmpeg ebur128` for ITU-R BS.1770-4 loudness/true-peak checks; otherwise it falls back automatically to its built-in pure-Python K-weighted BS.1770-4 engine.

### Step 2: Run the Audition Pass First (Default)
Run the bundled self-contained DSP script [scripts/clean_analog_master.py](./scripts/clean_analog_master.py):

```bash
python3 ./scripts/clean_analog_master.py \
  --input "/path/to/original.wav" \
  --stems-dir "/path/to/stems" \
  --profile auto \
  --audition-ranges "auto" \
  --out-dir "/path/to/output" \
  --mode audition
```

Omit `--stems-dir` if no stems exist (the script runs the contiguous full-mix chain directly; pass `--profile rock` explicitly for stemless rock/band tracks).

What the script enforces automatically:
- **Profile `electronic`**:
  - 87.5%-overlap exact additive 2-way Wiener stem split (`original == backing + target_stem`, error `~1e-15`) with correlated-bleed projection (`r > 0.35`), `1.4x` vocal guard, and sub-90 Hz protected in backing.
  - Target-stem re-amp: sub-protected (`>120 Hz`) M/S 4x-oversampled Class-A saturation -> 8-tap Lanczos-4 BBD stereo chorus above `900 Hz` -> `4.2%` warm BPM delay -> exact RMS restore.
  - Zero-STFT-framing full-mix chain: 4097-tap linear-phase FIR comb flattening (`<= 2.2 dB` cuts only), envelope-gated 4x harmonic air exciter above the detected HF cliff, analog allpass cascade (`>5.8 kHz`), slow tonal breath (`±0.40 dB`), stereo-linked Lanczos-8 micro-tempo breath, M/S Class-A bus sheen, and 2049-tap linear-phase reference balance trim.
- **Profile `rock`** (for guitar, live-drum, and vocal-driven tracks):
  - 87.5%-overlap exact additive 4-way Wiener partition (`Vocals + Drums + Guitar + Bass/Rest == original`, error `~1e-12`, sub-85 Hz kept in Drums + Bass).
  - Per-stem studio rock chains: Vocals (dynamic de-ess `6.5-11.5 kHz`, U47/Neve tone shaping, 4x tube preamp, `5%` diffused stereo studio plate, vocal allpass), Drums (`20%` New York parallel tape drum bus, `6.5%` stereo drum room, cymbal allpass), Guitar (minimum-phase 4x12 cab curve with `7.8 kHz` speaker cone rolloff, Class-A tube warmth, subtle Lanczos-4 width), Bass (`5.2 kHz` LP stripping ultrasonic shelf and 50 Hz codec spikes + mid-growl).
  - Full-mix rock chain: 4097-tap linear-phase comb FIR (`<= 3.0 dB` cuts only), 15-ips tape rolloff (`12.2 kHz`) + envelope-gated 4x analog HF harmonic regeneration, 5-stage allpass dispersion, `4.2%` tight shared live-room glue, gentle 3-band analog bus glue, `+1.15%` tape varispeed (configurable via `--varispeed-pct`) + human drummer phrase/bar pocket via Lanczos-8, M/S Class-A sheen, and `2%` analog console crosstalk.
- **Strict LUFS-matched 24-bit TPDF export**: writes clean 44-byte 24-bit PCM WAVs with true 1-LSB triangular-PDF (TPDF) dither (plus optional `--also-float32` archival export in `full` mode). In `audition` mode, exports `[Original 1] -> 1s silence -> [Processed 1] -> 2s silence -> [Original 2] -> 1s silence -> [Processed 2]`, loudness-matched per pair.

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
