# Model Recommendations & Harness Capability Matrix

Not every model brings the same strengths to audio cleanup and mastering. This workflow mixes two distinct capabilities:

1. **Multimodal Audio Ear-Checking**: Actually *listening* to rendered `.wav` clips to catch watery artifacts, vocal coloration, metallic ringing, or hiss in quiet gaps.
2. **Scientific Python & DSP Engineering**: Auditing stem correlations, writing/tuning `scipy.signal` linear-phase FIR filters, verifying ITU-R BS.1770-4 loudness/crest metrics, and running clean ablations.

---

## Quick Recommendation Matrix

| Harness | Primary (Best Fidelity & Reasoning) | Secondary (Fast / Budget) | Native `.wav` Ear-Check? | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Antigravity** | **Gemini 3 Pro** / **Gemini 2.5 Pro** | **Gemini 3 Flash** / **Gemini 2.5 Flash** | **Yes** | Best end-to-end choice: agent listens directly to original, audition, and 4 full-track spot clips via `read_file`. |
| **Claude Code** | **Claude Opus 4.6** | **Claude Sonnet 4.6** | No | Outstanding DSP/Python precision and ablation discipline; pair with your own ears on `--mode audition`. |
| **Claude Desktop / Web** | **Claude Opus 4.6** | **Claude Sonnet 4.6** | No | Best for single stereo bounces or reviewing scripts; use Claude Code locally for large 48 kHz multi-stem folders. |
| **OpenAI Codex (CLI / App)** | **GPT-5.3-Codex** / **GPT-5.4** | **GPT-5-mini** / **o4-mini** | No | Fast, disciplined terminal execution and clean DSP refactors; pair with your own ears on `--mode audition`. |

---

## Choosing the Right Model for Your Task

### 1. "I just want clean, reliable masters of my tracks using the v2 pipeline"
- **Antigravity**: **Gemini 3 Flash** or **Gemini 2.5 Pro** (Pro if you want the sharpest ear-check on subtle vocal/reverb tails).
- **Claude Code**: **Claude Sonnet 4.6** (fast, follows `SKILL.md` gates reliably, low cost).
- **Codex**: **GPT-5.3-Codex** (or **GPT-5-mini** for straightforward runs).

### 2. "A track has tricky stem bleed, unusual vocals, or needs custom DSP tuning / ablations"
- **Antigravity**: **Gemini 3 Pro** combines deep DSP diagnosis with direct audio listening across ablation clips (`--skip-stem-reamp` vs `--skip-mix-cleanup`).
- **Claude Code**: **Claude Opus 4.6** is the top choice when writing custom Python diagnostics (e.g., kick-band BPM detection, custom frequency-band guards, or alternative target stems like Guitar/Piano).
- **Codex**: **GPT-5.3-Codex** / **GPT-5.4** provide strong numerical Python debugging and clean script edits.

### 3. "I want to prevent watery artifacts automatically before I even listen"
- Use **Antigravity with Gemini 3 Pro / Gemini 2.5 Pro**. Because it directly auditions `10–16 s` WAV slices with `read_file`, it can hear subtle artifacts on quiet intros and exposed vocal bridges immediately and run ablations before handing you a render.

---

## Universal Guardrails (Regardless of Model)

No matter which harness or model you choose:
1. **Never skip `--mode audition`** on a new song. Even identical DSP settings behave differently on sparse acoustic ballads vs. dense EDM/trance drops.
2. **Always check all 5 quantitative QA red flags**:
   - `proc_lufs` matches `orig_lufs` (`±0.1 LU`)
   - `proc_lra` matches `orig_lra` (`±0.2 LU`)
   - `band_deltas_db` stays flat within `±0.20 dB` across `30 Hz – 16 kHz`
   - `proc_hf_crest_db` stays within `±0.35 dB` of `orig_hf_crest_db` (confirms crisp transients/cymbals)
   - `audio data file offset: 44` (confirms zero leftover source metadata chunks)
3. **Trust your own ears last**: Every human playback system reveals different details, so do a quick A/B on headphones or studio monitors before releasing a track.
