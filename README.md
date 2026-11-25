# AI Audio Clean Master (v2)

A reusable, zero-artifact, **reference-audiophile time-domain audio cleanup and linear-phase analog mastering skill** for **Antigravity**, **Claude Code / Claude Desktop**, and **OpenAI Codex**.

Built to preserve original studio tonal balance, vocal clarity, stereo image, and transient punch (`±0.08 dB` across `30 Hz – 16 kHz`) while reducing measurable AI-audio artifacts (stationary vocoder comb ripple, high-frequency bandwidth cliff, static HF phase locks, rigid machine timing, and leftover generator WAV metadata).

---

## Why This Pipeline Exists

Common audio-cleanup workarounds—General MIDI/FluidSynth stem swaps, STFT sinusoidal peak-tracking (`SMS`), broadband random-phase resynthesis, raw room slap, or independent-channel time warping—often introduce watery "MP3 birdie" artifacts, dull treble, phase flanging, or mix balance drift on quiet intros and vocal bridges.

**AI Audio Clean Master v2** eliminates those failure modes by keeping the entire mastering chain **zero-STFT-framing and strictly phase-/stereo-coherent**:

1. **87.5%-Overlap Exact Additive Wiener Stem Split (Vocal-Guarded)** — Auto-detects correlated bleed stems (e.g., Keyboard/Strings duplicating Synth with `r > 0.35`), protects vocal stems with a `1.4×` guard, protects sub-90 Hz in the backing mix, and splits `original == target_stem + backing` to floating-point precision (`~1e-16`).
2. **Pure Time-Domain Target-Stem Re-Amp** — Sub-protected (`>120 Hz`) Mid/Side 4×-oversampled Class-A asymmetric triode/tape warmth → 8-tap **Lanczos-4** quadrature Juno-style BBD stereo chorus above `900 Hz` (zero treble loss) → subtle BPM-synced warm LP analog delay tail (`4.2%` wet) → exact RMS restore.
3. **Zero-STFT-Framing Contiguous Full-Mix Mastering Chain**:
   - **4097-tap Blackman-windowed Linear-Phase FIR** (`0.000°` phase) for conservative track-wide stationary comb flattening (`2.2–18.2 kHz`, cuts `<= 2.2 dB` only where both mean and 25th-percentile floor show excess, with smooth broadband energy preservation).
   - **Envelope-following 4×-oversampled harmonic air exciter** above the detected HF bandwidth cliff (silky natural air that gates cleanly to zero during quiet gaps).
   - **2nd-order analog allpass cascade** above `5.8 kHz` (`0.000 dB` flat magnitude, gentle L/R phase dispersion).
   - **Slow time-domain biquad mastering tonal breath** (`±0.40 dB` across 4 bands).
   - **Stereo-linked Lanczos-8 (16-tap) organic micro-tempo breath** (`±0.11%` max rate, identical L/R read positions — zero flanging).
   - **Sub-protected (`>120 Hz`) Mid/Side 4×-oversampled Class-A bus sheen** (`11%` wet) + **2049-tap Linear-Phase FIR reference-sparkle balance trim** (`±0.85 dB` max) + `-82 dBFS` 1/f natural analog floor.
4. **Broadcast-Grade Export & QA**: Strict ITU-R BS.1770-4 (`ffmpeg ebur128`) LUFS matching, clean 44-byte RIFF/WAVE header (zero leftover source metadata), **24-bit PCM with 1-LSB TPDF dither** (plus optional `--also-float32` archival export), HF transient crest-factor verification (`6–16 kHz`), and 4 automatic spot-check clips.

---

## Repository Layout

```text
.
├── ai-audio-clean-master/             # Universal drop-in skill folder
│   ├── SKILL.md                       # Skill instructions (Antigravity, Claude, Codex)
│   ├── agents/openai.yaml             # Codex UI display metadata
│   └── scripts/clean_analog_master.py # Self-contained Audiophile v2 DSP CLI
├── dist/
│   └── ai-audio-clean-master-v2-skill.zip  # Pre-packaged zip (ready for Claude.ai/Desktop upload)
├── guides/
│   ├── antigravity.md                 # Detailed Antigravity setup & workflow guide
│   ├── claude.md                      # Detailed Claude Code & Claude Desktop/Web guide
│   ├── codex.md                       # Detailed OpenAI Codex CLI & App guide
│   └── models.md                      # Model recommendations & capability matrix
├── install.sh                         # One-command installer for all harnesses
├── requirements.txt                   # Python dependencies
└── LICENSE
```

---

## Quick Install

### Prerequisites
Install Python audio dependencies and `ffmpeg` (for ITU-R BS.1770-4 `ebur128` loudness and true-peak verification):

```bash
# macOS (Homebrew)
brew install ffmpeg

# Python dependencies (recommended inside a venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option A: One-Command Installer
```bash
git clone https://github.com/mfmiami/ai-audio-clean-master.git
cd ai-audio-clean-master

# Install into all three harnesses at once:
./install.sh --all

# Or install selectively:
./install.sh --antigravity
./install.sh --claude
./install.sh --codex
```

### Option B: Direct ZIP Install (No Git Required)
Download [`dist/ai-audio-clean-master-v2-skill.zip`](./dist/ai-audio-clean-master-v2-skill.zip):

- **Claude.ai / Claude Desktop**: Go to **Settings → Capabilities → Skills** and upload `ai-audio-clean-master-v2-skill.zip` directly.
- **Claude Code**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.claude/skills/`
- **Codex**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.codex/skills/`
- **Antigravity**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.gemini/config/skills/`

---

## Detailed Harness Guides

| Harness | Install Path / Method | Detailed Guide |
| :--- | :--- | :--- |
| **Google Antigravity** | `~/.gemini/config/skills/ai-audio-clean-master/` (or `<workspace>/.agents/skills/`) | [guides/antigravity.md](./guides/antigravity.md) |
| **Claude Code / Claude Desktop / Claude.ai** | `~/.claude/skills/ai-audio-clean-master/` or ZIP upload in Settings | [guides/claude.md](./guides/claude.md) |
| **OpenAI Codex (CLI & App)** | `~/.codex/skills/ai-audio-clean-master/` (`$ai-audio-clean-master`) | [guides/codex.md](./guides/codex.md) |
| **Model Selection & Tuning** | Recommended models per harness & audio-auditioning matrix | [guides/models.md](./guides/models.md) |

---

## Recommended Models at a Glance

| Harness | Best Overall | Fast / Everyday Runs | Native Audio Ear-Check? |
| :--- | :--- | :--- | :---: |
| **Antigravity** | **Gemini 3 Pro** / **Gemini 2.5 Pro** | **Gemini 3 Flash** / **Gemini 2.5 Flash** | **Yes** (agent listens to `.wav` clips natively) |
| **Claude Code / Desktop** | **Claude Opus 4.6** | **Claude Sonnet 4.6** | No — relies on QA metrics + your ear check on the audition WAV |
| **OpenAI Codex** | **GPT-5.3-Codex** / **GPT-5.4** | **GPT-5-mini** / **o4-mini** | No — relies on QA metrics + your ear check on the audition WAV |

See [guides/models.md](./guides/models.md) for detailed trade-offs, when to upgrade models, and how to pair non-audio models with local auditioning.

---

## Direct CLI Usage (Without an Agent)

You can also invoke the bundled DSP script directly from your terminal:

```bash
# Step 1: Build a loudness-matched A/B audition WAV first
python3 ai-audio-clean-master/scripts/clean_analog_master.py \
  --input "/path/to/track.wav" \
  --stems-dir "/path/to/stems" \
  --target-stem "Synth" \
  --audition-ranges "5-19,45-59,110-124" \
  --out-dir "./out/audition" \
  --mode audition

# Step 2: Render the full-length 24-bit TPDF reference master (omit --stems-dir if no stems)
python3 ai-audio-clean-master/scripts/clean_analog_master.py \
  --input "/path/to/track.wav" \
  --stems-dir "/path/to/stems" \
  --target-stem "Synth" \
  --bpm 128 \
  --out-dir "./out/full" \
  --mode full \
  --also-float32
```

### Ablation Flags
If you ever hear an issue on a specific track, isolate the stage immediately:
- `--skip-stem-reamp` — tests the full-mix linear-phase mastering chain alone
- `--skip-mix-cleanup` — tests the isolated Wiener stem re-amp alone

---

## Non-Negotiable Principles & Detector Honesty
- **Always audition before full render**: Run `--mode audition` first and check exposed quiet intros/bridges alongside dense choruses.
- **Honest QA reporting**: Local metrics (`ebur128`, spectral ripple std, HF cliff drop, `6–16 kHz` transient crest, L/R correlation, mono-sum delta, band deltas) are strictly acoustic measurements. No local tool can guarantee a third-party detector score (such as SubmitHub)—always test the exported WAV directly if targeting an external checker.

## License
MIT — see [LICENSE](./LICENSE).
