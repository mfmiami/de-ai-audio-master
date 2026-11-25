# De-AI Audio Master

An agent skill and Python DSP script for cleaning up common technical artifacts in AI-generated tracks (from tools like Suno or Udio) without wrecking the mix. Packaged as a reusable skill (`ai-audio-clean-master`) for Antigravity, Claude Code / Claude Desktop, and OpenAI Codex.

## What this actually fixes (and what it doesn't)

Tracks exported from generative music services usually share a handful of measurable technical artifacts:

- WAV header metadata left by the generator (like Suno's `LIST/INFO/ICMT` chunk). This script writes a plain 44-byte RIFF/WAVE header.
- Stationary narrowband comb resonances between 2.2 kHz and 18.2 kHz from the neural vocoder.
- A steep high-frequency cutoff around 18 to 20 kHz where the codec drops off.
- Overly static high-frequency phase alignment and sterile synth textures.
- Rigid, quantized timing.

What it does **not** touch: lyrics, vocal pronunciation, songwriting, or arrangement. And it does not guarantee any particular score on third-party AI detectors (SubmitHub, Deezer, IRCAM, etc.), since those inspect deeper neural features beyond surface audio artifacts.

## Why the chain is built this way

Earlier attempts using FluidSynth stem swaps, STFT sinusoidal peak tracking (`SMS`), random-phase resynthesis, raw room early reflections, or independent L/R time warping all caused audible problems: watery "MP3 birdie" artifacts on quiet intros, dull treble, or stereo flanging.

To avoid those issues, the script splits the work into two parts:

1. **Optional stem re-amp (if you have stems).** It finds correlated bleed stems (for example, Keyboard or Strings duplicating Synth with `r > 0.35`), shields vocal stems with a `1.4x` weight, and runs an 87.5%-overlap additive Wiener split (`original == target_stem + backing` to `~1e-16` error, with everything below 90 Hz kept in the backing mix). Only the isolated target stem gets 4x-oversampled asymmetric saturation above 120 Hz, a Lanczos-4 quadrature BBD stereo chorus above 900 Hz, and a 4.2% wet low-passed BPM delay tail.
2. **Full-mix pass without STFT framing.** The full mix goes through a 4097-tap Blackman-windowed linear-phase FIR to cut stationary comb peaks (`<= 2.2 dB`, cuts only), an envelope-gated 4x-oversampled harmonic exciter above the detected HF cutoff (gated to zero in quiet gaps), a flat-magnitude 2nd-order allpass cascade above 5.8 kHz, slow `±0.40 dB` tonal drift, stereo-linked Lanczos-8 micro-timing variation (`±0.11%`, identical L/R read pointers so there is no flanging), light M/S 4x-oversampled bus saturation above 120 Hz, and a 2049-tap linear-phase FIR balance trim (`±0.85 dB` cap) so 30 Hz to 16 kHz stays close to the original mix.

Output is matched to the original integrated LUFS via `ffmpeg ebur128` and written as 24-bit PCM with 1-LSB TPDF dither (plus an optional 32-bit float file).

## Repo layout

```text
.
├── ai-audio-clean-master/             # Drop-in skill folder
│   ├── SKILL.md                       # Skill instructions (Antigravity, Claude, Codex)
│   ├── agents/openai.yaml             # Codex UI metadata
│   └── scripts/clean_analog_master.py # Standalone Python DSP CLI
├── dist/
│   └── ai-audio-clean-master-v2-skill.zip
├── guides/
│   ├── antigravity.md
│   ├── claude.md
│   ├── codex.md
│   └── models.md
├── install.sh
├── requirements.txt
└── LICENSE
```

## Setup

You need `ffmpeg` (used for ITU-R BS.1770-4 loudness and true-peak measurement) and Python 3 with `numpy`, `scipy`, and `soundfile`:

```bash
brew install ffmpeg

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install from a cloned repo

```bash
git clone https://github.com/mfmiami/de-ai-audio-master.git
cd de-ai-audio-master

# Install into Antigravity, Claude Code, and Codex at once:
./install.sh --all

# Or pick one:
./install.sh --antigravity
./install.sh --claude
./install.sh --codex
```

### Install from the zip

Grab [`dist/ai-audio-clean-master-v2-skill.zip`](./dist/ai-audio-clean-master-v2-skill.zip):

- **Claude.ai / Claude Desktop**: Settings -> Capabilities -> Skills -> upload the zip directly.
- **Claude Code**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.claude/skills/`
- **Codex**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.codex/skills/`
- **Antigravity**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.gemini/config/skills/`

## Harness guides

- [Antigravity setup and usage](./guides/antigravity.md)
- [Claude Code and Claude Desktop setup](./guides/claude.md)
- [OpenAI Codex setup](./guides/codex.md)
- [Model recommendations](./guides/models.md)

### Quick model summary

| Harness | Recommended | Faster / cheaper option | Can listen to `.wav` files directly? |
| :--- | :--- | :--- | :---: |
| **Antigravity** | Gemini 3 Pro / Gemini 2.5 Pro | Gemini 3 Flash / Gemini 2.5 Flash | Yes |
| **Claude Code / Desktop** | Claude Opus 4.6 | Claude Sonnet 4.6 | No (you listen to the audition WAV) |
| **OpenAI Codex** | GPT-5.3-Codex / GPT-5.4 | GPT-5-mini / o4-mini | No (you listen to the audition WAV) |

See [guides/models.md](./guides/models.md) for details on when each model makes sense.

## Running the script directly

You don't need an agent to run the DSP script. Always do an audition pass first so you can compare the original and processed sections at matched loudness:

```bash
# 1. Build a loudness-matched A/B audition WAV
python3 ai-audio-clean-master/scripts/clean_analog_master.py \
  --input "/path/to/track.wav" \
  --stems-dir "/path/to/stems" \
  --target-stem "Synth" \
  --audition-ranges "5-19,45-59,110-124" \
  --out-dir "./out/audition" \
  --mode audition

# 2. Render the full track (omit --stems-dir if you only have a stereo mix)
python3 ai-audio-clean-master/scripts/clean_analog_master.py \
  --input "/path/to/track.wav" \
  --stems-dir "/path/to/stems" \
  --target-stem "Synth" \
  --bpm 128 \
  --out-dir "./out/full" \
  --mode full \
  --also-float32
```

If something sounds off on a specific track, use the ablation flags to find out which pass caused it:

- `--skip-stem-reamp`: runs only the full-mix pass
- `--skip-mix-cleanup`: runs only the stem re-amp pass

## License

MIT. See [LICENSE](./LICENSE).
