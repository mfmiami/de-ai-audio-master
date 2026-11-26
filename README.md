# De-AI Audio Master

> **Work in progress:** This tool is actively being developed and has so far been tested on electronic/synth tracks and guitar-driven rock tracks. More testing and tuning will be needed across other genres (solo acoustic, hip-hop, jazz, orchestral, etc.), so always run the audition pass first and trust your ears.

A standalone, 100% local Python DSP tool for cleaning up common technical artifacts in AI-generated tracks (from tools like Suno or Udio) without wrecking the mix.

It runs completely offline on CPU with only `numpy`, `scipy`, and `soundfile` (`ffmpeg` is optional). You can use it as a standalone CLI, a zero-install `uv` one-liner, a local drag-and-drop browser UI, or optionally pair it with an AI coding agent using the included skill for Antigravity, Claude Code / Claude Desktop, and OpenAI Codex.

## Run it standalone (no AI agent required)

Every audio step happens locally on your machine. Nothing is uploaded anywhere, and there are zero API or token costs.

### Option 1: Zero-install one-liner with `uv`

Because the script embeds PEP 723 dependency metadata, you don't even have to clone the repo or create a virtualenv:

```bash
# Audition pass first (--profile auto detects electronic vs rock from stems)
uv run https://raw.githubusercontent.com/mfmiami/de-ai-audio-master/main/ai-audio-clean-master/scripts/clean_analog_master.py \
  --input "/path/to/track.wav" \
  --stems-dir "/path/to/stems" \
  --profile auto \
  --mode audition

# Full 24-bit master (omit --stems-dir if you only have a stereo bounce; pass --profile rock for stemless rock tracks)
uv run https://raw.githubusercontent.com/mfmiami/de-ai-audio-master/main/ai-audio-clean-master/scripts/clean_analog_master.py \
  --input "/path/to/track.wav" \
  --stems-dir "/path/to/stems" \
  --profile auto \
  --mode full
```

### Option 2: Installable CLI + local web UI (`pipx` or `pip`)

```bash
pipx install git+https://github.com/mfmiami/de-ai-audio-master.git
# or inside a venv:
# pip install git+https://github.com/mfmiami/de-ai-audio-master.git
```

That gives you two commands:

```bash
# 1. Standalone CLI (single track OR a whole folder of WAVs in batch mode)
deai-master --input "/path/to/track.wav" --stems-dir "/path/to/stems" --profile auto --mode audition
deai-master --input "/path/to/album_folder" --profile auto --mode full --also-float32

# 2. Local browser UI (100% offline, zero extra dependencies)
deai-master-ui --open
```

The local web UI starts on `http://127.0.0.1:8765`, lets you pick a WAV, optional stems, and profile (`Auto`, `Electronic / Synth`, or `Rock / Band`), plays the loudness-matched A/B audition right in the browser with jump buttons for each pair, prints the QA table, and exports the final 24-bit WAV.

### Profiles (`--profile auto | electronic | rock`)

Different genres carry AI artifacts in different places, so the engine ships with two 8x-oversampled (384 kHz internal) profiles (pass `--oversample 4` if you want faster batch CPU runs):

- **`electronic`**: Built for synth- and pad-driven tracks. Uses a 2-way vocal-guarded Wiener split (re-amping the dominant synth/bleed stem with 8x Class-A warmth, 12-tap Lanczos-6 BBD chorus, and subtle BPM delay), 4097-tap linear-phase comb flattening, 8x-oversampled analog air above `15.2 kHz` (keeping `4 to 15 kHz` sparkling open), elliptical `<95 Hz` sub-mono tightening + M/S stereo bloom (`+0.75 dB` Side `>350 Hz`), a subtle 2049-tap linear-phase mastering contour (`+0.45 dB @ 55 Hz`, `-0.55 dB @ 240 Hz`, `+0.30 dB @ 9.5 kHz`), and `1.2%` console crosstalk. Default varispeed: `0.0%`.
- **`rock`**: Built for guitar-, live-drum-, and vocal-driven tracks where vocals and drums carry most of the mix energy. Uses a 4-way exact additive Wiener partition (`Vocals + Drums + Guitar + Bass/Rest`), applying 8x-oversampled per-stem rock chains (U47/Neve vocal de-ess/warmth + `5%` studio plate, `20%` New York parallel tape drum bus + `6.5%` drum room, minimum-phase 4x12 guitar cab with `7.8 kHz` cone rolloff, `5.2 kHz` bass LP), followed by 15-ips tape rolloff (`10.5 kHz`) + 8x-oversampled envelope-gated analog HF harmonic regeneration, tight `4.2%` shared live-room glue, `1.8:1` 3-band bus glue, `-1.1 dB @ 180 Hz` de-mud cleanup, elliptical `<105 Hz` sub-mono + M/S bloom (`+0.70 dB` Side `>300 Hz`), `2%` console crosstalk, and `+1.15%` analog tape varispeed + human drummer phrase/bar pocket. Pass `--varispeed-pct 0` if you need exact original pitch and duration for video sync or DAW stem alignment.
- **`auto` (default)**: Inspects stem RMS levels when `--stems-dir` is given and picks `rock` when Guitar/Drums/Vocals outweigh Synth, or `electronic` otherwise. Without stems, `auto` defaults to `electronic` (pass `--profile rock` explicitly for stereo-only rock tracks).

### Built-in portability details

- **No `ffmpeg` required**: Uses `ffmpeg ebur128` when available, and automatically falls back to a built-in pure-Python ITU-R BS.1770-4 K-weighted LUFS/LRA + 4x-oversampled True Peak engine when `ffmpeg` isn't installed (or when `--python-lufs` is passed).
- **Auto audition ranges**: `--audition-ranges auto` (the default) scans the track and picks an exposed opening, the strongest melodic/target-stem hook, and the densest chorus automatically.
- **Kick-band BPM detection**: When a Drums stem is present, BPM is measured from the 40 to 140 Hz kick band so the subtle analog delay tail locks to the real tempo instead of a subharmonic.
- **Batch folder mode**: Passing a folder to `--input` processes every `.wav` inside and automatically pairs sibling `<TrackName> Stems/` directories when they exist.

## What this actually fixes (and what it doesn't)

Tracks exported from generative music services usually share a handful of measurable technical artifacts:

- WAV header metadata left by the generator (like Suno's `LIST/INFO/ICMT` chunk). This script writes a plain 44-byte RIFF/WAVE header.
- Stationary narrowband comb resonances between 2.2 kHz and 18.2 kHz from the neural vocoder.
- A steep high-frequency cutoff around 18 to 20 kHz where the codec drops off.
- Overly static high-frequency phase alignment and sterile synth textures.
- Rigid, quantized timing.

What it does **not** touch: lyrics, vocal pronunciation, songwriting, or arrangement. And it does not guarantee any particular score on third-party AI detectors (SubmitHub, Deezer, IRCAM, etc.), since those inspect deeper neural features beyond surface audio artifacts.

## Why the DSP chain is built this way

Earlier attempts using FluidSynth stem swaps, STFT sinusoidal peak tracking (`SMS`), random-phase resynthesis, raw room early reflections, or independent L/R time warping all caused audible problems: watery "MP3 birdie" artifacts on quiet intros, dull treble, or stereo flanging.

To avoid those issues, the script splits the work into two parts:

1. **Optional stem re-amp (if you have stems).** It finds correlated bleed stems (for example, Keyboard or Strings duplicating Synth with `r > 0.35`), shields vocal stems with a `1.4x` weight, and runs an 87.5%-overlap additive Wiener split (`original == target_stem + backing` to `~1e-16` error, with everything below 90 Hz kept in the backing mix). Only the isolated target stem gets 4x-oversampled asymmetric saturation above 120 Hz, a Lanczos-4 quadrature BBD stereo chorus above 900 Hz, and a 4.2% wet low-passed BPM delay tail.
2. **Full-mix pass without STFT framing.** The full mix goes through a 4097-tap Blackman-windowed linear-phase FIR to cut stationary comb peaks (`<= 2.2 dB`, cuts only), an envelope-gated 4x-oversampled harmonic exciter above the detected HF cutoff (gated to zero in quiet gaps), a flat-magnitude 2nd-order allpass cascade above 5.8 kHz, slow `±0.40 dB` tonal drift, stereo-linked Lanczos-8 micro-timing variation (`±0.11%`, identical L/R read pointers so there is no flanging), light M/S 4x-oversampled bus saturation above 120 Hz, and a 2049-tap linear-phase FIR balance trim (`±0.85 dB` cap) so 30 Hz to 16 kHz stays close to the original mix.

Output is matched to the original integrated LUFS and written as 24-bit PCM with 1-LSB TPDF dither (plus an optional 32-bit float file). If something sounds off on a specific track, use `--skip-stem-reamp` or `--skip-mix-cleanup` to isolate which pass caused it.

## Optional: Use with an AI agent (Antigravity, Claude, Codex)

You don't need an LLM to run the audio engine, but an AI agent is genuinely helpful when you want hands-off track auditing, custom ablation runs on tricky mixes, DSP tweaks, or (in Antigravity with Gemini) having the agent directly listen to spot clips to catch subtle vocal or intro artifacts before you hear them.

### Install the agent skill

From a cloned repo:

```bash
git clone https://github.com/mfmiami/de-ai-audio-master.git
cd de-ai-audio-master

./install.sh --all          # Antigravity + Claude Code + Codex
# or selectively:
./install.sh --antigravity
./install.sh --claude
./install.sh --codex
```

Or from [`dist/ai-audio-clean-master-v2-skill.zip`](./dist/ai-audio-clean-master-v2-skill.zip):

- **Claude.ai / Claude Desktop**: Settings -> Capabilities -> Skills -> upload the zip directly.
- **Claude Code**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.claude/skills/`
- **Codex**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.codex/skills/`
- **Antigravity**: `unzip ai-audio-clean-master-v2-skill.zip -d ~/.gemini/config/skills/`

### Harness guides & model recommendations

- [Antigravity setup and usage](./guides/antigravity.md)
- [Claude Code and Claude Desktop setup](./guides/claude.md)
- [OpenAI Codex setup](./guides/codex.md)
- [Model recommendations](./guides/models.md)

| Harness | Recommended | Faster / cheaper option | Can listen to `.wav` files directly? |
| :--- | :--- | :--- | :---: |
| **Antigravity** | Gemini 3 Pro / Gemini 2.5 Pro | Gemini 3 Flash / Gemini 2.5 Flash | Yes |
| **Claude Code / Desktop** | Claude Opus 4.6 | Claude Sonnet 4.6 | No (you listen to the audition WAV) |
| **OpenAI Codex** | GPT-5.3-Codex / GPT-5.4 | GPT-5-mini / o4-mini | No (you listen to the audition WAV) |

## Repo layout

```text
.
├── ai-audio-clean-master/             # Self-contained agent skill folder (PEP 723 script)
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   └── scripts/clean_analog_master.py
├── deai_master/                       # Installable standalone package (CLI + offline web UI)
│   ├── __init__.py
│   ├── cli.py
│   ├── _engine.py
│   └── web_ui.py
├── dist/
│   └── ai-audio-clean-master-v2-skill.zip
├── guides/
│   ├── antigravity.md
│   ├── claude.md
│   ├── codex.md
│   └── models.md
├── pyproject.toml
├── install.sh
├── requirements.txt
└── LICENSE
```

## License

MIT. See [LICENSE](./LICENSE).
