# De-AI Audio Master

> **Work in progress:** This tool has been verified on electronic/synth, guitar-driven rock, and vocal/drum-driven hip-hop tracks, and includes a fourth `acoustic` archetype for unplugged, jazz, and classical mixes. Always run the audition pass first and trust your ears.

A standalone, 100% local Python DSP tool for cleaning up common technical artifacts in AI-generated tracks (from tools like Suno or Udio) without wrecking the mix.

It runs completely offline on CPU with only `numpy`, `scipy`, and `soundfile` (`ffmpeg` is optional). You can use it as a standalone CLI, a zero-install `uv` one-liner, a local drag-and-drop browser UI, or optionally pair it with an AI coding agent using the included skill for Antigravity, Claude Code / Claude Desktop, and OpenAI Codex.

## Run it standalone (no AI agent required)

Every audio step happens locally on your machine. Nothing is uploaded anywhere, and there are zero API or token costs.

### Option 1: Zero-install one-liner with `uv`

Because the script embeds PEP 723 dependency metadata, you don't even have to clone the repo or create a virtualenv:

```bash
# Audition pass first (--profile auto detects electronic, rock, or hiphop from stems)
uv run https://raw.githubusercontent.com/mfmiami/de-ai-audio-master/main/ai-audio-clean-master/scripts/clean_analog_master.py \
  --input "/path/to/track.wav" \
  --stems-dir "/path/to/stems" \
  --profile auto \
  --mode audition

# Full 24-bit master (omit --stems-dir for a stereo bounce; pass --profile rock or hiphop for stemless non-electronic tracks)
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

The local web UI starts on `http://127.0.0.1:8765`, lets you pick a WAV, optional stems, and profile (`Auto`, `Electronic / Synth`, `Rock / Guitar Band`, `Hip-Hop / R&B / Vocal-Drum`, or `Acoustic / Jazz / Classical`), plays the loudness-matched A/B audition right in the browser with jump buttons for each pair, prints the QA table, and exports the final 24-bit WAV.

### Profiles (`--profile auto | electronic | rock | hiphop | acoustic`)

Rather than chasing one-off presets per subgenre, the engine groups music into four 8x-oversampled (384 kHz internal) acoustic/stem-dominance archetypes with energy-adaptive per-stem drive scaling so hybrid mixes adapt smoothly (pass `--oversample 4` if you want faster batch CPU runs):

- **`electronic` (synth-led: EDM, House, Techno, Synthpop, Trance, Ambient)**: Uses a 2-way vocal-guarded Wiener split (re-amping the dominant synth/bleed stem with 8x Class-A warmth, 12-tap Lanczos-6 BBD chorus, and subtle BPM delay), 4097-tap linear-phase comb flattening, 8x-oversampled analog air above `15.2 kHz` (keeping `4 to 15 kHz` sparkling open), elliptical `<95 Hz` sub-mono tightening + M/S stereo bloom (`+0.75 dB` Side `>350 Hz`), a subtle 2049-tap linear-phase mastering contour (`+0.45 dB @ 55 Hz`, `-0.55 dB @ 240 Hz`, `+0.30 dB @ 9.5 kHz`), and `1.2%` console crosstalk. Default varispeed: `0.0%`.
- **`rock` (guitar/band-led: Rock, Metal, Indie, Punk, Country, Acoustic Band)**: Uses a 4-way exact additive Wiener partition (`Vocals + Drums + Guitar + Bass/Rest`), applying energy-adaptive 8x-oversampled per-stem rock chains (U47/Neve vocal de-ess/warmth + `5%` studio plate, `20%` New York parallel tape drum bus + `6.5%` drum room, minimum-phase 4x12 guitar cab with `7.8 kHz` cone rolloff, `5.2 kHz` bass LP), followed by 15-ips tape rolloff (`10.5 kHz`) + 8x-oversampled envelope-gated analog HF harmonic regeneration, tight `4.2%` shared live-room glue, `1.8:1` 3-band bus glue, `-1.1 dB @ 180 Hz` de-mud cleanup, elliptical `<105 Hz` sub-mono + M/S bloom (`+0.70 dB` Side `>300 Hz`), `2%` console crosstalk, and `+1.15%` analog tape varispeed + human drummer phrase/bar pocket.
- **`hiphop` (vocal/drum/bass-led: Hip-Hop, Trap, R&B, Vocal Pop, Soul, Afrobeats, Reggaeton)**: Uses a 4-way exact additive Wiener partition (`Vocals + Drums + Bass + Sample/Melodic`), applying energy-adaptive 8x-oversampled per-stem chains (upfront dry vocal preamp + gentle de-ess + tight `3.4%` plate, `22%` sub-95 Hz protected parallel tape drum bus + tight `3.6%` booth, 808-safe `4.5 kHz` LP bass harmonics above `95 Hz`, and Class-A + Lanczos-6 BBD sample/keys warmth), followed by `10.8 kHz` 8x analog HF regeneration (open to `21.5 kHz`), `2.4%` tight studio glue, sub-friendly 3-band bus glue, balanced hip-hop contour (`+1.0 dB @ 58 Hz`, `-1.0 dB @ 190 Hz`, `+1.4 dB @ 6.8 kHz`, `+1.0 dB @ 12.5 kHz`), elliptical `<95 Hz` sub-mono + M/S bloom (`+0.75 dB` Side `>320 Hz`), `1.8%` console crosstalk, and `+1.10%` MPC/tape varispeed + tight groove pocket.
- **`acoustic` (unplugged/orchestral: Solo Piano, Jazz, Classical, Strings, Acoustic Vocal Ballads)**: Uses a 4-way exact additive Wiener partition (`Vocals + Keys/Strings + Acoustic/Other + Bass/Light Perc`), applying energy-adaptive ribbon/condenser 8x warmth and natural wooden hall ambience (no heavy drum crush or electric guitar cab), followed by `11.8 kHz` 8x analog HF regeneration, ultra-gentle `1.4:1` dynamic preservation, gentle `<80 Hz` sub tightening + M/S bloom, `1.6%` console crosstalk, and pitch-true `0.0%` default varispeed + natural rubato micro-timing.
- **`auto` (default)**: With `--stems-dir`, inspects fullband and `2 to 16 kHz` stem energy across all four archetypes (`acoustic`, `rock`, `hiphop`, `electronic`). Without stems, falls back to a stereo-only spectral, Mid/Side, and transient-contrast classifier so single-WAV drops auto-route cleanly too. Pass `--varispeed-pct 0` on `rock` or `hiphop` whenever you need exact original pitch and duration for video sync or DAW stem alignment.

### Built-in regression test suite

Run the self-contained 4-archetype regression suite (synthesizes multi-stem fixtures for all four archetypes and verifies auto-routing, LUFS matching, `<= -1.0 dBTP` true-peak ceiling, and clean 44-byte 24-bit headers):

```bash
python3 tests/test_regression.py
```

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

1. **Optional stem re-amp (if you have stems).** In `electronic` mode, it finds correlated bleed stems (`r > 0.35`), shields vocals with a `1.4x` weight, and runs an 87.5%-overlap additive 2-way Wiener split (`original == target_stem + backing` to `~1e-16` error, sub-90 Hz protected) with 8x Class-A warmth and Lanczos-6 BBD chorus. In `rock` and `hiphop` modes, it runs an exact 4-way additive Wiener partition (`Vocals + Drums + Guitar/Bass + Rest/Sample`, sub-85 Hz protected in Drums + Bass) so every dominant stem gets dedicated 8x-oversampled studio processing.
2. **Full-mix pass without STFT framing.** The full mix goes through a 4097-tap Blackman-windowed linear-phase FIR to cut stationary comb peaks, 8x-oversampled envelope-gated harmonic air regeneration (gated to zero in quiet gaps), a flat-magnitude 2nd-order allpass cascade, slow tonal drift, stereo-linked Lanczos-8 micro-timing variation (identical L/R read pointers so there is no flanging), elliptical sub-bass mono tightening + M/S width bloom, light 8x M/S Class-A bus saturation above 115 to 120 Hz, and console crosstalk.

Output is matched to the original integrated LUFS (with an 8x-oversampled soft-knee true-peak safety ceiling when rare phase spikes exceed `-1.2 dBFS`) and written as 24-bit PCM with 1-LSB TPDF dither (plus an optional 32-bit float file). If something sounds off on a specific track, use `--skip-stem-reamp` or `--skip-mix-cleanup` to isolate which pass caused it.

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
├── tests/
│   └── test_regression.py             # 4-archetype synthetic & real-track regression suite
├── pyproject.toml
├── install.sh
├── requirements.txt
└── LICENSE
```

## License

MIT. See [LICENSE](./LICENSE).
