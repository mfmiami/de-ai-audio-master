# Antigravity Guide

The main reason Antigravity works well for this skill is that Gemini models can read `.wav` files directly through `read_file`. That means the agent can listen to clips of your original mix, check the audition pass, and listen to four spot clips from the full render (intro, hook, verse, bridge) to catch watery artifacts or quiet-gap noise before you even open the file.

## 1. Prerequisites

Install `ffmpeg` and a Python environment with `numpy`, `scipy`, and `soundfile`:

```bash
brew install ffmpeg

python3 -m venv ~/.venvs/audio-master
~/.venvs/audio-master/bin/pip install numpy scipy soundfile
```

If you already have a Python venv with those packages (from Demucs, ACE-Step, etc.), you can point the agent to that interpreter instead.

## 2. Install the skill

Global install (available in every conversation):

```bash
./install.sh --antigravity
```

Or from the zip:

```bash
mkdir -p ~/.gemini/config/skills
unzip dist/ai-audio-clean-master-v2-skill.zip -d ~/.gemini/config/skills/
```

For a single workspace instead of global install:

```bash
mkdir -p /path/to/your-project/.agents/skills
cp -R ai-audio-clean-master /path/to/your-project/.agents/skills/
```

Check that the files landed where expected:

```bash
ls -la ~/.gemini/config/skills/ai-audio-clean-master/
# SKILL.md, agents/openai.yaml, scripts/clean_analog_master.py
```

## 3. Model recommendations

- **Gemini 3 Pro** (or **Gemini 2.5 Pro**): Best default. Good at both hearing subtle audio problems in `.wav` clips and debugging DSP/Python issues if a track needs custom tweaks.
- **Gemini 3 Flash** (or **Gemini 2.5 Flash**): Faster and cheaper when you just want to run the existing script as-is with audio spot-checks.
- **Claude Opus 4.6** (inside Antigravity): Useful if you're refactoring the DSP script itself, though switch back to a Gemini model if you want the agent to listen to the audio directly.

## 4. Typical workflow

Start a chat and give the agent your WAV path, optional stems folder, and Python path:

> Use the `ai-audio-clean-master` skill on `/path/to/MyTrack.wav` with stems in `/path/to/MyTrack Stems/`.
> Use `~/.venvs/audio-master/bin/python`.
> Audit the track and stems first, listen to the original, pick an exposed intro/hook range and a dense chorus range, run `--mode audition`, ear-check the output, and wait for my okay before rendering the full track.

What happens next:

1. The agent inspects the WAV header, stem correlations, and section levels, then listens to short clips of the original.
2. It runs `--mode audition` and gives you a loudness-matched comparison WAV (`Original -> 1s silence -> Processed`) plus a table of metrics.
3. Listen to the audition file. If anything sounds off, ask the agent to run `--skip-stem-reamp` vs `--skip-mix-cleanup` to isolate the problem.
4. Once it sounds right, reply:

> Looks good. Render the full 24-bit master and save a copy to my Downloads folder.

The agent runs `--mode full`, checks LUFS, LRA, true peak, `6-16 kHz` transient crest, tonal deltas, and the 44-byte header, listens to four spot clips, and writes the final WAV.

## 5. Practical notes

- **macOS sandbox**: Antigravity sandboxes terminal commands by default. If your audio is in `~/Downloads` or `~/Desktop`, it will ask once for permission to stage a copy into its scratch directory. Everything after that runs inside the sandbox.
- **Auto-BPM subharmonics**: On electronic or trance tracks, simple envelope autocorrelation sometimes locks onto 2/3 or 1/2 tempo (for example, `94` instead of `141`). If you know the BPM, pass it in your prompt or ask the agent to check kick-band BPM on the Drums stem and pass `--bpm`.
- **Stereo mix only (no stems)**: Just leave out the stems folder. The script skips stem re-amping and runs the full-mix chain directly.
- **Archival float copy**: Ask for `--also-float32` if you also want a 32-bit float WAV for DAW work.
