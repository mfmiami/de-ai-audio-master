# OpenAI Codex Guide

The skill folder includes both `SKILL.md` and `agents/openai.yaml`, so Codex CLI and the Codex app pick it up as `$ai-audio-clean-master`.

## 1. Prerequisites

```bash
brew install ffmpeg

python3 -m venv ~/.venvs/audio-master
~/.venvs/audio-master/bin/pip install numpy scipy soundfile
```

## 2. Install the skill

Global install:

```bash
./install.sh --codex
```

Or from the zip:

```bash
mkdir -p ~/.codex/skills
unzip dist/ai-audio-clean-master-v2-skill.zip -d ~/.codex/skills/
```

Project-local install:

```bash
mkdir -p /path/to/your-project/.codex/skills
cp -R ai-audio-clean-master /path/to/your-project/.codex/skills/
```

Verify both `SKILL.md` and `agents/openai.yaml` are present, then start a fresh Codex session so the skill list refreshes:

```bash
ls -la ~/.codex/skills/ai-audio-clean-master/
ls -la ~/.codex/skills/ai-audio-clean-master/agents/
```

## 3. Model recommendations

- **GPT-5.3-Codex** (or **GPT-5.4**): Best default for inspecting stems, checking QA numbers carefully, and editing `scipy.signal` code when a track needs custom adjustments.
- **GPT-5-mini** / **o4-mini**: Good enough for routine `--mode audition` -> `--mode full` runs where you don't need any custom DSP edits.

## 4. Handling audio listening in Codex

Like Claude Code, Codex CLI does not natively hear `.wav` waveforms. Let Codex handle the stem audit, DSP run, and quantitative QA checks, and do the ear check yourself on the loudness-matched audition WAV (`afplay` or your DAW) before approving the full render.

## 5. Example prompts

Audition pass:

> Use `$ai-audio-clean-master` on `~/Downloads/MyTrack.wav` with stems in `~/Downloads/MyTrack Stems/`.
> Use `~/.venvs/audio-master/bin/python`.
> Check the format, WAV header chunks, stem correlations, and 15-second RMS map first, then build `--mode audition` covering an exposed intro/hook and the main chorus. Print the playback timestamps and QA table, and pause so I can listen.

Full render after listening:

> Audition sounds clean. Measure the kick BPM from the Drums stem, run `--mode full` with that `--bpm`, verify the 44-byte WAV header and QA summary, and copy the 24-bit master to `~/Downloads/`.

## 6. Practical notes

- **Workspace access**: If your Codex session is restricted to the current repo directory, either copy your track and stems into a scratch folder inside the workspace first or approve read/write access to `~/Downloads` when prompted.
- **No stems**: Omit `--stems-dir` and the script runs the full-mix chain only.
