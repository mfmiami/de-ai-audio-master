# Claude Code & Claude Desktop Guide

This skill runs locally in **Claude Code** (CLI or IDE) and can also be uploaded as a zip in **Claude Desktop / Claude.ai**.

## 1. Prerequisites (Claude Code)

Claude Code runs the Python DSP script on your machine, so install `ffmpeg` and the Python dependencies first:

```bash
brew install ffmpeg

python3 -m venv ~/.venvs/audio-master
~/.venvs/audio-master/bin/pip install numpy scipy soundfile
```

If you're working with full-length 48 kHz tracks and 10 to 12 stems (often 500 MB or more), use Claude Code locally rather than uploading large WAV files into Claude.ai web chat.

## 2. Install the skill

### Claude Code (global)

```bash
./install.sh --claude
```

Or unzip directly:

```bash
mkdir -p ~/.claude/skills
unzip dist/ai-audio-clean-master-v2-skill.zip -d ~/.claude/skills/
```

### Claude Code (single project)

```bash
mkdir -p /path/to/your-project/.claude/skills
cp -R ai-audio-clean-master /path/to/your-project/.claude/skills/
```

### Claude.ai / Claude Desktop (zip upload)

1. Download [`dist/ai-audio-clean-master-v2-skill.zip`](../dist/ai-audio-clean-master-v2-skill.zip).
2. Open **Settings -> Capabilities -> Skills**.
3. Upload the zip file.

## 3. Model recommendations

- **Claude Opus 4.6**: Best pick if you expect to tweak the DSP script, inspect weird stem bleed, or run ablations on a difficult track.
- **Claude Sonnet 4.6**: Faster and cheaper for normal runs where you just want the bundled script executed and verified cleanly.
- **Smaller/Haiku models**: Skip these for audio tuning. They are fine for running a single fixed command, but more likely to gloss over stem correlation warnings or QA deltas.

## 4. Handling audio listening in Claude

Claude cannot directly listen to `.wav` files the way native-audio models can. In practice, that just means **you** do the ear check during Step 2:

1. Have Claude run `--mode audition` and print the full QA table (LUFS, true peak, band deltas, `6-16 kHz` crest factor, L/R correlation, comb ripple, cliff drop).
2. Listen to the exported `*-Audition.wav` yourself (QuickTime, your DAW, or `afplay` on macOS).
3. Tell Claude to run `--mode full` only after your ears confirm there are no artifacts.

## 5. Example prompts for Claude Code

First pass (audit + audition):

> Use the `ai-audio-clean-master` skill on `~/Downloads/MyTrack.wav` with stems in `~/Downloads/MyTrack Stems/`.
> Run Python with `~/.venvs/audio-master/bin/python`.
> Check the file format, WAV header metadata, stem correlations, and energy map first, then run `--mode audition` on an exposed intro/hook section and a dense chorus. Print the exact playback timestamps and before/after QA metrics, then stop so I can listen.

After listening:

> Sounds good. Run `--mode full` with the actual kick BPM, save the 24-bit master to `~/Downloads/MyTrack-Master-24bit-48kHz.wav`, check the 44-byte header with `afinfo`, and print the final QA metrics.

## 6. Troubleshooting

- **Missing Python packages**: If your default `python3` doesn't have `numpy`/`scipy`/`soundfile`, either activate your venv before starting `claude` or include the full path (`~/.venvs/audio-master/bin/python`) in your prompt.
- **Something sounds off at a specific timestamp**: Tell Claude the exact second where you hear an issue (for example, "vocal reverb at 0:18 sounds phasey") and have it run `--skip-stem-reamp` vs `--skip-mix-cleanup` so you can compare the two isolated outputs.
