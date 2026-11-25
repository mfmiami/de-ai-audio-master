# How-To Guide: Claude Code, Claude Desktop & Claude.ai

This skill works seamlessly in **Claude Code (CLI / IDE)** and can also be uploaded directly as a ZIP into **Claude Desktop / Claude.ai**.

---

## 1. Install Prerequisites (Claude Code)

Claude Code executes the bundled Python DSP script locally, so install `ffmpeg` and the Python scientific stack first:

```bash
brew install ffmpeg

python3 -m venv ~/.venvs/audio-master
~/.venvs/audio-master/bin/pip install numpy scipy soundfile
```

*(For Claude.ai / Claude Desktop cloud code execution, check that `numpy`, `scipy`, `soundfile`, and `ffmpeg` are available in the container environment, or use Claude Code locally for full-length 48 kHz multi-stem tracks, which are often 50–800 MB total.)*

---

## 2. Install the Skill

### Option A: Claude Code (Global — Recommended)
```bash
# From a cloned copy of this repo:
./install.sh --claude

# Or from the pre-packaged zip:
mkdir -p ~/.claude/skills
unzip dist/ai-audio-clean-master-v2-skill.zip -d ~/.claude/skills/
```

### Option B: Claude Code (Project-Local)
```bash
mkdir -p /path/to/your-project/.claude/skills
cp -R ai-audio-clean-master /path/to/your-project/.claude/skills/
```

### Option C: Claude.ai / Claude Desktop (ZIP Upload)
1. Download [`dist/ai-audio-clean-master-v2-skill.zip`](../dist/ai-audio-clean-master-v2-skill.zip).
2. Open **Settings → Capabilities → Skills**.
3. Click **Upload Skill** and select `ai-audio-clean-master-v2-skill.zip`.
   *(The zip uses a single `ai-audio-clean-master/` root folder with `SKILL.md` and `scripts/clean_analog_master.py` and zero macOS/metadata clutter.)*

Verify in Claude Code:
```bash
ls -la ~/.claude/skills/ai-audio-clean-master/
# Expected: SKILL.md, agents/openai.yaml, scripts/clean_analog_master.py
```

---

## 3. Recommended Models for Claude

| Use Case | Recommended Model | Why |
| :--- | :--- | :--- |
| **Best Overall / Custom DSP Tuning** | **Claude Opus 4.6** | Best precision for stem-correlation analysis, custom `scipy.signal` filter tuning, ablation diagnosis, and strict QA verification. |
| **Best Speed & Cost Balance** | **Claude Sonnet 4.6** | Fast, highly reliable at invoking `clean_analog_master.py`, auditing QA JSON outputs, and enforcing the audition-first gate. |
| **Avoid for Audio DSP Editing** | Lightweight/Haiku-class models | Fine for invoking a fixed CLI command, but prone to skipping stem bleed diagnostics or misreading subtle acoustic QA metrics. |

---

## 4. Important Note on Audio Auditioning in Claude

Claude models do **not** natively listen to raw `.wav` waveforms the way audio-multimodal models do. To keep the **Audition First** rule airtight in Claude Code:

1. Let Claude run `--mode audition` and report the full quantitative QA table (`orig_lufs` vs `proc_lufs`, `band_deltas_db`, `orig_hf_crest_db` vs `proc_hf_crest_db`, `orig_lr_corr` vs `proc_lr_corr`, `ripple_std_db`, `cliff_drop_db`).
2. **You** listen directly to the exported `*-Reference-Master-Audition.wav` (on macOS, you can even ask Claude to run `open` or `afplay` on it, or drop it into your DAW / QuickTime).
3. Only after your ears confirm zero artifacts, approve `--mode full`!

---

## 5. Step-by-Step Prompting in Claude Code

Start `claude` in your working folder and run:

> Use the `ai-audio-clean-master` skill on `~/Downloads/MyTrack.wav` with stems in `~/Downloads/MyTrack Stems/`.
> Run with `~/.venvs/audio-master/bin/python`.
> First inspect the track format, header metadata, stem correlations, and energy map to pick the best exposed intro/hook and dense chorus ranges, then run `--mode audition`. Show me the exact playback timestamps and full before/after QA table, and pause so I can listen before you render the full track.

After listening to the audition WAV:

> Sounds clean—run `--mode full` with the exact kick BPM and save the 24-bit master to `~/Downloads/MyTrack-Reference-Analog-Master-v2-24bit-48kHz.wav`. Verify the 44-byte header with `afinfo` and print the full QA summary.

---

## 6. Tips & Troubleshooting

- **Large stem folders**: Full 48 kHz stereo stems (10–12 WAVs) can total 500+ MB. Always use **Claude Code locally** rather than uploading 500 MB into Claude.ai web chat.
- **Custom Python path**: If `python3` in your default shell lacks `numpy`/`scipy`/`soundfile`, explicitly tell Claude the venv path (`~/.venvs/audio-master/bin/python`) in your prompt or activate the venv before launching `claude`.
- **Inspecting subtle issues**: If you hear anything unnatural in the audition WAV, tell Claude *which timestamp* sounds off (e.g., *"the vocal at 0:18 sounds slightly phasey"*) and instruct it to run the ablation flags (`--skip-stem-reamp` vs `--skip-mix-cleanup`) to isolate and fix only that stage.
