# How-To Guide: Google Antigravity

Antigravity is the **highest-leverage harness** for `ai-audio-clean-master` because Gemini models can **natively listen to `.wav` files** via `read_file`. That lets the agent actually ear-check your original track, audition the processed A/B clips, and verify all 4 spot-check clips (intro, hook, verse, bridge) before delivering the final master.

---

## 1. Install Prerequisites

Make sure Python (`numpy`, `scipy`, `soundfile`) and `ffmpeg` are installed on your machine:

```bash
brew install ffmpeg

# Option A: dedicated venv (recommended)
python3 -m venv ~/.venvs/audio-master
~/.venvs/audio-master/bin/pip install numpy scipy soundfile

# Option B: reuse an existing audio Python venv (e.g., ACE-Step, Demucs, etc.)
```

---

## 2. Install the Skill in Antigravity

Choose **global install** (available in every conversation and workspace) or **workspace-local install**:

### Global Install (Recommended)
```bash
# From a cloned copy of this repo:
./install.sh --antigravity

# Or from the pre-packaged zip:
mkdir -p ~/.gemini/config/skills
unzip dist/ai-audio-clean-master-v2-skill.zip -d ~/.gemini/config/skills/
```

### Workspace-Local Install (Project-Specific)
```bash
mkdir -p /path/to/your-project/.agents/skills
cp -R ai-audio-clean-master /path/to/your-project/.agents/skills/
```

Verify the skill files are in place:
```bash
ls -la ~/.gemini/config/skills/ai-audio-clean-master/
# Expected: SKILL.md, agents/openai.yaml, scripts/clean_analog_master.py
```

---

## 3. Recommended Models in Antigravity

| Use Case | Recommended Model | Why |
| :--- | :--- | :--- |
| **Default / Best Quality** | **Gemini 3 Pro** (or **Gemini 2.5 Pro**) | Strongest combined **native audio perception** (hears subtle watery artifacts, vocal coloration, and quiet-gap noise) + precise DSP/Python reasoning when tuning parameters. |
| **Fast Standard Runs** | **Gemini 3 Flash** (or **Gemini 2.5 Flash**) | Fast and cost-effective when running the bundled v2 script end-to-end with native `.wav` ear-checks and minimal custom DSP edits. |
| **Custom DSP / FIR Refactors** | **Gemini 3 Pro** or **Claude Opus 4.6** | If modifying filter taps, Wiener masks, or adding new analog stages; switch back to a Gemini model for native audio ear-checking. |

---

## 4. Step-by-Step Usage Workflow

1. **Start a conversation** in Antigravity and point to your stereo mix (and optional stems folder).
2. **Copy/paste a prompt** like:

   > Use the `ai-audio-clean-master` skill on `/path/to/MyTrack.wav` (stems are in `/path/to/MyTrack Stems/`).
   > Use `~/.venvs/audio-master/bin/python` for Python.
   > Audit the track, listen to the original first, pick the best exposed intro/hook and dense chorus ranges, run `--mode audition`, ear-check the output, and wait for my approval before rendering the full 24-bit master.

3. **Review the Audition Pass**:
   - Antigravity stages your audio, checks stem correlations (projecting any duplicate bleed cleanly while guarding vocals at `1.4×`), runs `--mode audition`, and listens directly to the rendered clips.
   - Click the generated `*-Audition.wav` link to compare Original → Processed at matched LUFS.
   - If you hear anything off on a specific section, ask Antigravity to run an ablation (`--skip-stem-reamp` vs `--skip-mix-cleanup`) and adjust only the offending stage.
4. **Approve & Render the Full Master**:
   - Reply: *"Sounds great. Render the full 24-bit master and save a copy in my Downloads folder."*
   - Antigravity runs `--mode full`, verifies LUFS, LRA, true peak, `6–16 kHz` transient crest, tonal deltas, and clean 44-byte header, ear-checks 4 spot clips across the full song, and delivers the final master.

---

## 5. Tips & Troubleshooting

- **Sandbox permissions on macOS**: Antigravity runs shell commands inside a sandbox by default. When your original audio lives in `~/Downloads` or `~/Desktop`, Antigravity will ask once to copy/stage it into the session scratch folder; after that, all DSP rendering and audio auditing run cleanly inside the sandbox with zero extra prompts.
- **Wrong auto-detected BPM**: If your track is electronic/trance and auto-BPM picks a subharmonic (e.g., `94` instead of `141`), tell Antigravity the true BPM or ask it to measure kick-band BPM from the Drums stem and pass `--bpm <value>`.
- **No stems available**: Simply omit the stems folder. The skill automatically skips stem re-amping and runs the zero-STFT-framing linear-phase mastering chain on the full stereo mix.
- **32-bit float DAW archive**: Ask Antigravity to include `--also-float32` during `--mode full` if you want a 32-bit float archival WAV alongside the 24-bit TPDF master.
