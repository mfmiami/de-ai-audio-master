# How-To Guide: OpenAI Codex (CLI & App)

This skill ships with both `SKILL.md` and `agents/openai.yaml` metadata so it registers cleanly in the **Codex CLI** and **Codex Desktop App** as `$ai-audio-clean-master`.

---

## 1. Install Prerequisites

```bash
brew install ffmpeg

python3 -m venv ~/.venvs/audio-master
~/.venvs/audio-master/bin/pip install numpy scipy soundfile
```

---

## 2. Install the Skill in Codex

### Global Install (Recommended)
```bash
# From a cloned copy of this repo:
./install.sh --codex

# Or from the pre-packaged zip:
mkdir -p ~/.codex/skills
unzip dist/ai-audio-clean-master-v2-skill.zip -d ~/.codex/skills/
```

### Repo-Local Install
```bash
mkdir -p /path/to/your-project/.codex/skills
cp -R ai-audio-clean-master /path/to/your-project/.codex/skills/
```

Verify the installation (including Codex's `agents/openai.yaml`):
```bash
ls -la ~/.codex/skills/ai-audio-clean-master/
ls -la ~/.codex/skills/ai-audio-clean-master/agents/
# Expected: SKILL.md, agents/openai.yaml, scripts/clean_analog_master.py
```

Restart Codex (or start a new session) so the skill list reloads.

---

## 3. Recommended Models in Codex

| Use Case | Recommended Model | Why |
| :--- | :--- | :--- |
| **Best Overall / DSP Engineering** | **GPT-5.3-Codex** (or latest Codex flagship) / **GPT-5.4** | Highest reliability for multi-step audio diagnostics, stem-bleed inspection, accurate `scipy.signal` modifications, and strict adherence to QA gates. |
| **Fast Everyday Mastering** | **GPT-5-mini** / **o4-mini** | Quick and economical when you only need standard `--mode audition` → `--mode full` execution without custom DSP changes. |

---

## 4. Important Note on Audio Auditioning in Codex

Codex excels at terminal automation, DSP math, and quantitative QA checks, but does **not** natively listen to raw `.wav` waveforms in CLI mode. Always pair Codex's quantitative QA output (`ebur128` LUFS/LRA/True Peak, `6–16 kHz` transient crest, `band_deltas_db`, `ripple_std_db`, `cliff_drop_db`) with **your own direct ear check** on the exported audition WAV before approving the full master.

---

## 5. Step-by-Step Usage in Codex

Invoke the skill explicitly with `$ai-audio-clean-master`:

> Use `$ai-audio-clean-master` on `~/Downloads/MyTrack.wav` with stems in `~/Downloads/MyTrack Stems/`.
> Use `~/.venvs/audio-master/bin/python`.
> Audit the format, header metadata, stem correlations, and section energy map first, then build the loudness-matched audition WAV (`--mode audition`) covering an exposed intro/hook and the densest vocal chorus. Print the exact playback timestamps and QA table and stop so I can listen.

Listen to the audition WAV locally (e.g., `afplay` on macOS or in your DAW), then approve the full render:

> Audition sounds great. Run `--mode full` with `--bpm <exact_bpm>`, verify the clean 44-byte header and full-track QA metrics, and save the 24-bit TPDF master into `~/Downloads/`.

---

## 6. Tips & Troubleshooting

- **Codex sandbox / approval mode**: Because `clean_analog_master.py` reads audio from `~/Downloads` (often outside your current git workspace), either copy your track + stems into your active workspace folder first or allow Codex to read/write the target folder when prompted.
- **BPM lock for electronic tracks**: Ask Codex to measure the kick-band (`40–140 Hz`) autocorrelation BPM on the `Drums` stem and pass `--bpm <BPM>` explicitly so the subtle `4.2%` wet analog delay tail locks tightly to the tempo.
- **Zero-stems mode**: If you only have a stereo bounce, omit `--stems-dir`; Codex will run the zero-STFT-framing linear-phase mastering chain directly on the full mix.
