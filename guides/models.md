# Model Recommendations

Running this skill mostly involves two things: executing/tweaking Python DSP code and checking whether the rendered audio actually sounds clean. Different harnesses and models handle those two parts differently.

## Summary by harness

| Harness | Best default | Faster / cheaper | Native `.wav` listening? |
| :--- | :--- | :--- | :---: |
| **Antigravity** | Gemini 3 Pro / Gemini 2.5 Pro | Gemini 3 Flash / Gemini 2.5 Flash | Yes (`read_file` on `.wav` clips) |
| **Claude Code** | Claude Opus 4.6 | Claude Sonnet 4.6 | No |
| **Claude Desktop / Web** | Claude Opus 4.6 | Claude Sonnet 4.6 | No |
| **OpenAI Codex** | GPT-5.3-Codex / GPT-5.4 | GPT-5-mini / o4-mini | No |

## Which one to pick

- **Want the agent to catch audio artifacts before you listen?** Use **Antigravity with Gemini 3 Pro or Gemini 2.5 Pro**. Native audio input lets it listen to 10 to 16 second spot clips directly and run ablations if a quiet intro or vocal bridge sounds grainy.
- **Running standard tracks where you listen yourself?** **Gemini 3 Flash** (Antigravity), **Claude Sonnet 4.6** (Claude Code), or **GPT-5.3-Codex / GPT-5-mini** (Codex) are fast and handle the script cleanly.
- **Debugging tricky stems or modifying the DSP script?** **Claude Opus 4.6**, **Gemini 3 Pro**, or **GPT-5.3-Codex / GPT-5.4** are the most reliable when editing `scipy.signal` filter code or writing custom diagnostics (like kick-band BPM detection or alternative target stems like Guitar or Piano).

## QA numbers to check on any model

Regardless of harness or model, check these five values in the JSON summary before trusting a render:

1. `proc_lufs` within `±0.1 LU` of `orig_lufs`
2. `proc_lra` within `±0.2 LU` of `orig_lra`
3. `band_deltas_db` within `±0.20 dB` from `30 Hz` to `16 kHz`
4. `proc_hf_crest_db` within `±0.35 dB` of `orig_hf_crest_db` (makes sure cymbals and vocal consonants didn't get smeared)
5. `audio data file offset: 44` in `afinfo` (confirms no leftover generator metadata)

Numbers only catch so much. Always listen to the loudness-matched audition WAV on headphones or monitors before doing the full render.
