#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="${SCRIPT_DIR}/ai-audio-clean-master"

usage() {
  cat <<'EOF'
Usage: ./install.sh [--all | --antigravity | --claude | --codex]

Installs the ai-audio-clean-master v2 skill into one or more agent harnesses:
  --antigravity   ~/.gemini/config/skills/ai-audio-clean-master
  --claude        ~/.claude/skills/ai-audio-clean-master
  --codex         ~/.codex/skills/ai-audio-clean-master
  --all           Install into all three harnesses (default if no flag given)
EOF
}

install_target() {
  local name="$1"
  local dest_root="$2"
  local dest="${dest_root}/ai-audio-clean-master"
  mkdir -p "${dest_root}"
  rm -rf "${dest}"
  cp -R "${SKILL_SRC}" "${dest}"
  chmod +x "${dest}/scripts/clean_analog_master.py"
  echo "Installed ${name} skill -> ${dest}"
}

if [[ ! -f "${SKILL_SRC}/SKILL.md" ]]; then
  echo "Error: skill source not found at ${SKILL_SRC}" >&2
  exit 1
fi

DO_AGY=0
DO_CLAUDE=0
DO_CODEX=0

if [[ $# -eq 0 ]]; then
  DO_AGY=1; DO_CLAUDE=1; DO_CODEX=1
else
  for arg in "$@"; do
    case "$arg" in
      --all) DO_AGY=1; DO_CLAUDE=1; DO_CODEX=1 ;;
      --antigravity) DO_AGY=1 ;;
      --claude) DO_CLAUDE=1 ;;
      --codex) DO_CODEX=1 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
  done
fi

[[ $DO_AGY -eq 1 ]] && install_target "Antigravity" "${HOME}/.gemini/config/skills"
[[ $DO_CLAUDE -eq 1 ]] && install_target "Claude Code" "${HOME}/.claude/skills"
[[ $DO_CODEX -eq 1 ]] && install_target "Codex" "${HOME}/.codex/skills"

echo ""
echo "Done. Ensure Python (numpy, scipy, soundfile) and ffmpeg are available in your environment."
