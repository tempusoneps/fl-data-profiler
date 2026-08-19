#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS="$ROOT_DIR/AGENTS.md"
GEMINI="$ROOT_DIR/GEMINI.md"
CLAUDE="$ROOT_DIR/CLAUDE.md"

DOCS=(
  "docs/.ai/AI_AGENT_GUIDELINE.md"
  "docs/.ai/RULE.md"
  "docs/.ai/STRUCTURE.md"
)

HEADER="# Agent Guide

This file is generated from project documentation. Do not edit it directly.
"

echo "$HEADER" > "$AGENTS"
echo "$HEADER" > "$GEMINI"
echo "$HEADER" > "$CLAUDE"

for doc in "${DOCS[@]}"; do
  echo "" >> "$AGENTS"
  echo "" >> "$GEMINI"
  echo "" >> "$CLAUDE"
  cat "$ROOT_DIR/$doc" >> "$AGENTS"
  cat "$ROOT_DIR/$doc" >> "$GEMINI"
  cat "$ROOT_DIR/$doc" >> "$CLAUDE"
done

echo "Generated AGENTS.md, GEMINI.md, CLAUDE.md"
