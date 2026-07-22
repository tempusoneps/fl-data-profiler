#!/bin/bash

# Exit on error, pipe failures, and unset variables
set -euo pipefail

# Determine script & project directories to ensure relative path works from any working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${PROJECT_DIR}/datasets"

# Default dataset paths (output to datasets/ directory)
OHLCV_DATASET="${OHLCV_DATASET:-${DATASET_DIR}/VN30F1M_5m.csv}"
LABEL_DATASET="${LABEL_DATASET:-${DATASET_DIR}/label.csv}"
FEATURE_DATASET="${FEATURE_DATASET:-${DATASET_DIR}/feature.parquet}"

OHLCV_URL="https://raw.githubusercontent.com/tempusoneps/vn-stock-data/refs/heads/main/VN30F1M/data_ohlcv/VN30F1M_5m.csv"

FORCE=false

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--force)
      FORCE=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Automatically prepare OHLCV (CSV), Labels (CSV), and Features (Parquet) datasets inside datasets/ directory."
      echo ""
      echo "Options:"
      echo "  -f, --force    Force re-download and re-generation even if files already exist"
      echo "  -h, --help     Display this help message"
      exit 0
      ;;
    *)
      echo "[ERROR] Invalid option: $1" >&2
      echo "Run '$0 --help' for available options." >&2
      exit 1
      ;;
  esac
done

# Helper function to execute CLI commands (native, via uv, or from virtualenv)
run_cli() {
  local cmd="$1"
  if command -v "$cmd" &>/dev/null; then
    "$@"
  elif command -v uv &>/dev/null; then
    uv run "$@"
  elif [ -x "${PROJECT_DIR}/.venv/bin/${cmd}" ]; then
    "${PROJECT_DIR}/.venv/bin/$@"
  else
    echo "[ERROR] Command '$cmd' not found. Please setup environment using 'uv sync'." >&2
    exit 1
  fi
}

echo "=== START DATASET PREPARATION ==="
echo "[INFO] Target dataset directory: ${DATASET_DIR}"

# 0. Ensure datasets/ directory exists
mkdir -p "${DATASET_DIR}"

# 1. Check & Download OHLCV_DATASET (CSV)
if [ "$FORCE" = true ] || [ ! -f "$OHLCV_DATASET" ]; then
  echo "[INFO] OHLCV_DATASET does not exist or force requested. Downloading..."
  if command -v curl &>/dev/null; then
    curl -sSL -o "$OHLCV_DATASET" "$OHLCV_URL"
  elif command -v wget &>/dev/null; then
    wget -q -O "$OHLCV_DATASET" "$OHLCV_URL"
  else
    echo "[ERROR] curl or wget is required to download dataset." >&2
    exit 1
  fi
  echo "[PASS] Successfully downloaded OHLCV_DATASET (CSV): ${OHLCV_DATASET}"
else
  echo "[SKIP] OHLCV_DATASET (CSV) already exists: ${OHLCV_DATASET}"
fi

# 2. Check & Generate LABEL_DATASET (CSV)
if [ "$FORCE" = true ] || [ ! -f "$LABEL_DATASET" ]; then
  echo "[INFO] LABEL_DATASET does not exist or force requested. Generating..."
  run_cli labelohlcv "$OHLCV_DATASET" --mod vn30f1m --output "$LABEL_DATASET"
  echo "[PASS] Successfully generated LABEL_DATASET (CSV): ${LABEL_DATASET}"
else
  echo "[SKIP] LABEL_DATASET (CSV) already exists: ${LABEL_DATASET}"
fi

# 3. Check & Generate FEATURE_DATASET (Parquet directly via autofcholv)
if [ "$FORCE" = true ] || [ ! -f "$FEATURE_DATASET" ]; then
  echo "[INFO] FEATURE_DATASET does not exist or force requested. Extracting directly to Parquet..."
  run_cli autofcholv extract "$OHLCV_DATASET" --output "$FEATURE_DATASET"
  echo "[PASS] Successfully generated FEATURE_DATASET (Parquet): ${FEATURE_DATASET}"
else
  echo "[SKIP] FEATURE_DATASET (Parquet) already exists: ${FEATURE_DATASET}"
fi

echo "=== PREPARATION COMPLETE ==="
