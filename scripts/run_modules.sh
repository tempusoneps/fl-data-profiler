#!/usr/bin/env bash

# ==============================================================================
# Script: run_modules.sh
# Description: Sequentially execute fl-data-profiling modules one by one
#              with timing, error handling, and a final summary report.
#
# Default: Runs the 14 Fast & Robust profiling modules (finishes in ~2-4 mins).
# Slow/Heavy modules (MI, Permutation TS, Boruta, AutoML, etc.) are excluded
# by default and can be enabled with --all or --include-slow.
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${PROJECT_DIR}/datasets"

# Default paths and configurations
FEATURE_PATH=""
LABEL_PATH=""
OUTPUT_DIR="${PROJECT_DIR}/reports"
JOIN_KEY=""
LIMIT=""
FULL_MODE=false
FAIL_FAST=false
RUN_ALL=false
SELECTED_MODULES=""
SKIP_MODULES=""
EXTRA_TARGETS=()

# 1. Fast & Robust Profiling Modules (14 modules - Default)
# Typically completes in ~2-4 minutes total on VN30F1M dataset.
FAST_MODULES=(
  "statistics"
  "eda"
  "scipy"
  "statsmodels"
  "information_coefficient"
  "signal_analysis"
  "regime_scoring"
  "alphalens"
  "probability"
  "probability_2d"
  "probability_3d"
  "probability_drift"
  "kmean"
  "visual_regions"
  "sklearn"
  "regularized_linear"
  "xgboost"
  "lightgbm"
)

# 2. Slow & Resource-Intensive Modules (11 modules - Excluded by default)
# Excluded by default due to O(N * F^2) complexity, continuous kNN, multi-fold RF permutation, or AutoML budgets.
SLOW_MODULES=(
  "mutual_information"        # Continuous KNN mutual info on 50k rows × features (~1.5h)
  "permutation_importance_ts" # 20 rolling folds × 100+ RF permutation predictions (~1.5h)
  "timeseries_importance"     # Combines IC + Permutation Drop + Mutual Information (>3h)
  "mrmr"                      # Uses continuous mutual information & pairwise correlations
  "feature_interactions"      # Computes pairwise terms & runs mutual information twice
  "boruta"                    # 30 iterations of 300-tree RF shadow feature models
  "stability_selection"       # 30 resamples of L1 regularized logistic regression/lasso
  "shap"                      # TreeSHAP exact attribution across all features
  "flaml"                     # FLAML AutoML hyperparameter search
  "autogluon"                 # AutoGluon multi-layer stacking ensemble AutoML
  "pycaret"                   # PyCaret 15+ estimator AutoML pipeline
)

# Text formatting
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
  C_RESET=$'\e[0m'
  C_BOLD=$'\e[1m'
  C_RED=$'\e[31m'
  C_GREEN=$'\e[32m'
  C_YELLOW=$'\e[33m'
  C_BLUE=$'\e[34m'
  C_CYAN=$'\e[36m'
  C_MAGENTA=$'\e[35m'
else
  C_RESET=""
  C_BOLD=""
  C_RED=""
  C_GREEN=""
  C_YELLOW=""
  C_BLUE=""
  C_CYAN=""
  C_MAGENTA=""
fi

show_help() {
  cat << EOF
${C_BOLD}Usage:${C_RESET} $(basename "$0") [OPTIONS]

Sequentially run fl-data-profiler modules one by one.
By default, runs the ${C_GREEN}14 Fast & Recommended modules${C_RESET} (~2-4 mins total).

${C_BOLD}Options:${C_RESET}
  -f, --feature <path>       Path to feature dataset (Parquet or CSV).
                             Default: auto-detect datasets/feature.parquet or datasets/feature.csv
  -l, --label <path>         Path to label dataset (CSV or Parquet).
                             Default: auto-detect datasets/label.csv or datasets/label.parquet
  -o, --output-dir <dir>     Directory where module report folders will be created (default: reports)
  -k, --join-key <col>       Optional column name for joining feature and label rows
  -t, --target <col>         Label column to focus on (can be specified multiple times)
  -n, --limit <N>            Limit inputs to the first N rows before profiling
      --full                 Disable internal row downsampling across all modules
  -a, --all, --include-slow  Run all 25 modules including the 11 slow/resource-intensive modules
  -m, --modules <list>       Comma-separated list of specific modules to run
      --skip-modules <list>  Comma-separated list of modules to skip
      --fail-fast            Stop immediately if any module execution fails
  -h, --help                 Display this help message and exit

${C_BOLD}Default Fast Modules (14 - Running by default):${C_RESET}
  1.  statistics               - Descriptive stats, Pearson correlations & label quantiles (~2s)
  2.  eda                      - Exploratory Data Analysis & missingness profiling (~4s)
  3.  scipy                    - Hypothesis testing (ANOVA, t-test, Chi-square, Cohen's d) (~3s)
  4.  statsmodels              - Econometric OLS and Logit regressions (~5s)
  5.  information_coefficient  - Rolling walk-forward Pearson & Spearman Rank IC (~3s)
  6.  signal_analysis          - Trading signal evaluation (AUC, PR-AUC, F1, redundancy) (~6s)
  7.  regime_scoring           - Market regime-segmented feature scoring (~4s)
  8.  alphalens                - Factor tearsheet analysis & forward return quantiles (~5s)
  9.  probability              - 20-bin Quantile Conditional Probability & WoE/IV (~4s)
  10. probability_2d           - 2D Joint Probability Heatmap & Sweet Spots (~10s)
  11. probability_3d           - 3D Joint Probability & Hyper Sweet Spots (~15s)
  12. probability_drift        - Time-series probability stability & PSI drift (~5s)
  13. kmean                    - 2D KMeans clustering & label separability (~30s)
  14. visual_regions           - 2D grid decision rule generator (~30s)
  15. sklearn                  - Scikit-Learn baseline models (SGDClassifier, Ridge) (~5s)
  16. regularized_linear       - Lasso (L1) and Ridge (L2) regression shrinkage (~8s)
  17. xgboost                  - XGBoost GBDT importance & confusion matrix (~15s)
  18. lightgbm                 - LightGBM fast histogram GBDT importance (~10s)

${C_BOLD}Slow / Resource-Intensive Modules (11 - Excluded by default, enable via --all):${C_RESET}
  - mutual_information         - KNN mutual info across all continuous features (~90m)
  - permutation_importance_ts  - 20 rolling folds × 100+ RF permutation predictions (~90m)
  - timeseries_importance      - Unified IC + Permutation + Mutual Information (>180m)
  - mrmr                       - MRMR selection using continuous mutual information
  - feature_interactions       - Pairwise interaction discovery + mutual information
  - boruta                     - 30 iterations of 300-tree RF shadow feature models
  - stability_selection        - 30 resamples of L1 regularized logistic regression/lasso
  - shap                       - TreeSHAP value attribution & interpretability
  - flaml                      - Microsoft FLAML AutoML search
  - autogluon                  - Amazon AutoGluon multi-layer stacking AutoML
  - pycaret                    - PyCaret 15+ estimator AutoML pipeline

${C_BOLD}Examples:${C_RESET}
  $(basename "$0")                                     # Run 14 fast modules (default)
  $(basename "$0") --limit 1000                        # Fast profiling with 1000 rows
  $(basename "$0") --all                               # Run all 25 modules (including slow ones)
  $(basename "$0") --modules statistics,eda,xgboost    # Run only specific modules
  $(basename "$0") --skip-modules kmean,visual_regions # Skip specific modules
EOF
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--feature)
      FEATURE_PATH="$2"
      shift 2
      ;;
    -l|--label)
      LABEL_PATH="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -k|--join-key)
      JOIN_KEY="$2"
      shift 2
      ;;
    -t|--target)
      EXTRA_TARGETS+=("$2")
      shift 2
      ;;
    -n|--limit)
      LIMIT="$2"
      shift 2
      ;;
    --full)
      FULL_MODE=true
      shift
      ;;
    -a|--all|--include-slow)
      RUN_ALL=true
      shift
      ;;
    -m|--modules)
      SELECTED_MODULES="$2"
      shift 2
      ;;
    --skip-modules)
      SKIP_MODULES="$2"
      shift 2
      ;;
    --fail-fast)
      FAIL_FAST=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo -e "${C_RED}[ERROR] Unknown option: $1${C_RESET}" >&2
      echo "Run '$0 --help' for usage instructions." >&2
      exit 1
      ;;
  esac
done

# Auto-detect feature path if not provided
if [[ -z "$FEATURE_PATH" ]]; then
  if [[ -f "${DATASET_DIR}/feature.parquet" ]]; then
    FEATURE_PATH="${DATASET_DIR}/feature.parquet"
  elif [[ -f "${DATASET_DIR}/feature.csv" ]]; then
    FEATURE_PATH="${DATASET_DIR}/feature.csv"
  else
    echo -e "${C_RED}[ERROR] Feature file not found in ${DATASET_DIR}.${C_RESET}" >&2
    echo "Please specify --feature <path> or run: bash scripts/prepare_datasets.sh" >&2
    exit 1
  fi
fi

# Auto-detect label path if not provided
if [[ -z "$LABEL_PATH" ]]; then
  if [[ -f "${DATASET_DIR}/label.csv" ]]; then
    LABEL_PATH="${DATASET_DIR}/label.csv"
  elif [[ -f "${DATASET_DIR}/label.parquet" ]]; then
    LABEL_PATH="${DATASET_DIR}/label.parquet"
  else
    echo -e "${C_RED}[ERROR] Label file not found in ${DATASET_DIR}.${C_RESET}" >&2
    echo "Please specify --label <path> or run: bash scripts/prepare_datasets.sh" >&2
    exit 1
  fi
fi

# Validate input files exist
if [[ ! -f "$FEATURE_PATH" ]]; then
  echo -e "${C_RED}[ERROR] Feature file does not exist: $FEATURE_PATH${C_RESET}" >&2
  exit 1
fi
if [[ ! -f "$LABEL_PATH" ]]; then
  echo -e "${C_RED}[ERROR] Label file does not exist: $LABEL_PATH${C_RESET}" >&2
  exit 1
fi

# Function to run CLI commands
run_profiler_cmd() {
  local module_name="$1"
  local cmd_args=(fit "$FEATURE_PATH" "$LABEL_PATH" "--module" "$module_name" "--output-dir" "$OUTPUT_DIR")

  if [[ -n "$JOIN_KEY" ]]; then
    cmd_args+=("--join-key" "$JOIN_KEY")
  fi

  for target_col in "${EXTRA_TARGETS[@]}"; do
    cmd_args+=("--target" "$target_col")
  done

  if [[ -n "$LIMIT" ]]; then
    cmd_args+=("--limit" "$LIMIT")
  fi

  if [[ "$FULL_MODE" = true ]]; then
    cmd_args+=("--full")
  fi

  if command -v uv &>/dev/null; then
    uv run fldataprofiler "${cmd_args[@]}"
  elif command -v fldataprofiler &>/dev/null; then
    fldataprofiler "${cmd_args[@]}"
  elif [[ -x "${PROJECT_DIR}/.venv/bin/fldataprofiler" ]]; then
    "${PROJECT_DIR}/.venv/bin/fldataprofiler" "${cmd_args[@]}"
  else
    echo -e "${C_RED}[ERROR] Neither 'uv' nor 'fldataprofiler' found in PATH.${C_RESET}" >&2
    exit 1
  fi
}

# Determine the list of modules to execute
MODULES_TO_RUN=()
if [[ -n "$SELECTED_MODULES" ]]; then
  IFS=',' read -ra ADDR <<< "$SELECTED_MODULES"
  for m in "${ADDR[@]}"; do
    m="$(echo "$m" | xargs)"
    if [[ -n "$m" ]]; then
      MODULES_TO_RUN+=("$m")
    fi
  done
elif [[ "$RUN_ALL" = true ]]; then
  MODULES_TO_RUN=("${FAST_MODULES[@]}" "${SLOW_MODULES[@]}")
else
  # Default: run only fast modules
  MODULES_TO_RUN=("${FAST_MODULES[@]}")
fi

# Filter out skipped modules
FINAL_MODULES=()
if [[ -n "$SKIP_MODULES" ]]; then
  IFS=',' read -ra SKIP_ADDR <<< "$SKIP_MODULES"
  declare -A SKIP_MAP
  for s in "${SKIP_ADDR[@]}"; do
    s="$(echo "$s" | xargs)"
    if [[ -n "$s" ]]; then
      SKIP_MAP["$s"]=1
    fi
  done

  for m in "${MODULES_TO_RUN[@]}"; do
    if [[ -z "${SKIP_MAP[$m]:-}" ]]; then
      FINAL_MODULES+=("$m")
    fi
  done
else
  FINAL_MODULES=("${MODULES_TO_RUN[@]}")
fi

TOTAL_MODULES=${#FINAL_MODULES[@]}

if [[ $TOTAL_MODULES -eq 0 ]]; then
  echo -e "${C_YELLOW}[WARN] No modules selected to run.${C_RESET}"
  exit 0
fi

# Print banner
echo -e "${C_BOLD}${C_CYAN}================================================================================${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}                    FL-DATA-PROFILER: RUN PROFILING PIPELINE                    ${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}================================================================================${C_RESET}"
echo -e "${C_BOLD}Feature Data:${C_RESET}    $FEATURE_PATH"
echo -e "${C_BOLD}Label Data:${C_RESET}      $LABEL_PATH"
echo -e "${C_BOLD}Output Dir:${C_RESET}      $OUTPUT_DIR"
[[ -n "$JOIN_KEY" ]] && echo -e "${C_BOLD}Join Key:${C_RESET}        $JOIN_KEY"
[[ ${#EXTRA_TARGETS[@]} -gt 0 ]] && echo -e "${C_BOLD}Targets:${C_RESET}         ${EXTRA_TARGETS[*]}"
[[ -n "$LIMIT" ]] && echo -e "${C_BOLD}Row Limit:${C_RESET}       $LIMIT"
echo -e "${C_BOLD}Full Row Mode:${C_RESET}   $FULL_MODE"
echo -e "${C_BOLD}Mode:${C_RESET}            $([[ "$RUN_ALL" = true ]] && echo "${C_YELLOW}All 25 Modules (including Slow)${C_RESET}" || echo "${C_GREEN}Fast Modules Only (${TOTAL_MODULES} modules)${C_RESET}")"
echo -e "${C_BOLD}Total Modules:${C_RESET}   $TOTAL_MODULES"
echo -e "${C_BOLD}${C_CYAN}================================================================================${C_RESET}"
echo ""

# Results tracking
declare -A MODULE_STATUS
declare -A MODULE_DURATION
declare -A MODULE_ERROR_MSG

PASSED_COUNT=0
FAILED_COUNT=0
OVERALL_START_TIME=$(date +%s)

for idx in "${!FINAL_MODULES[@]}"; do
  module_name="${FINAL_MODULES[$idx]}"
  step_num=$((idx + 1))
  
  echo -e "${C_BOLD}${C_BLUE}[${step_num}/${TOTAL_MODULES}] Starting Module: ${C_MAGENTA}${module_name}${C_RESET} ..."
  
  START_TIME=$(date +%s)
  
  # Run the module and capture output/error
  MODULE_LOG_FILE=$(mktemp)
  if run_profiler_cmd "$module_name" > "$MODULE_LOG_FILE" 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MODULE_STATUS["$module_name"]="SUCCESS"
    MODULE_DURATION["$module_name"]="${DURATION}s"
    PASSED_COUNT=$((PASSED_COUNT + 1))
    echo -e "${C_GREEN}[PASS] Completed ${module_name} in ${DURATION}s${C_RESET}"
    # Show artifacts created if found in output
    grep "Report written to:" "$MODULE_LOG_FILE" || true
  else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MODULE_STATUS["$module_name"]="FAILED"
    MODULE_DURATION["$module_name"]="${DURATION}s"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    
    LAST_ERR=$(tail -n 5 "$MODULE_LOG_FILE" | tr '\n' ' ')
    MODULE_ERROR_MSG["$module_name"]="$LAST_ERR"
    
    echo -e "${C_RED}[FAIL] Module ${module_name} failed after ${DURATION}s${C_RESET}" >&2
    echo -e "${C_RED}Error output snippet:${C_RESET}" >&2
    tail -n 10 "$MODULE_LOG_FILE" >&2
    
    if [[ "$FAIL_FAST" = true ]]; then
      echo -e "${C_RED}[ABORT] --fail-fast is enabled. Stopping execution.${C_RESET}" >&2
      rm -f "$MODULE_LOG_FILE"
      break
    fi
  fi
  rm -f "$MODULE_LOG_FILE"
  echo ""
done

OVERALL_END_TIME=$(date +%s)
OVERALL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))

# Print summary table
echo ""
echo -e "${C_BOLD}${C_CYAN}================================================================================${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}                             PROFILING RUN SUMMARY                              ${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}================================================================================${C_RESET}"
printf "%-4s %-28s %-12s %-10s %s\n" "No." "Module Name" "Status" "Duration" "Report Location"
echo "--------------------------------------------------------------------------------"

for idx in "${!FINAL_MODULES[@]}"; do
  m="${FINAL_MODULES[$idx]}"
  no=$((idx + 1))
  status="${MODULE_STATUS[$m]:-NOT_RUN}"
  duration="${MODULE_DURATION[$m]:-0s}"
  report_path="${OUTPUT_DIR}/${m}"
  
  if [[ "$status" == "SUCCESS" ]]; then
    status_fmt="${C_GREEN}SUCCESS${C_RESET}"
  elif [[ "$status" == "FAILED" ]]; then
    status_fmt="${C_RED}FAILED${C_RESET} "
  else
    status_fmt="${C_YELLOW}SKIPPED${C_RESET}"
  fi
  
  printf "%-4d %-28s " "$no" "$m"
  echo -ne "$status_fmt"
  printf "     %-10s %s\n" "$duration" "$report_path"
done

echo -e "${C_BOLD}${C_CYAN}================================================================================${C_RESET}"
echo -e "Total: ${C_BOLD}${TOTAL_MODULES}${C_RESET} | Passed: ${C_GREEN}${PASSED_COUNT}${C_RESET} | Failed: ${C_RED}${FAILED_COUNT}${C_RESET} | Total Time: ${C_BOLD}${OVERALL_DURATION}s${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}================================================================================${C_RESET}"

if [[ $FAILED_COUNT -gt 0 ]]; then
  exit 1
fi
exit 0
