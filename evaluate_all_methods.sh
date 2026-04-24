#!/bin/bash

# Comprehensive evaluation script for all baseline methods across multiple models
# Tests: PromptFuzz (fuzz), ReAct-Leak (re), PLeak (pleak), and LeakAgent (leakagent)
# Models: Llama 3.1-8B, Llama 3.1-70B, Mistral-7B, GPT-OSS 20B, Qwen 3.5-27B

set -o pipefail

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="evaluation_logs"
RESULTS_DIR="evaluation_results"

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Main log file
MAIN_LOG="$LOG_DIR/evaluation_${TIMESTAMP}.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define methods and their result directories
declare -A METHODS=(
    ["fuzz"]="fuzz_results"
    ["re"]="re_results"
    ["pleak"]="pleak_results"
    ["leakagent"]="leakagent_results"
)

# Define models with their HuggingFace names
declare -A MODELS=(
    ["llama3.1-8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["llama3.1-70b"]="meta-llama/Llama-3.1-70B-Instruct"
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3"
    ["gpt-oss-20b"]="openai/gpt-oss-20b"
    ["qwen3.5-27b"]="Qwen/Qwen3.5-27B"
)

# Tracking variables
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
SKIPPED_EXPERIMENTS=0

# Function to log messages
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${message}" | tee -a "$MAIN_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${message}" | tee -a "$MAIN_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${message}" | tee -a "$MAIN_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${message}" | tee -a "$MAIN_LOG"
            ;;
        "DEBUG")
            echo -e "${YELLOW}[DEBUG]${NC} ${message}" | tee -a "$MAIN_LOG"
            ;;
    esac
}

# Function to run a single evaluation
run_evaluation() {
    local method=$1
    local method_display=$2
    local model=$3
    local model_display=$4
    local prompts_file=$5
    local method_log=$6

    ((TOTAL_EXPERIMENTS++))

    log_message "INFO" "Starting evaluation: $method_display vs $model_display"
    log_message "DEBUG" "Using prompts file: $prompts_file"

    # Check if prompts file exists
    if [ ! -f "$prompts_file" ]; then
        log_message "WARNING" "Skipping $method_display: Results file not found at $prompts_file"
        ((SKIPPED_EXPERIMENTS++))
        return 1
    fi

    # Run evaluation with error handling
    local output_file="$RESULTS_DIR/${method}_vs_${model}_${TIMESTAMP}.csv"

    if python evaluate_task.py \
        --prompts_data_path "$prompts_file" \
        --model_name "$3" \
        --n_samples 5 \
        --dataset_path test_data_pleak.csv \
        --disable_tqdm >> "$method_log" 2>&1; then

        log_message "SUCCESS" "Completed evaluation: $method_display vs $model_display"
        log_message "DEBUG" "Results saved to: $output_file"
        ((SUCCESSFUL_EXPERIMENTS++))
        return 0
    else
        log_message "ERROR" "Failed evaluation: $method_display vs $model_display"
        ((FAILED_EXPERIMENTS++))
        return 1
    fi
}

# Main evaluation loop
main() {
    log_message "INFO" "=================================================="
    log_message "INFO" "Starting comprehensive model evaluation"
    log_message "INFO" "Timestamp: $TIMESTAMP"
    log_message "INFO" "Log directory: $LOG_DIR"
    log_message "INFO" "Results directory: $RESULTS_DIR"
    log_message "INFO" "=================================================="

    # Print configuration
    echo "" | tee -a "$MAIN_LOG"
    log_message "INFO" "Methods to evaluate:"
    for method in "${!METHODS[@]}"; do
        log_message "DEBUG" "  - $method (${METHODS[$method]})"
    done

    echo "" | tee -a "$MAIN_LOG"
    log_message "INFO" "Models to test against:"
    for model in "${!MODELS[@]}"; do
        log_message "DEBUG" "  - $model (${MODELS[$model]})"
    done

    echo "" | tee -a "$MAIN_LOG"

    # Main evaluation loops
    for method in "${!METHODS[@]}"; do
        for model in "${!MODELS[@]}"; do
            prompts_file="${METHODS[$method]}/good_prompts.csv"
            method_log="$LOG_DIR/${method}_vs_${model}_${TIMESTAMP}.log"

            # Run evaluation with error handling
            if ! run_evaluation "$method" "$method" "$model" "$model" "${MODELS[$model]}" "$prompts_file" "$method_log"; then
                log_message "WARNING" "Evaluation failed or skipped for $method vs $model"
            fi

            echo "" | tee -a "$MAIN_LOG"
        done
    done

    # Summary
    echo "" | tee -a "$MAIN_LOG"
    log_message "INFO" "=================================================="
    log_message "INFO" "Evaluation Summary"
    log_message "INFO" "=================================================="
    log_message "INFO" "Total experiments: $TOTAL_EXPERIMENTS"
    log_message "SUCCESS" "Successful: $SUCCESSFUL_EXPERIMENTS"
    log_message "ERROR" "Failed: $FAILED_EXPERIMENTS"
    log_message "WARNING" "Skipped: $SKIPPED_EXPERIMENTS"
    log_message "INFO" "=================================================="
    log_message "INFO" "Main log: $MAIN_LOG"
    log_message "INFO" "Results directory: $RESULTS_DIR"
    log_message "INFO" "=================================================="
}

# Run main function
main "$@"
