#!/bin/bash

###############################################################################
# Full Evaluation Suite
#
# Orchestrates evaluation of all attack methods across all modes and defenses.
# Server lifecycle is managed by individual Python scripts.
#
# Attack Methods: pleak, fuzz, re, leakagent
# Models: llama3.1-8b, llama3.1-70b, mistral-7b, qwen3.5-27b
# Defenses: promptguard (all models), secalign (llama3.1-8b), sft (llama3.1-8b)
###############################################################################

set -o pipefail  # Fail if any part of a pipeline fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="evaluation_results"
LOG_DIR="evaluation_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$LOG_DIR/evaluation_summary_$TIMESTAMP.log"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$SUMMARY_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$SUMMARY_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$SUMMARY_LOG"
}

print_section() {
    echo "" | tee -a "$SUMMARY_LOG"
    echo "================================================================================" | tee -a "$SUMMARY_LOG"
    echo "$1" | tee -a "$SUMMARY_LOG"
    echo "================================================================================" | tee -a "$SUMMARY_LOG"
}

# Attack methods
declare -A ATTACK_METHODS=(
    [pleak]="pleak_results"
    [fuzz]="llama3_fuzz_finetune"
    [re]="llama3_re_finetune"
    [leakagent]="llama3_rl_finetune"
)

# Models (ordered: llama first, then others)
declare -A MODELS=(
    [llama3.1-8b]="meta-llama/Llama-3.1-8B-Instruct"
    [llama3.1-70b]="meta-llama/Llama-3.1-70B-Instruct"
    [mistral-7b]="mistralai/Mistral-7B-Instruct-v0.1"
    [qwen3.5-27b]="Qwen/Qwen3.5-27B"
)

# Ordered model list (llama models first)
MODEL_ORDER=("llama3.1-8b" "llama3.1-70b" "mistral-7b" "qwen3.5-27b")

# GPU config
PORT_BASE=8000
GPU_MEMORY_UTILIZATION=0.95
TENSOR_PARALLEL_SIZE=2

# Helper function to run evaluation
run_eval() {
    local script=$1
    local method=$2
    local model=$3
    local mode=$4
    local defense=${5:-}

    local model_hf="${MODELS[$model]}"
    local port=$((PORT_BASE + RANDOM % 100))
    local training_prompts="${ATTACK_METHODS[$method]}/good_prompts.csv"

    # Check if training prompts exist
    if [ ! -f "$training_prompts" ]; then
        log_error "Training prompts not found: $training_prompts"
        return 1
    fi

    log_info "Running: $method vs $model ($mode)${defense:+ with $defense defense}"

    local log_file="$LOG_DIR/eval_${method}_${model/\//}_${mode}${defense:+_vs_${defense}}_$TIMESTAMP.log"

    # Run Python script with proper argument passing (no eval needed)
    if [ -n "$defense" ]; then
        python "$script" \
            --method "$method" \
            --model "$model_hf" \
            --mode "$mode" \
            --training_prompts "$training_prompts" \
            --port "$port" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --output_dir "$OUTPUT_DIR" \
            --defense "$defense" \
            >> "$log_file" 2>&1
    else
        python "$script" \
            --method "$method" \
            --model "$model_hf" \
            --mode "$mode" \
            --training_prompts "$training_prompts" \
            --port "$port" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --output_dir "$OUTPUT_DIR" \
            >> "$log_file" 2>&1
    fi

    if [ $? -eq 0 ]; then
        log_success "$method vs $model ($mode)${defense:+ with $defense defense}"
        return 0
    else
        log_error "$method vs $model ($mode)${defense:+ with $defense defense} - see $log_file"
        return 1
    fi
}

# Phase 1: Base Attacks
phase_base() {
    print_section "PHASE 1: BASE ATTACKS (STANDARD MODE)"

    local total=0
    local successful=0

    for attack in "${!ATTACK_METHODS[@]}"; do
        for model in "${MODEL_ORDER[@]}"; do
            ((total++))
            if run_eval "evaluation_method.py" "$attack" "$model" "standard"; then
                ((successful++))
            fi
            sleep 1
        done
    done

    log_info "Phase 1 Summary: $successful/$total successful"
}

# Phase 2: Repeated Prompt
phase_repeated() {
    print_section "PHASE 2: REPEATED PROMPT MODE"

    local total=0
    local successful=0

    for attack in "${!ATTACK_METHODS[@]}"; do
        for model in "${MODEL_ORDER[@]}"; do
            ((total++))
            if run_eval "evaluation_multi_mode.py" "$attack" "$model" "repeated_prompt"; then
                ((successful++))
            fi
            sleep 1
        done
    done

    log_info "Phase 2 Summary: $successful/$total successful"
}

# Phase 3: Multi-Turn
phase_multiturn() {
    print_section "PHASE 3: MULTI-TURN MODE"

    local total=0
    local successful=0

    for attack in "${!ATTACK_METHODS[@]}"; do
        for model in "${MODEL_ORDER[@]}"; do
            ((total++))
            if run_eval "evaluation_multi_mode.py" "$attack" "$model" "multi_turn"; then
                ((successful++))
            fi
            sleep 1
        done
    done

    log_info "Phase 3 Summary: $successful/$total successful"
}

# Phase 4: PromptGuard Defense
phase_promptguard() {
    print_section "PHASE 4: DEFENSE - PROMPTGUARD (ALL MODELS)"

    local total=0
    local successful=0

    for attack in "${!ATTACK_METHODS[@]}"; do
        for model in "${MODEL_ORDER[@]}"; do
            ((total++))
            if run_eval "evaluation_method_with_defense.py" "$attack" "$model" "standard" "promptguard"; then
                ((successful++))
            fi
            sleep 1
        done
    done

    log_info "Phase 4 Summary: $successful/$total successful"
}

# Phase 5: SecAlign Defense
phase_secalign() {
    print_section "PHASE 5: DEFENSE - SECALIGN (LLAMA3.1-8B ONLY)"

    local total=0
    local successful=0

    for attack in "${!ATTACK_METHODS[@]}"; do
        ((total++))
        if run_eval "evaluation_method_with_defense.py" "$attack" "llama3.1-8b" "standard" "secalign"; then
            ((successful++))
        fi
        sleep 1
    done

    log_info "Phase 5 Summary: $successful/$total successful"
}

# Phase 6: SFT Defense
phase_sft() {
    print_section "PHASE 6: DEFENSE - SFT (LLAMA3.1-8B ONLY)"

    local total=0
    local successful=0

    for attack in "${!ATTACK_METHODS[@]}"; do
        ((total++))
        if run_eval "evaluation_method_with_defense.py" "$attack" "llama3.1-8b" "standard" "sft"; then
            ((successful++))
        fi
        sleep 1
    done

    log_info "Phase 6 Summary: $successful/$total successful"
}

# Main function
main() {
    log_info "Starting Full Evaluation Suite"
    log_info "Log file: $SUMMARY_LOG"

    local phases_to_run=()
    if [ $# -gt 0 ]; then
        phases_to_run=("$@")
    else
        phases_to_run=("base" "repeated" "multiturn" "promptguard" "secalign" "sft")
    fi

    # Run requested phases
    for phase in "${phases_to_run[@]}"; do
        case "$phase" in
            base) phase_base ;;
            repeated) phase_repeated ;;
            multiturn|multi_turn) phase_multiturn ;;
            promptguard) phase_promptguard ;;
            secalign) phase_secalign ;;
            sft) phase_sft ;;
            all)
                phase_base
                phase_repeated
                phase_multiturn
                phase_promptguard
                phase_secalign
                phase_sft
                ;;
            *)
                log_error "Unknown phase: $phase"
                echo "Valid phases: base, repeated, multiturn, promptguard, secalign, sft, all"
                exit 1
                ;;
        esac
    done

    print_section "EVALUATION SUITE COMPLETE"
    log_success "All evaluations completed!"
    log_info "Results directory: $OUTPUT_DIR"
    log_info "Summary log: $SUMMARY_LOG"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [PHASES]

Runs comprehensive evaluation of all attack methods.
Server lifecycle is automatically managed by each evaluation script.

PHASES (optional, defaults to all):
  base              - Base attacks (standard mode)
  repeated          - Repeated prompt injection
  multiturn         - Multi-turn conversations
  promptguard       - PromptGuard defense (all models)
  secalign          - SecAlign defense (llama3.1-8b only)
  sft               - SFT defense (llama3.1-8b only)
  all               - All phases (default)

Examples:
  $0                    # Run all phases
  $0 base repeated      # Run specific phases
  $0 promptguard        # Run single phase

Each Python script starts its own vLLM server, evaluates, then stops the server.
EOF
}

# Main
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
    exit 0
fi

cd "$SCRIPT_DIR"
main "$@"
