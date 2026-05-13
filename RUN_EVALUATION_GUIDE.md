# Full Evaluation Suite - Quick Start Guide

## Overview

The evaluation suite runs comprehensive experiments evaluating all attack methods (pleak, fuzz, re, leakagent) across:

1. **Base Attacks** - Direct attack with default temperature
2. **Prompt Repetition** - Same prompt applied twice to measure cumulative effects
3. **Multi-Turn** - Different prompts in conversation setting
4. **Defenses** - PromptGuard (all models), SecAlign (llama3.1-8b), SFT/LeakAgent-D (llama3.1-8b)

## Prerequisites

1. **vLLM Server** - Start before running evaluations
2. **Python Dependencies** - Installed (pandas, numpy, torch, etc.)
3. **Training Prompts** - Generated attack prompts in model-specific directories
4. **Disk Space** - ~10-20GB for full evaluation results

## Quick Start (5 minutes)

### Step 1: Start vLLM Server

```bash
# Start server for llama3.1-8b (primary model)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.5 &

# Wait for server to be ready
sleep 30
curl http://localhost:8000/health
```

### Step 2: Run Evaluation Suite

```bash
# Run all evaluations
cd /home/lgarza3/projects/rlagent
./run_full_evaluation.sh all

# Or run specific phases
./run_full_evaluation.sh base repeated multiturn promptguard secalign sft
```

### Step 3: Analyze Results

```bash
# Generate comprehensive report
python analyze_evaluation_results.py

# Or save to file
python analyze_evaluation_results.py --output evaluation_report.txt

# Or save summary to CSV
python analyze_evaluation_results.py --csv summary.csv
```

## Detailed Usage

### Running Specific Phases Only

```bash
# Run only base attacks (faster for initial testing)
./run_full_evaluation.sh base

# Run base attacks + repeated prompt (test persistence)
./run_full_evaluation.sh base repeated

# Run only defense evaluations
./run_full_evaluation.sh promptguard secalign sft

# Run multi-turn conversations
./run_full_evaluation.sh multiturn
```

### Configuration

Edit variables in `run_full_evaluation.sh`:

```bash
# Number of samples per prompt (default: 10)
N_SAMPLES=10

# Number of top prompts to select (default: 5)
TOP_K=5

# Server configuration
SERVER_URL="http://localhost:8000/v1"
API_KEY="not-set"

# Output directory
OUTPUT_DIR="evaluation_results"
```

Or set environment variables:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your-key-here"
./run_full_evaluation.sh all
```

## What Each Phase Does

### Phase 1: Base Attacks (Standard Mode)
- **What**: Direct attack with adversarial prompt
- **How**: Query model once per (system prompt, attack prompt) pair
- **Time**: ~2-3 hours (all attacks × all models)
- **Output**: `evaluation_results/standard/metrics_*.json`

```bash
# Run only this phase
./run_full_evaluation.sh base
```

### Phase 2: Prompt Repetition (Repeated Prompt Mode)
- **What**: Apply same attack prompt twice
- **How**: First query, then repeat same prompt to measure persistence
- **Time**: ~4-6 hours (two turns for each example)
- **Output**: `evaluation_results/repeated_prompt/metrics_*.json` with delta metrics
- **Key Metrics**: avg_first_similarity, avg_second_similarity, avg_delta_similarity

```bash
# Run only this phase
./run_full_evaluation.sh repeated
```

### Phase 3: Multi-Turn Conversations
- **What**: Different attack prompts in conversation
- **How**: First turn with one attack, second turn with different attack
- **Time**: ~4-6 hours
- **Output**: `evaluation_results/multi_turn/metrics_*.json` with delta metrics
- **Key Metrics**: Shows cumulative attack effectiveness

```bash
# Run only this phase
./run_full_evaluation.sh multiturn
```

### Phase 4: PromptGuard Defense (All Models)
- **What**: Evaluate attacks against PromptGuard defense
- **Models**: All (llama3.1-8b, llama3.1-70b, mistral-7b, qwen3.5-27b)
- **Time**: ~2-3 hours
- **Output**: `evaluation_results/standard/metrics_*_vs_promptguard_*.json`
- **Key Metric**: Compare avg_similarity vs base attacks to measure effectiveness

```bash
# Run only this phase
./run_full_evaluation.sh promptguard
```

### Phase 5: SecAlign Defense (llama3.1-8b only)
- **What**: Evaluate attacks against SecAlign defense
- **Models**: llama3.1-8b only (fine-tuned)
- **Time**: ~30-45 minutes
- **Output**: `evaluation_results/standard/metrics_*_vs_secalign_*.json`

```bash
# Run only this phase
./run_full_evaluation.sh secalign
```

### Phase 6: SFT/LeakAgent-D Defense (llama3.1-8b only)
- **What**: Evaluate attacks against SFT defense
- **Models**: llama3.1-8b only (fine-tuned)
- **Time**: ~30-45 minutes
- **Output**: `evaluation_results/standard/metrics_*_vs_sft_*.json`

```bash
# Run only this phase
./run_full_evaluation.sh sft
```

## Understanding the Output

### Metrics Files

Each `metrics_*.json` contains:

```json
{
  "method": "leakagent",
  "model": "llama3.1-8b",
  "mode": "standard",
  "defense": null,
  "total_samples": 250,
  "avg_similarity_rouge": 0.685,
  "max_similarity_rouge": 0.982,
  "runtime_seconds": 1843.52,
  "num_queries": 2500
}
```

**Key Fields**:
- **avg_similarity_rouge**: Average success rate (0-1, higher = more successful)
- **max_similarity_rouge**: Best-case attack success
- **total_samples**: Number of (system_prompt, attack_prompt) pairs tested
- **runtime_seconds**: Total evaluation time

### Top 5 Examples Files

`top5_examples_*.json` contains the 5 most successful attacks with:
- System prompt (target instruction)
- Attack prompt (adversarial input)
- 10 responses from target model
- Similarity scores for each response

### Examples Files (JSONL)

`examples_*.jsonl` has one JSON object per line with all evaluation examples.

## Analysis Commands

### View All Metrics as Table

```bash
# Print all standard mode results
python analyze_evaluation_results.py

# Print and save to file
python analyze_evaluation_results.py --output report.txt

# Save to CSV for spreadsheet analysis
python analyze_evaluation_results.py --csv summary.csv
```

### Manual Analysis

```bash
# View metrics for specific method
jq '.avg_similarity_rouge' evaluation_results/standard/metrics_leakagent_*.json | sort -nr

# Compare attack vs defense
echo "Base attacks:"
jq '.avg_similarity_rouge' evaluation_results/standard/metrics_leakagent_llama3.1-8b_*.json
echo "With PromptGuard:"
jq '.avg_similarity_rouge' evaluation_results/standard/metrics_leakagent_*_vs_promptguard_*.json

# View top 5 leaked prompts
python3 << 'EOF'
import json
with open('evaluation_results/standard/top5_examples_leakagent_llama3.1-8b_*.json') as f:
    data = json.load(f)
    for i, ex in enumerate(data[:3], 1):
        print(f"\nExample {i}:")
        print(f"  System: {ex['system_prompt'][:80]}")
        print(f"  Attack: {ex['attack_prompt'][:80]}")
        print(f"  Success: {ex['highest_similarity']:.4f}")
        print(f"  Response: {ex['responses'][0][:100]}")
EOF
```

## Expected Results

### Attack Success Rates (Typical)
- **Base**: 0.5-0.7 (50-70% success)
- **Repeated Prompt**: 0.6-0.8 (10-20% improvement)
- **Multi-Turn**: 0.6-0.8 (similar to repeated)

### Defense Effectiveness
- **PromptGuard**: 70-90% reduction
- **SecAlign**: 50-70% reduction
- **SFT**: 80-95% reduction

### Method Ranking (Typical)
1. LeakAgent - highest success rate
2. PromptFuzz - close to LeakAgent
3. ReAct - moderate success
4. PLeak - lower success (but fastest)

### Model Vulnerability (Typical)
1. Llama3.1-8B - most vulnerable
2. Mistral-7B - moderate
3. Qwen3.5-27B - similar
4. Llama3.1-70B - most robust

## Troubleshooting

### Server Connection Issues

```bash
# Check if server is running
curl http://localhost:8000/health

# Check logs
tail -f vllm_server.log

# Restart on different port
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8001 &

# Update script to use new port
export OPENAI_BASE_URL="http://localhost:8001/v1"
```

### Out of Memory

```bash
# Reduce scale to test
./run_full_evaluation.sh --n_samples 5 --top_k 2 base

# Or modify script:
# Edit N_SAMPLES=5 and TOP_K=2 in run_full_evaluation.sh
```

### Training Prompts Not Found

```bash
# Check if training prompts exist
ls -la pleak_results/llama3.1-8b_prompts.csv
ls -la llama3_fuzz_finetune/llama3.1-8b_prompts.csv
ls -la llama3_re_finetune/llama3.1-8b_prompts.csv
ls -la llama3_rl_finetune/llama3.1-8b_prompts.csv

# If missing, generate them first with respective methods
```

## Estimating Total Time

```
Phase 1 (Base):           3 hours
Phase 2 (Repeated):       5 hours
Phase 3 (Multi-Turn):     5 hours
Phase 4 (PromptGuard):    3 hours
Phase 5 (SecAlign):       1 hour
Phase 6 (SFT):            1 hour
-------
Total:                   ~18 hours

With parallelization:    Could be 8-10 hours
```

## Monitoring Progress

```bash
# Watch evaluation logs
tail -f evaluation_logs/evaluation_summary_*.log

# Check completed evaluations
ls evaluation_results/standard/metrics_*.json | wc -l
# Expected: 16 (4 methods × 4 models)

ls evaluation_results/repeated_prompt/metrics_*.json | wc -l
# Expected: 16

ls evaluation_results/multi_turn/metrics_*.json | wc -l
# Expected: 16
```

## Parallel Execution

To run multiple model servers in parallel:

```bash
# Terminal 1: llama3.1-8b on port 8000
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 &

# Terminal 2: mistral-7b on port 8001
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --port 8001 &

# Modify script to use different servers for different models
# Or run evaluations selectively:
./run_full_evaluation.sh base  # llama3.1-8b tests
# Then switch server and run again
```

## Next Steps

1. **Run Evaluation Suite**: `./run_full_evaluation.sh all`
2. **Monitor Progress**: Watch logs and check completed files
3. **Analyze Results**: `python analyze_evaluation_results.py --output report.txt`
4. **Review Top Examples**: Check top 5 examples JSON files
5. **Compare Methods**: Look at method rankings in report
6. **Assess Defenses**: Review defense effectiveness section

## Support

For issues:
1. Check logs in `evaluation_logs/`
2. Review troubleshooting section above
3. Verify vLLM server is running
4. Check training prompts exist
5. Ensure sufficient disk space

## Files Generated

```
evaluation_results/
├── standard/                     (base attacks)
│   ├── metrics_*.json
│   ├── top5_examples_*.json
│   └── examples_*.jsonl
├── repeated_prompt/              (persistence)
│   ├── metrics_*.json
│   ├── top5_examples_*.json
│   └── examples_*.jsonl
└── multi_turn/                   (conversation)
    ├── metrics_*.json
    ├── top5_examples_*.json
    └── examples_*.jsonl

evaluation_logs/
└── evaluation_summary_*.log

evaluation_report.txt             (generated by analysis)
```

---

**Happy Evaluating!** 🚀

For detailed information about the evaluation framework, see:
- `EVALUATION_README.md` - Framework overview
- `EVALUATION_GUIDE.md` - Detailed module documentation
- `EVALUATION_ARCHITECTURE.md` - Technical design
- `EVALUATION_OUTPUT_EXAMPLES.md` - Output format examples
