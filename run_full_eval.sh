#!/bin/bash
set -e  # Exit on error

echo "======================================"
echo "Starting comprehensive evaluation"
echo "======================================"

echo ""
echo "Phase 1: Repeated-Prompt Evaluation"
echo "Models: llama3.1-8b, llama3.1-70b, mistral-7b, qwen3.5-27b"
echo "Methods: fuzz, re, leakagent, pleak"
echo "N sweep: [1, 2] (prompt repeated 1x and 2x)"
echo "Samples: 5 per (attack_prompt, system_prompt) pair"
echo "GPU Memory: 0.7 × 2 GPUs (out of 3 available)"
echo ""
python evaluate_all_methods.py \
  --methods fuzz,re,leakagent,pleak \
  --exclude-models gpt \
  --repeated-prompt \
  --repeated-prompt-max-n 2 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2

echo ""
echo "======================================"
echo "Phase 1 Complete - Results in evaluation_results/repeated_prompt/"
echo "======================================"
echo ""
sleep 5

echo "Phase 2: Multi-Turn Evaluation"
echo "Models: llama3.1-8b, llama3.1-70b, mistral-7b, qwen3.5-27b"
echo "Methods: fuzz, re, leakagent, pleak"
echo "Samples: 1 per (attack_prompt, system_prompt) pair"
echo "GPU Memory: 0.7 × 2 GPUs (out of 3 available)"
echo ""
python evaluate_all_methods.py \
  --methods fuzz,re,leakagent,pleak \
  --exclude-models gpt \
  --multi-turn \
  --multi-turn-samples 1 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2

echo ""
echo "======================================"
echo "All evaluations complete!"
echo "======================================"
echo ""
echo "Results saved to:"
echo "  - evaluation_results/multi_turn/"
echo "  - evaluation_results/repeated_prompt/"
echo "Logs: evaluation_logs/"
