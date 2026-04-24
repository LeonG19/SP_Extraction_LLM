# PLeak: Prompt Leaking Attacks Implementation Guide

This document explains how to use the PLeak white-box attack implementation in this repository. PLeak is a gradient-based attack method that extracts system prompts and training data from language models.

## Overview

**PLeak** is a white-box attack that:
- Uses gradient information from the target model to craft adversarial prompts
- Optimizes adversarial tokens to maximize the likelihood of extracting system prompts
- Works with any open-source model available on Hugging Face
- Generates adversarial triggers that force the target LLM to reveal its hidden instructions

## Key Differences from Other Baselines

| Method | Type | Access | Approach |
|--------|------|--------|----------|
| **PLeak** | White-box | Full model access | Gradient-based token optimization |
| **PromptFuzz** | Black-box | API only | Genetic algorithm + LLM mutations |
| **ReAct-Leak** | Black-box | API only | Reasoning-based iterative refinement |
| **LeakAgent** | Black-box RL | API only | PPO-based RL training |

## Installation

PLeak is already integrated into your rlagent repository. No additional installation needed beyond the existing dependencies.

## Usage

### Method 1: Standalone Training Script

Train PLeak to generate adversarial triggers:

```bash
python train_pleak.py \
  --model llama3 \
  --dataset_path train_data_pleak.csv \
  --optim_token_length 20 \
  --num_triggers 5 \
  --num_iterations 100 \
  --output_dir pleak_results
```

**Parameters:**
- `--model`: Target model (llama3, llama, mixtral, etc.)
- `--dataset_path`: CSV file with system prompts to extract
- `--optim_token_length`: Number of adversarial tokens (higher = more flexible but slower)
- `--num_triggers`: Number of different triggers to generate
- `--num_iterations`: Optimization iterations per trigger (higher = better but slower)
- `--top_k`: Top-k candidates for token replacement (default: 50)
- `--temperature`: Temperature for candidate selection (default: 0.5)
- `--use_english_vocab`: Restrict to English tokens only
- `--output_dir`: Where to save results

### Method 2: Pipeline Integration

Run PLeak as part of your training pipeline:

```bash
python pipeline.py \
  --method pleak \
  --target_model llama3 \
  --prompts_data_path train_data_pleak.csv \
  --optim_token_length 20 \
  --num_iterations 100
```

### Method 3: Programmatic Usage

```python
from attacks.token_level.whitebox.PLeak import PLeakAttack
import pandas as pd

# Load system prompts to attack
system_prompts = pd.read_csv("train_data_pleak.csv")["text"].tolist()

# Initialize attack
attack = PLeakAttack(
    model="llama3",
    optim_token_length=20,
    num_trigger=5,
    num_iterations=100,
)

# Run attack
triggers = attack.train(system_prompts)

# Access results
for result in attack.generated_triggers:
    print(f"Trigger: {result['trigger']}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Reward: {1.0 / (1.0 + result['loss']):.4f}")
```

## How PLeak Works

### 1. Initialization
- Initialize random adversarial tokens (or from a seed)
- Load the target language model with 4-bit quantization for efficiency

### 2. Gradient-Based Optimization
For each iteration:

a) **Loss Computation**
   - Create prompts: [System Prompt] + [Adversarial Trigger]
   - Ask model to output the system prompt
   - Compute cross-entropy loss (lower = better at leaking)

b) **Gradient Computation**
   - Backpropagate loss to embedding layer
   - Compute gradients for each token position

c) **Token Replacement**
   - Use gradients to find best token candidates
   - HotFlip mechanism: find tokens whose embeddings align with negative gradient
   - Replace tokens that reduce loss the most

d) **Update**
   - Keep tokens that improve extraction capability
   - Iterate until convergence or max iterations reached

### 3. Output
- Generate adversarial triggers that maximize system prompt extraction
- Save as CSV with format compatible with LeakAgent evaluation

## Performance Considerations

### Memory Usage
- Default: ~15 GB with 4-bit quantization
- Adjust `--optim_token_length` if OOM (out of memory)
- Use smaller models (e.g., 7B instead of 70B)

### Speed
- Optimization iterations: 1-5 minutes per trigger depending on model size
- Total time for 5 triggers with 100 iterations: 5-25 minutes

### Quality
- More iterations → better triggers but slower
- Longer tokens → more powerful but slower
- More system prompts in dataset → better generalization

## Example Results Format

Results are saved as `good_prompts.csv`:

```
text,loss,reward
"adversarial trigger 1",0.5,0.667
"adversarial trigger 2",0.75,0.571
"adversarial trigger 3",0.3,0.769
...
```

The "reward" is computed as `1.0 / (1.0 + loss)` for compatibility with LeakAgent format.

## Evaluating Generated Triggers

Use the standard evaluation script to test your PLeak triggers:

```bash
python evaluate_task.py \
  --model_name llama3 \
  --prompts_data_path pleak_results/good_prompts.csv \
  --n_samples 5 \
  --dataset_path test_data_pleak.csv
```

This will:
- Test each adversarial trigger against the dataset
- Compute metrics: LCS, similarity, ROUGE, WES
- Show how effective the triggers are at extracting prompts

## Comparing with Other Baselines

Run all baselines for comparison:

```bash
# PromptFuzz
python save_baseline_results.py --method fuzz --output_dir fuzz_results

# ReAct-Leak
python save_baseline_results.py --method re --output_dir re_results

# PLeak (white-box)
python train_pleak.py --output_dir pleak_results

# Evaluate all
python evaluate_task.py --prompts_data_path fuzz_results/good_prompts.csv
python evaluate_task.py --prompts_data_path re_results/good_prompts.csv
python evaluate_task.py --prompts_data_path pleak_results/good_prompts.csv
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--optim_token_length` (e.g., 15 instead of 20)
- Use smaller model (e.g., `--model llama` instead of `llama3`)
- Reduce batch size in dataset

### Slow Training
- Reduce `--num_iterations` (e.g., 50 instead of 100)
- Reduce `--optim_token_length`
- Use smaller model

### Poor Quality Triggers
- Increase `--num_iterations`
- Increase `--optim_token_length`
- Use larger model
- Ensure `--top_k` is large enough (≥ 50)

### CUDA Out of Memory During Evaluation
- Reduce `--n_samples` in evaluate_task.py
- Evaluate on fewer prompts

## Research Context

PLeak is based on the paper:
> "Prompts Leaking Attacks against Large Language Model Applications"

Key contributions:
- First white-box prompt extraction attack using gradients
- Works on both system prompts and training data
- Demonstrates vulnerability of LLMs to gradient-based attacks

## Security Considerations

PLeak is designed for:
- ✅ Authorized security research
- ✅ Evaluating model vulnerabilities  
- ✅ Testing defensive mechanisms
- ✅ Educational purposes

This tool should only be used on models you have permission to test against (your own models or explicit authorization).

## References

- Original PLeak Repository: https://github.com/BHui97/PLeak
- LeakAgent Paper: https://arxiv.org/pdf/2412.05734
- HotFlip Attack: https://arxiv.org/abs/1712.06032
