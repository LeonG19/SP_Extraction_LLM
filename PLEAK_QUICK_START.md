# PLeak Quick Start Guide

## Installation
No additional installation needed - PLeak is integrated into rlagent.

## One-Liner Examples

### Fastest Demo (5 min)
```bash
python train_pleak.py --num_iterations 20 --num_triggers 2
```

### Standard Training (20 min)
```bash
python train_pleak.py --num_iterations 100 --num_triggers 5
```

### Full Training with Dataset (45 min)
```bash
python train_pleak.py \
  --optim_token_length 20 \
  --num_iterations 150 \
  --num_triggers 5 \
  --output_dir pleak_full_results
```

### With Different Model
```bash
# Using Llama 2
python train_pleak.py --model llama --num_iterations 100

# Using Mixtral
python train_pleak.py --model mixtral --num_iterations 100
```

### English-Only Tokens
```bash
python train_pleak.py --num_iterations 100 --use_english_vocab
```

## Comparison with Baselines

```bash
# Run all baselines
python train_pleak.py --output_dir pleak_results
python save_baseline_results.py --method fuzz --output_dir fuzz_results
python save_baseline_results.py --method re --output_dir re_results

# Evaluate all
for dir in pleak_results fuzz_results re_results; do
  python evaluate_task.py --prompts_data_path $dir/good_prompts.csv
done
```

## Programmatic API

```python
from attacks.token_level.whitebox.PLeak import PLeakAttack
import pandas as pd

# Load data
prompts = pd.read_csv("train_data_pleak.csv")["text"].tolist()

# Create attack
attack = PLeakAttack(
    model="llama3",
    optim_token_length=20,
    num_trigger=5,
    num_iterations=100,
)

# Run attack
results = attack.train(prompts)

# Save results
df = pd.DataFrame([
    {"text": r["trigger"], "loss": r["loss"], "reward": 1/(1+r["loss"])}
    for r in results
])
df.to_csv("my_results.csv", index=False)
```

## Common Parameters

| Parameter | Default | Recommendation |
|-----------|---------|-----------------|
| `--model` | llama3 | Change for different models |
| `--optim_token_length` | 20 | 15-30 (more = better but slower) |
| `--num_iterations` | 100 | 50-200 (more = better but slower) |
| `--num_triggers` | 5 | 3-10 triggers to generate |
| `--top_k` | 50 | 30-100 candidates to consider |
| `--temperature` | 0.5 | 0.1-1.0 (lower = more focused) |
| `--use_english_vocab` | False | Add flag for readable-only tokens |

## Troubleshooting

### Out of Memory
```bash
# Use smaller model
python train_pleak.py --model llama

# Reduce token length
python train_pleak.py --optim_token_length 15

# Reduce iterations
python train_pleak.py --num_iterations 50
```

### Slow Execution
```bash
# Quick demo
python train_pleak.py --num_iterations 30 --num_triggers 2

# Reduce search space
python train_pleak.py --top_k 30
```

### Check Installation
```bash
python -c "from attacks.token_level.whitebox.PLeak import PLeakAttack; print('✓ PLeak ready')"
```

## Output Files

After running `python train_pleak.py --output_dir pleak_results`:

```
pleak_results/
└── good_prompts.csv    # Results in LeakAgent format
```

Format:
```
text,loss,reward
"trigger_1_text",0.45,0.690
"trigger_2_text",0.60,0.625
```

## Evaluation

```bash
python evaluate_task.py --prompts_data_path pleak_results/good_prompts.csv
```

Outputs metrics:
- **LCS**: Longest common subsequence matching
- **SIM**: Semantic similarity
- **ROUGE**: Recall-oriented understudy for gisting evaluation
- **WES**: Word error similarity

## Integration with Pipeline

```bash
python pipeline.py \
  --method pleak \
  --target_model llama3 \
  --optim_token_length 20 \
  --num_iterations 100
```

## Key Differences from Baselines

| Aspect | PLeak | PromptFuzz | ReAct |
|--------|-------|-----------|-------|
| Access | White-box | Black-box | Black-box |
| Gradients | ✅ Yes | ❌ No | ❌ No |
| Speed | Fast | Slow | Medium |
| Quality | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## Next Steps

1. **Learn More**: Read `PLEAK_GUIDE.md`
2. **Try Examples**: Run `python examples_pleak.py`
3. **Your Dataset**: `python train_pleak.py --dataset_path your_data.csv`
4. **Compare**: Run all baselines and evaluate

---

**Status**: ✅ Ready to use
**Last Updated**: 2024
**Questions**: See PLEAK_GUIDE.md for detailed documentation
