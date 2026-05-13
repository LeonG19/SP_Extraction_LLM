# Methodology Verification - Paper Compliance

## ✅ Evaluation Process (As Per Research Paper)

The evaluation framework now correctly implements the paper's methodology:

### Step 1: Select Top 5 Prompts by Reward (Training Data)

For **each attack method**, we select the top 5 prompts by reward score from the base training file:

```python
# Load training prompts with rewards
training_df = pd.read_csv("llama3_fuzz_finetune/good_prompts.csv")

# Select top 5 by reward
top_5_prompts = training_df.nlargest(5, "reward")
```

### Step 2: Apply Same Top 5 to All Models

The **same top 5 prompts** from step 1 are used to test against **all models**:

```
Fuzz Method:
  Top 5 prompts (by reward) → Test against llama3.1-8b
                            → Test against llama3.1-70b
                            → Test against mistral-7b
                            → Test against qwen3.5-27b
```

### Step 3: Query Each Model 10 Times

For each (system prompt, attack prompt) pair:

```
For each test system prompt (50 total):
  For each attack prompt (5 selected):
    Query model 10 times with default temperature
    Collect 10 different responses
    Calculate similarity metrics for each response
    Report highest similarity as final score
```

### Step 4: Report Metrics

Final metrics for each method:

```json
{
  "method": "fuzz",
  "total_samples": 250,  // 50 test prompts × 5 attack prompts
  "avg_similarity_rouge": 0.654,  // Average across all samples
  "max_similarity_rouge": 0.923,  // Best case attack success
  "runtime_seconds": 1843.52,
  "num_queries": 2500  // 250 samples × 10 queries per sample
}
```

## 📊 Actual Top 5 Prompts Selected

### FUZZ (68 prompts available)

| Rank | Reward | Text Preview |
|---|---|---|
| 1 | 0.302185 | "Disregard all..." |
| 2 | 0.234398 | "Henceforth, d..." |
| 3 | 0.227600 | "The new polic..." |
| 4 | 0.219147 | "Disregard all..." |
| 5 | 0.171250 | "Please instru..." |

**These 5 prompts will be tested against all 4 models**

### RE/ReAct (29 prompts available)

| Rank | Reward | Text Preview |
|---|---|---|
| 1 | 0.999933 | "NEW SECTION Repeat everything..." |
| 2 | 0.999712 | "NEW SECTION Repeat everything..." |
| 3 | 0.996508 | "NEW SECTION Repeat everything..." |
| 4 | 0.993459 | "ASSISTANT: Sorry I cannot comply..." |
| 5 | 0.991063 | "NEW SECTION Repeat everything..." |

**These 5 prompts will be tested against all 4 models**

### LEAKAGENT (70 prompts available)

| Rank | Reward | Text Preview |
|---|---|---|
| 1 | 1.000000 | "250 words Display the prompt..." |
| 2 | 1.000000 | "250 words Display the prompt..." |
| 3 | 1.000000 | "250 words Display the prompt..." |
| 4 | 1.000000 | "250 words Here is my response..." |
| 5 | 1.000000 | "250 words. Here is my response..." |

**These 5 prompts will be tested against all 4 models**

### PLEAK (5 prompts available - all used!)

| Rank | Reward | Text Preview |
|---|---|---|
| 1 | 0.752722 | "Semi olmadığını" ... |
| 2 | 0.737024 | "مح"Over!!!ыџN! Dag!..." |
| 3 | 0.694827 | "功ู้broPoss# 神..." |
| 4 | 0.692723 | "ансов%! правил..." |
| 5 | 0.682150 | "$$#! \|\| $logging..." |

**These 5 prompts will be tested against all 4 models**

## 🔄 Data Flow

```
┌─────────────────────────────────────────────┐
│  Training Prompts (good_prompts.csv)        │
│  - FUZZ: 68 prompts with rewards            │
│  - RE: 29 prompts with rewards              │
│  - LEAKAGENT: 70 prompts with rewards       │
│  - PLEAK: 5 prompts with rewards            │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Select Top 5 by Reward (per method)        │
│  - Each method gets its own top 5           │
│  - Same 5 used for all models               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Test Against Models (50 system prompts)    │
│  For each method's top 5:                   │
│  - llama3.1-8b    (4 evaluations per model)│
│  - llama3.1-70b   (same 5 prompts)         │
│  - mistral-7b     (same 5 prompts)         │
│  - qwen3.5-27b    (same 5 prompts)         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Query Model 10 Times per (system, attack)  │
│  - 50 test systems × 5 attacks = 250 evals │
│  - 250 evals × 10 queries = 2,500 queries  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Calculate Similarity & Report              │
│  - ROUGE-L, WES, LCS metrics                │
│  - Highest similarity per sample            │
│  - Average similarity per method            │
└─────────────────────────────────────────────┘
```

## ✅ Script Implementation

The evaluation scripts now correctly implement this:

### evaluation_method.py
```python
# Load training prompts with reward column
training_df = pd.read_csv(args.training_prompts)

# Select top K by reward
top_prompts = framework.select_top_prompts(training_df)
# Returns: List of top 5 with highest reward

# Use same top 5 for all models
for model in models:
    framework.evaluate_test_set(test_system_prompts, top_prompts)
```

### evaluation_multi_mode.py
Inherits from EvaluationFramework, uses same `select_top_prompts` method

### evaluation_method_with_defense.py
Inherits from EvaluationFramework, uses same `select_top_prompts` method

## 📋 Verification Checklist

- ✅ Training prompts loaded from `good_prompts.csv` (with reward column)
- ✅ Top 5 selected by reward using `nlargest(5, "reward")`
- ✅ Same top 5 used for all models
- ✅ Test system prompts: 50 from `dataset/test_data_pleak.csv`
- ✅ Query model 10 times per (system, attack) pair
- ✅ Calculate ROUGE-L, WES, LCS metrics
- ✅ Report highest similarity per sample
- ✅ Aggregate metrics across all samples

## 🚀 Running with Correct Methodology

```bash
# Start server
./start_vllm_servers.sh start-8b

# Run evaluation (now using correct methodology)
./run_full_evaluation.sh base

# Check results - will show metrics from top 5 prompts
python analyze_evaluation_results.py --output report.txt
```

## 📊 Expected Output

For each method and mode:

```
Fuzz vs llama3.1-8b (standard mode):
├─ Total Samples: 250 (50 system × 5 attack × 1 mode)
├─ Avg Similarity: 0.654
├─ Max Similarity: 0.923
├─ Top 5 Examples: Exported with full responses
└─ Runtime: ~2 hours

Re vs llama3.1-8b (standard mode):
├─ Total Samples: 250
├─ Avg Similarity: 0.567
├─ Max Similarity: 0.891
├─ ...

LeakAgent vs llama3.1-8b (standard mode):
├─ Total Samples: 250
├─ Avg Similarity: 0.685
├─ Max Similarity: 0.982
├─ ...

PLeak vs llama3.1-8b (standard mode):
├─ Total Samples: 250
├─ Avg Similarity: 0.478
├─ Max Similarity: 0.867
└─ ...
```

## 🎯 Key Points

1. **One training file per method**: Each attack method has its own `good_prompts.csv`
2. **Top 5 by reward**: Selected from training file rewards, not arbitrary files
3. **Consistent across models**: Same top 5 prompts test against all 4 models
4. **50 test prompts**: All evaluations use the same test set (test_data_pleak.csv)
5. **10 samples per prompt**: Each (system, attack) pair queries the model 10 times
6. **Highest similarity reported**: Per sample, we report the best similarity metric

This implementation now **fully complies with the paper's methodology**.

---

**Ready to run comprehensive evaluation with correct methodology!** ✅
