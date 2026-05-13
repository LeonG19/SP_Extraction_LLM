# Path Verification Report

## ✅ All Paths Verified and Corrected

### Training Prompts Paths

#### PromptFuzz (Fuzz) Method
```
✓ llama3_fuzz_finetune/good_prompts.csv                          (fallback)
✓ llama3_fuzz_finetune/good_prompts_eval_top_5_Llama-3.1-8B-Instruct.csv
✓ llama3_fuzz_finetune/good_prompts_eval_top_5_Meta-Llama-3.1-70B-Instruct-AWQ-INT4.csv
✓ llama3_fuzz_finetune/good_prompts_eval_top_5_Mistral-7B-Instruct-v0.3.csv
✓ llama3_fuzz_finetune/good_prompts_eval_top_5_Qwen3.5-27B.csv
```

#### ReAct (Re) Method
```
✓ llama3_re_finetune/good_prompts.csv                            (fallback)
✓ llama3_re_finetune/good_prompts_eval_top_5_Llama-3.1-8B-Instruct.csv
✓ llama3_re_finetune/good_prompts_eval_top_5_Meta-Llama-3.1-70B-Instruct-AWQ-INT4.csv
✓ llama3_re_finetune/good_prompts_eval_top_5_Mistral-7B-Instruct-v0.3.csv
✓ llama3_re_finetune/good_prompts_eval_top_5_Qwen3.5-27B.csv
```

#### LeakAgent (RL) Method
```
✓ llama3_rl_finetune/good_prompts.csv                            (fallback)
✓ llama3_rl_finetune/good_prompts_eval_top_5_Llama-3.1-8B-Instruct.csv
✓ llama3_rl_finetune/good_prompts_eval_top_5_Meta-Llama-3.1-70B-Instruct-AWQ-INT4.csv
✓ llama3_rl_finetune/good_prompts_eval_top_5_Mistral-7B-Instruct-v0.3.csv
✓ llama3_rl_finetune/good_prompts_eval_top_5_Qwen3.5-27B.csv
```

#### PLeak Method
```
✓ pleak_results/good_prompts.csv                                 (only version)
```

### Test Data Path
```
✓ dataset/test_data_pleak.csv                                    (50 system prompts)
```

## Defense Paths

### PromptGuard
```
✓ prompt_guard.py (local implementation)
  - Jailbreak detection model
  - Available for all models
```

### SecAlign
```
✓ defenses/secalign/secalign.py
  - Loads models from HuggingFace
  - Models: RedAgent/SecAlign-*
  - Tested on: llama3.1-8b only (as per setup)
```

### SFT (LeakAgent-D Defense)
```
✓ defenses/SFTDefense.py
  - Supervised fine-tuned model
  - Tested on: llama3.1-8b only (as per setup)
  - Fine-tuned with LeakAgent prompts
```

## Script Corrections Made

### Changed in `run_full_evaluation.sh`

**OLD (Incorrect):**
```bash
local training_prompts="${method_dir}/${model}_prompts.csv"
# Would look for: llama3_fuzz_finetune/llama3.1-8b_prompts.csv ❌
```

**NEW (Correct):**
```bash
# Function now maps model names correctly and uses evaluation files
local training_prompts=$(get_training_prompts_path "$method" "$model")

# Maps:
# llama3.1-8b      → Llama-3.1-8B-Instruct
# llama3.1-70b     → Meta-Llama-3.1-70B-Instruct-AWQ-INT4
# mistral-7b       → Mistral-7B-Instruct-v0.3
# qwen3.5-27b      → Qwen3.5-27B

# Uses: llama3_fuzz_finetune/good_prompts_eval_top_5_Llama-3.1-8B-Instruct.csv ✓
```

## Model Name Mappings

The script now correctly maps short names to the evaluation file names:

| Short Name | File Name in Dataset |
|---|---|
| `llama3.1-8b` | `Llama-3.1-8B-Instruct` |
| `llama3.1-70b` | `Meta-Llama-3.1-70B-Instruct-AWQ-INT4` |
| `mistral-7b` | `Mistral-7B-Instruct-v0.3` |
| `qwen3.5-27b` | `Qwen3.5-27B` |

## Fallback Logic

The script implements smart fallback:
1. **First try**: Model-specific eval file (e.g., `good_prompts_eval_top_5_Llama-3.1-8B-Instruct.csv`)
2. **Fallback**: Generic `good_prompts.csv` file

This ensures evaluations work even if model-specific files are missing.

## Verification Checklist

```
Training Prompts:
  ✓ Fuzz:      5 files available (generic + 4 model-specific)
  ✓ Re:        5 files available (generic + 4 model-specific)
  ✓ LeakAgent: 9 files available (generic + 4 model-specific + epochs)
  ✓ PLeak:     1 file available (generic only)

Test Data:
  ✓ test_data_pleak.csv: 50 system prompts

Defenses:
  ✓ PromptGuard: prompt_guard.py (local)
  ✓ SecAlign:    defenses/secalign/secalign.py
  ✓ SFT:         defenses/SFTDefense.py

Models:
  ✓ All 4 models supported for standard evaluations
  ✓ SecAlign & SFT: llama3.1-8b only (fine-tuned)
  ✓ PromptGuard: All models supported
```

## What the Script Does Now

### 1. Get Training Prompts
```python
# For each attack method and model combination:
training_prompts = get_training_prompts_path(method, model)

# Example for leakagent + llama3.1-8b:
# Returns: llama3_rl_finetune/good_prompts_eval_top_5_Llama-3.1-8B-Instruct.csv

# Example for pleak (all models):
# Returns: pleak_results/good_prompts.csv
```

### 2. Load Test Data
```python
# All evaluations use:
test_system_prompts = pd.read_csv("dataset/test_data_pleak.csv")["text"].tolist()
# 50 system prompts for testing
```

### 3. Load Defense Models
```python
# PromptGuard: Loaded from prompt_guard.py (works for all models)
# SecAlign: Loaded from defenses/secalign/secalign.py (llama3.1-8b only)
# SFT: Loaded from defenses/SFTDefense.py (llama3.1-8b only)
```

## File Statistics

```
Attack Methods:
  fuzz:       5 CSV files (78KB base, ~500KB eval files each)
  re:         5 CSV files (12KB base, ~217KB eval files each)
  leakagent:  9 CSV files (15KB base, ~171KB eval files each)
  pleak:      1 CSV file (490B base only)

Test Data:
  system prompts:  50 prompts (24KB total)

Defenses:
  prompt_guard:    ~10KB Python implementation
  secalign:        Multiple model options from HuggingFace
  sft:             Fine-tuned on LeakAgent data
```

## Ready to Run

All paths have been verified and corrected. The script is now ready to run:

```bash
./run_full_evaluation.sh base
```

The script will:
1. ✅ Correctly find all training prompts
2. ✅ Correctly find test data
3. ✅ Correctly load defense models
4. ✅ Run evaluations with proper paths
5. ✅ Generate results in correct output directory

## Troubleshooting

### If evaluation fails to find training prompts:

```bash
# Check what files actually exist
ls llama3_*_finetune/*.csv
ls pleak_results/*.csv

# The script now logs which file it's trying to load
tail -f evaluation_logs/evaluation_summary_*.log
```

### If defenses fail to load:

```bash
# Check defense paths
ls -la defenses/*.py
python -c "import prompt_guard; print('PromptGuard OK')"
python -c "from defenses.secalign import secalign; print('SecAlign OK')"
```

---

**All paths verified and corrected.** Ready to run evaluations! ✅
