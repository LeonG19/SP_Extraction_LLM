# LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage

This repository studies **system prompt extraction** — adversarially forcing an LLM to reveal the confidential instructions embedded in its system prompt. It implements four attacks, three defenses, and a multi-mode evaluation pipeline.

## Attacks

| Method | Type | Description |
|--------|------|-------------|
| **PLeak** | White-box | Gradient-based HotFlip token optimization against a locally loaded model |
| **PromptFuzz** | Black-box | MCTS-guided genetic search with LLM-based prompt mutation |
| **RE (ReAct-Leak)** | Black-box | Reasoning-agent iterative refinement using a Think → Act → Observe loop |
| **LeakAgent** | Black-box RL | PPO fine-tuning of an LLM to generate adversarial prompts end-to-end |

A **multi-turn sycophancy** extension wraps any of the above: Turn 1 sends the adversarial prompt; Turn 2 appends a scripted follow-up asking the model to reproduce its instructions word for word. In experiments, Turn 2 ROUGE-L converges near 97–98% for all methods, compared with 6–88% for Turn 1 alone.

## Defenses

| Defense | Description |
|---------|-------------|
| **PromptGuard** | 86M classifier that blocks detected jailbreak / injection inputs |
| **SecAlign** | DPO-fine-tuned model variant trained to ignore prompt injection |
| **SFT** | Supervised fine-tuning on refusal examples via PEFT + LoRA |

## Dataset

All attacks train on `dataset/train_data_pleak.csv` (**87 system prompts**) and are evaluated on `dataset/test_data_pleak.csv` (**58 system prompts**), both sourced from the Awesome ChatGPT Prompts collection. The evaluation selects the top 5 generated attack prompts by reward, applies each to all 58 test prompts with 10 sampled responses, and reports the best-of-10 ROUGE-L and LCS similarity per example averaged across the test set.

---

## 1. Installation

```bash
pip install -r requirements.txt
```

---

## 2. Start the Victim Model Server

All attacks and evaluations query the victim model through a vLLM OpenAI-compatible server. Start it before running any attack or evaluation step.

```bash
# GPU 0 — victim model (Llama-3.1-8B)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --disable-log-requests \
    --enable-prefix-caching \
    --port 54321 \
    --api-key empty
```

For attacks that also use a helper / reasoning model (PromptFuzz, RE, LeakAgent), start a second server on a different port:

```bash
# GPU 1 — helper model (policy / reasoning model)
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --disable-log-requests \
    --port 54322 \
    --api-key empty
```

---

## 3. Generate Attack Prompts

Each attack reads `train_data_pleak.csv` and writes its best prompts to `{output_dir}/good_prompts.csv`.

### PLeak (white-box, gradient-based)

PLeak requires direct access to model weights; no vLLM server is needed for training.

```bash
python train_pleak.py \
    --model llama3 \
    --dataset_path dataset/train_data_pleak.csv \
    --optim_token_length 20 \
    --num_triggers 5 \
    --num_iterations 100 \
    --output_dir pleak_results
```

### PromptFuzz (black-box, MCTS + LLM mutation)

```bash
python pipeline.py \
    --method fuzz \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --helper_model meta-llama/Meta-Llama-3-8B-Instruct \
    --target_server_url http://localhost:54321/v1 \
    --target_api_key empty \
    --helper_server_url http://localhost:54322/v1 \
    --helper_api_key empty \
    --prompts_data_path dataset/train_data_pleak.csv
```

Generated prompts are saved to `llama3_fuzz_finetune/good_prompts.csv`.

### RE — ReAct-Leak (black-box, reasoning agent)

```bash
python pipeline.py \
    --method re \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --helper_model meta-llama/Meta-Llama-3-8B-Instruct \
    --target_server_url http://localhost:54321/v1 \
    --target_api_key empty \
    --helper_server_url http://localhost:54322/v1 \
    --helper_api_key empty \
    --reason_model gpt-4o-mini \
    --prompts_data_path dataset/train_data_pleak.csv
```

Generated prompts are saved to `llama3_re_finetune/good_prompts.csv`.

### LeakAgent (black-box RL fine-tuning)

LeakAgent PPO-fine-tunes an LLM policy against the victim model. Both the policy and victim model require GPU servers.

```bash
accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes 2 \
    --main_process_port 24599 \
    attacks/token_level/blackbox/FineTuneLLM.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer_name meta-llama/Meta-Llama-3-8B-Instruct \
    --load_in_4bit \
    --victim_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir llama3_rl_finetune/ \
    --batch_size 32 \
    --requests_per_minute 100 \
    --target_dataset dataset/train_data_pleak.csv \
    --epochs 60 \
    --save_freq 4 \
    --use_bonus_reward True \
    --server_url http://localhost:54322/v1 \
    --api_key empty
```

- Checkpoints are saved to `llama3_rl_finetune/epoch_{N}/`
- Best prompts are written to `llama3_rl_finetune/good_prompts.csv`
- Optional W&B logging: add `--log_with wandb --wandb_exp_name <name> --wandb_entity <entity>`

---

## 4. Evaluate

All evaluation commands assume the victim model vLLM server is running on port 54321. The evaluator selects the top 5 attack prompts by reward from `good_prompts.csv`, applies them to `test_data_pleak.csv`, and reports ROUGE-L and LCS similarity scores.

### Single-turn standard evaluation

```bash
python evaluation_method.py \
    --method leakagent \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --training_prompts llama3_rl_finetune/good_prompts.csv \
    --test_data dataset/test_data_pleak.csv \
    --n_samples 10 \
    --top_k 5 \
    --port 54321
```

Swap `--method` for `fuzz`, `re`, or `pleak` and point `--training_prompts` to the corresponding `good_prompts.csv`.

### Repeated prompt evaluation

The same attack prompt is sent in two independent conversations and scored separately.

```bash
python evaluation_method.py \
    --method fuzz \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --training_prompts llama3_fuzz_finetune/good_prompts.csv \
    --mode repeated_prompt \
    --port 54321
```

### Multi-turn sycophancy evaluation

Turn 1 sends the adversarial prompt; Turn 2 appends the model's response and sends a scripted follow-up asking it to reproduce its instructions verbatim.

```bash
python extensions/run_extensions.py \
    --mode multiturn \
    --attack fuzz \
    --target llama3.1-8b \
    --server_url http://127.0.0.1:54321/v1 \
    --api_key empty \
    --dataset_path dataset/test_data_pleak.csv \
    --n_attack_prompts 5 \
    --output_dir results/multiturn
```

### Evaluate against defenses

Run all attack methods against all defenses and models in one command:

```bash
python evaluate_methods_vs_defenses.py \
    --attack-methods fuzz re pleak leakagent \
    --defense-methods promptguard secalign sft \
    --models llama3.1-8b mistral-7b \
    --mode standard
```

To run only a specific combination:

```bash
# LeakAgent vs SecAlign on Llama-3.1-8B
python evaluate_methods_vs_defenses.py \
    --attack-methods leakagent \
    --defense-methods secalign \
    --models llama3.1-8b
```

Results are written to `evaluation_results/standard/` (or `repeated_prompt/` / `multi_turn/` depending on `--mode`).

---

## 5. Results Summary

Standard single-turn evaluation on `test_data_pleak.csv` (avg best-of-10 ROUGE-L):

| Attack | Llama-3.1-8B | Mistral-7B |
|--------|-------------|-----------|
| PLeak | 0.304 | 0.350 |
| PromptFuzz | 0.782 | 0.682 |
| LeakAgent | 0.542 | — |
| RE | **0.886** | 0.539 |

Multi-turn sycophancy (Turn 2 ROUGE-L after follow-up):

| Attack | Turn 1 ROUGE-L | Turn 2 ROUGE-L |
|--------|--------------|--------------|
| PLeak | 0.060 | **0.968** |
| PromptFuzz | 0.680 | **0.985** |
| LeakAgent | 0.337 | **0.977** |
| RE | 0.773 | **0.973** |

---

## 6. Reference

If you find this repository useful, please cite:

```bibtex
@inproceedings{nie2025leakagent,
  title={LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage},
  author={Yuzhou Nie and Zhun Wang and Ye Yu and Xian Wu and Xuandong Zhao and Wenbo Guo and Dawn Song},
  year={2025},
  booktitle={COLM},
}
```
