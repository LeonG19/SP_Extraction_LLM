"""
SysVec End-to-End Runner
========================
Ties together data synthesis, training, and evaluation into one CLI script.

Usage examples:

  # 1. Generate preference data, train a SysVec, and save the vector
  python run_sysvec.py train \
      --model  meta-llama/Meta-Llama-3-8B-Instruct \
      --system_prompt_file prompts/dnd.txt \
      --questions_file     data/dnd_questions.json \
      --output_dir         checkpoints/dnd \
      --epochs 25 \
      --injection_layer 15 \
      --alpha 1.0

  # 2. Run an attack against the defended model
  python run_sysvec.py attack \
      --model         meta-llama/Meta-Llama-3-8B-Instruct \
      --sysvec_path   checkpoints/dnd/sysvec_final.pt \
      --system_prompt_file prompts/dnd.txt \
      --questions_file     data/dnd_test_questions.json \
      --attack        remember \
      --known_prefix  "You are a" \
      --defense       sysvec

  # 3. Evaluate utility (RUS) of the defended model
  python run_sysvec.py eval_rus \
      --model         meta-llama/Meta-Llama-3-8B-Instruct \
      --sysvec_path   checkpoints/dnd/sysvec_final.pt \
      --system_prompt_file prompts/dnd.txt \
      --questions_file     data/dnd_test_questions.json \
      --openai_api_key sk-...
"""

import argparse
import json
import os
import sys

import torch

# ── Make sure sibling packages are importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from model import SysVec
from trainer import SysVecTrainer, PreferenceDataset, build_preference_samples
from attacks import apply_attack, ATTACK_REGISTRY
from defenses import apply_defense, DEFENSE_REGISTRY
from metrics import compute_pls, compute_ss, compute_rus


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_questions(path: str) -> list:
    """Load questions from a JSON file (list of strings) or plain text (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
            if isinstance(data, list):
                return [str(q) for q in data]
            return list(data.values())
        else:
            return [line.strip() for line in f if line.strip()]


def build_model(args) -> SysVec:
    return SysVec(
        model_name_or_path=args.model,
        injection_layer=args.injection_layer,
        alpha=args.alpha,
        device=args.device,
    )


def generate_response(
    model: SysVec,
    system_prompt: str,
    user_message: str,
    inject: bool = True,
    max_new_tokens: int = 512,
    defense: str = "none",
) -> str:
    """
    Generate a single response.

    If defense == 'sysvec': inject the learned vector (no textual system prompt).
    Otherwise: apply the named textual defense and do NOT inject.
    """
    tokenizer = model.tokenizer
    device = model.device

    if defense == "sysvec":
        # No textual system prompt; vsys carries the instructions
        messages = [{"role": "user", "content": user_message}]
        inject = True
    else:
        mod_sp, mod_up = apply_defense(system_prompt, user_message, defense)
        messages = [
            {"role": "system", "content": mod_sp},
            {"role": "user", "content": mod_up},
        ]
        inject = False

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if defense == "sysvec":
            text = f"<|user|>\n{user_message}\n<|assistant|>\n"
        else:
            text = (
                f"<|system|>\n{mod_sp}\n"
                f"<|user|>\n{mod_up}\n<|assistant|>\n"
            )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            inject=inject,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_ids = out_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_train(args):
    print("=" * 60)
    print("SysVec — TRAINING MODE")
    print("=" * 60)

    system_prompt = load_text(args.system_prompt_file)
    questions     = load_questions(args.questions_file)

    train_qs = questions[: args.train_size]
    print(f"Using {len(train_qs)} questions for training.")

    model = build_model(args)

    print("\n[Step 1] Generating preference data (yw, yl) pairs...")
    samples = build_preference_samples(
        sysvec_model=model,
        system_prompt=system_prompt,
        user_questions=train_qs,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
    )

    dataset = PreferenceDataset(samples)

    print("\n[Step 2] Optimising SysVec vector via DPO objective...")
    trainer = SysVecTrainer(
        sysvec_model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
    )
    final_path = trainer.train()
    print(f"\nDone. System vector saved to: {final_path}")


def cmd_attack(args):
    print("=" * 60)
    print(f"SysVec — ATTACK MODE  [{args.attack}]  defense=[{args.defense}]")
    print("=" * 60)

    system_prompt = load_text(args.system_prompt_file)
    questions     = load_questions(args.questions_file)
    test_qs       = questions[args.train_size: args.train_size + args.test_size]

    model = build_model(args)
    if args.sysvec_path and args.defense == "sysvec":
        model.load_sysvec(args.sysvec_path)

    results = []
    for i, q in enumerate(test_qs):
        adversarial_q = apply_attack(
            q,
            args.attack,
            known_prefix=args.known_prefix,
            end_phrase=args.end_phrase,
        )

        response = generate_response(
            model,
            system_prompt,
            adversarial_q,
            defense=args.defense,
            max_new_tokens=args.max_new_tokens,
        )

        print(f"\n[{i+1}/{len(test_qs)}] Q: {q[:80]}...")
        print(f"  Attack suffix applied: {args.attack}")
        print(f"  Response (first 200 chars): {response[:200]}")

        entry = {"question": q, "attack": args.attack, "response": response}

        # Optional PLS metric
        if args.openai_api_key:
            pls = compute_pls(system_prompt, response, api_key=args.openai_api_key)
            ss  = compute_ss(system_prompt, response)
            entry["pls"] = pls
            entry["ss"]  = ss
            print(f"  PLS={pls:.2f}  SS={ss:.3f}")

        results.append(entry)

    # Save results
    out_path = os.path.join(
        args.output_dir or ".",
        f"attack_{args.attack}_{args.defense}.json",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if args.openai_api_key:
        pls_vals = [r["pls"] for r in results if "pls" in r]
        ss_vals  = [r["ss"]  for r in results if "ss"  in r]
        import statistics
        print("\n── Summary ──────────────────────────────────────")
        print(f"  PLS: {statistics.mean(pls_vals):.2f} ± {statistics.stdev(pls_vals):.2f}")
        print(f"  SS:  {statistics.mean(ss_vals):.3f} ± {statistics.stdev(ss_vals):.3f}")


def cmd_eval_rus(args):
    print("=" * 60)
    print("SysVec — UTILITY EVALUATION (RUS)")
    print("=" * 60)

    if not args.openai_api_key:
        print("ERROR: --openai_api_key is required for RUS evaluation.")
        sys.exit(1)

    system_prompt = load_text(args.system_prompt_file)
    questions     = load_questions(args.questions_file)
    test_qs       = questions[args.train_size: args.train_size + args.test_size]

    model = build_model(args)
    if args.sysvec_path:
        model.load_sysvec(args.sysvec_path)

    rus_scores = []
    for i, q in enumerate(test_qs):
        response = generate_response(
            model,
            system_prompt,
            q,
            defense=args.defense,
            max_new_tokens=args.max_new_tokens,
        )
        rus = compute_rus(
            system_prompt, q, response,
            api_key=args.openai_api_key,
        )
        rus_scores.append(rus)
        print(f"[{i+1}/{len(test_qs)}] RUS={rus:.1f}  Q: {q[:60]}...")

    import statistics
    mean_rus = statistics.mean(rus_scores)
    std_rus  = statistics.stdev(rus_scores) if len(rus_scores) > 1 else 0.0
    print(f"\n── RUS Summary ──────────────────────────────────")
    print(f"  RUS: {mean_rus:.2f} ± {std_rus:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="SysVec — Prompt Leakage Defense")
    subs = p.add_subparsers(dest="command", required=True)

    # ── Shared args ──────────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model",               required=True, help="HuggingFace model id or local path")
    shared.add_argument("--system_prompt_file",  required=True, help="Path to .txt file with system prompt")
    shared.add_argument("--questions_file",      required=True, help="Path to JSON or .txt file with questions")
    shared.add_argument("--injection_layer",     type=int,   default=15)
    shared.add_argument("--alpha",               type=float, default=1.0)
    shared.add_argument("--device",              default="cuda")
    shared.add_argument("--output_dir",          default="./output")
    shared.add_argument("--train_size",          type=int,   default=800)
    shared.add_argument("--test_size",           type=int,   default=200)
    shared.add_argument("--max_new_tokens",      type=int,   default=256)
    shared.add_argument("--openai_api_key",      default=None)
    shared.add_argument("--sysvec_path",         default=None, help="Path to a saved .pt SysVec checkpoint")

    # ── Train ────────────────────────────────────────────────────────────────
    train_p = subs.add_parser("train", parents=[shared])
    train_p.add_argument("--epochs",            type=int,   default=25)
    train_p.add_argument("--lr",                type=float, default=5e-4)
    train_p.add_argument("--weight_decay",      type=float, default=0.05)
    train_p.add_argument("--beta",              type=float, default=0.1)
    train_p.add_argument("--batch_size",        type=int,   default=2)
    train_p.add_argument("--grad_accum_steps",  type=int,   default=4)
    train_p.add_argument("--warmup_steps",      type=int,   default=100)
    train_p.add_argument("--gen_batch_size",    type=int,   default=4)

    # ── Attack ───────────────────────────────────────────────────────────────
    attack_p = subs.add_parser("attack", parents=[shared])
    attack_p.add_argument("--attack",
        choices=list(ATTACK_REGISTRY.keys()), default="naive")
    attack_p.add_argument("--defense",
        choices=list(DEFENSE_REGISTRY.keys()) + ["sysvec"], default="sysvec")
    attack_p.add_argument("--known_prefix",  default="You are GPT")
    attack_p.add_argument("--end_phrase",    default="initialization done")

    # ── Eval RUS ─────────────────────────────────────────────────────────────
    rus_p = subs.add_parser("eval_rus", parents=[shared])
    rus_p.add_argument("--defense",
        choices=list(DEFENSE_REGISTRY.keys()) + ["sysvec"], default="sysvec")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "attack":
        cmd_attack(args)
    elif args.command == "eval_rus":
        cmd_eval_rus(args)
