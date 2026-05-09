"""
SysVec End-to-End Runner (flat-folder version)
===============================================
All files live in the same directory — no package structure needed.

Usage:

  # Generate questions from system prompt only
  python run_sysvec.py generate \
      --system_prompt_file prompts/dnd.txt \
      --output_file        data/dnd_questions.json \
      --backend            local \
      --model              meta-llama/Meta-Llama-3-8B-Instruct

  python run_sysvec.py generate \
      --system_prompt_file prompts/dnd.txt \
      --output_file        data/dnd_questions.json \
      --backend            openai --openai_api_key sk-...

  python run_sysvec.py generate \
      --system_prompt_file prompts/dnd.txt \
      --output_file        data/dnd_questions.json \
      --backend            claude --anthropic_api_key sk-ant-...

  # Train (auto-generates questions if --questions_file is absent)
  python run_sysvec.py train \
      --model              meta-llama/Meta-Llama-3-8B-Instruct \
      --system_prompt_file prompts/dnd.txt \
      --output_dir         checkpoints/dnd

  # Train with existing questions file
  python run_sysvec.py train \
      --model              meta-llama/Meta-Llama-3-8B-Instruct \
      --system_prompt_file prompts/dnd.txt \
      --questions_file     data/dnd_questions.json \
      --output_dir         checkpoints/dnd

  # Attack
  python run_sysvec.py attack \
      --model              meta-llama/Meta-Llama-3-8B-Instruct \
      --sysvec_path        checkpoints/dnd/sysvec_final.pt \
      --system_prompt_file prompts/dnd.txt \
      --questions_file     data/dnd_questions.json \
      --attack             remember --defense sysvec

  # Evaluate utility (RUS)
  python run_sysvec.py eval_rus \
      --model              meta-llama/Meta-Llama-3-8B-Instruct \
      --sysvec_path        checkpoints/dnd/sysvec_final.pt \
      --system_prompt_file prompts/dnd.txt \
      --questions_file     data/dnd_questions.json \
      --openai_api_key     sk-...
"""

import argparse
import json
import os
import sys
import statistics

import torch

# All imports are flat — just filenames, no packages
from model             import SysVec
from trainer           import SysVecTrainer, PreferenceDataset, build_preference_samples
from attacks           import apply_attack, ATTACK_REGISTRY
from defenses          import apply_defense, DEFENSE_REGISTRY
from metrics           import compute_pls, compute_ss, compute_rus
from question_generator import generate_questions


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
            return [str(q) for q in data] if isinstance(data, list) else list(data.values())
        return [line.strip() for line in f if line.strip()]


def build_model(args):
    return SysVec(
        model_name_or_path=args.model,
        injection_layer=args.injection_layer,
        alpha=args.alpha,
        device=args.device,
    )


def generate_response(model, system_prompt, user_message,
                      max_new_tokens=512, defense="none"):
    tokenizer = model.tokenizer
    device    = model.device

    if defense == "sysvec":
        messages = [{"role": "user", "content": user_message}]
        inject = True
    else:
        mod_sp, mod_up = apply_defense(system_prompt, user_message, defense)
        messages = [
            {"role": "system", "content": mod_sp},
            {"role": "user",   "content": mod_up},
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
            text = f"<|system|>\n{mod_sp}\n<|user|>\n{mod_up}\n<|assistant|>\n"

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


def _resolve_questions(args):
    """Return questions list, auto-generating if no --questions_file given."""
    if args.questions_file:
        return load_questions(args.questions_file)

    sp_stem = os.path.splitext(os.path.basename(args.system_prompt_file))[0]
    auto_q_file = os.path.join(args.output_dir, f"{sp_stem}_questions.json")

    if os.path.exists(auto_q_file):
        print(f"[QuestionGen] Reusing existing file: {auto_q_file}")
        return load_questions(auto_q_file)

    system_prompt = load_text(args.system_prompt_file)
    n_needed = args.train_size + args.test_size

    print(f"[QuestionGen] No --questions_file given — generating {n_needed} questions.")
    print(f"[QuestionGen] Backend: {args.question_backend}")
    print(f"[QuestionGen] Saving to: {auto_q_file}")
    os.makedirs(args.output_dir, exist_ok=True)

    questions = generate_questions(
        system_prompt=system_prompt,
        n_questions=n_needed,
        output_file=auto_q_file,
        backend=args.question_backend,
        model_name_or_path=args.model,
        device=args.device,
        openai_api_key=getattr(args, "openai_api_key", None),
        anthropic_api_key=getattr(args, "anthropic_api_key", None),
        batch_size=args.question_batch_size,
        temperature=args.question_temperature,
    )

    if len(questions) < n_needed:
        print(f"[QuestionGen] WARNING: got {len(questions)}, need {n_needed}.")

    return questions


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_generate(args):
    print("=" * 60)
    print("SysVec -- QUESTION GENERATION")
    print("=" * 60)
    system_prompt = load_text(args.system_prompt_file)
    generate_questions(
        system_prompt=system_prompt,
        n_questions=args.n_questions,
        output_file=args.output_file,
        backend=args.backend,
        model_name_or_path=getattr(args, "model", None),
        device=getattr(args, "device", "cuda"),
        openai_api_key=getattr(args, "openai_api_key", None),
        anthropic_api_key=getattr(args, "anthropic_api_key", None),
        batch_size=args.question_batch_size,
        temperature=args.question_temperature,
    )


def cmd_train(args):
    print("=" * 60)
    print("SysVec -- TRAINING MODE")
    print("=" * 60)

    system_prompt = load_text(args.system_prompt_file)
    questions     = _resolve_questions(args)
    train_qs      = questions[: args.train_size]
    print(f"Using {len(train_qs)} questions for training.")

    model = build_model(args)

    # DPO dataset save path — lives next to the sysvec checkpoints
    sp_stem   = os.path.splitext(os.path.basename(args.system_prompt_file))[0]
    dpo_path  = os.path.join(args.output_dir, f"{sp_stem}_dpo_pairs.pt")

    print(f"\n[Step 1] Synthesising preference data (yw, yl) pairs...")
    print(f"  Save path : {dpo_path}")
    print(f"  Resume    : {not args.force_regen}  (use --force_regen to rebuild)")
    samples = build_preference_samples(
        sysvec_model=model,
        system_prompt=system_prompt,
        user_questions=train_qs,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
        save_path=dpo_path,
        resume=not args.force_regen,
    )

    print("\n[Step 2] Optimising SysVec vector via DPO objective...")
    trainer = SysVecTrainer(
        sysvec_model=model,
        dataset=PreferenceDataset(samples),
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
    print(f"\nDone. Vector saved to: {final_path}")


def cmd_attack(args):
    print("=" * 60)
    print(f"SysVec -- ATTACK  [{args.attack}]  defense=[{args.defense}]")
    print("=" * 60)

    system_prompt = load_text(args.system_prompt_file)
    questions     = _resolve_questions(args)
    test_qs       = questions[args.train_size: args.train_size + args.test_size]

    model = build_model(args)
    if args.sysvec_path and args.defense == "sysvec":
        model.load_sysvec(args.sysvec_path)

    results = []
    for i, q in enumerate(test_qs):
        adv_q = apply_attack(q, args.attack,
                              known_prefix=args.known_prefix,
                              end_phrase=args.end_phrase)
        response = generate_response(
            model, system_prompt, adv_q,
            defense=args.defense, max_new_tokens=args.max_new_tokens,
        )
        print(f"\n[{i+1}/{len(test_qs)}] Q: {q[:80]}...")
        print(f"  Response: {response[:200]}")

        entry = {"question": q, "attack": args.attack, "response": response}
        if args.openai_api_key:
            pls = compute_pls(system_prompt, response, api_key=args.openai_api_key)
            ss  = compute_ss(system_prompt, response)
            entry.update(pls=pls, ss=ss)
            print(f"  PLS={pls:.2f}  SS={ss:.3f}")
        results.append(entry)

    out_path = os.path.join(
        args.output_dir or ".", f"attack_{args.attack}_{args.defense}.json"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if args.openai_api_key:
        pls_vals = [r["pls"] for r in results if "pls" in r]
        ss_vals  = [r["ss"]  for r in results if "ss"  in r]
        print(f"\n-- Summary --")
        print(f"  PLS: {statistics.mean(pls_vals):.2f} +/- {statistics.stdev(pls_vals):.2f}")
        print(f"  SS:  {statistics.mean(ss_vals):.3f} +/- {statistics.stdev(ss_vals):.3f}")


def cmd_eval_rus(args):
    print("=" * 60)
    print("SysVec -- UTILITY EVALUATION (RUS)")
    print("=" * 60)

    if not args.openai_api_key:
        print("ERROR: --openai_api_key required for RUS.")
        sys.exit(1)

    system_prompt = load_text(args.system_prompt_file)
    questions     = _resolve_questions(args)
    test_qs       = questions[args.train_size: args.train_size + args.test_size]

    model = build_model(args)
    if args.sysvec_path:
        model.load_sysvec(args.sysvec_path)

    rus_scores = []
    for i, q in enumerate(test_qs):
        response = generate_response(
            model, system_prompt, q,
            defense=args.defense, max_new_tokens=args.max_new_tokens,
        )
        rus = compute_rus(system_prompt, q, response, api_key=args.openai_api_key)
        rus_scores.append(rus)
        print(f"[{i+1}/{len(test_qs)}] RUS={rus:.1f}  Q: {q[:60]}...")

    print(f"\n-- RUS Summary --")
    print(f"  RUS: {statistics.mean(rus_scores):.2f} +/- "
          f"{statistics.stdev(rus_scores) if len(rus_scores)>1 else 0:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="SysVec -- Prompt Leakage Defense")
    subs = p.add_subparsers(dest="command", required=True)

    # Shared question-gen args
    qgen = argparse.ArgumentParser(add_help=False)
    qgen.add_argument("--question_backend",
        choices=["local", "openai", "claude"], default="local")
    qgen.add_argument("--question_batch_size",  type=int,   default=50)
    qgen.add_argument("--question_temperature", type=float, default=0.7)
    qgen.add_argument("--anthropic_api_key",    default=None)

    # Shared model/data args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model",              default=None)
    shared.add_argument("--system_prompt_file", required=True)
    shared.add_argument("--questions_file",     default=None,
        help="Path to questions JSON. If omitted, questions are auto-generated.")
    shared.add_argument("--injection_layer",    type=int,   default=15)
    shared.add_argument("--alpha",              type=float, default=1.0)
    shared.add_argument("--device",             default="cuda")
    shared.add_argument("--output_dir",         default="./output")
    shared.add_argument("--train_size",         type=int,   default=800)
    shared.add_argument("--test_size",          type=int,   default=200)
    shared.add_argument("--max_new_tokens",     type=int,   default=256)
    shared.add_argument("--openai_api_key",     default=None)
    shared.add_argument("--sysvec_path",        default=None)

    # generate
    gen_p = subs.add_parser("generate", parents=[qgen, shared])
    gen_p.add_argument("--n_questions", type=int, default=1000)
    gen_p.add_argument("--output_file", required=True)
    gen_p.add_argument("--backend",
        choices=["local", "openai", "claude"], default="local")

    # train
    train_p = subs.add_parser("train", parents=[shared, qgen])
    train_p.add_argument("--epochs",           type=int,   default=25)
    train_p.add_argument("--lr",               type=float, default=5e-4)
    train_p.add_argument("--weight_decay",     type=float, default=0.05)
    train_p.add_argument("--beta",             type=float, default=0.1)
    train_p.add_argument("--batch_size",       type=int,   default=2)
    train_p.add_argument("--grad_accum_steps", type=int,   default=4)
    train_p.add_argument("--warmup_steps",     type=int,   default=100)
    train_p.add_argument("--gen_batch_size",   type=int,   default=4)
    train_p.add_argument("--force_regen",      action="store_true", default=False,
        help="Force regeneration of DPO pairs even if a saved dataset exists")

    # attack
    attack_p = subs.add_parser("attack", parents=[shared])
    attack_p.add_argument("--attack",
        choices=list(ATTACK_REGISTRY.keys()), default="naive")
    attack_p.add_argument("--defense",
        choices=list(DEFENSE_REGISTRY.keys()) + ["sysvec"], default="sysvec")
    attack_p.add_argument("--known_prefix", default="You are GPT")
    attack_p.add_argument("--end_phrase",   default="initialization done")

    # eval_rus
    rus_p = subs.add_parser("eval_rus", parents=[shared])
    rus_p.add_argument("--defense",
        choices=list(DEFENSE_REGISTRY.keys()) + ["sysvec"], default="sysvec")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if   args.command == "generate": cmd_generate(args)
    elif args.command == "train":    cmd_train(args)
    elif args.command == "attack":   cmd_attack(args)
    elif args.command == "eval_rus": cmd_eval_rus(args)