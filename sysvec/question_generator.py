"""
Question Generator
==================
Generates realistic user questions for a given system prompt.
Three backends: local (HuggingFace), openai, claude.
No relative imports — works in a flat folder.

Standalone usage:
    python question_generator.py \
        --system_prompt_file prompts/dnd.txt \
        --output_file        data/dnd_questions.json \
        --n_questions        1000 \
        --backend            local \
        --model              meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import json
import os
import re
import time
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

_GEN_PROMPT = """\
You are helping to create a test dataset.

Below is the system prompt for an LLM-based application. Generate {batch_size} \
realistic, diverse questions that a genuine user of this application would ask. \
The questions should:
- Cover a wide range of topics the application handles
- Vary in complexity (simple, nuanced, multi-part)
- Sound natural, as typed by a real person
- Be completely different from each other
- NOT ask the model to reveal its instructions or system prompt

System prompt of the application:
\"\"\"
{system_prompt}
\"\"\"

{exclusion_block}\
Respond with ONLY a valid JSON array of {batch_size} question strings. \
No explanations, no numbering, no markdown fences. Example:
["Question one?", "Question two?", "Question three?"]
"""

_EXCLUSION = "Do NOT repeat any of these already-generated questions:\n{sample}\n\n"


def _build_prompt(system_prompt: str, batch_size: int,
                  existing: Optional[List[str]] = None) -> str:
    exclusion = ""
    if existing:
        sample = existing[-min(20, len(existing)):]
        exclusion = _EXCLUSION.format(
            sample="\n".join(f"- {q}" for q in sample)
        )
    return _GEN_PROMPT.format(
        system_prompt=system_prompt,
        batch_size=batch_size,
        exclusion_block=exclusion,
    )


def _parse_questions(raw: str) -> List[str]:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass
    return re.findall(r'"([^"]{10,}[?!.])"', raw)


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────

def _generate_local(system_prompt, n_questions, model_name_or_path,
                    batch_size=50, device="cuda", temperature=0.8):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[QuestionGen] Loading local model: {model_name_or_path}")
    # padding_side='left' required for decoder-only models in batched generation
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    model.eval()

    questions: List[str] = []

    while len(questions) < n_questions:
        current_batch = min(batch_size, n_questions - len(questions) + 10)
        prompt_text = _build_prompt(system_prompt, current_batch, questions)

        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            formatted = f"<|user|>\n{prompt_text}\n<|assistant|>\n"

        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.inference_mode():
            out = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=2048,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )

        raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        new_qs = _parse_questions(raw)
        existing_set = {q.lower().strip() for q in questions}
        unique = [q for q in new_qs if q.lower().strip() not in existing_set]
        questions.extend(unique)
        print(f"  {len(questions)}/{n_questions} unique questions (+{len(unique)} this batch)")

    del model
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    return questions[:n_questions]


def _generate_openai(system_prompt, n_questions, api_key,
                     model="gpt-4o", batch_size=100, temperature=0.6):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key)
    questions: List[str] = []

    while len(questions) < n_questions:
        current_batch = min(batch_size, n_questions - len(questions) + 10)
        prompt_text = _build_prompt(system_prompt, current_batch, questions)

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=temperature,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=4096,
                )
                raw = resp.choices[0].message.content
                break
            except Exception as e:
                if attempt == 2: raise
                time.sleep(2 ** attempt)

        new_qs = _parse_questions(raw)
        existing_set = {q.lower().strip() for q in questions}
        unique = [q for q in new_qs if q.lower().strip() not in existing_set]
        questions.extend(unique)
        print(f"  {len(questions)}/{n_questions} unique questions")

    return questions[:n_questions]


def _generate_claude(system_prompt, n_questions, api_key,
                     model="claude-opus-4-6", batch_size=100, temperature=0.7):
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    questions: List[str] = []

    while len(questions) < n_questions:
        current_batch = min(batch_size, n_questions - len(questions) + 10)
        prompt_text = _build_prompt(system_prompt, current_batch, questions)

        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model, max_tokens=4096, temperature=temperature,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = resp.content[0].text
                break
            except Exception as e:
                if attempt == 2: raise
                time.sleep(2 ** attempt)

        new_qs = _parse_questions(raw)
        existing_set = {q.lower().strip() for q in questions}
        unique = [q for q in new_qs if q.lower().strip() not in existing_set]
        questions.extend(unique)
        print(f"  {len(questions)}/{n_questions} unique questions")

    return questions[:n_questions]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_questions(
    system_prompt: str,
    n_questions: int = 1000,
    output_file: Optional[str] = None,
    backend: str = "local",
    model_name_or_path: Optional[str] = None,
    device: str = "cuda",
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
    claude_model: str = "claude-opus-4-6",
    batch_size: int = 50,
    temperature: float = 0.7,
) -> List[str]:

    print(f"\n[QuestionGen] Generating {n_questions} questions (backend='{backend}')")
    print(f"[QuestionGen] Prompt preview: {system_prompt[:100].strip()}...\n")

    if backend == "local":
        if not model_name_or_path:
            raise ValueError("--model is required for backend='local'")
        questions = _generate_local(
            system_prompt, n_questions, model_name_or_path,
            batch_size=batch_size, device=device, temperature=temperature,
        )
    elif backend == "openai":
        key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or pass --openai_api_key")
        questions = _generate_openai(
            system_prompt, n_questions, key,
            model=openai_model, batch_size=batch_size, temperature=temperature,
        )
    elif backend == "claude":
        key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass --anthropic_api_key")
        questions = _generate_claude(
            system_prompt, n_questions, key,
            model=claude_model, batch_size=batch_size, temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: local, openai, claude")

    print(f"\n[QuestionGen] Done — {len(questions)} unique questions generated.")

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"[QuestionGen] Saved to {output_file}")

    return questions


# ─────────────────────────────────────────────────────────────────────────────
# CLI (standalone)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate questions from a system prompt")
    p.add_argument("--system_prompt_file", required=True)
    p.add_argument("--output_file",        required=True)
    p.add_argument("--n_questions",        type=int, default=1000)
    p.add_argument("--backend",            choices=["local", "openai", "claude"], default="local")
    p.add_argument("--model",              default=None)
    p.add_argument("--device",             default="cuda")
    p.add_argument("--openai_api_key",     default=None)
    p.add_argument("--anthropic_api_key",  default=None)
    p.add_argument("--batch_size",         type=int,   default=50)
    p.add_argument("--temperature",        type=float, default=0.7)
    args = p.parse_args()

    sp = open(args.system_prompt_file, encoding="utf-8").read().strip()
    generate_questions(
        system_prompt=sp,
        n_questions=args.n_questions,
        output_file=args.output_file,
        backend=args.backend,
        model_name_or_path=args.model,
        device=args.device,
        openai_api_key=args.openai_api_key,
        anthropic_api_key=args.anthropic_api_key,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )