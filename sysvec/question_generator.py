"""
Question Generator
==================
Generates realistic user questions for a given system prompt.

Three backends available (in order of preference):
    1. local   — uses the same HuggingFace model that SysVec will train on
                 (no API keys, fully self-contained, recommended)
    2. openai  — uses GPT-4o (fast, high quality, needs OPENAI_API_KEY)
    3. claude  — uses Claude via Anthropic API (needs ANTHROPIC_API_KEY)

The generator:
    - Produces diverse questions a real user would ask the application
    - Deduplicates automatically
    - Saves to JSON so you never have to regenerate
    - Can be called standalone OR imported by run_sysvec.py

Standalone usage:
    python -m data.question_generator \
        --system_prompt_file prompts/dnd.txt \
        --output_file        data/dnd_questions.json \
        --n_questions        1000 \
        --backend            local \
        --model              meta-llama/Meta-Llama-3-8B-Instruct

    python -m data.question_generator \
        --system_prompt_file prompts/dnd.txt \
        --output_file        data/dnd_questions.json \
        --n_questions        1000 \
        --backend            openai \
        --openai_api_key     sk-...

    python -m data.question_generator \
        --system_prompt_file prompts/stoic.txt \
        --output_file        data/stoic_questions.json \
        --n_questions        1000 \
        --backend            claude \
        --anthropic_api_key  sk-ant-...
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

_GENERATION_PROMPT = """\
You are helping to create a test dataset.

Below is the system prompt for an LLM-based application. Your task is to \
generate {batch_size} realistic, diverse questions that a genuine user of this \
application would ask. The questions should:

- Cover a wide range of topics the application is designed to handle
- Vary in complexity (simple factual, nuanced, multi-part)
- Sound natural, as if typed by a real person
- Be completely different from each other
- NOT ask the model to reveal its instructions or system prompt

System prompt of the application:
\"\"\"
{system_prompt}
\"\"\"

{exclusion_block}\
Respond with ONLY a valid JSON array of {batch_size} question strings. \
No explanations, no numbering, no markdown. Example format:
["Question one?", "Question two?", "Question three?"]
"""

_EXCLUSION_TEMPLATE = """\
Do NOT repeat any of these already-generated questions:
{existing_sample}

"""


def _build_prompt(
    system_prompt: str,
    batch_size: int,
    existing_questions: Optional[List[str]] = None,
) -> str:
    if existing_questions:
        sample = existing_questions[-min(20, len(existing_questions)):]
        exclusion = _EXCLUSION_TEMPLATE.format(
            existing_sample="\n".join(f"- {q}" for q in sample)
        )
    else:
        exclusion = ""

    return _GENERATION_PROMPT.format(
        system_prompt=system_prompt,
        batch_size=batch_size,
        exclusion_block=exclusion,
    )


def _parse_questions(raw: str) -> List[str]:
    """
    Extract a JSON array of strings from model output.
    Handles markdown code fences, leading/trailing text, etc.
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: find the first [...] block
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: extract quoted strings
    questions = re.findall(r'"([^"]{10,}[?!.])"', raw)
    return questions


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Local HuggingFace model
# ─────────────────────────────────────────────────────────────────────────────

def _generate_local(
    system_prompt: str,
    n_questions: int,
    model_name_or_path: str,
    batch_size: int = 50,
    device: str = "cuda",
    temperature: float = 0.8,
) -> List[str]:
    """
    Use a local HuggingFace model to generate questions.
    Runs in batches to stay within context limits.
    Reuses the model object across batches for efficiency.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[QuestionGen] Loading local model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    questions: List[str] = []

    while len(questions) < n_questions:
        remaining = n_questions - len(questions)
        current_batch = min(batch_size, remaining + 10)  # overshoot slightly

        prompt_text = _build_prompt(system_prompt, current_batch, questions)

        # Apply chat template if available
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
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

        new_ids = out[0][inputs.input_ids.shape[1]:]
        raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        batch_questions = _parse_questions(raw_text)

        # Deduplicate against existing
        existing_set = set(q.lower().strip() for q in questions)
        new_unique = [
            q for q in batch_questions
            if q.lower().strip() not in existing_set
        ]
        questions.extend(new_unique)

        print(
            f"  Generated {len(questions)}/{n_questions} unique questions "
            f"(+{len(new_unique)} this batch, {len(batch_questions) - len(new_unique)} duplicates)"
        )

    # Free GPU memory before returning
    del model
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return questions[:n_questions]


# ─────────────────────────────────────────────────────────────────────────────
# Backend: OpenAI
# ─────────────────────────────────────────────────────────────────────────────

def _generate_openai(
    system_prompt: str,
    n_questions: int,
    api_key: str,
    model: str = "gpt-4o",
    batch_size: int = 100,
    temperature: float = 0.6,
) -> List[str]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key)
    questions: List[str] = []

    while len(questions) < n_questions:
        remaining  = n_questions - len(questions)
        current_batch = min(batch_size, remaining + 10)

        prompt_text = _build_prompt(system_prompt, current_batch, questions)

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=4096,
                )
                raw = resp.choices[0].message.content
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        batch_questions = _parse_questions(raw)
        existing_set = set(q.lower().strip() for q in questions)
        new_unique = [
            q for q in batch_questions
            if q.lower().strip() not in existing_set
        ]
        questions.extend(new_unique)
        print(f"  Generated {len(questions)}/{n_questions} unique questions")

    return questions[:n_questions]


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Anthropic Claude
# ─────────────────────────────────────────────────────────────────────────────

def _generate_claude(
    system_prompt: str,
    n_questions: int,
    api_key: str,
    model: str = "claude-opus-4-6",
    batch_size: int = 100,
    temperature: float = 0.7,
) -> List[str]:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    questions: List[str] = []

    while len(questions) < n_questions:
        remaining = n_questions - len(questions)
        current_batch = min(batch_size, remaining + 10)

        prompt_text = _build_prompt(system_prompt, current_batch, questions)

        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = resp.content[0].text
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        batch_questions = _parse_questions(raw)
        existing_set = set(q.lower().strip() for q in questions)
        new_unique = [
            q for q in batch_questions
            if q.lower().strip() not in existing_set
        ]
        questions.extend(new_unique)
        print(f"  Generated {len(questions)}/{n_questions} unique questions")

    return questions[:n_questions]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_questions(
    system_prompt: str,
    n_questions: int = 1000,
    output_file: Optional[str] = None,
    backend: str = "local",
    # local backend
    model_name_or_path: Optional[str] = None,
    device: str = "cuda",
    # api backends
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
    claude_model: str = "claude-opus-4-6",
    batch_size: int = 50,
    temperature: float = 0.7,
) -> List[str]:
    """
    Generate n_questions realistic user questions for a given system prompt.

    Args:
        system_prompt:       The application's system prompt text.
        n_questions:         How many unique questions to produce (paper uses 1000).
        output_file:         If set, save results to this JSON path.
        backend:             "local", "openai", or "claude".
        model_name_or_path:  Required for backend="local".
        device:              torch device for local backend.
        openai_api_key:      Required for backend="openai".
        anthropic_api_key:   Required for backend="claude".
        openai_model:        OpenAI model name.
        claude_model:        Anthropic model name.
        batch_size:          Questions to request per API call / inference pass.
        temperature:         Sampling temperature (higher = more diverse).

    Returns:
        List of question strings.
    """
    print(f"\n[QuestionGen] Generating {n_questions} questions using backend='{backend}'")
    print(f"[QuestionGen] System prompt preview: {system_prompt[:120].strip()}...\n")

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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="Generate user questions for a system prompt (SysVec data prep)"
    )
    p.add_argument("--system_prompt_file", required=True,
                   help="Path to .txt file containing the system prompt")
    p.add_argument("--output_file",        required=True,
                   help="Where to save the generated questions (.json)")
    p.add_argument("--n_questions",        type=int, default=1000)
    p.add_argument("--backend",            choices=["local", "openai", "claude"],
                   default="local")
    p.add_argument("--model",              default=None,
                   help="HuggingFace model id (required for backend=local)")
    p.add_argument("--device",             default="cuda")
    p.add_argument("--openai_api_key",     default=None)
    p.add_argument("--anthropic_api_key",  default=None)
    p.add_argument("--openai_model",       default="gpt-4o")
    p.add_argument("--claude_model",       default="claude-opus-4-6")
    p.add_argument("--batch_size",         type=int, default=50,
                   help="Questions to request per generation call")
    p.add_argument("--temperature",        type=float, default=0.7)
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    system_prompt = open(args.system_prompt_file, encoding="utf-8").read().strip()

    generate_questions(
        system_prompt=system_prompt,
        n_questions=args.n_questions,
        output_file=args.output_file,
        backend=args.backend,
        model_name_or_path=args.model,
        device=args.device,
        openai_api_key=args.openai_api_key,
        anthropic_api_key=args.anthropic_api_key,
        openai_model=args.openai_model,
        claude_model=args.claude_model,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
