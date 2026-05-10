"""Shared utilities for experimental extensions: metrics, async inference, logging, checkpointing."""

import asyncio
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from aiolimiter import AsyncLimiter
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# Global state for embedding model
_EMBED_MODEL = None


def setup_logging(timestamp: str, log_dir: str = "logs") -> logging.Logger:
    """Setup logging to file and stdout."""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load system prompt dataset. Returns DataFrame with columns: index, text."""
    df = pd.read_csv(dataset_path)
    return df[["index", "text"]]


def load_attack_prompts(attack: str, n: int = 5, base_dir: str = ".") -> List[str]:
    """Load attack prompts from good_prompts.csv for the given attack type.

    Returns top-n by reward, as a list of strings.
    """
    attack_to_dir = {
        "fuzz": "llama3_fuzz_finetune",
        "re": "llama3_re_finetune",
        "leakagent": "llama3_rl_finetune",
        "pleak": "pleak_results",
    }

    if attack not in attack_to_dir:
        raise ValueError(f"Unknown attack: {attack}. Must be one of {list(attack_to_dir.keys())}")

    csv_path = Path(base_dir) / attack_to_dir[attack] / "good_prompts.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Attack prompts not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("reward", ascending=False)
    prompts = df["text"].head(n).tolist()
    return prompts


def make_client(server_url: str, api_key: str) -> "openai.AsyncOpenAI":
    """Create an async OpenAI client (vLLM or OpenAI-compatible)."""
    import openai

    return openai.AsyncOpenAI(base_url=server_url, api_key=api_key)


async def batch_sample(
    client,
    model: str,
    messages: List[Dict],
    n: int,
    temperature: float = 0.8,
    max_tokens: int = 512,
    limiter: Optional[AsyncLimiter] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Sample n responses from the model for the given messages.

    Returns list of n response strings.
    Retries up to 5 times with exponential backoff on failure.
    """
    for attempt in range(5):
        try:
            if limiter:
                async with limiter:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # Extract response texts
            responses = [choice.message.content for choice in response.choices]
            return responses

        except Exception as e:
            if logger:
                logger.warning(f"Attempt {attempt + 1}/5 failed: {e}")
            await asyncio.sleep(10 * (attempt + 1))
            continue

    raise RuntimeError(f"Failed to sample after 5 attempts")


def get_embed_model(device: str = "cuda") -> SentenceTransformer:
    """Get or create the embedding model (lazy singleton)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=device,
        )
    return _EMBED_MODEL


def compute_metrics(pred: str, ref: str, embed_model: SentenceTransformer) -> Dict[str, float]:
    """
    Compute all metrics: ROUGE-L, BLEU, exact match, cosim.

    Returns dict with keys: rouge_l, bleu, exact_match, cosim
    """
    metrics = {}

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_result = scorer.score(ref, pred)
    metrics["rouge_l"] = rouge_result["rougeL"].fmeasure

    # BLEU (sentence-level with smoothing)
    ref_tokens = ref.split()
    pred_tokens = pred.split()
    try:
        bleu_score = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
        metrics["bleu"] = bleu_score
    except Exception:
        metrics["bleu"] = 0.0

    # Exact match (substring)
    metrics["exact_match"] = float(ref in pred)

    # Cosine similarity via embeddings
    try:
        embeddings = embed_model.encode([pred, ref], normalize_embeddings=True)
        pred_emb = embeddings[0]
        ref_emb = embeddings[1]
        cosim = np.dot(pred_emb, ref_emb)
        metrics["cosim"] = float(cosim)
    except Exception:
        metrics["cosim"] = 0.0

    return metrics


def completed_triples(
    output_dir: str,
    prefix: str,
) -> Set[Tuple[str, str, int]]:
    """
    Scan existing JSONL files to find completed (system_prompt_id, attack, sample_index) triples.

    Used for resuming. Returns a set of tuples.
    """
    completed = set()
    output_path = Path(output_dir)
    if not output_path.exists():
        return completed

    for jsonl_file in output_path.glob(f"{prefix}*.jsonl"):
        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "system_prompt_id" in record and "attack" in record:
                        sample_index = record.get("sample_index", 0)
                        triple = (record["system_prompt_id"], record["attack"], sample_index)
                        completed.add(triple)
        except Exception:
            continue

    return completed


def append_jsonl(path: str, record: Dict) -> None:
    """Append a single record (dict) to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_progress(
    logger: logging.Logger,
    done: int,
    total: int,
    start_time: float,
    running_rouge: float,
) -> None:
    """Log progress with ETA and running mean ROUGE-L."""
    import time

    if done == 0:
        return

    elapsed = time.time() - start_time
    rate = done / elapsed
    remaining = total - done
    eta = remaining / rate if rate > 0 else 0

    logger.info(
        f"Progress: {done}/{total} ({100*done//total}%) "
        f"ETA={eta:.0f}s mean_rougeL={running_rouge:.4f}"
    )


def create_scaling_curve_csv(
    output_path: str,
    records: List[Dict],
) -> None:
    """
    Aggregate raw records into a scaling curve CSV.

    Group by (n, selector), compute mean/std ROUGE-L and mean BLEU/exact/cosim.
    Write to CSV with columns: n, selector, mean_rouge_l, std_rouge_l, mean_bleu, mean_exact_match, mean_cosim
    """
    if not records:
        return

    df = pd.DataFrame(records)
    grouped = df.groupby(["n", "selector"]).agg(
        {
            "rouge_l": ["mean", "std"],
            "bleu": "mean",
            "exact_match": "mean",
            "cosim": "mean",
        }
    )

    # Flatten multi-level columns
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    # Rename for clarity
    grouped = grouped.rename(
        columns={
            "rouge_l_mean": "mean_rouge_l",
            "rouge_l_std": "std_rouge_l",
            "bleu_mean": "mean_bleu",
            "exact_match_mean": "mean_exact_match",
            "cosim_mean": "mean_cosim",
        }
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)


def save_best_responses_csv(
    output_path: str,
    records: List[Dict],
    dataset: pd.DataFrame = None,
    metric_key: str = "rouge_l",
) -> None:
    """
    For each (system_prompt_id, attack), select the best response based on metric_key.

    Records should have: system_prompt_id, attack, target, response, and metric fields.
    Dataset should have: index (system_prompt_id), text (ground truth).
    Saves CSV with columns: system_prompt_id, ground_truth, attack, target, best_response, rouge_l, bleu, exact_match, cosim
    """
    if not records:
        return

    df = pd.DataFrame(records)

    # Group by system_prompt_id and attack, select best by metric
    best_responses = []
    for (sys_id, attack), group in df.groupby(["system_prompt_id", "attack"]):
        # Get row with max metric value
        best_idx = group[metric_key].idxmax()
        best_row = df.loc[best_idx].copy()

        # Add ground truth if dataset provided
        if dataset is not None:
            ground_truth_row = dataset[dataset["index"] == sys_id]
            if not ground_truth_row.empty:
                best_row["ground_truth"] = ground_truth_row.iloc[0]["text"]
            else:
                best_row["ground_truth"] = ""

        best_responses.append(best_row)

    best_df = pd.DataFrame(best_responses)

    # Select and order columns
    cols_to_keep = [
        "system_prompt_id", "ground_truth", "attack", "target", "best_response",
        "rouge_l", "bleu", "exact_match", "cosim"
    ]
    cols_to_keep = [c for c in cols_to_keep if c in best_df.columns]
    best_df = best_df[cols_to_keep]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(output_path, index=False)
