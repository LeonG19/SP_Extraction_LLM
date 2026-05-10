"""Extension 1: Best-of-N Repeated Sampling with multiple selector strategies."""

import asyncio
import logging
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from extensions.utils import (
    append_jsonl,
    batch_sample,
    compute_metrics,
    completed_triples,
    create_scaling_curve_csv,
    get_embed_model,
    load_attack_prompts,
    load_dataset,
    log_progress,
)


def select_oracle(responses: List[str], ground_truth: str) -> int:
    """Oracle selector: pick response with max ROUGE-L vs ground truth."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ground_truth, r)["rougeL"].fmeasure for r in responses]
    return int(np.argmax(scores))


def select_longest_overlap(responses: List[str]) -> int:
    """Longest overlap: pick response with longest common substring with any other response."""
    if len(responses) == 1:
        return 0

    max_overlaps = []
    for i, resp_i in enumerate(responses):
        max_overlap = 0
        for j, resp_j in enumerate(responses):
            if i != j:
                matcher = SequenceMatcher(None, resp_i, resp_j)
                longest_block = max([m.size for m in matcher.get_matching_blocks()], default=0)
                max_overlap = max(max_overlap, longest_block)
        max_overlaps.append(max_overlap)

    return int(np.argmax(max_overlaps))


def select_majority_vote(responses: List[str]) -> str:
    """Majority vote: token-level plurality voting, return closest response to synthetic result."""
    if len(responses) == 1:
        return responses[0]

    # Tokenize all responses
    tokenized = [r.split() for r in responses]

    # Find max length
    max_len = max(len(tokens) for tokens in tokenized)

    # Per-position majority vote
    synthetic_tokens = []
    for pos in range(max_len):
        tokens_at_pos = [tokenized[i][pos] for i in range(len(responses)) if pos < len(tokenized[i])]
        if tokens_at_pos:
            counter = Counter(tokens_at_pos)
            most_common_token = counter.most_common(1)[0][0]
            synthetic_tokens.append(most_common_token)

    synthetic = " ".join(synthetic_tokens)

    # Return response closest to synthetic (by ROUGE-L)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(synthetic, r)["rougeL"].fmeasure for r in responses]
    best_idx = int(np.argmax(scores))
    return responses[best_idx]


def select_embed_centroid(responses: List[str], embed_model: SentenceTransformer) -> int:
    """Embed-centroid: cluster embeddings, return response closest to largest cluster centroid."""
    if len(responses) == 1:
        return 0

    embeddings = embed_model.encode(responses, normalize_embeddings=True)

    k = max(2, min(3, len(responses) // 2))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Find largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]

    # Centroid of largest cluster
    cluster_embeddings = embeddings[labels == largest_cluster]
    centroid = cluster_embeddings.mean(axis=0)

    # Return index of closest response to centroid
    distances = [1 - np.dot(embeddings[i], centroid) for i in range(len(embeddings))]
    return int(np.argmin(distances))


async def run_best_of_n(
    client,
    model: str,
    attack: str,
    target: str,
    n_values: List[int],
    temperature: float = 0.8,
    selectors: List[str] = None,
    dataset_path: str = "dataset/test_data_pleak.csv",
    n_attack_prompts: int = 5,
    output_dir: str = "results/bon",
    resume: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run Best-of-N sampling experiment.

    Samples N responses for each (system_prompt, attack_prompt) pair, selects best via strategies.
    Saves raw samples, selected responses, and scaling curves.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if selectors is None:
        selectors = ["oracle", "longest_overlap", "majority_vote", "embed_centroid"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset and attack prompts
    dataset = load_dataset(dataset_path)
    attack_prompts = load_attack_prompts(attack, n=n_attack_prompts)
    embed_model = get_embed_model()

    # Timestamp for output files
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output file paths
    raw_path = Path(output_dir) / f"raw_samples_{attack}_{target}_{timestamp}.jsonl"
    scaling_curve_path = Path(output_dir) / f"scaling_curve_{attack}_{target}_{timestamp}.csv"

    # Track completed triples for resume
    completed = completed_triples(output_dir, f"raw_samples_{attack}_{target}")

    logger.info(f"Starting Best-of-N: attack={attack}, target={target}, max_n={max(n_values)}")
    logger.info(f"Selectors: {selectors}")
    logger.info(f"Dataset: {len(dataset)} system prompts, {len(attack_prompts)} attack prompts")

    start_time = time.time()
    total_prompts = len(dataset)
    done_prompts = 0
    all_metrics = []

    limiter = AsyncLimiter(10, 60)

    # Iterate over system prompts
    for _, row in dataset.iterrows():
        sys_id = row["index"]
        sys_text = row["text"]

        # Iterate over attack prompts
        for attack_prompt in attack_prompts:

            # Sample at the largest N value
            max_n = max(n_values)
            messages = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": attack_prompt},
            ]

            responses = await batch_sample(
                client,
                model,
                messages,
                n=max_n,
                temperature=temperature,
                limiter=limiter,
                logger=logger,
            )

            # Save all raw responses
            for sample_idx, response in enumerate(responses):
                record = {
                    "system_prompt_id": sys_id,
                    "attack": attack,
                    "target": target,
                    "n": len(responses),
                    "sample_index": sample_idx,
                    "response": response,
                }
                append_jsonl(str(raw_path), record)

            # Apply selectors to different N values
            for n in sorted(n_values):
                responses_n = responses[:n]

                # Apply each selector
                for selector in selectors:
                    if selector == "oracle":
                        best_idx = select_oracle(responses_n, sys_text)
                        best_response = responses_n[best_idx]
                    elif selector == "longest_overlap":
                        best_idx = select_longest_overlap(responses_n)
                        best_response = responses_n[best_idx]
                    elif selector == "majority_vote":
                        best_response = select_majority_vote(responses_n)
                    elif selector == "embed_centroid":
                        best_idx = select_embed_centroid(responses_n, embed_model)
                        best_response = responses_n[best_idx]
                    else:
                        raise ValueError(f"Unknown selector: {selector}")

                    # Compute metrics
                    metrics = compute_metrics(best_response, sys_text, embed_model)

                    # Record metrics for scaling curve
                    all_metrics.append(
                        {
                            "system_prompt_id": sys_id,
                            "attack": attack,
                            "target": target,
                            "n": n,
                            "selector": selector,
                            "response": best_response,
                            **metrics,
                        }
                    )

        done_prompts += 1
        if done_prompts % 5 == 0:
            running_rouge = np.mean([m["rouge_l"] for m in all_metrics[-10:]])
            log_progress(logger, done_prompts, total_prompts, start_time, running_rouge)

    # Create scaling curve
    create_scaling_curve_csv(str(scaling_curve_path), all_metrics)
    logger.info(f"Saved scaling curve to {scaling_curve_path}")

    # Save best responses per (system_prompt_id, attack) pair with ground truth
    from extensions.utils import save_best_responses_csv
    best_responses_path = Path(output_dir) / f"best_responses_{attack}_{target}_{timestamp}.csv"

    # Prepare records with response text
    best_response_records = []
    for record in all_metrics:
        best_response_records.append({
            "system_prompt_id": record["system_prompt_id"],
            "attack": record["attack"],
            "target": record["target"],
            "best_response": record["response"],
            "rouge_l": record["rouge_l"],
            "bleu": record["bleu"],
            "exact_match": record["exact_match"],
            "cosim": record["cosim"],
        })

    save_best_responses_csv(
        str(best_responses_path),
        best_response_records,
        dataset=dataset,
        metric_key="rouge_l"
    )
    logger.info(f"Saved best responses to {best_responses_path}")

    logger.info(f"Total time: {time.time() - start_time:.1f}s")
