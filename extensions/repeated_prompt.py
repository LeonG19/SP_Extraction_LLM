"""Extension 4: Repeated Prompt - test effect of prompt repetition within a single query."""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter

from extensions.utils import (
    append_jsonl,
    batch_sample,
    compute_metrics,
    create_scaling_curve_csv,
    get_embed_model,
    load_attack_prompts,
    load_dataset,
    log_progress,
)


async def run_repeated_prompt(
    client,
    model: str,
    attack: str,
    target: str,
    n_values: List[int],
    temperature: float = 0.8,
    dataset_path: str = "dataset/test_data_pleak.csv",
    n_attack_prompts: int = 5,
    output_dir: str = "results/repeated_prompt",
    resume: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run Repeated Prompt experiment: repeat the same attack prompt N times within a single query.

    For each (system_prompt, attack_prompt) pair, sends:
        {prompt} {prompt} {prompt} ... {prompt}  (N times, space-separated)

    Tests whether prompt repetition within a single query affects system prompt extraction.
    Saves raw responses and scaling curves grouped by N.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset and attack prompts
    dataset = load_dataset(dataset_path)
    attack_prompts = load_attack_prompts(attack, n=n_attack_prompts)
    embed_model = get_embed_model()

    # Timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output file paths
    raw_path = Path(output_dir) / f"raw_repeated_{attack}_{target}_{timestamp}.jsonl"
    scaling_curve_path = Path(output_dir) / f"scaling_curve_{attack}_{target}_{timestamp}.csv"

    logger.info(f"Starting Repeated Prompt: attack={attack}, target={target}, max_n={max(n_values)}")
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

            # For each repetition count
            for n in sorted(n_values):

                # Create repeated prompt: prompt + space + prompt + ... (n times)
                repeated_attack = " ".join([attack_prompt] * n)

                messages = [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": repeated_attack},
                ]

                response = (
                    await batch_sample(
                        client,
                        model,
                        messages,
                        n=1,
                        temperature=temperature,
                        limiter=limiter,
                        logger=logger,
                    )
                )[0]

                # Compute metrics
                metrics = compute_metrics(response, sys_text, embed_model)

                # Save raw response
                record = {
                    "system_prompt_id": sys_id,
                    "attack": attack,
                    "target": target,
                    "n_repetitions": n,
                    "response": response,
                    **metrics,
                }
                append_jsonl(str(raw_path), record)

                # Record metrics for scaling curve
                all_metrics.append(
                    {
                        "system_prompt_id": sys_id,
                        "attack": attack,
                        "target": target,
                        "n": n,
                        "selector": "repeated",
                        "response": response,
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

    best_responses_path = Path(output_dir) / f"best_responses_{attack}_{target}_{timestamp}.csv"
    save_best_responses_csv(
        str(best_responses_path),
        best_response_records,
        dataset=dataset,
        metric_key="rouge_l"
    )
    logger.info(f"Saved best responses to {best_responses_path}")

    logger.info(f"Total time: {time.time() - start_time:.1f}s")
