"""Extension 3: Combined 2x2 - Best-of-N over Multi-Turn trajectories."""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter

from extensions.best_of_n import (
    select_embed_centroid,
    select_longest_overlap,
    select_majority_vote,
    select_oracle,
)
from extensions.multi_turn import TURN_2_FOLLOWUP
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


async def run_combined(
    client,
    model: str,
    attack: str,
    target: str,
    n_values: List[int],
    temperature: float = 0.8,
    selectors: List[str] = None,
    dataset_path: str = "dataset/test_data_pleak.csv",
    n_attack_prompts: int = 5,
    output_dir: str = "results/combined",
    resume: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run Combined experiment: Best-of-N sampling over complete 2-turn conversations.

    For each (system_prompt, attack_prompt) pair:
    - Generate N complete 2-turn conversations
    - Apply selectors to the set of N turn-2 responses
    - Compute metrics and record all trajectories and selections

    Saves raw trajectories, selected responses, and scaling curves.
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output file paths
    raw_path = Path(output_dir) / f"raw_{attack}_{target}_{timestamp}.jsonl"
    scaling_curve_path = Path(output_dir) / f"scaling_curve_{attack}_{target}_{timestamp}.csv"

    logger.info(f"Starting Combined (BoN + Multi-Turn): attack={attack}, target={target}, max_n={max(n_values)}")
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

            # For each N value
            for n in sorted(n_values):

                # Generate N complete 2-turn conversations
                turn2_responses = []
                all_trajectories = []

                for trajectory_idx in range(n):

                    # Turn 1
                    turn1_messages = [
                        {"role": "system", "content": sys_text},
                        {"role": "user", "content": attack_prompt},
                    ]

                    response1 = (
                        await batch_sample(
                            client,
                            model,
                            turn1_messages,
                            n=1,
                            temperature=temperature,
                            limiter=limiter,
                            logger=logger,
                        )
                    )[0]

                    # Turn 2
                    turn2_messages = [
                        {"role": "system", "content": sys_text},
                        {"role": "user", "content": attack_prompt},
                        {"role": "assistant", "content": response1},
                        {"role": "user", "content": TURN_2_FOLLOWUP},
                    ]

                    response2 = (
                        await batch_sample(
                            client,
                            model,
                            turn2_messages,
                            n=1,
                            temperature=temperature,
                            limiter=limiter,
                            logger=logger,
                        )
                    )[0]

                    turn2_responses.append(response2)

                    # Store full trajectory
                    trajectory = {
                        "system_prompt_id": sys_id,
                        "attack": attack,
                        "target": target,
                        "n": n,
                        "trajectory_index": trajectory_idx,
                        "turn1_response": response1,
                        "turn2_response": response2,
                    }
                    all_trajectories.append(trajectory)
                    append_jsonl(str(raw_path), trajectory)

                # Apply selectors to turn-2 responses
                for selector in selectors:

                    if selector == "oracle":
                        best_idx = select_oracle(turn2_responses, sys_text)
                        best_response = turn2_responses[best_idx]
                    elif selector == "longest_overlap":
                        best_idx = select_longest_overlap(turn2_responses)
                        best_response = turn2_responses[best_idx]
                    elif selector == "majority_vote":
                        best_response = select_majority_vote(turn2_responses)
                    elif selector == "embed_centroid":
                        best_idx = select_embed_centroid(turn2_responses, embed_model)
                        best_response = turn2_responses[best_idx]
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
