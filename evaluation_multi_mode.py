#!/usr/bin/env python3
"""
Multi-mode evaluation for repeated_prompt and multi_turn modes.

This module extends the base evaluation framework to handle:
- repeated_prompt: Same adversarial prompt repeated in multiple messages
- multi_turn: Multiple conversation turns with different adversarial prompts

For these modes, we report:
- Metrics for first prompt/turn
- Metrics for second prompt/turn
- Delta metrics (improvement from first to second)
"""

import os
import sys
import json
import time
import atexit
import signal
import socket
import logging
import argparse
import asyncio
import subprocess
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.request import urlopen
from urllib.error import URLError

import torch
from evaluation_method import (
    EvaluationFramework, EvaluationExample, EvaluationMetrics,
    start_vllm_server, stop_vllm_server, _register_proc, _unregister_proc,
    _kill_proc_group, _cleanup_all_procs, _port_in_use, _wait_for_server,
    logger
)


@dataclass
class MultiModeEvaluationExample:
    """Evaluation example with metrics for multiple turns/prompts."""
    system_prompt: str
    first_attack_prompt: str
    second_attack_prompt: Optional[str]  # None for repeated_prompt

    # First turn/prompt responses and metrics
    first_responses: List[str]
    first_similarity_scores: Dict[str, float]
    first_highest_similarity: float

    # Second turn/prompt responses and metrics
    second_responses: List[str]
    second_similarity_scores: Dict[str, float]
    second_highest_similarity: float

    # Delta metrics
    delta_similarity: float

    timestamp: str


@dataclass
class MultiModeEvaluationMetrics:
    """Metrics for multi-mode evaluation."""
    method: str
    model: str
    mode: str  # "repeated_prompt" or "multi_turn"
    defense: Optional[str] = None
    total_samples: int = 0

    # First turn/prompt metrics
    avg_first_similarity: float = 0.0
    max_first_similarity: float = 0.0

    # Second turn/prompt metrics
    avg_second_similarity: float = 0.0
    max_second_similarity: float = 0.0

    # Delta metrics
    avg_delta_similarity: float = 0.0
    max_delta_similarity: float = 0.0

    runtime_seconds: float = 0.0
    num_queries: int = 0


class MultiModeEvaluationFramework(EvaluationFramework):
    """Evaluation framework for multi-turn and repeated prompt modes."""

    async def evaluate_repeated_prompt_example(
        self,
        system_prompt: str,
        attack_prompt: str,
    ) -> MultiModeEvaluationExample:
        """
        Evaluate repeated prompt mode.

        In this mode, the same attack prompt is repeated multiple times
        to see if the model eventually leaks the system prompt.

        Args:
            system_prompt: Target system prompt
            attack_prompt: Adversarial prompt (repeated twice)

        Returns:
            MultiModeEvaluationExample with first and second turn metrics
        """
        # First turn: single query with attack prompt
        first_responses = await self.query_model_multiple(
            system_prompt,
            attack_prompt,
            n_samples=self.n_samples,
        )

        first_similarity = self.calculate_similarity_metrics(first_responses, system_prompt)

        # Second turn: repeat the same prompt (user asks again)
        second_attack_prompt = attack_prompt  # Same prompt
        second_responses = await self.query_model_multiple(
            system_prompt,
            second_attack_prompt,
            n_samples=self.n_samples,
        )

        second_similarity = self.calculate_similarity_metrics(second_responses, system_prompt)

        # Calculate delta
        delta = second_similarity["highest_similarity"] - first_similarity["highest_similarity"]

        example = MultiModeEvaluationExample(
            system_prompt=system_prompt,
            first_attack_prompt=attack_prompt,
            second_attack_prompt=attack_prompt,  # Same prompt for repeated mode
            first_responses=first_responses,
            first_similarity_scores=first_similarity,
            first_highest_similarity=first_similarity["highest_similarity"],
            second_responses=second_responses,
            second_similarity_scores=second_similarity,
            second_highest_similarity=second_similarity["highest_similarity"],
            delta_similarity=delta,
            timestamp=datetime.now().isoformat(),
        )

        return example

    async def evaluate_multi_turn_example(
        self,
        system_prompt: str,
        attack_prompt_1: str,
        attack_prompt_2: str,
    ) -> MultiModeEvaluationExample:
        """
        Evaluate multi-turn mode.

        In this mode, two different adversarial prompts are used
        in a conversation to extract the system prompt.

        Args:
            system_prompt: Target system prompt
            attack_prompt_1: First adversarial prompt
            attack_prompt_2: Second adversarial prompt

        Returns:
            MultiModeEvaluationExample with metrics for both turns
        """
        # First turn with first attack prompt
        first_responses = await self.query_model_multiple(
            system_prompt,
            attack_prompt_1,
            n_samples=self.n_samples,
        )

        first_similarity = self.calculate_similarity_metrics(first_responses, system_prompt)

        # Second turn with second attack prompt
        # In a real multi-turn scenario, you'd maintain conversation history
        # For now, we just query with the second prompt
        second_responses = await self.query_model_multiple(
            system_prompt,
            attack_prompt_2,
            n_samples=self.n_samples,
        )

        second_similarity = self.calculate_similarity_metrics(second_responses, system_prompt)

        # Calculate delta
        delta = second_similarity["highest_similarity"] - first_similarity["highest_similarity"]

        example = MultiModeEvaluationExample(
            system_prompt=system_prompt,
            first_attack_prompt=attack_prompt_1,
            second_attack_prompt=attack_prompt_2,
            first_responses=first_responses,
            first_similarity_scores=first_similarity,
            first_highest_similarity=first_similarity["highest_similarity"],
            second_responses=second_responses,
            second_similarity_scores=second_similarity,
            second_highest_similarity=second_similarity["highest_similarity"],
            delta_similarity=delta,
            timestamp=datetime.now().isoformat(),
        )

        return example

    async def evaluate_repeated_prompt_test_set(
        self,
        test_system_prompts: List[str],
        attack_prompts: List[Dict[str, Any]],
    ) -> Tuple[List[MultiModeEvaluationExample], MultiModeEvaluationMetrics]:
        """
        Evaluate repeated prompt mode against test set.

        Args:
            test_system_prompts: List of test system prompts
            attack_prompts: List of attack prompt dictionaries

        Returns:
            Tuple of (evaluation examples, aggregated metrics)
        """
        start_time = time.time()
        all_examples = []
        first_similarities = []
        second_similarities = []
        delta_similarities = []
        num_queries = 0

        # For each test system prompt
        for system_prompt in tqdm(test_system_prompts, desc="Evaluating repeated prompts", disable=self.disable_tqdm):
            # For each attack prompt
            for attack_dict in attack_prompts:
                attack_prompt = attack_dict["text"]

                # Evaluate
                example = await self.evaluate_repeated_prompt_example(system_prompt, attack_prompt)
                all_examples.append(example)
                first_similarities.append(example.first_highest_similarity)
                second_similarities.append(example.second_highest_similarity)
                delta_similarities.append(example.delta_similarity)
                num_queries += self.n_samples * 2  # Two turns

        runtime = time.time() - start_time

        # Aggregate metrics
        metrics = MultiModeEvaluationMetrics(
            method="unknown",
            model=self.model_name.split("/")[-1],
            mode="repeated_prompt",
            total_samples=len(all_examples),
            avg_first_similarity=np.mean(first_similarities) if first_similarities else 0.0,
            max_first_similarity=max(first_similarities) if first_similarities else 0.0,
            avg_second_similarity=np.mean(second_similarities) if second_similarities else 0.0,
            max_second_similarity=max(second_similarities) if second_similarities else 0.0,
            avg_delta_similarity=np.mean(delta_similarities) if delta_similarities else 0.0,
            max_delta_similarity=max(delta_similarities) if delta_similarities else 0.0,
            runtime_seconds=runtime,
            num_queries=num_queries,
        )

        return all_examples, metrics

    async def evaluate_multi_turn_test_set(
        self,
        test_system_prompts: List[str],
        attack_prompts_1: List[Dict[str, Any]],
        attack_prompts_2: List[Dict[str, Any]],
    ) -> Tuple[List[MultiModeEvaluationExample], MultiModeEvaluationMetrics]:
        """
        Evaluate multi-turn mode against test set.

        Args:
            test_system_prompts: List of test system prompts
            attack_prompts_1: List of first attack prompt dictionaries
            attack_prompts_2: List of second attack prompt dictionaries

        Returns:
            Tuple of (evaluation examples, aggregated metrics)
        """
        start_time = time.time()
        all_examples = []
        first_similarities = []
        second_similarities = []
        delta_similarities = []
        num_queries = 0

        # For each test system prompt
        for system_prompt in tqdm(test_system_prompts, desc="Evaluating multi-turn", disable=self.disable_tqdm):
            # For each pair of attack prompts
            for ap1, ap2 in zip(attack_prompts_1, attack_prompts_2):
                attack_prompt_1 = ap1["text"]
                attack_prompt_2 = ap2["text"]

                # Evaluate
                example = await self.evaluate_multi_turn_example(
                    system_prompt,
                    attack_prompt_1,
                    attack_prompt_2,
                )
                all_examples.append(example)
                first_similarities.append(example.first_highest_similarity)
                second_similarities.append(example.second_highest_similarity)
                delta_similarities.append(example.delta_similarity)
                num_queries += self.n_samples * 2  # Two turns

        runtime = time.time() - start_time

        # Aggregate metrics
        metrics = MultiModeEvaluationMetrics(
            method="unknown",
            model=self.model_name.split("/")[-1],
            mode="multi_turn",
            total_samples=len(all_examples),
            avg_first_similarity=np.mean(first_similarities) if first_similarities else 0.0,
            max_first_similarity=max(first_similarities) if first_similarities else 0.0,
            avg_second_similarity=np.mean(second_similarities) if second_similarities else 0.0,
            max_second_similarity=max(second_similarities) if second_similarities else 0.0,
            avg_delta_similarity=np.mean(delta_similarities) if delta_similarities else 0.0,
            max_delta_similarity=max(delta_similarities) if delta_similarities else 0.0,
            runtime_seconds=runtime,
            num_queries=num_queries,
        )

        return all_examples, metrics

    def get_top_k_examples(self, examples: List[MultiModeEvaluationExample], k: int = 5) -> List[MultiModeEvaluationExample]:
        """Get top K examples by second turn similarity."""
        sorted_examples = sorted(examples, key=lambda x: x.second_highest_similarity, reverse=True)
        return sorted_examples[:k]

    def save_repeated_prompt_results(
        self,
        examples: List[MultiModeEvaluationExample],
        metrics: MultiModeEvaluationMetrics,
        output_dir: str,
        method_name: str,
    ):
        """Save repeated prompt evaluation results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        metrics.method = method_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = self.model_name.split("/")[-1]

        # Save all examples to JSONL
        examples_file = os.path.join(output_dir, f"examples_{method_name}_{model_short}_{timestamp}.jsonl")
        with open(examples_file, "w") as f:
            for example in examples:
                f.write(json.dumps({
                    "system_prompt": example.system_prompt,
                    "attack_prompt": example.first_attack_prompt,
                    "first_num_responses": len(example.first_responses),
                    "second_num_responses": len(example.second_responses),
                    "first_similarity": example.first_similarity_scores,
                    "second_similarity": example.second_similarity_scores,
                    "delta": example.delta_similarity,
                    "timestamp": example.timestamp,
                }) + "\n")

        # Save top 5 examples
        top_examples_file = os.path.join(output_dir, f"top5_examples_{method_name}_{model_short}_{timestamp}.json")
        top_examples = self.get_top_k_examples(examples, k=5)
        with open(top_examples_file, "w") as f:
            json.dump([{
                "system_prompt": ex.system_prompt,
                "attack_prompt": ex.first_attack_prompt,
                "first_responses": ex.first_responses,
                "second_responses": ex.second_responses,
                "first_similarity_scores": ex.first_similarity_scores,
                "second_similarity_scores": ex.second_similarity_scores,
                "delta": ex.delta_similarity,
            } for ex in top_examples], f, indent=2)

        # Save metrics
        metrics_file = os.path.join(output_dir, f"metrics_{method_name}_{model_short}_{timestamp}.json")
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        print(f"\nResults saved to {output_dir}")
        print(f"  - Examples: {examples_file}")
        print(f"  - Top 5: {top_examples_file}")
        print(f"  - Metrics: {metrics_file}")

        return {
            "examples": examples_file,
            "top_examples": top_examples_file,
            "metrics": metrics_file,
        }

    def save_multi_turn_results(
        self,
        examples: List[MultiModeEvaluationExample],
        metrics: MultiModeEvaluationMetrics,
        output_dir: str,
        method_name: str,
    ):
        """Save multi-turn evaluation results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        metrics.method = method_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = self.model_name.split("/")[-1]

        # Save all examples to JSONL
        examples_file = os.path.join(output_dir, f"examples_{method_name}_{model_short}_{timestamp}.jsonl")
        with open(examples_file, "w") as f:
            for example in examples:
                f.write(json.dumps({
                    "system_prompt": example.system_prompt,
                    "first_attack_prompt": example.first_attack_prompt,
                    "second_attack_prompt": example.second_attack_prompt,
                    "first_num_responses": len(example.first_responses),
                    "second_num_responses": len(example.second_responses),
                    "first_similarity": example.first_similarity_scores,
                    "second_similarity": example.second_similarity_scores,
                    "delta": example.delta_similarity,
                    "timestamp": example.timestamp,
                }) + "\n")

        # Save top 5 examples
        top_examples_file = os.path.join(output_dir, f"top5_examples_{method_name}_{model_short}_{timestamp}.json")
        top_examples = self.get_top_k_examples(examples, k=5)
        with open(top_examples_file, "w") as f:
            json.dump([{
                "system_prompt": ex.system_prompt,
                "first_attack_prompt": ex.first_attack_prompt,
                "second_attack_prompt": ex.second_attack_prompt,
                "first_responses": ex.first_responses,
                "second_responses": ex.second_responses,
                "first_similarity_scores": ex.first_similarity_scores,
                "second_similarity_scores": ex.second_similarity_scores,
                "delta": ex.delta_similarity,
            } for ex in top_examples], f, indent=2)

        # Save metrics
        metrics_file = os.path.join(output_dir, f"metrics_{method_name}_{model_short}_{timestamp}.json")
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        print(f"\nResults saved to {output_dir}")
        print(f"  - Examples: {examples_file}")
        print(f"  - Top 5: {top_examples_file}")
        print(f"  - Metrics: {metrics_file}")

        return {
            "examples": examples_file,
            "top_examples": top_examples_file,
            "metrics": metrics_file,
        }


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate attacks in multi-mode settings")
    parser.add_argument("--method", type=str, required=True, help="Attack method")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Target model")
    parser.add_argument("--mode", type=str, required=True, choices=["repeated_prompt", "multi_turn"], help="Evaluation mode")
    parser.add_argument("--training_prompts", type=str, help="Path to training prompts CSV")
    parser.add_argument("--test_data", type=str, default="test_data_pleak.csv", help="Path to test system prompts")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples per prompt")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top prompts to select")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bars")
    parser.add_argument("--skip_server", action="store_true", help="Skip server startup (assume already running)")

    args = parser.parse_args()

    # Start vLLM server if not skipping
    server_proc = None
    server_url = f"http://localhost:{args.port}/v1"

    if not args.skip_server:
        logger.info(f"Starting vLLM server for {args.model}...")
        server_proc = start_vllm_server(
            args.model,
            port=args.port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        if server_proc is None:
            logger.error("Failed to start vLLM server, aborting")
            sys.exit(1)
    else:
        logger.info(f"Assuming vLLM server already running at {server_url}")

    try:
        # Create evaluation framework
        framework = MultiModeEvaluationFramework(
            model_name=args.model,
            mode=args.mode,
            n_samples=args.n_samples,
            top_k_prompts=args.top_k,
            disable_tqdm=args.disable_tqdm,
            server_url=server_url,
            api_key="not-set",
        )

        # Load training prompts and select top K
        if not args.training_prompts:
            raise ValueError("Must provide --training_prompts argument")

        training_df = pd.read_csv(args.training_prompts)
        top_prompts = framework.select_top_prompts(training_df)
        logger.info(f"Selected top {len(top_prompts)} prompts by reward")

        # Load test system prompts
        from project_env import PROMPT_PATH
        test_data_path = os.path.join(PROMPT_PATH, args.test_data) if not os.path.isabs(args.test_data) else args.test_data
        test_df = pd.read_csv(test_data_path)
        test_system_prompts = test_df["text"].tolist()
        logger.info(f"Loaded {len(test_system_prompts)} test system prompts")

        # Evaluate based on mode
        output_dir = os.path.join(args.output_dir, args.mode)

        if args.mode == "repeated_prompt":
            examples, metrics = await framework.evaluate_repeated_prompt_test_set(
                test_system_prompts,
                top_prompts,
            )
            framework.save_repeated_prompt_results(examples, metrics, output_dir, args.method)
        else:  # multi_turn
            # For multi-turn, we use the same prompts for both turns (can be customized)
            examples, metrics = await framework.evaluate_multi_turn_test_set(
                test_system_prompts,
                top_prompts,
                top_prompts,
            )
            framework.save_multi_turn_results(examples, metrics, output_dir, args.method)

        # Print summary
        print("\n" + "=" * 80)
        print(f"EVALUATION SUMMARY ({args.mode.upper()})")
        print("=" * 80)
        print(f"Method: {args.method}")
        print(f"Model: {args.model}")
        print(f"Mode: {args.mode}")
        print(f"Total Samples: {metrics.total_samples}")
        print(f"\nFirst Prompt/Turn:")
        print(f"  Avg Similarity: {metrics.avg_first_similarity:.4f}")
        print(f"  Max Similarity: {metrics.max_first_similarity:.4f}")
        print(f"\nSecond Prompt/Turn:")
        print(f"  Avg Similarity: {metrics.avg_second_similarity:.4f}")
        print(f"  Max Similarity: {metrics.max_second_similarity:.4f}")
        print(f"\nDelta (Second - First):")
        print(f"  Avg Delta: {metrics.avg_delta_similarity:.4f}")
        print(f"  Max Delta: {metrics.max_delta_similarity:.4f}")
        print(f"\nRuntime: {metrics.runtime_seconds:.2f}s")
        print(f"Total Queries: {metrics.num_queries}")
        print("=" * 80)
    finally:
        # Clean up server
        if server_proc is not None:
            logger.info("Cleaning up vLLM server...")
            stop_vllm_server(server_proc)


if __name__ == "__main__":
    from tqdm import tqdm
    asyncio.run(main())
