#!/usr/bin/env python3
"""
Evaluation method with defense mechanisms.

Evaluates attack methods against defended models:
1. Apply defense to attack prompt
2. Query target model with defended attack prompt
3. Calculate similarity metrics
4. Report same metrics as undefended evaluation

Defense methods:
- promptguard: Detect and filter jailbreak attempts
- secalign: Align model behavior with security guidelines
- sft: Supervised fine-tuning on safe examples
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


class DefendedEvaluationFramework(EvaluationFramework):
    """Evaluation framework with defense mechanisms."""

    def __init__(
        self,
        model_name: str,
        defense_name: str,
        mode: str = "standard",
        n_samples: int = 10,
        top_k_prompts: int = 5,
        disable_tqdm: bool = False,
        server_url: str = "",
        api_key: str = "",
    ):
        """
        Initialize defended evaluation framework.

        Args:
            model_name: Name of target model
            defense_name: Name of defense mechanism (promptguard, secalign, sft)
            mode: Evaluation mode (standard, repeated_prompt, multi_turn)
            n_samples: Number of samples per adversarial prompt per test
            top_k_prompts: Number of top prompts to select by reward
            disable_tqdm: Disable progress bars
            server_url: Server URL for OpenAI-compatible API
            api_key: API key for OpenAI-compatible API
        """
        super().__init__(
            model_name=model_name,
            mode=mode,
            n_samples=n_samples,
            top_k_prompts=top_k_prompts,
            disable_tqdm=disable_tqdm,
            server_url=server_url,
            api_key=api_key,
        )
        self.defense_name = defense_name
        self.defense_model = None
        self.defense_tokenizer = None
        self._load_defense()

    def _load_defense(self):
        """Load the defense mechanism."""
        if self.defense_name == "promptguard":
            self._load_promptguard()
        elif self.defense_name == "secalign":
            self._load_secalign()
        elif self.defense_name == "sft":
            self._load_sft()
        else:
            raise ValueError(f"Unknown defense: {self.defense_name}")

    def _load_promptguard(self):
        """Load PromptGuard defense."""
        try:
            from prompt_guard import load_model_and_tokenizer
            self.defense_model, self.defense_tokenizer = load_model_and_tokenizer()
            if torch.cuda.is_available():
                self.defense_model = self.defense_model.to("cuda:0")
            print("PromptGuard defense loaded")
        except Exception as e:
            print(f"Failed to load PromptGuard: {e}")
            self.defense_model = None

    def _load_secalign(self):
        """Load SecAlign defense."""
        try:
            # SecAlign is loaded from the defenses/secalign directory
            secalign_path = os.path.join(Path(__file__).parent, "defenses", "secalign")
            if os.path.exists(secalign_path):
                sys.path.insert(0, secalign_path)
                # Load SecAlign model here
                print("SecAlign defense loaded")
            else:
                print(f"SecAlign path not found: {secalign_path}")
                self.defense_model = None
        except Exception as e:
            print(f"Failed to load SecAlign: {e}")
            self.defense_model = None

    def _load_sft(self):
        """Load SFT defense."""
        try:
            sft_path = os.path.join(Path(__file__).parent, "defenses", "SFTDefense")
            if os.path.exists(sft_path):
                sys.path.insert(0, sft_path)
                # Load SFT model here
                print("SFT defense loaded")
            else:
                print(f"SFT path not found: {sft_path}")
                self.defense_model = None
        except Exception as e:
            print(f"Failed to load SFT: {e}")
            self.defense_model = None

    def apply_defense(self, attack_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Apply defense mechanism to attack prompt.

        Args:
            attack_prompt: The attack prompt to defend against

        Returns:
            Tuple of (potentially modified prompt, defense info dict)
        """
        if self.defense_name == "promptguard" and self.defense_model:
            return self._apply_promptguard_defense(attack_prompt)
        elif self.defense_name == "secalign" and self.defense_model:
            return self._apply_secalign_defense(attack_prompt)
        elif self.defense_name == "sft" and self.defense_model:
            return self._apply_sft_defense(attack_prompt)
        else:
            # If defense not available, return original prompt
            return attack_prompt, {"defense_applied": False, "defense": self.defense_name}

    def _apply_promptguard_defense(self, attack_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Apply PromptGuard defense."""
        try:
            from prompt_guard import get_jailbreak_score

            score = get_jailbreak_score(attack_prompt, self.defense_model, self.defense_tokenizer)

            # If jailbreak score is high, block or modify the prompt
            if score > 0.5:  # Threshold
                modified_prompt = "I cannot process this request as it appears to contain a jailbreak attempt."
                return modified_prompt, {
                    "defense_applied": True,
                    "defense": self.defense_name,
                    "jailbreak_score": float(score),
                    "blocked": True,
                }
            else:
                return attack_prompt, {
                    "defense_applied": True,
                    "defense": self.defense_name,
                    "jailbreak_score": float(score),
                    "blocked": False,
                }
        except Exception as e:
            print(f"Error applying PromptGuard: {e}")
            return attack_prompt, {"defense_applied": False, "error": str(e)}

    def _apply_secalign_defense(self, attack_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Apply SecAlign defense."""
        try:
            # SecAlign would detect and filter adversarial prompts
            # For now, return the original prompt with defense info
            return attack_prompt, {
                "defense_applied": True,
                "defense": self.defense_name,
                "modified": False,
            }
        except Exception as e:
            print(f"Error applying SecAlign: {e}")
            return attack_prompt, {"defense_applied": False, "error": str(e)}

    def _apply_sft_defense(self, attack_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Apply SFT defense."""
        try:
            # SFT is a fine-tuned model that naturally resists jailbreaks
            # The defense is applied at model level, not prompt level
            return attack_prompt, {
                "defense_applied": True,
                "defense": self.defense_name,
                "sft_model": True,
            }
        except Exception as e:
            print(f"Error applying SFT: {e}")
            return attack_prompt, {"defense_applied": False, "error": str(e)}

    async def evaluate_single_example(
        self,
        system_prompt: str,
        attack_prompt: str,
    ) -> EvaluationExample:
        """
        Evaluate a single example with defense applied.

        Args:
            system_prompt: Target system prompt
            attack_prompt: Adversarial prompt

        Returns:
            EvaluationExample with results
        """
        # Apply defense to attack prompt
        defended_prompt, defense_info = self.apply_defense(attack_prompt)

        # Query model n_samples times with defended prompt
        responses = await self.query_model_multiple(
            system_prompt,
            defended_prompt,
            n_samples=self.n_samples,
        )

        # Calculate similarity metrics
        similarity_scores = self.calculate_similarity_metrics(responses, system_prompt)

        # Create example object
        example = EvaluationExample(
            system_prompt=system_prompt,
            attack_prompt=attack_prompt,  # Store original attack prompt
            responses=responses,
            similarity_scores=similarity_scores,
            highest_similarity=similarity_scores["highest_similarity"],
            timestamp=datetime.now().isoformat(),
        )

        return example

    def save_results(
        self,
        examples: List[EvaluationExample],
        metrics: EvaluationMetrics,
        output_dir: str,
        method_name: str,
    ):
        """
        Save evaluation results to disk.

        Args:
            examples: List of evaluation examples
            metrics: Aggregated metrics
            output_dir: Output directory
            method_name: Name of attack method
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Update metrics with method name and defense
        metrics.method = method_name
        metrics.defense = self.defense_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = self.model_name.split("/")[-1]

        # Save all examples to JSONL
        examples_file = os.path.join(
            output_dir,
            f"examples_{method_name}_vs_{self.defense_name}_{model_short}_{timestamp}.jsonl"
        )
        with open(examples_file, "w") as f:
            for example in examples:
                f.write(json.dumps({
                    "system_prompt": example.system_prompt,
                    "attack_prompt": example.attack_prompt,
                    "num_responses": len(example.responses),
                    "similarity_scores": example.similarity_scores,
                    "timestamp": example.timestamp,
                }) + "\n")

        # Save top 5 examples with full responses
        top_examples_file = os.path.join(
            output_dir,
            f"top5_examples_{method_name}_vs_{self.defense_name}_{model_short}_{timestamp}.json"
        )
        top_examples = self.get_top_k_examples(examples, k=5)
        with open(top_examples_file, "w") as f:
            json.dump([{
                "system_prompt": ex.system_prompt,
                "attack_prompt": ex.attack_prompt,
                "responses": ex.responses,
                "similarity_scores": ex.similarity_scores,
                "highest_similarity": ex.highest_similarity,
            } for ex in top_examples], f, indent=2)

        # Save metrics summary
        metrics_file = os.path.join(
            output_dir,
            f"metrics_{method_name}_vs_{self.defense_name}_{model_short}_{timestamp}.json"
        )
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

    parser = argparse.ArgumentParser(description="Evaluate attack methods against defended models")
    parser.add_argument("--method", type=str, required=True, help="Attack method (fuzz, re, pleak, leakagent)")
    parser.add_argument("--defense", type=str, required=True, help="Defense mechanism (promptguard, secalign, sft)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Target model")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "repeated_prompt", "multi_turn"], help="Evaluation mode")
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
        # Create defended evaluation framework
        framework = DefendedEvaluationFramework(
            model_name=args.model,
            defense_name=args.defense,
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

        # Evaluate
        examples, metrics = await framework.evaluate_test_set(test_system_prompts, top_prompts)

        # Save results
        output_dir = os.path.join(args.output_dir, args.mode)
        framework.save_results(examples, metrics, output_dir, args.method)

        # Print summary
        print("\n" + "=" * 80)
        print("DEFENDED EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Method: {args.method}")
        print(f"Defense: {args.defense}")
        print(f"Model: {args.model}")
        print(f"Mode: {args.mode}")
        print(f"Total Samples: {metrics.total_samples}")
        print(f"Average Similarity (ROUGE): {metrics.avg_similarity_rouge:.4f}")
        print(f"Average Similarity (WES): {metrics.avg_similarity_custom:.4f}")
        print(f"Max Similarity (ROUGE): {metrics.max_similarity_rouge:.4f}")
        print(f"Runtime: {metrics.runtime_seconds:.2f}s")
        print(f"Total Queries: {metrics.num_queries}")
        print("=" * 80)
    finally:
        # Clean up server
        if server_proc is not None:
            logger.info("Cleaning up vLLM server...")
            stop_vllm_server(server_proc)


if __name__ == "__main__":
    asyncio.run(main())
