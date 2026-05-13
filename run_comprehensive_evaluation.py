#!/usr/bin/env python3
"""
Master script for comprehensive evaluation across all attack methods, models, and modes.

This script orchestrates:
1. Standard evaluation for all attack methods
2. Multi-mode evaluation (repeated_prompt, multi_turn)
3. Defense evaluation (promptguard, secalign, sft)
4. Aggregated results and comparison

Run without arguments for interactive mode, or provide specific arguments for automated runs.
"""

import os
import sys
import argparse
import asyncio
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Evaluation configuration
ATTACK_METHODS = {
    "fuzz": "llama3_fuzz_finetune",
    "re": "llama3_re_finetune",
    "pleak": "pleak_results",
    "leakagent": "llama3_rl_finetune",
}

MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
}

DEFENSES = ["promptguard", "secalign", "sft"]

EVALUATION_MODES = ["standard", "repeated_prompt", "multi_turn"]

class ComprehensiveEvaluationRunner:
    """Orchestrates comprehensive evaluation runs."""

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        modes: Optional[List[str]] = None,
        defenses: Optional[List[str]] = None,
        output_dir: str = "evaluation_results",
        server_url: str = "http://localhost:8000/v1",
        api_key: str = "not-set",
        n_samples: int = 10,
        top_k: int = 5,
        dry_run: bool = False,
    ):
        """
        Initialize evaluation runner.

        Args:
            methods: List of attack methods to evaluate
            models: List of models to evaluate
            modes: List of evaluation modes
            defenses: List of defenses to evaluate
            output_dir: Output directory for results
            server_url: Server URL for local models
            api_key: API key for local models
            n_samples: Number of samples per prompt
            top_k: Number of top prompts to select
            dry_run: If True, only print commands without running
        """
        self.methods = methods or list(ATTACK_METHODS.keys())
        self.models = models or list(MODELS.keys())
        self.modes = modes or EVALUATION_MODES
        self.defenses = defenses or DEFENSES
        self.output_dir = output_dir
        self.server_url = server_url
        self.api_key = api_key
        self.n_samples = n_samples
        self.top_k = top_k
        self.dry_run = dry_run
        self.results = []

    def get_training_prompts_path(self, method: str, model: str) -> str:
        """Get training prompts path for method and model."""
        method_dir = ATTACK_METHODS.get(method, method)
        return f"{method_dir}/{model}_prompts.csv"

    def run_standard_evaluation(self, method: str, model: str) -> bool:
        """Run standard evaluation."""
        print(f"\n{'='*80}")
        print(f"Standard Evaluation: {method} vs {model}")
        print(f"{'='*80}")

        training_prompts = self.get_training_prompts_path(method, model)
        model_hf = MODELS[model]

        cmd = [
            "python", "evaluation_method.py",
            "--method", method,
            "--model", model_hf,
            "--mode", "standard",
            "--training_prompts", training_prompts,
            "--n_samples", str(self.n_samples),
            "--top_k", str(self.top_k),
            "--output_dir", self.output_dir,
            "--server_url", self.server_url,
            "--api_key", self.api_key,
        ]

        return self._run_command(cmd)

    def run_multi_mode_evaluation(self, method: str, model: str, mode: str) -> bool:
        """Run multi-mode evaluation."""
        print(f"\n{'='*80}")
        print(f"{mode.replace('_', ' ').title()} Evaluation: {method} vs {model}")
        print(f"{'='*80}")

        training_prompts = self.get_training_prompts_path(method, model)
        model_hf = MODELS[model]

        cmd = [
            "python", "evaluation_multi_mode.py",
            "--method", method,
            "--model", model_hf,
            "--mode", mode,
            "--training_prompts", training_prompts,
            "--n_samples", str(self.n_samples),
            "--top_k", str(self.top_k),
            "--output_dir", self.output_dir,
            "--server_url", self.server_url,
            "--api_key", self.api_key,
        ]

        return self._run_command(cmd)

    def run_defense_evaluation(self, method: str, model: str, defense: str) -> bool:
        """Run defense evaluation."""
        print(f"\n{'='*80}")
        print(f"Defense Evaluation: {method} vs {defense} on {model}")
        print(f"{'='*80}")

        training_prompts = self.get_training_prompts_path(method, model)
        model_hf = MODELS[model]

        cmd = [
            "python", "evaluation_method_with_defense.py",
            "--method", method,
            "--defense", defense,
            "--model", model_hf,
            "--mode", "standard",
            "--training_prompts", training_prompts,
            "--n_samples", str(self.n_samples),
            "--top_k", str(self.top_k),
            "--output_dir", self.output_dir,
            "--server_url", self.server_url,
            "--api_key", self.api_key,
        ]

        return self._run_command(cmd)

    def _run_command(self, cmd: List[str]) -> bool:
        """Run a command and return success status."""
        if self.dry_run:
            print(f"[DRY RUN] {' '.join(cmd)}")
            return True

        try:
            result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def run_all_evaluations(self):
        """Run all configured evaluations."""
        print("\n" + "="*100)
        print("COMPREHENSIVE EVALUATION FRAMEWORK")
        print("="*100)
        print(f"Methods: {self.methods}")
        print(f"Models: {self.models}")
        print(f"Modes: {self.modes}")
        print(f"Defenses: {self.defenses}")
        print(f"Output: {self.output_dir}")
        print("="*100)

        total_evaluations = 0
        successful_evaluations = 0

        # Standard evaluations
        if "standard" in self.modes:
            print("\n" + "="*100)
            print("PHASE 1: STANDARD EVALUATIONS")
            print("="*100)
            for method in self.methods:
                for model in self.models:
                    total_evaluations += 1
                    if self.run_standard_evaluation(method, model):
                        successful_evaluations += 1
                    else:
                        print(f"⚠ Failed: {method} vs {model}")

        # Multi-mode evaluations
        for mode in self.modes:
            if mode != "standard":
                print("\n" + "="*100)
                print(f"PHASE: {mode.replace('_', ' ').upper()} EVALUATIONS")
                print("="*100)
                for method in self.methods:
                    for model in self.models:
                        total_evaluations += 1
                        if self.run_multi_mode_evaluation(method, model, mode):
                            successful_evaluations += 1
                        else:
                            print(f"⚠ Failed: {method} vs {model} ({mode})")

        # Defense evaluations
        print("\n" + "="*100)
        print("PHASE: DEFENSE EVALUATIONS")
        print("="*100)
        for method in self.methods:
            for model in self.models:
                for defense in self.defenses:
                    total_evaluations += 1
                    if self.run_defense_evaluation(method, model, defense):
                        successful_evaluations += 1
                    else:
                        print(f"⚠ Failed: {method} vs {defense} on {model}")

        # Summary
        print("\n" + "="*100)
        print("EVALUATION SUMMARY")
        print("="*100)
        print(f"Total Evaluations: {total_evaluations}")
        print(f"Successful: {successful_evaluations}")
        print(f"Failed: {total_evaluations - successful_evaluations}")
        print(f"Success Rate: {100.0 * successful_evaluations / total_evaluations:.1f}%")
        print(f"Results Directory: {self.output_dir}")
        print("="*100)

    def run_selective_evaluation(
        self,
        selected_method: str,
        selected_model: str,
        selected_modes: List[str],
        selected_defenses: Optional[List[str]] = None,
    ):
        """Run evaluation for specific method, model, and modes."""
        print(f"\nRunning selective evaluation for {selected_method} on {selected_model}")

        # Standard evaluation
        if "standard" in selected_modes:
            self.run_standard_evaluation(selected_method, selected_model)

        # Multi-mode evaluations
        for mode in selected_modes:
            if mode != "standard":
                self.run_multi_mode_evaluation(selected_method, selected_model, mode)

        # Defense evaluations
        if selected_defenses:
            for defense in selected_defenses:
                self.run_defense_evaluation(selected_method, selected_model, defense)


def interactive_mode():
    """Run in interactive mode."""
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION FRAMEWORK - INTERACTIVE MODE")
    print("="*100)

    print("\nAvailable attack methods:")
    for i, method in enumerate(ATTACK_METHODS.keys(), 1):
        print(f"  {i}. {method}")

    print("\nAvailable models:")
    for i, model in enumerate(MODELS.keys(), 1):
        print(f"  {i}. {model}")

    print("\nAvailable evaluation modes:")
    for i, mode in enumerate(EVALUATION_MODES, 1):
        print(f"  {i}. {mode}")

    print("\nAvailable defenses:")
    for i, defense in enumerate(DEFENSES, 1):
        print(f"  {i}. {defense}")

    # Get user selections
    print("\n" + "-"*100)
    method_input = input("Select method(s) (comma-separated indices or 'all'): ").strip()
    if method_input.lower() == "all":
        methods = list(ATTACK_METHODS.keys())
    else:
        method_indices = [int(x.strip()) - 1 for x in method_input.split(",")]
        methods = [list(ATTACK_METHODS.keys())[i] for i in method_indices]

    model_input = input("Select model(s) (comma-separated indices or 'all'): ").strip()
    if model_input.lower() == "all":
        models = list(MODELS.keys())
    else:
        model_indices = [int(x.strip()) - 1 for x in model_input.split(",")]
        models = [list(MODELS.keys())[i] for i in model_indices]

    mode_input = input("Select mode(s) (comma-separated indices or 'all'): ").strip()
    if mode_input.lower() == "all":
        modes = EVALUATION_MODES
    else:
        mode_indices = [int(x.strip()) - 1 for x in mode_input.split(",")]
        modes = [EVALUATION_MODES[i] for i in mode_indices]

    defense_input = input("Include defense evaluation? (y/n): ").strip().lower()
    defenses = DEFENSES if defense_input == "y" else []

    # Create runner and execute
    runner = ComprehensiveEvaluationRunner(
        methods=methods,
        models=models,
        modes=modes,
        defenses=defenses,
    )

    runner.run_all_evaluations()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation runner for attack and defense methods"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Specific attack method to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to evaluate"
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="standard,repeated_prompt,multi_turn",
        help="Comma-separated list of modes"
    )
    parser.add_argument(
        "--defenses",
        type=str,
        default="promptguard,secalign,sft",
        help="Comma-separated list of defenses"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all evaluations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory"
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Server URL"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="not-set",
        help="API key"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples per prompt"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top prompts"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if args.all:
        runner = ComprehensiveEvaluationRunner(
            output_dir=args.output_dir,
            server_url=args.server_url,
            api_key=args.api_key,
            n_samples=args.n_samples,
            top_k=args.top_k,
            dry_run=args.dry_run,
        )
        runner.run_all_evaluations()
    elif args.method and args.model:
        modes = args.modes.split(",")
        defenses = args.defenses.split(",") if args.defenses else []

        runner = ComprehensiveEvaluationRunner(
            methods=[args.method],
            models=[args.model],
            modes=modes,
            defenses=defenses,
            output_dir=args.output_dir,
            server_url=args.server_url,
            api_key=args.api_key,
            n_samples=args.n_samples,
            top_k=args.top_k,
            dry_run=args.dry_run,
        )
        runner.run_all_evaluations()
    else:
        print("Please use --all for comprehensive evaluation, --method and --model for specific evaluation,")
        print("or --interactive for interactive mode.")
        print("Run with --help for more options.")


if __name__ == "__main__":
    main()
