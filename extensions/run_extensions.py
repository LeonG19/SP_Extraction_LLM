#!/usr/bin/env python3
"""CLI entry point for experimental extensions: Best-of-N, Multi-Turn, Combined."""

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from extensions.best_of_n import run_best_of_n
from extensions.combined import run_combined
from extensions.multi_turn import run_multi_turn
from extensions.repeated_prompt import run_repeated_prompt
from extensions.utils import make_client, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experimental extensions for LeakAgent prompt extraction attacks."
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["bon", "multiturn", "combined", "repeated_prompt", "all"],
        default="bon",
        help="Which extension to run: bon (Best-of-N), multiturn (2-turn), combined (BoN+Multi-Turn), repeated_prompt (prompt repetition), or all",
    )

    # Attack and target
    parser.add_argument(
        "--attack",
        choices=["fuzz", "re", "pleak", "leakagent"],
        required=True,
        help="Attack method",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target model name (e.g., llama3.1-8b)",
    )

    # Server connection
    parser.add_argument(
        "--server_url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible server URL",
    )
    parser.add_argument(
        "--api_key",
        default="EMPTY",
        help="API key for server",
    )

    # BoN-specific
    parser.add_argument(
        "--max_n",
        type=int,
        default=64,
        help="Maximum N for Best-of-N sweep",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--selector",
        choices=["oracle", "longest_overlap", "majority_vote", "embed_centroid", "all"],
        default="all",
        help="Selector strategy for BoN",
    )

    # Dataset
    parser.add_argument(
        "--dataset_path",
        default="dataset/test_data_pleak.csv",
        help="Path to test dataset CSV",
    )
    parser.add_argument(
        "--n_attack_prompts",
        type=int,
        default=5,
        help="Number of top attack prompts to use",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Base output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(timestamp)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Attack: {args.attack}, Target: {args.target}")
    logger.info(f"Server: {args.server_url}")

    # Create client
    client = make_client(args.server_url, args.api_key)

    # Determine output directories
    if args.mode == "bon":
        output_dir = f"{args.output_dir}/bon"
    elif args.mode == "multiturn":
        output_dir = f"{args.output_dir}/multiturn"
    elif args.mode == "combined":
        output_dir = f"{args.output_dir}/combined"
    elif args.mode == "repeated_prompt":
        output_dir = f"{args.output_dir}/repeated_prompt"
    else:
        output_dir = args.output_dir

    # Parse N values
    n_values = [n for n in [1, 2, 4, 8, 16, 32, 64] if n <= args.max_n]

    # Parse selectors
    if args.selector == "all":
        selectors = ["oracle", "longest_overlap", "majority_vote", "embed_centroid"]
    else:
        selectors = [args.selector]

    try:
        if args.mode == "bon" or args.mode == "all":
            logger.info("=== Running Best-of-N ===")
            await run_best_of_n(
                client=client,
                model=args.target,
                attack=args.attack,
                target=args.target,
                n_values=n_values,
                temperature=args.temperature,
                selectors=selectors,
                dataset_path=args.dataset_path,
                n_attack_prompts=args.n_attack_prompts,
                output_dir=f"{output_dir}/bon" if args.mode == "all" else output_dir,
                resume=args.resume,
                logger=logger,
            )

        if args.mode == "multiturn" or args.mode == "all":
            logger.info("=== Running Multi-Turn ===")
            await run_multi_turn(
                client=client,
                model=args.target,
                attack=args.attack,
                target=args.target,
                temperature=args.temperature,
                dataset_path=args.dataset_path,
                n_attack_prompts=args.n_attack_prompts,
                output_dir=f"{output_dir}/multiturn" if args.mode == "all" else output_dir,
                resume=args.resume,
                logger=logger,
            )

        if args.mode == "combined" or args.mode == "all":
            logger.info("=== Running Combined (BoN + Multi-Turn) ===")
            await run_combined(
                client=client,
                model=args.target,
                attack=args.attack,
                target=args.target,
                n_values=n_values,
                temperature=args.temperature,
                selectors=selectors,
                dataset_path=args.dataset_path,
                n_attack_prompts=args.n_attack_prompts,
                output_dir=f"{output_dir}/combined" if args.mode == "all" else output_dir,
                resume=args.resume,
                logger=logger,
            )

        if args.mode == "repeated_prompt" or args.mode == "all":
            logger.info("=== Running Repeated Prompt ===")
            await run_repeated_prompt(
                client=client,
                model=args.target,
                attack=args.attack,
                target=args.target,
                n_values=n_values,
                temperature=args.temperature,
                dataset_path=args.dataset_path,
                n_attack_prompts=args.n_attack_prompts,
                output_dir=f"{output_dir}/repeated_prompt" if args.mode == "all" else output_dir,
                resume=args.resume,
                logger=logger,
            )

        logger.info("=== All extensions completed successfully ===")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
