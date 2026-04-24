import argparse
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from attacks.sentence_level.method.GeneticMethod import FuzzingMethod
from attacks.sentence_level.method.ReactMethod import ReActMethod
from attacks.token_level.whitebox.PLeak import PLeakAttack
from project_env import PROMPT_PATH


def extract_fuzz_results(method):
    """Extract prompts from FuzzingMethod"""
    prompts = []

    def traverse_tree(node):
        if node.prompt is not None:
            prompts.append({"text": node.prompt, "reward": node.reward})
        for child in node.children:
            traverse_tree(child)

    traverse_tree(method.root)
    return prompts


def extract_react_results(method):
    """Extract prompts from ReActMethod"""
    prompts = []
    for seed, info in method.seeds.items():
        prompts.append({"text": seed, "reward": info.get("reward", 0)})
    return prompts


def extract_pleak_results(method):
    """Extract prompts from PLeakAttack"""
    prompts = []
    for result in method.generated_triggers:
        # Convert loss to reward: lower loss = higher reward
        reward = 1.0 / (1.0 + result["loss"])
        prompts.append({"text": result["trigger"], "reward": reward})
    return prompts


def save_results(method_obj, method_name, output_dir):
    """Save baseline results to CSV in LeakAgent format"""
    os.makedirs(output_dir, exist_ok=True)

    if method_name == "fuzz":
        prompts = extract_fuzz_results(method_obj)
    elif method_name == "re":
        prompts = extract_react_results(method_obj)
    elif method_name == "pleak":
        prompts = extract_pleak_results(method_obj)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Create DataFrame with same format as LeakAgent
    df = pd.DataFrame(prompts)
    df = df.sort_values(by="reward", ascending=False)

    # Save with LeakAgent naming convention
    output_path = os.path.join(output_dir, "good_prompts.csv")
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Training completed for {method_name.upper()} baseline!")
    print(f"{'='*60}")
    print(f"Total prompts generated: {len(df)}")
    print(f"Best reward: {df['reward'].max():.4f}")
    print(f"Average reward: {df['reward'].mean():.4f}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")

    return output_path


if __name__ == "__main__":
    load_dotenv()
    argument = argparse.ArgumentParser()
    argument.add_argument("--helper_model", type=str, default="llama3")
    argument.add_argument("--target_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    argument.add_argument("--prompts_data_path", type=str, default="train_data_pleak.csv")
    argument.add_argument("--method", type=str, choices=["fuzz", "re", "pleak"], required=True)
    argument.add_argument("--target_server_url", type=str, default=None)
    argument.add_argument("--target_api_key", type=str, default=None)
    argument.add_argument("--output_dir", type=str, default=None)
    argument.add_argument("--env_num", type=int, default=2)
    argument.add_argument("--reason_model", type=str, default="llama3")
    argument.add_argument("--reflect_model", type=str, default="llama3")
    argument.add_argument("--pleak_model", type=str, default="llama3",
                        help="Model for PLeak (llama3, llama, mixtral, falcon, etc.)")
    argument.add_argument("--optim_token_length", type=int, default=20, help="Token length for PLeak")
    argument.add_argument("--num_iterations", type=int, default=100, help="Iterations for PLeak")

    args = argument.parse_args()

    # Use method name for output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"llama3_{args.method}_finetune"

    train_set = pd.read_csv(os.path.join(PROMPT_PATH, args.prompts_data_path))["text"].tolist()
    seeds = pd.read_csv(os.path.join(PROMPT_PATH, "injection-prompt.csv"))["text"].tolist()

    print(f"\n{'='*60}")
    print(f"Training {args.method.upper()} baseline")
    print(f"{'='*60}")
    print(f"Helper model: {args.helper_model}")
    print(f"Target model: {args.target_model}")
    print(f"Training samples: {len(train_set)}")
    print(f"Initial seeds: {len(seeds)}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    if args.method == "fuzz":
        method = FuzzingMethod(
            seeds, train_set, helper_model=args.helper_model, target_model=args.target_model,
            target_server_url=args.target_server_url, target_api_key=args.target_api_key
        )
    elif args.method == "re":
        method = ReActMethod(
            seeds, train_set, helper_model=args.helper_model, target_model=args.target_model,
            reason_model=args.reason_model, reflect_model=args.reflect_model,
            target_server_url=args.target_server_url, target_api_key=args.target_api_key
        )
    elif args.method == "pleak":
        method = PLeakAttack(
            model=args.pleak_model,
            optim_token_length=args.optim_token_length,
            num_trigger=5,
            num_iterations=args.num_iterations,
        )

    if args.method == "pleak":
        method.train(train_set)
    else:
        method.train()

    save_results(method, args.method, args.output_dir)
