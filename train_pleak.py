import argparse
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from attacks.token_level.whitebox.PLeak import PLeakAttack
from project_env import PROMPT_PATH


def save_pleak_results(method_obj, output_dir):
    """Save PLeak results to CSV in LeakAgent format"""
    os.makedirs(output_dir, exist_ok=True)

    # Extract triggers with their metrics
    results = []
    for result in method_obj.generated_triggers:
        results.append({
            "text": result["trigger"],
            "loss": result["loss"],
            "reward": 1.0 / (1.0 + result["loss"]),  # Convert loss to reward (lower loss = higher reward)
        })

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(by="reward", ascending=False)

    # Save with LeakAgent naming convention
    output_path = os.path.join(output_dir, "good_prompts.csv")
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"PLeak Training Completed!")
    print(f"{'='*60}")
    print(f"Total triggers generated: {len(df)}")
    print(f"Best loss: {df['loss'].min():.4f}")
    print(f"Average loss: {df['loss'].mean():.4f}")
    print(f"Best reward: {df['reward'].max():.4f}")
    print(f"Average reward: {df['reward'].mean():.4f}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")

    return output_path


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train PLeak attack to extract system prompts")

    parser.add_argument("--model", type=str, default="llama3",
                        help="Target model name (llama3, llama, etc.)")
    parser.add_argument("--dataset_path", type=str, default="train_data_pleak.csv",
                        help="Path to dataset containing system prompts to extract")
    parser.add_argument("--optim_token_length", type=int, default=20,
                        help="Number of adversarial tokens to optimize")
    parser.add_argument("--num_triggers", type=int, default=5,
                        help="Number of adversarial triggers to generate")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top candidates for token replacement")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for candidate selection")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate for optimization")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Number of optimization iterations per trigger")
    parser.add_argument("--use_english_vocab", action="store_true",
                        help="Restrict to English tokens only")
    parser.add_argument("--output_dir", type=str, default="pleak_results",
                        help="Output directory for results")

    args = parser.parse_args()

    # Load dataset
    dataset_full_path = os.path.join(PROMPT_PATH, args.dataset_path)
    print(f"\nLoading dataset from: {dataset_full_path}")
    system_prompts = pd.read_csv(dataset_full_path)["text"].tolist()

    print(f"\n{'='*60}")
    print(f"PLeak Attack Configuration")
    print(f"{'='*60}")
    print(f"Target model: {args.model}")
    print(f"System prompts to extract: {len(system_prompts)}")
    print(f"Adversarial token length: {args.optim_token_length}")
    print(f"Number of triggers: {args.num_triggers}")
    print(f"Optimization iterations: {args.num_iterations}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    # Initialize PLeak attack
    pleak = PLeakAttack(
        model=args.model,
        optim_token_length=args.optim_token_length,
        num_trigger=args.num_triggers,
        top_k=args.top_k,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        use_english_vocab=args.use_english_vocab,
    )

    # Run attack
    pleak.train(system_prompts)

    # Save results
    save_pleak_results(pleak, args.output_dir)
