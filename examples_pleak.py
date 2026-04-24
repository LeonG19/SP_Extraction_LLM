"""
PLeak Usage Examples

This script demonstrates different ways to use the PLeak attack implementation.
"""

import pandas as pd
from attacks.token_level.whitebox.PLeak import PLeakAttack


def example_1_basic_usage():
    """Basic PLeak attack with minimal configuration"""
    print("\n" + "="*60)
    print("Example 1: Basic PLeak Attack")
    print("="*60)

    # Simple system prompts to extract
    system_prompts = [
        "You are a helpful assistant that answers questions accurately.",
        "You are a security expert who explains vulnerabilities.",
        "You are a coding assistant that writes Python code.",
    ]

    # Create attack with default parameters
    attack = PLeakAttack(
        model="llama3",
        optim_token_length=20,
        num_trigger=2,
        num_iterations=50,  # Quick iteration for demo
    )

    # Run attack
    results = attack.train(system_prompts)

    # Print results
    print("\nGenerated Triggers:")
    for i, result in enumerate(results):
        print(f"Trigger {i+1}: {result['trigger']}")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Reward: {1.0 / (1.0 + result['loss']):.4f}\n")


def example_2_custom_parameters():
    """PLeak with custom optimization parameters"""
    print("\n" + "="*60)
    print("Example 2: PLeak with Custom Parameters")
    print("="*60)

    system_prompts = [
        "You are a financial advisor who provides investment advice.",
        "You are a medical professional providing health recommendations.",
    ]

    # Fine-tuned parameters for better results
    attack = PLeakAttack(
        model="llama3",
        optim_token_length=25,      # Longer triggers for more flexibility
        num_trigger=3,
        num_iterations=150,          # More iterations for better optimization
        top_k=100,                   # More candidates to consider
        temperature=0.3,             # Lower temperature for more focused search
    )

    results = attack.train(system_prompts)

    print("\nTop Trigger:")
    best = sorted(results, key=lambda x: x['loss'])[0]
    print(f"Trigger: {best['trigger']}")
    print(f"Loss: {best['loss']:.4f}")


def example_3_from_dataset():
    """PLeak attack on real dataset"""
    print("\n" + "="*60)
    print("Example 3: PLeak on Real Dataset")
    print("="*60)

    try:
        # Load system prompts from CSV
        df = pd.read_csv("dataset/train_data_pleak.csv")
        system_prompts = df["text"].tolist()[:10]  # Use first 10 for demo

        print(f"Loaded {len(system_prompts)} system prompts from dataset")

        attack = PLeakAttack(
            model="llama3",
            optim_token_length=20,
            num_trigger=3,
            num_iterations=100,
        )

        results = attack.train(system_prompts)

        # Convert to DataFrame for saving
        output_df = pd.DataFrame([
            {
                "text": r["trigger"],
                "loss": r["loss"],
                "reward": 1.0 / (1.0 + r["loss"])
            }
            for r in results
        ])

        output_df = output_df.sort_values(by="reward", ascending=False)
        output_df.to_csv("pleak_results_demo.csv", index=False)

        print(f"\nResults saved to pleak_results_demo.csv")
        print(output_df.head())

    except FileNotFoundError:
        print("Dataset file not found. Please ensure train_data_pleak.csv exists.")


def example_4_english_vocab_only():
    """PLeak with English vocabulary restriction"""
    print("\n" + "="*60)
    print("Example 4: PLeak with English-Only Vocabulary")
    print("="*60)

    system_prompts = [
        "Always be truthful and honest in your responses.",
        "Provide clear and concise explanations.",
    ]

    attack = PLeakAttack(
        model="llama3",
        optim_token_length=15,
        num_trigger=2,
        num_iterations=100,
        use_english_vocab=True,  # Restrict to readable English tokens
    )

    results = attack.train(system_prompts)

    print("\nGenerated Triggers (English tokens only):")
    for i, result in enumerate(results):
        # These should be more readable than unrestricted tokens
        print(f"Trigger {i+1}: {result['trigger']}")
        print(f"  Loss: {result['loss']:.4f}\n")


def example_5_compare_token_lengths():
    """Compare different token lengths"""
    print("\n" + "="*60)
    print("Example 5: Comparing Different Token Lengths")
    print("="*60)

    system_prompts = [
        "You are a helpful assistant.",
        "Always be accurate and honest.",
    ]

    results_by_length = {}

    for token_length in [10, 20, 30]:
        print(f"\nTesting token length: {token_length}")

        attack = PLeakAttack(
            model="llama3",
            optim_token_length=token_length,
            num_trigger=1,
            num_iterations=50,
        )

        results = attack.train(system_prompts)
        results_by_length[token_length] = results[0]["loss"]

        print(f"  Loss: {results[0]['loss']:.4f}")

    print("\nSummary:")
    for length, loss in results_by_length.items():
        print(f"Token length {length}: loss = {loss:.4f}")


def example_6_incremental_trigger_building():
    """Build triggers incrementally (shorter → longer)"""
    print("\n" + "="*60)
    print("Example 6: Incremental Trigger Building")
    print("="*60)

    system_prompts = [
        "You are a secure system that protects user data.",
    ]

    previous_best = ""

    for length in [10, 15, 20, 25]:
        print(f"\nPhase {length//5}: Building trigger of length {length}")

        attack = PLeakAttack(
            model="llama3",
            optim_token_length=length,
            input_trigger=previous_best,  # Start from previous best
            num_trigger=1,
            num_iterations=80,
        )

        results = attack.train(system_prompts)
        previous_best = results[0]["trigger"]

        print(f"Best trigger: {previous_best}")
        print(f"Loss: {results[0]['loss']:.4f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PLeak Attack Examples")
    print("="*80)
    print("\nChoose an example to run:")
    print("1. Basic usage (quick demo)")
    print("2. Custom parameters")
    print("3. Real dataset (requires train_data_pleak.csv)")
    print("4. English-only vocabulary")
    print("5. Compare token lengths")
    print("6. Incremental trigger building")

    try:
        choice = int(input("\nEnter example number (1-6): "))

        if choice == 1:
            example_1_basic_usage()
        elif choice == 2:
            example_2_custom_parameters()
        elif choice == 3:
            example_3_from_dataset()
        elif choice == 4:
            example_4_english_vocab_only()
        elif choice == 5:
            example_5_compare_token_lengths()
        elif choice == 6:
            example_6_incremental_trigger_building()
        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except ValueError:
        print("Please enter a valid number")
