#!/usr/bin/env python3
"""
Calculate comprehensive metrics comparison with LCS.
Selects top samples per system_prompt_id based on ROUGE-L scores.
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def word_edit_similarity(text1: str, text2: str) -> float:
    """Calculate word-level edit similarity (normalized)."""
    words1 = text1.split()
    words2 = text2.split()

    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    max_len = max(m, n)
    if max_len == 0:
        return 1.0
    return 1.0 - (dp[m][n] / max_len)


def lcs_similarity(text1: str, text2: str) -> float:
    """Calculate LCS (Longest Common Subsequence) similarity."""
    words1 = text1.split()
    words2 = text2.split()

    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    max_len = max(m, n)
    if max_len == 0:
        return 1.0
    return lcs_length / max_len


def load_system_prompts(csv_path: str) -> Dict[str, str]:
    """Load system prompts from CSV."""
    prompts = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt_id = row.get('index') or row.get('name') or row.get('system_prompt_id')
                prompt_text = row.get('text') or row.get('prompt')
                if prompt_id and prompt_text:
                    prompts[prompt_id.strip()] = prompt_text.strip()
    except Exception as e:
        print(f"Error loading system prompts: {e}", file=sys.stderr)
    return prompts


def process_jsonl_file(jsonl_path: str, system_prompts: Dict[str, str]) -> List[dict]:
    """Process JSONL and add LCS metrics."""
    entries = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt_id = entry.get('system_prompt_id')
                if not prompt_id or prompt_id not in system_prompts:
                    continue

                ground_truth = system_prompts[prompt_id]

                # Calculate LCS for each turn
                for turn_key in sorted([k for k in entry.keys() if k.startswith('turn')]):
                    if isinstance(entry[turn_key], dict) and 'response' in entry[turn_key]:
                        response = entry[turn_key]['response']
                        entry[turn_key]['lcs'] = lcs_similarity(ground_truth, response)
                        entry[turn_key]['wes'] = word_edit_similarity(ground_truth, response)

                entries.append(entry)
            except json.JSONDecodeError:
                continue

    return entries


def main():
    multi_turn_dir = Path('/home/lgarza3/projects/rlagent/evaluation_results/multi_turn')
    dataset_csv = Path('/home/lgarza3/projects/rlagent/dataset/awesome_chatgpt_prompts.csv')

    if not dataset_csv.exists():
        print(f"Error: Dataset CSV not found", file=sys.stderr)
        return 1

    print("Loading system prompts...")
    system_prompts = load_system_prompts(str(dataset_csv))
    print(f"Loaded {len(system_prompts)} system prompts\n")

    # Define file mappings
    methods = {
        'FUZZ': 'raw_fuzz_llama3.1-8b_20260509_233917.jsonl',
        'LEAKAGENT': 'raw_leakagent_llama3.1-8b_20260510_091721.jsonl',
        'PLEAK': 'raw_pleak_llama3.1-8b_20260510_140623.jsonl',
        'RE': 'raw_re_llama3.1-8b_20260510_042819.jsonl',
    }

    print("Processing files and selecting top samples by ROUGE-L...\n")

    comparison_data = []

    for method_name, filename in methods.items():
        jsonl_path = multi_turn_dir / filename
        if not jsonl_path.exists():
            print(f"Warning: {filename} not found", file=sys.stderr)
            continue

        print(f"Processing {method_name}...")

        # Load and process file
        entries = process_jsonl_file(str(jsonl_path), system_prompts)

        # Group by system_prompt_id and select by highest ROUGE-L in Turn 2
        grouped = defaultdict(list)
        for entry in entries:
            prompt_id = entry.get('system_prompt_id')
            if prompt_id:
                grouped[prompt_id].append(entry)

        # Select best by ROUGE-L (turn2)
        top_samples = []
        for prompt_id, entries_list in grouped.items():
            # Find entry with highest turn2 ROUGE-L
            best = max(entries_list, key=lambda e: e.get('turn2', {}).get('rouge_l', 0))
            top_samples.append(best)

        # Calculate statistics
        turn1_rouge = []
        turn2_rouge = []
        turn1_wes = []
        turn2_wes = []
        turn1_lcs = []
        turn2_lcs = []
        turn1_bleu = []
        turn2_bleu = []
        turn1_cosim = []
        turn2_cosim = []
        delta_rouge = []
        delta_wes = []
        delta_lcs = []
        delta_bleu = []

        for sample in top_samples:
            t1 = sample.get('turn1', {})
            t2 = sample.get('turn2', {})

            turn1_rouge.append(t1.get('rouge_l', 0))
            turn2_rouge.append(t2.get('rouge_l', 0))
            turn1_wes.append(t1.get('wes', 0))
            turn2_wes.append(t2.get('wes', 0))
            turn1_lcs.append(t1.get('lcs', 0))
            turn2_lcs.append(t2.get('lcs', 0))
            turn1_bleu.append(t1.get('bleu', 0))
            turn2_bleu.append(t2.get('bleu', 0))
            turn1_cosim.append(t1.get('cosim', 0))
            turn2_cosim.append(t2.get('cosim', 0))

            delta_rouge.append(t2.get('rouge_l', 0) - t1.get('rouge_l', 0))
            delta_wes.append(t2.get('wes', 0) - t1.get('wes', 0))
            delta_lcs.append(t2.get('lcs', 0) - t1.get('lcs', 0))
            delta_bleu.append(t2.get('bleu', 0) - t1.get('bleu', 0))

        # Calculate means
        def mean(values):
            return sum(values) / len(values) if values else 0

        comparison_data.append({
            'Method': method_name,
            'Samples': len(top_samples),
            'Turn1_ROUGE': f"{mean(turn1_rouge):.4f}",
            'Turn2_ROUGE': f"{mean(turn2_rouge):.4f}",
            'Δ_ROUGE': f"{mean(delta_rouge):.4f}",
            'Turn1_WES': f"{mean(turn1_wes):.4f}",
            'Turn2_WES': f"{mean(turn2_wes):.4f}",
            'Δ_WES': f"{mean(delta_wes):.4f}",
            'Turn1_LCS': f"{mean(turn1_lcs):.4f}",
            'Turn2_LCS': f"{mean(turn2_lcs):.4f}",
            'Δ_LCS': f"{mean(delta_lcs):.4f}",
            'Turn1_BLEU': f"{mean(turn1_bleu):.4f}",
            'Turn2_BLEU': f"{mean(turn2_bleu):.4f}",
            'Δ_BLEU': f"{mean(delta_bleu):.4f}",
        })

        print(f"  ✓ Selected {len(top_samples)} top samples by ROUGE-L\n")

    # Save comparison CSV
    output_csv = multi_turn_dir / 'all_metrics_comparison_with_lcs.csv'
    if comparison_data:
        keys = comparison_data[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(comparison_data)

        print("="*130)
        print(f"✓ Saved to {output_csv.name}\n")

        # Display results
        import pandas as pd
        df = pd.read_csv(output_csv)
        print("ALL METRICS COMPARISON (Top Samples Selected by ROUGE-L):")
        print("="*130)
        print(df.to_string(index=False))

        # Summary
        print("\n" + "="*130)
        print("METRIC SUMMARY:")
        print("="*130)

        for _, row in df.iterrows():
            method = row['Method']
            samples = row['Samples']
            print(f"\n{method} (n={samples} top samples):")
            print(f"  ROUGE:  T1={row['Turn1_ROUGE']} → T2={row['Turn2_ROUGE']} (Δ {row['Δ_ROUGE']})")
            print(f"  WES:    T1={row['Turn1_WES']} → T2={row['Turn2_WES']} (Δ {row['Δ_WES']})")
            print(f"  LCS:    T1={row['Turn1_LCS']} → T2={row['Turn2_LCS']} (Δ {row['Δ_LCS']})")
            print(f"  BLEU:   T1={row['Turn1_BLEU']} → T2={row['Turn2_BLEU']} (Δ {row['Δ_BLEU']})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
