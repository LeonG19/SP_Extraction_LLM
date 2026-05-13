#!/usr/bin/env python3
"""Calculate WES (Word Edit Similarity) metrics for multi-turn evaluations."""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict


def word_edit_similarity(text1: str, text2: str) -> float:
    """Calculate word-level edit similarity (normalized)."""
    words1 = text1.split()
    words2 = text2.split()

    # Levenshtein distance at word level
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


def load_system_prompts(csv_path: str) -> Dict[str, str]:
    """Load system prompts from CSV."""
    prompts = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Handle CSV that may have quotes and escaping
            reader = csv.DictReader(f)
            for row in reader:
                # The index column should be "index" or similar
                prompt_id = row.get('index') or row.get('name') or row.get('system_prompt_id')
                prompt_text = row.get('text') or row.get('prompt')
                if prompt_id and prompt_text:
                    prompts[prompt_id.strip()] = prompt_text.strip()
    except Exception as e:
        print(f"Error loading system prompts: {e}", file=sys.stderr)
    return prompts


def process_jsonl_file(jsonl_path: str, system_prompts: Dict[str, str]) -> Tuple[list, dict]:
    """Process a JSONL file and calculate WES metrics."""
    entries_with_wes = []
    stats = defaultdict(list)

    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)

                # Get the ground truth system prompt
                prompt_id = entry.get('system_prompt_id')
                if not prompt_id or prompt_id not in system_prompts:
                    print(f"Warning: Unknown system_prompt_id '{prompt_id}' at line {line_num}", file=sys.stderr)
                    continue

                ground_truth = system_prompts[prompt_id]

                # Calculate WES for each turn
                wes_scores = {}
                for turn_key in sorted([k for k in entry.keys() if k.startswith('turn')]):
                    if isinstance(entry[turn_key], dict) and 'response' in entry[turn_key]:
                        response = entry[turn_key]['response']
                        wes = word_edit_similarity(ground_truth, response)
                        entry[turn_key]['wes'] = wes
                        wes_scores[turn_key] = wes

                # Calculate delta_wes if we have turn1 and turn2
                if 'turn1' in wes_scores and 'turn2' in wes_scores:
                    entry['delta_wes'] = wes_scores['turn2'] - wes_scores['turn1']
                    stats['delta_wes'].append(entry['delta_wes'])

                # Track statistics
                for turn_key, wes in wes_scores.items():
                    stats[f'{turn_key}_wes'].append(wes)

                entries_with_wes.append(entry)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}", file=sys.stderr)
                continue

    return entries_with_wes, dict(stats)


def calculate_statistics(values: list) -> dict:
    """Calculate statistics for a list of values."""
    if not values:
        return {}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    return {
        'count': n,
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / n,
        'median': sorted_vals[n // 2],
        'std': (sum((x - sum(values)/n)**2 for x in values) / n) ** 0.5 if n > 1 else 0.0,
    }


def main():
    # Setup paths
    multi_turn_dir = Path('/home/lgarza3/projects/rlagent/evaluation_results/multi_turn')
    dataset_csv = Path('/home/lgarza3/projects/rlagent/dataset/awesome_chatgpt_prompts.csv')

    if not dataset_csv.exists():
        print(f"Error: Dataset CSV not found at {dataset_csv}", file=sys.stderr)
        return 1

    # Load system prompts
    print("Loading system prompts...")
    system_prompts = load_system_prompts(str(dataset_csv))
    print(f"Loaded {len(system_prompts)} system prompts")

    # Find all raw JSONL files
    raw_files = sorted(multi_turn_dir.glob('raw_*.jsonl'))
    print(f"\nFound {len(raw_files)} raw JSONL files:")
    for f in raw_files:
        print(f"  - {f.name}")

    print("\n" + "="*80)

    # Process each file
    all_stats = {}
    for jsonl_file in raw_files:
        print(f"\nProcessing {jsonl_file.name}...")
        entries, stats = process_jsonl_file(str(jsonl_file), system_prompts)

        # Save updated JSONL with WES metrics
        output_file = jsonl_file.parent / f"wes_{jsonl_file.name[4:]}"  # Remove 'raw_' prefix
        with open(output_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        print(f"  ✓ Saved {len(entries)} entries to {output_file.name}")

        # Print statistics
        print(f"\n  WES Statistics for {jsonl_file.name}:")
        for stat_key in sorted(stats.keys()):
            values = stats[stat_key]
            calc_stats = calculate_statistics(values)
            print(f"    {stat_key}:")
            print(f"      count:  {calc_stats.get('count', 0):>6}")
            print(f"      min:    {calc_stats.get('min', 0):>6.4f}")
            print(f"      mean:   {calc_stats.get('mean', 0):>6.4f}")
            print(f"      median: {calc_stats.get('median', 0):>6.4f}")
            print(f"      max:    {calc_stats.get('max', 0):>6.4f}")
            print(f"      std:    {calc_stats.get('std', 0):>6.4f}")

        all_stats[jsonl_file.name] = stats

    # Overall summary
    print("\n" + "="*80)
    print("\nOVERALL SUMMARY:\n")
    for file_name, stats in all_stats.items():
        print(f"{file_name}:")
        if 'delta_wes' in stats:
            calc_stats = calculate_statistics(stats['delta_wes'])
            print(f"  delta_wes mean:   {calc_stats.get('mean', 0):>6.4f}")
        if 'turn1_wes' in stats:
            calc_stats = calculate_statistics(stats['turn1_wes'])
            print(f"  turn1_wes mean:   {calc_stats.get('mean', 0):>6.4f}")
        if 'turn2_wes' in stats:
            calc_stats = calculate_statistics(stats['turn2_wes'])
            print(f"  turn2_wes mean:   {calc_stats.get('mean', 0):>6.4f}")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
