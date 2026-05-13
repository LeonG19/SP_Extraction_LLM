import pandas as pd
import os

metrics_files = [
    "/home/lgarza3/projects/rlagent/evaluation_results/multi_turn/metrics_fuzz_llama3.1-8b_20260509_233917.csv",
    "/home/lgarza3/projects/rlagent/evaluation_results/multi_turn/metrics_leakagent_llama3.1-8b_20260510_091721.csv",
    "/home/lgarza3/projects/rlagent/evaluation_results/multi_turn/metrics_re_llama3.1-8b_20260510_042819.csv",
]

# Numeric columns (metrics to average)
metric_columns = [
    'turn1_rouge_l', 'turn2_rouge_l', 'delta_rouge_l',
    'turn1_bleu', 'turn2_bleu',
    'turn1_exact', 'turn2_exact',
    'turn1_cosim', 'turn2_cosim'
]

results = {}

for file_path in metrics_files:
    file_name = os.path.basename(file_path)
    print(f"\nProcessing: {file_name}")
    print("=" * 80)

    # Read CSV
    df = pd.read_csv(file_path)

    print(f"Total rows: {len(df)}")
    print(f"Unique system_prompt_ids: {df['system_prompt_id'].nunique()}")

    # For each system_prompt_id, get the row with highest combined turn1_rouge_l + turn2_rouge_l
    df['rouge_combined_score'] = df['turn1_rouge_l'] + df['turn2_rouge_l']
    highest_scoring_samples = df.loc[df.groupby('system_prompt_id')['rouge_combined_score'].idxmax()]

    print(f"Samples selected (highest turn1_rouge_l + turn2_rouge_l per system_prompt_id): {len(highest_scoring_samples)}")

    # Calculate averages across all metrics
    averages = {}
    for col in metric_columns:
        averages[col] = highest_scoring_samples[col].mean()

    results[file_name] = {
        'averages': averages,
        'num_samples': len(highest_scoring_samples),
        'num_system_prompts': df['system_prompt_id'].nunique()
    }

    print("\nAverages across selected samples:")
    print("-" * 80)
    for metric, value in averages.items():
        print(f"{metric:20s}: {value:.6f}")

# Summary table
print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

summary_data = []
for file_name, data in results.items():
    row = {'File': file_name, 'Num Samples': data['num_samples']}
    row.update(data['averages'])
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save results to file
output_file = "/home/lgarza3/projects/rlagent/metrics_averages_summary.csv"
summary_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
