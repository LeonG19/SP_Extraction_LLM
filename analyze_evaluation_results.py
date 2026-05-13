#!/usr/bin/env python3
"""
Analyze evaluation results and generate comprehensive comparison reports.

This script reads all evaluation results and generates:
1. Summary tables comparing methods, models, and defenses
2. Attack success rankings
3. Defense effectiveness analysis
4. Multi-mode comparison (base vs repeated vs multi-turn)
"""

import os
import json
import glob
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class EvaluationAnalyzer:
    """Analyze evaluation results."""

    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = results_dir
        self.metrics = {}
        self.load_all_metrics()

    def load_all_metrics(self):
        """Load all metrics files from results directory."""
        for metrics_file in glob.glob(f"{self.results_dir}/**/metrics_*.json", recursive=True):
            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                    key = f"{data['method']}_{data['model']}_{data['mode']}"
                    if 'defense' in data and data['defense']:
                        key += f"_vs_{data['defense']}"
                    self.metrics[key] = data
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")

    def generate_summary_table(self, mode: str = "standard", defense: str = None) -> pd.DataFrame:
        """Generate summary table for a specific mode and defense."""
        rows = []

        for key, data in self.metrics.items():
            if data['mode'] == mode:
                if defense:
                    if not data.get('defense') or data['defense'] != defense:
                        continue
                else:
                    if data.get('defense'):
                        continue

                rows.append({
                    'Method': data['method'],
                    'Model': data['model'],
                    'Avg Similarity': f"{data['avg_similarity_rouge']:.4f}",
                    'Max Similarity': f"{data['max_similarity_rouge']:.4f}",
                    'Samples': data['total_samples'],
                    'Runtime (s)': f"{data['runtime_seconds']:.1f}",
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df.sort_values('Avg Similarity', ascending=False)

    def generate_method_ranking(self, mode: str = "standard") -> pd.DataFrame:
        """Rank attack methods by average similarity."""
        method_stats = defaultdict(lambda: {'total': 0, 'count': 0, 'max': 0})

        for key, data in self.metrics.items():
            if data['mode'] == mode and not data.get('defense'):
                method = data['method']
                method_stats[method]['total'] += data['avg_similarity_rouge']
                method_stats[method]['count'] += 1
                method_stats[method]['max'] = max(method_stats[method]['max'], data['max_similarity_rouge'])

        rows = []
        for method, stats in method_stats.items():
            rows.append({
                'Rank': 0,  # Will be set later
                'Method': method,
                'Avg Similarity': f"{stats['total'] / stats['count']:.4f}" if stats['count'] > 0 else 0,
                'Max Similarity': f"{stats['max']:.4f}",
                'Models Tested': stats['count'],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('Avg Similarity', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        return df[['Rank', 'Method', 'Avg Similarity', 'Max Similarity', 'Models Tested']]

    def generate_defense_effectiveness(self) -> pd.DataFrame:
        """Calculate defense effectiveness."""
        defense_stats = defaultdict(lambda: defaultdict(lambda: {'undefended': [], 'defended': []}))

        for key, data in self.metrics.items():
            if data['mode'] == 'standard':
                method = data['method']
                model = data['model']

                if data.get('defense'):
                    defense = data['defense']
                    defense_stats[method][defense]['defended'].append(data['avg_similarity_rouge'])
                else:
                    defense_stats[method]['undefended']['undefended'].append(data['avg_similarity_rouge'])

        # Calculate effectiveness
        rows = []
        for method in defense_stats:
            undefended_scores = defense_stats[method].get('undefended', {}).get('undefended', [])
            avg_undefended = sum(undefended_scores) / len(undefended_scores) if undefended_scores else 0

            for defense in ['promptguard', 'secalign', 'sft']:
                defended_scores = defense_stats[method].get(defense, {}).get('defended', [])
                if defended_scores:
                    avg_defended = sum(defended_scores) / len(defended_scores)
                    effectiveness = (avg_undefended - avg_defended) / avg_undefended * 100 if avg_undefended > 0 else 0

                    rows.append({
                        'Attack': method,
                        'Defense': defense,
                        'Undefended Avg': f"{avg_undefended:.4f}",
                        'Defended Avg': f"{avg_defended:.4f}",
                        'Effectiveness': f"{effectiveness:.1f}%",
                    })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def generate_multimode_comparison(self) -> pd.DataFrame:
        """Compare base, repeated, and multi-turn modes."""
        comparison = defaultdict(lambda: {})

        for key, data in self.metrics.items():
            if not data.get('defense'):
                method = data['method']
                model = data['model']
                mode = data['mode']

                key_str = f"{method}_{model}"
                if key_str not in comparison:
                    comparison[key_str] = {}

                if mode == 'standard':
                    comparison[key_str]['base'] = f"{data['avg_similarity_rouge']:.4f}"
                elif mode == 'repeated_prompt':
                    comparison[key_str]['repeated'] = f"{data['avg_similarity_rouge']:.4f}"
                    comparison[key_str]['repeated_delta'] = f"{data['avg_delta_similarity']:.4f}"
                elif mode == 'multi_turn':
                    comparison[key_str]['multi_turn'] = f"{data['avg_similarity_rouge']:.4f}"
                    comparison[key_str]['multi_turn_delta'] = f"{data['avg_delta_similarity']:.4f}"

        rows = []
        for key, modes in comparison.items():
            method, model = key.rsplit('_', 1)
            row = {
                'Method': method,
                'Model': model,
                'Base': modes.get('base', 'N/A'),
                'Repeated': modes.get('repeated', 'N/A'),
                'Repeated Δ': modes.get('repeated_delta', 'N/A'),
                'Multi-Turn': modes.get('multi_turn', 'N/A'),
                'Multi-Turn Δ': modes.get('multi_turn_delta', 'N/A'),
            }
            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def generate_model_comparison(self) -> pd.DataFrame:
        """Compare model vulnerability."""
        model_stats = defaultdict(lambda: {'total': 0, 'count': 0, 'max': 0})

        for key, data in self.metrics.items():
            if data['mode'] == 'standard' and not data.get('defense'):
                model = data['model']
                model_stats[model]['total'] += data['avg_similarity_rouge']
                model_stats[model]['count'] += 1
                model_stats[model]['max'] = max(model_stats[model]['max'], data['max_similarity_rouge'])

        rows = []
        for model, stats in model_stats.items():
            rows.append({
                'Rank': 0,
                'Model': model,
                'Avg Vulnerability': f"{stats['total'] / stats['count']:.4f}" if stats['count'] > 0 else 0,
                'Max Vulnerability': f"{stats['max']:.4f}",
                'Methods Tested': stats['count'],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('Avg Vulnerability', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        return df[['Rank', 'Model', 'Avg Vulnerability', 'Max Vulnerability', 'Methods Tested']]

    def print_report(self, output_file: str = None):
        """Print comprehensive evaluation report."""
        report_lines = []

        def print_and_log(text):
            print(text)
            report_lines.append(text)

        # Header
        print_and_log("=" * 100)
        print_and_log("COMPREHENSIVE EVALUATION REPORT")
        print_and_log("=" * 100)
        print_and_log(f"Total Evaluations: {len(self.metrics)}")
        print_and_log("")

        # Method Rankings
        print_and_log("\n" + "=" * 100)
        print_and_log("ATTACK METHOD RANKINGS (Standard Mode)")
        print_and_log("=" * 100)
        method_ranking = self.generate_method_ranking("standard")
        if not method_ranking.empty:
            print_and_log(method_ranking.to_string(index=False))
        else:
            print_and_log("No standard mode results available")
        print_and_log("")

        # Model Vulnerability Rankings
        print_and_log("\n" + "=" * 100)
        print_and_log("MODEL VULNERABILITY RANKINGS")
        print_and_log("=" * 100)
        model_ranking = self.generate_model_comparison()
        if not model_ranking.empty:
            print_and_log(model_ranking.to_string(index=False))
        else:
            print_and_log("No model comparison data available")
        print_and_log("")

        # Standard Mode Summary
        print_and_log("\n" + "=" * 100)
        print_and_log("STANDARD MODE - ALL ATTACKS AND MODELS")
        print_and_log("=" * 100)
        summary_table = self.generate_summary_table("standard")
        if not summary_table.empty:
            print_and_log(summary_table.to_string(index=False))
        else:
            print_and_log("No standard mode results available")
        print_and_log("")

        # Repeated Prompt Mode Summary
        print_and_log("\n" + "=" * 100)
        print_and_log("REPEATED PROMPT MODE - CUMULATIVE ATTACK EFFECTS")
        print_and_log("=" * 100)
        repeated_table = self.generate_summary_table("repeated_prompt")
        if not repeated_table.empty:
            print_and_log(repeated_table.to_string(index=False))
        else:
            print_and_log("No repeated prompt mode results available")
        print_and_log("")

        # Multi-Turn Mode Summary
        print_and_log("\n" + "=" * 100)
        print_and_log("MULTI-TURN MODE - CONVERSATION ATTACKS")
        print_and_log("=" * 100)
        multiturn_table = self.generate_summary_table("multi_turn")
        if not multiturn_table.empty:
            print_and_log(multiturn_table.to_string(index=False))
        else:
            print_and_log("No multi-turn mode results available")
        print_and_log("")

        # Multi-Mode Comparison
        print_and_log("\n" + "=" * 100)
        print_and_log("MULTI-MODE COMPARISON (Base vs Repeated vs Multi-Turn)")
        print_and_log("=" * 100)
        multimode = self.generate_multimode_comparison()
        if not multimode.empty:
            print_and_log(multimode.to_string(index=False))
        else:
            print_and_log("No multi-mode comparison data available")
        print_and_log("")

        # Defense Effectiveness
        print_and_log("\n" + "=" * 100)
        print_and_log("DEFENSE EFFECTIVENESS ANALYSIS")
        print_and_log("=" * 100)
        defense_eff = self.generate_defense_effectiveness()
        if not defense_eff.empty:
            print_and_log(defense_eff.to_string(index=False))
        else:
            print_and_log("No defense results available")
        print_and_log("")

        # PromptGuard Results (All Models)
        print_and_log("\n" + "=" * 100)
        print_and_log("PROMPTGUARD DEFENSE (All Models)")
        print_and_log("=" * 100)
        promptguard_table = self.generate_summary_table("standard", "promptguard")
        if not promptguard_table.empty:
            print_and_log(promptguard_table.to_string(index=False))
        else:
            print_and_log("No PromptGuard results available")
        print_and_log("")

        # SecAlign Results (llama3.1-8b)
        print_and_log("\n" + "=" * 100)
        print_and_log("SECALIGN DEFENSE (llama3.1-8b only)")
        print_and_log("=" * 100)
        secalign_table = self.generate_summary_table("standard", "secalign")
        if not secalign_table.empty:
            print_and_log(secalign_table.to_string(index=False))
        else:
            print_and_log("No SecAlign results available")
        print_and_log("")

        # SFT Results (llama3.1-8b)
        print_and_log("\n" + "=" * 100)
        print_and_log("SFT/LEAKAGENT-D DEFENSE (llama3.1-8b only)")
        print_and_log("=" * 100)
        sft_table = self.generate_summary_table("standard", "sft")
        if not sft_table.empty:
            print_and_log(sft_table.to_string(index=False))
        else:
            print_and_log("No SFT results available")
        print_and_log("")

        # Footer
        print_and_log("=" * 100)
        print_and_log("END OF REPORT")
        print_and_log("=" * 100)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation_results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to file (optional)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Save summary table to CSV",
    )

    args = parser.parse_args()

    analyzer = EvaluationAnalyzer(args.results_dir)

    if not analyzer.metrics:
        print(f"No evaluation results found in {args.results_dir}")
        return

    print(f"Loaded {len(analyzer.metrics)} evaluation results\n")

    # Print report
    analyzer.print_report(args.output)

    # Optionally save summary to CSV
    if args.csv:
        summary = analyzer.generate_summary_table("standard")
        if not summary.empty:
            summary.to_csv(args.csv, index=False)
            print(f"\nSummary table saved to: {args.csv}")


if __name__ == "__main__":
    main()
