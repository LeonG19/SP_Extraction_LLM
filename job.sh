#!/bin/bash
#SBATCH --job-name=eval_all_methods
#SBATCH --output=logs/%j.out          # stdout log (%j = job ID)
#SBATCH --error=logs/%j.err           # stderr log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                  # single H200 GPU
#SBATCH --mem=90G
#SBATCH --time=24:00:00               # 4 methods × 5 models × ~1hr each

# --- Environment Setup ---
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate se_leak

# --- Create output directories ---
mkdir -p logs evaluation_logs evaluation_results

# --- Run Comprehensive Evaluation ---
python evaluate_all_methods.py
