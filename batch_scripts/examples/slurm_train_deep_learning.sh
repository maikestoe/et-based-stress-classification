#!/bin/bash
#SBATCH --job-name=et_stress_dl
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/examples/vr_goalkeeper_cnn_pd.json}"

cd "$REPO_ROOT"
python scripts/training/train_deep_learning.py --config "$CONFIG"

