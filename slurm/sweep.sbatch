#!/bin/bash
#SBATCH --job-name=bsfx
#SBATCH -p scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-03
#SBATCH --array=1-77%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Project paths — edit if you move the repo
NETID=af375
REPO=/work/${NETID}/transformer590
ENV=/work/${NETID}/envs/bsfx
SAVE_DIR=/work/${NETID}/binary_suffix_experiment

# HF / cache redirect (keeps $HOME quota safe)
export HF_HOME=/work/${NETID}/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Avoid picking up random packages from $HOME/.local
export PYTHONNOUSERSITE=1

# Micromamba activation
export MAMBA_ROOT_PREFIX=/work/${NETID}/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV"

# CUDA runtime libs (flash-attn / nvjitlink path)
export LD_LIBRARY_PATH="${ENV}/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${ENV}/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
mkdir -p logs "$SAVE_DIR"

# Read this task's (base, p) from manifest.txt — 1-indexed
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest.txt)
BASE=$(echo "$LINE" | awk '{print $1}')
P=$(echo "$LINE" | awk '{print $2}')

if [[ -z "$BASE" || -z "$P" ]]; then
    echo "ERROR: empty manifest line for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} base=${BASE} p=${P} host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python -u run.py \
    --base "$BASE" \
    --p "$P" \
    --save_dir "$SAVE_DIR" \
    --epochs 400

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} done"
