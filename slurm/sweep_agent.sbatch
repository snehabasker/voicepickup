#!/usr/bin/env bash
# One W&B sweep trial per array task. Two trials share the GPU via `shard` GRES
# (each gets half the card; the 1.2M-parameter model needs < 1 GB).
#
#   SWEEP=$(wandb sweep --project voicepickup configs/sweep.yaml 2>&1 | sed -n 's/.*wandb agent //p')
#   sbatch --array=0-11%2 slurm/sweep_agent.sbatch "$SWEEP"     # 12 trials, 2 at a time
#
#SBATCH --job-name=vp-sweep
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
VENV="${VENV:-$HOME/venvs/voicepickup}"
SWEEP_ID="${1:?usage: sbatch --array=0-N%2 slurm/sweep_agent.sbatch <entity/project/sweep_id>}"
echo "[$(date -Is)] sweep ${SWEEP_ID} task ${SLURM_ARRAY_TASK_ID} gpu=${CUDA_VISIBLE_DEVICES:-none}"
# --count 1: exactly one trial per array task, so Slurm (not the agent) controls concurrency
"$VENV/bin/wandb" agent --count 1 "$SWEEP_ID"
