#!/usr/bin/env bash
# One training run. Preemption-safe: 120 s before the time limit (or on `scancel -s USR1`)
# Slurm signals the training step, which checkpoints and exits 99; we then requeue the job,
# and when it restarts it resumes from runs/<jobid>/last.pt into the same W&B run.
#
#   sbatch slurm/train.sbatch                       # defaults from configs/base.yaml
#   sbatch slurm/train.sbatch --set train.lr=5e-4   # any vp-train arguments pass through
#
#SBATCH --job-name=vp-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm-%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
VENV="${VENV:-$HOME/venvs/voicepickup}"
echo "[$(date -Is)] job ${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES:-none}"

srun --kill-on-bad-exit=1 "$VENV/bin/python" -m voicepickup.train --run-dir "runs/${SLURM_JOB_ID}" "$@"
code=$?
if [ "$code" -eq 99 ]; then
  echo "[$(date -Is)] preempted: checkpoint saved, requeueing"
  scontrol requeue "${SLURM_JOB_ID}"
fi
exit "$code"
