# Running on Slurm

Training and sweeps go through Slurm, on a one-node cluster running in WSL2 on a laptop
with an RTX 5050 (8 GB). The job scripts in this folder work unchanged on a real
multi-node GPU cluster. Only the `--partition` and `--gres` names may need editing.

## Setup (once)

```bash
sudo bash slurm/install_wsl_slurm.sh
```

The script does the following:

- Installs `munge`, `slurmctld` and `slurmd`.
- Reads the real CPU, socket and memory layout from `slurmd -C`.
- Registers the GPU twice: as `gpu:1` (the whole card) and as `shard:4` (quarters of
  the card, so several small jobs can share it).
- Starts the daemons under systemd.
- Finishes with a test job that prints `CUDA_VISIBLE_DEVICES` and `nvidia-smi -L`.

Two WSL-specific details:

- WSL exposes the GPU as `/dev/dxg`, not `/dev/nvidia0`. The GRES is bound to that device,
  so Slurm still sets `CUDA_VISIBLE_DEVICES`. cgroup device isolation does not apply to
  it, so the config uses `proctrack/linuxproc` and `task/none`, with no cgroup setup.
- The Blackwell GPU (sm_120) needs a PyTorch wheel built for CUDA 12.8 or later. Older
  wheels import without errors but fail at the first kernel launch.
  `scripts/setup_env.sh` pins the right wheel and checks it by actually launching a kernel.

## One training run, with preemption and requeue

```bash
sbatch slurm/train.sbatch                          # configs/base.yaml
sbatch slurm/train.sbatch --set train.lr=5e-4      # extra args go to vp-train
```

The script handles preemption as follows:

1. `#SBATCH --signal=USR1@120` makes Slurm send `SIGUSR1` 120 s before the time limit.
   `scancel -s USR1 <jobid>` sends the same signal by hand, which simulates preemption.
2. `vp-train` catches the signal, saves `runs/<jobid>/last.pt` and exits with code 99.
   The save is atomic: it writes a temporary file and then renames it.
3. The batch script sees code 99 and runs `scontrol requeue $SLURM_JOB_ID`.
4. The requeued job has the same job ID, so it finds `runs/<jobid>/last.pt`. It restores
   the model, optimizer and step, and resumes the **same W&B run**, because the run ID
   is stored in the checkpoint. W&B shows one continuous curve, and
   `SLURM_RESTART_COUNT` is logged in the run config.

## Hyperparameter sweep as a job array

```bash
SWEEP=$(~/venvs/voicepickup/bin/wandb sweep --project voicepickup configs/sweep.yaml 2>&1 \
        | sed -n 's/.*wandb agent //p')
sbatch --array=0-11%2 slurm/sweep_agent.sbatch "$SWEEP"
```

- Each array task runs `wandb agent --count 1`, so it runs exactly one trial. Slurm
  controls how many trials run at once, not the agent.
- `%2` together with `--gres=shard:2` runs two trials side by side on one GPU. The
  1.2M-parameter model needs well under 1 GB of GPU memory, so a single trial leaves
  the GPU mostly idle.
- The sweep is a W&B Bayesian search over learning rate, GRU width, loss weighting and
  wind probability. Hyperband stops clearly bad trials early.

## Useful commands

```bash
squeue -o "%.10i %.12j %.8T %.10M %.6D %b"     # %b = GRES requested
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode,Restarts   # needs accounting; see below
scontrol show job <jobid> | grep -E "Restarts|TRES"
tail -f logs/slurm-vp-train-<jobid>.out
```

Job accounting (`slurmdbd`) is not set up on this single node. `scontrol` and the job logs
cover what is needed here.
