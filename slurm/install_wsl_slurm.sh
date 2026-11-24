#!/usr/bin/env bash
# Single-node Slurm "cluster" on WSL2 (Ubuntu 24.04+/systemd) with the GPU as a GRES.
# Run once:  sudo bash slurm/install_wsl_slurm.sh
#
# What it sets up
#   * munge (auth) + slurmctld (controller) + slurmd (compute daemon) on this machine
#   * node resources taken from `slurmd -C` (real CPU/socket/memory layout)
#   * GRES: gpu:1 (whole GPU) and shard:4 (GPU sharing: four jobs can each hold 1/4 of
#     the card, which is how two small sweep trials run side by side on an 8 GB GPU)
#   * one partition "gpu", job requeue enabled (for preemption + checkpoint/resume)
#
# WSL specifics
#   * WSL exposes the GPU as /dev/dxg (not /dev/nvidia0). We bind the GRES to that device
#     so Slurm sets CUDA_VISIBLE_DEVICES; there is no per-job device isolation (cgroups
#     device control does not apply to /dev/dxg), which is fine on a one-user box.
#   * proctrack/linuxproc + task/none: no cgroup setup needed.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq munge slurm-wlm >/dev/null

NODE="$(hostname -s)"
NODE_LINE="$(slurmd -C | head -1)"            # NodeName=... CPUs=... RealMemory=...
MEM="$(sed -n 's/.*RealMemory=\([0-9]*\).*/\1/p' <<<"$NODE_LINE")"
MEM=$(( MEM - 1024 ))                         # leave 1 GB for the OS
NODE_LINE="$(sed "s/RealMemory=[0-9]*/RealMemory=$MEM/" <<<"$NODE_LINE")"

mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
chown slurm:slurm /var/spool/slurmctld /var/log/slurm

cat > /etc/slurm/slurm.conf <<EOF
ClusterName=wsl
SlurmctldHost=$NODE
SlurmUser=slurm
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
ReturnToService=2
JobRequeue=1
MinJobAge=300
KillWait=60
GresTypes=gpu,shard
$NODE_LINE Gres=gpu:1,shard:4 State=UNKNOWN
PartitionName=gpu Nodes=$NODE Default=YES MaxTime=INFINITE State=UP DefMemPerCPU=512
EOF

cat > /etc/slurm/gres.conf <<EOF
NodeName=$NODE Name=gpu File=/dev/dxg
NodeName=$NODE Name=shard Count=4 File=/dev/dxg
EOF

[ -s /etc/munge/munge.key ] || { mungekey --create 2>/dev/null || create-munge-key -f; }
chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key

systemctl enable --now munge
systemctl restart slurmctld slurmd
systemctl enable slurmctld slurmd >/dev/null 2>&1

for _ in $(seq 30); do
  state="$(sinfo -h -N -o %T 2>/dev/null || true)"
  [ "$state" = "idle" ] && break
  sleep 1
done
scontrol update NodeName="$NODE" State=RESUME 2>/dev/null || true
sinfo -N -o "%N %T %C %m %G"
echo
echo "GPU test job:"
su -l "${SUDO_USER:-$USER}" -c "srun -p gpu --gres=gpu:1 -t 1 bash -c 'echo CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES; nvidia-smi -L'"
