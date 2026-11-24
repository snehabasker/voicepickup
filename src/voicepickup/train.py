"""Train the causal CRN with on-the-fly mixing, tracked in Weights & Biases.

    vp-train --config configs/base.yaml --run-dir runs/crn_base [--set train.lr=5e-4 ...]

* Resumable: if ``<run-dir>/last.pt`` exists, training continues from it *and* resumes the
  same W&B run (the run id is stored in the checkpoint), so a Slurm job that is preempted
  and requeued produces one continuous curve, not fragments.
* Preemption-safe: SIGUSR1/SIGTERM (Slurm sends one before killing a job) triggers a
  checkpoint and a clean exit with code 99, which the sbatch script uses to requeue.
* Sweep-aware: under ``wandb agent`` the sweep's dotted keys (``train.lr``...) override the
  YAML config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset

from .data.mixer import DynamicMixer, MixConfig
from .data.prepare import data_root, open_corpus
from .model import CRN, CRNConfig, braun_loss, n_params, stft

PREEMPT_EXIT_CODE = 99


# ---------------------------------------------------------------- config helpers

def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def unflatten(flat: dict) -> dict:
    out: dict = {}
    for key, v in flat.items():
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    return out


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = flatten(yaml.safe_load(Path(path).read_text()))
    for item in overrides:
        key, _, val = item.partition("=")
        if key not in cfg:
            raise KeyError(f"unknown config key {key!r}")
        cfg[key] = yaml.safe_load(val)
    return cfg


# ---------------------------------------------------------------- data

class MixStream(IterableDataset):
    """Endless stream of fresh noisy/clean pairs. Each worker gets an independent RNG
    seeded from (seed, start step, worker id) so a resumed run doesn't replay old data."""

    def __init__(self, root: Path, mix_cfg: MixConfig, seed: int, start_step: int):
        self.root, self.mix_cfg, self.seed, self.start_step = root, mix_cfg, seed, start_step

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, self.start_step, wid])
        mixer = DynamicMixer(open_corpus("vbd_train_clean", self.root), open_corpus("noise_train", self.root),
                             self.mix_cfg)
        while True:
            s = mixer.sample(rng)
            yield torch.from_numpy(s["noisy"]), torch.from_numpy(s["clean"])


def make_val_set(root: Path, mix_cfg: MixConfig, n: int, seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Fixed validation mixtures: unseen val speakers x val noise mics, 3 s each."""
    cfg = MixConfig(**{**mix_cfg.__dict__, "segment_s": 3.0})
    mixer = DynamicMixer(open_corpus("vbd_val_clean", root), open_corpus("noise_val", root), cfg)
    rng = np.random.default_rng(seed)
    items = [mixer.sample(rng) for _ in range(n)]
    noisy = torch.from_numpy(np.stack([s["noisy"] for s in items]))
    clean = torch.from_numpy(np.stack([s["clean"] for s in items]))
    return noisy, clean, [{"noise": s["noise"], "snr_db": s["snr_db"]} for s in items]


# ---------------------------------------------------------------- metrics / logging helpers

def si_sdr_t(ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    target = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-8) * ref
    return 10 * torch.log10(target.pow(2).sum(-1) / ((est - target).pow(2).sum(-1) + 1e-8) + 1e-8)


def spectrogram_image(x: np.ndarray) -> np.ndarray:
    """Log-magnitude spectrogram as a uint8 image (freq up), no matplotlib needed."""
    spec = np.abs(stft(torch.from_numpy(x)[None])[0].numpy()).T
    db = 20 * np.log10(spec + 1e-6)
    db = np.clip((db - (db.max() - 80)) / 80, 0, 1)
    return (np.flipud(db) * 255).astype(np.uint8)


def lr_at(step: int, cfg: dict) -> float:
    base, warm, total = cfg["train.lr"], cfg["train.warmup_steps"], cfg["train.steps"]
    if step < warm:
        return base * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


# ---------------------------------------------------------------- training

class Preempted(Exception):
    pass


def train(cfg: dict, run_dir: Path, root: Path) -> dict:
    import wandb

    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_path = run_dir / "last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else None
    if ckpt:
        cfg = {**ckpt["cfg"], **{k: v for k, v in cfg.items() if k.startswith("wandb.")}}

    run = wandb.init(
        project=cfg["wandb.project"], entity=cfg["wandb.entity"], mode=cfg["wandb.mode"],
        tags=cfg["wandb.tags"] or None, dir=str(run_dir), config=cfg,
        id=ckpt["wandb_id"] if ckpt else None, resume="allow",
        name=run_dir.name if not ckpt else None,
    )
    cfg = {**cfg, **{k: v for k, v in dict(wandb.config).items() if k in cfg}}  # sweep overrides
    slurm = {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                       "SLURM_RESTART_COUNT", "SLURMD_NODENAME") if k in os.environ}
    wandb.config.update({"slurm": slurm, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
                        allow_val_change=True)

    torch.manual_seed(cfg["seed"])
    model_cfg = CRNConfig(channels=tuple(cfg["model.channels"]), rnn_hidden=cfg["model.rnn_hidden"],
                          rnn_layers=cfg["model.rnn_layers"], compress=cfg["model.compress"])
    model = CRN(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train.lr"], weight_decay=cfg["train.weight_decay"])
    step, best = 0, {"val/pesq": -1.0}
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"resumed from step {step}")
    wandb.summary["params"] = n_params(model)

    mix_cfg = MixConfig(segment_s=cfg["data.segment_s"], snr_db=tuple(cfg["data.snr_db"]),
                        level_dbfs=tuple(cfg["data.level_dbfs"]), p_wind=cfg["data.p_wind"],
                        p_second_noise=cfg["data.p_second_noise"])
    loader = DataLoader(MixStream(root, mix_cfg, cfg["seed"], step), batch_size=cfg["train.batch_size"],
                        num_workers=cfg["train.num_workers"], pin_memory=device == "cuda",
                        persistent_workers=cfg["train.num_workers"] > 0)
    val_noisy, val_clean, val_meta = make_val_set(root, mix_cfg, cfg["train.val_items"])

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "cfg": cfg, "model_cfg": model_cfg.__dict__, "wandb_id": run.id}, tmp)
        os.replace(tmp, path)  # atomic: a kill mid-save never leaves a corrupt checkpoint

    stop = {"flag": False}

    def on_signal(signum, _frame):
        print(f"signal {signum}: checkpointing at step {step} and exiting", flush=True)
        stop["flag"] = True

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, on_signal)

    @torch.no_grad()
    def validate() -> tuple[dict, dict]:
        from .metrics import pesq_wb, stoi

        model.eval()
        losses, sdrs, outs = [], [], []
        for i in range(0, len(val_noisy), 32):
            noisy, clean = val_noisy[i : i + 32].to(device), val_clean[i : i + 32].to(device)
            est_c, _ = model.forward_spec(model.compress(stft(noisy)))
            losses.append(braun_loss(est_c, model.compress(stft(clean)), cfg["train.loss_lambda"]).item())
            est = model(noisy)
            sdrs.append(si_sdr_t(clean, est).cpu())
            outs.append(est.cpu())
        est_all = torch.cat(outs).numpy()
        sdr = torch.cat(sdrs).numpy()
        k = cfg["train.val_pesq_items"]
        idx = np.linspace(0, len(est_all) - 1, k).astype(int)
        pesqs = [pesq_wb(val_clean[j].numpy(), est_all[j]) for j in idx]
        stois = [stoi(val_clean[j].numpy(), est_all[j]) for j in idx]
        noisy_sdr = si_sdr_t(val_clean, val_noisy).numpy()
        logs = {"val/loss": float(np.mean(losses)), "val/si_sdr": float(np.mean(sdr)),
                "val/si_sdr_improvement": float(np.mean(sdr - noisy_sdr)),
                "val/pesq": float(np.nanmean(pesqs)), "val/stoi": float(np.mean(stois))}
        wind = [j for j, m in enumerate(val_meta) if "WIND" in m["noise"]]
        if wind:
            logs["val/si_sdr_wind"] = float(np.mean(sdr[wind]))
        media = {}
        for j in range(cfg["train.log_audio_items"]):
            tag = f"{val_meta[j]['noise']} @ {val_meta[j]['snr_db']:.1f} dB"
            for name, x in (("noisy", val_noisy[j].numpy()), ("enhanced", est_all[j]), ("clean", val_clean[j].numpy())):
                media[f"audio/{j}_{name}"] = wandb.Audio(x, sample_rate=16_000, caption=f"{name}: {tag}")
                media[f"spec/{j}_{name}"] = wandb.Image(spectrogram_image(x), caption=f"{name}: {tag}")
        model.train()
        return logs, media

    model.train()
    t0, seen, it = time.time(), 0, iter(loader)
    try:
        while step < cfg["train.steps"]:
            t_data = time.time()
            noisy, clean = next(it)
            t_data = time.time() - t_data
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            est_c, _ = model.forward_spec(model.compress(stft(noisy)))
            loss = braun_loss(est_c, model.compress(stft(clean)), cfg["train.loss_lambda"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train.grad_clip"])
            opt.step()
            step += 1
            seen += noisy.shape[0]

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % cfg["train.log_every"] == 0:
                dt = time.time() - t0
                logs = {"train/loss": loss.item(), "train/grad_norm": gnorm.item(), "train/lr": opt.param_groups[0]["lr"],
                        "perf/steps_per_s": cfg["train.log_every"] / dt, "perf/audio_x_realtime": seen * cfg["data.segment_s"] / dt,
                        "perf/data_wait_frac": t_data * cfg["train.log_every"] / dt}
                if device == "cuda":
                    logs["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                wandb.log(logs, step=step)
                if step % (cfg["train.log_every"] * 10) == 0:  # plain-text progress for Slurm logs
                    print(f"step {step} loss={logs['train/loss']:.4f} lr={logs['train/lr']:.2e} "
                          f"{logs['perf/steps_per_s']:.1f} it/s data_wait={logs['perf/data_wait_frac']:.0%}", flush=True)
                t0, seen = time.time(), 0
            if step % cfg["train.eval_every"] == 0 or step == cfg["train.steps"]:
                logs, media = validate()
                if logs["val/pesq"] > best["val/pesq"]:
                    best = {**logs, "step": step}
                    save(run_dir / "best.pt")
                for k, v in best.items():
                    wandb.summary[f"best/{k.removeprefix('val/')}"] = v
                wandb.log({**logs, **media}, step=step)
                print(f"step {step}: " + " ".join(f"{k}={v:.3f}" for k, v in logs.items()), flush=True)
            if step % cfg["train.ckpt_every"] == 0:
                save(ckpt_path)
            if stop["flag"]:
                raise Preempted
    except Preempted:
        save(ckpt_path)
        if hasattr(run, "mark_preempting"):
            run.mark_preempting()  # tells W&B the run will be resumed, not crashed
        run.finish(exit_code=PREEMPT_EXIT_CODE)
        sys.exit(PREEMPT_EXIT_CODE)

    save(ckpt_path)
    (run_dir / "summary.json").write_text(json.dumps({"best": best, "steps": step, "wandb_url": run.url}, indent=1))
    run.finish()
    return best


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vp-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--run-dir", type=Path, default=None, help="default: runs/<slurm job or timestamp>")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    name = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID"))
    if name and "SLURM_ARRAY_TASK_ID" in os.environ:
        name = f"{name}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    run_dir = a.run_dir or Path("runs") / (name or time.strftime("%Y%m%d-%H%M%S"))
    train(cfg, run_dir, a.data_root or data_root())


if __name__ == "__main__":
    main()
