"""Run an enhancement method over a fixed test set and score every item.

Writes ``results/<set>/<method>.csv`` (one row per item, all metrics) and returns a
summary: the overall mean plus breakdowns by noise type and by SNR, which is where the
interesting differences between methods show up (e.g. wind at -5 dB versus office at 15).
"""

from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import baselines
from .audio import read, write
from .data.testsets import read_manifest
from .metrics import all_metrics

METRICS = ["pesq", "stoi", "estoi", "si_sdr", "snr", "ssnr", "csig", "cbak", "covl",
           "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"]
HEADLINE = ["pesq", "csig", "cbak", "covl", "ssnr", "stoi", "si_sdr", "dnsmos_ovrl"]


def _model_method(path: str, two_channel: bool) -> Callable:
    import torch

    from .model import load_checkpoint

    torch.set_num_threads(1)  # one process per CPU core already
    model = load_checkpoint(path, "cpu")

    @torch.no_grad()
    def enhance(noisy, spacing_m=None):
        x = noisy[0] if two_channel else noisy  # single-channel model: front mic on the headset set
        return model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None])[0].numpy()

    return enhance


def get_method(name: str, two_channel: bool) -> Callable:
    if name.startswith("model:"):
        return _model_method(name.removeprefix("model:"), two_channel)
    table = baselines.TWO_CHANNEL if two_channel else baselines.SINGLE_CHANNEL
    if name not in table:
        raise KeyError(f"unknown {'2-ch' if two_channel else '1-ch'} method {name!r}; have {sorted(table)}")
    return table[name]


_STATE: dict = {}


def _init(set_dir: str, method: str, two_channel: bool, with_dnsmos: bool, save_dir: str | None) -> None:
    _STATE.update(set_dir=Path(set_dir), fn=get_method(method, two_channel), two=two_channel,
                  dnsmos=with_dnsmos, save=Path(save_dir) if save_dir else None)


def _score(row: dict) -> dict:
    s = _STATE
    clean = read(s["set_dir"] / row["clean"])
    noisy = read(s["set_dir"] / row["noisy"], mono=not s["two"])
    est = s["fn"](noisy, row["extra"]["mic_spacing_m"]) if s["two"] else s["fn"](noisy)
    if s["save"] is not None:
        write(s["save"] / f"{row['id']}.wav", est)
    return {"id": row["id"], "noise": row["noise"], "snr_db": row["snr_db"],
            **all_metrics(clean, est, with_dnsmos=s["dnsmos"])}


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r and r[key] not in ("", None) and not math.isnan(float(r[key]))]
    return float(np.mean([float(v) for v in vals])) if vals else float("nan")


def summarise(rows: list[dict]) -> dict:
    def block(rs):
        return {"n": len(rs), **{m: _mean(rs, m) for m in METRICS}}

    # Sets with mixed noises (headset2mic: "TBUS+OOFFICE+WIND") or continuous SNRs would give one
    # row per item; collapse them to wind vs acoustic-only and 5 dB SNR bins.
    many_noises = len({r["noise"] for r in rows}) > 12
    snrs = {float(r["snr_db"]) for r in rows if not math.isnan(float(r["snr_db"]))}
    bin_snr = len(snrs) > 12

    def noise_key(r):
        if not many_noises:
            return r["noise"]
        return "with WIND" if "WIND" in r["noise"].split("+") else "acoustic only"

    def snr_key(snr):
        if not bin_snr:
            return f"{snr:g}", snr
        lo = math.floor(snr / 5) * 5
        return f"[{lo:g},{lo + 5:g})", lo

    by_noise, by_snr, order = defaultdict(list), defaultdict(list), {}
    for r in rows:
        by_noise[noise_key(r)].append(r)
        snr = float(r["snr_db"])
        if not math.isnan(snr):
            key, sort = snr_key(snr)
            by_snr[key].append(r)
            order[key] = sort
    return {"overall": block(rows),
            "by_noise": {k: block(v) for k, v in sorted(by_noise.items())},
            "by_snr": {k: block(by_snr[k]) for k in sorted(by_snr, key=order.get)}}


def evaluate(set_dir: Path, method: str, out_dir: Path, workers: int | None = None, limit: int | None = None,
             with_dnsmos: bool = True, save_audio: bool = False, label: str | None = None) -> dict:
    rows = read_manifest(set_dir)
    if limit:
        rows = rows[:: max(1, len(rows) // limit)][:limit]  # spread across conditions
    two = set_dir.name == "headset2mic"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = label or method
    save = str(out_dir / "audio" / label) if save_audio else None
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    # spawn, not fork: forking a process that already started torch/BLAS threads can deadlock
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(set_dir), method, two, with_dnsmos, save)) as pool:
        results = list(tqdm(pool.imap(_score, rows, chunksize=4), total=len(rows), desc=f"{set_dir.name}/{label}"))
    with open(out_dir / f"{label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "noise", "snr_db", *METRICS], extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    return summarise(results)


def format_table(summaries: dict[str, dict], metrics: list[str] = HEADLINE) -> str:
    """Markdown table, one row per method, overall means."""
    head = "| method | " + " | ".join(metrics) + " |"
    sep = "|---|" + "---:|" * len(metrics)
    lines = [head, sep]
    for name, s in summaries.items():
        o = s["overall"]
        lines.append(f"| {name} | " + " | ".join("–" if math.isnan(o[m]) else f"{o[m]:.2f}" for m in metrics) + " |")
    return "\n".join(lines)


def format_breakdown(summaries: dict[str, dict], axis: str, metric: str) -> str:
    """Markdown table: rows = noise types or SNRs, columns = methods, cells = ``metric``."""
    keys = list(next(iter(summaries.values()))[axis])
    lines = [f"| {axis.removeprefix('by_')} | " + " | ".join(summaries) + " |", "|---|" + "---:|" * len(summaries)]
    for k in keys:
        cells = [summaries[m][axis].get(k, {}).get(metric, float("nan")) for m in summaries]
        lines.append(f"| {k} | " + " | ".join("–" if math.isnan(c) else f"{c:.2f}" for c in cells) + " |")
    return "\n".join(lines)
