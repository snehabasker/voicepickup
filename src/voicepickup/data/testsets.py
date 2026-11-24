"""Fixed, seeded evaluation sets written to disk as WAV + manifest.csv.

* ``vbd_official``: the standard VoiceBank-DEMAND test set (824 utterances, 4 SNRs:
  2.5/7.5/12.5/17.5 dB). Kept so numbers are comparable with published results.
* ``grid``: unseen test speakers x the 5 held-out DEMAND environments + wind x SNR in
  {-5, 0, 5, 10, 15} dB. The official set has no negative SNRs and no wind, which is
  exactly where headset calls struggle, so this is the set for per-condition breakdowns.
* ``headset2mic``: simulated two-mic earcup recordings (see ``headset.py``), noisy is
  stereo, clean is the speech image at the front mic.

Manifest columns: id, noisy, clean, noise, snr_db, plus set-specific extras (JSON).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..audio import SR, write
from .corpus import Corpus
from .headset import meta_dict, simulate_headset
from .mixer import WIND, mix
from .wind import wind_noise

GRID_SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0)
MANIFEST_FIELDS = ["id", "noisy", "clean", "noise", "snr_db", "extra"]


def _write_manifest(out: Path, rows: list[dict]) -> None:
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(set_dir: Path) -> list[dict]:
    with open(Path(set_dir) / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["snr_db"] = float(r["snr_db"]) if r["snr_db"] not in ("", None) else float("nan")
        r["extra"] = json.loads(r["extra"]) if r.get("extra") else {}
    return rows


def parse_vbd_log(path: Path) -> dict[str, tuple[str, float]]:
    """VoiceBank-DEMAND log lines look like ``p232_001 living 17.5``."""
    labels = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                labels[parts[0]] = (parts[1], float(parts[2]))
    return labels


def make_vbd_official(clean: Corpus, noisy: Corpus, out: Path, log: Path | None = None) -> None:
    labels = parse_vbd_log(log) if log else {}
    rows = []
    for i, name in enumerate(tqdm(clean.names, desc="vbd_official")):
        assert noisy.names[i] == name
        write(out / "noisy" / f"{name}.wav", noisy.get(i))
        write(out / "clean" / f"{name}.wav", clean.get(i))
        kind, snr = labels.get(name, ("unknown", float("nan")))
        rows.append({"id": name, "noisy": f"noisy/{name}.wav", "clean": f"clean/{name}.wav",
                     "noise": kind, "snr_db": snr, "extra": "{}"})
    _write_manifest(out, rows)


def _noise_for(kind: str, noise: Corpus, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == WIND:
        return wind_noise(n, rng, SR)
    i = int(rng.choice(noise.indices_of(kind)))
    return noise.segment(i, int(rng.integers(noise.lengths[i])), n)


def make_grid(speech: Corpus, noise: Corpus, out: Path, n_utts: int = 120, seed: int = 0,
              snrs: tuple[float, ...] = GRID_SNRS, level_dbfs: float = -25.0) -> None:
    rng = np.random.default_rng(seed)
    utts = sorted(rng.choice(len(speech), size=min(n_utts, len(speech)), replace=False).tolist())
    kinds = [*noise.group_names, WIND]
    rows = []
    for u in tqdm(utts, desc="grid"):
        clean = speech.get(u)
        for kind in kinds:
            for snr in snrs:
                cid = f"{speech.names[u]}__{kind}__{snr:+.0f}dB"
                noisy, c = mix(clean, _noise_for(kind, noise, len(clean), rng), snr, level_dbfs)
                write(out / "noisy" / f"{cid}.wav", noisy)
                write(out / "clean" / f"{cid}.wav", c)
                rows.append({"id": cid, "noisy": f"noisy/{cid}.wav", "clean": f"clean/{cid}.wav",
                             "noise": kind, "snr_db": snr, "extra": json.dumps({"utt": speech.names[u]})})
    _write_manifest(out, rows)


def make_headset(speech: Corpus, noise: Corpus, out: Path, n_utts: int = 200, seed: int = 0,
                 p_wind: float = 0.3, snr_range: tuple[float, float] = (-5.0, 15.0)) -> None:
    rng = np.random.default_rng(seed)
    utts = rng.choice(len(speech), size=min(n_utts, len(speech)), replace=False).tolist()
    envs = noise.group_names
    rows = []
    for k, u in enumerate(tqdm(utts, desc="headset2mic")):
        clean = speech.get(u)
        n_src = int(rng.integers(1, 4))
        wind = bool(rng.random() < p_wind)
        if wind and rng.random() < 0.3:
            n_src = 0  # wind alone, e.g. walking outdoors in a quiet street
        kinds = [str(rng.choice(envs)) for _ in range(n_src)]
        noises = [_noise_for(kd, noise, len(clean), rng) for kd in kinds]
        snr = float(rng.uniform(*snr_range))
        sim = simulate_headset(clean, noises, snr, rng, wind=wind)
        cid = f"hs{k:04d}_{speech.names[u]}"
        write(out / "noisy" / f"{cid}.wav", sim["noisy"])
        write(out / "clean" / f"{cid}.wav", sim["target"])
        label = "+".join(sorted(set(kinds)) + ([WIND] if wind else []))
        rows.append({"id": cid, "noisy": f"noisy/{cid}.wav", "clean": f"clean/{cid}.wav", "noise": label,
                     "snr_db": round(snr, 2), "extra": json.dumps({**meta_dict(sim["meta"]), "utt": speech.names[u]})})
    _write_manifest(out, rows)
