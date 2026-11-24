"""Turn the raw downloads into 16 kHz memory-mapped corpora.

Splits (all speaker- or environment-disjoint where it matters):

* VoiceBank train (28 speakers) -> train (26) + validation (p226, p287), clean and noisy.
* VoiceBank test (p232, p257, unseen speakers) -> test, clean and noisy.
* DEMAND noise: the 5 environments used to build the official *test* set (DLIVING,
  OOFFICE, TBUS, PCAFETER, SPSQUARE) are held out for testing; the other 12 are used for
  training (mics 1-12) and validation (mics 13-16). So our test noises are never heard in
  training, matching the official protocol.
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..audio import read, to_int16
from .corpus import Corpus

VAL_SPEAKERS = ("p226", "p287")
TEST_NOISE_ENVS = ("DLIVING", "OOFFICE", "PCAFETER", "SPSQUARE", "TBUS")
TRAIN_NOISE_ENVS = ("DKITCHEN", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING",
                    "PRESTO", "PSTATION", "STRAFFIC", "TCAR", "TMETRO")
VAL_NOISE_MICS = (13, 14, 15, 16)


def data_root() -> Path:
    return Path(os.environ.get("VP_DATA_ROOT", Path.home() / "data" / "voicepickup"))


def _load(path: Path) -> np.ndarray:
    return to_int16(read(path))


def _build(paths: list[Path], names: list[str], groups: list[str], stem: Path, workers: int) -> Corpus:
    with Pool(workers) as pool:
        ints = pool.map(_load, paths, chunksize=16)
    lengths = np.array([len(a) for a in ints], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    c = Corpus(np.concatenate(ints), names, groups, offsets, lengths)
    c.save(stem)
    print(f"  {stem.name}: {len(c)} clips, {c.hours():.2f} h")
    return c


def prepare(root: Path | None = None, workers: int | None = None, force: bool = False) -> None:
    root = root or data_root()
    raw, out = root / "raw", root / "prepared"
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    def todo(stem: str) -> bool:
        return force or not (out / f"{stem}.npy").exists()

    # Each corpus is built as soon as its raw files exist, so evaluation can start while the
    # (slow) training download is still running. The official noisy train set is optional.
    for split, folder in (("train", "trainset_28spk_wav"), ("test", "testset_wav")):
        names = sorted(p.stem for p in (raw / "vbd" / f"clean_{folder}").glob("*.wav"))
        if not names:
            print(f"  skipping vbd {split}: {raw / 'vbd' / f'clean_{folder}'} not downloaded yet")
            continue
        subsets = {"test": names} if split == "test" else {
            "train": [n for n in names if n.split("_")[0] not in VAL_SPEAKERS],
            "val": [n for n in names if n.split("_")[0] in VAL_SPEAKERS],
        }
        for sub, sub_names in subsets.items():
            for kind in ("clean", "noisy"):
                stem = f"vbd_{sub}_{kind}"
                src = raw / "vbd" / f"{kind}_{folder}"
                if not src.is_dir():
                    print(f"  skipping {stem}: {src} not downloaded")
                elif todo(stem):
                    paths = [src / f"{n}.wav" for n in sub_names]
                    _build(paths, sub_names, [n.split("_")[0] for n in sub_names], out / stem, workers)

    demand = raw / "demand"
    plan = {
        "noise_train": [(e, m) for e in TRAIN_NOISE_ENVS for m in range(1, 17) if m not in VAL_NOISE_MICS],
        "noise_val": [(e, m) for e in TRAIN_NOISE_ENVS for m in VAL_NOISE_MICS],
        "noise_test": [(e, m) for e in TEST_NOISE_ENVS for m in range(1, 17)],
    }
    for stem, items in plan.items():
        if todo(stem):
            paths = [demand / e / f"ch{m:02d}.wav" for e, m in items]
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f"  skipping {stem}: {len(missing)} DEMAND files not downloaded yet, e.g. {missing[0]}")
                continue
            _build(paths, [f"{e}_ch{m:02d}" for e, m in items], [e for e, _ in items], out / stem, workers)


def open_corpus(stem: str, root: Path | None = None) -> Corpus:
    return Corpus.open((root or data_root()) / "prepared" / stem)
