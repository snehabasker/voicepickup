"""A corpus is many clips stored back to back in one int16 .npy, plus a CSV index.

Memory-mapping one file is far faster for random-access training than opening
tens of thousands of small WAVs, and it halves RAM/disk versus float32.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio import from_int16, to_int16


@dataclass
class Corpus:
    data: np.ndarray  # int16, possibly a read-only memmap
    names: list[str]
    groups: list[str]  # speaker for speech, environment for noise
    offsets: np.ndarray
    lengths: np.ndarray

    def __len__(self) -> int:
        return len(self.names)

    def get(self, i: int) -> np.ndarray:
        o, n = int(self.offsets[i]), int(self.lengths[i])
        return from_int16(self.data[o : o + n])

    def segment(self, i: int, start: int, n: int) -> np.ndarray:
        """``n`` samples of clip ``i`` from ``start``, wrapping around if the clip is short."""
        clip_len = int(self.lengths[i])
        o = int(self.offsets[i])
        if start + n <= clip_len:
            return from_int16(self.data[o + start : o + start + n])
        idx = (start + np.arange(n)) % clip_len
        return from_int16(self.data[o + idx])

    def indices_of(self, group: str) -> list[int]:
        return [i for i, g in enumerate(self.groups) if g == group]

    @property
    def group_names(self) -> list[str]:
        return sorted(set(self.groups))

    @classmethod
    def from_arrays(cls, arrays: list[np.ndarray], names: list[str], groups: list[str]) -> Corpus:
        ints = [to_int16(a) for a in arrays]
        lengths = np.array([len(a) for a in ints], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        data = np.concatenate(ints) if ints else np.zeros(0, np.int16)
        return cls(data, list(names), list(groups), offsets, lengths)

    def save(self, stem: str | Path) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.save(stem.with_suffix(".npy"), self.data)
        with open(stem.with_suffix(".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "group", "offset", "length"])
            w.writerows(zip(self.names, self.groups, self.offsets.tolist(), self.lengths.tolist()))

    @classmethod
    def open(cls, stem: str | Path) -> Corpus:
        stem = Path(stem)
        data = np.load(stem.with_suffix(".npy"), mmap_mode="r")
        with open(stem.with_suffix(".csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        return cls(
            data,
            [r["name"] for r in rows],
            [r["group"] for r in rows],
            np.array([int(r["offset"]) for r in rows], dtype=np.int64),
            np.array([int(r["length"]) for r in rows], dtype=np.int64),
        )

    def hours(self, sr: int = 16_000) -> float:
        return float(self.lengths.sum()) / sr / 3600.0
