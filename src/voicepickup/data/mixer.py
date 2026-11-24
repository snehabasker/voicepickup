"""On-the-fly (dynamic) mixing of clean speech with noise for training.

Every call draws a fresh clean segment, noise segment, SNR and level, so the model
never sees the same pair twice. This generalises better than the fixed 11.5k pairs of
the official VoiceBank-DEMAND train set and lets us add noise types it lacks (wind).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio import SR, noise_gain_for_snr, normalize_pair
from .corpus import Corpus
from .wind import wind_noise

WIND = "WIND"


@dataclass
class MixConfig:
    segment_s: float | None = 2.0  # None = whole utterance
    snr_db: tuple[float, float] = (-5.0, 15.0)
    level_dbfs: tuple[float, float] = (-35.0, -15.0)  # RMS of the noisy mixture
    p_wind: float = 0.15  # probability the noise is synthetic wind
    p_second_noise: float = 0.2  # probability of adding a second, weaker noise
    second_noise_rel_db: tuple[float, float] = (-10.0, 0.0)


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float, level_dbfs: float | None,
        active: slice | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mix at an exact SNR (measured on the ``active`` part of clean, default all of it),
    then scale both signals by one shared gain to the requested mixture level."""
    ref = clean if active is None else clean[active]
    noisy = clean + noise * noise_gain_for_snr(ref, noise, snr_db)
    return normalize_pair(noisy, clean, level_dbfs)


class DynamicMixer:
    def __init__(self, speech: Corpus, noise: Corpus, cfg: MixConfig | None = None, sr: int = SR):
        self.speech, self.noise, self.cfg, self.sr = speech, noise, cfg or MixConfig(), sr

    def _noise(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if rng.random() < self.cfg.p_wind:
            return wind_noise(n, rng, self.sr), WIND
        i = int(rng.integers(len(self.noise)))
        start = int(rng.integers(max(1, int(self.noise.lengths[i]))))
        return self.noise.segment(i, start, n), self.noise.groups[i]

    def sample(self, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        i = int(rng.integers(len(self.speech)))
        utt = self.speech.get(i)
        if cfg.segment_s is None:
            clean, active = utt, None
        else:
            n = int(cfg.segment_s * self.sr)
            if len(utt) >= n:
                s = int(rng.integers(len(utt) - n + 1))
                clean, active = utt[s : s + n], None
            else:  # pad short utterances; SNR is measured on the real speech only
                clean = np.zeros(n, np.float32)
                clean[: len(utt)] = utt
                active = slice(0, len(utt))
        n = len(clean)

        noise, kind = self._noise(n, rng)
        if rng.random() < cfg.p_second_noise:
            extra, kind2 = self._noise(n, rng)
            rel = 10.0 ** (rng.uniform(*cfg.second_noise_rel_db) / 20.0)
            noise = noise + extra * rel * noise_gain_for_snr(noise, extra, 0.0)
            kind = f"{kind}+{kind2}"

        snr = float(rng.uniform(*cfg.snr_db))
        level = float(rng.uniform(*cfg.level_dbfs))
        noisy, clean = mix(clean, noise, snr, level, active)
        return {"noisy": noisy, "clean": clean, "noise": kind, "snr_db": snr,
                "speaker": self.speech.groups[i], "utt": self.speech.names[i]}
