"""Small audio helpers: I/O, resampling, levels and SNR-controlled mixing."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16_000
EPS = 1e-10


def resample(x: np.ndarray, fs_in: int, fs_out: int = SR) -> np.ndarray:
    """Polyphase resampling along the last axis (anti-aliased, exact rational ratio)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float32)
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g, axis=-1).astype(np.float32)


def read(path: str | Path, sr: int = SR, mono: bool = True) -> np.ndarray:
    """Read a file as float32, resampled to ``sr``. Mono -> (n,), else (channels, n)."""
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1) if mono else x.T
    return resample(x, fs, sr)


def write(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    """Write 16-bit PCM. Accepts (n,) or (channels, n)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    sf.write(str(path), x.T if x.ndim == 2 else x, sr, subtype="PCM_16")


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x) * 32767.0), -32768, 32767).astype(np.int16)


def from_int16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / 32767.0


def rms(x: np.ndarray, axis: int | None = -1) -> np.ndarray | float:
    return np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=axis) + EPS)


def db(x: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.asarray(x) + EPS)


def noise_gain_for_snr(speech_ref: np.ndarray, noise_ref: np.ndarray, snr_db: float) -> float:
    """Gain to apply to the noise so that 20*log10(rms(speech)/rms(gain*noise)) == snr_db."""
    return float(rms(speech_ref, axis=None) / rms(noise_ref, axis=None) * 10.0 ** (-snr_db / 20.0))


def normalize_pair(
    noisy: np.ndarray, clean: np.ndarray, level_dbfs: float | None, ref: np.ndarray | None = None,
    peak: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale noisy and clean by the same gain: noisy (or ``ref``) to ``level_dbfs`` RMS, then
    back off if the noisy peak would clip. A shared gain keeps the SNR and the target exact."""
    g = 1.0
    if level_dbfs is not None:
        g = 10.0 ** (level_dbfs / 20.0) / rms(noisy if ref is None else ref, axis=None)
    pk = float(np.max(np.abs(noisy))) * g
    if pk > peak:
        g *= peak / pk
    return (noisy * g).astype(np.float32), (clean * g).astype(np.float32)
