"""Synthetic wind noise for headset microphones.

Wind noise is not sound travelling through the air; it is turbulence at the mic
port. Three properties matter for enhancement and are what this generator models
(following the measurements in Nelke & Vary, "Measurement, analysis and
simulation of wind noise signals for mobile communication devices", IWAENC 2014):

1. Spectrum: energy is concentrated below a few hundred Hz with a steep roll-off.
2. Gusts: the level fluctuates strongly and non-stationarily over 0.1-2 s, with
   lulls where it nearly stops. Stationary noise estimators handle this badly.
3. Low inter-mic coherence: turbulence is local, so two mics 1-2 cm apart see
   almost independent waveforms (unlike acoustic noise, which is coherent at low
   frequencies). Dual-mic wind detectors rely on exactly this.

High wind also overloads the mic, so a random fraction of clips is soft-clipped.
"""

from __future__ import annotations

import numpy as np

from ..audio import SR, rms

CONTROL_RATE = 100  # Hz, rate of the gust envelope before interpolation


def _ou(m: int, tau_s: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck process sampled at CONTROL_RATE (stationary, std ``sigma``)."""
    a = np.exp(-1.0 / (CONTROL_RATE * tau_s))
    w = rng.standard_normal(m) * sigma * np.sqrt(1.0 - a * a)
    z = np.empty(m)
    z[0] = rng.standard_normal() * sigma
    for t in range(1, m):
        z[t] = a * z[t - 1] + w[t]
    return z


def gust_envelope(n: int, rng: np.random.Generator, sr: int = SR, lulls: bool | None = None) -> np.ndarray:
    """Positive, log-normal gust envelope of length ``n`` with optional lulls."""
    m = int(np.ceil(n / sr * CONTROL_RATE)) + 2
    log_env = _ou(m, tau_s=rng.uniform(0.2, 1.5), sigma=rng.uniform(0.4, 1.0), rng=rng)
    env = np.exp(log_env)
    if lulls if lulls is not None else rng.random() < 0.5:
        gate_z = _ou(m, tau_s=rng.uniform(0.5, 2.0), sigma=1.0, rng=rng)
        env *= 1.0 / (1.0 + np.exp(-4.0 * (gate_z - rng.uniform(-1.0, 0.5))))  # smooth on/off gate
    t_ctrl = np.arange(m) / CONTROL_RATE
    return np.interp(np.arange(n) / sr, t_ctrl, env)


def wind_noise(
    n: int,
    rng: np.random.Generator | int | None = None,
    sr: int = SR,
    n_mics: int = 1,
    shared_envelope: float = 0.8,
    saturate_prob: float = 0.3,
) -> np.ndarray:
    """Return wind noise of shape (n,) for one mic, or (n_mics, n), with unit overall RMS.

    ``shared_envelope`` in [0, 1] is how much of the gust envelope the mics share (the
    same wind gust hits both ports); the fine structure is always independent per mic.
    """
    rng = np.random.default_rng(rng)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    fc = rng.uniform(40.0, 250.0)  # corner frequency
    slope = rng.uniform(1.0, 2.0)  # power falls as f^(-2*slope) above fc
    shape = 1.0 / np.sqrt(1.0 + (f / fc) ** (2.0 * slope))
    shape[f < 15.0] = 0.0  # mics and codecs high-pass below ~15-20 Hz

    shared = gust_envelope(n, rng, sr)
    out = np.empty((n_mics, n), dtype=np.float64)
    for m in range(n_mics):
        fine = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n)
        own = gust_envelope(n, rng, sr)
        out[m] = fine * shared**shared_envelope * own ** (1.0 - shared_envelope)

    out /= rms(out, axis=None)
    if rng.random() < saturate_prob:
        clip = rng.uniform(1.5, 3.0)  # in units of RMS
        out = clip * np.tanh(out / clip)
        out /= rms(out, axis=None)
    out = out.astype(np.float32)
    return out[0] if n_mics == 1 else out
