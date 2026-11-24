"""Classical (non-learned) enhancement baselines.

Everything is causal and runs on 20 ms frames with a 10 ms hop (sqrt-Hann analysis and
synthesis, so 20 ms algorithmic latency: one window), the same framing as the neural
model, so the comparison is fair for a real-time headset.

Single-channel (``enhance(noisy) -> enhanced``):

* ``passthrough``      - the noisy input, for reference.
* ``specsub``          - power spectral subtraction with over-subtraction and a floor
                         (Berouti et al., 1979).
* ``wiener``           - Wiener gain with the decision-directed a-priori SNR
                         (Ephraim & Malah, 1984).
* ``mmse_lsa``         - MMSE log-spectral amplitude estimator (Ephraim & Malah, 1985).

All three need a noise PSD; it is tracked with the speech-presence-probability estimator
of Gerkmann & Hendriks (2012), which follows non-stationary noise better than minimum
statistics and needs no voice activity detector.

Two-channel, for the simulated headset (``enhance(noisy_2ch, spacing_m)``):

* ``front_mic``        - front mic only, then MMSE-LSA.
* ``diff_cardioid``    - first-order differential (delay-and-subtract) endfire beamformer
                         steered at the mouth, with low-frequency equalisation, then
                         MMSE-LSA as a post-filter. The classic small-array headset/hearing
                         aid design: a rear-facing null and ~4.8 dB directivity index in a
                         diffuse field, but it amplifies uncorrelated noise (mic self-noise
                         and wind) at low frequencies, which the EQ makes worse.
"""

from __future__ import annotations

import numpy as np
from scipy.special import exp1

from .audio import SR

WIN = 320  # 20 ms
HOP = 160  # 10 ms
SPEED_OF_SOUND = 343.0


def _window() -> np.ndarray:
    return np.sqrt(np.hanning(WIN + 1)[:-1])  # periodic sqrt-Hann: perfect reconstruction at 50% overlap


def stft(x: np.ndarray) -> np.ndarray:
    """(n,) -> (frames, WIN//2+1). Pads so every sample is covered by two frames."""
    x = np.concatenate([np.zeros(HOP), np.asarray(x, np.float64), np.zeros(WIN)])
    n = (len(x) - WIN) // HOP + 1
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[idx] * _window(), axis=1)


def istft(spec: np.ndarray, length: int) -> np.ndarray:
    frames = np.fft.irfft(spec, WIN, axis=1) * _window()
    out = np.zeros(HOP * (len(frames) - 1) + WIN)
    for t, f in enumerate(frames):
        out[t * HOP : t * HOP + WIN] += f
    return out[HOP : HOP + length].astype(np.float32)


def track_noise_psd(power: np.ndarray, init_frames: int = 5) -> np.ndarray:
    """Speech-presence-probability noise PSD tracker (Gerkmann & Hendriks, 2012)."""
    xi_h1 = 10 ** (15 / 10)  # fixed a-priori SNR under speech presence
    g = xi_h1 / (1 + xi_h1)
    noise = np.empty_like(power)
    est = power[:init_frames].mean(0) + 1e-12
    p_bar = np.zeros(power.shape[1])
    for t, y in enumerate(power):
        post = y / est
        p = 1 / (1 + (1 + xi_h1) * np.exp(-np.minimum(post * g, 50)))
        p_bar = 0.9 * p_bar + 0.1 * p
        p = np.where(p_bar > 0.99, np.minimum(p, 0.99), p)
        est = 0.8 * est + 0.2 * ((1 - p) * y + p * est)
        noise[t] = est
    return noise


def _gain_dd(power: np.ndarray, rule: str, alpha: float = 0.98, gmin_db: float = -25.0) -> np.ndarray:
    noise = track_noise_psd(power)
    gmin = 10 ** (gmin_db / 20)
    gains = np.empty_like(power)
    prev_clean = np.zeros(power.shape[1])
    for t in range(len(power)):
        gamma = np.minimum(power[t] / noise[t], 1e4)  # a-posteriori SNR
        xi = alpha * prev_clean / noise[t] + (1 - alpha) * np.maximum(gamma - 1, 0)  # decision-directed
        xi = np.maximum(xi, 10 ** (-25 / 10))
        if rule == "wiener":
            gain = xi / (1 + xi)
        else:  # mmse_lsa
            v = xi / (1 + xi) * gamma
            gain = xi / (1 + xi) * np.exp(0.5 * exp1(np.maximum(v, 1e-8)))
        gain = np.clip(gain, gmin, 1.0)
        gains[t] = gain
        prev_clean = (gain**2) * power[t]
    return gains


def specsub(noisy: np.ndarray, over: float = 2.0, floor: float = 0.02) -> np.ndarray:
    spec = stft(noisy)
    power = np.abs(spec) ** 2
    noise = track_noise_psd(power)
    clean_pow = np.maximum(power - over * noise, floor * noise)
    gain = np.sqrt(clean_pow / (power + 1e-12))
    return istft(spec * np.minimum(gain, 1.0), len(noisy))


def wiener(noisy: np.ndarray) -> np.ndarray:
    spec = stft(noisy)
    return istft(spec * _gain_dd(np.abs(spec) ** 2, "wiener"), len(noisy))


def mmse_lsa(noisy: np.ndarray) -> np.ndarray:
    spec = stft(noisy)
    return istft(spec * _gain_dd(np.abs(spec) ** 2, "mmse_lsa"), len(noisy))


def passthrough(noisy: np.ndarray) -> np.ndarray:
    return np.asarray(noisy, np.float32)


def diff_cardioid_beam(noisy: np.ndarray, spacing_m: float, max_eq_db: float = 15.0, sr: int = SR) -> np.ndarray:
    """First-order differential endfire beam: Y = X_front - X_rear * e^{-jw d/c}, equalised so
    the on-axis (mouth) response is flat. EQ gain is capped to limit the noise boost at low f."""
    front, rear = stft(noisy[0]), stft(noisy[1])
    w = 2 * np.pi * np.fft.rfftfreq(WIN, 1 / sr)
    tau = spacing_m / SPEED_OF_SOUND
    # Response to a far-field source on the front axis is 1 - e^{-2jwt} = 2j sin(wt) e^{-jwt}: a
    # 6 dB/octave high-pass with a 90 degree phase shift. Invert both (magnitude capped) so mouth
    # speech comes out time-aligned with the front-mic target; magnitude-only EQ would leave a
    # Hilbert-like phase error that wrecks waveform metrics such as SI-SDR.
    on_axis = 1 - np.exp(-2j * w * tau)
    eq = np.exp(-1j * np.angle(on_axis)) / np.maximum(np.abs(on_axis), 10 ** (-max_eq_db / 20))
    eq[0] = 0.0
    y = (front - rear * np.exp(-1j * w * tau)) * eq
    return istft(y, noisy.shape[1])


SINGLE_CHANNEL = {"passthrough": passthrough, "specsub": specsub, "wiener": wiener, "mmse_lsa": mmse_lsa}


def front_mic(noisy2: np.ndarray, spacing_m: float) -> np.ndarray:
    return mmse_lsa(noisy2[0])


def diff_cardioid(noisy2: np.ndarray, spacing_m: float) -> np.ndarray:
    return mmse_lsa(diff_cardioid_beam(noisy2, spacing_m))


def front_mic_raw(noisy2: np.ndarray, spacing_m: float) -> np.ndarray:
    return np.asarray(noisy2[0], np.float32)


def diff_cardioid_raw(noisy2: np.ndarray, spacing_m: float) -> np.ndarray:
    return diff_cardioid_beam(noisy2, spacing_m)


TWO_CHANNEL = {"front_mic_raw": front_mic_raw, "diff_cardioid_raw": diff_cardioid_raw,
               "front_mic": front_mic, "diff_cardioid": diff_cardioid}
