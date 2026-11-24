"""Two-microphone headset simulation with pyroomacoustics (image-source method).

Geometry (per clip, randomised):

* A head somewhere in a random shoebox room (RT60 0.2-0.8 s), facing a random way.
* Two mics on the outside of the right earcup, 1.2-2.0 cm apart, on the line that points
  from the earcup towards the mouth ("endfire" towards the talker), as on many
  boomless headsets. The front mic (index 0) is the one nearer the mouth.
* The mouth is ~15 cm from the mics. That is near-field: the mouth-to-mic distance
  differs by ~10% between the two mics, so direct-path speech is ~1 dB louder at the
  front mic, while distant noise reaches both at the same level. That
  level difference plus the time difference is what dual-mic voice pickup uses.
* 1-3 noise sources at least 1 m away, plus (optionally) wind, which is added at the mic
  ports with independent fine structure per mic (see ``wind.py``) instead of being
  propagated through the room.

The training/evaluation target is the speech image at the front mic: what a perfect
noise remover would output. It includes the (weak, at this distance) room reverb.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pyroomacoustics as pra

from ..audio import SR, noise_gain_for_snr, normalize_pair
from .wind import wind_noise

MAX_IMAGE_ORDER = 12  # caps simulation time; later reflections are weak at these distances


@dataclass
class HeadsetMeta:
    room_dims: list[float]
    rt60_s: float
    mic_spacing_m: float
    mouth_to_front_mic_m: float
    mouth_to_rear_mic_m: float
    n_noise_sources: int
    wind: bool
    snr_db: float


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _random_point(dims: np.ndarray, rng: np.random.Generator, margin: float = 0.3) -> np.ndarray:
    return np.array([rng.uniform(margin, d - margin) for d in dims])


def simulate_headset(
    speech: np.ndarray,
    noises: list[np.ndarray],
    snr_db: float,
    rng: np.random.Generator,
    wind: bool = False,
    level_dbfs: float | None = -25.0,
    sr: int = SR,
) -> dict:
    """Return {"noisy": (2, n), "target": (n,), "meta": HeadsetMeta}.

    ``noises`` are dry noise signals (each at least len(speech) long) to place as point
    sources; pass an empty list for a wind-only scene (then ``wind`` must be True).
    """
    if not noises and not wind:
        raise ValueError("need at least one noise source or wind")
    n = len(speech)
    dims = np.array([rng.uniform(3.0, 10.0), rng.uniform(3.0, 8.0), rng.uniform(2.5, 4.0)])
    rt60 = float(rng.uniform(0.2, 0.8))
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_abs), max_order=min(max_order, MAX_IMAGE_ORDER))

    head = np.array([rng.uniform(1.0, dims[0] - 1.0), rng.uniform(1.0, dims[1] - 1.0), rng.uniform(1.1, 1.8)])
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    mouth = head + 0.09 * fwd - 0.07 * up
    cup = head + 0.10 * right  # outer surface of the right earcup
    axis = _unit(mouth - cup)
    spacing = float(rng.uniform(0.012, 0.020))
    mics = np.stack([cup + axis * spacing / 2, cup - axis * spacing / 2], axis=1)  # (3, 2), front first

    room.add_source(mouth, signal=speech)
    for noise in noises:
        while True:
            pos = _random_point(dims, rng)
            if np.linalg.norm(pos - head) >= 1.0:
                break
        room.add_source(pos, signal=noise[:n])
    room.add_microphone_array(mics)
    premix = room.simulate(return_premix=True)  # (sources, mics, samples incl. RIR tail)

    speech_img = premix[0][:, :n]
    noise_img = premix[1:, :, :n].sum(axis=0) if noises else np.zeros((2, n))
    if wind:
        w = wind_noise(n, rng, sr, n_mics=2).astype(np.float64)
        if noises:  # wind level relative to the acoustic noise at the front mic
            w *= noise_gain_for_snr(noise_img[0], w[0], rng.uniform(-5.0, 5.0))
        noise_img = noise_img + w

    noisy = speech_img + noise_img * noise_gain_for_snr(speech_img[0], noise_img[0], snr_db)
    noisy, target = normalize_pair(noisy, speech_img[0], level_dbfs, ref=noisy[0])
    meta = HeadsetMeta(
        room_dims=[round(float(d), 3) for d in dims], rt60_s=round(rt60, 3), mic_spacing_m=round(spacing, 4),
        mouth_to_front_mic_m=round(float(np.linalg.norm(mouth - mics[:, 0])), 4),
        mouth_to_rear_mic_m=round(float(np.linalg.norm(mouth - mics[:, 1])), 4),
        n_noise_sources=len(noises), wind=wind, snr_db=float(snr_db),
    )
    return {"noisy": noisy.astype(np.float32), "target": target.astype(np.float32), "meta": meta}


def meta_dict(meta: HeadsetMeta) -> dict:
    return asdict(meta)
