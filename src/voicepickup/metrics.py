"""Speech enhancement metrics.

* PESQ (ITU-T P.862.2 wide-band) and STOI: perceptual quality / intelligibility.
* SI-SDR: scale-invariant signal-to-distortion ratio (Le Roux et al., 2019).
* SNR and segmental SNR (frame SNR clamped to [-10, 35] dB).
* CSIG / CBAK / COVL: the Hu & Loizou (2008) composite predictors of signal distortion,
  background intrusiveness and overall quality, built from PESQ, LLR, WSS and segSNR.
  Ported from Loizou's ``composite.m`` (as used in every VoiceBank-DEMAND paper), so our
  numbers are comparable with published tables.
* DNSMOS P.835 (SIG/BAK/OVRL): Microsoft's non-intrusive neural MOS predictor; needs no
  clean reference, so it also scores real recordings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import toeplitz

from .audio import SR

EPS = np.finfo(np.float64).eps


def _align(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref), len(est))
    return np.asarray(ref[:n], np.float64), np.asarray(est[:n], np.float64)


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = _align(ref, est)
    ref, est = ref - ref.mean(), est - est.mean()
    target = (est @ ref) / (ref @ ref + EPS) * ref
    return float(10 * np.log10((target @ target + EPS) / ((est - target) @ (est - target) + EPS)))


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Frames used by composite.m: ``floor(len/hop - win/hop)`` frames, MATLAB hanning window."""
    n_frames = int(len(x) / hop - win / hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, win + 1) / (win + 1)))
    return x[idx] * window


def snr_segsnr(ref: np.ndarray, est: np.ndarray, sr: int = SR) -> tuple[float, float]:
    ref, est = _align(ref, est)
    overall = 10 * np.log10(np.sum(ref**2) / (np.sum((ref - est) ** 2) + EPS) + EPS)
    win = round(30 * sr / 1000)
    c, p = _frames(ref, win, win // 4), _frames(est, win, win // 4)
    seg = 10 * np.log10(np.sum(c**2, 1) / (np.sum((c - p) ** 2, 1) + EPS) + EPS)
    return float(overall), float(np.mean(np.clip(seg, -10, 35)))


def _lpc(frames: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation (frames, order+1) and LPC polynomial [1, -a1..-ap] via Levinson-Durbin."""
    n = frames.shape[1]
    r = np.stack([np.sum(frames[:, : n - k] * frames[:, k:], 1) for k in range(order + 1)], 1)
    a = np.zeros((len(frames), order))
    e = r[:, 0].copy()
    for i in range(order):
        past = a[:, :i].copy()
        k = (r[:, i + 1] - np.sum(past * r[:, i:0:-1], 1)) / (e + EPS)
        a[:, i] = k
        a[:, :i] = past - k[:, None] * past[:, ::-1]
        e = (1 - k**2) * e
    return r, np.concatenate([np.ones((len(frames), 1)), -a], 1)


def llr(ref: np.ndarray, est: np.ndarray, sr: int = SR) -> np.ndarray:
    ref, est = _align(ref, est)
    win = round(30 * sr / 1000)
    c, p = _frames(ref, win, win // 4), _frames(est, win, win // 4)
    order = 10 if sr < 10_000 else 16
    r_c, a_c = _lpc(c, order)
    _, a_p = _lpc(p, order)
    out = np.zeros(len(c))
    for t in range(len(c)):
        if r_c[t, 0] < 1e-12:  # digitally silent reference frame: LLR is undefined, count as no distortion
            continue
        rc = toeplitz(r_c[t])
        out[t] = np.log((a_p[t] @ rc @ a_p[t]) / (a_c[t] @ rc @ a_c[t]))
    return np.maximum(out, 0.0)  # >= 0 in exact arithmetic (a_c minimises the clean residual)


_CENT = np.array([50.0, 120, 190, 260, 330, 400, 470, 540, 617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30,
                  1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97, 2978.04, 3276.17, 3597.63])
_BW = np.array([70.0, 70, 70, 70, 70, 70, 70, 77.3724, 86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423, 153.823,
                168.154, 183.457, 199.776, 217.153, 235.631, 255.255, 276.072, 298.126, 321.465, 346.136])


def wss(ref: np.ndarray, est: np.ndarray, sr: int = SR) -> np.ndarray:
    """Weighted spectral slope distance (Klatt, 1982), per frame."""
    ref, est = _align(ref, est)
    win = round(30 * sr / 1000)
    n_fft = 2 ** int(np.ceil(np.log2(2 * win)))
    half = n_fft // 2
    kmax, klocmax = 20.0, 1.0
    f0 = _CENT / (sr / 2) * half
    bw = _BW / (sr / 2) * half
    j = np.arange(half)
    filt = np.exp(-11 * ((j[None, :] - np.floor(f0)[:, None]) / bw[:, None]) ** 2
                  + (np.log(_BW[0]) - np.log(_BW))[:, None])
    filt *= filt > np.exp(-30 / (2 * 2.303))

    def band_db(x):
        spec = np.abs(np.fft.fft(_frames(x, win, win // 4), n_fft, axis=1)[:, :half]) ** 2
        return 10 * np.log10(np.maximum(spec @ filt.T, 1e-10))

    ec, ep = band_db(ref), band_db(est)
    sc, sp = np.diff(ec, axis=1), np.diff(ep, axis=1)
    n_crit = ec.shape[1]

    def loc_peaks(energy, slope):
        peaks = np.empty_like(slope)
        for t in range(len(slope)):
            for i in range(n_crit - 1):
                n = i
                if slope[t, i] > 0:
                    while n < n_crit - 1 and slope[t, n] > 0:
                        n += 1
                    peaks[t, i] = energy[t, n - 1]
                else:
                    while n >= 0 and slope[t, n] <= 0:
                        n -= 1
                    peaks[t, i] = energy[t, n + 1]
        return peaks

    def weight(energy, peaks):
        wmax = kmax / (kmax + energy.max(1, keepdims=True) - energy[:, :-1])
        wloc = klocmax / (klocmax + peaks - energy[:, :-1])
        return wmax * wloc

    w = (weight(ec, loc_peaks(ec, sc)) + weight(ep, loc_peaks(ep, sp))) / 2
    return np.sum(w * (sc - sp) ** 2, 1) / np.sum(w, 1)


def pesq_wb(ref: np.ndarray, est: np.ndarray, sr: int = SR) -> float:
    from pesq import PesqError, pesq

    ref, est = _align(ref, est)
    try:
        return float(pesq(sr, ref, est, "wb", on_error=PesqError.RAISE_EXCEPTION))
    except (PesqError, ValueError):  # e.g. no utterance detected in a silent clip
        return float("nan")


def stoi(ref: np.ndarray, est: np.ndarray, sr: int = SR, extended: bool = False) -> float:
    from pystoi import stoi as _stoi

    ref, est = _align(ref, est)
    return float(_stoi(ref, est, sr, extended=extended))


def composite(ref: np.ndarray, est: np.ndarray, sr: int = SR, pesq_score: float | None = None) -> dict:
    alpha = 0.95
    pq = pesq_wb(ref, est, sr) if pesq_score is None else pesq_score
    w = np.sort(wss(ref, est, sr))
    w_mean = float(np.mean(w[: round(len(w) * alpha)]))
    ll = np.sort(llr(ref, est, sr))
    ll_mean = float(np.mean(ll[: round(len(ll) * alpha)]))
    _, seg = snr_segsnr(ref, est, sr)
    clamp = lambda v: float(min(max(v, 1.0), 5.0))
    return {
        "csig": clamp(3.093 - 1.029 * ll_mean + 0.603 * pq - 0.009 * w_mean),
        "cbak": clamp(1.634 + 0.478 * pq - 0.007 * w_mean + 0.063 * seg),
        "covl": clamp(1.594 + 0.805 * pq - 0.512 * ll_mean - 0.007 * w_mean),
    }


# ---------------------------------------------------------------- DNSMOS P.835

DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"


def dnsmos_model_path() -> Path:
    from .data.prepare import data_root

    return Path(os.environ.get("VP_DNSMOS", data_root() / "models" / "sig_bak_ovr.onnx"))


@lru_cache(maxsize=1)
def _dnsmos_session():
    import onnxruntime as ort

    path = dnsmos_model_path()
    if not path.exists():
        return None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # we parallelise over files instead
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def dnsmos(est: np.ndarray, sr: int = SR) -> dict:
    """DNSMOS P.835 (non-personalised), following Microsoft's ``dnsmos_local.py``."""
    sess = _dnsmos_session()
    if sess is None:
        return {}
    assert sr == 16_000
    seg_len = int(9.01 * sr)
    x = np.asarray(est, np.float32)
    while len(x) < seg_len:
        x = np.concatenate([x, x])
    n_hops = int(np.floor(len(x) / sr) - 9.01) + 1
    name = sess.get_inputs()[0].name
    raw = []
    for h in range(n_hops):
        seg = x[h * sr : h * sr + seg_len]
        if len(seg) == seg_len:
            raw.append(sess.run(None, {name: seg[None, :]})[0][0])
    sig, bak, ovr = np.mean(raw, 0)
    return {
        "dnsmos_sig": float(np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig)),
        "dnsmos_bak": float(np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak)),
        "dnsmos_ovrl": float(np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr)),
    }


def all_metrics(ref: np.ndarray, est: np.ndarray, sr: int = SR, with_dnsmos: bool = True) -> dict:
    ref, est = _align(ref, est)
    pq = pesq_wb(ref, est, sr)
    snr, seg = snr_segsnr(ref, est, sr)
    out = {"pesq": pq, "stoi": stoi(ref, est, sr), "estoi": stoi(ref, est, sr, extended=True),
           "si_sdr": si_sdr(ref, est), "snr": snr, "ssnr": seg}
    if not np.isnan(pq):
        out.update(composite(ref, est, sr, pq))
    if with_dnsmos:
        out.update(dnsmos(est, sr))
    return out
