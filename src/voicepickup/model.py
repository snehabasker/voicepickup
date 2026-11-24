"""Causal convolutional-recurrent network (CRN) for real-time speech enhancement.

Signal path (16 kHz, 20 ms sqrt-Hann window, 10 ms hop, same as the classical baselines):

    noisy -> STFT -> power-law compress |X|^0.3 (keep phase) -> [real, imag] (2 x 161 bins)
          -> conv encoder over frequency (no look-ahead in time)
          -> 2-layer GRU over time (the only long-term memory; stateful when streaming)
          -> conv decoder with skip connections
          -> complex ratio mask M (bounded by tanh) applied to the compressed spectrum
          -> decompress -> inverse STFT -> enhanced

Causality: every conv has time kernel 2 with one frame of *past* padding, the GRU is
unidirectional and the mask is applied frame by frame, so output frame t depends only on
input frames <= t. Algorithmic latency is one window (20 ms). ``tests/test_model.py``
checks causality by perturbing the future and asserting the past output is unchanged.

The loss is the compressed complex/magnitude spectral loss of Braun & Tashev (2021,
"A consolidated view of loss functions for supervised deep learning-based speech
enhancement"), which they found best across PESQ/STOI/SI-SDR for this kind of model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

WIN, HOP = 320, 160
N_FREQ = WIN // 2 + 1


def _window(device=None) -> torch.Tensor:
    return torch.hann_window(WIN, periodic=True, device=device).sqrt()


def stft(x: torch.Tensor) -> torch.Tensor:
    """(B, n) -> complex (B, T, F); identical framing/padding to ``baselines.stft``."""
    x = F.pad(x, (HOP, WIN))
    frames = x.unfold(-1, WIN, HOP) * _window(x.device)
    return torch.fft.rfft(frames, dim=-1)


def istft(spec: torch.Tensor, length: int) -> torch.Tensor:
    frames = torch.fft.irfft(spec, WIN, dim=-1) * _window(spec.device)  # (B, T, WIN)
    b, t, _ = frames.shape
    out = F.fold(frames.transpose(1, 2), output_size=(1, HOP * (t - 1) + WIN), kernel_size=(1, WIN), stride=(1, HOP))
    return out.reshape(b, -1)[:, HOP : HOP + length]


@dataclass
class CRNConfig:
    channels: tuple[int, ...] = (16, 32, 48, 64)
    rnn_hidden: int = 256
    rnn_layers: int = 2
    compress: float = 0.3


class CausalConv(nn.Module):
    """Conv2d over (time, freq) with kernel (2, 3), stride 2 in freq, past-only time padding."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
        self.norm = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x):  # (B, C, T, F)
        return self.act(self.norm(self.conv(F.pad(x, (0, 0, 1, 0)))))


class CausalDeconv(nn.Module):
    def __init__(self, cin: int, cout: int, out_freq: int, last: bool = False):
        super().__init__()
        self.out_freq = out_freq
        self.conv = nn.ConvTranspose2d(cin, cout, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
        self.post = nn.Identity() if last else nn.Sequential(nn.BatchNorm2d(cout), nn.PReLU(cout))

    def forward(self, x):
        t = x.shape[2]
        y = self.conv(x)[:, :, :t]  # transposed conv adds one frame at the end: drop it (keeps causality)
        return self.post(y[..., : self.out_freq])


class CRN(nn.Module):
    def __init__(self, cfg: CRNConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or CRNConfig()
        chans = (2, *cfg.channels)
        freqs = [N_FREQ]
        for _ in cfg.channels:
            freqs.append((freqs[-1] + 1) // 2)
        self.freqs = freqs
        self.enc = nn.ModuleList(CausalConv(chans[i], chans[i + 1]) for i in range(len(cfg.channels)))
        flat = cfg.channels[-1] * freqs[-1]
        self.pre = nn.Linear(flat, cfg.rnn_hidden)
        self.rnn = nn.GRU(cfg.rnn_hidden, cfg.rnn_hidden, cfg.rnn_layers, batch_first=True)
        self.post = nn.Linear(cfg.rnn_hidden, flat)
        n = len(cfg.channels)
        self.dec = nn.ModuleList(
            CausalDeconv(chans[i + 1] * 2, chans[i] if i > 0 else 2, freqs[i], last=(i == 0))
            for i in reversed(range(n))
        )

    def compress(self, spec: torch.Tensor) -> torch.Tensor:
        mag = spec.abs().clamp_min(1e-8)
        return spec / mag * mag ** self.cfg.compress

    def decompress(self, spec: torch.Tensor) -> torch.Tensor:
        mag = spec.abs().clamp_min(1e-8)
        return spec / mag * mag ** (1.0 / self.cfg.compress)

    def forward_spec(self, cspec: torch.Tensor, h: torch.Tensor | None = None):
        """Compressed complex spectrum (B, T, F) -> (enhanced compressed spectrum, GRU state)."""
        x = torch.stack([cspec.real, cspec.imag], 1)  # (B, 2, T, F)
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
        b, c, t, f = x.shape
        z = self.pre(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        z, h = self.rnn(z, h)
        x = self.post(z).reshape(b, t, c, f).permute(0, 2, 1, 3)
        for layer, skip in zip(self.dec, reversed(skips)):
            x = layer(torch.cat([x, skip], 1))
        mask = torch.complex(torch.tanh(x[:, 0]), torch.tanh(x[:, 1])) * 1.5  # |M| up to ~2: allows mild boost
        return mask * cspec, h

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """(B, n) waveform -> (B, n) enhanced waveform."""
        est, _ = self.forward_spec(self.compress(stft(noisy)))
        return istft(self.decompress(est), noisy.shape[-1])


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def braun_loss(est_c: torch.Tensor, ref_c: torch.Tensor, lam: float = 0.3) -> torch.Tensor:
    """Compressed-spectrum loss: lam * complex MSE + (1 - lam) * magnitude MSE."""
    complex_term = (est_c - ref_c).abs().pow(2).mean()
    mag_term = (est_c.abs() - ref_c.abs()).pow(2).mean()
    return lam * complex_term + (1 - lam) * mag_term


def load_checkpoint(path: str, device: str = "cpu") -> CRN:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CRN(CRNConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)
