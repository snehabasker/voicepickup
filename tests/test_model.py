import numpy as np
import pytest
import torch

from voicepickup import baselines as bl
from voicepickup.model import CRN, braun_loss, istft, n_params, stft


def test_torch_stft_matches_numpy_and_reconstructs():
    x = np.random.default_rng(0).standard_normal(16_000).astype(np.float32)
    s_np = bl.stft(x)
    s_t = stft(torch.from_numpy(x)[None])[0].numpy()
    assert s_t.shape == s_np.shape and np.max(np.abs(s_t - s_np)) < 1e-3
    y = istft(stft(torch.from_numpy(x)[None]), len(x))[0].numpy()
    assert np.max(np.abs(y - x)) < 1e-5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return CRN().eval()


def test_shapes_and_size(model):
    x = torch.randn(3, 16_000)
    assert model(x).shape == x.shape
    assert 0.5e6 < n_params(model) < 2.5e6  # small enough for an embedded DSP/NPU budget


def test_causal_no_lookahead(model):
    """Changing the future must not change the past: output frame t depends only on input <= t."""
    torch.manual_seed(1)
    x = torch.randn(1, 16_000)
    y = model(x)
    x2 = x.clone()
    cut = 8_000
    x2[:, cut:] = torch.randn(1, 16_000 - cut)
    y2 = model(x2)
    # samples before cut - WIN are produced only by frames that end before the perturbation
    safe = cut - 320
    assert torch.allclose(y[:, :safe], y2[:, :safe], atol=1e-5)
    assert not torch.allclose(y[:, cut:], y2[:, cut:])


def test_loss_zero_at_target_and_trains(model):
    torch.manual_seed(2)
    ref = torch.randn(2, 50, 161, dtype=torch.complex64)
    assert braun_loss(ref, ref).item() == 0.0
    m = CRN().train()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    clean = torch.randn(4, 8_000) * 0.1
    noisy = clean + 0.1 * torch.randn_like(clean)
    target = m.compress(stft(clean))
    losses = []
    for _ in range(30):
        est, _ = m.forward_spec(m.compress(stft(noisy)))
        loss = braun_loss(est, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.7 * losses[0]
