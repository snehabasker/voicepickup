import numpy as np
import pytest

from voicepickup import baselines as bl
from voicepickup.audio import SR
from voicepickup.data.mixer import mix
from voicepickup.metrics import all_metrics, composite, llr, si_sdr, snr_segsnr, wss

from .test_data import speechlike


@pytest.fixture(scope="module")
def pair():
    rng = np.random.default_rng(0)
    clean = np.concatenate([np.zeros(SR // 4, np.float32), speechlike(2 * SR, rng)])
    noisy, clean = mix(clean, rng.standard_normal(len(clean)).astype(np.float32), 0.0, -25.0)
    return clean, noisy


def test_stft_perfect_reconstruction():
    x = np.random.default_rng(1).standard_normal(SR).astype(np.float32)
    assert np.max(np.abs(bl.istft(bl.stft(x), len(x)) - x)) < 1e-5


def test_identity_scores(pair):
    clean, _ = pair
    assert si_sdr(clean, clean) > 100
    voiced = clean[SR // 4 :]  # silent frames score the -10 dB floor, as in composite.m
    assert snr_segsnr(voiced, voiced)[1] == pytest.approx(35.0)
    assert np.allclose(wss(clean, clean), 0) and np.allclose(llr(clean, clean), 0, atol=1e-9)
    m = all_metrics(clean, clean, with_dnsmos=False)
    assert m["pesq"] > 4.5 and m["stoi"] > 0.99 and m["csig"] == 5.0


def test_si_sdr_is_scale_invariant(pair):
    clean, noisy = pair
    assert si_sdr(clean, 3 * noisy) == pytest.approx(si_sdr(clean, noisy), abs=1e-6)
    assert si_sdr(clean, noisy) == pytest.approx(0.0, abs=0.3)


def test_composite_components_degrade_with_noise(pair):
    # Absolute composite values are only meaningful on real speech (validated against the
    # published "Noisy" row of VoiceBank-DEMAND, see README); here we check monotonicity.
    clean, _ = pair
    noise = np.random.default_rng(2).standard_normal(len(clean))
    better, c1 = mix(clean, noise, 15.0, -25.0)
    worse, c2 = mix(clean, noise, 5.0, -25.0)
    assert np.mean(wss(c1, better)) < np.mean(wss(c2, worse))
    assert np.mean(llr(c1, better)) < np.mean(llr(c2, worse))
    assert composite(c1, better)["cbak"] > composite(c2, worse)["cbak"]


@pytest.mark.parametrize("name", ["specsub", "wiener", "mmse_lsa"])
def test_single_channel_baselines_improve(pair, name):
    clean, noisy = pair
    enhanced = bl.SINGLE_CHANNEL[name](noisy)
    assert enhanced.shape == noisy.shape
    assert snr_segsnr(clean, enhanced)[1] > snr_segsnr(clean, noisy)[1] + 3


def test_differential_beam_nulls_rear_and_passes_front():
    rng = np.random.default_rng(3)
    s, d = rng.standard_normal(SR), 0.016
    shift = d / bl.SPEED_OF_SOUND * SR  # fractional delay in samples

    def delayed(x, k):
        f = np.fft.rfftfreq(len(x))
        return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * k), len(x))

    front_src = np.stack([s, delayed(s, shift)])  # hits the front mic first
    rear_src = np.stack([delayed(s, shift), s])
    out_front = bl.diff_cardioid_beam(front_src, d)
    out_rear = bl.diff_cardioid_beam(rear_src, d)
    # on-axis speech comes out aligned with the front mic (no 90 degree phase error)
    band = lambda x: np.fft.irfft(np.fft.rfft(x) * ((np.fft.rfftfreq(len(x), 1 / SR) > 500)
                                                     & (np.fft.rfftfreq(len(x), 1 / SR) < 7000)), len(x))
    assert si_sdr(band(s)[400:-400], band(out_front)[400:-400]) > 20
    assert np.std(out_rear[400:-400]) < 0.05 * np.std(out_front[400:-400])
