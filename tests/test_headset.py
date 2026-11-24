import numpy as np
import pytest

from voicepickup.audio import SR, db, rms
from voicepickup.data.headset import simulate_headset

from .test_data import speechlike


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(0)
    speech = speechlike(2 * SR, rng)
    noises = [0.1 * rng.standard_normal(2 * SR).astype(np.float32) for _ in range(2)]
    return speech, noises


def test_shapes_snr_and_target(scene):
    speech, noises = scene
    out = simulate_headset(speech, noises, snr_db=5.0, rng=np.random.default_rng(1))
    noisy, target, meta = out["noisy"], out["target"], out["meta"]
    assert noisy.shape == (2, len(speech)) and target.shape == (len(speech),)
    assert db(rms(target) / rms(noisy[0] - target)) == pytest.approx(5.0, abs=0.05)
    assert db(rms(noisy[0])) == pytest.approx(-25.0, abs=0.05)
    assert 0.012 <= meta.mic_spacing_m <= 0.020
    assert meta.mouth_to_rear_mic_m > meta.mouth_to_front_mic_m


def test_near_field_speech_is_louder_at_front_mic(scene):
    speech, _ = scene
    diffs = []
    for seed in range(4):
        # clean speech only (noise at +80 dB SNR) so the level difference is the speech's
        out = simulate_headset(speech, [np.zeros_like(speech) + 1e-6], 80.0, np.random.default_rng(seed))
        diffs.append(db(rms(out["noisy"][0]) / rms(out["noisy"][1])))
    assert np.mean(diffs) > 0.3  # direct path alone gives ~0.9 dB; reverb dilutes it


def test_wind_only_scene_and_determinism(scene):
    speech, _ = scene
    a = simulate_headset(speech, [], 0.0, np.random.default_rng(5), wind=True)
    b = simulate_headset(speech, [], 0.0, np.random.default_rng(5), wind=True)
    assert np.array_equal(a["noisy"], b["noisy"]) and a["meta"].wind
    with pytest.raises(ValueError):
        simulate_headset(speech, [], 0.0, np.random.default_rng(5), wind=False)
