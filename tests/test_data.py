import numpy as np
import pytest
from scipy.signal import coherence, welch

from voicepickup.audio import SR, db, read, resample, rms, write
from voicepickup.data.corpus import Corpus
from voicepickup.data.mixer import WIND, DynamicMixer, MixConfig, mix
from voicepickup.data.wind import wind_noise


def tone(f, s=1.0, sr=SR, amp=0.1):
    return (amp * np.sin(2 * np.pi * f * np.arange(int(s * sr)) / sr)).astype(np.float32)


def speechlike(n, rng):
    """Amplitude-modulated harmonic signal: cheap stand-in for speech in unit tests."""
    t = np.arange(n) / SR
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 20))
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t + rng.uniform(0, 6)))
    return (0.1 * x * env).astype(np.float32)


def test_resample_keeps_tone_and_length():
    x48 = tone(1000, sr=48_000)
    x16 = resample(x48, 48_000, SR)
    assert len(x16) == SR
    f, p = welch(x16, SR, nperseg=1024)
    assert abs(f[np.argmax(p)] - 1000) < 20


def test_wav_roundtrip(tmp_path):
    x = np.stack([tone(440), tone(880)])
    write(tmp_path / "a.wav", x)
    y = read(tmp_path / "a.wav", mono=False)
    assert y.shape == x.shape and np.max(np.abs(y - x)) < 1e-3


def test_mix_hits_exact_snr_and_level():
    rng = np.random.default_rng(0)
    clean, noise = speechlike(SR, rng), rng.standard_normal(SR).astype(np.float32)
    for snr in (-5.0, 0.0, 12.5):
        noisy, c = mix(clean, noise, snr, level_dbfs=-25.0)
        assert db(rms(c) / rms(noisy - c)) == pytest.approx(snr, abs=0.01)
        assert db(rms(noisy)) == pytest.approx(-25.0, abs=0.01)


def test_mix_peak_guard_preserves_snr():
    rng = np.random.default_rng(1)
    clean, noise = speechlike(SR, rng), rng.standard_normal(SR).astype(np.float32)
    noisy, c = mix(clean, noise, 0.0, level_dbfs=-3.0)  # would clip
    assert np.max(np.abs(noisy)) <= 0.99 + 1e-6
    assert db(rms(c) / rms(noisy - c)) == pytest.approx(0.0, abs=0.01)


def test_corpus_roundtrip(tmp_path):
    arrays = [tone(300, 0.5), tone(500, 0.25)]
    c = Corpus.from_arrays(arrays, ["a", "b"], ["spk1", "spk2"])
    c.save(tmp_path / "x")
    d = Corpus.open(tmp_path / "x")
    assert len(d) == 2 and d.names == ["a", "b"] and d.group_names == ["spk1", "spk2"]
    assert np.max(np.abs(d.get(1) - arrays[1])) < 1e-4
    seg = d.segment(1, 3000, 2000)  # wraps past the end of a 4000-sample clip
    assert len(seg) == 2000 and np.allclose(seg[1000:], d.get(1)[:1000])


class TestWind:
    def test_energy_is_low_frequency(self):
        for seed in range(5):
            w = wind_noise(4 * SR, seed)
            f, p = welch(w, SR, nperseg=2048)
            assert p[f < 500].sum() / p.sum() > 0.8
            assert rms(w) == pytest.approx(1.0, rel=1e-3)

    def test_is_gusty_not_stationary(self):
        w = wind_noise(8 * SR, 3)
        frame_db = db(rms(w[: len(w) // 1600 * 1600].reshape(-1, 1600), axis=1))  # 100 ms frames
        assert np.percentile(frame_db, 95) - np.percentile(frame_db, 5) > 6.0

    def test_two_mics_are_incoherent_above_low_frequencies(self):
        rng = np.random.default_rng(4)
        w = wind_noise(4 * SR, rng, n_mics=2)
        f, coh = coherence(w[0], w[1], SR, nperseg=1024)
        assert coh[(f > 100) & (f < 2000)].mean() < 0.2
        # A diffuse acoustic field at 1.6 cm spacing would be highly coherent here (sinc(kd) ~ 0.9 at 1 kHz).

    def test_deterministic_with_seed(self):
        assert np.array_equal(wind_noise(SR, 7), wind_noise(SR, 7))


def test_dynamic_mixer_contract():
    rng = np.random.default_rng(0)
    speech = Corpus.from_arrays([speechlike(n, rng) for n in (SR // 2, 3 * SR)], ["u1", "u2"], ["p1", "p2"])
    noise = Corpus.from_arrays([0.05 * rng.standard_normal(5 * SR)], ["n1"], ["DKITCHEN"])
    m = DynamicMixer(speech, noise, MixConfig(segment_s=1.0, p_wind=0.5, p_second_noise=0.5))
    kinds = set()
    for k in range(40):
        s = m.sample(np.random.default_rng(k))
        assert s["noisy"].shape == s["clean"].shape == (SR,)
        assert -5 <= s["snr_db"] <= 15 and np.max(np.abs(s["noisy"])) <= 0.99 + 1e-6
        active = s["clean"] != 0  # short utterances are zero-padded; SNR is on the speech part
        measured = db(rms(s["clean"][: active.nonzero()[0][-1] + 1]) / rms(
            (s["noisy"] - s["clean"])[: active.nonzero()[0][-1] + 1]))
        if s["utt"] == "u2":
            assert measured == pytest.approx(s["snr_db"], abs=0.05)
        kinds.update(s["noise"].split("+"))
    assert kinds == {WIND, "DKITCHEN"}
    a, b = m.sample(np.random.default_rng(99)), m.sample(np.random.default_rng(99))
    assert np.array_equal(a["noisy"], b["noisy"])
