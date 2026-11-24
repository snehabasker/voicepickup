import numpy as np
import pytest

from voicepickup.audio import SR, db, read, rms
from voicepickup.data.corpus import Corpus
from voicepickup.data.testsets import make_grid, make_headset, parse_vbd_log, read_manifest

from .test_data import speechlike


@pytest.fixture(scope="module")
def corpora():
    rng = np.random.default_rng(0)
    speech = Corpus.from_arrays([speechlike(SR, rng) for _ in range(3)], ["p232_001", "p232_002", "p257_001"],
                                ["p232", "p232", "p257"])
    noise = Corpus.from_arrays([0.05 * rng.standard_normal(3 * SR) for _ in range(2)], ["TBUS_ch01", "OOFFICE_ch01"],
                               ["TBUS", "OOFFICE"])
    return speech, noise


def test_grid(tmp_path, corpora):
    make_grid(*corpora, tmp_path, n_utts=2, snrs=(-5.0, 10.0))
    rows = read_manifest(tmp_path)
    assert len(rows) == 2 * 3 * 2  # utts x (2 envs + wind) x snrs
    assert {r["noise"] for r in rows} == {"TBUS", "OOFFICE", "WIND"}
    for r in rows[:6]:
        noisy, clean = read(tmp_path / r["noisy"]), read(tmp_path / r["clean"])
        assert db(rms(clean) / rms(noisy - clean)) == pytest.approx(r["snr_db"], abs=0.2)  # 16-bit rounding


def test_headset(tmp_path, corpora):
    make_headset(*corpora, tmp_path, n_utts=3, p_wind=0.5)
    rows = read_manifest(tmp_path)
    assert len(rows) == 3
    assert read(tmp_path / rows[0]["noisy"], mono=False).shape[0] == 2
    assert "rt60_s" in rows[0]["extra"]


def test_parse_vbd_log(tmp_path):
    (tmp_path / "log.txt").write_text("p232_001 living 17.5\np232_002 bus 2.5\n")
    assert parse_vbd_log(tmp_path / "log.txt") == {"p232_001": ("living", 17.5), "p232_002": ("bus", 2.5)}
