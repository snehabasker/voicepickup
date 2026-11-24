"""Smoke test the full loop on a tiny synthetic data root: train -> checkpoint -> resume -> eval."""

import numpy as np
import pytest
import torch

from voicepickup.audio import SR
from voicepickup.data.corpus import Corpus
from voicepickup.data.testsets import make_grid
from voicepickup.evaluate import evaluate
from voicepickup.train import load_config, train

from .test_data import speechlike


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for stem, spk in (("vbd_train_clean", "p001"), ("vbd_val_clean", "p226"), ("vbd_test_clean", "p232")):
        Corpus.from_arrays([speechlike(int(2.5 * SR), rng) for _ in range(4)], [f"{spk}_{i:03d}" for i in range(4)],
                           [spk] * 4).save(root / "prepared" / stem)
    for stem, env in (("noise_train", "DKITCHEN"), ("noise_val", "DKITCHEN"), ("noise_test", "TBUS")):
        Corpus.from_arrays([0.05 * rng.standard_normal(5 * SR)], [f"{env}_ch01"], [env]).save(root / "prepared" / stem)
    make_grid(Corpus.open(root / "prepared" / "vbd_test_clean"), Corpus.open(root / "prepared" / "noise_test"),
              root / "testsets" / "grid", n_utts=1, snrs=(0.0,))
    return root


def cfg(steps):
    return load_config("configs/base.yaml", [
        f"train.steps={steps}", "train.batch_size=2", "train.num_workers=0", "train.eval_every=3",
        "train.ckpt_every=3", "train.val_items=4", "train.val_pesq_items=2", "train.log_audio_items=1",
        "train.warmup_steps=2", "train.log_every=2", "model.rnn_hidden=32", "model.channels=[8,8,8,8]",
        "wandb.mode=disabled",
    ])


def test_train_resume_eval(root, tmp_path):
    run = tmp_path / "run"
    train(cfg(4), run, root)
    assert (run / "last.pt").exists() and (run / "best.pt").exists()
    assert torch.load(run / "last.pt", weights_only=False)["step"] == 4

    train(cfg(6), run, root)  # resumes from step 4 (stored cfg wins except wandb keys)
    ck = torch.load(run / "last.pt", weights_only=False)
    assert ck["step"] == 4  # stored config says steps=4, so nothing more to do: config is pinned on resume

    summary = evaluate(root / "testsets" / "grid", f"model:{run / 'best.pt'}", tmp_path / "res", workers=1,
                       with_dnsmos=False, label="tiny")
    assert summary["overall"]["n"] == 2  # 1 utt x (TBUS + WIND) x 1 SNR
    assert (tmp_path / "res" / "tiny.csv").exists()


def test_unknown_override_rejected():
    with pytest.raises(KeyError):
        load_config("configs/base.yaml", ["train.lrr=1"])
