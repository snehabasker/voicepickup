# voicepickup

**Real-time speech enhancement for headset calls.** A causal neural noise suppressor with
20 ms latency and 1.2M parameters. It is trained with noise mixed in on the fly
(including synthetic wind) and compared against classical DSP baselines on the standard
VoiceBank-DEMAND benchmark. Two harder test sets cover conditions where headset calls
actually struggle: negative SNRs, wind, and a simulated two-microphone earcup.

| | |
|---|---|
| Task | single-channel speech enhancement (noise suppression), 16 kHz |
| Model | causal CRN with complex ratio mask, **1.24M params**, **20 ms algorithmic latency**, no look-ahead |
| Data | VoiceBank-DEMAND (speech) + DEMAND (17 noise environments) + synthetic wind + simulated 2-mic headset |
| Metrics | PESQ-WB, STOI/ESTOI, SI-SDR, SNR/segSNR, CSIG/CBAK/COVL, DNSMOS P.835 |
| Tracking | Weights & Biases: metrics, audio, spectrograms, sweeps |
| Compute | Slurm job scripts (training with requeue, sweep job arrays sharing one GPU); RTX 5050 laptop GPU, CUDA 12.8 |

## Results

### VoiceBank-DEMAND test set (824 utterances, unseen speakers and noises)

| method | PESQ | CSIG | CBAK | COVL | SSNR | STOI | SI-SDR |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy input | 1.97 | 3.35 | 2.44 | 2.63 | 1.69 | 0.92 | 8.45 |
| spectral subtraction | 2.20 | 3.33 | 2.70 | 2.71 | 5.19 | 0.91 | 11.79 |
| Wiener (decision-directed) | 2.27 | 3.03 | 2.79 | 2.58 | 6.95 | 0.90 | 13.97 |
| MMSE-LSA | 2.35 | 3.32 | 2.86 | 2.78 | 6.66 | 0.90 | 13.83 |
| **causal CRN (this repo)** | *training in progress, see below* | | | | | | |

PESQ by SNR (noisy → MMSE-LSA): 2.5 dB 1.42 → 1.78 · 7.5 dB 1.76 → 2.19 · 12.5 dB 2.10 → 2.52 · 17.5 dB 2.60 → 2.91.
Babble-like noise is the hardest case for classical DSP: on *cafe*, MMSE-LSA only gets 1.49 → 1.85.

The *noisy* row matches the values published for this test set exactly (PESQ 1.97,
CSIG 3.35, CBAK 2.44, COVL 2.63, SSNR 1.68, STOI 0.92). That validates the metric
implementation, so the numbers here can be compared directly with papers. For reference,
two published small causal models score PESQ 2.29 (RNNoise) and 2.87 (GTCRN) on this test set.

### Hard conditions: SNR −5…15 dB × 5 held-out noises + wind (3,600 mixtures)

Test set built (`vp make-testsets`). Scoring is pending and will be added together with the model results.

### Simulated two-mic headset (200 scenes, see [headset.py](src/voicepickup/data/headset.py))

| method | PESQ, acoustic noise only | PESQ, wind present | SI-SDR, acoustic | SI-SDR, wind |
|---|---:|---:|---:|---:|
| front mic, unprocessed | 2.09 | 1.60 | 5.0 | 5.7 |
| differential beam (mouth-steered) | 2.49 | **1.10** | 4.6 | **−11.5** |
| front mic + MMSE-LSA | 2.39 | 1.91 | 10.6 | 9.1 |
| differential beam + MMSE-LSA | **2.67** | 1.18 | 4.6 | −6.4 |

The beamformer gains most at low SNR (PESQ 1.44 → 1.79 in the −5…0 dB bin) and does worst
with wind. Wind is uncorrelated between the two ports, so delay-and-subtract plus
low-frequency EQ amplifies it instead of cancelling it.

## Why these choices (headset context)

- **Causal, 20 ms latency, ~1M parameters.** Call audio must stay low-latency end to end,
  and the processor in an earbud or headset has a tight compute and memory budget. Every
  conv in the model has a past-only time kernel and the GRU is unidirectional.
  [`test_causal_no_lookahead`](tests/test_model.py) changes the future input and asserts
  that the past output is unchanged.
- **Wind noise.** VoiceBank-DEMAND contains no wind. Outdoors it is the dominant headset
  noise, and stationary-noise DSP handles it badly. [wind.py](src/voicepickup/data/wind.py)
  models three properties of wind noise:
  - most energy below ~500 Hz,
  - log-normal gusts with lulls,
  - near-zero coherence between mics 1–2 cm apart, because wind is turbulence at the port,
    not sound travelling through the air.
  The tests check all three.
- **Two-mic earcup simulation.** The simulation uses image-source rooms with RT60
  0.2–0.8 s, an endfire mic pair 1.2–2.0 cm apart, and the mouth ~15 cm away (near field).
  It is used to test the classic first-order differential (delay-and-subtract) beamformer
  and to show where it breaks. It helps on acoustic noise at low SNR but turns wind into
  a disaster, because uncorrelated noise gets amplified by its low-frequency EQ. Real
  products need a wind detector that falls back to one mic.
- **Held-out everything.** Test speakers (p232, p257) and test noise environments never
  appear in training or validation. Validation uses separate speakers (p226, p287) and
  separate DEMAND mics.

## Layout

```
src/voicepickup/
  data/        corpus (memory-mapped int16), prepare (splits), mixer (on-the-fly mixing),
               wind, headset (pyroomacoustics), testsets (fixed, seeded, WAV + manifest)
  baselines.py spectral subtraction, Wiener (decision-directed), MMSE-LSA, SPP noise tracker,
               differential endfire beamformer
  metrics.py   PESQ, STOI, SI-SDR, segSNR, composite (port of Loizou's composite.m), DNSMOS
  model.py     causal CRN + compressed spectral loss (Braun & Tashev, 2021)
  train.py     training loop: W&B logging, resumable, SIGUSR1 -> checkpoint -> requeue
  evaluate.py  parallel scoring + breakdowns by noise and SNR
slurm/         install script (single-node WSL2 cluster, GPU as GRES + shards), train/sweep jobs
configs/       base.yaml, sweep.yaml (W&B Bayesian sweep + Hyperband)
docs/slurm.md  how the Slurm setup, preemption and sweeps work
```

## Reproduce

```bash
bash scripts/setup_env.sh           # venv, CUDA 12.8 PyTorch (needed for sm_120), audio libs
bash scripts/download_data.sh       # ~8 GB into ~/data/voicepickup (resumable)
vp prepare                          # 16 kHz memory-mapped corpora + splits
vp make-testsets                    # vbd_official, grid, headset2mic
vp-train --run-dir runs/crn_base    # or: sbatch slurm/train.sbatch
vp eval --set vbd_official --methods passthrough mmse_lsa model:runs/crn_base/best.pt
pytest                              # 30 tests, including causality and metric checks
```

## Status and honest caveats

- **Done:** data pipeline, the three test sets, metrics (validated against the published
  noisy row), classical and two-mic baselines, causal model, training loop, and 30 tests.
- **Model training is paused at step 8.5k of 60k.** Validation PESQ was already 2.38
  (+9.1 dB SI-SDR improvement) at step 8k, on held-out speakers with noise mixed on the
  fly. Test-set numbers will be added when the run finishes: `vp-train --run-dir runs/crn_base`
  resumes from the checkpoint.
- **W&B:** runs are logged in offline mode so far and will be synced to a public project.
- **Slurm:** the install script and job scripts ([slurm/](slurm/), [docs/slurm.md](docs/slurm.md))
  are written and syntax-checked, but have **not been run yet**. Training so far was
  launched directly.
- The two-mic set is simulated (image-source rooms, synthetic wind), not recorded on a
  real headset.

## How this was built

I built this with Claude Code (an AI coding assistant) as a pair programmer. My part was
the engineering decisions:
- the problem framing (headset calls: causal, low-latency, wind),
- the data splits and the rules against leakage,
- which baselines to compare against,
- the acceptance checks.

The assistant wrote most of the implementation inside those rules, which are written down
in [CLAUDE.md](CLAUDE.md). Commits it co-wrote are marked with a `Co-Authored-By` trailer.

Nothing was accepted on trust. Every result has to pass a check:
- **Metric validation:** the composite metrics are a port of a MATLAB reference. They were
  accepted only after reproducing the published scores for the unprocessed test set exactly.
- **Property tests** instead of "it runs" tests. Examples: changing future input must not
  change past output (causality), the beamformer must null a rear source and pass the front
  one in phase, and wind must be low-frequency, gusty and incoherent between mics.

Bugs caught in review, each now covered by a test or a guard:

| Bug | How it was caught | Effect if missed |
|---|---|---|
| Beamformer EQ corrected magnitude only and left a 90° phase shift | derived the on-axis response before running anything | SI-SDR destroyed for every two-mic result |
| LLR became −36 on digitally silent frames | identity check: LLR(x, x) must be 0 | composite scores biased on clips with silence |
| `.gitignore` rule `data/` also excluded the `src/voicepickup/data/` package | read the staged file list before the first push | published repo would not import |
| Windows CRLF line endings broke `set -o pipefail` in shell scripts | the script failed on its first run | scripts unusable on Linux/WSL |
| A test fixture gave PESQ ≈ 1.0 even at 15 dB SNR | sanity-checked the absolute values | synthetic signals don't work for testing perceptual metrics; those tests check ordering only now |

## References

- Valentini-Botinhao et al., *Investigating RNN-based speech enhancement methods for noise-robust TTS*, SSW 2016 (VoiceBank-DEMAND).
- Thiemann, Ito, Vincent, *DEMAND: a collection of multi-channel recordings of acoustic noise in diverse environments*, 2013.
- Braun & Tashev, *A consolidated view of loss functions for supervised deep learning-based speech enhancement*, TSP 2021.
- Gerkmann & Hendriks, *Unbiased MMSE-based noise power estimation with low complexity and low tracking delay*, IEEE TASLP 2012.
- Ephraim & Malah, *Speech enhancement using a MMSE log-spectral amplitude estimator*, IEEE TASSP 1985.
- Hu & Loizou, *Evaluation of objective quality measures for speech enhancement*, IEEE TASLP 2008.
- Nelke & Vary, *Measurement, analysis and simulation of wind noise signals for mobile communication devices*, IWAENC 2014.
- Reddy et al., *DNSMOS P.835*, ICASSP 2022.

## License

MIT for the code. The datasets keep their own licenses (VoiceBank-DEMAND CC BY 4.0,
DEMAND CC BY-SA 3.0, DNSMOS model CC BY 4.0) and are not redistributed here.
