# voicepickup: working agreement for AI coding assistants

These are the rules I give the AI assistant (Claude Code) when it works in this repo.
I set the design and the acceptance checks. The assistant writes code within these rules,
and nothing counts as done until the checks below pass.

## Commands (run in WSL2 / Linux; venv at ~/venvs/voicepickup)
- Environment: `bash scripts/setup_env.sh` (pins a CUDA 12.8 PyTorch build; sm_120 GPUs need it)
- Data: `bash scripts/download_data.sh`, then `vp prepare` and `vp make-testsets`
- Test: `python -m pytest`
- Lint: `ruff check src tests`
- Train: `vp-train --run-dir runs/<name>` (or `sbatch slurm/train.sbatch`)
- Evaluate: `vp eval --set <vbd_official|grid|headset2mic> --methods <baseline|model:path.pt>`
- Tables from saved results: `vp report --set <set> --methods ... --metrics pesq si_sdr`

## Definition of done (every change)
1. Tests are added or updated for the new behaviour, including at least one edge case.
2. `pytest` and `ruff check src tests` both pass.
3. Every bug that gets fixed gets a regression test that fails without the fix.
4. Any change to data, model or metrics is re-run on real data, and the numbers are reported.

## Evaluation integrity (never break these)
- **No leakage.** Test speakers (p232, p257) and test noise environments (DLIVING, OOFFICE,
  PCAFETER, SPSQUARE, TBUS) never appear in training or validation. Validation uses
  p226/p287 and DEMAND mics 13-16.
- **Never tune on a test set.** Choose hyperparameters and checkpoints on validation only.
- **Metrics are validated.** Scoring the unprocessed VoiceBank-DEMAND test set must reproduce
  the published row (PESQ 1.97, CSIG 3.35, CBAK 2.44, COVL 2.63). If a metric change moves
  that row, the change is wrong.
- **Same framing for every method.** Baselines and the model all use 20 ms windows with a
  10 ms hop, so the comparison is fair.
- Test sets are seeded and fixed. Don't regenerate them to change a number.

## Model constraints (headset use case)
- **Causal only.** No look-ahead: past-padded convs, a unidirectional GRU.
  `test_causal_no_lookahead` must keep passing.
- Latency budget: 20 ms algorithmic. Parameter budget: about 2M or fewer.
- Training must stay resumable and preemption-safe: checkpoint on SIGUSR1/SIGTERM and exit
  with code 99. Checkpoints are written atomically (temp file, then rename).

## Reporting
- The README states only what was actually run. Anything written but not yet executed
  (for example the Slurm scripts) is labelled as such under "Status and honest caveats".
- Report negative results too (for example, the beamformer failing with wind).

## Repo hygiene
- No data, audio, checkpoints or W&B logs in git (see `.gitignore`).
  Results CSVs and summaries are committed.
- Shell scripts use LF line endings (enforced by `.gitattributes`).
- No secrets in code. W&B credentials come from `wandb login`, never from files in the repo.
