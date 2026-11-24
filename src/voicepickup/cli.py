"""Command line: ``vp prepare``, ``vp make-testsets``, ``vp info``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data.prepare import data_root, open_corpus, prepare
from .data.testsets import make_grid, make_headset, make_vbd_official


def cmd_prepare(a: argparse.Namespace) -> None:
    prepare(a.data_root, a.workers, a.force)


def cmd_make_testsets(a: argparse.Namespace) -> None:
    root = a.data_root or data_root()
    out = root / "testsets"
    speech = open_corpus("vbd_test_clean", root)
    noise = open_corpus("noise_test", root)
    if "vbd_official" in a.sets:
        make_vbd_official(speech, open_corpus("vbd_test_noisy", root), out / "vbd_official",
                          root / "raw" / "vbd" / "log_testset.txt")
    if "grid" in a.sets:
        make_grid(speech, noise, out / "grid", n_utts=a.grid_utts, seed=a.seed)
    if "headset2mic" in a.sets:
        make_headset(speech, noise, out / "headset2mic", n_utts=a.headset_utts, seed=a.seed)


def method_label(m: str) -> str:
    """``model:runs/crn_base/best.pt`` -> ``crn_base``; baselines keep their name."""
    if m.startswith("model:"):
        p = Path(m.removeprefix("model:"))
        return p.parent.name if p.stem in ("best", "last") else p.stem
    return m


def cmd_eval(a: argparse.Namespace) -> None:
    import json

    from .evaluate import evaluate, format_breakdown, format_table

    root = a.data_root or data_root()
    out = a.results / a.set
    summaries = {}
    for m in a.methods:
        label = method_label(m)
        summaries[label] = evaluate(root / "testsets" / a.set, m, out, a.workers, a.limit, not a.no_dnsmos,
                                    a.save_audio, label)
        (out / f"{label}.summary.json").write_text(json.dumps(summaries[label], indent=1))
    print(format_table(summaries))
    if len(next(iter(summaries.values()))["by_noise"]) > 1:
        print("\nPESQ by noise\n" + format_breakdown(summaries, "by_noise", "pesq"))
    if len(next(iter(summaries.values()))["by_snr"]) > 1:
        print("\nPESQ by SNR\n" + format_breakdown(summaries, "by_snr", "pesq"))


def cmd_report(a: argparse.Namespace) -> None:
    """Rebuild summary tables from saved per-item CSVs (no re-scoring)."""
    import csv

    from .evaluate import format_breakdown, format_table, summarise

    out = a.results / a.set
    summaries = {}
    for m in a.methods:
        with open(out / f"{m}.csv", newline="") as f:
            summaries[m] = summarise(list(csv.DictReader(f)))
    print(format_table(summaries))
    for metric in a.metrics:
        print(f"\n{metric} by noise\n" + format_breakdown(summaries, "by_noise", metric))
        print(f"\n{metric} by SNR\n" + format_breakdown(summaries, "by_snr", metric))


def cmd_info(a: argparse.Namespace) -> None:
    root = a.data_root or data_root()
    print(f"data root: {root}")
    for p in sorted((root / "prepared").glob("*.npy")):
        c = open_corpus(p.stem, root)
        print(f"  {p.stem:18s} {len(c):6d} clips {c.hours():7.2f} h  groups={len(c.group_names)}")
    for d in sorted((root / "testsets").glob("*/manifest.csv")):
        print(f"  testset {d.parent.name:14s} {len(d.read_text().splitlines()) - 1:6d} items")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vp", description=__doc__)
    p.add_argument("--data-root", type=Path, default=None, help="default: $VP_DATA_ROOT or ~/data/voicepickup")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("prepare", help="resample raw downloads into memory-mapped corpora")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("make-testsets", help="write the fixed evaluation sets")
    s.add_argument("--sets", nargs="+", default=["vbd_official", "grid", "headset2mic"])
    s.add_argument("--grid-utts", type=int, default=120)
    s.add_argument("--headset-utts", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_make_testsets)

    s = sub.add_parser("eval", help="enhance + score a test set; writes results/<set>/<method>.csv")
    s.add_argument("--set", required=True, choices=["vbd_official", "grid", "headset2mic"])
    s.add_argument("--methods", nargs="+", required=True,
                   help="baseline names (see baselines.py) or model:<checkpoint.pt>")
    s.add_argument("--results", type=Path, default=Path("results"))
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--limit", type=int, default=None, help="score a spread-out subset of N items")
    s.add_argument("--no-dnsmos", action="store_true")
    s.add_argument("--save-audio", action="store_true")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("report", help="tables from saved results/<set>/<method>.csv")
    s.add_argument("--set", required=True)
    s.add_argument("--methods", nargs="+", required=True)
    s.add_argument("--metrics", nargs="+", default=["pesq"])
    s.add_argument("--results", type=Path, default=Path("results"))
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("info", help="summarise prepared data")
    s.set_defaults(func=cmd_info)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
