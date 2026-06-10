"""Command-line allocation recommendation.

Examples
--------
Offline (synthetic data, no network)::

    python scripts/recommend.py --synthetic

Real data, with your current holdings, writing a markdown report::

    python scripts/recommend.py --holdings "SPY:0.4,TLT:0.3,GLD:0.1" --report

NOTE: research/education tool, not financial advice.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mlportfolio.config import Config  # noqa: E402
from mlportfolio.recommend import recommend, format_report  # noqa: E402


def _parse_holdings(text: str):
    if not text:
        return None
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        tic, _, wt = part.partition(":")
        out[tic.strip().upper()] = float(wt)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Forward-looking allocation recommendation")
    p.add_argument("--config", default=None)
    p.add_argument("--synthetic", action="store_true", help="Run offline on synthetic data")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--holdings", default=None, help='e.g. "SPY:0.4,TLT:0.3,GLD:0.1"')
    p.add_argument("--no-regime", action="store_true", help="Disable regime adjustment")
    p.add_argument("--report", action="store_true", help="Write a markdown report to outputs/")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config.load(args.config)
    overrides = {"dates.start": args.start, "dates.end": args.end, "seed": args.seed}
    cfg = cfg.override({k: v for k, v in overrides.items() if v is not None})

    print(f"Building recommendation (synthetic={args.synthetic}) ...")
    rec = recommend(
        cfg,
        synthetic=args.synthetic,
        current_holdings=_parse_holdings(args.holdings),
        regime_aware=not args.no_regime,
    )

    report = format_report(rec)
    print("\n" + report)

    if args.report:
        outdir = cfg.get("output", "dir", default="outputs")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"recommendation_{rec.asof.date()}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"\nReport written to: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
