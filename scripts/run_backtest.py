"""Command-line entry point for the full backtest comparison.

Examples
--------
Offline smoke test on synthetic data (no network needed)::

    python scripts/run_backtest.py --synthetic

Real market data::

    python scripts/run_backtest.py --start 2010-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``src`` importable when run directly from the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402

from mlportfolio.config import Config  # noqa: E402
from mlportfolio.pipeline import Pipeline, summary_table  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ML-enhanced portfolio backtest")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--synthetic", action="store_true", help="Run offline on synthetic data")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--outdir", default=None, help="Output directory")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config.load(args.config)
    overrides = {
        "dates.start": args.start,
        "dates.end": args.end,
        "seed": args.seed,
        "output.dir": args.outdir,
    }
    cfg = cfg.override({k: v for k, v in overrides.items() if v is not None})

    outdir = cfg.get("output", "dir", default="outputs")
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading data (synthetic={args.synthetic}) ...")
    pipe = Pipeline(cfg, synthetic=args.synthetic).load()
    print(f"Universe: {pipe.data.tickers}")
    print(f"Sample: {pipe.data.returns.index[0].date()} -> {pipe.data.returns.index[-1].date()} "
          f"({len(pipe.data.returns)} days)")

    print("Running walk-forward backtest across strategies ...")
    artifacts = pipe.run()

    table = summary_table(artifacts)
    print("\n=== Performance summary ===")
    print(table.to_string())
    print(f"\nRandom Forest avg in-sample RMSE: {artifacts.rmse['rf_avg_in_sample_rmse']:.5f}")

    # Persist outputs.
    table.to_csv(os.path.join(outdir, "summary.csv"))
    curves = pd.DataFrame({n: r.equity_curve for n, r in artifacts.results.items()})
    curves.to_csv(os.path.join(outdir, "equity_curves.csv"))

    # Compute the headline "improvement vs Markowitz" figure for the CV.
    base = artifacts.results["Markowitz"].summary["sharpe"]
    best_ml = max(
        artifacts.results["BlackLitterman_RF"].summary["sharpe"],
        artifacts.results["RegimeAware_BL_RF"].summary["sharpe"],
    )
    if base and base == base and base != 0:  # not NaN, not zero
        improvement = (best_ml - base) / abs(base) * 100
        print(f"\nBest ML Sharpe vs Markowitz Sharpe: {improvement:+.1f}%")

    try:
        _save_plot(curves, outdir)
        print(f"Saved equity-curve plot to {os.path.join(outdir, 'equity_curves.png')}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(Skipped plot: {exc})")

    print(f"\nOutputs written to: {outdir}/")
    return 0


def _save_plot(curves: pd.DataFrame, outdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in curves.columns:
        ax.plot(curves.index, curves[col], label=col, linewidth=1.5)
    ax.set_title("Walk-forward equity curves (growth of $1)")
    ax.set_ylabel("Cumulative growth")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity_curves.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
