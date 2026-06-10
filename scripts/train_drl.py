"""Train and evaluate the Deep RL allocation agent (experimental).

The agent is trained on the first part of the sample and evaluated out-of-sample
through the SAME walk-forward backtester as the other strategies, so the numbers
are directly comparable.

Training needs PyTorch + Stable-Baselines3:

    pip install -e ".[drl]"

Examples
--------
    python scripts/train_drl.py --synthetic --timesteps 20000
    python scripts/train_drl.py --start 2010-01-01 --end 2024-12-31 --algo PPO

Without the [drl] extra installed, the script still runs a random-policy agent so
you can see the evaluation plumbing (it will not be any good — that's the point of
the comparison).

NOTE: experimental / research module. DRL results are noisy and seed-dependent and
frequently fail to beat the simple baselines. Not financial advice.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mlportfolio.config import Config  # noqa: E402
from mlportfolio.pipeline import Pipeline, summary_table  # noqa: E402
from mlportfolio.backtest.engine import WalkForwardBacktester  # noqa: E402
from mlportfolio.backtest.metrics import performance_summary  # noqa: E402
from mlportfolio.drl.agent import (  # noqa: E402
    build_rebalance_dates,
    random_policy_strategy,
    DRLStrategy,
    train_drl,
    HAS_SB3,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train/evaluate the DRL allocation agent")
    p.add_argument("--config", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--timesteps", type=int, default=20000)
    p.add_argument("--algo", default="PPO", choices=["PPO", "A2C"])
    p.add_argument("--train-frac", type=float, default=0.6, help="Fraction of dates for training")
    p.add_argument("--rebalance", default="M")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default=None, help="Path to save the trained model (.zip)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config.load(args.config)
    overrides = {"dates.start": args.start, "dates.end": args.end, "seed": args.seed,
                 "optimize.rebalance": args.rebalance}
    cfg = cfg.override({k: v for k, v in overrides.items() if v is not None})

    print(f"Loading data (synthetic={args.synthetic}) ...")
    pipe = Pipeline(cfg, synthetic=args.synthetic).load()
    data = pipe.data
    cost = cfg.get("backtest", "transaction_cost_bps", default=10) / 1e4
    warmup = cfg.get("backtest", "warmup_days", default=504)

    all_dates = build_rebalance_dates(data.returns.index, args.rebalance, warmup_days=warmup)
    if len(all_dates) < 12:
        print("Not enough history for a train/eval split.")
        return 1
    split = int(len(all_dates) * args.train_frac)
    train_dates = all_dates[:split]
    split_date = all_dates[split]
    print(f"{len(all_dates)} rebalances · train on {len(train_dates)} "
          f"(through {train_dates[-1].date()}), evaluate out-of-sample from {split_date.date()}")

    if HAS_SB3:
        print(f"Training {args.algo} for {args.timesteps} timesteps (this can take a few minutes) ...")
        model = train_drl(
            data.returns, train_dates, vix=data.vix,
            total_timesteps=args.timesteps, cost=cost, algo=args.algo, seed=args.seed,
        )
        drl_strat = DRLStrategy(model, data.tickers, vix=data.vix)
        if args.save:
            model.save(args.save)
            print(f"Saved model to {args.save}")
    else:
        print("stable-baselines3 not installed -> using a RANDOM-policy agent for the demo.")
        print('   Install training support with:  pip install -e ".[drl]"')
        model = None
        drl_strat = random_policy_strategy(data.tickers, vix=data.vix, seed=args.seed)

    print("Evaluating all strategies out-of-sample ...")
    bt = WalkForwardBacktester(
        rebalance=args.rebalance,
        transaction_cost_bps=cfg.get("backtest", "transaction_cost_bps", default=10),
        warmup_days=warmup,
        smoothing=cfg.get("backtest", "smoothing", default=1.0),
        no_trade_band=cfg.get("backtest", "no_trade_band", default=0.0),
        risk_free_annual=cfg.get("backtest", "risk_free_annual", default=0.0),
    )
    drl_res = bt.run(data, drl_strat, name="DRL_agent")

    artifacts = pipe.run()
    name = "DRL_agent" if HAS_SB3 else "DRL_random"

    # Compare every strategy on the out-of-sample window only.
    rows = []
    streams = {n: r.returns for n, r in artifacts.results.items()}
    streams[name] = drl_res.returns
    for sname, rets in streams.items():
        oos = rets.loc[split_date:]
        summ = performance_summary(oos)
        summ["name"] = sname
        rows.append(summ)
    table = pd.DataFrame(rows).set_index("name")[
        ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"]
    ].round(4)

    print(f"\n=== Out-of-sample performance (from {split_date.date()}) ===")
    print(table.to_string())

    outdir = cfg.get("output", "dir", default="outputs")
    os.makedirs(outdir, exist_ok=True)
    table.to_csv(os.path.join(outdir, "drl_comparison.csv"))
    print(f"\nSaved comparison to {os.path.join(outdir, 'drl_comparison.csv')}")
    print("\nNote: experimental module. DRL results are noisy and often do not beat "
          "the baselines. Not financial advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
