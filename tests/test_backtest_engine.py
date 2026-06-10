import numpy as np
import pandas as pd

from mlportfolio.backtest.engine import WalkForwardBacktester
from mlportfolio.optimize.markowitz import equal_weights


def test_no_lookahead_strategy_only_sees_past(synthetic_data):
    seen_dates = []

    def strat(data_slice, asof):
        # The slice handed to the strategy must never contain future data.
        assert data_slice.returns.index.max() <= asof
        seen_dates.append(asof)
        return equal_weights(data_slice.tickers)

    bt = WalkForwardBacktester(rebalance="M", transaction_cost_bps=10, warmup_days=300)
    res = bt.run(synthetic_data, strat, name="eq")
    assert len(seen_dates) > 0
    assert len(res.returns) == len(synthetic_data.returns)


def test_equal_weight_runs_and_is_finite(synthetic_data):
    bt = WalkForwardBacktester(rebalance="M", transaction_cost_bps=10, warmup_days=300)

    def strat(data_slice, asof):
        return equal_weights(data_slice.tickers)

    res = bt.run(synthetic_data, strat, name="eq")
    assert np.isfinite(res.equity_curve.iloc[-1])
    assert res.equity_curve.iloc[-1] > 0
    assert "sharpe" in res.summary


def test_transaction_costs_reduce_return(synthetic_data):
    def churn(data_slice, asof):
        # Alternate concentration to force turnover.
        w = equal_weights(data_slice.tickers)
        if asof.month % 2 == 0:
            w[:] = 0.0
            w.iloc[0] = 1.0
        return w

    cheap = WalkForwardBacktester("M", transaction_cost_bps=0, warmup_days=300).run(
        synthetic_data, churn
    )
    pricey = WalkForwardBacktester("M", transaction_cost_bps=100, warmup_days=300).run(
        synthetic_data, churn
    )
    assert pricey.equity_curve.iloc[-1] <= cheap.equity_curve.iloc[-1] + 1e-9


def test_weights_logged_at_each_rebalance(synthetic_data):
    bt = WalkForwardBacktester("M", transaction_cost_bps=5, warmup_days=300)
    res = bt.run(synthetic_data, lambda d, a: equal_weights(d.tickers))
    # Each logged weight row sums to ~1.
    sums = res.weights.sum(axis=1)
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-6)


def test_normalize_freq_valid_for_installed_pandas():
    from pandas.tseries.frequencies import to_offset
    from mlportfolio.backtest.engine import _normalize_freq
    # Whatever the installed pandas accepts, the normalized alias must parse.
    for legacy in ["M", "Q", "Y"]:
        to_offset(_normalize_freq(legacy))
