import numpy as np
import pytest

from mlportfolio.drl.env import compute_observation, normalize_action, PortfolioEnv
from mlportfolio.drl.agent import build_rebalance_dates, random_policy_strategy
from mlportfolio.backtest.engine import WalkForwardBacktester


def test_normalize_action_sums_to_one():
    w = normalize_action(np.array([1.0, 3.0, 0.0, 0.0]), 4)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()
    # all-zero action -> equal weights
    eq = normalize_action(np.zeros(4), 4)
    assert np.allclose(eq, 0.25)


def test_observation_shape_and_finite(synthetic_data):
    obs = compute_observation(synthetic_data.returns, vix_value=18.0)
    assert obs.shape[0] == 3 * len(synthetic_data.tickers) + 2
    assert np.isfinite(obs).all()


def test_env_reset_step_and_episode(synthetic_data):
    dates = build_rebalance_dates(synthetic_data.returns.index, "M", warmup_days=252)
    env = PortfolioEnv(synthetic_data.returns, dates, vix=synthetic_data.vix, seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    steps = 0
    done = False
    while not done and steps < 1000:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        assert abs(info["weights"].sum() - 1.0) < 1e-6
        done = term or trunc
        steps += 1
    assert steps == len(dates) - 1


def test_random_agent_runs_through_backtester(synthetic_data):
    strat = random_policy_strategy(synthetic_data.tickers, vix=synthetic_data.vix, seed=2)
    bt = WalkForwardBacktester(rebalance="M", transaction_cost_bps=10, warmup_days=252)
    res = bt.run(synthetic_data, strat, name="DRL_random")
    assert np.isfinite(res.equity_curve.iloc[-1])
    assert np.allclose(res.weights.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert "sharpe" in res.summary
