from __future__ import annotations
import numpy as np
import pandas as pd
from src.utils import to_returns

def winsorize(s: pd.Series, k: float = 3.0) -> pd.Series:
    if s.std() == 0 or not np.isfinite(s.std()):
        return s.fillna(0.0)
    z = (s - s.mean()) / (s.std() + 1e-12)
    return z.clip(-k, k)

def momentum_12_1_scores(prices_train: pd.DataFrame, gap_days: int = 21, lookback_days: int = 252) -> pd.Series:
    rets = to_returns(prices_train)
    prod_lookback = (1.0 + rets).rolling(lookback_days).apply(np.prod, raw=True) - 1.0
    mom = prod_lookback.shift(gap_days)
    return mom.iloc[-1].dropna()

def aggregate_period_returns(ret: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    sl = ret.loc[(ret.index > start) & (ret.index <= end)]
    if sl.empty:
        return pd.Series(0.0, index=ret.columns)
    return (1.0 + sl).prod() - 1.0

def build_rebalance_index(prices: pd.DataFrame, rets: pd.DataFrame, rebalance_freq: str) -> pd.DatetimeIndex:
    rebal_dates = rets.resample(rebalance_freq).last().index
    if len(rebal_dates) < 2:
        rebal_dates = prices.resample(rebalance_freq).last().index
    return rebal_dates
