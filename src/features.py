# src/features.py
import pandas as pd
import numpy as np

def trailing_mean_returns(returns: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """Placeholder μ: cross-sectional rolling-mean of recent returns (last row)."""
    mu = returns.rolling(lookback).mean()
    return mu.iloc[-1].dropna()

def cs_momentum_6_1(prices: pd.DataFrame, months: int = 6, gap: int = 1) -> pd.Series:
    """
    Cross-sectional 6-1 momentum on the last available date:
      cumulative return from (t - 6 months) to (t - 1 month).
    Assumes ~21 trading days per month.
    """
    lookback = months * 21
    gap_days = gap * 21
    past = prices.shift(gap_days)                          # exclude most-recent month
    mom = past.pct_change(periods=lookback - gap_days)     # 6m return ending 1m ago
    return mom.iloc[-1].dropna()

def cs_momentum_12_1(prices: pd.DataFrame) -> pd.Series:
    """Cross-sectional 12-1 momentum on last date: return from t-12m to t-1m."""
    months, gap = 12, 1
    lookback = months * 21
    gap_days = gap * 21
    past = prices.shift(gap_days)
    mom = past.pct_change(periods=lookback - gap_days)
    return mom.iloc[-1].dropna()

def compute_betas_to_market(train_returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    """
    Rolling betas of each asset (columns of train_returns) to a market series.
    Beta_i = Cov(r_i, r_m) / Var(r_m), computed over the training window.
    """
    idx = train_returns.index.intersection(market_returns.index)
    R = train_returns.loc[idx]
    rm = market_returns.loc[idx]
    if len(R) < 5:
        return pd.Series(0.0, index=train_returns.columns)

    rm_mean = rm.mean()
    var_m = ((rm - rm_mean) ** 2).sum() / max(len(rm) - 1, 1)
    if var_m <= 0:
        return pd.Series(0.0, index=train_returns.columns)

    cov_im = ((R - R.mean()).mul(rm - rm_mean, axis=0)).sum() / max(len(rm) - 1, 1)
    betas = cov_im / var_m
    return betas.reindex(train_returns.columns).fillna(0.0)

def neutralize_mu_against_beta(mu: pd.Series, betas: pd.Series) -> pd.Series:
    """
    Cross-sectional regression: mu_i = a + b * beta_i + eps_i.
    Return residuals eps_i (mu with linear beta effect removed).
    """
    x = betas.reindex(mu.index).astype(float)
    y = mu.astype(float)
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return mu.fillna(0.0)

    x_m = x[mask]; y_m = y[mask]
    var_x = ((x_m - x_m.mean())**2).sum() / max(len(x_m) - 1, 1)
    if var_x <= 0:
        return mu.fillna(0.0)

    cov_xy = ((x_m - x_m.mean()) * (y_m - y_m.mean())).sum() / max(len(x_m) - 1, 1)
    b = cov_xy / var_x
    a = y_m.mean() - b * x_m.mean()

    resid = y - (a + b * x)
    return resid.fillna(0.0)
