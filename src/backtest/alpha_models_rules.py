from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from src.signals import compute_residual_returns, mean_reversion_signal
from .helpers import momentum_12_1_scores, winsorize

def momentum_mu(prices_train: pd.DataFrame) -> pd.Series:
    return momentum_12_1_scores(prices_train, gap_days=21, lookback_days=252)

def mr_mu(rets_train: pd.DataFrame, spy_rets: Optional[pd.Series], lk: int, mr_beta_lookback: int) -> pd.Series:
    if spy_rets is not None:
        common_idx = rets_train.index.intersection(spy_rets.index)
        resid_train = compute_residual_returns(rets_train.loc[common_idx], spy_rets.loc[common_idx], lookback=mr_beta_lookback)
    else:
        resid_train = rets_train.copy()
    mr_full = mean_reversion_signal(resid_train, lookback=lk)
    return mr_full.iloc[-1] if len(mr_full) else pd.Series(0.0, index=rets_train.columns)

def combo_mom_mr_mu(prices_train: pd.DataFrame, rets_train: pd.DataFrame,
                    spy_rets: Optional[pd.Series], mr_lookback: int, mr_beta_lookback: int,
                    weight: float) -> pd.Series:
    mom = momentum_mu(prices_train).reindex(rets_train.columns).fillna(0.0)
    if spy_rets is not None:
        common_idx = rets_train.index.intersection(spy_rets.index)
        resid_train = compute_residual_returns(rets_train.loc[common_idx], spy_rets.loc[common_idx], lookback=mr_beta_lookback)
    else:
        resid_train = rets_train.copy()
    mr_full = mean_reversion_signal(resid_train, lookback=mr_lookback)
    mr = (mr_full.iloc[-1] if len(mr_full) else pd.Series(0.0, index=rets_train.columns)).reindex(rets_train.columns).fillna(0.0)

    def _z(v: pd.Series) -> pd.Series:
        s = v.std()
        return v*0.0 if (s is None or not np.isfinite(s) or s == 0) else (v - v.mean()) / (s + 1e-12)

    return _z(mom) + float(weight) * _z(mr)

def overlay_mr(mu: pd.Series,
               rets_train: pd.DataFrame,
               spy_rets: Optional[pd.Series],
               mr_lookback: int,
               mr_beta_lookback: int,
               overlay_weight: float,
               standardize_mu: bool) -> pd.Series:
    if spy_rets is None:
        return mu
    common_idx = rets_train.index.intersection(spy_rets.index)
    resid_train_overlay = compute_residual_returns(rets_train.loc[common_idx], spy_rets.loc[common_idx], lookback=mr_beta_lookback)
    mr_overlay_full = mean_reversion_signal(resid_train_overlay, lookback=mr_lookback) if len(resid_train_overlay) else pd.DataFrame()
    if not len(mr_overlay_full):
        return mu
    mr_today = mr_overlay_full.iloc[-1].reindex(mu.index).fillna(0.0)
    mu_z = (mu - mu.mean()) / (mu.std() + 1e-12) if mu.std() > 0 else mu
    disagree = (np.sign(mu_z) != np.sign(mr_today)).astype(float)
    scale = 1.0 - float(overlay_weight) * disagree * mr_today.abs().clip(0.0, 1.0)
    mu_new = (mu_z * scale).fillna(0.0)
    if standardize_mu and mu_new.std() > 0:
        mu_new = (mu_new - mu_new.mean()) / (mu_new.std() + 1e-12)
    return mu_new

def finalize_mu(mu: pd.Series, sector_map: Optional[pd.Series | dict], *,
                sector_neutral_signal: bool, standardize_mu: bool, winsor_mu_k: float) -> pd.Series:
    if sector_neutral_signal and sector_map is not None:
        sec = sector_map if isinstance(sector_map, pd.Series) else pd.Series(sector_map)
        sec = sec.reindex(mu.index)
        mu = mu.groupby(sec).transform(lambda s: s - s.mean())
    if standardize_mu and mu.std() > 0:
        mu = (mu - mu.mean()) / (mu.std() + 1e-12)
    mu = winsorize(mu, k=winsor_mu_k)
    return mu
