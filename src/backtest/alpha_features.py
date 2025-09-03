from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from src.signals import compute_residual_returns, mean_reversion_signal
from src.features import compute_betas_to_market
from .helpers import momentum_12_1_scores, aggregate_period_returns

def _zscore_cs(vec: pd.Series) -> pd.Series:
    s = float(vec.std())
    if not np.isfinite(s) or s == 0:
        return vec.reindex(vec.index).fillna(0.0) * 0.0
    return (vec - vec.mean()) / (s + 1e-12)

def build_features_for_date(
    d: pd.Timestamp,
    prices: pd.DataFrame,
    rets: pd.DataFrame,
    spy_rets: Optional[pd.Series],
    beta_lookback: int,
    mr_lookback: int,
) -> pd.DataFrame:
    tickers = list(prices.columns)
    px = prices.loc[:d]
    rx = rets.loc[:d]
    if rx.empty or px.empty:
        return pd.DataFrame(index=tickers, columns=["z_mom", "z_mr", "z_idvol", "z_beta", "tau"], dtype=float)

    mom = momentum_12_1_scores(px).reindex(tickers)
    z_mom = _zscore_cs(mom.fillna(0.0))

    if spy_rets is not None:
        common = rx.index.intersection(spy_rets.index)
        resid = compute_residual_returns(rx.loc[common], spy_rets.loc[common], lookback=max(beta_lookback, 63))
    else:
        resid = rx.copy()
    mr_full = mean_reversion_signal(resid, lookback=mr_lookback) if len(resid) else pd.DataFrame()
    mr_today = (mr_full.iloc[-1] if len(mr_full) else pd.Series(0.0, index=tickers)).reindex(tickers).fillna(0.0)
    z_mr = _zscore_cs(mr_today)

    idv = rx.tail(21).std().reindex(tickers).fillna(0.0)
    z_idv = _zscore_cs(idv)

    if spy_rets is not None:
        start = d - pd.Timedelta(days=beta_lookback)
        rx_win = rx.loc[rx.index > start]
        mkt_win = spy_rets.loc[spy_rets.index > start]
        common2 = rx_win.index.intersection(mkt_win.index)
        if len(common2) >= 5:
            betas = compute_betas_to_market(rx_win.loc[common2], mkt_win.loc[common2])
            z_beta = _zscore_cs(betas.reindex(tickers).fillna(0.0))
        else:
            z_beta = pd.Series(0.0, index=tickers)
    else:
        z_beta = pd.Series(0.0, index=tickers)

    start_date = prices.index.min()
    tau = float((d - start_date).days) / 252.0
    tau_col = pd.Series(tau, index=tickers)

    feats = pd.DataFrame({
        "z_mom": z_mom,
        "z_mr": z_mr,
        "z_idvol": z_idv,
        "z_beta": z_beta,
        "tau": tau_col,
    }).fillna(0.0)
    return feats

def build_train_set(
    prices: pd.DataFrame,
    rets: pd.DataFrame,
    spy_rets: Optional[pd.Series],
    rebal_dates: pd.DatetimeIndex,
    t: pd.Timestamp,
    train_lookback_days: int,
    beta_lookback: int,
    mr_lookback: int,
    *,
    per_date_name_frac: Optional[float] = None,  # subset of names per date
    label_mode: str = "none",                    # "none" | "zmad" | "rank"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (rebal_dates < t) & (rebal_dates >= (t - pd.Timedelta(days=train_lookback_days)))
    d_list = list(rebal_dates[mask])
    if len(d_list) < 3:
        return np.empty((0, 5)), np.empty((0,)), np.empty((0,))

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    d_rows: List[pd.Timestamp] = []

    for j in range(len(d_list) - 1):
        d = d_list[j]
        d_next = d_list[j + 1]

        feats = build_features_for_date(d, prices, rets, spy_rets, beta_lookback, mr_lookback)
        y_vec = aggregate_period_returns(rets, d, d_next).reindex(feats.index).astype(float)

        m = y_vec.notna()
        if m.sum() == 0:
            continue

        if per_date_name_frac is not None and 0 < per_date_name_frac < 1:
            idx_names = np.where(m.values)[0]
            k = max(1, int(len(idx_names) * per_date_name_frac))
            rs = np.random.RandomState(int(pd.Timestamp(d).value % (2**32 - 1)))
            pick = rs.choice(idx_names, size=k, replace=False)
            mask_pick = np.zeros_like(m.values, dtype=bool)
            mask_pick[pick] = True
            m = pd.Series(mask_pick, index=feats.index)

        y_vals = y_vec.loc[m].values.astype(float)
        lm = (label_mode or "none").lower()
        if lm == "rank":
            r = pd.Series(y_vals).rank(method="average", pct=True).to_numpy()
            y_vals = (r - 0.5).astype(float)
        elif lm == "zmad":
            med = float(np.median(y_vals))
            mad = float(np.median(np.abs(y_vals - med))) + 1e-12
            y_vals = np.clip((y_vals - med) / (1.4826 * mad), -3.0, 3.0)

        X_rows.append(feats.loc[m, ["z_mom", "z_mr", "z_idvol", "z_beta", "tau"]].values)
        y_rows.append(y_vals)
        d_rows.append(np.array([d] * len(y_vals)))

    if not X_rows:
        return np.empty((0, 5)), np.empty((0,)), np.empty((0,))

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    dates_arr = np.concatenate(d_rows)

    y_mean, y_std = float(np.mean(y)), float(np.std(y) + 1e-12)
    y = (y - y_mean) / y_std
    return X, y, dates_arr
