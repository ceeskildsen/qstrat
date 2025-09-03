# src/metrics.py
import pandas as pd
import numpy as np

# ---------------- Core metrics ----------------

def compute_metrics(
    pnl: pd.Series,
    weights: pd.DataFrame,
    ann_factor: int = 52,
    risk_free_annual: float | None = None,
    risk_free_periodic: pd.Series | None = None,  # per-period RF aligned to pnl.index
) -> dict:
    """
    Computes annualized return/vol/Sharpe on *excess* returns if a risk-free is provided.
    - If risk_free_periodic is given, subtract it date-by-date (must align to pnl.index).
    - Else if risk_free_annual is given, convert to per-period and subtract.
    - Else compute metrics on raw pnl.
    Also returns turnover stats and (NEW) max_drawdown (magnitude).
    """
    pnl = pd.Series(pnl).astype(float).dropna()

    # Align / compute excess returns
    if risk_free_periodic is not None:
        rf = pd.Series(risk_free_periodic).reindex(pnl.index).fillna(0.0)
        excess = pnl - rf
    elif risk_free_annual is not None:
        rf_per = (1.0 + float(risk_free_annual)) ** (1.0 / ann_factor) - 1.0
        excess = pnl - rf_per
    else:
        excess = pnl

    # Annualized return/vol/Sharpe
    avg = excess.mean()
    vol = excess.std()
    ann_return = avg * ann_factor
    ann_vol = vol * np.sqrt(ann_factor)
    sharpe = 0.0 if ann_vol == 0 else ann_return / ann_vol

    # Turnover per rebalance (L1 change in weights)
    w = weights.sort_index()
    dw = w.diff().abs()
    if not w.empty:
        dw.iloc[0] = w.iloc[0].abs()  # cost to open the book
    turnover_per_reb = dw.sum(axis=1)
    avg_turnover = float(turnover_per_reb.mean())
    ann_turnover = avg_turnover * ann_factor

    # --- NEW: Max drawdown (magnitude, NaN-safe) on the same series used for metrics ---
    eq = (1.0 + pd.Series(excess, copy=False).dropna()).cumprod()
    if eq.size >= 2:
        peak = eq.cummax()
        dd = eq / peak - 1.0
        mdd_mag = float(-dd.min()) if np.isfinite(dd.min()) else 0.0  # positive magnitude
    else:
        mdd_mag = 0.0

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "avg_turnover_per_rebalance": float(avg_turnover),
        "ann_turnover": float(ann_turnover),
        "periods": int(len(excess)),
        "max_drawdown": mdd_mag,  # <- added
    }



# ---------------- IC utilities (kept for your signal lab) ----------------

def information_coefficient(signal: pd.DataFrame, future_returns: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """
    Cross-sectional IC time series: correlation across names between today's signal and next-period returns.
    Returns a daily Series of IC values.
    """
    common = signal.index.intersection(future_returns.index)
    sig = signal.loc[common]
    fut = future_returns.loc[common]
    out = []

    for d in common:
        x = sig.loc[d]
        y = fut.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= 3:
            if method == "spearman":
                rho = x[m].rank().corr(y[m].rank())
            else:
                rho = x[m].corr(y[m])
            out.append(rho)
        else:
            out.append(np.nan)

    return pd.Series(out, index=common).dropna()


def information_coefficient_safe(signal: pd.DataFrame, future_returns: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """Guarded IC (avoids rare zero-variance crashes)."""
    common = signal.index.intersection(future_returns.index)
    sig = signal.loc[common]
    fut = future_returns.loc[common]
    out = []

    for d in common:
        x = sig.loc[d]
        y = fut.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= 3:
            xr = x[m].rank() if method == "spearman" else x[m]
            yr = y[m].rank() if method == "spearman" else y[m]
            if xr.std() == 0 or yr.std() == 0:
                out.append(np.nan)
            else:
                out.append(xr.corr(yr))
        else:
            out.append(np.nan)

    return pd.Series(out, index=common).dropna()
