# src/signals.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import to_returns

# ============================================================
# Cross-sectional helpers (robust, NaN-safe, performance-tuned)
# ============================================================

def _cs_zscore_row(s: pd.Series, eps: float = 1e-12, min_non_nan: int = 3) -> pd.Series:
    """Standard cross-sectional z for one date (row). Returns 0s if too sparse."""
    x = s.to_numpy(copy=False)
    mask = np.isfinite(x)
    if mask.sum() < min_non_nan:
        return pd.Series(0.0, index=s.index)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=s.index)
    z = (x - mu) / (sd + eps)
    return pd.Series(z, index=s.index).fillna(0.0)


def _winsor_row(s: pd.Series, k: float = 3.0) -> pd.Series:
    """Clip to [-k, k] (NaN-safe)."""
    return s.clip(lower=-k, upper=k)


def _cs_robust_z_row(
    s: pd.Series,
    method: str = "mad",
    k: float = 3.0,
    eps: float = 1e-12,
    min_non_nan: int = 3,
) -> pd.Series:
    """
    Robust cross-sectional z per date. Short-circuits sparse rows.
    method: "mad" (default) or "std".
    Returns 0s where insufficient data.
    """
    x = s.to_numpy(copy=False)
    mask = np.isfinite(x)
    if mask.sum() < min_non_nan:
        # Too few names with data — return zeros to avoid NaN storms
        return pd.Series(0.0, index=s.index)

    if method == "mad":
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scale = 1.4826 * (mad if np.isfinite(mad) and mad > 0 else 0.0) + eps
        z = (x - med) / scale
    else:
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(0.0, index=s.index)
        z = (x - mu) / (sd + eps)

    # Clip only on finite entries for speed/stability
    z = np.where(np.isfinite(z), np.clip(z, -k, k), 0.0)

    # Re-center to ~0 mean over finite entries
    z_mean = np.nanmean(np.where(np.isfinite(z), z, np.nan))
    if np.isfinite(z_mean):
        z = np.where(np.isfinite(z), z - z_mean, 0.0)
    else:
        z = np.where(np.isfinite(z), z, 0.0)

    return pd.Series(z, index=s.index).fillna(0.0)


# ============================================================
# Residual returns (vectorized)
# ============================================================

def compute_residual_returns(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Vectorized rolling OLS residuals of asset returns vs market:
      eps_t = r_t - beta_t * m_t,
      with beta_t computed on the last `lookback` days.
    """
    idx = returns.index.intersection(market_returns.index)
    if len(idx) == 0:
        return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    R = returns.loc[idx]
    m = pd.Series(market_returns).loc[idx]

    # Rolling moments (guard against empty windows)
    Em = m.rolling(lookback, min_periods=max(5, lookback // 5)).mean()
    Vm = m.rolling(lookback, min_periods=max(5, lookback // 5)).var(ddof=1).replace(0, np.nan)

    Er  = R.rolling(lookback, min_periods=max(5, lookback // 5)).mean()
    Erm = (R.mul(m, axis=0)).rolling(lookback, min_periods=max(5, lookback // 5)).mean()

    beta = (Erm - Er.mul(Em, axis=0)).div(Vm, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    resid = R - beta.mul(m, axis=0)
    return resid.dropna(how="all")


def mean_reversion_signal(
    residuals: pd.DataFrame,
    lookback: int = 5,
    vol_norm_window: int = 60,
    winsor_k: float = 3.0,
    robust_z: bool = True,
    min_non_nan: int = 3,
) -> pd.DataFrame:
    """
    MR on market-residualized returns:
      - per-name vol normalization (rolling std)
      - sum over short lookback (negative for reversion)
      - robust cross-sectional z by date (MAD by default)
    """
    if residuals.empty:
        return residuals.copy()

    # 1) per-name vol normalization
    vol = residuals.rolling(vol_norm_window, min_periods=max(5, vol_norm_window // 5)).std()
    res_n = residuals.div(vol.replace(0, np.nan))

    # 2) rolling sum over lookback (negative for reversion)
    raw = -res_n.rolling(lookback, min_periods=max(2, lookback // 2)).sum()

    # Only z-score on rows with enough finite names
    valid_rows = raw.count(axis=1) >= int(min_non_nan)
    out = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)

    if robust_z:
        z_valid = raw.loc[valid_rows].apply(
            lambda r: _cs_robust_z_row(r, method="mad", k=winsor_k, min_non_nan=min_non_nan),
            axis=1
        )
    else:
        # (winsor -> standard z)
        raw_w = raw.loc[valid_rows].apply(lambda r: _winsor_row(r, winsor_k), axis=1)
        z_valid = raw_w.apply(lambda r: _cs_zscore_row(r, min_non_nan=min_non_nan), axis=1)

    out.loc[valid_rows, :] = z_valid
    # For invalid rows, fill zeros (neutral) instead of NaN to avoid downstream churn
    out.loc[~valid_rows, :] = 0.0
    return out


# ============================================================
# Momentum 12–1 signal (ratio-based, vectorized)
# ============================================================

def momentum_12_1_signal(
    prices: pd.DataFrame,
    *,
    gap_days: int = 21,          # ~1M gap between lookback and today
    lookback_days: int = 252,    # ~12M lookback
    zscore: bool = True,
    winsor_k: float | None = 3.0,
    sector_map: pd.Series | dict | None = None,  # optional soft neutralization of the signal
    sector_neutral_signal: bool = False,
    robust_z: bool = True,       # robust z (MAD) by default
    robust_method: str = "mad",
    min_non_nan: int = 3,
) -> pd.DataFrame:
    """
    Cross-sectional momentum 12–1 at date t (vectorized):
        mom_t = P_{t-21} / P_{t-252} - 1
    Returns a DataFrame aligned to 'prices' dates, with optional cross-sectional standardization.
    """
    if prices.empty:
        return prices.copy()

    P = prices.dropna(how="all").sort_index()

    # Vectorized ratio (creates NaNs at the start — expected)
    mom = P.shift(gap_days) / P.shift(lookback_days) - 1.0

    # Optional soft sector de-meaning on the signal (skip if row is all NaN)
    if sector_neutral_signal and sector_map is not None and mom.shape[1] > 1:
        sec = pd.Series(sector_map) if isinstance(sector_map, dict) else sector_map
        sec = sec.reindex(mom.columns)
        if sec.notna().any():
            # Demean each valid row within sectors
            def _demean_row(row: pd.Series) -> pd.Series:
                if row.count() < min_non_nan:
                    return row  # skip sparse rows; they’ll be handled by z-step
                return row.groupby(sec).transform(lambda r: r - r.mean())
            mom = mom.apply(_demean_row, axis=1)

    if not zscore:
        return mom

    # Z-score only on rows with enough finite values
    valid_rows = mom.count(axis=1) >= int(min_non_nan)
    out = pd.DataFrame(index=mom.index, columns=mom.columns, dtype=float)

    if robust_z:
        z_valid = mom.loc[valid_rows].apply(
            lambda r: _cs_robust_z_row(r, method=robust_method, k=winsor_k if winsor_k is not None else 3.0, min_non_nan=min_non_nan),
            axis=1
        )
    else:
        z_valid = mom.loc[valid_rows].apply(lambda r: _cs_zscore_row(r, min_non_nan=min_non_nan), axis=1)
        if winsor_k is not None:
            z_valid = z_valid.apply(lambda r: _winsor_row(r, k=winsor_k), axis=1)

    out.loc[valid_rows, :] = z_valid
    # Fill invalid rows with 0 (neutral) to avoid downstream NaN propagation/perf hits
    out.loc[~valid_rows, :] = 0.0
    return out


# ============================================================
# Simple z-score combiner
# ============================================================

def combine_signals_z(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    weight_secondary: float = 0.3,
    winsor_k: float = 3.0,
    restandardize: bool = True,
    min_non_nan: int = 3,
) -> pd.DataFrame:
    """
    Combine two signals by cross-sectional z-score each date:
        combo = z(primary) + weight_secondary * z(secondary)
    Then optionally restandardize and winsorize per date.

    NOTE: You can treat `weight_secondary` as your λ (mr_lambda) when the
          secondary is the MR score.
    """
    idx = primary.index.intersection(secondary.index)
    cols = primary.columns.intersection(secondary.columns)
    if len(idx) == 0 or len(cols) == 0:
        return pd.DataFrame(index=primary.index, columns=primary.columns, dtype=float)

    P = primary.loc[idx, cols].apply(lambda r: _cs_zscore_row(r, min_non_nan=min_non_nan), axis=1)
    S = secondary.loc[idx, cols].apply(lambda r: _cs_zscore_row(r, min_non_nan=min_non_nan), axis=1)
    combo = P + float(weight_secondary) * S

    if restandardize:
        combo = combo.apply(lambda r: _cs_zscore_row(r, min_non_nan=min_non_nan), axis=1)

    if winsor_k is not None:
        combo = combo.apply(lambda r: _winsor_row(r, winsor_k), axis=1)

    out = pd.DataFrame(index=primary.index, columns=primary.columns, dtype=float)
    out.loc[idx, cols] = combo
    # Leave other dates as 0 (neutral)
    out = out.fillna(0.0)
    return out


# ============================================================
# NEW: MR helpers with explicit mr_lambda for overlays/blends
# ============================================================

def mr_score_from_prices(
    prices: pd.DataFrame,
    *,
    market_returns: pd.Series | None = None,
    mr_lookback: int = 3,
    vol_norm_window: int = 60,
    winsor_k: float = 3.0,
    robust_z: bool = True,
    min_non_nan: int = 3,
    market_beta_lookback: int = 252,
) -> pd.DataFrame:
    """
    Build a cross-sectional MR z-score from prices.

    If `market_returns` is provided, we first compute market-residualized returns
    (rolling beta over `market_beta_lookback`) and apply MR to those residuals.
    Otherwise we apply MR to raw daily returns.

    Returns: MR z-score DataFrame aligned to `prices` (date x tickers).
    """
    rets = to_returns(prices).replace([np.inf, -np.inf], np.nan)
    if market_returns is not None:
        resid = compute_residual_returns(rets, market_returns, lookback=market_beta_lookback)
    else:
        resid = rets

    mr = mean_reversion_signal(
        resid,
        lookback=mr_lookback,
        vol_norm_window=vol_norm_window,
        winsor_k=winsor_k,
        robust_z=robust_z,
        min_non_nan=min_non_nan,
    )
    # Align to prices index/columns (fill missing with 0 to be neutral)
    mr = mr.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return mr


def apply_mr_overlay(
    alpha: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    mr_lambda: float = 0.10,
    market_returns: pd.Series | None = None,
    mr_lookback: int = 3,
    vol_norm_window: int = 60,
    winsor_k: float = 3.0,
    robust_z: bool = True,
    min_non_nan: int = 3,
    market_beta_lookback: int = 252,
) -> pd.DataFrame:
    """
    Blend an existing alpha with a mean-reversion score built from prices:

        alpha_blend = (1 - mr_lambda) * alpha + mr_lambda * mr_z

    - `mr_lambda` is your blend weight (λ).
    - If `market_returns` is provided, MR is computed on market-residualized returns.
    - Output is aligned to `alpha` and fills missing with 0 for neutrality.
    """
    if alpha.empty or prices.empty:
        return alpha.copy()

    mr = mr_score_from_prices(
        prices,
        market_returns=market_returns,
        mr_lookback=mr_lookback,
        vol_norm_window=vol_norm_window,
        winsor_k=winsor_k,
        robust_z=robust_z,
        min_non_nan=min_non_nan,
        market_beta_lookback=market_beta_lookback,
    )

    # Align and blend
    idx = alpha.index.intersection(mr.index)
    cols = alpha.columns.intersection(mr.columns)
    if len(idx) == 0 or len(cols) == 0:
        return alpha.fillna(0.0)

    blended = ((1.0 - float(mr_lambda)) * alpha.loc[idx, cols] +
               float(mr_lambda) * mr.loc[idx, cols])

    out = alpha.copy()
    out.loc[idx, cols] = blended
    return out.fillna(0.0)
