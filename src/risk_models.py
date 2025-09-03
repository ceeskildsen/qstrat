# src/risk_models.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# PSD guard via eigen-floor (symmetrize → eig → floor → rebuild)
# ============================================================

def _ensure_psd_eigfloor(
    S_df: pd.DataFrame,
    floor_frac: float = 1e-6,   # floor at ≈ 1e-6 × mean eigenvalue
    abs_floor: float = 1e-12    # absolute fallback floor
) -> pd.DataFrame:
    """
    Make a covariance matrix numerically PSD:
      - Symmetrize
      - Eigen-decompose
      - Floor eigenvalues at max(floor_frac * mean_eigenvalue, abs_floor)
      - Reconstruct

    Returns a DataFrame with the same index/columns.
    """
    if S_df.empty:
        return S_df

    S = 0.5 * (S_df.values + S_df.values.T)
    vals, vecs = np.linalg.eigh(S)

    mean_ev = float(np.mean(vals)) if vals.size else 0.0
    floor = max(floor_frac * max(mean_ev, 0.0), abs_floor)

    vals = np.clip(vals, floor, None)
    S_psd = (vecs * vals) @ vecs.T
    S_psd = 0.5 * (S_psd + S_psd.T)

    return pd.DataFrame(S_psd, index=S_df.index, columns=S_df.columns)


# ============================================================
# Sample covariance (centered across time)
# ============================================================

def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Unbiased sample covariance:
        Σ = (1/(T-1)) * R_c^T R_c
    where R_c is returns centered across time (per column).
    Uses pairwise deletion for NaNs.
    """
    cols = list(returns.columns)
    if not cols:
        return pd.DataFrame(index=[], columns=[])

    S = returns.cov()  # unbiased, ddof=1
    if S.isna().all().all():
        n = len(cols)
        S = pd.DataFrame(np.eye(n) * 1e-8, index=cols, columns=cols)

    return _ensure_psd_eigfloor(S)


# ============================================================
# EWMA covariance (RiskMetrics-style, zero-mean per period)
# ============================================================

def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """
    Exponentially Weighted Moving Average covariance:
        Σ_t = λ Σ_{t-1} + (1-λ) r_t r_t^T
    - Zero-mean per period (standard in risk practice).
    - NaNs treated as 0.0 for stability.
    """
    cols = list(returns.columns)
    if not cols:
        return pd.DataFrame(index=[], columns=[])

    X = returns.fillna(0.0).to_numpy(dtype=float, copy=False)  # T x N
    N = len(cols)
    S = np.zeros((N, N), dtype=float)

    alpha = 1.0 - lam
    for r in X:
        S = lam * S + alpha * np.outer(r, r)

    S_df = pd.DataFrame(0.5 * (S + S.T), index=cols, columns=cols)
    return _ensure_psd_eigfloor(S_df)


# ============================================================
# Ledoit–Wolf shrinkage covariance (sklearn)
# ============================================================

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit–Wolf shrinkage covariance toward scaled identity (sklearn).
    Robust in small samples.
    """
    cols = list(returns.columns)
    if not cols:
        return pd.DataFrame(index=[], columns=[])

    try:
        from sklearn.covariance import LedoitWolf
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for Ledoit–Wolf. Install via: pip install scikit-learn"
        ) from e

    X = returns.fillna(0.0).to_numpy(dtype=float, copy=False)  # T x N
    if X.shape[0] < 2:
        n = len(cols)
        S_df = pd.DataFrame(np.eye(n) * 1e-8, index=cols, columns=cols)
        return _ensure_psd_eigfloor(S_df)

    lw = LedoitWolf().fit(X)
    S_df = pd.DataFrame(lw.covariance_, index=cols, columns=cols)
    return _ensure_psd_eigfloor(S_df)


# ============================================================
# GP-like kernel-weighted covariance (time-kernel smoother)
# ============================================================

def _time_kernel_weights(
    T: int,
    *,
    kernel: str = "matern32",
    length_scale: float = 32.0
) -> np.ndarray:
    """
    Build normalized, trailing time weights w_t for t=0..T-1 (0 oldest, T-1 latest).
    Weights peak at the most recent sample (t = T-1) and decay backward in time
    according to the chosen kernel.

    Supported kernels:
      - "rbf"        : exp(-0.5 * (Δ/ℓ)^2)
      - "exp"        : exp(-Δ/ℓ)                      (≈ EWMA with λ = exp(-1/ℓ))
      - "matern32"   : (1 + √3 x) exp(-√3 x),         x = Δ/ℓ
      - "matern52"   : (1 + √5 x + 5x²/3) exp(-√5 x), x = Δ/ℓ
    """
    if T <= 0:
        return np.zeros((0,), dtype=float)

    # Δ = distance (in days) from the newest observation
    # oldest row: Δ = T-1, newest row: Δ = 0
    Delta = (np.arange(T, dtype=float)[::-1])  # [T-1, ..., 1, 0]
    ell = max(1e-6, float(length_scale))
    x = Delta / ell

    k = kernel.lower()
    if k == "rbf":
        w = np.exp(-0.5 * x * x)
    elif k == "exp":
        w = np.exp(-x)
    elif k in ("matern32", "m32"):
        sqrt3 = np.sqrt(3.0)
        w = (1.0 + sqrt3 * x) * np.exp(-sqrt3 * x)
    elif k in ("matern52", "m52"):
        sqrt5 = np.sqrt(5.0)
        w = (1.0 + sqrt5 * x + 5.0 * x * x / 3.0) * np.exp(-sqrt5 * x)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Use 'rbf', 'exp', 'matern32', or 'matern52'.")

    w_sum = float(np.sum(w))
    if w_sum <= 0:
        # fallback to uniform
        w = np.ones(T, dtype=float) / float(T)
    else:
        w = w / w_sum
    return w


def gp_kernel_cov(
    returns: pd.DataFrame,
    *,
    kernel: str = "matern32",
    length_scale: float = 32.0,
    zero_mean: bool = True,
    shrink_to_identity: float | None = None
) -> pd.DataFrame:
    """
    Gaussian-process–like (time-kernel) covariance estimate:

        Σ ≈  Σ_t w_t · r_t r_tᵀ,  with w_t from a time kernel centered at "now".

    - Trailing-only smoother (weights peak at the most recent sample).
    - Kernels: RBF, Exp, Matérn 3/2 (default), Matérn 5/2.
    - zero_mean=True matches standard EWMA practice (faster, stabler).
      If False, will subtract weighted mean before forming Σ (rarely needed).

    Parameters
    ----------
    returns : (T x N) DataFrame
        Asset returns (dates x tickers).
    kernel : str
        'matern32' (default), 'rbf', 'exp', 'matern52'.
    length_scale : float
        Smoothing length in days; larger => smoother / slower decay.
        Note: Exp kernel with length_scale=ℓ corresponds roughly to EWMA λ=exp(-1/ℓ).
    zero_mean : bool
        Use zero-mean per period (True) or subtract the weighted cross-sectional mean (False).
    shrink_to_identity : float or None
        Optional scalar shrink α in [0,1] that blends toward σ̄² I:
            Σ ← (1-α) Σ + α·σ̄² I
        where σ̄² is the average variance (trace/N). Useful with very small T.

    Returns
    -------
    Σ_df : (N x N) DataFrame
        PSD-corrected covariance matrix.
    """
    cols = list(returns.columns)
    if not cols:
        return pd.DataFrame(index=[], columns=[])

    X = returns.to_numpy(dtype=float, copy=False)  # T x N
    T = X.shape[0]
    if T < 2:
        n = len(cols)
        S_df = pd.DataFrame(np.eye(n) * 1e-8, index=cols, columns=cols)
        return _ensure_psd_eigfloor(S_df)

    # Build trailing time weights
    w = _time_kernel_weights(T, kernel=kernel, length_scale=length_scale)  # shape (T,)

    # Optionally subtract weighted time-mean (rarely necessary for risk)
    if zero_mean:
        Xw = X  # zero-mean per period assumption
    else:
        mu_w = np.average(X, axis=0, weights=w)  # shape (N,)
        Xw = X - mu_w[None, :]

    # Σ = Xᵀ diag(w) X  (efficient)
    S = Xw.T @ (w[:, None] * Xw)

    # Optional small-sample shrinkage to identity
    if shrink_to_identity is not None and 0.0 < float(shrink_to_identity) <= 1.0:
        tr = float(np.trace(S))
        n = S.shape[0]
        sigma_bar2 = tr / max(1, n)
        S = (1.0 - float(shrink_to_identity)) * S + float(shrink_to_identity) * (sigma_bar2 * np.eye(n))

    S_df = pd.DataFrame(0.5 * (S + S.T), index=cols, columns=cols)
    return _ensure_psd_eigfloor(S_df)
