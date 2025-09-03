# src/optimizer.py
from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp


def mean_variance_opt(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    *,
    risk_aversion: float = 5.0,
    dollar_neutral: bool = True,
    long_only: bool = False,
    position_bound: float = 0.10,
    gross_limit: float | None = None,
    # accept BOTH names for robustness
    betas: np.ndarray | list | pd.Series | None = None,
    beta:  np.ndarray | list | pd.Series | None = None,
    beta_limit: float | None = None,
    risk_budget_daily: float | None = None,   # daily stdev cap (converted from annual outside)
    # sector neutrality (per-sector sums)
    sector_neutral_mat: np.ndarray | None = None,        # shape (K_sec x N)
    sector_cap_abs: float | np.ndarray | None = None,    # scalar or per-sector vector
    # NEW: generic factor neutrality (e.g., PCA)
    factor_neutral_mat: np.ndarray | None = None,        # shape (K_fac x N)
    factor_cap_abs: float | np.ndarray | None = None,    # scalar or per-factor vector
) -> pd.Series:
    """
    Maximize  mu^T w - λ * w^T Σ w
    subject to:
      - long_only: w >= 0, sum(w) = 1, w <= position_bound
      - else:     -position_bound <= w <= position_bound
                   dollar_neutral -> sum(w) = 0
      - gross_limit: ||w||_1 <= gross_limit
      - beta constraint: |β^T w| <= beta_limit
      - risk budget: w^T Σ w <= (risk_budget_daily)^2
      - sector neutrality (optional):  hard A_sec w = 0  or  soft |A_sec w| <= cap
      - factor neutrality (optional):  hard A_fac w = 0  or  soft |A_fac w| <= cap
    """
    # accept either 'beta' or 'betas'
    if betas is None and beta is not None:
        betas = beta

    mu = pd.Series(mu)
    Sigma = pd.DataFrame(Sigma).reindex(index=mu.index, columns=mu.index).fillna(0.0)

    n = len(mu)
    if n == 0:
        raise ValueError("Empty asset universe in optimizer.")

    w = cp.Variable(n)
    Q = Sigma.values
    mu_vec = mu.values

    cons = []

    if long_only:
        cons += [w >= 0, cp.sum(w) == 1.0, w <= position_bound]
    else:
        cons += [w <= position_bound, w >= -position_bound]
        if dollar_neutral:
            cons += [cp.sum(w) == 0]
        if gross_limit is not None:
            cons += [cp.norm1(w) <= float(gross_limit)]

    if risk_budget_daily is not None and risk_budget_daily > 0:
        cons += [cp.quad_form(w, Q) <= float(risk_budget_daily) ** 2]

    if betas is not None and beta_limit is not None:
        b = pd.Series(np.asarray(betas).reshape(-1), index=mu.index).reindex(mu.index).fillna(0.0).values
        cons += [cp.abs(b @ w) <= float(beta_limit)]

    # Sector neutrality: hard (equality) or soft (cap)
    if (not long_only) and (sector_neutral_mat is not None):
        A = np.asarray(sector_neutral_mat, dtype=float)
        if A.size > 0:
            if sector_cap_abs is None:
                cons += [A @ w == 0]
            else:
                cap = sector_cap_abs
                cap_vec = np.full(A.shape[0], float(cap)) if np.isscalar(cap) else np.asarray(cap, dtype=float).reshape(-1)
                if cap_vec.shape[0] != A.shape[0]:
                    raise ValueError("sector_cap_abs length must match number of sector rows.")
                cons += [cp.abs(A @ w) <= cap_vec]

    # NEW: Factor neutrality (e.g., PCA rows)
    if (not long_only) and (factor_neutral_mat is not None):
        F = np.asarray(factor_neutral_mat, dtype=float)
        if F.size > 0:
            if factor_cap_abs is None:
                cons += [F @ w == 0]
            else:
                capf = factor_cap_abs
                capf_vec = np.full(F.shape[0], float(capf)) if np.isscalar(capf) else np.asarray(capf, dtype=float).reshape(-1)
                if capf_vec.shape[0] != F.shape[0]:
                    raise ValueError("factor_cap_abs length must match number of factor rows.")
                cons += [cp.abs(F @ w) <= capf_vec]

    obj = cp.Maximize(mu_vec @ w - float(risk_aversion) * cp.quad_form(w, Q))
    prob = cp.Problem(obj, cons)

    # Solve (QP first, then conic fallback)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            prob.solve(verbose=False)

    if w.value is None:
        raise RuntimeError("Optimizer failed to solve.")

    return pd.Series(np.array(w.value).reshape(-1), index=mu.index)
