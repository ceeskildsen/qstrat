# src/diagnostics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---- Core series helpers ----

def equity_from_pnl(pnl: pd.Series) -> pd.Series:
    pnl = pd.Series(pnl).dropna()
    return (1 + pnl).cumprod().rename("equity")

def drawdown(equity: pd.Series) -> pd.Series:
    eq = pd.Series(equity).dropna()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.rename("drawdown")

def rolling_sharpe(pnl: pd.Series, window: int, ann_factor: int) -> pd.Series:
    r = pd.Series(pnl).dropna()
    m = r.rolling(window).mean()
    s = r.rolling(window).std()
    rs = (m / s).replace([np.inf, -np.inf], np.nan) * np.sqrt(ann_factor)
    return rs.rename(f"rolling_sharpe_{window}")

def realized_vol(pnl: pd.Series, window: int, ann_factor: int) -> pd.Series:
    r = pd.Series(pnl).dropna()
    rv = r.rolling(window).std() * np.sqrt(ann_factor)
    return rv.rename(f"realized_vol_{window}")

def compute_turnover_series(weights: pd.DataFrame) -> pd.Series:
    w = weights.sort_index()
    dw = w.diff().abs()
    if len(w) > 0:
        dw.iloc[0] = w.iloc[0].abs()  # cost to open the book
    return dw.sum(axis=1).rename("turnover")

# ---- Exposures & constraints ----

def compute_betas(returns: pd.DataFrame, market_rets: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """
    Rolling market betas per name using OLS: cov(name, mkt)/var(mkt).
    Returns a DataFrame aligned to returns.index (dates) with columns = tickers.
    """
    rets = returns.copy().dropna(how="all")
    mkt = pd.Series(market_rets).dropna()
    common = rets.index.intersection(mkt.index)
    rets = rets.loc[common]
    mkt = mkt.loc[common]

    # Pre-compute rolling stats of market
    mkt_mean = mkt.rolling(lookback).mean()
    mkt_var = mkt.rolling(lookback).var()

    betas = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for col in rets.columns:
        x = rets[col]
        x_mean = x.rolling(lookback).mean()
        cov = (x.rolling(lookback).mean() * 0)  # placeholder index
        # cov(x,m) = E[x*m] - E[x]E[m]
        em = (x * mkt).rolling(lookback).mean()
        cov = em - x_mean * mkt_mean
        beta = cov / (mkt_var.replace(0, np.nan))
        betas[col] = beta
    return betas

def exposure_series(weights: pd.DataFrame, factor_loads: pd.DataFrame) -> pd.Series:
    """
    Dot each row of weights with the corresponding row of factor loads.
    Returns a Series indexed by weights.index.
    """
    idx = weights.index.intersection(factor_loads.index)
    W = weights.loc[idx].copy()
    B = factor_loads.loc[idx].copy()
    # Align columns; missing betas -> 0
    B = B.reindex(columns=W.columns).fillna(0.0)
    exp = (W * B).sum(axis=1)
    return exp.rename("exposure")

def constraint_binding_stats(
    weights: pd.DataFrame,
    betas_by_date: Optional[pd.DataFrame],
    position_bound: Optional[float],
    gross_limit: Optional[float],
    beta_limit: Optional[float],
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    For each rebalance date, report:
      - pos_bind_rate: share of names at |w| >= position_bound - eps
      - gross_bind:    boolean if sum(|w|) >= gross_limit - eps
      - beta_bind:     boolean if |w^T beta| >= beta_limit - eps  (requires betas_by_date)
      - beta_exposure: w^T beta
    """
    idx = weights.index
    out = pd.DataFrame(index=idx, columns=["pos_bind_rate","gross_bind","beta_bind","beta_exposure"], dtype=float)

    abs_w_sum = weights.abs().sum(axis=1)
    if gross_limit is not None:
        out["gross_bind"] = (abs_w_sum >= (gross_limit - eps)).astype(float)
    else:
        out["gross_bind"] = np.nan

    if position_bound is not None:
        pos_bind = (weights.abs() >= (position_bound - eps)).sum(axis=1) / weights.shape[1]
        out["pos_bind_rate"] = pos_bind
    else:
        out["pos_bind_rate"] = np.nan

    if beta_limit is not None and betas_by_date is not None and not betas_by_date.empty:
        # compute beta exposure per date
        idx2 = idx.intersection(betas_by_date.index)
        B = betas_by_date.reindex(idx2).reindex(columns=weights.columns, fill_value=0.0)
        W = weights.reindex(idx2).fillna(0.0)
        beta_exp = (W * B).sum(axis=1)
        bind = (beta_exp.abs() >= (beta_limit - eps)).astype(float)
        out.loc[idx2, "beta_exposure"] = beta_exp
        out.loc[idx2, "beta_bind"] = bind
    else:
        out["beta_exposure"] = np.nan
        out["beta_bind"] = np.nan

    return out

# ---- Cost sensitivity ----

def reprice_with_costs(pnl_after_costs: pd.Series, turnover: pd.Series, current_bps: float, test_bps: Iterable[float], ann_factor:int) -> pd.DataFrame:
    """
    Recover pre-cost pnl (approx) by adding back current cost, then re-apply alt costs.
    cost_t = bps * 1e-4 * turnover_t
    """
    pnl = pd.Series(pnl_after_costs).dropna()
    to = turnover.reindex(pnl.index).fillna(0.0)
    cost_now = (current_bps * 1e-4) * to
    pre = pnl + cost_now

    rows = []
    for bps in test_bps:
        new_cost = (bps * 1e-4) * to
        new_pnl = pre - new_cost
        mu = new_pnl.mean() * ann_factor
        sig = new_pnl.std() * np.sqrt(ann_factor)
        sharpe = 0.0 if sig == 0 else mu / sig
        rows.append({"bps": float(bps), "ann_return": float(mu), "ann_vol": float(sig), "sharpe": float(sharpe)})

    return pd.DataFrame(rows).sort_values("bps")

# ---- Risk model sensitivity (ex-ante vol from weights) ----

def ex_ante_vol_series(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    train_window_days: int,
    model: str = "ewma",
    ewma_lambda: float = 0.97,
    ann_factor: int = 252,
) -> pd.Series:
    """
    For each rebalance date t, compute sqrt(w_t^T Sigma_t w_t) annualized,
    where Sigma_t is estimated from returns prior to t using the chosen model.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    idx = weights.index
    names = list(weights.columns)
    out = pd.Series(index=idx, dtype=float, name=f"ex_ante_vol_{model}")

    rets = returns.dropna(how="all")

    for t in idx:
        # training window ends strictly before t
        hist = rets.loc[:t].iloc[:-1].tail(train_window_days)
        if len(hist) < max(21, train_window_days // 6):  # need some data
            out.loc[t] = np.nan
            continue

        if model == "ewma":
            S = ewma_cov(hist, ewma_lambda)
        elif model == "ledoit_wolf":
            S = ledoit_wolf_cov(hist)
        else:
            S = sample_cov(hist)

        w = weights.loc[t].reindex(S.columns).fillna(0.0).values
        try:
            sigma = float(np.sqrt(np.clip(w @ S.values @ w, 0, None)) * np.sqrt(ann_factor))
        except Exception:
            sigma = np.nan
        out.loc[t] = sigma

    return out

# simple covariances (import your implementations if present)
def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    S = returns.cov()
    return S.fillna(0.0)

def ewma_cov(returns: pd.DataFrame, lam: float = 0.97) -> pd.DataFrame:
    X = returns.fillna(0.0).values
    T, N = X.shape
    S = np.zeros((N, N))
    w = 1.0
    denom = 0.0
    for t in range(T):
        x = X[t:t+1].T
        S = lam * S + (1 - lam) * (x @ x.T)
        denom = lam * denom + (1 - lam)
    S = S / max(denom, 1e-12)
    cols = list(returns.columns)
    return pd.DataFrame(S, index=cols, columns=cols)

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Simple LW shrink to identity (ridge-style) for robustness.
    Not the full CC model, but sufficient for diagnostics sensitivity.
    """
    X = returns.dropna().values
    S = np.cov(X, rowvar=False)
    mu = np.trace(S) / S.shape[0]
    F = mu * np.eye(S.shape[0])
    # shrinkage intensity via a crude plug-in (bounded)
    diff = S - F
    num = np.sum(diff**2)
    den = np.sum((S - np.mean(S))**2) + 1e-12
    alpha = np.clip(num / den, 0.0, 1.0)
    S_hat = alpha * F + (1 - alpha) * S
    cols = list(returns.columns)
    return pd.DataFrame(S_hat, index=cols, columns=cols)

# ---- Regime slicing ----

def metrics_by_regime(pnl: pd.Series, market_rets: pd.Series, ann_factor:int) -> pd.DataFrame:
    """
    Slice pnl by simple up/down market regimes (based on market return sign in the same period).
    """
    pnl = pd.Series(pnl).dropna()
    mkt = pd.Series(market_rets).reindex(pnl.index).fillna(0.0)
    regime = pd.Series(np.where(mkt >= 0, "UP", "DOWN"), index=pnl.index, name="regime")
    out = []
    for rg, s in pnl.groupby(regime):
        mu = s.mean() * ann_factor
        sig = s.std() * np.sqrt(ann_factor)
        sharpe = 0.0 if sig == 0 else mu / sig
        out.append({"regime": rg, "ann_return": float(mu), "ann_vol": float(sig), "sharpe": float(sharpe), "periods": int(len(s))})
    return pd.DataFrame(out).set_index("regime")

# ---- Signal–weights alignment ----

def cross_sectional_alignment(signal: pd.DataFrame, weights: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """
    Per date, correlation across names between signal(d) and weights(d).
    """
    idx = weights.index.intersection(signal.index)
    out = []
    for d in idx:
        s = signal.loc[d]
        w = weights.loc[d].reindex(s.index)
        m = s.notna() & w.notna()
        if m.sum() < 3:
            out.append(np.nan); continue
        if method == "spearman":
            rho = s[m].rank().corr(w[m].rank())
        else:
            rho = s[m].corr(w[m])
        out.append(rho)
    return pd.Series(out, index=idx, name=f"align_{method}")

# ---- Plotting helpers (matplotlib optional) ----

def safe_plot(fn):
    def wrapper(*args, **kwargs):
        try:
            import matplotlib.pyplot as plt  # noqa
            return fn(*args, **kwargs)
        except Exception:
            return None
    return wrapper

@safe_plot
def plot_ex_ante_vs_realized(dates, ex_ante: pd.Series, realized: pd.Series, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    ex_ante.plot(label="Ex-ante vol")
    realized.plot(label="Realized vol", alpha=0.8)
    plt.legend(); plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Vol (ann.)")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

@safe_plot
def plot_series(series: pd.Series, out_path: Path, title: str, ylabel: str):
    import matplotlib.pyplot as plt
    plt.figure()
    series.plot()
    plt.title(title); plt.xlabel("Date"); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

@safe_plot
def plot_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, xlabel: str, ylabel: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(df[x].astype(str), df[y])
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

@safe_plot
def plot_onepager(
    equity: pd.Series,
    dd: pd.Series,
    roll_sharpe: pd.Series,
    ex_ante: pd.Series,
    realized: pd.Series,
    beta_exp: pd.Series,
    turnover: pd.Series,
    out_path: Path,
    title: str = "Portfolio diagnostics (one-pager)"
):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    # 1) Equity + drawdown (right axis)
    axs[0,0].plot(equity.index, equity.values, label="Equity")
    ax2 = axs[0,0].twinx()
    ax2.plot(dd.index, dd.values, color="gray", alpha=0.5, label="Drawdown")
    axs[0,0].set_title("Equity (left) & Drawdown (right)")

    # 2) Rolling Sharpe
    axs[0,1].plot(roll_sharpe.index, roll_sharpe.values)
    axs[0,1].set_title(roll_sharpe.name or "Rolling Sharpe")

    # 3) Ex-ante vs Realized vol
    axs[1,0].plot(ex_ante.index, ex_ante.values, label="Ex-ante")
    axs[1,0].plot(realized.index, realized.values, label="Realized")
    axs[1,0].legend(); axs[1,0].set_title("Volatility (annualized)")

    # 4) Market beta exposure
    axs[1,1].plot(beta_exp.index, beta_exp.values)
    axs[1,1].set_title("Market beta exposure (wᵀβ)")

    # 5) Turnover
    axs[2,0].plot(turnover.index, turnover.values)
    axs[2,0].set_title("Turnover per rebalance")

    # Empty panel (kept for future: constraint bind rate or cost sensitivity)
    axs[2,1].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)
