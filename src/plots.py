# src/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------ helpers ------------

def _to_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path = _to_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _as_dt_index(x: pd.Index | Iterable) -> pd.DatetimeIndex:
    """Return tz-naive DatetimeIndex; tz-aware → UTC → naive."""
    if isinstance(x, pd.DatetimeIndex):
        return x.tz_convert("UTC").tz_localize(None) if x.tz is not None else x
    di = pd.to_datetime(list(x), utc=True, errors="coerce")
    return di.tz_localize(None)

def _normalize_first_valid(s: pd.Series) -> pd.Series:
    v = pd.Series(s).dropna()
    if v.empty:
        return pd.Series(s)
    base = float(v.iloc[0]) or 1.0
    out = pd.Series(s) / base
    out.index = pd.Series(s).index
    return out

def _to_equity(s: pd.Series) -> pd.Series:
    """
    If it looks like returns, cumprod to equity.
    If it's already equity OR a drawdown/underwater series (<= 0), return as-is.
    """
    s = pd.Series(s).astype(float).sort_index()
    s.index = _as_dt_index(s.index)
    sn = s.dropna()
    if sn.empty:
        return s

    # --- IMPORTANT: do NOT transform true drawdown / underwater series ---
    # Drawdown is ≤ 0 with max around 0. Underwater is in [0,1] but you
    # won't call _to_equity for it except by mistake; the key bug was with true DD.
    if float(sn.max()) <= 1e-12:
        return s  # already a drawdown-like series (≤ 0)

    # Heuristic: returns are small numbers around 0; equity is O(1+) and positive.
    q05, q95 = sn.quantile(0.05), sn.quantile(0.95)
    m = sn.mean()
    is_returns = (q95 < 0.5) and (q05 > -0.5) and (abs(m) < 0.2)

    return (1.0 + s.fillna(0.0)).cumprod() if is_returns else s


# ------------ generic lines ------------

def save_equity_curve(equity: pd.Series, out_path, title="Equity", y_label="NAV") -> None:
    e = _to_equity(equity)  # safe even if already equity
    e.index = _as_dt_index(e.index)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(e.index, e.values)
    ax.set_title(title); ax.set_ylabel(y_label); ax.grid(True, alpha=0.3)
    _save(fig, out_path)

def save_line_series(series: pd.Series, out_path, title="", y_label="") -> None:
    s = pd.Series(series).sort_index()
    s.index = _as_dt_index(s.index)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s.index, s.values)
    ax.set_title(title); ax.set_ylabel(y_label); ax.grid(True, alpha=0.3)
    _save(fig, out_path)

def save_multi_equity(
    curves: Dict[str, pd.Series],
    out_path,
    title: str = "Equity (normalized)",
    y_label: str = "Normalized NAV",
    normalize: bool = True,
) -> None:
    """
    Plot multiple time series. Each series is converted to equity if it
    looks like returns; true drawdown (≤0) is detected and left unchanged.
    """
    clean: Dict[str, pd.Series] = {}
    idx_union: Optional[pd.DatetimeIndex] = None

    for label, s in curves.items():
        eq = _to_equity(pd.Series(s))
        if eq.dropna().empty:
            print(f"[info] plot: dropping empty series '{label}'")
            continue
        eq.index = _as_dt_index(eq.index)
        clean[label] = eq
        idx_union = eq.index if idx_union is None else idx_union.union(eq.index)

    if not clean:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center"); ax.axis("off")
        _save(fig, out_path)
        return

    idx_union = idx_union.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, eq in clean.items():
        y = eq.reindex(idx_union).astype(float).ffill().bfill()
        if normalize:
            y = _normalize_first_valid(y)
        ax.plot(idx_union, y.values, label=label)

    ax.set_title(title); ax.set_ylabel(y_label); ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    _save(fig, out_path)

def save_drawdown_from_equity(equity: pd.Series, out_path, title="Drawdown", y_label="Drawdown") -> None:
    e = _to_equity(equity); e.index = _as_dt_index(e.index)
    dd = (e / e.cummax()) - 1.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dd.index, dd.values)
    ax.set_title(title); ax.set_ylabel(y_label); ax.grid(True, alpha=0.3)
    _save(fig, out_path)

def save_equity_vs_benchmark(
    equity: pd.Series,
    benchmark: pd.Series,
    out_path,
    title: str = "Equity vs Risk-Free",
    y_label: str = "NAV",
    equity_label: str = "Portfolio",
    benchmark_label: str = "Risk-free",
    normalize: bool = True,
    labels: Optional[Tuple[str, str]] = None,  # backward-compat
) -> None:
    """Overlay portfolio equity and a benchmark. Accepts legacy labels=(strat, bench)."""
    if labels is not None and len(labels) == 2:
        equity_label, benchmark_label = labels  # type: ignore[misc]
    e = _to_equity(equity); b = _to_equity(benchmark)
    idx = _as_dt_index(e.index.union(b.index))
    e = e.reindex(idx).ffill().bfill(); b = b.reindex(idx).ffill().bfill()
    if normalize:
        e, b = _normalize_first_valid(e), _normalize_first_valid(b)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(idx, e.values, label=equity_label)
    ax.plot(idx, b.values, label=benchmark_label)
    ax.set_title(title); ax.set_ylabel(y_label); ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    _save(fig, out_path)


# ------------ IC utilities ------------

def save_ic_timeseries(ic_series: pd.Series, out_path, title="IC (timeseries)") -> None:
    s = pd.Series(ic_series).sort_index(); s.index = _as_dt_index(s.index)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s.index, s.values); ax.axhline(0.0, lw=1)
    ax.set_title(title); ax.set_ylabel("IC"); ax.grid(True, alpha=0.3)
    _save(fig, out_path)

def save_ic_hist(ic_series: pd.Series, out_path, title="IC histogram", bins=40) -> None:
    s = pd.Series(ic_series).replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s.values, bins=bins); ax.set_title(title)
    ax.set_xlabel("IC"); ax.set_ylabel("Count"); ax.grid(True, alpha=0.3)
    _save(fig, out_path)

def save_icir_heatmap(
    df: pd.DataFrame,
    *,
    row: str,
    col: str,
    value: str,
    out_path,
    title="IC IR (annualized)",
    annotate=True,
) -> None:
    pivot = df.pivot_table(index=row, columns=col, values=value, aggfunc="mean")
    r_labels, c_labels = list(pivot.index), list(pivot.columns)
    M = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, aspect="auto", origin="lower")
    ax.set_title(title); ax.set_xlabel(str(col)); ax.set_ylabel(str(row))
    ax.set_xticks(range(len(c_labels))); ax.set_xticklabels(c_labels)
    ax.set_yticks(range(len(r_labels))); ax.set_yticklabels(r_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    _save(fig, out_path)


# ------------ bars & heatmap ------------

def save_bar_df(
    df: pd.DataFrame, *, x: str, y: str, out_path, title="", rotate_xticks: int = 0
) -> None:
    d = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(d[x].astype(str).values, d[y].astype(float).values)
    ax.set_title(title); ax.set_xlabel(x); ax.set_ylabel(y)
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)

def save_constraint_binding(
    summary: pd.DataFrame, out_path, title="Constraint Binding (avg % of time active)"
) -> None:
    if "constraint" not in summary or "bind_pct" not in summary:
        raise ValueError("summary must have columns ['constraint','bind_pct']")
    df = summary.sort_values("bind_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["constraint"].astype(str).values, df["bind_pct"].astype(float).values)
    ax.set_title(title); ax.set_ylabel("% of periods")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)

def save_sector_heatmap(
    exposures: pd.DataFrame, out_path, title="Sector exposures (heatmap)", resample_to: Optional[str] = "ME"
) -> None:
    if exposures is None or exposures.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No sector exposure data", ha="center", va="center")
        ax.axis("off")
        _save(fig, out_path)
        return
    X = exposures.copy()
    X.index = _as_dt_index(X.index)
    if resample_to:
        X = X.resample(resample_to).mean()
    vmax = float(max(np.nanmax(np.abs(X.values)) if np.isfinite(X.values).any() else 1.0, 1e-6))
    vmin = -vmax
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(X.values.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.set_xlabel("date"); ax.set_ylabel("sector")
    ax.set_yticks(range(len(X.columns))); ax.set_yticklabels([str(c) for c in X.columns])
    n = len(X.index)
    if n > 1:
        step = max(1, n // 12)
        locs = list(range(0, n, step))
        ax.set_xticks(locs)
        ax.set_xticklabels(
            [pd.Timestamp(d).strftime("%Y-%m") for d in X.index[locs]],
            rotation=45,
            ha="right",
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_path)
