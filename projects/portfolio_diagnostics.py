# projects/portfolio_diagnostics.py
from __future__ import annotations

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

# Output controls
MINIMAL = True                 # keep outputs lean
INCLUDE_ROLLING_SHARPE = True  # include rolling Sharpe even if MINIMAL

from src.utils import to_returns, daily_to_period_rf
from src.data.prices import get_prices
from src.data.sectors import get_sector_map
from src.diagnostics import (
    equity_from_pnl, drawdown, rolling_sharpe,
    compute_turnover_series, compute_betas,
)
from src.plots import (
    save_drawdown_from_equity,
    save_line_series,
    save_constraint_binding,
    save_sector_heatmap,
)

# ---------------------------------------------------------------------------

def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1] if len(here.parents) >= 2 else Path.cwd()

def _load_alpha_outputs(root: Path):
    out_alpha = root / "outputs" / "alpha_to_portfolio"
    pnl = pd.read_csv(out_alpha / "pnl.csv", index_col=0, parse_dates=True)["pnl"]
    weights = pd.read_csv(out_alpha / "weights.csv", index_col=0, parse_dates=True)
    with open(out_alpha / "run_config.json", "r", encoding="utf-8") as f:
        rc = json.load(f)
    cfg = rc["config"]
    start, end = rc["start"], rc["end"]
    market_ticker = rc["market_ticker"]
    ann_factor = cfg.get("ann_factor", 252)
    available_tickers = rc.get("available_tickers", list(weights.columns))
    return pnl, weights, cfg, start, end, market_ticker, ann_factor, available_tickers

def _daily_to_period_returns_df(daily_returns: pd.DataFrame, period_index: pd.Index) -> pd.DataFrame:
    """
    Aggregate daily per-name returns to the supplied period endpoints (e.g., monthly).
    Compounds between endpoints: per = cumprod(1+r)_t / cumprod(1+r)_{t-1} - 1.
    """
    if daily_returns.empty or len(period_index) == 0:
        return pd.DataFrame(index=period_index, columns=daily_returns.columns, dtype=float)
    cum = (1.0 + daily_returns).cumprod()
    aligned = cum.reindex(period_index.union(cum.index)).ffill().reindex(period_index)
    per = aligned.pct_change().fillna(0.0)
    return per

def main() -> None:
    ROOT = _repo_root_from_here()
    OUT = ROOT / "outputs" / "portfolio_diagnostics"
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Load prior run (alpha_to_portfolio) ----
    pnl, weights, cfg, start, end, market_ticker, annF, available = _load_alpha_outputs(ROOT)

    # ---- Equity (not plotted) + Drawdown (plotted) ----
    equity = equity_from_pnl(pnl)
    dd = drawdown(equity)
    dd.to_csv(OUT / "drawdown.csv")
    save_drawdown_from_equity(equity, OUT / "drawdown.png", title="Drawdown")

    # ---- Per-name periodic returns aligned to portfolio periods ----
    per_name_periodic = pd.DataFrame(index=pnl.index)
    try:
        px = get_prices(available, start=start, end=end)
        daily_rets = to_returns(px)  # DataFrame (dates x tickers)
        per_name_periodic = _daily_to_period_returns_df(daily_rets, pnl.index)
        per_name_periodic = per_name_periodic.reindex(columns=weights.columns)
    except Exception as e:
        print(f"[info] per-name periodic returns unavailable; some parts may be skipped: {e}")

    # ---- Beta exposure vs market ----
    try:
        mkt_px = get_prices(market_ticker, start=start, end=end)[market_ticker]
        mkt_daily = to_returns(mkt_px)
        mkt_periodic = daily_to_period_rf(mkt_daily, pnl.index)

        lookback = int(max(6, min(annF, len(per_name_periodic))))
        betas_by_name = compute_betas(per_name_periodic, mkt_periodic, lookback=lookback)

        idx = weights.index.intersection(betas_by_name.index)
        cols = weights.columns.intersection(betas_by_name.columns)
        beta_exp = (weights.loc[idx, cols] * betas_by_name.loc[idx, cols]).sum(axis=1)
        beta_exp.name = "beta_exposure"

        beta_exp.to_csv(OUT / "beta_exposure.csv")
        save_line_series(beta_exp, OUT / "beta_exposure.png", title="Market Beta", y_label="Beta")
    except Exception as e:
        print(f"[info] beta_exposure skipped: {e}")
        betas_by_name = pd.DataFrame()

    # ---- Constraint binding summary ----
    try:
        if betas_by_name.empty:
            raise ValueError("betas_by_name empty")
        from src.diagnostics import constraint_binding_stats
        position_bound = float(cfg.get("max_pos", 0.02))
        gross_limit = float(cfg.get("gross_limit", 1.0))
        beta_limit = float(cfg.get("beta_limit", 0.10))  # label only (engine uses its own cap)
        bind = constraint_binding_stats(
            weights=weights,
            betas_by_date=betas_by_name,
            position_bound=position_bound,
            gross_limit=gross_limit,
            beta_limit=beta_limit,
            eps=1e-6,
        )
        bind.to_csv(OUT / "constraint_binding.csv")

        if isinstance(bind, pd.DataFrame) and {"constraint", "bind_pct"}.issubset(bind.columns):
            summary = (bind[["constraint", "bind_pct"]]
                       .groupby("constraint", as_index=False)["bind_pct"].mean()
                       .sort_values("bind_pct", ascending=False))
            if summary["bind_pct"].max() <= 1.0:
                summary["bind_pct"] *= 100.0
            plot_df = summary
        else:
            num = bind.select_dtypes(include=[np.number, "bool"]).astype(float)
            summary = pd.DataFrame({
                "constraint": num.columns.astype(str),
                "bind_pct": (num.mean(axis=0) * 100.0).values
            }).sort_values("bind_pct", ascending=False)
            plot_df = summary

        summary.to_csv(OUT / "constraint_binding_summary.csv", index=False)
        save_constraint_binding(plot_df, OUT / "constraint_binding_summary.png")
    except Exception as e:
        print(f"[info] constraint_binding skipped: {e}")

    # ---- Rolling Sharpe ----
    if (not MINIMAL) or INCLUDE_ROLLING_SHARPE:
        try:
            rs = rolling_sharpe(pnl, ann_factor=annF, window=12)  # 12 portfolio periods
            rs.to_csv(OUT / "rolling_sharpe.csv")
            save_line_series(rs, OUT / "rolling_sharpe.png",
                             title="Rolling Sharpe", y_label="Sharpe")
        except Exception as e:
            print(f"[info] rolling_sharpe skipped: {e}")

    # ---- Sector exposure — always emit heatmap ----
    try:
        sec_map = get_sector_map(weights.columns).reindex(weights.columns).fillna("Other")
        sectors = pd.unique(sec_map.values)
        exp_df = pd.DataFrame(index=weights.index, columns=sectors, dtype=float)
        for sec in sectors:
            cols = sec_map.index[sec_map.values == sec]
            exp_df[sec] = weights.loc[:, cols].sum(axis=1) if len(cols) else 0.0
        exp_df = exp_df.fillna(0.0)
        exp_df.to_csv(OUT / "sector_exposure.csv")
        save_sector_heatmap(exp_df, OUT / "sector_exposure.png",
                            title="Sector Exposures (heatmap)")
    except Exception as e:
        print(f"[info] sector_exposure heatmap skipped: {e}")

    print(f"Portfolio diagnostics written to: {OUT.as_posix()}")

if __name__ == "__main__":
    main()
