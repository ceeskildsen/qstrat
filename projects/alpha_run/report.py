# projects/alpha_run/report.py
from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import json

import pandas as pd

from src.utils import to_returns, daily_to_period_rf
from src.metrics import compute_metrics
from src.data.prices import get_prices
from src.plots import save_equity_vs_benchmark, save_equity_curve


def _repo_root_from_here() -> Path:
    """
    Resolve the repository root assuming this file lives at:
    <repo>/projects/alpha_run/report.py
    """
    here = Path(__file__).resolve()
    # parents: [alpha_run, projects, <repo>]
    if len(here.parents) >= 3:
        return here.parents[2]
    return Path.cwd()


def summarize_and_save(
    bt: Dict[str, pd.DataFrame | pd.Series],
    cfg: Dict[str, Any],
    *,
    available_tickers: list[str],
    start: str,
    end: str,
    market_ticker: str,
    output_subdir: str = "alpha_to_portfolio",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Compute metrics, build plots (centralized in src/plots.py), and save CSV + config.
    Returns a dict containing pnl, weights, metrics, and out_dir path.
    """
    ROOT = _repo_root_from_here()
    out_dir = ROOT / "outputs" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Unpack backtest outputs ---
    pnl: pd.Series = bt["pnl"]  # periodic returns, index aligned to rebalance freq
    weights: pd.DataFrame = bt["weights"]  # weights per date x ticker

    # --- Risk-free via BIL aligned to portfolio periods ---
    # Get daily BIL prices in [start, end], convert to daily returns,
    # then aggregate to the portfolio's period index.
    bil = get_prices("BIL", start=start, end=end)["BIL"]
    bil_daily = to_returns(bil)
    rf_periodic = daily_to_period_rf(bil_daily, pnl.index)

    # --- Metrics ---
    m_raw = compute_metrics(pnl, weights, ann_factor=cfg["ann_factor"])
    m_excess = compute_metrics(
        pnl, weights, ann_factor=cfg["ann_factor"], risk_free_periodic=rf_periodic
    )

    # --- Console summary (optional for callers) ---
    print("\nPnL (tail):")
    print(pnl.tail().to_string())

    equity = (1 + pnl).cumprod()
    cum_return = equity.iloc[-1] - 1
    print(f"\nTotal cumulative return over backtest: {cum_return:.2%}")

    print("\n=== Performance (RAW, after costs) ===")
    print(f"Periods: {m_raw.get('periods')}")
    print(f"Ann. Return: {m_raw.get('ann_return', float('nan')):.2%}")
    print(f"Ann. Vol:    {m_raw.get('ann_vol', float('nan')):.2%}")
    print(f"Sharpe:      {m_raw.get('sharpe', float('nan')):.2f}")
    print(f"MaxDD:       {m_raw.get('max_drawdown', float('nan')):.2%}")

    print("\n=== Performance (Excess over risk-free) ===")
    print(f"Ann. Excess Return: {m_excess.get('ann_return', float('nan')):.2%}")
    print(f"Sharpe (excess):    {m_excess.get('sharpe', float('nan')):.2f}")

    if save:
        # --- Save CSVs ---
        pnl.to_csv(out_dir / "pnl.csv", header=["pnl"])
        weights.to_csv(out_dir / "weights.csv")

        # --- Save run config ---
        run_cfg = {
            "config": cfg,
            "available_tickers": available_tickers,
            "start": start,
            "end": end,
            "market_ticker": market_ticker,
        }
        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, indent=2)

        # --- Save plots via centralized plot module ---
        # 1) Strategy equity vs risk-free (BIL)
        rf_equity = (1 + rf_periodic).cumprod()
        save_equity_vs_benchmark(
            equity,
            rf_equity,
            out_path=out_dir / "equity_vs_riskfree.png",
            title=f"Equity vs Risk-Free — {cfg.get('label', cfg.get('MODE', 'run'))}",
            labels=("Portfolio", "Risk-free"),
        )

        # 2) Excess equity over BIL
        excess_periodic = (pnl - rf_periodic).astype(float)
        equity_excess = (1 + excess_periodic).cumprod()
        save_equity_curve(
            equity_excess,
            out_path=out_dir / "equity_excess_over_bil.png",
            title=f"Excess equity over BIL — {cfg.get('label', cfg.get('MODE', 'run'))}",
            y_label="NAV",
        )

        print(f"\nSaved alpha outputs to: {out_dir.as_posix()}")

    return {
        "pnl": pnl,
        "weights": weights,
        "metrics_raw": m_raw,
        "metrics_excess": m_excess,
        "out_dir": out_dir,
    }
