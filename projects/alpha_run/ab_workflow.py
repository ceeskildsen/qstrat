# projects/alpha_run/ab_workflow.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Mapping
import json

import pandas as pd

from src.utils import daily_to_period_rf
from src.metrics import compute_metrics
from src.data.prices import get_prices
from projects.alpha_run.data import load_universe
from projects.alpha_run.config import ann_factor_from_freq
from projects.alpha_run.runner import run_backtest
from src.plots import save_multi_equity


def _repo_root_from_here() -> Path:
    """
    Resolve the repository root assuming this file lives at:
    <repo>/projects/alpha_run/ab_workflow.py
    """
    here = Path(__file__).resolve()
    # parents: [alpha_run, projects, <repo>]
    if len(here.parents) >= 3:
        return here.parents[2]
    return Path.cwd()


def load_base_run_context(output_subdir: str = "alpha_to_portfolio") -> Tuple[Dict[str, Any], List[str], str, str, str]:
    """
    Load the most recent alpha_to_portfolio run context to reuse its universe and dates.
    Expects: outputs/<output_subdir>/run_config.json created by projects/alpha_run/report.py
    Returns: (base_cfg, available_tickers, start, end, market_ticker)
    """
    root = _repo_root_from_here()
    cfg_path = root / "outputs" / output_subdir / "run_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        rc = json.load(f)
    base_cfg = rc["config"]
    available_tickers = rc["available_tickers"]
    start = rc["start"]
    end = rc["end"]
    market_ticker = rc["market_ticker"]
    # Ensure ann_factor exists
    base_cfg.setdefault("ann_factor", ann_factor_from_freq(base_cfg.get("FREQ", "W-FRI")))
    return base_cfg, available_tickers, start, end, market_ticker


def make_variant(base: Dict[str, Any], **overrides) -> Dict[str, Any]:
    """Shallow-copy base config and apply overrides."""
    v = dict(base)
    v.update(overrides)
    return v


def _equity_from_pnl(pnl: pd.Series) -> pd.Series:
    return (1.0 + pd.Series(pnl).astype(float)).cumprod()


def _normalize_to_common_start(curves: Mapping[str, pd.Series]) -> Dict[str, pd.Series]:
    """Align to the union index and normalize each curve to 1 at its first valid point."""
    # Union index
    all_idx = None
    for s in curves.values():
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    all_idx = pd.Index(sorted(all_idx)) if all_idx is not None else pd.Index([])
    out = {}
    for label, s in curves.items():
        s2 = pd.Series(s).reindex(all_idx).ffill()
        first_valid = s2.dropna()
        first = first_valid.iloc[0] if not first_valid.empty else 1.0
        denom = first if first != 0 else 1.0
        out[label] = s2 / denom
    return out


def run_ab_variant(
    cfg: Dict[str, Any],
    *,
    available_tickers: List[str],
    start: str,
    end: str,
    market_ticker: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run a single A/B variant and compute metrics (raw and excess over risk-free).
    Returns dict with pnl, weights, metrics_raw, metrics_excess, ann_factor, equity.
    """
    # Load data for the fixed universe/dates
    prices, market, sector_map, _avail = load_universe(available_tickers, start, end, market_ticker)

    # Backtest
    bt = run_backtest(prices, market, sector_map, cfg, debug=debug)
    pnl: pd.Series = bt["pnl"]
    weights: pd.DataFrame = bt["weights"]

    # Risk-free (BIL) --> periodic RF aligned to pnl.index
    bil = get_prices("BIL", start=start, end=end)["BIL"].pct_change().dropna()
    rf_periodic = daily_to_period_rf(bil, pnl.index)

    # Metrics
    annF = cfg.get("ann_factor", ann_factor_from_freq(cfg.get("FREQ", "W-FRI")))
    m_raw = compute_metrics(pnl, weights, ann_factor=annF)
    m_ex = compute_metrics(pnl, weights, ann_factor=annF, risk_free_periodic=rf_periodic)

    return {
        "pnl": pnl,
        "weights": weights,
        "metrics_raw": m_raw,
        "metrics_excess": m_ex,
        "ann_factor": annF,
        "equity": _equity_from_pnl(pnl),
    }


def run_ab_test(
    base_cfg: Dict[str, Any],
    variants: List[Tuple[str, Dict[str, Any]]],
    *,
    out_dir: Path,
    available_tickers: List[str],
    start: str,
    end: str,
    market_ticker: str,
    plot_title: str = "A/B Test – Equity (normalized)",
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Orchestrate A/B test: run each variant, write summary.csv and equity_ab.png.
    Returns (summary_df, curves_dict).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    curves: Dict[str, pd.Series] = {}

    for label, overrides in variants:
        cfg = make_variant(base_cfg, **overrides)
        cfg["label"] = label
        res = run_ab_variant(
            cfg,
            available_tickers=available_tickers,
            start=start,
            end=end,
            market_ticker=market_ticker,
        )
        # Collect equity
        curves[label] = res["equity"]

        # Build one summary row combining raw & excess metrics
        r = {"label": label}
        for k, v in res["metrics_raw"].items():
            r[f"{k}"] = v
        for k, v in res["metrics_excess"].items():
            r[f"excess_{k}"] = v
        rows.append(r)

    # Save summary
    summary = pd.DataFrame(rows)
    # Put label first
    cols = ["label"] + [c for c in summary.columns if c != "label"]
    summary = summary[cols]
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Save normalized equity overlay
    curves_norm = _normalize_to_common_start(curves)
    save_multi_equity(curves_norm, out_dir / "equity_ab.png", title=plot_title, y_label="Normalized NAV")

    return summary, curves
