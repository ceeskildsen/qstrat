# projects/ab_test.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd  # NEW: needed for explicit drawdown calc

# Make repo importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.alpha_run.ab_workflow import (
    load_base_run_context,
    make_variant,
    run_ab_test,
)
from src.plots import save_multi_equity  # used for both equity & drawdown overlays


def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    # parents: [projects, <repo>]
    if len(here.parents) >= 2:
        return here.parents[1]
    return Path.cwd()


def main() -> None:
    # Reuse last alpha_to_portfolio run context (universe/dates/market/config)
    base_cfg, available, start, end, market_ticker = load_base_run_context()

    # === A/B Variants ===
    VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
        ("A: baseline", {}),
        ("B: baseline without pca neutralize", {"pca_neutralize": False}),
        ("C: baseline without mean reversion", {"overlay_mr": False}),
        # (examples you can swap back in)
        # ("B: gross limit 5, beta limit 0.05", {"gross_limit": 5, "beta_limit": 0.05}),
        # ("C: gross limit 4, beta limit 0.04", {"gross_limit": 4, "beta_limit": 0.04}),
    ]

    root = _repo_root_from_here()
    out_dir = root / "outputs" / "ab_test"

    # Run and save summary.csv + equity_ab.png (normalized overlay handled inside)
    summary, curves = run_ab_test(
        base_cfg,
        VARIANTS,
        out_dir=out_dir,
        available_tickers=available,
        start=start,
        end=end,
        market_ticker=market_ticker,
        plot_title="A/B Test – Equity (normalized)",
    )

    # --- Drawdown overlay (TRUE drawdown: E/peak - 1, matches portfolio_diagnostics) ---
    # Compute drawdown here explicitly instead of using src.diagnostics.drawdown (which returns 0..1 underwater).
    dd_curves: Dict[str, pd.Series] = {}
    for label, eq in curves.items():
        s = pd.Series(eq).astype(float).sort_index()
        # make index tz-naive to be safe with plotting
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce").tz_localize(None)
        dd = (s / s.cummax()) - 1.0    # 0 at peaks, negative when underwater
        dd_curves[label] = dd

    save_multi_equity(
        dd_curves,
        out_dir / "drawdown_ab.png",
        title="A/B Test – Drawdown",
        y_label="Drawdown",
        normalize=False,   # IMPORTANT: keep raw (negative) drawdown values
    )

    # Light console view
    print("\nA/B summary:")
    print(summary.head().to_string(index=False))
    print(f"\nSaved figures to: {out_dir.as_posix()}")
    print(" - equity_ab.png")
    print(" - drawdown_ab.png")
    print(" - summary.csv")


if __name__ == "__main__":
    main()
