# projects/alpha_to_portfolio.py
# Ensure repo root is importable when run from /projects
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Quiet TF logs just in case (won't import TF in sklearn/Nyström modes)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from projects.alpha_run import (
    make_config,
    load_universe,
    run_backtest,
    summarize_and_save,
)

# --- Choose a preset here (same options as before) ---
# Options: "MOM_MONTHLY", "MR_WEEKLY", "COMBO_MONTHLY", "GP_MONTHLY", "NYSTROM_MONTHLY"
MODE = "MOM_MONTHLY"

# Optional per-run overrides (can leave empty)
OVERRIDES = {
    # example: "start": "2018-01-01",
    # example: "ny_time_half_life_days": 180,
}


def main():
    # Build configuration (DEFAULTS + PRESET + OVERRIDES)
    cfg = make_config(mode=MODE, overrides=OVERRIDES)

    # ---------- Data ----------
    prices, spy, sector_map, available = load_universe(
        cfg["tickers"], cfg["start"], cfg["end"], cfg["market_ticker"]
    )
    print(f"Universe size: {len(available)} / {len(cfg['tickers'])} usable")

    # ---------- Backtest ----------
    print(f"\n=== Walk-forward backtest ({cfg.get('label', cfg['MODE'])}) ===")
    bt = run_backtest(prices, spy, sector_map, cfg, debug=True)

    # ---------- Report / Save ----------
    summarize_and_save(
        bt, cfg,
        available_tickers=available,
        start=cfg["start"], end=cfg["end"],
        market_ticker=cfg["market_ticker"],
        output_subdir="alpha_to_portfolio",
        save=True,
    )


if __name__ == "__main__":
    main()
