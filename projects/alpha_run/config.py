# projects/alpha_run/config.py
from __future__ import annotations

from typing import Dict, Any


def ann_factor_from_freq(freq: str) -> int:
    f = (freq or "").upper()
    if f.startswith("W"):
        return 52
    if f in ("M", "ME", "MS", "BM", "BME"):
        return 12
    return 252


# === Defaults cloned from your previous alpha_to_portfolio ===
DEFAULTS: Dict[str, Any] = {
    # Universe & dates
    "tickers": [
        "AAPL","MSFT","AMZN","GOOGL","META","NVDA","AVGO","TSLA","JPM","UNH",
        "V","MA","HD","KO","PEP","MRK","ABBV","COST","PFE","CVX",
        "XOM","WMT","CSCO","ORCL","ACN","ADBE","CRM","NFLX","TXN","NKE",
        "LIN","AMD","QCOM","TMO","ABT","DHR","MCD","INTC","CMCSA","VZ",
        "PM","UPS","CAT","HON","IBM","LOW","AMAT","SBUX","BKNG","GILD"
    ],
    "start": "2020-01-01",
    "end":   "2025-07-31",
    "market_ticker": "SPY",

    # Cadence & windows
    "rebalance_freq": "ME",
    "train_window_days": 504,
    "beta_lookback": 252,

    # Signal (overridden by PRESETS)
    "signal_type": "mom_12_1",
    "sector_neutral_signal": False,

    # Sector neutrality (optimizer-level)
    "sector_neutral": False,
    "sector_neutral_tol": None,

    # PCA neutrality (optimizer-level)
    "pca_neutralize": True,
    "pca_k": 2,
    "pca_neutral_tol": 0.03,

    # Risk model
    "risk_model": "lw",
    "ewma_lambda": 0.97,

    # Constraints
    "dollar_neutral": True,
    "long_only": False,
    "position_bound": 0.10,
    "gross_limit": 2.0,
    "beta_neutralize": True,
    "beta_limit": 0.02,

    # Risk budget
    "risk_budget_annual": 0.20,
    "risk_aversion": 5.0,

    # Costs & gating
    "transaction_cost_bps": 5.0,
    "min_name_change": 0.002,
    "min_port_change": 0.05,
    "benefit_cost_buffer": 1,
    "step_fraction": 1,

    # Momentum overlay MR (for pure momentum only)
    "overlay_mr": True,
    "overlay_mr_weight": 0.30,
    "mr_lookback": 3,
    "mr_lambda": 0.3,
    "mr_beta_lookback": 126,

    # Filters / crash protection
    "market_filter": False,
    "mkt_fast_win": 63, "mkt_slow_win": 252, "risk_off_shrink": 0.5,
    "crash_protect": True,
    "crash_fast_win": 21, "crash_slow_win": 63,
    "crash_fast_th": 0.07, "crash_slow_th": 0.12, "crash_shrink_neg": 0.5,

    # Ex-ante vol targeting
    "vol_target_ex_ante": True,
    "vol_scale_bounds": (0.7, 1.3),

    # COMBO-only
    "combo_mr_weight": 0.20,

    # --- Sklearn GP knobs (exact) ---
    "gp_backend": "sklearn",
    "gp_train_lookback_days": 100,
    "gp_uncertainty_sizing": True,
    "gp_sigma_floor": 0.25,
    "gp_n_restarts": 0,
    "gp_alpha": 3e-5,
    "gp_max_train_points": 200,
    "gp_refit_every": 1,
    "gp_float32": True,
    "gp_per_date_name_frac": 0.4,
    "gp_label_mode": "zmad",         # 'none' | 'zmad' | 'rank'

    # --- Nyström KRR knobs ---
    "ny_kernel": "rbf",
    "ny_components": 200,
    "ny_alpha": 1e-2,
    "ny_gamma": None,              # None => median heuristic
    "ny_random_state": 0,
    "ny_time_half_life_days": 180,   # time-decay half-life (days)
}

PRESETS: Dict[str, Any] = {
    "MR_WEEKLY": {
        "label": "weekly • mean-reversion (mr_5)",
        "signal_type": "mr_5",
        "rebalance_freq": "W-FRI",
        "transaction_cost_bps": 10.0,
        "overlay_mr": True,
    },
    "MOM_MONTHLY": {
        "label": "monthly • momentum (12-1)",
        "signal_type": "mom_12_1",
        "rebalance_freq": "ME",
        "transaction_cost_bps": 5.0,
        "overlay_mr": True,
    },
    "COMBO_MONTHLY": {
        "label": "monthly • combo (MOM + 0.20·MR)",
        "signal_type": "combo_mom_mr",
        "rebalance_freq": "ME",
        "transaction_cost_bps": 5.0,
        "overlay_mr": True,
        "combo_mr_weight": 0.20,
        "mr_lookback": 3,
        "mr_beta_lookback": 126,
    },
    "GP_MONTHLY": {
        "label": "monthly • GP (sklearn, fast)",
        "signal_type": "gp_xs",
        "gp_backend": "sklearn",
        "rebalance_freq": "ME",
        "transaction_cost_bps": 5.0,
        "overlay_mr": True,
    },
    "NYSTROM_MONTHLY": {
        "label": "monthly • Nyström KRR (fast)",
        "signal_type": "gp_xs",
        "gp_backend": "nystrom_krr",
        "rebalance_freq": "ME",
        "transaction_cost_bps": 5.0,
        "overlay_mr": False,
    },
}


def make_config(mode: str = "MOM_MONTHLY", overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a full config dict from a PRESET + DEFAULTS + optional overrides."""
    if mode not in PRESETS:
        raise ValueError(f"Unknown MODE '{mode}'. Choose one of: {list(PRESETS)}")
    cfg = {**DEFAULTS, **PRESETS[mode]}
    if overrides:
        cfg.update(overrides)
    cfg["MODE"] = mode
    cfg["ann_factor"] = ann_factor_from_freq(cfg["rebalance_freq"])
    return cfg
