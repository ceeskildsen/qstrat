# projects/alpha_run/runner.py
from __future__ import annotations

from typing import Dict, Any
import pandas as pd

# Import the split backtest. Keep a fallback to the legacy single file.
try:
    from src.backtest.engine import backtest_alpha_to_portfolio
except Exception:  # pragma: no cover
    from src.backtest import backtest_alpha_to_portfolio  # type: ignore


def run_backtest(
    prices: pd.DataFrame,
    spy: pd.Series,
    sector_map: dict[str, str] | pd.Series | None,
    cfg: Dict[str, Any],
    *,
    debug: bool = True,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Thin wrapper that passes your config into the backtest engine.

    Notes
    -----
    - `mr_lambda` is treated as an alias for `overlay_mr_weight`.
      If both are present, `mr_lambda` wins.
    """
    # Backward/forward compatible mapping for MR overlay intensity
    overlay_mr_weight = cfg.get("mr_lambda", cfg.get("overlay_mr_weight", 0.20))

    return backtest_alpha_to_portfolio(
        prices,

        # cadence & window
        rebalance_freq=cfg["rebalance_freq"],
        train_window_days=cfg["train_window_days"],

        # signal
        signal_type=cfg["signal_type"],
        combo_mr_weight=cfg.get("combo_mr_weight", 0.20),
        gp_label_mode=cfg.get("gp_label_mode", "none"),

        # GP knobs / backends
        gp_backend=cfg.get("gp_backend", "sklearn"),
        gp_train_lookback_days=cfg["gp_train_lookback_days"],
        gp_uncertainty_sizing=cfg["gp_uncertainty_sizing"],
        gp_sigma_floor=cfg["gp_sigma_floor"],
        gp_n_restarts=cfg["gp_n_restarts"],
        gp_alpha=cfg["gp_alpha"],
        gp_max_train_points=cfg["gp_max_train_points"],
        gp_refit_every=cfg["gp_refit_every"],
        gp_float32=cfg["gp_float32"],
        gp_per_date_name_frac=cfg["gp_per_date_name_frac"],

        # Nyström params
        ny_kernel=cfg["ny_kernel"],
        ny_components=cfg["ny_components"],
        ny_alpha=cfg["ny_alpha"],
        ny_gamma=cfg["ny_gamma"],
        ny_random_state=cfg["ny_random_state"],
        ny_time_half_life_days=cfg["ny_time_half_life_days"],

        # risk model
        risk_model=cfg["risk_model"],
        ewma_lambda=cfg["ewma_lambda"],

        # constraints
        dollar_neutral=cfg["dollar_neutral"],
        long_only=cfg["long_only"],
        position_bound=cfg["position_bound"],
        gross_limit=cfg["gross_limit"],

        # beta / market
        market_prices=spy,
        beta_neutralize=cfg["beta_neutralize"],
        beta_limit=cfg["beta_limit"],
        beta_lookback=cfg["beta_lookback"],

        # sector neutrality
        sector_neutral=cfg["sector_neutral"],
        sector_neutral_tol=cfg["sector_neutral_tol"],
        sector_map=sector_map,

        # PCA neutrality
        pca_neutralize=cfg["pca_neutralize"],
        pca_k=cfg["pca_k"],
        pca_neutral_tol=cfg["pca_neutral_tol"],

        # risk budget & aversion
        risk_budget_annual=cfg["risk_budget_annual"],
        risk_aversion=cfg["risk_aversion"],

        # costs & gating
        transaction_cost_bps=cfg["transaction_cost_bps"],
        min_name_change=cfg["min_name_change"],
        min_port_change=cfg["min_port_change"],
        benefit_cost_buffer=cfg["benefit_cost_buffer"],
        step_fraction=cfg["step_fraction"],

        # signal scaling / sector soft neutrality
        sector_neutral_signal=cfg["sector_neutral_signal"],

        # MR knobs
        overlay_mr=cfg["overlay_mr"],
        overlay_mr_weight=overlay_mr_weight,  # <- accepts cfg["mr_lambda"] if provided
        mr_lookback=cfg["mr_lookback"],
        mr_beta_lookback=cfg["mr_beta_lookback"],

        # filters / crash
        market_filter=cfg["market_filter"],
        mkt_fast_win=cfg["mkt_fast_win"],
        mkt_slow_win=cfg["mkt_slow_win"],
        risk_off_shrink=cfg["risk_off_shrink"],
        crash_protect=cfg["crash_protect"],
        crash_fast_win=cfg["crash_fast_win"],
        crash_slow_win=cfg["crash_slow_win"],
        crash_fast_th=cfg["crash_fast_th"],
        crash_slow_th=cfg["crash_slow_th"],
        crash_shrink_neg=cfg["crash_shrink_neg"],

        # vol targeting
        vol_target_ex_ante=cfg["vol_target_ex_ante"],
        vol_scale_bounds=cfg["vol_scale_bounds"],

        debug=debug,
    )
