from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List

from src.utils import to_returns
from src.optimizer import mean_variance_opt

from .helpers import build_rebalance_index, aggregate_period_returns, momentum_12_1_scores, winsorize
from .alpha_features import build_features_for_date, build_train_set
from .alpha_models_rules import (
    momentum_mu, mr_mu, combo_mom_mr_mu,
    overlay_mr as overlay_mr_fn,
    finalize_mu as finalize_mu_fn,
)
from .alpha_models_ml import (
    sklearn_gp_fit_predict,
    nystrom_krr_fit_predict,
    _subsample_balanced,  # used to cap train size consistently
)
from .constraints import build_sector_matrix, build_pca_matrix, compute_beta_vector
from .risk_cov import choose_cov
from .cooldown import compute_cooldown_params

def backtest_alpha_to_portfolio(
    prices: pd.DataFrame,
    *,
    # cadence & window
    rebalance_freq: str = "ME",
    train_window_days: int = 504,
    # signal
    signal_type: str = "mom_12_1",
    combo_mr_weight: float = 0.20,
    gp_label_mode: str = "none",
    # === GP config (shared) ===
    gp_backend: str = "sklearn",          # "sklearn" | "nystrom_krr"
    gp_train_lookback_days: int = 540,
    gp_uncertainty_sizing: bool = True,
    gp_sigma_floor: float = 0.15,
    gp_n_restarts: int = 0,               # sklearn only
    # sklearn / Nyström speed knobs
    gp_alpha: float = 1e-6,
    gp_max_train_points: Optional[int] = 600,
    gp_refit_every: int = 2,
    gp_float32: bool = True,
    gp_per_date_name_frac: Optional[float] = None,
    # === Nyström KRR knobs ===
    ny_kernel: str = "rbf",
    ny_components: int = 200,
    ny_alpha: float = 1e-2,
    ny_gamma: Optional[float] = None,
    ny_random_state: int = 0,
    ny_time_half_life_days: Optional[float] = None,
    # risk model
    risk_model: str = "ewma",
    ewma_lambda: float = 0.97,
    # basic constraints
    dollar_neutral: bool = True,
    long_only: bool = False,
    position_bound: float = 0.10,
    gross_limit: float | None = 2.0,
    # beta / market
    market_prices: pd.Series | None = None,
    beta_neutralize: bool = True,
    beta_limit: float | None = 0.02,
    beta_lookback: int = 252,
    # sector neutrality
    sector_neutral: bool = False,
    sector_neutral_tol: float | None = None,
    sector_map: Dict[str, str] | pd.Series | None = None,
    sector_soft_row_norm: bool = True,
    # PCA factor neutrality
    pca_neutralize: bool = False,
    pca_k: int = 2,
    pca_neutral_tol: float | None = None,
    pca_equalize_risk: bool = False,
    # risk budget (annual) -> converted to daily internally
    risk_budget_annual: float | None = 0.20,
    risk_aversion: float = 5.0,
    # costs & trade gating
    transaction_cost_bps: float = 5.0,
    min_name_change: float = 0.002,
    min_port_change: float = 0.05,
    benefit_cost_buffer: float = 1.0,
    step_fraction: float = 1.0,
    # feature scaling
    standardize_mu: bool = True,
    winsor_mu_k: float = 3.0,
    # sector-neutral SIGNAL (soft)
    sector_neutral_signal: bool = False,
    # MR overlay (only applies to pure momentum path)
    overlay_mr: bool = False,
    overlay_mr_weight: float = 0.15,
    mr_lookback: int = 3,
    mr_beta_lookback: int = 126,
    # filters
    market_filter: bool = False,
    mkt_fast_win: int = 63,
    mkt_slow_win: int = 252,
    risk_off_shrink: float = 0.5,
    # crash protect
    crash_protect: bool = False,
    crash_fast_win: int = 21,
    crash_slow_win: int = 63,
    crash_fast_th: float = 0.07,
    crash_slow_th: float = 0.12,
    crash_shrink_neg: float = 0.5,
    # ex-ante vol targeting
    vol_target_ex_ante: bool = False,
    vol_scale_bounds: tuple[float, float] = (0.5, 1.5),
    # DEBUG
    debug: bool = False,
) -> dict[str, pd.DataFrame | pd.Series]:

    prices = prices.dropna(how="all").sort_index()
    rets = to_returns(prices)
    tickers = list(prices.columns)

    DAILY_FACTOR = 252
    rb_daily = float(risk_budget_annual) / np.sqrt(DAILY_FACTOR) if risk_budget_annual is not None else None

    # Rebalance dates
    rebal_dates = build_rebalance_index(prices, rets, rebalance_freq)
    if len(rebal_dates) < 2:
        first = prices.index.min() if len(prices.index) else "NA"
        last  = prices.index.max() if len(prices.index) else "NA"
        raise ValueError(
            f"Not enough rebalance points for freq='{rebalance_freq}'. "
            f"price_rows={len(prices)}, return_rows={len(rets)}, "
            f"first={first}, last={last}. "
            "This usually means the price download was truncated by timeouts."
        )

    # Market series
    spy_prices = None
    spy_rets = None
    if market_prices is not None:
        spy_prices = market_prices.dropna().sort_index()
        spy_rets = spy_prices.pct_change(fill_method=None).dropna()

    # Risk-off filter (MA cross on market)
    risk_off_mask = pd.Series(False, index=rebal_dates)
    if market_filter and (spy_prices is not None):
        fast = spy_prices.rolling(mkt_fast_win, min_periods=1).mean()
        slow = spy_prices.rolling(mkt_slow_win, min_periods=1).mean()
        ok = (fast > slow)
        risk_off_mask = ~ok.reindex(rebal_dates, method="ffill").fillna(False)

    weights_records: list[tuple[pd.Timestamp, pd.Series]] = []
    pnl_records: list[tuple[pd.Timestamp, float]] = []

    w_prev = pd.Series(0.0, index=tickers)
    cost_rate = transaction_cost_bps / 10000.0

    # Sector matrix (once, on full ticker set)
    sector_A = None
    if sector_neutral and (not long_only) and (sector_map is not None):
        sector_A = build_sector_matrix(tickers, sector_map)
        if (sector_A is not None) and (sector_neutral_tol is not None) and sector_soft_row_norm:
            norms = np.linalg.norm(sector_A, axis=1, keepdims=True)
            norms = np.where(norms <= 1e-12, 1.0, norms)
            sector_A = sector_A / norms

    # ML model warm states
    prev_gp_state: Optional[dict] = None
    prev_krr_state: Optional[dict] = None
    gp_fit_counter = 0
    printed_gp_header = False

    for i, t in enumerate(rebal_dates):
        rets_train = rets.loc[:t].tail(train_window_days)
        if len(rets_train) < 60:
            if debug: print(f"[{t.date()}] skip (too few train obs): {len(rets_train)}")
            continue
        prices_train = prices.loc[rets_train.index]

        # ---------- μ (alpha) ----------
        st = (signal_type or "").lower()

        if st == "mom_12_1":
            mu = momentum_mu(prices_train)

        elif st.startswith("mr"):
            try:
                lk = int(st.split("_")[1])
            except Exception:
                lk = mr_lookback
            mu = mr_mu(rets_train, spy_rets, lk, mr_beta_lookback)

        elif st == "combo_mom_mr":
            mu = combo_mom_mr_mu(prices_train, rets_train, spy_rets, mr_lookback, mr_beta_lookback, combo_mr_weight)

        elif st in ("gp_xs", "gpflow_xs"):  # gpflow_xs aliased to gp_xs backends we keep
            X_train, y_train, dates_train = build_train_set(
                prices, rets, spy_rets, rebal_dates, t,
                train_lookback_days=gp_train_lookback_days,
                beta_lookback=beta_lookback,
                mr_lookback=mr_lookback,
                per_date_name_frac=gp_per_date_name_frac,
                label_mode=gp_label_mode,
            )
            feats_now = build_features_for_date(t, prices, rets, spy_rets, beta_lookback, mr_lookback)
            X_curr = feats_now[["z_mom", "z_mr", "z_idvol", "z_beta", "tau"]].values

            if gp_float32:
                X_train = X_train.astype(np.float32, copy=False)
                y_train = y_train.astype(np.float32, copy=False)
                X_curr  = X_curr.astype(np.float32,  copy=False)

            mu_hat = None
            std_hat = None

            if X_train.shape[0] >= 50:
                backend = gp_backend.lower()
                if not printed_gp_header and debug:
                    print(f"[{t.date()}] GP backend={backend}, label_mode={gp_label_mode}, ny_hl={ny_time_half_life_days}")
                    printed_gp_header = True

                Xs, ys = _subsample_balanced(X_train, y_train, gp_max_train_points, tail_frac=0.25, rng=42)

                if backend == "nystrom_krr":
                    need_refit = (prev_krr_state is None) or (gp_refit_every <= 1) or (gp_fit_counter % int(max(1, gp_refit_every)) == 0)
                    mu_hat, std_hat, prev_krr_state = nystrom_krr_fit_predict(
                        Xs, ys, X_curr,
                        prev_state=prev_krr_state,
                        refit=need_refit,
                        kernel=ny_kernel,
                        n_components=int(ny_components),
                        alpha=float(ny_alpha),
                        gamma=ny_gamma,
                        random_state=int(ny_random_state),
                        dates_train=dates_train,
                        as_of=t,
                        time_half_life_days=ny_time_half_life_days,
                    )
                    gp_fit_counter += 1
                else:  # sklearn exact GP
                    need_refit = (prev_gp_state is None) or (gp_refit_every <= 1) or (gp_fit_counter % int(max(1, gp_refit_every)) == 0)
                    mu_hat, std_hat, prev_gp_state = sklearn_gp_fit_predict(
                        Xs, ys, X_curr,
                        prev_state=prev_gp_state,
                        refit=need_refit,
                        n_restarts=int(max(0, gp_n_restarts)),
                        alpha=float(gp_alpha),
                    )
                    gp_fit_counter += 1

                if gp_uncertainty_sizing and (std_hat is not None):
                    eff_std = np.maximum(std_hat, float(gp_sigma_floor))
                    score = mu_hat / (np.sqrt(eff_std**2 + 1e-12))
                else:
                    score = mu_hat

                mu = pd.Series(score, index=feats_now.index).astype(float)
            else:
                mu = momentum_12_1_scores(prices_train, gap_days=21, lookback_days=252).reindex(rets_train.columns).fillna(0.0)
        else:
            raise ValueError(f"Unknown signal_type '{signal_type}'.")

        mu = mu.reindex(tickers).fillna(0.0)

        # Optional MR overlay (only on pure momentum)
        if overlay_mr and (signal_type.lower() == "mom_12_1") and (spy_rets is not None):
            mu = overlay_mr_fn(mu, rets_train, spy_rets, mr_lookback, mr_beta_lookback, overlay_mr_weight, standardize_mu)

        # Market filter shrink
        if market_filter and risk_off_mask.loc[t]:
            mu = mu * float(risk_off_shrink)

        # Finalize μ (sector-neutralize signal, standardize, winsorize)
        mu = finalize_mu_fn(mu, sector_map, sector_neutral_signal=sector_neutral_signal,
                            standardize_mu=standardize_mu, winsor_mu_k=winsor_mu_k)

        # Crash protect (mask strong short signals in melt-up regimes)
        if (market_prices is not None) and crash_protect and (spy_prices.index.min() <= t):
            s = spy_prices.loc[:t]
            if len(s) >= max(crash_slow_win + 1, crash_fast_win + 1):
                fast_ret = float(s.iloc[-1] / s.iloc[-1 - crash_fast_win] - 1.0)
                slow_ret = float(s.iloc[-1] / s.iloc[-1 - crash_slow_win] - 1.0)
                if (fast_ret >= crash_fast_th) and (slow_ret >= crash_slow_th):
                    neg = mu < 0
                    if neg.any():
                        mu.loc[neg] = mu.loc[neg] * float(crash_shrink_neg)

        # ---------- COOLDOWN (pre-optimizer) ----------
        cd = compute_cooldown_params(pnl_records)
        if cd["is_cool"] and debug:
            print(f"[{t.date()}] COOLDOWN on: dd_recent={cd['dd_recent']:.2%}, sum_recent={cd['sum_recent']:+.2%}, near_peak={cd['near_peak']}")
        if cd["is_cool"]:
            mu = mu * cd["mu_scale"]

        opt_position_bound = position_bound * cd["position_bound_mult"]
        opt_gross_limit    = (gross_limit * cd["gross_limit_mult"]) if (gross_limit is not None) else None
        eff_risk_aversion  = risk_aversion * cd["risk_aversion_mult"]

        eff_step_fraction    = step_fraction * cd["step_fraction_mult"]
        eff_min_port_change  = min_port_change * cd["min_port_change_mult"]
        eff_benefit_cost_buf = benefit_cost_buffer * cd["benefit_cost_buffer_mult"]
        eff_risk_budget_ann  = (risk_budget_annual * cd["risk_budget_annual_mult"]) if (risk_budget_annual is not None) else None

        # ---------- risk model ----------
        Sigma = choose_cov(rets_train, risk_model=risk_model, ewma_lambda=ewma_lambda)
        Sigma = Sigma.reindex(index=tickers, columns=tickers).fillna(0.0)

        # ---------- PCA factor neutrality ----------
        pca_A = None
        if pca_neutralize and (not long_only) and (pca_k is not None) and (pca_k > 0):
            pca_A = build_pca_matrix(Sigma, pca_k, pca_equalize_risk)

        # ---------- beta vector ----------
        beta_vec = None
        if beta_neutralize and (market_prices is not None) and (spy_rets is not None):
            beta_vec = compute_beta_vector(rets, spy_rets, t, beta_lookback, tickers)

        # ---------- optimize ----------
        try:
            beta_lim = float(beta_limit) if (beta_vec is not None and beta_limit is not None) else None
            w_target = mean_variance_opt(
                mu, Sigma,
                risk_aversion=eff_risk_aversion,
                dollar_neutral=dollar_neutral,
                long_only=long_only,
                position_bound=opt_position_bound,
                gross_limit=opt_gross_limit,
                beta=beta_vec,
                beta_limit=beta_lim,
                risk_budget_daily=(float(eff_risk_budget_ann) / np.sqrt(252.0)) if eff_risk_budget_ann is not None else None,
                sector_neutral_mat=sector_A,
                sector_cap_abs=sector_neutral_tol,
                factor_neutral_mat=pca_A,
                factor_cap_abs=pca_neutral_tol,
            )
        except Exception as e:
            if debug:
                print(f"[{t.date()}] optimizer failed: {e}; keep previous weights")
            w_target = w_prev.copy()

        w_target = pd.Series(w_target, index=tickers).fillna(0.0)

        # Ex-ante vol targeting
        if vol_target_ex_ante and (eff_risk_budget_ann is not None):
            num = float(np.sqrt(max(1e-12, w_target.values @ Sigma.values @ w_target.values)))
            tgt = float(eff_risk_budget_ann) / np.sqrt(252.0)
            if num > 0:
                scale_v = np.clip(tgt / num, vol_scale_bounds[0], vol_scale_bounds[1])
                w_target = w_target * scale_v

        # Trade gating & costs
        w_prop = w_target.copy()
        small = (w_prop - w_prev).abs() < float(min_name_change)
        w_prop[small] = w_prev[small]
        turnover = float((w_prop - w_prev).abs().sum())
        execute = (i == 0) or (turnover >= float(eff_min_port_change))
        if execute:
            def U(mu_vec: pd.Series, w_vec: pd.Series) -> float:
                risk_term = float(w_vec.values @ Sigma.values @ w_vec.values)
                return float(mu_vec.reindex(tickers).fillna(0.0).values @ w_vec.values) - risk_aversion * risk_term
            delta_u = U(mu, w_prop) - U(mu, w_prev)
            est_cost = cost_rate * turnover
            if (i > 0) and (delta_u <= eff_benefit_cost_buf * est_cost):
                execute = False

        if execute:
            w_new = w_prev + float(eff_step_fraction) * (w_prop - w_prev)
            trade_cost = cost_rate * float((w_new - w_prev).abs().sum())
        else:
            w_new = w_prev.copy()
            trade_cost = 0.0

        weights_records.append((t, w_new.copy()))
        t_next = rebal_dates[i + 1] if (i + 1) < len(rebal_dates) else t
        if t_next > t:
            r_agg = aggregate_period_returns(rets, t, t_next)
            pnl_period = float((w_new * r_agg.reindex(tickers).fillna(0.0)).sum()) - trade_cost
            pnl_records.append((t_next, pnl_period))

        w_prev = w_new

    weights_df = (pd.DataFrame([w for _, w in weights_records],
                               index=[d for d, _ in weights_records],
                               columns=tickers).sort_index()
                  if weights_records else
                  pd.DataFrame(index=pd.Index([], name="date"), columns=tickers, dtype=float))
    pnl_s = (pd.Series([p for _, p in pnl_records],
                       index=pd.Index([d for d, _ in pnl_records], name="date"),
                       name="pnl").sort_index()
             if pnl_records else
             pd.Series(dtype=float, name="pnl"))

    return {"pnl": pnl_s, "weights": weights_df}
