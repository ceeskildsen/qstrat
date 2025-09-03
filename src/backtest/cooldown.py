from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

def compute_cooldown_params(
    pnl_records: List[Tuple[pd.Timestamp, float]],
    *,
    cool_lookback: int = 2,
    dd_window: int = 4,
    loss_thresh: float = -0.002,
    dd_thresh: float = 0.10,
    near_peak_th: float = 0.01,
    mu_cool_scale: float = 0.90,
    bound_cool_mult: float = 0.90,
    gross_cool_mult: float = 0.90,
    risk_aversion_boost: float = 1.15,
):
    # equity rebuild
    eq_hist = [1.0]
    for _, p in pnl_records:
        eq_hist.append(eq_hist[-1] * (1.0 + float(p)))
    eq_series = np.array(eq_hist, dtype=float)

    is_cool = False
    dd_recent = 0.0
    near_peak = True
    sum_recent = 0.0

    if len(pnl_records) >= max(cool_lookback, 1):
        recent = np.array([p for _, p in pnl_records[-cool_lookback:]], dtype=float)
        sum_recent = np.nansum(recent)
        neg_count = int((recent < 0).sum())

        if len(eq_series) >= dd_window + 1:
            eq_win = eq_series[-(dd_window + 1):]
            peak_win = np.maximum.accumulate(eq_win)
            dd_recent = 1.0 - (eq_win[-1] / peak_win[-1])
            near_peak = (peak_win[-1] - eq_win[-1]) / peak_win[-1] <= near_peak_th

        is_cool = ((sum_recent < loss_thresh) or (neg_count >= 2)) and (dd_recent > dd_thresh) and (not near_peak)

    params = {
        "is_cool": is_cool,
        "mu_scale": mu_cool_scale if is_cool else 1.0,
        "position_bound_mult": bound_cool_mult if is_cool else 1.0,
        "gross_limit_mult": gross_cool_mult if is_cool else 1.0,
        "risk_aversion_mult": risk_aversion_boost if is_cool else 1.0,
        # late-stage knobs
        "step_fraction_mult": 0.5 if is_cool else 1.0,
        "min_port_change_mult": 1.5 if is_cool else 1.0,
        "benefit_cost_buffer_mult": 1.5 if is_cool else 1.0,
        "risk_budget_annual_mult": 0.8 if is_cool else 1.0,
        # debug
        "dd_recent": dd_recent,
        "sum_recent": sum_recent,
        "near_peak": near_peak,
    }
    return params
