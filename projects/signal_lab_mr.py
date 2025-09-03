# projects/signal_lab_mr.py
# Make repo root importable
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

MINIMAL = True  # <<< set False to generate all extra series/hists/weekly-monthly summaries

from src.data.prices import get_prices
from src.utils import to_returns
from src.plots import (
    save_equity_curve,
    save_ic_timeseries,
    save_ic_hist,
    save_icir_heatmap,
    save_bar_df,
)
from src.signals import (
    compute_residual_returns,
    mean_reversion_signal,
    momentum_12_1_signal,
    combine_signals_z,
)

try:
    from src.metrics import information_coefficient_safe as information_coefficient
except Exception:
    from src.metrics import information_coefficient as information_coefficient

def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1] if len(here.parents) >= 2 else Path.cwd()

ROOT = _repo_root_from_here()
ALPHA_CFG = ROOT / "outputs" / "alpha_to_portfolio" / "run_config.json"

DEFAULTS = {"tickers": [], "start": "2020-01-01", "end": "2024-12-31",
            "market_ticker": "SPY", "rebalance_freq": "ME", "mr_beta_lookback": 252}

cfg_last = json.loads(ALPHA_CFG.read_text(encoding="utf-8")) if ALPHA_CFG.exists() else {}
TICKERS        = cfg_last.get("available_tickers", DEFAULTS["tickers"])
START          = cfg_last.get("start", DEFAULTS["start"])
END            = cfg_last.get("end", DEFAULTS["end"])
MARKET_TICKER  = cfg_last.get("market_ticker", DEFAULTS["market_ticker"])
REBAL_FREQ     = str(cfg_last.get("config", {}).get("FREQ", DEFAULTS["rebalance_freq"])).upper()
MR_BETA_LOOKBACK = int(cfg_last.get("config", {}).get("mr_beta_lookback", DEFAULTS["mr_beta_lookback"]))

MR_LOOKBACKS   = [3, 5, 10]
FWD_HORIZONS   = [1, 3, 5]
BLEND_WEIGHTS  = [0.0, 0.1, 0.2, 0.3, 0.5]

OUT_DIR = ROOT / "outputs" / "signal_lab_mr"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PREFIX = "sl_mr"

def p(path: Path) -> str: return path.as_posix()

# Data
prices = get_prices(TICKERS, start=START, end=END).sort_index()
prices = prices[[c for c in prices.columns if c in TICKERS]]
rets   = to_returns(prices)
mkt    = get_prices(MARKET_TICKER, start=START, end=END)[MARKET_TICKER].pct_change().dropna()

print(f"MR lab: {len(prices.columns)} names | {START} → {END} | Market={MARKET_TICKER} | Freq={REBAL_FREQ}")

resid   = compute_residual_returns(rets, mkt, lookback=MR_BETA_LOOKBACK)
signal0 = mean_reversion_signal(resid, lookback=5)

def decile_long_short(signal_df: pd.DataFrame, fwd_returns: pd.DataFrame,
                      top_q=0.1, bottom_q=0.1) -> pd.Series:
    common = signal_df.index.intersection(fwd_returns.index)
    pnl = []
    for dt in common:
        srow = signal_df.loc[dt]
        rrow = fwd_returns.loc[dt]
        mask = srow.notna() & rrow.notna()
        if mask.sum() < 20:
            pnl.append(np.nan); continue
        s = srow[mask].rank(pct=True)
        long = s >= (1.0 - top_q)
        short = s <= bottom_q
        if long.sum() == 0 or short.sum() == 0:
            pnl.append(np.nan); continue
        pnl.append(rrow[long].mean() - rrow[short].mean())
    return pd.Series(pnl, index=common).dropna()

# Grid: L × H
rows = []
for L in MR_LOOKBACKS:
    sigL = mean_reversion_signal(resid, lookback=int(L))
    for H in FWD_HORIZONS:
        fut = rets.shift(-int(H))
        idx = sigL.index.intersection(fut.index)
        if idx.empty: continue
        ic = information_coefficient(sigL.loc[idx], fut.loc[idx], method="spearman")
        mu, sd = float(ic.mean()), float(ic.std(ddof=1))
        ir = 0.0 if sd == 0 else (mu / sd) * np.sqrt(252.0)
        hit = 100.0 * float((ic > 0).mean())
        rows.append({"mr_lookback": int(L), "fwd_days": int(H),
                     "ic_mean": mu, "ic_std": sd, "ic_ir_ann": ir,
                     "hit_rate_pct": hit, "n_days": int(ic.size)})

df_res = pd.DataFrame(rows).sort_values(["ic_ir_ann","ic_mean"], ascending=[False, False])
(df_res).to_csv(OUT_DIR / f"{OUT_PREFIX}_grid_summary.csv", index=False)
if not df_res.empty:
    try:
        save_icir_heatmap(df_res, row="mr_lookback", col="fwd_days", value="ic_ir_ann",
                          out_path=OUT_DIR / f"{OUT_PREFIX}_ic_ir_heatmap.png",
                          title="MR: IC IR (annualized)")
    except Exception as e:
        print(f"[info] heatmap skipped: {e}")

# Blend grid
mom = momentum_12_1_signal(prices, gap_days=21, lookback_days=252)
blend_rows = []
for w in BLEND_WEIGHTS:
    combo = combine_signals_z(mom, signal0, weight_secondary=w, winsor_k=3.0, restandardize=True)
    fut = rets.shift(-1)
    idx = combo.index.intersection(fut.index)
    if idx.empty: continue
    ic = information_coefficient(combo.loc[idx], fut.loc[idx])
    mu, sd = float(ic.mean()), float(ic.std(ddof=1))
    ir = 0.0 if sd == 0 else (mu / sd) * np.sqrt(252.0)
    blend_rows.append({"mr_weight": float(w), "ic_ir_ann": ir})

df_blend = pd.DataFrame(blend_rows)
df_blend.to_csv(OUT_DIR / f"{OUT_PREFIX}_blend_mom_mr_grid.csv", index=False)
if not df_blend.empty:
    try:
        save_bar_df(df_blend.rename(columns={"mr_weight":"x","ic_ir_ann":"y"}),
                    x="x", y="y",
                    out_path=OUT_DIR / f"{OUT_PREFIX}_blend_icir_bar.png",
                    title="MOM + w·MR — IC IR (ann.)")
    except Exception as e:
        print(f"[info] blend bar skipped: {e}")

# Best showcase: choose F=1, top L
if not df_res.empty:
    cand = df_res[df_res["fwd_days"] == 1]
    if not cand.empty:
        Lbest = int(cand.iloc[0]["mr_lookback"])
        sig_best = mean_reversion_signal(resid, lookback=Lbest)
        fut = rets.shift(-1)
        idx = sig_best.index.intersection(fut.index)
        if not idx.empty:
            ls = decile_long_short(sig_best.loc[idx], fut.loc[idx])
            if not ls.empty:
                ann_ret = float(ls.mean() * 252.0)
                ann_vol = float(ls.std(ddof=1) * np.sqrt(252.0))
                sharpe  = 0.0 if ann_vol == 0 else ann_ret / ann_vol
                pd.Series({"L": Lbest, "F": 1, "ann_ret": ann_ret,
                           "ann_vol": ann_vol, "sharpe": sharpe,
                           "n_days": int(ls.size)}).to_csv(OUT_DIR / f"{OUT_PREFIX}_best_stats_L{Lbest}_F1.csv")
                eq = (1 + ls).cumprod()
                try:
                    save_equity_curve(eq, OUT_DIR / f"{OUT_PREFIX}_decile_equity_L{Lbest}_F1.png",
                                      title=f"MR decile L/S equity — L={Lbest}, F=1", y_label="NAV")
                except Exception as e:
                    print(f"[info] decile equity skipped: {e}")

            if not MINIMAL:
                # Optional IC series & hist for the best daily config
                ic_best = information_coefficient(sig_best.loc[idx], fut.loc[idx])
                ic_best.to_csv(OUT_DIR / f"{OUT_PREFIX}_ic_daily_L{Lbest}_F1.csv", header=["ic"])
                try:
                    save_ic_timeseries(ic_best, OUT_DIR / f"{OUT_PREFIX}_ic_timeseries_L{Lbest}_F1.png",
                                       title=f"IC (daily) — MR L={Lbest}, F=1")
                except Exception as e:
                    print(f"[info] IC timeseries skipped: {e}")
                try:
                    save_ic_hist(ic_best, OUT_DIR / f"{OUT_PREFIX}_ic_hist_L{Lbest}_F1.png",
                                 title=f"IC histogram — MR L={Lbest}, F=1", bins=40)
                except Exception as e:
                    print(f"[info] IC hist skipped: {e}")

# Weekly/monthly summaries (OPTIONAL)
if not MINIMAL:
    sigW = signal0.resample("W-FRI").last().dropna(how="all")
    retW = rets.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1).dropna(how="all")
    icW  = information_coefficient(sigW.loc[sigW.index.intersection(retW.index)],
                                   retW.loc[sigW.index.intersection(retW.index)])
    pd.Series({"ic_mean": float(icW.mean()), "ic_std": float(icW.std(ddof=1)),
               "ic_ir_ann": float((icW.mean() / icW.std(ddof=1)) * np.sqrt(52.0)) if icW.std(ddof=1) else 0.0,
               "n_weeks": int(icW.size)}).to_csv(OUT_DIR / f"{OUT_PREFIX}_weekly_ic_summary.csv")

    lsW = decile_long_short(sigW, retW.shift(-1))
    if not lsW.empty:
        ann_ret_W = float(lsW.mean() * 52.0)
        ann_vol_W = float(lsW.std(ddof=1) * np.sqrt(52.0))
        sharpe_W  = 0.0 if ann_vol_W == 0 else ann_ret_W / ann_vol_W
        pd.Series({"ann_ret": ann_ret_W, "ann_vol": ann_vol_W,
                   "sharpe": sharpe_W, "n_weeks": int(lsW.size)}).to_csv(OUT_DIR / f"{OUT_PREFIX}_weekly_decile_summary.csv")
    else:
        pd.Series({"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "n_weeks": 0}).to_csv(OUT_DIR / f"{OUT_PREFIX}_weekly_decile_summary.csv")

    sigM = signal0.resample("ME").last().dropna(how="all")
    retM = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna(how="all")
    icM  = information_coefficient(sigM.loc[sigM.index.intersection(retM.index)],
                                   retM.loc[sigM.index.intersection(retM.index)])
    pd.Series({"ic_mean": float(icM.mean()), "ic_std": float(icM.std(ddof=1)),
               "ic_ir_ann": float((icM.mean() / icM.std(ddof=1)) * np.sqrt(12.0)) if icM.std(ddof=1) else 0.0,
               "n_months": int(icM.size)}).to_csv(OUT_DIR / f"{OUT_PREFIX}_monthly_ic_summary.csv")
