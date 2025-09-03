# projects/signal_lab_mom.py
# Make repo root importable
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.prices import get_prices
from src.plots import save_equity_curve
from src.signals import momentum_12_1_signal

# Prefer the safe IC if present
try:
    from src.metrics import information_coefficient_safe as information_coefficient
except Exception:
    from src.metrics import information_coefficient as information_coefficient

# ---------------- Paths & run context ----------------
def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1] if len(here.parents) >= 2 else Path.cwd()

ROOT = _repo_root_from_here()

# Load last alpha_to_portfolio config for dates/universe
CFG_PATH = ROOT / "outputs" / "alpha_to_portfolio" / "run_config.json"
DEFAULTS = {
    "tickers": [],
    "start": "2015-01-01",
    "end": "2025-01-01",
    "rebalance_freq": "W-FRI",
}
if CFG_PATH.exists():
    cfg_last = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg_last = {
        "tickers": cfg_last.get("available_tickers", DEFAULTS["tickers"]),
        "start": cfg_last.get("start", DEFAULTS["start"]),
        "end": cfg_last.get("end", DEFAULTS["end"]),
        "rebalance_freq": str(cfg_last.get("config", {}).get("FREQ", DEFAULTS["rebalance_freq"])).upper(),
    }
else:
    cfg_last = DEFAULTS

TICKERS    = cfg_last.get("tickers", DEFAULTS["tickers"])
START      = cfg_last.get("start", DEFAULTS["start"])
END        = cfg_last.get("end", DEFAULTS["end"])
REBAL_FREQ = str(cfg_last.get("rebalance_freq", DEFAULTS["rebalance_freq"])).upper()

# ---------------- Params & output ----------------
GAPS       = [10, 21, 30]       # business day gap (≈0.5m, 1m, ~1.5m)
LOOKBACKS  = [126, 189, 252]    # 6m, 9m, 12m
OUT_DIR    = ROOT / "outputs" / "signal_lab_mom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def p(path: Path) -> str:
    return path.as_posix()

# ---------------- Data ----------------
prices = get_prices(TICKERS, start=START, end=END).sort_index()
# forward (next-day) returns: return realized after signal is known at t
# NB: we do NOT shift again later.
returns_fwd = prices.pct_change().shift(-1)

# ---------------- Helpers ----------------
def _ic_series(signal_df: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """
    Daily cross-sectional IC using the library function, which expects DataFrames
    (dates × tickers) and returns a per-date Series.
    """
    # Align columns and index
    common_cols = sorted(set(signal_df.columns) & set(fwd_returns.columns))
    if not common_cols:
        return pd.Series(dtype=float)
    sig = signal_df[common_cols].dropna(how="all")
    fut = fwd_returns[common_cols].dropna(how="all")
    common_idx = sig.index.intersection(fut.index)
    if common_idx.empty:
        return pd.Series(dtype=float)
    sig = sig.loc[common_idx]
    fut = fut.loc[common_idx]
    ic = information_coefficient(sig, fut)
    ic.name = "IC"
    return ic

def _ic_ir(ic: pd.Series) -> float:
    if ic.empty:
        return 0.0
    mu = ic.mean()
    sd = ic.std(ddof=1)
    return 0.0 if sd == 0 else (mu / sd) * np.sqrt(252.0)

def _hit_rate(ic: pd.Series) -> float:
    if ic.empty:
        return 0.0
    return 100.0 * (ic > 0).mean()

def decile_long_short(signal_df: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Form a D10–D1 long/short (equal-weight) from cross-sectional deciles each date."""
    common = signal_df.index.intersection(fwd_returns.index)
    signal_df = signal_df.loc[common]
    fwd_returns = fwd_returns.loc[common]
    pnl = []
    for dt in common:
        srow = signal_df.loc[dt]
        rrow = fwd_returns.loc[dt]
        mask = srow.notna() & rrow.notna()
        if mask.sum() < 20:
            pnl.append(np.nan)
            continue
        s = srow[mask].rank(pct=True)
        long = s >= 0.9
        short = s <= 0.1
        if long.sum() == 0 or short.sum() == 0:
            pnl.append(np.nan)
            continue
        ls_ret = rrow[long].mean() - rrow[short].mean()
        pnl.append(ls_ret)
    return pd.Series(pnl, index=common).dropna()

# ---------------- Grid sweep ----------------
rows = []
monthly_ic_rows = []

for gap in GAPS:
    for lb in LOOKBACKS:
        sig = momentum_12_1_signal(prices, gap_days=int(gap), lookback_days=int(lb))
        ic_series = _ic_series(sig, returns_fwd)
        ic_ir_ann = _ic_ir(ic_series)
        hit = _hit_rate(ic_series)
        rows.append({
            "gap": int(gap),
            "lookback": int(lb),
            "ic_mean": ic_series.mean(),
            "ic_std": ic_series.std(ddof=1),
            "ic_ir_ann": ic_ir_ann,
            "hit_rate_pct": hit,
            "n_days": int(ic_series.size)
        })

        # Monthly IC summary
        if not ic_series.empty:
            ic_month = ic_series.resample("ME").mean()
            for dt, val in ic_month.items():
                monthly_ic_rows.append({
                    "gap": int(gap),
                    "lookback": int(lb),
                    "month": dt.strftime("%Y-%m"),
                    "ic_mean": float(val)
                })

grid = pd.DataFrame(rows)
if grid.empty:
    print("No results (empty grid) — likely due to data gaps. Try a shorter date range or fewer names.")
    grid_path = OUT_DIR / "mom_grid_summary.csv"
    grid.to_csv(grid_path, index=False)
else:
    grid = grid.sort_values("ic_ir_ann", ascending=False)
    grid_path = OUT_DIR / "mom_grid_summary.csv"
    grid.to_csv(grid_path, index=False)
    print(f"Saved MOM sweep summary: {p(grid_path)}")
    print(grid.head(10).to_string(index=False))

# Monthly IC summary CSV
monthly_ic_df = pd.DataFrame(monthly_ic_rows)
monthly_ic_df.to_csv(OUT_DIR / "mom_monthly_ic_summary.csv", index=False)

# ---------------- Best decile long-short equity (single chart, same filename convention) ----------------
def equity_from_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r.astype(float)).cumprod()

if not grid.empty:
    gapB, lbB = int(grid.iloc[0]["gap"]), int(grid.iloc[0]["lookback"])
    momB = momentum_12_1_signal(prices, gap_days=gapB, lookback_days=lbB)
    # use the already-forward returns (NO extra shift here)
    ls = decile_long_short(momB, returns_fwd)
    if not ls.empty:
        ann_ret = ls.mean() * 252
        ann_vol = ls.std(ddof=1) * np.sqrt(252)
        sharpe  = 0.0 if ann_vol == 0 else ann_ret / ann_vol
        s_csv = OUT_DIR / f"mom_decile_stats_gap{gapB}_lb{lbB}.csv"
        pd.Series({
            "gap": gapB,
            "lookback": lbB,
            "ann_ret": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "n_days": int(ls.size)
        }).to_csv(s_csv)
        print(f"Saved decile stats: {p(s_csv)}")
        eq = (1 + ls).cumprod()
        save_equity_curve(
            eq,
            OUT_DIR / f"mom_decile_equity_gap{gapB}_lb{lbB}.png",
            title=f"MOM decile L/S equity — gap={gapB}, lb={lbB}",
            y_label="NAV"
        )
    else:
        print("Decile L/S empty (not enough valid days).")
else:
    print("Skipping decile L/S (no best config).")
