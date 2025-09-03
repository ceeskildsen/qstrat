from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from src.features import compute_betas_to_market

def build_sector_matrix(tickers: List[str], sector_map: Dict[str, str] | pd.Series | None) -> np.ndarray | None:
    if sector_map is None:
        return None
    if isinstance(sector_map, pd.Series):
        s = sector_map.reindex(tickers)
    else:
        s = pd.Series({t: sector_map.get(t, "Other") for t in tickers})
    s = s.fillna("Other")
    counts = s.value_counts()
    keep = [sec for sec, c in counts.items() if c >= 2]
    if len(keep) <= 1:
        return None
    rows = []
    for sec in keep:
        mask = (s == sec).astype(float).reindex(tickers).fillna(0.0).values
        rows.append(mask)
    return np.vstack(rows) if rows else None

def build_pca_matrix(Sigma: pd.DataFrame, k: int, equalize_risk: bool) -> np.ndarray | None:
    if (k is None) or (k <= 0) or Sigma.empty:
        return None
    vals, vecs = np.linalg.eigh(Sigma.values)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    K = min(int(k), vecs.shape[1])
    if K <= 0:
        return None
    if equalize_risk:
        scales = np.sqrt(np.clip(vals[:K], 1e-12, None))
        return (vecs[:, :K] * scales.reshape(1, -1)).T   # (K x N)
    return vecs[:, :K].T

def compute_beta_vector(
    rets: pd.DataFrame,
    spy_rets: Optional[pd.Series],
    t: pd.Timestamp,
    beta_lookback: int,
    tickers: list[str],
) -> np.ndarray | None:
    if (spy_rets is None):
        return None
    look_mask = (rets.index <= t) & (rets.index > (t - pd.Timedelta(days=beta_lookback)))
    beta_window = rets.loc[look_mask]
    mkt_slice = spy_rets.loc[beta_window.index]
    if beta_window.empty or len(beta_window) != len(mkt_slice):
        return None
    betas = compute_betas_to_market(beta_window, mkt_slice)
    return betas.reindex(tickers).fillna(0.0).values
