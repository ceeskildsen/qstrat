from __future__ import annotations
import numpy as np
import pandas as pd
from src.risk_models import sample_cov, ewma_cov
try:
    from src.risk_models import ledoit_wolf_cov
    HAS_LW = True
except Exception:
    HAS_LW = False

# Optional GP-kernel covariance (only used if risk_model starts with "gp_kernel")
try:
    from src.risk_models import gp_kernel_cov
    HAS_GP_KERNEL = True
except Exception:
    HAS_GP_KERNEL = False

def choose_cov(rets_train: pd.DataFrame, *, risk_model: str, ewma_lambda: float) -> pd.DataFrame:
    rm = (risk_model or "sample").lower()

    if rm.startswith("gp_kernel"):
        if not HAS_GP_KERNEL:
            raise ValueError("risk_model='gp_kernel' requested but gp_kernel_cov not available in src.risk_models.")
        kernel = "rbf"
        parts = rm.split(":")
        if len(parts) >= 2:
            kernel = parts[1].strip() or "rbf"
        ls_days = None
        try:
            if 0.0 < float(ewma_lambda) < 1.0:
                ls_days = max(1.0, float(-1.0 / np.log(float(ewma_lambda))))
        except Exception:
            ls_days = None
        S = gp_kernel_cov(rets_train, kernel=kernel, length_scale_days=ls_days)

    elif rm in ("sample", "sample_cov"):
        S = sample_cov(rets_train)

    elif rm in ("ewma", "ewma_cov"):
        S = ewma_cov(rets_train, lam=ewma_lambda)

    elif rm in ("lw", "ledoit_wolf", "ledoit-wolf") and HAS_LW:
        S = ledoit_wolf_cov(rets_train)

    else:
        raise ValueError(
            f"Unknown risk_model '{risk_model}'. "
            "Use 'sample', 'ewma', 'lw' (if available), or 'gp_kernel[:rbf|matern52]' (if installed)."
        )

    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S.values)
    ridge = 1e-6 * max(1.0, float(np.mean(vals)))
    vals = np.clip(vals, ridge, None)
    S_psd = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(S_psd, index=S.index, columns=S.columns)
