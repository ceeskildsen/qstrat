from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple

def _subsample_balanced(X: np.ndarray, y: np.ndarray, max_points: Optional[int], tail_frac: float = 0.25, rng: int = 42):
    if (max_points is None) or (X.shape[0] <= max_points):
        return X, y
    n = X.shape[0]
    k = int(max_points)
    tail_n = max(int(tail_frac * k), 1)
    order = np.argsort(-np.abs(y))
    tail_idx = order[:2 * tail_n]
    keep = set(tail_idx.tolist())
    mid_idx = [i for i in range(n) if i not in keep]
    rs = np.random.RandomState(rng)
    need = k - len(keep)
    if need > 0:
        pick = rs.choice(mid_idx, size=need, replace=False)
        keep.update(pick.tolist())
    keep_idx = np.array(sorted(list(keep)))
    return X[keep_idx], y[keep_idx]

def _median_gamma(Xs: np.ndarray, subsample: int = 200) -> float:
    from sklearn.metrics import pairwise_distances
    n = Xs.shape[0]
    if n <= 2:
        return 1.0
    m = min(n, subsample)
    rng = np.random.RandomState(0)
    idx = rng.choice(n, size=m, replace=False)
    D2 = pairwise_distances(Xs[idx], Xs[idx], metric="sqeuclidean")
    med = np.median(D2[D2 > 0])
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return 1.0 / (2.0 * med)

def sklearn_gp_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_curr: np.ndarray,
    *,
    prev_state: Optional[dict] = None,
    refit: bool = True,
    n_restarts: int = 0,
    alpha: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler
        try:
            from sklearn.exceptions import ConvergenceWarning
        except Exception:
            ConvergenceWarning = Warning
        import warnings
    except Exception as e:
        raise ImportError("scikit-learn is required for sklearn GP alpha model.") from e

    if (not refit) and (prev_state is not None):
        gpr = prev_state["model"]
        scaler = prev_state["scaler"]
        Xc = scaler.transform(X_curr)
        mu, std = gpr.predict(Xc, return_std=True)
        return mu, std, prev_state

    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xc = scaler.transform(X_curr)

    n_features = Xt.shape[1]
    k_linear = DotProduct(sigma_0=1.0)
    k_matern = Matern(length_scale=np.ones(n_features), nu=1.5)
    kernel = ConstantKernel(1.0) * (k_linear + k_matern) + WhiteKernel(noise_level=alpha)

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(alpha),
        normalize_y=False,
        n_restarts_optimizer=int(max(0, n_restarts)),
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.gaussian_process.kernels")
        gpr.fit(Xt, y_train)

    mu, std = gpr.predict(Xc, return_std=True)
    state = {"model": gpr, "scaler": scaler}
    return mu, std, state

def nystrom_krr_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_curr: np.ndarray,
    *,
    prev_state: Optional[dict] = None,
    refit: bool = True,
    kernel: str = "rbf",
    n_components: int = 200,
    alpha: float = 1e-2,
    gamma: Optional[float] = None,
    random_state: int = 0,
    dates_train: Optional[np.ndarray] = None,
    as_of: Optional[pd.Timestamp] = None,
    time_half_life_days: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.kernel_approximation import Nystroem
        import numpy as np
    except Exception as e:
        raise ImportError("scikit-learn is required for Nyström KRR alpha model.") from e

    if (not refit) and (prev_state is not None):
        scaler = prev_state["scaler"]
        nys    = prev_state["map"]
        w      = prev_state["w"]
        L      = prev_state["L"]
        resid_std = prev_state["resid_std"]

        Xc = scaler.transform(X_curr)
        Phic = nys.transform(Xc)
        mu = Phic @ w
        std_proxy = np.zeros(Phic.shape[0], dtype=float)
        for i in range(Phic.shape[0]):
            v = np.linalg.solve(L, Phic[i])
            std_proxy[i] = resid_std * np.sqrt(max(1e-12, np.dot(v, v)))
        return mu, std_proxy, prev_state

    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xc = scaler.transform(X_curr)

    n_samples = Xt.shape[0]
    M_eff = int(min(n_components, max(1, n_samples - 1)))

    g = float(gamma) if (gamma is not None) else _median_gamma(Xt)
    nys = Nystroem(kernel=kernel, n_components=M_eff, gamma=g, random_state=random_state)
    Phit = nys.fit_transform(Xt)

    w_sample = None
    if (time_half_life_days is not None) and (time_half_life_days > 0) and (dates_train is not None) and (as_of is not None):
        dt = pd.to_datetime(dates_train)
        if isinstance(dt, pd.DatetimeIndex):
            delta_days = (pd.Timestamp(as_of) - dt).days
        else:
            delta_days = (pd.Timestamp(as_of) - dt).astype("timedelta64[D]").astype(int)
        delta_days = np.asarray(delta_days, dtype=float)
        w_sample = 0.5 ** (delta_days / float(time_half_life_days))
        w_sample = np.clip(w_sample, 1e-3, 1.0)

    M = Phit.shape[1]
    if w_sample is not None:
        A = Phit.T @ (Phit * w_sample[:, None]) + float(alpha) * np.eye(M)
        b = Phit.T @ (y_train * w_sample)
    else:
        A = Phit.T @ Phit + float(alpha) * np.eye(M)
        b = Phit.T @ y_train

    L = np.linalg.cholesky(A)
    w = np.linalg.solve(L.T, np.linalg.solve(L, b))

    resid = y_train - Phit @ w
    if w_sample is not None:
        denom = max(1.0, float(np.sum(w_sample)) - M)
        resid_std = float(np.sqrt(np.maximum(1e-12, np.sum(w_sample * (resid**2)) / denom)))
    else:
        denom = max(1, Phit.shape[0] - M)
        resid_std = float(np.sqrt(np.maximum(1e-12, np.sum(resid**2) / denom)))

    Phic = nys.transform(Xc)
    mu = Phic @ w

    std_proxy = np.zeros(Phic.shape[0], dtype=float)
    for i in range(Phic.shape[0]):
        v = np.linalg.solve(L, Phic[i])
        std_proxy[i] = resid_std * np.sqrt(max(1e-12, np.dot(v, v)))

    state = {"scaler": scaler, "map": nys, "w": w, "L": L, "resid_std": resid_std, "alpha": float(alpha)}
    return mu, std_proxy, state
