from __future__ import annotations
import numpy as np
import pandas as pd

from .compute_likelihood import (
    neg_loglik_prelec_logistic,
    neg_loglik_logistic_unweighted,
)

def _try_scipy():
    try:
        import scipy.optimize as opt  # type: ignore
        return opt
    except Exception:
        return None

def _bounded_minimize(fun, x0, bounds):
    opt = _try_scipy()
    if opt is not None:
        res = opt.minimize(
            lambda v: fun(v),
            x0=np.array([*x0.values()], dtype=float),
            bounds=bounds,
            method="L-BFGS-B",
        )
        return res.x, float(res.fun)
    # Fallback: coarse grid + local random search
    best_v, best_f = None, np.inf
    grids = [np.linspace(lo, hi, 7) for (lo, hi) in bounds]
    for cand in np.array(np.meshgrid(*grids)).T.reshape(-1, len(bounds)):
        f = fun(cand)
        if f < best_f:
            best_f, best_v = f, cand.copy()
    # small random perturbations
    rng = np.random.default_rng(0)
    for _ in range(200):
        cand = best_v + rng.normal(scale=0.05, size=len(bounds))
        cand = np.clip(cand, [b[0] for b in bounds], [b[1] for b in bounds])
        f = fun(cand)
        if f < best_f:
            best_f, best_v = f, cand.copy()
    return best_v, float(best_f)

def fit_prelec_logistic(df: pd.DataFrame) -> dict:
    """
    Fit per-participant Prelec-weighted logistic with lapse.
    Bounds are conservative to keep it stable on small N.
    """
    def wrap(v):
        params = dict(alpha=v[0], beta=v[1], kappa=v[2], bias=v[3], lapse=v[4])
        return neg_loglik_prelec_logistic(params, df)

    x0 = dict(alpha=1.0, beta=1.0, kappa=10.0, bias=0.0, lapse=0.02)
    bounds = [(0.3, 2.5),  # alpha
              (0.3, 2.5),  # beta
              (0.1, 50.0), # kappa
              (-2.0, 2.0), # bias
              (0.0, 0.1)]  # lapse
    v, f = _bounded_minimize(wrap, x0, bounds)
    return dict(alpha=v[0], beta=v[1], kappa=v[2], bias=v[3], lapse=v[4], nll=f)

def fit_logistic_unweighted(df: pd.DataFrame) -> dict:
    def wrap(v):
        params = dict(kappa=v[0], bias=v[1], lapse=v[2])
        return neg_loglik_logistic_unweighted(params, df)
    x0 = dict(kappa=10.0, bias=0.0, lapse=0.02)
    bounds = [(0.1, 50.0), (-2.0, 2.0), (0.0, 0.1)]
    v, f = _bounded_minimize(wrap, x0, bounds)
    return dict(kappa=v[0], bias=v[1], lapse=v[2], nll=f)

def aic(nll: float, k: int) -> float:
    return 2 * k + 2 * nll

def bic(nll: float, k: int, n: int) -> float:
    return k * np.log(max(n, 1)) + 2 * nll

def fit_models_by_participant(df: pd.DataFrame, id_cols=("participantId",)) -> pd.DataFrame:
    rows = []
    for pid, d in df.groupby(list(id_cols)):
        n = len(d)
        m1 = fit_logistic_unweighted(d)
        m2 = fit_prelec_logistic(d)
        rows.append({
            **{c: pid[i] if len(id_cols) > 1 else pid for i, c in enumerate(id_cols)},
            "n": n,
            "m1_nll": m1["nll"], "m1_aic": aic(m1["nll"], k=3), "m1_bic": bic(m1["nll"], k=3, n=n),
            "m1_kappa": m1["kappa"], "m1_bias": m1["bias"], "m1_lapse": m1["lapse"],
            "m2_nll": m2["nll"], "m2_aic": aic(m2["nll"], k=5), "m2_bic": bic(m2["nll"], k=5, n=n),
            "m2_alpha": m2["alpha"], "m2_beta": m2["beta"], "m2_kappa": m2["kappa"],
            "m2_bias": m2["bias"], "m2_lapse": m2["lapse"],
            "dAIC": (aic(m1["nll"], 3) - aic(m2["nll"], 5)),
            "dBIC": (bic(m1["nll"], 3, n) - bic(m2["nll"], 5, n)),
        })
    return pd.DataFrame(rows)
