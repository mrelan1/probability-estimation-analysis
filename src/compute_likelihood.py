from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def prelec_weight(p: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Prelec (1998) weighting:
      w(p) = exp( -beta * (-ln p)^alpha ), for p in (0,1)
    """
    p = np.clip(p, EPS, 1 - EPS)
    return np.exp(-beta * np.power(-np.log(p), alpha))

def choice_prob_prelec_logistic(
    p: np.ndarray,
    t: np.ndarray,
    kappa: float,
    bias: float,
    alpha: float,
    beta: float,
    lapse: float = 0.0,
) -> np.ndarray:
    """
    P(response='higher') = (1 - lapse) * sigmoid(kappa * (w(p) - t) + bias) + lapse * 0.5
    """
    w = prelec_weight(p, alpha, beta)
    core = logistic(kappa * (w - t) + bias)
    return (1.0 - lapse) * core + 0.5 * lapse

def neg_loglik_prelec_logistic(params: dict, df: pd.DataFrame) -> float:
    """
    params: dict(alpha, beta, kappa, bias, lapse)
    df must contain: 'probability', 'threshold', 'is_higher'
    """
    alpha = float(params.get("alpha", 1.0))  # curvature
    beta  = float(params.get("beta", 1.0))   # elevation
    kappa = float(params.get("kappa", 10.0)) # slope (noise)
    bias  = float(params.get("bias", 0.0))   # response bias
    lapse = float(params.get("lapse", 0.0))  # random lapses in [0, 0.1]

    p = choice_prob_prelec_logistic(
        df["probability"].to_numpy(),
        df["threshold"].to_numpy(),
        kappa=kappa, bias=bias, alpha=alpha, beta=beta, lapse=lapse
    )
    y = df["is_higher"].to_numpy()
    p = np.clip(p, EPS, 1 - EPS)
    ll = np.where(y == 1, np.log(p), np.log(1 - p)).sum()
    return -float(ll)

def neg_loglik_logistic_unweighted(params: dict, df: pd.DataFrame) -> float:
    """
    Baseline model without probability weighting (w(p) = p).
    params: dict(kappa, bias, lapse)
    """
    kappa = float(params.get("kappa", 10.0))
    bias  = float(params.get("bias", 0.0))
    lapse = float(params.get("lapse", 0.0))

    p = (1.0 - lapse) * logistic(kappa * (df["probability"] - df["threshold"]) + bias) + 0.5 * lapse
    y = df["is_higher"].to_numpy()
    p = np.clip(p, EPS, 1 - EPS)
    ll = np.where(y == 1, np.log(p), np.log(1 - p)).sum()
    return -float(ll)
