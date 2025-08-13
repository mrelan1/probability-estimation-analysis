
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

# SciPy is preferred for Beta CDF; fall back to mpmath if needed.
try:
    from scipy.stats import beta as scipy_beta
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    try:
        import mpmath as mp # type: ignore
    except Exception:
        mp = None

@dataclass
class Prior:
    a: float = 1.0   # Beta(a, b); a=b=1 -> uniform prior
    b: float = 1.0

def binomial_likelihood(k: int, n: int, p: np.ndarray) -> np.ndarray:
    """Likelihood P(tf=k/n | P_T = p) ignoring order: Binomial PMF."""
    from math import comb
    k = int(k); n = int(n)
    coef = comb(n, k)
    p = np.asarray(p, dtype=float)
    lk = coef * (p**k) * ((1.0 - p)**(n - k))
    lk[~np.isfinite(lk)] = 0.0
    return lk

def posterior_grid(k: int, n: int, grid: np.ndarray, prior: Prior = Prior()) -> np.ndarray:
    """
    Compute unnormalized posterior on a grid and normalize to sum to 1.
    With Beta(a,b) prior, posterior ∝ p^(k+a-1) (1-p)^(n-k+b-1).
    """
    a, b = prior.a, prior.b
    p = np.asarray(grid, dtype=float)
    post = (p**(k + a - 1.0)) * ((1.0 - p)**(n - k + b - 1.0))
    area = np.trapz(post, p)
    if area <= 0 or not np.isfinite(area):
        return np.full_like(p, 0.0)
    return post / area

def beta_cdf(x: float, a: float, b: float) -> float:
    """CDF of Beta(a,b) at x. Uses SciPy if available, else mpmath or numeric grid."""
    x = float(x)
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    if _HAVE_SCIPY:
        return float(scipy_beta.cdf(x, a, b))
    if 'mp' in globals() and mp is not None:
        return float(mp.betainc(a, b, 0, x, regularized=True))
    # crude fallback
    g = np.linspace(0, x, 2001)
    num = np.trapz((g**(a-1))*((1-g)**(b-1)), g)
    g2 = np.linspace(0, 1, 2001)
    den = np.trapz((g2**(a-1))*((1-g2)**(b-1)), g2)
    return float(num/den) if den > 0 else 0.0

def posterior_prob_gt_threshold(k: int, n: int, T: float, prior: Prior = Prior()) -> float:
    """
    Compute P(P_T > T | tf=k/n) under Beta(a,b) prior using the analytic Beta posterior.
    Posterior is Beta(k+a, n-k+b). Then:
        P(P_T > T | data) = 1 - BetaCDF(T; k+a, n-k+b)
    """
    a_post = k + prior.a
    b_post = (n - k) + prior.b
    cdf_at_T = beta_cdf(T, a_post, b_post)
    return float(max(0.0, min(1.0, 1.0 - cdf_at_T)))

def bayes_optimal_decision(k: int, n: int, T: float, prior: Prior = Prior(), tie_tol: float = 1e-12) -> Tuple[str, float, float, bool]:
    """
    Return (decision, prob_gt, prob_lt, is_tie).
      decision ∈ {"higher","lower"} based on which posterior probability is larger.
      prob_gt = P(P_T > T | data)
      prob_lt = 1 - prob_gt
      is_tie: True if |prob_gt - prob_lt| <= tie_tol
    """
    prob_gt = posterior_prob_gt_threshold(k, n, T, prior=prior)
    prob_lt = 1.0 - prob_gt
    is_tie = abs(prob_gt - prob_lt) <= tie_tol
    decision = "higher" if prob_gt > prob_lt else "lower"
    return decision, prob_gt, prob_lt, is_tie

def autodetect_columns(df, hints: Optional[Dict[str,str]] = None) -> Dict[str,str]:
    """
    Try to map expected fields to actual column names in trials_clean.csv.
    You can override with `hints` dict.
    Expected keys: participant_id, threshold, choice, rt, target_count, n_samples
    """
    mapping = {
        "participant_id": None,
        "threshold": None,
        "choice": None,
        "rt": None,
        "target_count": None,
        "n_samples": None,
    }
    cand = {c.lower(): c for c in df.columns}
    # heuristic guesses
    for key, options in {
        "participant_id": ["participantid","pid","participant_id","subj","subject"],
        "threshold": ["threshold","thresh","t"],
        "choice": ["choice","response","resp","decision","answer"],
        "rt": ["rt","reaction_time","rt_ms","latency"],
        "target_count": ["target_count","k","n_target","n_success","n_hits","count_target"],
        "n_samples": ["n_samples","n","samples_n","sample_count","nsamp","ntrialsample"],
    }.items():
        for o in options:
            if o in cand:
                mapping[key] = cand[o]
                break
    # allow user overrides
    if hints:
        for k,v in hints.items():
            if v is not None:
                mapping[k] = v
    return mapping

def coerce_choice_to_label(val) -> Optional[str]:
    """
    Map heterogeneous choice encodings to {"higher","lower"} when possible.
    Returns None if mapping cannot be determined.
    """
    if val is None: return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"higher","h","+","1","yes","y",">","above","hi","more"}: return "higher"
        if s in {"lower","l","-","0","no","n","<","below","lo","less"}: return "lower"
        return None
    if isinstance(val, (int, float)):
        if val == 1: return "higher"
        if val == 0: return "lower"
    return None
