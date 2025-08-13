from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy import stats
try:
    import statsmodels.api as sm
    from statsmodels.stats.anova import AnovaRM
except Exception:
    sm = None
    AnovaRM = None

from .utils import make_abs_delta_bins

def aggregate_by_abs_delta_bins(
    df: pd.DataFrame,
    abs_delta_edges=(0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00),
    participant_col="participantId",
) -> pd.DataFrame:
    """
    Returns long-format per-participant, per-bin aggregates:
      - rt_med: median RT per bin
      - acc_mean: mean accuracy per bin
      - n: trials per bin
    """
    d = df.copy()
    d["abs_delta_bin"] = make_abs_delta_bins(d, edges=abs_delta_edges)
    g = d.groupby([participant_col, "abs_delta_bin"])
    out = g.agg(
        rt_med=("rt", "median"),
        acc_mean=("correct_int", "mean"),
        n=("correct_int", "size")
    ).reset_index()
    # clean category labels for readability
    out["abs_delta_bin"] = out["abs_delta_bin"].astype(str)
    return out

def _pivot_for_rm(data: pd.DataFrame, dv_col: str, subject_col: str, within_col: str):
    """
    Produces a matrix shape (n_subjects, k_levels) needed for Friedman,
    ensuring aligned subjects and within levels.
    """
    wide = data.pivot_table(
        index=subject_col,
        columns=within_col,
        values=dv_col,
        aggfunc="first"
    )
    # Drop rows with any missing levels to keep tests balanced
    wide = wide.dropna(axis=0, how="any")
    return wide

def friedman_test(
    data: pd.DataFrame,
    dv_col: str,
    subject_col="participantId",
    within_col="abs_delta_bin"
):
    """
    Friedman test across within-factor levels using per-subject summaries.
    Returns dict with chi2, p, n_subjects, k_levels, and Kendall's W.
    """
    wide = _pivot_for_rm(data, dv_col, subject_col, within_col)
    if wide.shape[0] < 2 or wide.shape[1] < 2:
        return {"ok": False, "msg": "Not enough data for Friedman.", "n": wide.shape[0], "k": wide.shape[1]}
    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    n, k = wide.shape
    W = stat / (n * (k - 1)) if k > 1 and n > 0 else np.nan
    return {
        "ok": True,
        "chi2": float(stat),
        "p": float(p),
        "n_subjects": int(n),
        "k_levels": int(k),
        "kendalls_W": float(W),
        "levels": list(wide.columns.astype(str))
    }

def pairwise_wilcoxon_holm(
    data: pd.DataFrame,
    dv_col: str,
    subject_col="participantId",
    within_col="abs_delta_bin"
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank across within-levels with Holm correction.
    Returns a dataframe with (level_i, level_j, stat, p_raw, p_holm).
    """
    wide = _pivot_for_rm(data, dv_col, subject_col, within_col)
    levels = list(wide.columns)
    rows = []
    for i, j in itertools.combinations(range(len(levels)), 2):
        a, b = wide.iloc[:, i], wide.iloc[:, j]
        # Only keep pairs without NaNs for both a and b
        mask = ~(a.isna() | b.isna())
        if mask.sum() < 2:
            continue
        stat, p = stats.wilcoxon(a[mask], b[mask], zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        rows.append({"level_i": str(levels[i]), "level_j": str(levels[j]), "stat": float(stat), "p_raw": float(p)})
    if not rows:
        return pd.DataFrame(rows)
    dfp = pd.DataFrame(rows)
    # Holm correction
    m = len(dfp)
    dfp = dfp.sort_values("p_raw").reset_index(drop=True)
    dfp["p_holm"] = [min(1.0, p * (m - idx)) for idx, p in enumerate(dfp["p_raw"])]
    # Return in original (i,j) order but keep p_holm values
    dfp = dfp.sort_values(["level_i", "level_j"]).reset_index(drop=True)
    return dfp

def rm_anova(
    data: pd.DataFrame,
    dv_col: str,
    subject_col="participantId",
    within_col="abs_delta_bin"
):
    """
    Repeated-measures ANOVA via statsmodels (if available).
    Expects long-format rows: [subject, within, dv]
    Returns dict: ok, table (as DataFrame) or msg if statsmodels unavailable.
    """
    if AnovaRM is None:
        return {"ok": False, "msg": "statsmodels not installed; rm_anova unavailable."}
    # Drop any missing values in dv
    d = data[[subject_col, within_col, dv_col]].dropna()
    # Keep only subjects with complete data across within levels
    wide = _pivot_for_rm(d, dv_col, subject_col, within_col)
    d = d[d[subject_col].isin(wide.index)]
    aov = AnovaRM(d, depvar=dv_col, subject=subject_col, within=[within_col]).fit()
    table = aov.anova_table.reset_index().rename(columns={"index": "effect"})
    return {"ok": True, "table": table}
