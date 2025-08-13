from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .utils import ensure_dir

def _savefig(out_dir, name):
    ensure_dir(out_dir)
    path = Path(out_dir) / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)

def psychometric(df: pd.DataFrame, out_dir="data/figures", bins=21) -> str:
    """
    Empirical P(response='higher') vs delta = p - t.
    """
    d = df.copy()
    d["delta_bin"] = pd.cut(d["delta"], bins=np.linspace(-1, 1, bins), include_lowest=True)
    g = d.groupby("delta_bin")["is_higher"].mean().reset_index()
    x = g["delta_bin"].apply(lambda iv: 0.5*(iv.left+iv.right)).to_numpy()
    y = g["is_higher"].to_numpy()

    plt.figure(figsize=(6,4))
    plt.axhline(0.5, ls="--", lw=1, alpha=0.5)
    plt.axvline(0.0, ls="--", lw=1, alpha=0.5)
    plt.plot(x, y, marker="o")
    plt.xlabel("Δ = probability − threshold")
    plt.ylabel("P(response = 'higher')")
    plt.title("Psychometric curve")
    return _savefig(out_dir, "psychometric_delta")

def accuracy_vs_absdelta(df: pd.DataFrame, out_dir="data/figures",
                         edges=(0,0.05,0.1,0.2,0.3,0.4,0.5,1.0)) -> str:
    """
    Mean accuracy with 95% CI (normal approx) across |Δ| bins.
    """
    d = df.copy()
    d["abs_delta_bin"] = pd.cut(d["abs_delta"], bins=np.array(edges), include_lowest=True)
    g = d.groupby("abs_delta_bin")["correct_int"].agg(["mean","size"]).reset_index()
    x = g["abs_delta_bin"].apply(lambda iv: 0.5*(iv.left+iv.right)).to_numpy()
    p = g["mean"].to_numpy()
    n = g["size"].to_numpy()
    se = np.sqrt(p*(1-p)/np.maximum(n,1))
    ci = 1.96*se

    plt.figure(figsize=(6,4))
    plt.errorbar(x, p, yerr=ci, fmt="o-", capsize=3)
    plt.ylim(0,1)
    plt.xlabel("|Δ| = |probability − threshold|")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. boundary distance")
    return _savefig(out_dir, "accuracy_vs_absdelta")

def rt_vs_absdelta(df: pd.DataFrame, out_dir="data/figures",
                   edges=(0,0.05,0.1,0.2,0.3,0.4,0.5,1.0)) -> str:
    """
    Median RT with IQR whiskers across |Δ| bins.
    """
    d = df.copy()
    d["abs_delta_bin"] = pd.cut(d["abs_delta"], bins=np.array(edges), include_lowest=True)
    q = d.groupby("abs_delta_bin")["rt"].quantile([0.25,0.5,0.75]).unstack()
    q = q.rename(columns={0.25:"q1",0.5:"q2",0.75:"q3"}).reset_index()
    x = q["abs_delta_bin"].apply(lambda iv: 0.5*(iv.left+iv.right)).to_numpy()
    med = q["q2"].to_numpy()
    lower = med - q["q1"].to_numpy()
    upper = q["q3"].to_numpy() - med

    plt.figure(figsize=(6,4))
    plt.errorbar(x, med, yerr=[lower, upper], fmt="o-", capsize=3)
    plt.xlabel("|Δ| = |probability − threshold|")
    plt.ylabel("RT (ms), median ± IQR")
    plt.title("RT vs. boundary distance")
    return _savefig(out_dir, "rt_vs_absdelta")

def weighting_curves(fits_df: pd.DataFrame, out_dir="data/figures") -> str:
    """
    Prelec weighting curves per participant + group mean.
    Expects columns m2_alpha, m2_beta.
    """
    p = np.linspace(1e-3, 1-1e-3, 200)
    def w(p, a, b): return np.exp(-b * (-np.log(p))**a)

    plt.figure(figsize=(6,5))
    # individual
    for _, r in fits_df.iterrows():
        plt.plot(p, w(p, r["m2_alpha"], r["m2_beta"]), alpha=0.3)
    # group mean curve (mean of w(p) across participants)
    W = []
    for _, r in fits_df.iterrows():
        W.append(w(p, r["m2_alpha"], r["m2_beta"]))
    if W:
        mean_w = np.mean(np.stack(W, axis=0), axis=0)
        plt.plot(p, mean_w, lw=2.5, label="Group mean")
    plt.plot(p, p, ls="--", lw=1, alpha=0.6, label="Identity")
    plt.xlabel("Objective probability p")
    plt.ylabel("Weighted probability w(p)")
    plt.title("Prelec weighting")
    plt.legend()
    return _savefig(out_dir, "prelec_weighting")
