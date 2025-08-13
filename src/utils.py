from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns used across the pipeline:
      - delta = probability - threshold
      - abs_delta = |delta|
      - is_higher = 1 if response == 'higher' else 0
      - correct_int = 1/0
      - difficulty ~ proximity to boundary (higher = harder)
    """
    out = df.copy()
    if "probability" not in out or "threshold" not in out:
        raise ValueError("Expected columns 'probability' and 'threshold'.")
    out["delta"] = out["probability"] - out["threshold"]
    out["abs_delta"] = out["delta"].abs()
    out["is_higher"] = (out["response"].astype(str).str.lower() == "higher").astype(int)
    out["correct_int"] = out["correct"].astype(int)
    # Difficulty proxy in [0,1], highest at boundary:
    out["difficulty"] = 1.0 - out["abs_delta"].clip(0, 1)
    # Clean obvious RT outliers (optional, light winsorization)
    if "rt" in out:
        out["rt"] = pd.to_numeric(out["rt"], errors="coerce")
        q_low, q_hi = out["rt"].quantile([0.01, 0.99])
        out["rt_win"] = out["rt"].clip(q_low, q_hi)
    return out

def make_probability_bins(df: pd.DataFrame, step: float = 0.1) -> pd.Series:
    edges = np.arange(0.0, 1.0 + step, step)
    return pd.cut(df["probability"], bins=edges, include_lowest=True)

def make_abs_delta_bins(df: pd.DataFrame, edges=(0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)) -> pd.Series:
    return pd.cut(df["abs_delta"], bins=np.array(edges), include_lowest=True)
