# src/fetch_data.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.utils import add_derived_columns, make_probability_bins, make_abs_delta_bins, ensure_dir

def load_cleaned_trials(path_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    return add_derived_columns(df)

def summarize(df: pd.DataFrame) -> dict:
    out = {
        "participants": int(df["participantId"].nunique()),
        "trials_total": int(len(df)),
        "trials_per_session": df.groupby(["participantId", "sessionId"]).size().describe().to_dict(),
        "accuracy_overall": float(df["correct_int"].mean()),
        "rt_ms_mean": float(df["rt"].mean()),
        "rt_ms_median": float(df["rt"].median()),
    }
    pbin = make_probability_bins(df)
    out["acc_by_p_bin"] = df.groupby(pbin)["correct_int"].mean().to_dict()
    dbin = make_abs_delta_bins(df)
    out["rt_median_by_abs_delta"] = df.groupby(dbin)["rt"].median().to_dict()
    out["acc_by_abs_delta"] = df.groupby(dbin)["correct_int"].mean().to_dict()
    return out

def save_artifacts(df: pd.DataFrame, out_dir: str | Path = "data/derived"):
    """
    CSV-only saver to avoid pyarrow/Parquet issues.
    """
    ensure_dir(out_dir)
    Path(out_dir, "trials_clean.csv").write_text(df.to_csv(index=False))
    by_pid = df.groupby("participantId").agg(
        n=("correct_int", "size"),
        acc=("correct_int", "mean"),
        rt_med=("rt", "median"),
    ).reset_index()
    by_pid.to_csv(Path(out_dir, "participant_summary.csv"), index=False)
    return {"parquet_written": False}

if __name__ == "__main__":
    cleaned_csv = "data/exports/trials_20250811T043446Z_cleaned.csv"
    df = load_cleaned_trials(cleaned_csv)
    print("Summary:", summarize(df))
    print(save_artifacts(df))
    print("Wrote CSV artifacts to data/derived/")
