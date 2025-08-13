#!/usr/bin/env python3
"""
postprocess_data.py

Cleans Firestore export data by:
  1) Removing entire sessions that do NOT have the full number of non-practice trials.
     - "Full" is inferred as the mode (most common) number of non-practice trials per session.
     - You can override this with --expected-trials if you already know the correct count.
  2) Removing practice trials (practice == true) from the remaining sessions.

Inputs default to the most recent files in the current directory that match:
  - sessions_*.json
  - trials_*.csv

Outputs:
  - <sessions_stem>_cleaned.json
  - <trials_stem>_cleaned.csv
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd # pyright: ignore[reportMissingModuleSource]


def _latest(path_glob: str) -> Path:
    paths = sorted(Path(".").glob(path_glob))
    if not paths:
        sys.exit(f"ERROR: No files found for pattern: {path_glob}")
    return paths[-1]


def _coerce_bool(series):
    # Accepts real bools, 0/1, or strings like "true"/"false"
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def infer_expected_trials(trials_df: pd.DataFrame) -> int:
    """
    Infer the expected (full) number of non-practice trials per session
    using the mode across sessions. If there's a tie, choose the largest.
    """
    if "sessionId" not in trials_df.columns:
        sys.exit("ERROR: trials CSV must contain a 'sessionId' column.")

    has_practice = "practice" in trials_df.columns
    practice = _coerce_bool(trials_df["practice"]) if has_practice else pd.Series(False, index=trials_df.index)

    counts = (
        trials_df.loc[~practice]
        .groupby("sessionId")
        .size()
        .to_dict()
    )
    if not counts:
        sys.exit("ERROR: Couldn't compute non-practice counts; is the data empty?")

    # Mode of counts; break ties by choosing the maximum
    freq = Counter(counts.values())
    max_freq = max(freq.values())
    modal_counts = [k for k, v in freq.items() if v == max_freq]
    expected = max(modal_counts)
    return expected


def find_full_sessions(trials_df: pd.DataFrame, expected_trials: int) -> set:
    has_practice = "practice" in trials_df.columns
    practice = _coerce_bool(trials_df["practice"]) if has_practice else pd.Series(False, index=trials_df.index)
    non_practice_counts = (
        trials_df.loc[~practice]
        .groupby("sessionId")
        .size()
    )
    full_sessions = set(non_practice_counts[non_practice_counts == expected_trials].index)
    return full_sessions


def clean_trials(trials_df: pd.DataFrame, keep_sessions: set) -> pd.DataFrame:
    # Drop sessions not in keep set, then drop practice trials
    out = trials_df.copy()
    has_practice = "practice" in out.columns
    if has_practice:
        out["practice"] = _coerce_bool(out["practice"])
    else:
        out["practice"] = False

    out = out[out["sessionId"].isin(keep_sessions)]
    out = out[~out["practice"]].drop(columns=["practice"], errors="ignore")
    return out


def load_sessions_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit("ERROR: sessions JSON must be a list of session objects.")
    return data


def clean_sessions_json(sessions_list: list, keep_sessions: set) -> list:
    """
    sessions_list: [
      {
        "sessionId": "...",
        "participantId": "...",
        "trials": [ { ... "practice": true/false ... }, ... ],
        ...
      },
      ...
    ]
    """
    cleaned = []
    for sess in sessions_list:
        sid = sess.get("sessionId")
        if sid in keep_sessions:
            trials = sess.get("trials", [])
            # Drop practice trials; keep others as-is
            filtered_trials = [t for t in trials if not bool(t.get("practice", False))]
            new_sess = dict(sess)
            new_sess["trials"] = filtered_trials
            cleaned.append(new_sess)
    return cleaned


def main():
    ap = argparse.ArgumentParser(description="Postprocess Firestore export data.")
    ap.add_argument("--sessions", type=Path, default=None, help="Path to sessions_*.json")
    ap.add_argument("--trials", type=Path, default=None, help="Path to trials_*.csv")
    ap.add_argument("--expected-trials", type=int, default=None,
                    help="Override the inferred full number of non-practice trials per session.")
    args = ap.parse_args()

    sessions_path = args.sessions or _latest("sessions_*.json")
    trials_path = args.trials or _latest("trials_*.csv")

    print(f"Using sessions file: {sessions_path}")
    print(f"Using trials   file: {trials_path}")

    # Load trials CSV
    trials_df = pd.read_csv(trials_path)
    if "sessionId" not in trials_df.columns:
        sys.exit("ERROR: trials CSV missing 'sessionId' column.")

    # Determine expected non-practice count
    expected = args.expected_trials or infer_expected_trials(trials_df)
    print(f"Inferred expected non-practice trials per session: {expected}")

    # Identify sessions with a full set of non-practice trials
    keep_sessions = find_full_sessions(trials_df, expected)
    all_sessions = set(trials_df["sessionId"].unique())
    drop_sessions = all_sessions - keep_sessions

    print(f"Sessions (total): {len(all_sessions)}")
    print(f"Sessions kept   : {len(keep_sessions)}")
    print(f"Sessions dropped: {len(drop_sessions)}")

    # Clean trials and write CSV
    cleaned_trials = clean_trials(trials_df, keep_sessions)
    trials_out = trials_path.with_name(f"{trials_path.stem}_cleaned.csv")
    cleaned_trials.to_csv(trials_out, index=False)
    print(f"Wrote cleaned trials CSV: {trials_out} (rows: {len(cleaned_trials)})")

    # Clean sessions JSON and write JSON
    sessions_list = load_sessions_json(sessions_path)
    cleaned_sessions = clean_sessions_json(sessions_list, keep_sessions)
    sessions_out = sessions_path.with_name(f"{sessions_path.stem}_cleaned.json")
    with sessions_out.open("w", encoding="utf-8") as f:
        json.dump(cleaned_sessions, f, ensure_ascii=False, indent=2)
    print(f"Wrote cleaned sessions JSON: {sessions_out} (sessions: {len(cleaned_sessions)})")


if __name__ == "__main__":
    main()
