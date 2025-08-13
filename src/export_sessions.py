# src/export_sessions.py
import os
import json
import argparse
from datetime import datetime
import pandas as pd # type: ignore
from firestore_admin import get_db

def fetch_sessions():
    db = get_db()
    sessions = []
    for doc in db.collection('sessions').stream():
        d = doc.to_dict() or {}
        d['__doc_id'] = doc.id
        sessions.append(d)
    return sessions

def flatten_trials(sessions: list[dict]) -> pd.DataFrame:
    rows = []
    for sess in sessions:
        sid = sess.get('sessionId') or sess.get('__doc_id')
        pid = sess.get('participantId')
        trials = sess.get('trials') or []
        for idx, t in enumerate(trials):
            row = {'sessionId': sid, 'participantId': pid, 'trial_index': idx}
            row.update(t or {})
            rows.append(row)
    return pd.DataFrame(rows)

def main(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    sessions = fetch_sessions()
    # Save raw JSON
    with open(os.path.join(outdir, f"sessions_{ts}.json"), "w") as f:
        json.dump(sessions, f, indent=2)

    # Save flattened trials CSV
    df = flatten_trials(sessions)
    csv_path = os.path.join(outdir, f"trials_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Exported {len(sessions)} sessions and {len(df)} trials → {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/exports", help="Where to write JSON/CSV")
    args = parser.parse_args()
    main(args.outdir)
