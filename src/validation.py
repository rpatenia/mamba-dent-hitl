"""Validation-decision logging. Append-only CSV — a saved decision is
never overwritten in place, including across app restarts.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

COLUMNS = ["case_id", "label_type", "reviewer", "result", "notes", "timestamp"]
RESULTS = ("PASS", "NEEDS_CORRECTION", "REJECT")


def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(csv_path)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COLUMNS]


def append_result(csv_path: str, case_id: str, label_type: str, reviewer: str,
                   result: str, notes: str) -> None:
    if result not in RESULTS:
        raise ValueError(f"result must be one of {RESULTS}, got {result!r}")

    row = {
        "case_id": case_id,
        "label_type": label_type,
        "reviewer": reviewer or "unknown",
        "result": result,
        "notes": notes or "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    df_row = pd.DataFrame([row], columns=COLUMNS)
    df_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)


def reviewed_case_label_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    """(case_id, label_type) pairs that have at least one decision logged,
    by any reviewer."""
    if df.empty:
        return set()
    return set(zip(df["case_id"], df["label_type"]))


def summary_stats(df: pd.DataFrame, total_cases: int, label_types: list[str]) -> dict:
    stats = {
        "total_cases": total_cases,
        "reviewed_cases": len({c for c, _ in reviewed_case_label_pairs(df)}),
    }
    stats["remaining_cases"] = max(total_cases - stats["reviewed_cases"], 0)
    for result in RESULTS:
        stats[result] = int((df["result"] == result).sum()) if not df.empty else 0

    by_label = {}
    for lt in label_types:
        sub = df[df["label_type"] == lt] if not df.empty else df
        by_label[lt] = {r: int((sub["result"] == r).sum()) if not sub.empty else 0 for r in RESULTS}
    stats["by_label_type"] = by_label
    return stats
