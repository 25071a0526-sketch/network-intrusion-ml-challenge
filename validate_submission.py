#!/usr/bin/env python3
"""Validate a saved prediction CSV against competition test/sample files."""
import argparse
from pathlib import Path
import json
import pandas as pd

LABELS = {"Benign", "Availability", "Reconnaissance", "Spoofing"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--sample", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    sub, test, sample = (pd.read_csv(a.submission), pd.read_csv(a.test), pd.read_csv(a.sample))
    checks = {
        "columns_exactly_id_Target": sub.columns.tolist() == ["id", "Target"],
        "row_count_matches_test": len(sub) == len(test),
        "row_count_matches_sample": len(sub) == len(sample),
        "ids_match_test_order": len(sub) == len(test) and sub["id"].equals(test["id"]),
        "ids_match_sample_order": len(sub) == len(sample) and sub["id"].equals(sample["id"]),
        "ids_unique": sub["id"].is_unique,
        "no_nulls": not sub.isna().any().any(),
        "labels_valid": set(sub["Target"].dropna().unique()).issubset(LABELS),
    }
    report = {
        "submission": str(a.submission),
        "rows": int(len(sub)),
        "predicted_class_counts": {str(k): int(v) for k, v in sub["Target"].value_counts().items()},
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if a.output:
        a.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["all_pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
