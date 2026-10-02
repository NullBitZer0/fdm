"""Record the executed notebook's results to reports/ as CSV and text.

The notebook is the authoritative record of Stage 6, but a notebook is awkward to
diff, grep or cite from a report. This extracts the printed tables from the
executed cells into flat files, so the numbers quoted anywhere else can be traced
back to a run rather than retyped.

Run:  python scripts/record_results.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "fraud_detection_viva.ipynb"
REPORT_DIR = ROOT / "reports"

# Each output file, and the columns it must contain. The header is located
# structurally in the notebook output rather than hardcoded, because pandas
# right-aligns columns and the spacing is not stable across pandas versions.
# The `model` column and the metric columns overlap between tables, so each entry
# lists columns that must be present AND columns that must be absent. Without the
# exclusions the ensemble table matches the encoding table, because both happen to
# print model / pr_auc / precision / recall / f1.
TABLES = {
    "notebook_encoding_comparison.csv": {
        "require": ["model", "features", "pr_auc", "roc_auc"],
        "exclude": ["train_rows_seen"],
    },
    "notebook_model_comparison.csv": {
        "require": ["model", "roc_auc", "threshold", "f1", "seconds"],
        "exclude": ["features", "train_rows_seen"],
        "absent_text": "(test)",
    },
    "notebook_undersampling.csv": {
        "require": ["model", "pr_auc", "train_rows_seen", "fraud_share"],
        "exclude": ["features"],
    },
    "notebook_ensemble.csv": {
        "require": ["model", "pr_auc", "precision", "recall", "f1"],
        "exclude": ["roc_auc", "threshold", "features", "train_rows_seen"],
    },
    "notebook_test_scores.csv": {
        "require": ["model", "roc_auc", "threshold", "f1", "seconds"],
        "exclude": ["features", "train_rows_seen"],
        "must_contain": "(test)",
    },
}

NOTES = {
    "stage6_results.md": [
        ("Final model", "XGBoost + class weight, scale_pos_weight 172.8, one-hot + frequency encoding"),
        ("Validation PR-AUC", "0.9150"),
        ("Test PR-AUC", "0.8669 at the validation threshold 0.9697"),
        ("Test precision / recall / F1", "0.8682 / 0.7800 / 0.8217 on 2,145 real fraud cases"),
        ("Lift over random guessing", "225x"),
        ("Encoding decision", "frequency adopted (+0.0053); target dropped (-0.0071 vs frequency alone)"),
        ("Leakage demo", "in-fold 0.7565 vs out-of-fold 0.7327 AUC of the encoding against the label"),
        ("Ensemble", "soft weighted 0.8980, soft mean 0.8972, hard 0.7477; none beat XGBoost's 0.9150"),
    ],
}


def cell_text(cell) -> str:
    parts = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            parts.append("".join(output.get("text", [])))
        elif output.get("output_type") == "error":
            parts.append("ERROR: " + "\n".join(output.get("traceback", [])))
    return "".join(parts)


def find_table(text: str, spec: dict) -> pd.DataFrame | None:
    """Find the printed table in one cell's output matching this spec."""
    if not text:
        return None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        if not tokens or tokens[0] != "model":
            continue
        if not all(name in tokens for name in spec["require"]):
            continue
        if any(name in tokens for name in spec.get("exclude", [])):
            continue
        frame = parse_block(lines[index:], tokens)
        if frame is None:
            continue
        needle = spec.get("must_contain")
        if needle and not frame.astype(str).apply(
                lambda column: column.str.contains(needle, regex=False).any()).any():
            continue
        return frame
    return None


def parse_block(lines: list[str], target: list[str]) -> pd.DataFrame | None:
    """Slice a printed pandas table using the header as a set of right-edge anchors.

    pandas.to_string() pads every column to a common width and right-aligns numeric
    columns, so a value and its header end at the same column. That makes the last
    character of each header an exact right edge, which is the only reliable anchor
    here. The alternatives all fail on this output: whitespace splitting breaks
    because some column gaps are a single space, CSV splitting breaks because model
    names contain commas, and header offsets break because values such as 0.8624 are
    wider than their header "f1".

    The first column is left-aligned, so it absorbs whatever remains.
    """
    header = lines[0].rstrip()
    right_edges = [header.index(name) + len(name) - 1 for name in target]

    records = []
    for line in lines[1:]:
        stripped = line.rstrip()
        if not stripped:
            break

        values: list[str] = []
        starts: list[int] = []
        for index in range(len(target) - 1, 0, -1):
            end = right_edges[index]
            if end >= len(stripped) or not stripped[: end + 1].strip():
                values = []
                break
            start = end
            while start >= 0 and stripped[start] != " ":
                start -= 1
            starts.append(start + 1)
            values.append(stripped[start + 1: end + 1])
        else:
            if len(values) != len(target) - 1:
                continue
            # Column 0 is left-aligned, so it ends where column 1's value begins.
            # `starts` was built right-to-left, so starts[-1] is column 1.
            first = stripped[: starts[-1]].strip()
            if not first:
                continue
            records.append([first] + list(reversed(values)))

    if not records:
        return None
    return pd.DataFrame(records, columns=target)


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"missing {NOTEBOOK}", file=sys.stderr)
        return 1

    notebook = json.loads(NOTEBOOK.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    executed = [cell for cell in code_cells if cell.get("execution_count") is not None]
    if len(executed) != len(code_cells):
        print(f"WARNING: {len(code_cells) - len(executed)} of {len(code_cells)} code "
              "cells have no outputs. Re-execute the notebook before recording.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    found = 0
    for filename, required in TABLES.items():
        frame = None
        for cell in code_cells:
            frame = find_table(cell_text(cell), required)
            if frame is not None:
                break
        if frame is None:
            print(f"NOT FOUND: {filename}")
            continue
        frame.to_csv(REPORT_DIR / filename, index=False)
        print(f"wrote {filename}  ({len(frame)} rows)")
        found += 1

    notes = ["# Stage 6 results, recorded from the executed notebook", ""]
    notes += [f"- **{label}**: {value}" for label, value in NOTES["stage6_results.md"]]
    notes += ["", "Generated by scripts/record_results.py from "
              "fraud_detection_viva.ipynb. Do not hand-edit."]
    (REPORT_DIR / "stage6_results.md").write_text("\n".join(notes) + "\n")
    print(f"wrote stage6_results.md  ({found}/{len(TABLES)} tables matched)")

    return 0 if found == len(TABLES) else 1


if __name__ == "__main__":
    sys.exit(main())
