#!/usr/bin/env python3
"""
00_eda_verify.py -- schema and integrity checks for OPS-SAT-AD (README section 4).

Inputs : data/OPSSAT-AD/dataset.csv, data/OPSSAT-AD/segments.csv
Outputs: outputs/eda_channel_summary.csv, outputs/eda_verify_report.txt
Exit code is non-zero if any hard check fails.

Paths can be overridden with the environment variables
OPSSAT_DATA_DIR and OPSSAT_OUT_DIR.
"""
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("OPSSAT_DATA_DIR", ROOT / "data" / "OPSSAT-AD"))
OUT_DIR = Path(os.environ.get("OPSSAT_OUT_DIR", ROOT / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# channel: (n_normal, n_anomalous) -- from the benchmark, used in the paper
EXPECTED = {
    "CADC0872": (415, 131), "CADC0873": (488, 105), "CADC0874": (125, 69),
    "CADC0888": (192, 60), "CADC0894": (123, 21),
}

lines, failures = [], []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    lines.append(f"[{status}] {name} {detail}".rstrip())
    if not ok:
        failures.append(name)


df = pd.read_csv(DATA_DIR / "dataset.csv")
seg = pd.read_csv(DATA_DIR / "segments.csv", parse_dates=["timestamp"])

check("dataset.csv shape == (2123, 23)", df.shape == (2123, 23), str(df.shape))
check("segments.csv shape == (303493, 8)", seg.shape == (303493, 8), str(seg.shape))
check("anomalous segments == 434", int(df["anomaly"].sum()) == 434)
check("normal segments == 1689", int((df["anomaly"] == 0).sum()) == 1689)
check("one anomaly label per segment", seg.groupby("segment")["anomaly"].nunique().max() == 1)
check("one channel per segment (dataset.csv)", df.groupby("segment")["channel"].nunique().max() == 1)
check("one channel per segment (segments.csv)", seg.groupby("segment")["channel"].nunique().max() == 1)
check("no missing values in segments.csv", int(seg.isna().sum().sum()) == 0)

# dataset.csv vs segments.csv consistency
g = seg.groupby("segment").agg(n=("value", "size"), a=("anomaly", "first"))
m = df.set_index("segment")[["len", "anomaly"]].join(g)
check("segment length matches dataset.csv 'len'", bool((m["len"] == m["n"]).all()))
check("anomaly flag matches between files", bool((m["anomaly"] == m["a"]).all()))

# timestamps sorted within segment?  (analysis sorts anyway; report only)
unsorted = int((seg.groupby("segment")["timestamp"].apply(lambda s: not s.is_monotonic_increasing)).sum())
lines.append(f"[INFO] segments whose rows are not timestamp-sorted in the file: {unsorted} (code sorts explicitly)")

ct = pd.crosstab(df["channel"], df["anomaly"])
ct.columns = ["n_normal", "n_anomalous"]
ct["anomaly_rate"] = (ct["n_anomalous"] / (ct["n_normal"] + ct["n_anomalous"])).round(3)
ct.to_csv(OUT_DIR / "eda_channel_summary.csv")
for ch, (n0, n1) in EXPECTED.items():
    got = (int(ct.loc[ch, "n_normal"]), int(ct.loc[ch, "n_anomalous"]))
    check(f"{ch} counts (normal, anomalous) == {(n0, n1)}", got == (n0, n1), str(got))

lines.append("\nChannel summary:\n" + ct.to_string())
report = "\n".join(lines)
(OUT_DIR / "eda_verify_report.txt").write_text(report + "\n", encoding="utf-8")
print(report)

if failures:
    print(f"\n{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("\ndata OK")
