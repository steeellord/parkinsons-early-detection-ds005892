#!/usr/bin/env python3
"""Merge the two v2 preprocessing halves into one features_v2/ folder.

Run A wrote features_v2a/ (28 subjects), run B wrote features_v2b/
(27 subjects). This copies every subject file into features_v2/,
merges labels.csv / labels_qc.csv (sorted by subject), and checks the
count is 55 with no missing atlas files. Idempotent: safe to re-run.
"""
import csv, shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "features_v2"
PARTS = [BASE / "features_v2a", BASE / "features_v2b"]


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "motion").mkdir(exist_ok=True)
    labels, qc = [], []
    for part in PARTS:
        for f in sorted(part.glob("sub-*_fc*.npy")):
            shutil.copy2(f, OUT / f.name)
        for f in sorted((part / "motion").glob("*.csv")):
            shutil.copy2(f, OUT / "motion" / f.name)
        with open(part / "labels.csv") as fh:
            labels.extend(list(csv.DictReader(fh)))
        qc_path = part / "labels_qc.csv"
        if qc_path.exists():
            with open(qc_path) as fh:
                qc.extend(list(csv.DictReader(fh)))
    labels.sort(key=lambda r: r["subject"])
    qc.sort(key=lambda r: r["subject"])
    with open(OUT / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "group", "label3",
                                           "label_bin", "n_vol"])
        w.writeheader()
        w.writerows(labels)
    with open(OUT / "labels_qc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "group", "fd_mean_mm",
                                           "fd_max_mm", "n_scrubbed", "n_kept"])
        w.writeheader()
        w.writerows(qc)
    n_fc100 = len(list(OUT.glob("sub-*_fc100.npy")))
    n_fc200 = len(list(OUT.glob("sub-*_fc200.npy")))
    n_motion = len(list((OUT / "motion").glob("*.csv")))
    print(f"subjects in labels.csv: {len(labels)}")
    print(f"fc100 files: {n_fc100}, fc200 files: {n_fc200}, "
          f"motion csvs: {n_motion}")
    assert len(labels) == 55, "expected 55 subjects"
    assert n_fc100 == 55 and n_fc200 == 55, "missing FC files"
    assert n_motion == 55, "missing motion records"
    print("MERGE OK: 55/55 subjects, both atlases, motion records complete")


if __name__ == "__main__":
    main()
