#!/usr/bin/env python3
"""Build pooled binary (HC vs PD) Schaefer-100 FC features from the
BrainNet Auckland derived sets (PPMI, Neurocon, Tao Wu).

These sets ship per-subject Schaefer-100 timeseries (.mat). FCs are
recomputed here with plain Pearson correlation + Fisher-z — the same
step preprocess_v2 uses for atlas-100 — instead of trusting the
bundled correlation matrices (checked 9 Nov: bundled matrices differ
from corrcoef(timeseries) by up to 0.06, so they were made some other
way). Using our own step keeps pooled features comparable.

Labels come from the BrainNet Metadata_v5 CSVs. PPMI Prodromal and
SWEDD subjects are written out with group labels kept but excluded
from the binary labels file (they are neither HC nor PD); a separate
all-labels file keeps them for later use.

Outputs (training/features_pooled/):
  <sub>_fc100.npy, <sub>_fc100_fisherz.npy per subject
  labels.csv        binary pool (HC=0, PD=1) with a `source` column
  labels_all.csv    every subject incl. Prodromal/SWEDD
  build_log.txt     counts per source/group
"""
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data" / "brainnet-figshare"
OUT = BASE / "features_pooled"

# zip name -> (metadata csv name, id prefix mapping)
SOURCES = {
    "ppmi_v2.zip": "PPMI_metadata.csv",
    "neurocon.zip": "Neurocon_metadata.csv",
    "taowu.zip": "TaoWu_metadata.csv",
}


def load_groups():
    """Return {source_zip: {subject_number: group}} from Metadata_v5."""
    groups = {}
    with zipfile.ZipFile(DATA / "Metadata_v5.zip") as z:
        for zip_name, csv_name in SOURCES.items():
            rows = z.read(f"Metadata_v5/{csv_name}").decode().splitlines()
            per_subject = {}
            for r in csv.DictReader(rows):
                # PPMI csv has one row per image; Neurocon/TaoWu one per subject
                key = subject_number(r["Subject"])
                per_subject.setdefault(key, set()).add(r["Group"])
            groups[zip_name] = {k: next(iter(v)) for k, v in per_subject.items()}
    return groups


def subject_number(zip_sub):
    """'patient3332' / 'control3351' / 'prodromal101523' / 'swedd3101'
    -> bare subject number, matching the metadata Subject column."""
    for p in ("patient", "control", "prodromal", "swedd"):
        if zip_sub.startswith(p):
            return zip_sub[len(p):]
    return zip_sub


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = load_groups()
    rows_bin, rows_all = [], []
    log = []
    for zip_name in SOURCES:
        src = zip_name.replace(".zip", "")
        with zipfile.ZipFile(DATA / zip_name) as z:
            names = [n for n in z.namelist()
                     if n.endswith("_schaefer100_features_timeseries.mat")]
            n_ok = 0
            for name in sorted(names):
                # ppmi/sub-patient3332/sub-patient3332_schaefer100_....mat
                parts = name.split("/")
                zip_sub = parts[1].replace("sub-", "")
                num = subject_number(zip_sub)
                grp = groups[zip_name].get(num)
                if grp is None:
                    log.append(f"{src} {zip_sub}: no group in metadata, skipped")
                    continue
                ts = loadmat(io.BytesIO(z.read(name)))["data"]
                if ts.ndim != 2 or ts.shape[1] != 100:
                    log.append(f"{src} {zip_sub}: bad shape {ts.shape}, skipped")
                    continue
                ts = np.asarray(ts, dtype=np.float64)
                fc = np.corrcoef(ts.T)
                fc = np.nan_to_num(fc, nan=0.0)
                np.fill_diagonal(fc, 1.0)
                fz = np.arctanh(np.clip(fc, -0.9999, 0.9999))
                sid = f"{src}-{zip_sub}"
                np.save(OUT / f"{sid}_fc100.npy", fc.astype(np.float32))
                np.save(OUT / f"{sid}_fc100_fisherz.npy", fz.astype(np.float32))
                n_ok += 1
                grp_norm = {"Control": "Control", "PD": "PD"}.get(grp, grp)
                rows_all.append({"subject": sid, "group": grp_norm,
                                 "label3": "", "label_bin": "",
                                 "n_vol": int(ts.shape[0]), "source": src})
                if grp_norm == "Control":
                    rows_bin.append({"subject": sid, "group": "Control",
                                     "label3": 0, "label_bin": 0,
                                     "n_vol": int(ts.shape[0]), "source": src})
                elif grp_norm == "PD":
                    rows_bin.append({"subject": sid, "group": "PD",
                                     "label3": 1, "label_bin": 1,
                                     "n_vol": int(ts.shape[0]), "source": src})
        log.append(f"{src}: {n_ok} subjects written")
    fields = ["subject", "group", "label3", "label_bin", "n_vol", "source"]
    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_bin)
    with open(OUT / "labels_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_all)
    from collections import Counter
    log.append(f"binary pool: {len(rows_bin)} "
               f"({Counter(r['group'] for r in rows_bin)})")
    log.append(f"all: {len(rows_all)} "
               f"({Counter(r['group'] for r in rows_all)})")
    (OUT / "build_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
