#!/usr/bin/env python3
"""Preprocessing for Parkinson's early detection (ds005892, raw BIDS).

Minimal, documented pipeline (no fMRIPrep derivatives exist for this release):
  1. Motion correction SKIPPED by default on this run: ANTs rigid motion
     correction measured 327 s for one subject on this 1.5-vCPU VM
     (~5.5 min/subject, ~5 h for 55 subjects before any registration).
     Skipping it is a documented limitation — head motion is NOT corrected
     and no motion regressors exist. Bandpass filtering + parcel averaging
     mitigate some motion noise, but motion remains a confound.
  2. Affine normalisation to MNI152: BOLD-mean -> T1 (rigid), T1 -> MNI (affine)
     NOTE: affine-only (no nonlinear SyN) — a documented limitation, chosen
     because this VM has no GPU/FSL and SyN on 55 subjects would take many
     hours. Atlas-based FC after affine normalisation is standard practice
     for a quick pipeline but less precise than nonlinear normalisation.
  3. Nilearn NiftiLabelsMasker, Schaefer 2018 100 parcels / 7 networks:
     6 mm smoothing, bandpass 0.01-0.1 Hz, TR=2 s, z-score standardisation.
     No confound regressors are available (no motion params in raw BIDS and
     no fMRIPrep confounds.tsv); this is logged as a limitation.
  4. Pearson functional-connectivity matrix (100x100) per subject.

Outputs: training/features/<sub>_fc.npy (Pearson r), <sub>_fc_fisherz.npy,
training/features/labels.csv (subject, group, label3, label_bin).
Logs: training/logs/preprocessing.log
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent / "features"
LOG = Path(__file__).resolve().parent / "logs" / "preprocessing.log"

GROUP_TO_LABEL3 = {"Control": 0, "PD-NC": 1, "PD-MCI": 2}
LABEL3_NAMES = ["HC", "PD-NC", "PD-MCI"]


def setup_logging():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("preprocess")


def load_participants(raw_dir):
    import csv
    rows = []
    with open(raw_dir / "participants.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subset of subject ids, e.g. sub-MJF001")
    ap.add_argument("--skip-motion", action="store_true",
                    help="skip ANTs motion correction (faster, less clean)")
    ap.add_argument("--skip-normalize", action="store_true",
                    help="skip MNI normalisation (atlas applied in native space - NOT valid, testing only)")
    args = ap.parse_args()
    log = setup_logging()
    t0 = time.time()

    import nibabel as nib
    from nilearn import datasets, image
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure

    try:
        import ants
        have_ants = True
    except ImportError:
        have_ants = False
        log.warning("ANTsPy not available - normalisation will be skipped")

    log.info("=== Preprocessing ds005892 (raw BIDS, no fMRIPrep derivatives) ===")
    log.info("LIMITATION: affine-only MNI normalisation; motion correction "
             "skipped (327 s/subject on this VM); no confound regressors "
             "(no motion params in raw BIDS). See module docstring.")

    # Schaefer 100-parcel atlas (MNI space) — LOCAL copy in training/atlas/.
    # (nilearn's fetch_atlas_schaefer_2018 fails on this VM: the proxy's
    # self-signed cert breaks requests SSL verification to
    # raw.githubusercontent.com. The same files were downloaded with curl
    # and verified: 91x109x91, 2mm, labels 0-100.)
    log.info("Loading local Schaefer 2018 100-parcel atlas ...")
    atlas_path = Path(__file__).resolve().parent / "atlas" / \
        "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    labels_img = str(atlas_path)
    log.info("Atlas: %s", labels_img)

    mni_template = None
    if have_ants and not args.skip_normalize:
        log.info("Loading MNI152 template via ANTsPy ...")
        mni_template = ants.image_read(ants.get_ants_data("mni"))
        # Resample to 2 mm: applying the BOLD transform at 1 mm would create a
        # 182x218x182x200 float image (~5.8 GB) and OOM this VM (killed test 3).
        # 2 mm matches the Schaefer atlas resolution.
        mni_template = ants.resample_image(mni_template, (2, 2, 2),
                                           use_voxels=False, interp_type=1)
        log.info("MNI template (2mm) shape %s spacing %s",
                 mni_template.shape, mni_template.spacing)

    participants = load_participants(args.raw)
    if args.subjects:
        participants = [p for p in participants
                        if p["participant_id"] in args.subjects]
    log.info("Subjects to process: %d", len(participants))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = []
    for p in participants:
        sid = p["participant_id"]
        grp = p["group"]
        label3 = GROUP_TO_LABEL3[grp]
        label_bin = 0 if grp == "Control" else 1
        # Resume support: skip subjects already processed (crash-safe).
        if (out_dir / f"{sid}_fc.npy").exists():
            log.info("[%s] already processed, skipping", sid)
            import csv as _csv2
            rows.append({"subject": sid, "group": grp, "label3": label3,
                         "label_bin": label_bin,
                         "n_vol": int(nib.load(
                             str(args.raw / sid / "func" /
                                 f"{sid}_task-rest_bold.nii.gz")).shape[3])})
            continue
        ts = time.time()
        try:
            bold_path = args.raw / sid / "func" / f"{sid}_task-rest_bold.nii.gz"
            t1_path = args.raw / sid / "anat" / f"{sid}_T1w.nii.gz"
            bold_img = nib.load(str(bold_path))
            t1_img = nib.load(str(t1_path))
            n_vol = bold_img.shape[3]
            log.info("[%s] %s n_vol=%d bold_shape=%s t1_shape=%s",
                     sid, grp, n_vol, bold_img.shape, t1_img.shape)

            bold_proc = bold_img
            if have_ants and not args.skip_normalize:
                # 1) motion correction (rigid, mean reference)
                if not args.skip_motion:
                    log.info("[%s] motion correction ...", sid)
                    bold_ants = ants.image_read(str(bold_path))
                    mc = ants.motion_correction(bold_ants, fixed=None,
                                                more_accurate=0)
                    bold_proc = mc["motion_corrected"]
                    log.info("[%s] motion correction done (%.1fs)",
                             sid, time.time() - ts)
                else:
                    bold_ants = ants.image_read(str(bold_path))
                    bold_proc = bold_ants
                # 2) BOLD-mean -> T1 (rigid), T1 -> MNI (affine)
                log.info("[%s] normalisation (bold->T1 rigid, T1->MNI affine) ...", sid)
                mean_bold = ants.get_average_of_timeseries(bold_proc) \
                    if hasattr(bold_proc, "shape") and len(bold_proc.shape) == 4 \
                    else None
                # ants image objects: use image_read results directly
                if not args.skip_motion:
                    mean_bold = ants.get_average_of_timeseries(mc["motion_corrected"])
                else:
                    mean_bold = ants.get_average_of_timeseries(bold_ants)
                t1_ants = ants.image_read(str(t1_path))
                reg_b2t = ants.registration(fixed=t1_ants, moving=mean_bold,
                                            type_of_transform="Rigid")
                reg_t2m = ants.registration(fixed=mni_template, moving=t1_ants,
                                            type_of_transform="Affine")
                bold_mni = ants.apply_transforms(
                    fixed=mni_template, moving=bold_proc,
                    transformlist=reg_t2m["fwdtransforms"] + reg_b2t["fwdtransforms"],
                    interpolator="linear", imagetype=3)  # 3 = 4D time series
                # Build the nibabel (RAS) affine from the ANTs image geometry.
                # ANTs stores direction/spacing/origin in LPS; RAS = F @ LPS
                # with F = diag(-1,-1,1). (Verified: reproduces the standard
                # MNI152 2mm affine [[-2,0,0,90],[0,2,0,-126],[0,0,2,-72]].)
                # A disk round-trip via ants.image_write was tried and did not
                # create the file for the 4D image, so no round-trip is used.
                # 4D images carry a 4x4 direction / 4-vector spacing+origin;
                # use the first 3 (spatial) components only.
                D = np.array(bold_mni.direction, dtype=float)[:3, :3]
                s = np.array(bold_mni.spacing, dtype=float)[:3]
                o = np.array(bold_mni.origin, dtype=float)[:3]
                A_lps = np.eye(4)
                A_lps[:3, :3] = D @ np.diag(s)
                A_lps[:3, 3] = o
                F = np.diag([-1.0, -1.0, 1.0, 1.0])
                bold_arr = bold_mni.numpy()
                # Free ANTs intermediates before the memory-heavy masking step.
                del bold_mni, reg_b2t, reg_t2m
                try:
                    del bold_ants, t1_ants, mean_bold, mc
                except NameError:
                    pass
                import gc
                gc.collect()
                bold_proc = nib.Nifti1Image(bold_arr, affine=F @ A_lps)
                del bold_arr
                log.info("[%s] normalisation done (%.1fs)", sid, time.time() - ts)
            elif args.skip_normalize:
                log.warning("[%s] SKIP-NORMALIZE: atlas applied in native space "
                            "(invalid for group analysis - testing only)", sid)

            # 3) parcel time series
            masker = NiftiLabelsMasker(
                labels_img=labels_img, standardize="zscore_sample",
                smoothing_fwhm=6, low_pass=0.1, high_pass=0.01, t_r=2.0,
                detrend=True, dtype="float32", verbose=0)
            ts_series = masker.fit_transform(bold_proc)
            log.info("[%s] time series %s (%.1fs)", sid, ts_series.shape,
                     time.time() - ts)
            # 4) Pearson FC
            measure = ConnectivityMeasure(kind="correlation")
            fc = measure.fit_transform([ts_series])[0]
            fc = np.nan_to_num(fc, nan=0.0)
            np.fill_diagonal(fc, 1.0)
            fisherz = np.arctanh(np.clip(fc, -0.9999, 0.9999))
            np.save(out_dir / f"{sid}_fc.npy", fc)
            np.save(out_dir / f"{sid}_fc_fisherz.npy", fisherz)
            rows.append({"subject": sid, "group": grp, "label3": label3,
                         "label_bin": label_bin, "n_vol": n_vol})
            log.info("[%s] OK fc saved, mean|r|=%.3f (total %.1fs)",
                     sid, np.mean(np.abs(fc[np.triu_indices(100, 1)])),
                     time.time() - ts)
        except Exception as e:
            log.exception("[%s] FAILED: %s", sid, e)
            failed.append(sid)

    import csv as _csv
    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["subject", "group", "label3",
                                           "label_bin", "n_vol"])
        w.writeheader()
        w.writerows(rows)
    log.info("=== DONE: %d ok, %d failed (%s) in %.1f min ===",
             len(rows), len(failed), failed, (time.time() - t0) / 60)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
