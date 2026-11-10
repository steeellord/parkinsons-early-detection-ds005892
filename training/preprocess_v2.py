#!/usr/bin/env python3
"""Preprocessing v2 for Parkinson's early detection (ds005892, raw BIDS).

Pipeline per subject (v2 changes vs v1 marked):
  1. Motion correction ON (ANTs rigid, mean reference). v1 skipped this
     after one subject was slow; measured 312 s/subject on this VM, so the
     full run takes ~5 h for motion correction alone. The rigid parameters
     are parsed out of the per-volume transform files and saved, so the
     motion record is reproducible without re-running ANTs.
  2. Motion confounds: Friston-24 (6 rigid params + derivatives + squares)
     plus framewise displacement (Power 2012, FD in mm, 50 mm radius).
     Volumes with FD > 0.5 mm are censored (scrubbed) from the time series.
  3. High-variance confounds (nilearn's CompCor-style components, 5
     components from the highest-variance voxels) regressed out alongside
     the motion regressors.
  4. Affine normalisation to MNI152 2 mm (BOLD-mean -> T1 rigid,
     T1 -> MNI affine), same as v1. Affine-only is a known limitation;
     nonlinear SyN was not used (no GPU, too slow for 55 subjects).
  5. Parcel time series at TWO resolutions: Schaefer 100 and Schaefer 200
     (7 networks each), 6 mm smoothing, bandpass 0.01-0.1 Hz, TR 2 s,
     z-score standardisation, detrend.
  6. Pearson FC matrices (100x100 and 200x200) + Fisher-z versions.

Outputs: training/features_v2/<sub>_fc{100,200}.npy,
  <sub>_fc{100,200}_fisherz.npy, motion/<sub>_motion.csv (6 params + FD),
  labels.csv, labels_qc.csv (mean FD, max FD, volumes scrubbed/kept).
Logs: training/logs/preprocessing_v2.log
"""
import argparse, csv, gc, json, logging, sys, time
from pathlib import Path
import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent / "features_v2"
LOG = Path(__file__).resolve().parent / "logs" / "preprocessing_v2.log"
ATLAS_DIR = Path(__file__).resolve().parent / "atlas"

GROUP_TO_LABEL3 = {"Control": 0, "PD-NC": 1, "PD-MCI": 2}
FD_THRESH = 0.5  # mm; volumes above this are censored


def setup_logging():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered file stream: on this machine a plain FileHandler can
    # sit with records stuck in its buffer (flush calls fail here), so
    # records would only reach disk when the buffer fills. Line buffering
    # writes each record out at its newline instead.
    log_stream = open(LOG, "a", buffering=1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(log_stream),
                  logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("preprocess_v2")


def parse_rigid_mat(path):
    """Read one ANTs rigid transform file -> (translation mm x3,
    rotation rad x3).

    The file is read through ants.read_transform (ITK), not parsed as
    text: some transform files carry bytes that are not valid UTF-8,
    which broke a plain-text parse. Parameters are the 12 rigid values
    (9 rotation-matrix entries, then 3 translations, LPS coordinates);
    rotation is converted to XYZ Euler angles in radians.
    """
    import ants
    from scipy.spatial.transform import Rotation
    tx = ants.read_transform(path)
    params = np.array(tx.parameters, dtype=float)
    if params.shape != (12,):
        raise ValueError(f"expected 12 rigid params, got {params.shape} "
                         f"from {path}")
    R = params[:9].reshape(3, 3)
    t = params[9:12]
    angles = Rotation.from_matrix(R).as_euler("xyz")
    return np.concatenate([t, angles])


def motion_params_from_mc(mc, n_vol, log):
    """Extract 6 rigid parameters (tx,ty,tz mm; rx,ry,rz rad) per volume
    from an ants.motion_correction result. Returns (params, fd)."""
    mp = mc.get("motion_parameters")
    paths = []
    n_na = 0
    if isinstance(mp, (list, tuple)):
        for item in mp:
            if isinstance(item, str) and item == "NA":
                n_na += 1
                paths.append(None)
            elif isinstance(item, (list, tuple)):
                mats = [p for p in item if isinstance(p, str)
                        and p.endswith(".mat")]
                paths.append(mats[-1] if mats else None)
            elif isinstance(item, str) and item.endswith(".mat"):
                paths.append(item)
            else:
                paths.append(None)
    if n_na:
        log.warning("%d volumes returned no transform ('NA'); "
                    "their motion is treated as zero", n_na)
    if len(paths) != n_vol:
        log.warning("motion_parameters gave %d entries for %d volumes",
                    len(paths), n_vol)
    rows = []
    for p in paths[:n_vol]:
        rows.append(parse_rigid_mat(p) if p else np.zeros(6))
    params = np.array(rows)
    # Framewise displacement, Power et al. 2012: sum of absolute
    # translational + rotational (x50 mm radius) differences.
    d = np.abs(np.diff(params, axis=0))
    fd = np.zeros(n_vol)
    fd[1:] = d[:, :3].sum(axis=1) + 50.0 * d[:, 3:].sum(axis=1)
    # Cross-check against ANTs' own FD if it came back as numbers.
    ants_fd = mc.get("FD")
    try:
        afd = np.asarray(ants_fd, dtype=float).ravel()
        if afd.shape == (n_vol,):
            log.info("FD check: ours mean=%.3f max=%.3f | ANTs mean=%.3f max=%.3f",
                     fd.mean(), fd.max(), afd.mean(), afd.max())
    except Exception:
        pass
    return params, fd


def friston24(params):
    """Friston-24 expansion: R, R', R^2, R'^2 (24 columns)."""
    R = params
    dR = np.vstack([np.zeros((1, 6)), np.diff(R, axis=0)])
    return np.hstack([R, dR, R ** 2, dR ** 2])


def load_participants(raw_dir):
    rows = []
    with open(raw_dir / "participants.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--skip-motion", action="store_true",
                    help="skip motion correction (v1 behaviour, not recommended)")
    ap.add_argument("--skip-normalize", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    t0 = time.time()

    import nibabel as nib
    from nilearn import image
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure

    try:
        import ants
        have_ants = True
    except ImportError:
        have_ants = False
        log.warning("ANTsPy not available - normalisation will be skipped")

    log.info("=== Preprocessing v2 ds005892: motion correction ON, "
             "Friston-24 + FD scrubbing + high-variance confounds, "
             "Schaefer-100 and Schaefer-200 ===")
    atlases = {
        100: str(ATLAS_DIR /
                 "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"),
        200: str(ATLAS_DIR /
                 "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"),
    }
    for n, p in atlases.items():
        assert Path(p).exists(), f"atlas missing: {p}"
    log.info("Atlases: Schaefer-100 and Schaefer-200 (2 mm MNI)")

    mni_template = None
    if have_ants and not args.skip_normalize:
        log.info("Loading MNI152 template via ANTsPy ...")
        mni_template = ants.image_read(ants.get_ants_data("mni"))
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
    (out_dir / "motion").mkdir(parents=True, exist_ok=True)
    rows, qc_rows, failed = [], [], []
    for p in participants:
        sid = p["participant_id"]
        grp = p["group"]
        label3 = GROUP_TO_LABEL3[grp]
        label_bin = 0 if grp == "Control" else 1
        expected = [out_dir / f"{sid}_fc{n}.npy" for n in (100, 200)]
        if all(f.exists() for f in expected):
            log.info("[%s] already processed, skipping", sid)
            n_vol = int(nib.load(
                str(args.raw / sid / "func" /
                    f"{sid}_task-rest_bold.nii.gz")).shape[3])
            rows.append({"subject": sid, "group": grp, "label3": label3,
                         "label_bin": label_bin, "n_vol": n_vol})
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

            confounds = None
            sample_mask = None
            fd_mean = fd_max = n_scrub = None
            bold_proc = bold_img
            if have_ants and not args.skip_normalize:
                if not args.skip_motion:
                    log.info("[%s] motion correction ...", sid)
                    bold_ants = ants.image_read(str(bold_path))
                    # Transforms go to a stable folder under training/,
                    # not /tmp: one run lost a subject when ANTs could not
                    # read back a transform file from the /tmp tmpfs, and
                    # keeping the files also preserves the motion record.
                    mc_dir = (Path(__file__).resolve().parent / "tmp_mc"
                              / sid)
                    mc_dir.mkdir(parents=True, exist_ok=True)
                    mc = ants.motion_correction(
                        bold_ants, fixed=None, more_accurate=0,
                        outprefix=str(mc_dir / "mc"))
                    bold_proc = mc["motion_corrected"]
                    log.info("[%s] motion correction done (%.1fs)",
                             sid, time.time() - ts)
                    # motion record: parse now, the .mat files live in /tmp
                    params, fd = motion_params_from_mc(mc, n_vol, log)
                    fd_mean, fd_max = float(fd.mean()), float(fd.max())
                    keep = fd <= FD_THRESH
                    n_scrub = int((~keep).sum())
                    sample_mask = np.where(keep)[0]
                    confounds = friston24(params)
                    with open(out_dir / "motion" / f"{sid}_motion.csv",
                              "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["tx_mm", "ty_mm", "tz_mm",
                                    "rx_rad", "ry_rad", "rz_rad", "fd_mm"])
                        for row_p, row_f in zip(params, fd):
                            w.writerow([f"{v:.6f}" for v in row_p] +
                                      [f"{row_f:.6f}"])
                    log.info("[%s] FD mean=%.3f max=%.3f mm, scrubbed %d/%d volumes",
                             sid, fd_mean, fd_max, n_scrub, n_vol)
                    if len(sample_mask) < 100:
                        log.warning("[%s] only %d volumes survive scrubbing",
                                    sid, len(sample_mask))
                    del mc
                else:
                    bold_ants = ants.image_read(str(bold_path))
                    bold_proc = bold_ants
                log.info("[%s] normalisation (bold->T1 rigid, T1->MNI affine) ...", sid)
                if not args.skip_motion:
                    mean_bold = ants.get_average_of_timeseries(bold_proc)
                else:
                    mean_bold = ants.get_average_of_timeseries(bold_ants)
                t1_ants = ants.image_read(str(t1_path))
                reg_b2t = ants.registration(fixed=t1_ants, moving=mean_bold,
                                            type_of_transform="Rigid")
                reg_t2m = ants.registration(fixed=mni_template, moving=t1_ants,
                                            type_of_transform="Affine")
                # Apply the spatial normalisation in blocks of 25 volumes
                # instead of the whole 4D image at once: a single call
                # peaks near 3.7 GB on this machine and gets killed when
                # anything else allocates at the same moment. The transform
                # is spatial, so per-block results are identical to the
                # whole-image call (verified bit-identical on a test image).
                bold_np = bold_proc.numpy()
                n_vol_b = bold_np.shape[3]
                tx_list = (reg_t2m["fwdtransforms"]
                           + reg_b2t["fwdtransforms"])
                blocks = []
                for s0 in range(0, n_vol_b, 25):
                    blk = ants.from_numpy(
                        bold_np[:, :, :, s0:s0 + 25],
                        origin=bold_proc.origin, spacing=bold_proc.spacing,
                        direction=bold_proc.direction)
                    out = ants.apply_transforms(
                        fixed=mni_template, moving=blk,
                        transformlist=tx_list,
                        interpolator="linear", imagetype=3)
                    blocks.append(out.numpy())
                    del blk, out
                bold_arr = np.concatenate(blocks, axis=3)
                del blocks, bold_np, bold_proc
                # Output grid is the fixed (MNI template) grid.
                D = np.array(mni_template.direction, dtype=float)[:3, :3]
                s = np.array(mni_template.spacing, dtype=float)[:3]
                o = np.array(mni_template.origin, dtype=float)[:3]
                A_lps = np.eye(4)
                A_lps[:3, :3] = D @ np.diag(s)
                A_lps[:3, 3] = o
                F = np.diag([-1.0, -1.0, 1.0, 1.0])
                del reg_b2t, reg_t2m
                try:
                    del bold_ants, t1_ants, mean_bold
                except NameError:
                    pass
                gc.collect()
                bold_proc = nib.Nifti1Image(bold_arr, affine=F @ A_lps)
                del bold_arr
                log.info("[%s] normalisation done (%.1fs)", sid, time.time() - ts)
            elif args.skip_normalize:
                log.warning("[%s] SKIP-NORMALIZE: atlas in native space (testing only)",
                            sid)

            for n_parcels, atlas_path in atlases.items():
                masker = NiftiLabelsMasker(
                    labels_img=atlas_path, standardize="zscore_sample",
                    smoothing_fwhm=6, low_pass=0.1, high_pass=0.01, t_r=2.0,
                    detrend=True, dtype="float32", verbose=0,
                    high_variance_confounds=True)
                ts_series = masker.fit_transform(
                    bold_proc, confounds=confounds,
                    sample_mask=sample_mask)
                log.info("[%s] atlas-%d time series %s (%.1fs)",
                         sid, n_parcels, ts_series.shape, time.time() - ts)
                measure = ConnectivityMeasure(kind="correlation")
                fc = measure.fit_transform([ts_series])[0]
                fc = np.nan_to_num(fc, nan=0.0)
                np.fill_diagonal(fc, 1.0)
                fisherz = np.arctanh(np.clip(fc, -0.9999, 0.9999))
                np.save(out_dir / f"{sid}_fc{n_parcels}.npy", fc)
                np.save(out_dir / f"{sid}_fc{n_parcels}_fisherz.npy", fisherz)
                log.info("[%s] atlas-%d OK fc saved, mean|r|=%.3f (total %.1fs)",
                         sid, n_parcels,
                         np.mean(np.abs(fc[np.triu_indices(n_parcels, 1)])),
                         time.time() - ts)
            rows.append({"subject": sid, "group": grp, "label3": label3,
                         "label_bin": label_bin, "n_vol": n_vol})
            qc_rows.append({"subject": sid, "group": grp,
                            "fd_mean_mm": fd_mean, "fd_max_mm": fd_max,
                            "n_scrubbed": n_scrub,
                            "n_kept": (None if sample_mask is None
                                       else int(len(sample_mask)))})
        except Exception as e:
            log.exception("[%s] FAILED: %s", sid, e)
            failed.append(sid)

    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "group", "label3",
                                          "label_bin", "n_vol"])
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "labels_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "group", "fd_mean_mm",
                                          "fd_max_mm", "n_scrubbed", "n_kept"])
        w.writeheader()
        w.writerows(qc_rows)
    log.info("=== DONE v2: %d ok, %d failed (%s) in %.1f min ===",
             len(rows), len(failed), failed, (time.time() - t0) / 60)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
