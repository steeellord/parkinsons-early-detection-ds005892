# Additional Parkinson's datasets — download and audit notes

Audited 2026-11-09 on the project VM. Every dataset below was downloaded in full
from its public source and checked file-by-file against the source listing
(byte sizes, subject tables, NIfTI headers). Nothing here is sampled or estimated.
The original ds005892 audit is in `AUDIT.md` next to this file.

## Pooled totals (raw resting-state fMRI, verified on disk)

| Dataset | Subjects | PD | HC | BOLD | Site / scanner |
|---|---:|---:|---:|---|---|
| ds005892 (already on disk) | 55 | 33 (18 PD-NC + 15 PD-MCI) | 22 | 200 vols, TR 2.0 s | NYU, Siemens TrioTim 3 T |
| ds000245 (OpenNeuro) | 45 | 30 (15 severe hyposmia + 15 not) | 15 | 198 vols, TR 2.5 s | Japan, Siemens Verio 3 T |
| ds001354 (OpenNeuro) | 28 | 14 de novo PD | 14 | 300 vols, TR 2.13 s | Italy, de novo (untreated) |
| INDI NEUROCON | 43 | 27 | 16 | 137 vols, TR 3.48 s, 1–2 runs | Romania, Siemens Avanto 1.5 T |
| INDI Tao Wu | 40 | 20 | 20 | 239 vols, TR 2.0 s | Beijing, Siemens Trio 3 T |
| **Raw fMRI total** | **211** | **124** | **87** | — | 5 sites, 4 scanners, TR 2.0–3.48 s |

Plus ready-made connectivity matrices (Figshare, Auckland BrainNet collection):
PPMI 195 subjects, NEUROCON 41, Tao Wu 40 — see below.

Warning for any pooled training run: five sites, different TRs (2.0–3.48 s),
different voxel sizes and field strengths (1.5 T vs 3 T), different run lengths.
A naive pool will learn site, not disease. Use leave-one-dataset-out validation
and check site as a confound before trusting any pooled number.

## ds000245 — `data/ds000245/` — COMPLETE

- Source: OpenNeuro ds000245, Yoneyama et al., Japan. CC0.
  https://openneuro.org/datasets/ds000245
- Download: public S3 bucket, 100/100 files, 2,167,414,614 bytes — exact match
  with the S3 listing. No registration needed.
- Subjects: 45 (15 sub-ODP, 15 sub-ODN, 15 sub-CTL). Group is encoded in the
  subject ID, not in a column: ODP = Parkinson's with severe hyposmia
  (OSITJ smell-test scores 0–3), ODN = Parkinson's without severe hyposmia
  (OSITJ 6–11), CTL = healthy controls (OSITJ 9–12). Binary split: 30 PD / 15 HC.
- Files: 45 T1w + 45 task-rest BOLD. BOLD 64×64×39, 198 vols, TR 2.5 s,
  TE 30 ms, flip 80°, Siemens Verio 3 T, eyes closed. T1w 192×256×256.
- Clinical columns in participants.tsv: Age, Gender, ACER (Addenbrooke's
  cognitive score), MMSE, OSITJ smell test. No UPDRS column.
- Download note: the first attempt stalled with no bytes written for several
  minutes (all 6 workers stuck on slow S3 connections). Killed and restarted
  with 12 workers and a 90 s timeout; finished clean, 0 failures.

## ds001354 — `data/ds001354/` — COMPLETE

- Source: OpenNeuro ds001354, Tessa et al., "PD De Novo: Resting State fMRI
  and Physiological Signals". CC-BY 4.0. DOI 10.18112/openneuro.ds001354.v1.0.0.
  https://openneuro.org/datasets/ds001354
- Download: public S3 bucket, 149/149 files, 1.49 GB, 0 mismatches against
  the S3 listing. No registration needed.
- Subjects: 28 — 14 CONTROL, 14 PD-denovo (newly diagnosed). Age range is
  wider than ds005892 (a 36-year-old patient, a 38-year-old control); check
  age as a confound.
- Files: 28 T1w (256×256×176) + 28 task-rest BOLD (64×64×32, 300 vols,
  TR 2.13 s, ~10.7 min). Plus cardiac/respiratory physio recordings
  (sub-pddn14 physio files were the last two to arrive).
- Bonus: de novo = untreated at scan time, so medication is not a confound
  here — the cleanest early-detection subset of the five.

## INDI Parkinson's — `data/indi-parkinsons/` — COMPLETE

- Source: 1000 Functional Connectomes Project / INDI, retrospective release.
  http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html
  Paper: Badea et al. 2017, PLoS ONE 12(11): e0188196.
- Download: public S3 bucket fcp-indi, no registration. Two tarballs, sizes
  verified against the S3 listing:
  neurocon.tar.gz 1,617,681,469 bytes; taowu.tar.gz 1,436,300,817 bytes.
  Single-stream download ran at ~0.3 MB/s, so both were fetched with
  8-way ranged (HTTP Range) downloads; final sizes match exactly.

### NEUROCON — `data/indi-parkinsons/neurocon/` — 43 subjects

- 27 PD / 16 NORMAL (status column in neurocon_patients.tsv). UPDRS motor
  ON/OFF, treatment, Hoehn & Yahr and age/sex recorded for all 27 patients.
- Files: 43 T1w (175×217×203) + 85 BOLD (64×64×27, 137 vols, TR 3.48 s,
  TE 50 ms, Siemens Avanto 1.5 T, eyes closed, voxel 3.8×3.8×5 mm).
  Some subjects have 2 resting runs (run-1/run-2); most have 1.
- All 128 NIfTI files are genuine gzip and load correctly.

### Tao Wu — `data/indi-parkinsons/taowu/` — 40 subjects

- 20 PD / 20 controls (taowu_patients.tsv). MMSE, disease duration (years)
  and Hoehn & Yahr recorded for patients. n_timepoints = 239 for all.
- Files: 40 T1w (176×188×200) + 40 task-rest BOLD (64×64×28, 239 vols,
  TR 2.0 s, Siemens Magnetom Trio 3 T, eyes closed, voxel 4×4×5 mm).
- Upstream quirk, verified: the 40 BOLD files carry a `.nii.gz` extension
  but are UNCOMPRESSED NIfTI-1 data (header magic `\\\x01`, confirmed with
  `file` and by loading after a rename to `.nii`). The 40 T1w files are
  genuine gzip. Any pipeline step that assumes `.nii.gz` means gzipped
  must rename the Tao Wu BOLD files to `.nii` first (or handle both).
  Content verified: shape 64×64×28×239, all values finite, range 0–1760.
- Extraction note: the first `tar xzf` of both archives produced 0-byte
  files for 149 of 208 NIfTI (NEUROCON subjects from sub-patient032034 on,
  all of Tao Wu). The archives themselves are intact (member bytes read
  correctly, gzip CRC clean). Both directories were wiped and re-extracted
  with plain `tar xzf --no-same-owner`; the second extraction is verified
  complete — 208/208 NIfTI non-zero, every file size checked against the
  tar index. The tarballs are kept as the pristine download record.

## BrainNet Figshare (Auckland) — `data/brainnet-figshare/` — COMPLETE

- Source: Xu et al., "Data-Driven Network Neuroscience" (arXiv:2211.12421).
  Collection DOI 10.17608/k6.auckland.21397377, licence CC BY 4.0.
  Precomputed functional-connectivity matrices, not raw images.
- Downloaded (direct file links, no registration):
  ppmi_v2.zip 200,195,280 bytes; neurocon.zip 31,491,813 bytes;
  taowu.zip 45,271,308 bytes; Metadata_v5.zip 21,364 bytes.
  All CRC-clean (zip testzip).
- ppmi_v2.zip: 195 subjects × 10 .mat files (5 parcellations — AAL116,
  Harvard-Oxford 48, Schaefer-100, Ward-100, kmeans-100 — each with a
  correlation matrix and a feature timeseries). Groups: 113 PD patients,
  53 prodromal, 15 controls, 14 SWEDD. Note the paper's table says PPMI
  N=209; the v2 zip on Figshare holds 195. Only 15 controls — a binary
  PD-vs-control model from this file alone is 113 vs 15, heavily imbalanced.
  The 53 prodromal subjects are the interesting early-detection group.
- neurocon.zip: 41 subjects (26 PD / 15 control), 10 .mat each. Two fewer
  than the raw 43 — two subjects were dropped in their QC.
- taowu.zip: 40 subjects (20 PD / 20 control), 10 .mat each.
- Metadata_v5.zip: PPMI_metadata.csv (418 image-level rows: PD 226,
  prodromal 134, control 30, SWEDD 28 — multiple visits per subject),
  Neurocon_metadata.csv (41 rows), TaoWu_metadata.csv (40 rows).
  These CSVs give age/sex/group labels for the matrix files.
- NOT downloaded: the preprocessed-image zips in the same collection
  (Ppmi_Preprocessed_v2 ~107 GB total, Neurocon_Preprocessed ~40 GB,
  Taowu_Preprocessed ~24 GB, ABIDE ~470 GB). Far too large and unnecessary —
  the raw images are already on disk above.
- Usefulness: the Schaefer-100 correlation matrices are the same feature
  type the current pipeline builds, so these 276 subjects (195 + 41 + 40)
  can be tried as a fast second experiment without re-preprocessing —
  but they come from a different pipeline (fMRIPrep settings, parcellation
  build), so do not mix them into the same training table as the
  locally computed matrices without checking the distributions first.

## EEG-only Parkinson's sets on OpenNeuro (recorded, NOT downloaded)

These are EEG, not fMRI — out of scope for the current pipeline, listed so
they are not searched for twice:

- ds002778 (UC San Diego resting-state EEG, Rockhill et al.): 31 subjects,
  0.57 GB, 0 NIfTI.
- ds003490 (UNM EEG, Cavanagh; oddball + rest, PD on/off medication):
  50 subjects (25 PD / 25 control), 6.28 GB, 0 NIfTI.
- ds008768 / ds008875 (resting-state EEG in PD, uploaded Sep 2026):
  listed on OpenNeuro but the public S3 listing returned 400 for both —
  not yet in the public snapshot. EEG in any case.

## PPMI — the big one — NOT downloaded, needs Abhishek personally

PPMI (Parkinson's Progression Markers Initiative, Michael J. Fox Foundation)
is the largest PD cohort. It is free, but it is NOT an open download:
access is granted per person under a signed Data Use Agreement.

What it has: longitudinal study, 1,612 participants at baseline (PD,
genetic PD — LRRK2/GBA/SNCA, prodromal, healthy controls). Imaging:
rs-fMRI (TR 2.4–2.5 s, ~206–240 vols, 3 T, multi-site), T1, DTI, DAT
SPECT, plus MDS-UPDRS, MoCA, smell test, genetics, biospecimens,
alpha-synuclein seed amplification assay. The rs-fMRI subset with usable
scans is much smaller than 1,612 — published subsets run 90 to 331
subjects depending on QC (e.g. 181 = 113 PD + 68 non-PD; 331 with complete
rs-fMRI; 90 in one 2026 analysis).

Exact steps for Abhishek (all in his own name — this cannot be done for him):

1. Create a LONI IDA account at https://ida.loni.usc.edu/
   (register with his own email, verify it).
2. Go to https://www.ppmi-info.org/access-data-specimens/download-data
   ("Access data") and submit the Data Use Application: personal details,
   institution/affiliation, and the research purpose and planned analyses
   (early detection of Parkinson's from resting-state fMRI connectivity).
3. Read and electronically sign the PPMI Data Use Agreement v5.0
   (April 2026) at
   https://ida.loni.usc.edu/collaboration/access/appLicense.jsp
   Key terms: de-identified data only; no attempt to re-identify or
   contact participants; no redistribution of participant-level data;
   everyone on the team who touches the data must register separately;
   annual updates on the analyses; derived data returned to PPMI.
   The v5.0 agreement also forbids putting participant-level data into
   public-facing AI tools — local training code on this VM is fine,
   uploading scans or rows to a chatbot is not.
4. Wait for the study committee's review — the MJFF community guide says
   they respond within about a week. No fee was found anywhere.
5. Once approved, log in to LONI IDA, Download > Study Data, build an
   image collection (rs-fMRI + T1 + DTI) and download with the IDA
   Advanced Downloader.
6. Only if genetic data is ever needed: separate Genetic Data Request Form
   + supplemental agreement to ppmi@loni.usc.edu. Tier 3 restricted data:
   email resources@michaeljfox.org.

Open points, not assumed: the application asks for an institution /
affiliation. Abhishek is not currently attached to a lab, so he should
expect to name VIT and/or a faculty supervisor, or apply as an independent
researcher. Approval is not guaranteed. He also needs to decide whether
to apply for imaging only or the full clinical package.

Reality check on accuracy: published graph-model baselines on PPMI rs-fMRI
are around 55–57% accuracy (e.g. GCN 55.2, GAT 56.9 on 181 subjects;
GCN 54.0, GAT 55.0 on the 209-subject BrainNet matrices). One paper claims
AUC 0.93 on a 748-subject multimodal (rs-fMRI + DTI) set — treat that with
caution, it is an outlier against the honest benchmarks. More data helps,
but 90% balanced accuracy on honest held-out folds is not an established
result for rs-fMRI PD-vs-HC anywhere. The defensible target is to beat the
current 0.745 AUC / 0.683 balanced accuracy and report exactly what was
measured, per fold, per dataset.

## Other routes checked and closed

- AMP PD / AMP PDRD (amp-pd.org): harmonised genetics + clinical cohorts
  (HBS, PDBP, PPMI, BioFIND, STEADY-PD3, SURE-PD) on Terra / Verily
  Workbench. DUA v14 (June 2026) needs a Requester AND an Institutional
  Signing Official. Genetics-focused; no rs-fMRI download route. Needs
  Abhishek + an institution — same blocker as PPMI, less imaging payoff.
- PDBP via NDA (pdbp.ninds.nih.gov): DUA v3.0 (Aug 2023), PI + Institutional
  Signing Official, Data Access Request, 1-year access. Imaging exists but
  scale not verified from here; application route, no open download.
- UK Biobank: ~500,000 volunteers, imaging in ~100,000, rs-fMRI included,
  but it is a general-population cohort, not a PD cohort (PD cases accrue
  through health records, ~1,200 by 2022, ~2,800 projected by 2027).
  Application + Material Transfer Agreement + Tier 3 fee £9,000 for 3 years
  (student reduced rate £500 reported). New applications were PAUSED as of
  August 2026 while the Research Analysis Platform is closed. Not viable now.
- ADNI: Alzheimer's disease only (MCI / AD). No Parkinson's subjects —
  wrong disease for this classifier.
- OASIS-3: Alzheimer's / ageing. No Parkinson's subjects — wrong disease.
- C-BIG / LORIS (Montreal Neuro): PD imaging exists but behind tiered
  registered access; the "99 PD + 43 HC" LORIS cohort cited in a 2026
  preprint is not a named openly downloadable dataset.
- BioFIND / Fox Insight (MJFF data-sets page): clinical, biospecimen and
  survey data only, no MRI.
- SKIP (OpenNeuro, 7 T): v1.1.1 is healthy controls only (28 subjects);
  PD data not released.
- MJFF data-sets page for reference: https://www.michaeljfox.org/data-sets

## What is on disk now (all under `data/`)

- `raw/` — ds005892, 55 subjects, 2.37 GB (see AUDIT.md)
- `ds000245/` — 45 subjects, 2.17 GB
- `ds001354/` — 28 subjects, 1.49 GB
- `indi-parkinsons/neurocon.tar.gz` + `neurocon/` — 43 subjects, 1.62 GB
  tarball + 1.62 GB extracted
- `indi-parkinsons/taowu.tar.gz` + `taowu/` — 40 subjects, 1.44 GB
  tarball + 2.48 GB extracted (BOLD files are uncompressed despite .nii.gz)
- `brainnet-figshare/` — ppmi_v2.zip, neurocon.zip, taowu.zip,
  Metadata_v5.zip, 0.28 GB total (matrices, not images)
