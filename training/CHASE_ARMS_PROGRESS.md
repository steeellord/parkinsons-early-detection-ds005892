# Chase arms — progress ledger (rounds 4 / 4B / 4C)

Running record of every trained arm, written as its numbers land so
progress is visible without waiting for a whole round to finish.
One entry per arm per phase. Pooled = stratified 5-fold CV on the
pooled subjects; LOSO = leave-one-source-out (each fold holds out one
whole dataset — the honest test for a new scanner). Balanced accuracy
is the headline number: 0.50 is chance for the binary task, 0.33 for
the 3-class task. The 90% target means LOSO balanced accuracy above
0.90, not pooled.

Best honest numbers so far (12 Nov 2026): binary pooled 0.740
(arm209_SGDAM, LR) but LOSO only 0.541 — the pooled score is flattered
by scanner signal. No arm is near 90% on LOSO yet.

## Round 4 screening (done 11 Nov 2026, 12 arms)

Screening pool: n=264 (BrainNet derived 209 + ds005892 55, old v2
pipeline for ds005892) and n=209 derived-only arms; 3-class n=55.

- arm209_SGDAM/pooled_5fold — best LR, balacc 0.740, AUC 0.773, n=209.
  LOSO best LR, balacc 0.541, AUC 0.548. Verdict: best pooled number
  in the project so far, but it collapses to near chance on an unseen
  scanner — most of the pooled gain is site signal, not disease.
- arm209_SGDAM/loso — best LR, balacc 0.541, AUC 0.548 (same arm as
  above, LOSO scheme).
- arm209_SG/pooled_5fold — best SVM, balacc 0.664, AUC 0.711, n=209.
  LOSO best RF, balacc 0.509, AUC 0.485. Verdict: chance on a new
  scanner.
- arm209_SGDAM_combat/pooled_5fold — best LR, balacc 0.653, AUC 0.744.
  LOSO best XGB, balacc 0.525, AUC 0.546. Verdict: ComBat
  harmonisation hurts, again — pooled falls from 0.740 to 0.653 and
  LOSO stays at chance.
- arm264_SG/pooled_5fold — best SVM, balacc 0.601, AUC 0.668, n=264.
  LOSO best LR, balacc 0.498, AUC 0.510. Verdict: adding ds005892
  (old pipeline) pulls pooled down and LOSO is exactly chance.
- arm3cls_direct/pooled_5fold — best RF, balacc 0.427, AUC 0.583,
  n=55, 3-class (chance 0.333). Verdict: barely above chance.
- arm3cls_stage2/pooled_5fold — best XGB, balacc 0.633, AUC 0.539,
  n=33 (PD only, stage 2 of the two-stage 3-class). Verdict: the
  strongest 3-class number, but n=33 and no LOSO — treat as a lead,
  not a result.
- arm3cls_binary/pooled_5fold — best LR, balacc 0.509, AUC 0.508,
  n=55. Verdict: chance.

## Round 4 finisher (running since 11 Nov, restarted 12 Nov 11:42 UTC after VM restart #21)

Final pool: n=395 (101 HC / 294 PD), 7 sources, uniform pipeline
for the raw sets. Cognition pool n=78 (NC=18, MCI=51, PDD=9) —
not 91 as first estimated; 13 PD subjects have no usable cognition
label in the reprocessed data.

- final_SG_all/pooled_5fold — PARTIAL (12 Nov ~12:15 UTC): LR
  balacc 0.641, AUC 0.695; RF and SVM done, XGB on fold 3 of 5,
  ensemble still to run. Verdict so far: unifying the pipeline did
  not recover the 0.740 — the 395-subject pool sits at 0.641 pooled.
  The clean pipeline test is the 4B arm b264_SG_unified (same 264
  subjects, unified ds005892) against screening arm264_SG (0.601).

## Round 4B (queued — starts when the finisher writes logs/r4_finisher_done.txt)

B block: per-parcel time-series stats + brain states, 614 floats.
Status 12 Nov: 338/395 subjects built; the other 57 are ds004392,
whose reprocessing only finished after the B build ran. The chain
rebuilds the B block first (skips the 338 already done).
ds005906 audit verified 12 Nov: download complete (75/75 files,
10 subjects, ses-01 OFF run-01 present, gzip OK) — recorded in
data/DATASETS.md; build_r4b_5906.py already written.
S400 note: only 35 bold_mni files survive (the rest were deleted
in the 12 Nov disk-quota fix), so the S400 build can only cover 35
raw subjects. It runs after the core 4B training, not before it,
so a side arm does not delay the main run.

## Round 4C (queued — starts after 4B training)

C block: lag-1 cross-correlation FC, Ledoit-Wolf partial
correlation, edge co-fluctuation stats. Status 12 Nov: 2/395
subjects (smoke test only). Full build is the first 4C step.

<!-- Entries below are appended by training/watch_4b4c_progress.py
     as arms complete. Do not edit by hand. -->

- final_SG_all/pooled_5fold [full] — 12 Nov 2026 12:45 UTC: best logistic_regression, balacc 0.641, AUC 0.695, n=395. LOSO best balacc: n/a. Verdict: No LOSO number recorded for this arm.

- final_SG_all/loso [full] — 12 Nov 2026 15:35 UTC: best xgboost, balacc 0.520, AUC 0.518, n=395. Pooled balacc for the same arm: 0.641. Verdict: No better than chance on an unseen scanner.

- b264_SG_unified/pooled_5fold [screening] — 12 Nov 2026 15:40 UTC: best logistic_regression, balacc 0.633, AUC 0.705, n=264. LOSO best balacc: 0.495. Verdict: No better than chance on an unseen scanner.

- final_common_all/pooled_5fold [full] — 14 Nov 2026 14:28 UTC: best logistic_regression, balacc 0.611, AUC 0.679, n=395. LOSO best balacc: 0.534. Verdict: No better than chance on an unseen scanner.

- final_common_all/loso [full] — 14 Nov 2026 14:28 UTC: best xgboost, balacc 0.534, AUC 0.518, n=395. Pooled balacc for the same arm: 0.611. Verdict: No better than chance on an unseen scanner.

- final_fusion_derived/pooled_5fold [full] — 14 Nov 2026 14:28 UTC: best svm_rbf, balacc 0.706, AUC 0.705, n=209. LOSO best balacc: 0.564. Verdict: Barely above chance on an unseen scanner.

- final_fusion_derived/loso [full] — 14 Nov 2026 14:28 UTC: best ensemble, balacc 0.564, AUC 0.541, n=209. Pooled balacc for the same arm: 0.706. Verdict: Barely above chance on an unseen scanner.

- final_fusion_raw/pooled_5fold [full] — 14 Nov 2026 14:28 UTC: best logistic_regression, balacc 0.621, AUC 0.705, n=186. LOSO best balacc: 0.611. Verdict: Weak but real signal on an unseen scanner.

- final_fusion_raw/loso [full] — 14 Nov 2026 14:28 UTC: best svm_rbf, balacc 0.611, AUC 0.574, n=186. Pooled balacc for the same arm: 0.621. Verdict: Weak but real signal on an unseen scanner.

- final_fusion_raw_struct/pooled_5fold [full] — 14 Nov 2026 14:28 UTC: best logistic_regression, balacc 0.610, AUC 0.695, n=186. LOSO best balacc: 0.603. Verdict: Weak but real signal on an unseen scanner.

- final_fusion_raw_struct/loso [full] — 14 Nov 2026 14:28 UTC: best ensemble, balacc 0.603, AUC 0.582, n=186. Pooled balacc for the same arm: 0.610. Verdict: Weak but real signal on an unseen scanner.

- final_cog/pooled_5fold [full] — 14 Nov 2026 14:28 UTC: best logistic_regression, balacc 0.570, AUC 0.720, n=78. LOSO best balacc: 0.450. Verdict: Barely above chance on an unseen scanner.

- final_cog/loso [full] — 14 Nov 2026 14:28 UTC: best svm_rbf, balacc 0.450, AUC nan, n=78. Pooled balacc for the same arm: 0.570. Verdict: Barely above chance on an unseen scanner.

- final_common_combat/pooled_5fold [full] — 14 Nov 2026 14:28 UTC: best logistic_regression, balacc 0.619, AUC 0.658, n=395. LOSO best balacc: 0.592. Verdict: Barely above chance on an unseen scanner.

- final_common_combat/loso [full] — 14 Nov 2026 14:28 UTC: best logistic_regression, balacc 0.592, AUC 0.554, n=395. Pooled balacc for the same arm: 0.619. Verdict: Barely above chance on an unseen scanner.

- b264_SG_unified/loso [screening] — 14 Nov 2026 14:28 UTC: best xgboost, balacc 0.517, AUC 0.507, n=264. Pooled balacc for the same arm: 0.633. Verdict: No better than chance on an unseen scanner.

- b209_fusion_B/pooled_5fold [screening] — 14 Nov 2026 14:33 UTC: best logistic_regression, balacc 0.671, AUC 0.760, n=209. LOSO best balacc: 0.584. Verdict: Barely above chance on an unseen scanner.

- b209_fusion_B/loso [screening] — 14 Nov 2026 14:33 UTC: best logistic_regression, balacc 0.584, AUC 0.560, n=209. Pooled balacc for the same arm: 0.671. Verdict: Barely above chance on an unseen scanner.

- b395_common_B/pooled_5fold [screening] — 14 Nov 2026 14:38 UTC: best svm_rbf, balacc 0.617, AUC 0.665, n=395. LOSO best balacc: 0.551. Verdict: Barely above chance on an unseen scanner.

- b395_common_B/loso [screening] — 14 Nov 2026 14:38 UTC: best random_forest, balacc 0.551, AUC nan, n=395. Pooled balacc for the same arm: 0.617. Verdict: Barely above chance on an unseen scanner.

- b395_fusion_B/pooled_5fold [screening] — 14 Nov 2026 14:43 UTC: best svm_rbf, balacc 0.620, AUC 0.718, n=395. LOSO best balacc: 0.507. Verdict: No better than chance on an unseen scanner.

- b395_fusion_B/loso [screening] — 14 Nov 2026 14:43 UTC: best xgboost, balacc 0.507, AUC nan, n=395. Pooled balacc for the same arm: 0.620. Verdict: No better than chance on an unseen scanner.

- b186_raw_B/pooled_5fold [screening] — 14 Nov 2026 14:48 UTC: best svm_rbf, balacc 0.587, AUC 0.723, n=186. LOSO best balacc: 0.617. Verdict: Weak but real signal on an unseen scanner.

- b186_raw_B/loso [screening] — 14 Nov 2026 14:48 UTC: best svm_rbf, balacc 0.617, AUC nan, n=186. Pooled balacc for the same arm: 0.587. Verdict: Weak but real signal on an unseen scanner.

- bcog_B/pooled_5fold [screening] — 14 Nov 2026 14:48 UTC: best logistic_regression, balacc 0.427, AUC 0.632, n=78. LOSO best balacc: 0.332. Verdict: No better than chance on an unseen scanner.

- bcog_B/loso [screening] — 14 Nov 2026 14:48 UTC: best logistic_regression, balacc 0.332, AUC nan, n=78. Pooled balacc for the same arm: 0.427. Verdict: No better than chance on an unseen scanner.

<!-- Relay-monitor entries (14 Nov 2026 ~14:52 UTC) — early-stop enforcement, not watcher output. -->

- EARLY STOP — 4B full run KILLED 14 Nov 2026 14:47 UTC: the chain started the monolithic full run (`train_r4b.py`, no --fast, PID 10848) 16 s after the fast screening finished; the relay guard killed it (rc=143) before it could burn hours on dead arms. Fast-screening LOSO that justified the kill: b264_SG_unified 0.517, b395_common_B 0.551, b395_fusion_B 0.507, bcog_B 0.332 (3-class, chance 0.333) — all at chance. Only b209_fusion_B (0.584) and b186_raw_B (0.617) cleared 0.55. The chain stopped (STEP FAILED) and was relaunched from the patched chain script (run_4b4c_chain_es.sh) with logs/skip_4b_full.txt so the monolithic full run is skipped; full runs for the two surviving arms only were launched manually (`train_r4b.py --arms b186_raw_B b209_fusion_B`, no --fast).

- b186_raw_B/loso [screening] — HONESTY NOTE: best svm_rbf balacc 0.617 is carried by the all-PD ds004392 LOSO fold (0.897 = PD recall, no AUC); the other three folds are 0.600 / 0.500 / 0.470. Excluding the all-PD fold the mean is 0.523 — chance. Same caveat applies to the finisher's final_fusion_raw LOSO 0.611. Do not quote 0.617 as disease signal without the fold breakdown.
- b209_fusion_B/loso [screening] — HONESTY NOTE: best logistic_regression balacc 0.584, folds 0.632 / 0.519 / 0.600 over neurocon / ppmi_v2 / taowu — no all-PD fold in this arm, so this is the cleanest above-chance LOSO in 4B, though still weak.
- b395_common_B/loso [screening] — HONESTY NOTE: best random_forest 0.551 includes the all-PD ds004392 fold at 0.845; excluding it the mean is 0.502 — chance.

- c_rank_common/pooled_5fold [complete] — 14 Nov 2026 15:03 UTC: best svm_rbf, balacc 0.593, AUC 0.651, n=395. LOSO best balacc: 0.508. Verdict: No better than chance on an unseen scanner.

- c_rank_common/loso [complete] — 14 Nov 2026 15:08 UTC: best random_forest, balacc 0.508, AUC nan, n=395. Pooled balacc for the same arm: 0.593. Verdict: No better than chance on an unseen scanner.

- c_stable_common/pooled_5fold [complete] — 14 Nov 2026 15:08 UTC: best svm_rbf, balacc 0.622, AUC 0.667, n=395. LOSO best balacc: 0.507. Verdict: No better than chance on an unseen scanner.

- c_stable_common/loso [complete] — 14 Nov 2026 15:13 UTC: best logistic_regression, balacc 0.507, AUC nan, n=395. Pooled balacc for the same arm: 0.622. Verdict: No better than chance on an unseen scanner.

- b209_fusion_B/pooled_5fold [full] — 14 Nov 2026 15:23 UTC: best stacking, balacc 0.718, AUC 0.750, n=209. LOSO best balacc: 0.584. Verdict: Barely above chance on an unseen scanner.

- b209_fusion_B/loso [full] — 14 Nov 2026 15:33 UTC: best balanced_bagging, balacc 0.591, AUC 0.548, n=209. Pooled balacc for the same arm: 0.718. Verdict: Barely above chance on an unseen scanner.

- b186_raw_B/pooled_5fold [screening] (updated numbers) — 14 Nov 2026 16:03 UTC: best gaussian_process, balacc 0.610, AUC 0.489, n=186. LOSO best balacc: 0.617. Verdict: Weak but real signal on an unseen scanner.

- c_rankgauss_common/pooled_5fold [complete] — 14 Nov 2026 16:03 UTC: best logistic_regression, balacc 0.538, AUC 0.568, n=395. LOSO best balacc: 0.592. Verdict: Barely above chance on an unseen scanner.

- b209_fusion_B [full] — 14 Nov 2026 ~15:48 UTC (manual survivor run): pooled best stacking balacc 0.718, AUC 0.750, n=209. LOSO best balanced_bagging balacc 0.591, AUC 0.548, folds 0.613 / 0.510 / 0.650 (neurocon / ppmi_v2 / taowu — no all-PD fold, so this is the cleanest above-chance LOSO in 4B). LR 0.584, stacking 0.564, ensemble 0.520. Verdict: weak but real signal; the full run added almost nothing over the fast LR 0.584.
- b186_raw_B [full, in progress] — classical full-grid re-run confirms the fast result: LOSO LR 0.598 / SVM 0.617, but both are carried by the all-PD ds004392 fold (0.810 / 0.897); excluding it the means are 0.528 / 0.523 — chance. Pooled best so far gaussian_process 0.610.
- c_rankgauss_common/loso [fast, partial] — LR balacc 0.592 looks above chance but folds are 0.500 / 0.679 / 1.000 / 0.462 / 0.500 / 0.500 / 0.500; the 1.000 is the all-PD ds004392 fold (PD recall). Excluding it the mean is 0.524 — chance. Same all-PD-fold inflation as b186_raw_B and the finisher's final_fusion_raw 0.611. Pooled best LR 0.538 — the rank-gauss transform hurts pooled too.
- 4C GUARD — 14 Nov 2026 15:48 UTC: early-stop guard for train_r4c.py installed (/tmp/es_watchdog_4c.sh, PID 17508, log /tmp/earlystop_4c.log). It kills any non-fast train_r4c.py the moment it starts, so the chain cannot run a monolithic 4C full run of at-chance arms unattended. Stop it before any deliberate manual full run with --arms. The patched chain also skips the 4C full run if logs/skip_4c_full.txt exists.

- EARLY STOP (2) — b186_raw_B manual full run KILLED 14 Nov 2026 ~16:04 UTC: the classical full-grid re-run already proved the arm dead — LOSO LR 0.598 / SVM 0.617 are carried entirely by the all-PD ds004392 fold (0.810 / 0.897); excluding it the means are 0.528 / 0.523, chance. The remaining ensemble/extra models on LOSO were killed before they could burn more CPU that the 4C fast screening needs. b209_fusion_B full run had already completed before the kill. No further 4B full work is warranted.

- c_rankgauss_common/loso [complete] — 14 Nov 2026 16:15 UTC: best logistic_regression, balacc 0.592, AUC nan, n=395. Pooled balacc for the same arm: 0.538. Verdict: Barely above chance on an unseen scanner.

- c_ageresid_common/pooled_5fold [complete] — 14 Nov 2026 16:20 UTC: best svm_rbf, balacc 0.611, AUC 0.653, n=395. LOSO best balacc: 0.514. Verdict: No better than chance on an unseen scanner.

- c_ageresid_common/loso [complete] — 14 Nov 2026 16:20 UTC: best xgboost, balacc 0.514, AUC nan, n=395. Pooled balacc for the same arm: 0.611. Verdict: No better than chance on an unseen scanner.

- c_zscore_common/pooled_5fold [complete] — 14 Nov 2026 16:20 UTC: best svm_rbf, balacc 0.553, AUC 0.608, n=395. LOSO best balacc: n/a. Verdict: No LOSO number recorded for this arm.

- CHAIN RELAUNCH — 14 Nov 2026 ~16:12 UTC: the 4C fast run's original chain (run_4b4c_chain_es.sh PID 2719) and training (PID 2760) were replaced — new chain run_4b4c_chain.sh PID 3016, training PID 3054, resumed from per-fold checkpoints (log shows "resumed from partial checkpoint"). Watcher PID 2731 and both guards (2100, 2140) survived. Both chain scripts are identical and honour logs/skip_4c_full.txt.
- c_rankgauss_common/loso [fast, complete] — 14 Nov 2026 ~16:12 UTC: best LR balacc 0.592, folds 0.500 / 0.679 / 1.000 / 0.462 / 0.500 / 0.500 / 0.500 (sources ds000245 / ds001354 / ds004392 / ds005892 / neurocon / ppmi_v2 / taowu). HONESTY NOTE: the 1.000 is the all-PD ds004392 fold; excluding it the LR mean is 0.524 — chance. SVM 0.571 (ex-fold 0.500), RF 0.566 (ex-fold 0.493), XGB 0.565 (ex-fold 0.493). Pooled best LR 0.538. Verdict: chance; the rank-gauss transform helps nothing.
- c_ageresid_common/pooled_5fold [complete] — 14 Nov 2026 ~16:16 UTC: best svm_rbf, balacc 0.611, AUC 0.653, n=395. LOSO best balacc: 0.514. Verdict: pooled-only gain, no unseen-scanner signal.
- c_ageresid_common/loso [complete] — 14 Nov 2026 ~16:16 UTC: best xgboost, balacc 0.514, AUC nan, n=395. Folds 0.450 / 0.536 / 0.552 / 0.508 / 0.470 / 0.559 / 0.525 — the all-PD fold is 0.552 here, NOT inflated, so 0.514 is plain chance. Pooled balacc for the same arm: 0.611. Verdict: No better than chance on an unseen scanner.
- c_zscore_common/pooled_5fold [complete] — 14 Nov 2026 ~16:20 UTC: best svm_rbf, balacc 0.553, AUC 0.608, n=395. LOSO best balacc (partial): 0.588. Verdict: weak pooled, LOSO pending.
- c_zscore_common/loso [fast, partial] — LR 0.577, folds 0.517 / 0.500 / 0.948 / 0.515 / 0.528 / 0.533 / 0.500; SVM 0.588, folds 0.500 / 0.500 / 0.966 / 0.485 / 0.567 / 0.501 / 0.600. HONESTY NOTE: both are carried by the all-PD ds004392 fold (0.948 / 0.966); excluding it the means are 0.516 / 0.526 — chance. RF/XGB still pending.
- SKIP MARKER PLACED — 14 Nov 2026 16:22 UTC: training/logs/skip_4c_full.txt created preemptively by the follow-up monitor while c_loso_select / c_stack_loso / c_lagprec were still pending; every landed 4C arm is at chance ex-all-PD-fold. Remove ONLY if a fast arm lands genuinely above chance ex-all-PD-fold (bar: 4B honest LOSO 0.591), then run only that arm with --arms after stopping the 4C guard.

- c_zscore_common/loso [complete] — 14 Nov 2026 16:25 UTC: best svm_rbf, balacc 0.588, AUC nan, n=395. Pooled balacc for the same arm: 0.553. Verdict: Barely above chance on an unseen scanner.

- c_stack_loso/pooled_5fold [complete] — 14 Nov 2026 16:30 UTC: best block_stack, balacc 0.477, AUC 0.484, n=395. LOSO best balacc: 0.476. Verdict: No better than chance on an unseen scanner.

- c_stack_loso/loso [complete] — 14 Nov 2026 16:30 UTC: best block_stack, balacc 0.476, AUC 0.484, n=395. Pooled balacc for the same arm: 0.477. Verdict: No better than chance on an unseen scanner.

- c_loso_select/loso [complete] — 14 Nov 2026 16:30 UTC: best loso_selected, balacc 0.449, AUC 0.457, n=395. Pooled balacc for the same arm: n/a. Verdict: No better than chance on an unseen scanner.

- c_zscore_common/loso [fast, complete] — 14 Nov 2026 ~16:24 UTC: best svm_rbf, balacc 0.588, AUC nan, n=395. HONESTY NOTE: carried by the all-PD ds004392 fold (0.966); excluding it the SVM mean is 0.526 — chance. LR 0.577 (ex-fold 0.516), RF 0.574 (ex-fold 0.503), XGB 0.569 (ex-fold 0.506). Pooled best SVM 0.553. Verdict: chance on an unseen scanner.
- c_loso_select/loso [fast, complete] — 14 Nov 2026 ~16:26 UTC: loso_selected balacc 0.449, AUC 0.457, n=395 — BELOW chance. Folds 0.483 / 0.536 / 0.155 / 0.500 / 0.476 / 0.495 / 0.500; the 0.155 is the all-PD ds004392 fold (model predicts HC there). Verdict: LOSO-based model selection does not help; dead.
- c_stack_loso/pooled_5fold [fast, complete] — 14 Nov 2026 ~16:27 UTC: block_stack balacc 0.477, AUC 0.484, n=395. Verdict: below chance.
- c_stack_loso/loso [fast, complete] — 14 Nov 2026 ~16:27 UTC: block_stack balacc 0.476, AUC 0.484, n=395. Folds 0.550 / 0.500 / 0.379 / 0.515 / 0.429 / 0.431 / 0.525. Verdict: below chance; block-wise stacking does not help.
- c_lagprec/pooled_5fold [fast, complete] — 14 Nov 2026 ~16:30 UTC: best logistic_regression, balacc 0.607, AUC 0.703, n=395 (lag-1 FC + partial correlation + RSS co-fluctuation features). LOSO best (partial): 0.560. Verdict: pooled-only gain so far.
- c_lagprec/loso [fast, partial] — LR 0.530, folds 0.583 / 0.500 / 0.586 / 0.462 / 0.576 / 0.406 / 0.600 (all-PD fold 0.586, not inflated; ex-fold mean 0.521); SVM 0.560, folds 0.583 / 0.571 / 0.724 / 0.500 / 0.628 / 0.390 / 0.525 (ex-fold mean 0.533). RF/XGB still pending.

- c_lagprec/pooled_5fold [complete] — 14 Nov 2026 16:35 UTC: best logistic_regression, balacc 0.607, AUC 0.703, n=395. LOSO best balacc: 0.569. Verdict: Barely above chance on an unseen scanner.

- c_lagprec/loso [complete] — 14 Nov 2026 16:35 UTC: best random_forest, balacc 0.569, AUC nan, n=395. Pooled balacc for the same arm: 0.607. Verdict: Barely above chance on an unseen scanner.

- c_lagprec_S/pooled_5fold [complete] — 14 Nov 2026 16:35 UTC: best svm_rbf, balacc 0.592, AUC 0.701, n=395. LOSO best balacc: 0.536. Verdict: No better than chance on an unseen scanner.

- c_rank_common/loso [fast, complete] — HONESTY NOTE (added in final audit 14 Nov 2026 ~16:40 UTC): best random_forest 0.508, folds 0.483 / 0.500 / 0.707 / 0.500 / 0.462 / 0.401 / 0.500; excluding the all-PD ds004392 fold (0.707) the mean is 0.474 — chance. LR 0.485 (ex-fold 0.500), XGB 0.505 (ex-fold 0.469). Pooled best SVM 0.593, AUC 0.651. Verdict: chance.
- c_stable_common/loso [fast, complete] — HONESTY NOTE (final audit): best logistic_regression 0.507, folds 0.583 / 0.464 / 0.431 / 0.485 / 0.527 / 0.535 / 0.525; the all-PD fold is 0.431 here (not inflated), ex-fold mean 0.520 — chance. Pooled best SVM 0.622, AUC 0.667. Verdict: chance.
- c_lagprec/loso [fast, complete] — 14 Nov 2026 ~16:33 UTC: best random_forest, balacc 0.569, AUC nan, n=395. HONESTY NOTE: carried by the all-PD ds004392 fold (1.000); excluding it the RF mean is 0.497 — chance. XGB 0.564 (ex-fold 0.503), SVM 0.560 (ex-fold 0.533), LR 0.530 (ex-fold 0.521). Pooled best LR 0.607, AUC 0.703. Verdict: the new C-block features (lag-1 FC, partial correlation, RSS) give a pooled-only gain, no unseen-scanner signal.
- c_lagprec_S/pooled_5fold [fast, complete] — 14 Nov 2026 ~16:35 UTC: best svm_rbf, balacc 0.592, AUC 0.701, n=395. LOSO best balacc: 0.542. Verdict: pooled-only gain.
- c_lagprec_S/loso [fast, complete] — 14 Nov 2026 ~16:36 UTC (watcher had not recorded it by 16:40 UTC; metrics_round4c.json is the source): best xgboost, balacc 0.542, AUC nan, n=395. Folds 0.517 / 0.500 / 0.707 / 0.485 / 0.523 / 0.511 / 0.550; excluding the all-PD ds004392 fold (0.707) the mean is 0.514 — chance. SVM 0.536 (ex-fold 0.522), RF 0.517 (ex-fold 0.494), LR 0.491 (ex-fold 0.490). Pooled best SVM 0.592. Verdict: chance on an unseen scanner.
- 4C FAST SCREENING COMPLETE — 14 Nov 2026 16:36:53 UTC ("=== Round 4C training done ==="): all 17 arm/scheme entries landed in training/metrics_round4c.json (59 models). NO arm beats the 4B bar of honest LOSO 0.591 excluding the all-PD fold. Best honest 4C LOSO ex-all-PD-fold: c_lagprec SVM 0.533, c_rankgauss LR 0.524, c_lagprec_S SVM 0.522, c_stable LR 0.520 — all chance. Two arms were BELOW chance: c_loso_select 0.449, c_stack_loso 0.476.
- 4C FULL RUN SKIPPED — 14 Nov 2026 16:36:55 UTC: chain (run_4b4c_chain.sh PID 3016) honoured logs/skip_4c_full.txt and skipped the monolithic 4C full run. No non-fast train_r4c.py was started; the 4C guard (PID 2140) never had to kill anything. No arm warrants a full run.
- NOTE — mid-run failure: the fast screening failed once with rc=1 at 16:16:21 UTC and the chain relaunched it; the relaunched run resumed from checkpoints and completed. Original chain/training (PIDs 2719/2760) were replaced at ~16:12 UTC by PIDs 3016/3054, also resuming from checkpoints.
- FOLLOW-UP (not 4C): at 16:36:55 UTC the chain moved to "FOLLOW-UP: S400 build" — build_r4b_s400.py (PID 6742) is running. The monitor's task had Schaefer-400 marked deferred (only 35 bold_mni files survive); this step was in the chain script itself and was left running — parent to confirm whether it should continue.

- c_lagprec_S/loso [complete] — 14 Nov 2026 16:40 UTC: best xgboost, balacc 0.542, AUC nan, n=395. Pooled balacc for the same arm: 0.592. Verdict: No better than chance on an unseen scanner.
