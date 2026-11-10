# Hey Rishabh — read this first

This file is for you, Rishabh. Abhishek built this whole project for you, and this is the plain-English version of everything: what it is, what all the words mean, how it works, where it stands right now, and what the results will mean when they land. No background needed. If you only read one file in this repo, read this one.

---

## 1. What this is, in one paragraph

Parkinson's disease changes the brain years before the visible symptoms (like tremor) show up. This project asks one question: **can a resting brain scan show Parkinson's before the tremor does?** We take brain scans of 55 people — some healthy, some with Parkinson's — and teach computer models to tell them apart just from how different parts of their brain talk to each other while they lie still and do nothing. There is also a website where you can explore it yourself.

- **Website:** https://parkinsons-early-detection-ds005892.vercel.app/
- **This repo (all the code, logs and results):** https://github.com/steeellord/parkinsons-early-detection-ds005892
- **The brain scans we used:** http://openneuro.org/datasets/ds005892/versions/1.0.0

---

## 2. Why Parkinson's, and why "early" matters

Parkinson's disease is a brain condition where the cells that make dopamine — a chemical the brain uses to control movement — slowly stop working. The classic signs are tremor (shaking), stiffness, and slow movement. But by the time those signs are obvious, a lot of brain change has already happened. If a scan could spot the pattern earlier, doctors could act earlier. That is the whole motivation here. This project does not diagnose anyone — it is a research experiment to see how well a computer can spot the pattern.

---

## 3. The data — where the brains come from

The scans come from a public research dataset called **ds005892** on OpenNeuro (an open library of brain scans shared by scientists), published by Kemp and colleagues in 2025. Everything is anonymised — no names, just scan files and a group label.

**55 people with complete data:**

| Group | People | What it means |
|---|---|---|
| **HC — Healthy Controls** | 22 | People without Parkinson's. The "normal" reference group. |
| **PD-NC — Parkinson's, Normal Cognition** | 18 | People with Parkinson's whose thinking and memory are still normal. |
| **PD-MCI — Parkinson's, Mild Cognitive Impairment** | 15 | People with Parkinson's who also have mild problems with thinking or memory. |

So the computer has to sort every brain into one of **three** groups, not just "sick vs healthy". That is harder, and more useful — telling PD-NC from PD-MCI apart is the part doctors actually find tricky.

**What a scan looks like:** each person lay in an MRI scanner for about **6.7 minutes**, doing nothing — just resting. The scanner took **200 snapshots** of their brain in that time (one snapshot roughly every 2 seconds). That is called a **resting-state fMRI** scan (explained below). In total the dataset is **2.37 GB, 275 files**. The scans are raw — nobody had cleaned or pre-processed them before, so we do all of that ourselves.

---

## 4. Words you'll see everywhere, explained simply

**Parkinson's disease (PD):** a brain condition where dopamine-making cells gradually fail, causing tremor, stiffness and slow movement, and often thinking/memory problems later.

**HC / PD-NC / PD-MCI:** the three groups above — healthy controls, Parkinson's with normal cognition, Parkinson's with mild cognitive impairment.

**MCI (Mild Cognitive Impairment):** memory or thinking problems that are noticeable but not severe enough to stop someone living independently.

**MRI:** a scanner that takes pictures of the inside of the body using magnets — no radiation, no pain.

**fMRI (functional MRI):** a type of MRI that doesn't just photograph the brain's shape — it watches the brain *working*, by tracking blood flow, which rises where brain cells are active.

**Resting-state fMRI (rs-fMRI):** an fMRI scan taken while the person just lies still and thinks of nothing in particular. Even at rest, brain regions keep chatting with each other — and the *pattern* of that chatter is what we study.

**Volume:** one 3D snapshot of the whole brain. Each person's scan has 200 volumes.

**Brain parcel (region):** to make sense of a brain, scientists divide it into patches. We use the **Schaefer 100 atlas**, which splits the brain into **100 parcels**. Think of it like dividing a country into 100 districts so you can talk about each one separately.

**Functional connectivity:** for each pair of parcels, we ask: do these two districts activate in sync, or independently? If two regions rise and fall together, they are "connected". With 100 parcels, there are **4,950 pairs** — and the strength of all 4,950 connections is the fingerprint we feed the models.

**Connectome:** the full map of all those connections — the brain's wiring diagram, but for activity rather than physical wires.

**Preprocessing:** cleaning the raw scans before any learning happens — fixing head movement, lining the brain up to a standard template, removing noise. Raw scans are messy; models need clean input.

**Motion correction:** one preprocessing step that fixes the small head movements every person makes inside the scanner. (Honest note: the very first subject's scan kept failing this step, so the full run skips motion correction and says so in the logs. It is a known caveat, not a secret.)

**Feature:** one number the model is allowed to look at. Here, each of the 4,950 connection strengths is one feature.

**Machine learning (ML):** teaching a computer to find patterns from examples, instead of writing rules by hand. We show it 55 brains with labels; it learns the pattern that separates the groups.

**Deep learning:** machine learning with multi-layered neural networks — powerful, but hungry for data. With only 55 brains, the biggest deep models tend to just memorise the training set and fail on new people.

**Model:** one trained pattern-finder. We train **seven** different ones and compare them (Section 6).

**Epoch:** one full pass through all the training brains. We allow up to **1000 epochs**, but with **early stopping** — training halts automatically when the model stops improving, so it usually stops after 50–200 epochs instead of grinding through all 1000 and memorising.

**Training / validation / test:** the brains are split into groups — most are used to teach the model, some are held back to check whether it actually learned or just memorised. The held-back brains are the honest test.

**5-fold cross-validation:** we split the 55 brains into 5 parts, train on 4 parts and test on the 5th, then rotate so every part gets tested once. Every model gets judged on the exact same splits, so the comparison is fair.

**Accuracy:** the share of test brains the model puts in the right group. 100% would be perfect; 33% is random guessing with three groups.

**Confusion matrix:** a table showing *which* mistakes the model makes — e.g. does it confuse PD-NC with PD-MCI? That matters more than one overall number.

**Overfitting:** when a model memorises the training brains instead of learning the real pattern, so it looks great in training and fails on new people. The main enemy of this project, because 55 brains is small data.

**CPU / GPU:** the computer chips that do the work. A GPU is a specialist chip that trains models fast. We train on an ordinary CPU-only machine, so everything is slower — that is why training takes hours, not minutes.

**Vercel:** the service that hosts the website. **GitHub:** where all the code, logs and results live. **Next.js:** the framework the website is built with.

---

## 5. What we actually do to a brain scan (the pipeline)

1. **Download and audit:** fetch ds005892, check every file is present and intact (275 of 275 files, 2.37 GB, all 55 subjects complete).
2. **Preprocess each brain:** clean the 200-volume scan — align it, remove noise, and (except for the motion-correction caveat above) correct for head movement. About 2 minutes per person on our machine.
3. **Split into 100 parcels:** apply the Schaefer atlas so every brain is measured in the same 100 districts.
4. **Build the connectome:** for each person, compute the 4,950 connection strengths — the fingerprint.
5. **Train seven models:** each model learns on the same fingerprints and the same 5-fold splits, up to 1000 epochs with early stopping.
6. **Score and compare:** measure accuracy and the confusion matrix for each model on held-back brains, pick the honest winner.
7. **Publish:** real numbers go into the website's results section and into this repo (metrics, logs, model card). No fake numbers, ever — while training runs, the site says "training in progress".

---

## 6. The seven models, in one line each

| Model | Plain-English version |
|---|---|
| **SVM (Support Vector Machine)** | Draws the best dividing lines between the three groups. Simple, strong on small data. |
| **Logistic Regression** | Estimates the probability of each group from the connections. The classic baseline. |
| **Random Forest** | A crowd of decision trees that vote together. Hard to fool, easy to trust. |
| **XGBoost** | Trees that learn from each other's mistakes, one after another. Often the best of the simple methods. |
| **MLP (Multi-Layer Perceptron)** | A small neural network on the 4,950 connections. |
| **1D-CNN** | A neural network that slides a small window across the connection pattern looking for local shapes. |
| **GCN (Graph Convolutional Network)** | A neural network built for graphs — it treats the brain as a network of connected districts, which is exactly what a connectome is. Usually the best fit for this kind of data. |

**Why test all seven?** Because with only 55 brains, nobody can guess the winner in advance — the simple models (SVM, XGBoost) often beat the fancy ones here. We let held-out data decide, not hype. A raw-scan 3D deep network was deliberately *not* used: with 55 brains it would just memorise the scans.

---

## 7. The website — a tour

- **Hero — the brain network:** the connectome itself, live on the page. Hover any parcel to see its details, click one to isolate its connections, drag nodes around, watch pulses travel along the edges. Switch between **HC / PD-NC / PD-MCI** and the whole network reshapes to show that group's average pattern. The glow follows your mouse.
- **Compare slider:** drag it to morph between two groups and see what actually changes in the wiring.
- **Method:** the pipeline from Section 5, drawn out as a timeline you can expand.
- **Models:** cards for all seven models — what each one is, in plain words.
- **Interactive demo:** pick a sample subject, or upload a scan, and walk through the steps — choose input, see the connectivity, get a prediction.
- **Results:** while training runs this honestly says **"Training in progress"**. When training finishes, the real numbers land here — best model, accuracy, and the confusion matrix — and the same numbers get saved in this repo.

---

## 8. Where things stand right now

- **Website:** live — v3, the premium interactive version, at https://parkinsons-early-detection-ds005892.vercel.app/
- **Training, round 1 — finished, 9 November 2026, about 8:23 PM:** all 55 brains preprocessed, all seven models trained and compared. The honest headline: the best model for telling all three groups apart was **GAT, at 40% balanced accuracy** — only a little above the 33% you get from guessing, so the three-way task is genuinely hard on 55 brains. The simpler two-way question — healthy vs Parkinson's — worked better: **logistic regression reached AUC 0.745** (about 71% accuracy). The hardest pair to separate was PD-NC from PD-MCI, which is also the pair doctors find trickiest. The full numbers, tables and charts are on the website's Results section and in the repo.
- **Training, round 2 — finished, 10 November 2026, about 5:17 PM:** motion correction on, better noise removal, both the 100- and 200-region maps, stricter testing (nested cross-validation, so no information leaks), a two-step classifier, tuned models, and the trained models saved. The three-way task improved: **logistic regression now gets 50% balanced accuracy** (up from 32%), and the Parkinson's-without-memory-problems group went from 2 out of 18 correct to 10 out of 18. The finer 200-region map did not help (best 44%). The two-step classifier did not help either (40%). **But the two-way question got worse, and we know why:** after proper cleaning, healthy-vs-Parkinson's fell to chance on the 100-region map (best AUC 0.49). A careful check proved the round-1 signal of 0.745 lived in the uncleaned data — the old method on the new cleaned data gives 0.48, the new method on the old data gives 0.69, and the old number reproduces exactly on the old data. Head movement alone is barely predictive (AUC 0.52), so the removed signal was motion and body noise hiding inside the brain connections, not real brain activity. The best two-way model now is XGBoost on the 200-region map (AUC 0.69). **Bigger data did not save us either:** pooling 264 people from four datasets scores about 0.60–0.62 when mixed, but drops to pure chance when tested on a dataset it has never seen — the model learns each scanner's fingerprint, not the disease. So the next step is making the data from different scanners comparable before training, not just adding more scans. PPMI, the big Parkinson's dataset, still needs Abhishek to register and sign a data-use agreement himself (the exact steps are in `data/DATASETS.md`). The full numbers are on the website's Results section and in the repo.

---

## 9. How to read the results when they land

- **Best model + accuracy:** the headline. Remember 33% is chance; anything in the 70–90% range on 55 brains would be genuinely good for three groups.
- **Confusion matrix:** the more interesting part — it shows *which* groups get mixed up. If PD-NC and PD-MCI are the confused pair, that tells us the models struggle exactly where doctors struggle too.
- **Per-group scores:** accuracy for HC, PD-NC and PD-MCI separately. A model that aces healthy brains but fails on PD-MCI is less useful than one that is solid on all three.

---

## 10. The honest limits (read this before believing any number)

- **This is not a diagnosis.** It is a research prototype built from 55 brains. It cannot tell any individual person whether they have Parkinson's, and it is not a medical device.
- **Small data.** 55 people is small for machine learning. The numbers will be honest, but they come with wide uncertainty — a different 55 people could shift them.
- **One dataset, one scanner setup.** Results on ds005892 do not automatically hold for other hospitals or scanners.
- **CPU-only training.** Slow, and it limits how big a model we can fairly train.
- **Known caveats, stated openly:** round 1 skipped motion correction after the first subject kept failing that step; round 2 switched it back on, and that change is exactly what exposed that the round-1 two-way score was partly noise. The audit flagged a couple of unusual scans. Both are in the logs.
- **No fake numbers anywhere.** If a section says "training in progress", it means the numbers are not in yet — not that they are hidden.

---

## 11. Links

- **Website:** https://parkinsons-early-detection-ds005892.vercel.app/
- **This repo:** https://github.com/steeellord/parkinsons-early-detection-ds005892
- **Dataset (OpenNeuro ds005892 v1.0.0):** http://openneuro.org/datasets/ds005892/versions/1.0.0
- **Made by Rishabh.** Questions? Ask Abhishek — he has the logs.

---

## 12. Build log — problems we faced & how we tackled them

This is the running diary of the project: every real problem, what went wrong, what we tried, and what worked. Newest at the bottom.

**9 Nov 2026, evening — Round 1 finished, and the headline was honest but weak.**
The first full training run finished at about 8:23 PM: all 55 brains cleaned, all seven models trained. The best three-way score was GAT at 40% balanced accuracy — barely above the 33% you get by guessing. The healthy-vs-Parkinson's task was stronger (logistic regression, AUC 0.745). The problem was not the code — it was the size of the data. With only 55 brains, the models simply do not have enough examples to learn the subtle difference between PD-NC and PD-MCI. Nothing was hidden or dressed up: the real numbers went on the website and into the repo exactly as they came out.

**9 Nov 2026, evening — Round 1 had a gap: no trained model was saved.**
The first run saved scores, charts and logs, but not the trained models themselves — so there was nothing to load later to make a real prediction on a new scan. That was an oversight in the training script. Round 2 fixes it: every model's final version gets saved to disk.

**9 Nov 2026, evening — Motion correction was skipped in round 1, and that hurt.**
In round 1, the very first brain kept failing the motion-correction step, and that step is slow — about five and a half minutes per brain on our machine, which would have added five hours. So the whole run skipped it and said so openly. But uncorrected head movement is noise the models have to fight through, so round 2 switches motion correction back on and budgets the time for it. It also adds proper noise removal (motion-based regressors, censoring the worst movement frames, and high-variance noise signals), which round 1 did not have at all.

**9 Nov 2026, evening — The computer restarted mid-run once, and the results survived.**
During round 1 the machine restarted and killed the training run halfway. Because the script saved its results after the three-way models finished (a checkpoint), the re-run only had to redo the last part and produced exactly the same numbers. Lesson learned: save early, save often. Round 2 does the same — it writes its results to disk after each atlas finishes, so a restart can never wipe a whole run.

**9 Nov 2026, evening — Round 2 test: the first attempt failed on a file-reading bug, and the fix worked.**
The new cleaning script was tested on one brain first (sub-MJF001). Motion correction itself worked (about six minutes), but the script then crashed trying to read the motion files as plain text — one file contained bytes that are not valid text, so the read failed. The fix was to read those files through the same brain-imaging library that wrote them, instead of parsing the text by hand. The second test passed end to end in 10.2 minutes: motion correction, motion records saved, 36 of 200 volumes censored for excess movement, and clean connectivity maps for both the 100- and 200-region atlases. One honest side-effect of the cleaning: the average connection strength for this brain dropped from 0.31 to 0.19 — the noise the cleaning removed had been inflating the numbers.

**9 Nov 2026, evening — Round 2 full run launched.**
All 55 brains are now being re-cleaned with the fixed script, in two parallel halves (28 + 27 brains) because one brain takes about 10 minutes and a single sequential run would take nine hours. Expected finish: roughly five hours. Each brain's motion record, framewise-displacement numbers and censored-volume counts are saved alongside its connectivity maps, so every step can be checked later.

**9 Nov 2026, evening — One of the two parallel runs died silently; the other carries on.**
About twenty minutes into the full run, the first half (run A) vanished mid-step — no error in its log, it simply stopped during normalisation. The most likely cause is memory pressure: two heavy brain-imaging jobs at once, at the exact step where a whole 4D brain is held in memory. The second half (run B) was unaffected and is still running, one brain done. Rather than risk losing work again, the halves now run one at a time: run B finishes its 27 brains first, then run A restarts (it will skip the one brain already done in testing) and finishes its 28. Slower than planned, but nothing is lost — every finished brain stays saved.

**9 Nov 2026, evening — Run B also died, on a transform file the software could not read back.**
Run B finished two brains, then stopped during the third brain's motion correction: the imaging library reported it could not read back a motion file it had just written to the temporary folder. The file lives in /tmp, which on this machine is a small temporary disk — the most likely cause is that the file vanished or was never fully written there. Two fixes went in: motion files are now written to a stable folder inside the project (and kept, so the motion record survives), and the run now retries automatically — if a brain fails, the run picks up where it left off instead of stopping. Run B restarted with these fixes.

**9 Nov 2026, evening — The restarted run was killed by the machine running out of memory.**
The restarted run B died five times in a row, each time within seconds of starting, with exit code 139 — the code the system uses when it kills a process for using memory that is not there. At the time, the machine's memory was completely full (0 free): another job on the same machine was running a large software install (over 1 GB) at the same moment. The motion file left behind was 0 bytes long — not a bug in the file, but the run being killed halfway through writing it. The fix here is not code: the run waits until memory is free and then restarts. The retry wrapper is ready; the run resumes as soon as there is room.

**9 Nov 2026, evening — The service restarted and wiped the temporary folder, killing both runs.**
The machine's service restarted without warning. That killed run B mid-brain and the training dry run, and wiped everything in the temporary folder — including the retry script itself, which had been saved there. Lesson learned the hard way: nothing that must survive goes in the temporary folder. The retry script now lives inside the project (training/run_v2.sh), and run B was restarted from it. The dry run was rebuilt from scratch and restarted too. Finished brains are never lost — the run always skips brains it has already done.

**9 Nov 2026, evening — The training script's parallel search crashed, and was fixed the same way: one at a time.**
A dry run of the new training script (on 20 made-up brains, just to check the code) crashed partway: the parallel job system hit a limit in this machine's sandbox and could not start new workers. The fix was to run all searches one at a time instead of in parallel — slower, but it cannot hit that limit. The dry run is being repeated with the fix to prove the full training path works before the real brains finish cleaning.

**9 Nov 2026, evening — The dry run found a real bug in the training script: the model-saving step crashed.**
The dry run (classical models only) finished cleanly in 21.6 minutes — nested cross-validation, the two-stage classifier, full-data refits saved as files, figures, and the results file all worked. But when the neural-network models were added, their training ran and then the save step crashed: the code used `torch.save` in two places where the `torch` name was never imported (the import only existed inside other functions). Two one-line fixes — an `import torch` at the top of each of those two functions — and a focused re-test proved it: all five fold checkpoints and the full-data model file saved correctly. The lesson: the dry run earned its keep. The same crash would have wasted the whole real training run.

**9 Nov 2026, evening — The cleaning run was killed again, this time by the dry run itself.**
While the neural-network dry run was loading its libraries, the cleaning run (brain MJF040) was killed by the machine running out of memory — exit code 137. The two jobs plus other work on the shared machine peaked at the same moment. The retry script caught it and restarted the run automatically, picking up at MJF040. Nothing was lost: finished brains are always skipped on restart.

**9 Nov 2026, evening — The same kill kept happening, roughly every 15 minutes, even with nothing else of mine running.**
Brain MJF040 was killed three times in a row during its normalisation step, then finished on the fourth try; the next brain was killed the same way. Each kill was exit code 137 — the machine's out-of-memory killer — and each time other jobs on the shared machine (a video render, other agents' work) were using memory at the same moment. The cleaning job is simply the biggest memory user, so it is the one that gets killed. Two changes: the retry script now tries up to 40 times instead of 8, and it writes down the machine's free memory at the start of every try, so the record shows what was happening. Progress is still safe — every finished brain stays finished, and the run keeps coming back.

**9 Nov 2026, evening — The service restarted and killed even the retry script, so a watchdog was added.**
Around 18:45 the machine's service restarted without warning. That killed the cleaning run and the retry script itself, and wiped the temporary folder again. A retry script cannot restart itself after that, so a small watchdog was set up: every 20 minutes it checks whether the cleaning run is going, and if it is not, it starts the half that is still unfinished (the second half first, then the first half). It does nothing while a run is already going, and it stops starting anything once all 55 brains are done. The watchdog is a temporary helper — it can be removed once the cleaning is finished and the results are merged.

**9 Nov 2026, evening — Found the real cause of the kills: one step in the cleaning briefly uses 3.7 GB.**
Watching the run's memory second by second showed the pattern: most of the time it uses about 500 MB, but during normalisation — when each brain is warped onto the standard brain template — it briefly jumps to 3.7 GB. That one step is the biggest memory user on the whole machine, so it is the one the system kills whenever anything else needs memory at the same moment. The fix was to warp the brain in blocks of 25 scans at a time instead of all 200 at once. The maths is identical for each scan, so the result is the same — proved bit-for-bit identical on a test image, then proved again on a real brain (MJF001): the new code peaks at about 0.8 GB instead of 3.7 GB and produces valid results. Every run from now on uses the new code, so the kills should become rare. The test files were deleted after the check so they cannot be mixed into the real results.

**9 Nov 2026, evening — The new code survived normalisation on a real brain; the run was restarted after another service restart.**
Brain MJF043 was the first to go through normalisation with the new block code, and it finished that step cleanly. Minutes later the service restarted again and killed the run just as it was computing that brain's results, so the run was restarted by hand (the watchdog would also have caught it within 20 minutes). MJF043 starts over from the beginning, as always — only finished brains are kept.

**9 Nov 2026, evening — The pooled-data tools are built and dry-run.**
While the cleaning runs, the tools for the bigger-datasets half were built and tested on made-up data: a builder that turns the BrainNet timeseries (PPMI, Neurocon, Tao Wu) into connectivity matrices using our own calculation, giving a binary pool of 209 brains (159 Parkinson's, 50 healthy); and a pooled training script that tests models two honest ways — ordinary cross-validation, and holding out one whole dataset at a time to see if the model works on a scanner it never saw. Both dry runs passed. The real pooled training waits until the main cleaning finishes, so the real numbers are not mixed with test ones.

**10 Nov 2026, early morning — The cleaning run died again with no exit line; restarted by hand.**
Around 22:24 UTC the second-half run vanished mid-brain (MJF054, two minutes into motion correction): no process left, no exit line in the log — the whole wrapper and its worker were gone together, which points to another service restart rather than the memory killer (that one leaves an exit line). The watchdog would have restarted it within 20 minutes, but it was restarted by hand straight away. 15 of 27 second-half brains done; 12 left.

**10 Nov 2026, 00:00 UTC — Second half of the cleaning is done; first half started.**
The second half finished cleanly: 27 of 27 brains, none failed, in about 95 minutes. The first half (28 brains, one already done) was started by hand straight away — the automatic watchdog would have picked it up within 20 minutes, but starting it now saves the wait. About four hours left for the first half, then the training runs.

**10 Nov 2026, ~02:33 UTC — Another silent death; watchdog and hand restart both in play.**
The first-half run vanished again mid-brain (MJF026, in normalisation) with no exit line — the same signature as before, a service restart taking the whole run and the pipeline driver with it. Earlier, at 00:34, the watchdog had already caught one such silent death and restarted the run correctly on its own. This time the run was restarted by hand within a minute, and the pipeline driver too. The watchdog stays on as the safety net.

**10 Nov 2026, 03:51 UTC — All 55 brains cleaned; the merge caught a stale folder, fixed.**
The first half finished cleanly: 28 of 28, none failed. The merge step then tripped on a leftover folder from an early test — it held one brain's files and a labels file, so the pipeline wrongly thought the merge was already done and the training crashed on a single brain. The stale folder was deleted, the merge re-run by hand (55 of 55, both atlases, all motion records — confirmed), and the pipeline's check was tightened so it counts the brains instead of just looking for a file. Training is now running on the real 55 brains. First number in: logistic regression, balanced accuracy 0.50, up from 0.32 in the first round — the cleaning and the proper tuning are already paying off.

**10 Nov 2026, ~04:38 UTC — Training was killed mid-way; it now saves its work as it goes.**
The training run was cut off by another service restart just as it reached the last fold of the graph model — 44 minutes of work lost, because it only saved its results at the end of each atlas. The training script was changed so it saves after every model it finishes, and the change was tested on made-up data first (a re-run picked up exactly where the first stopped, in 0.1 minutes instead of 7.8). Training restarted from the beginning and is already reproducing the same numbers as before, which is a good sign they are solid. From here, any interruption costs at most one model, not the whole run.

**10 Nov 2026, ~05:42 UTC — A real bug found in the new save-as-you-go code; fixed and resumed.**
While checking the training log, the binary (healthy-vs-Parkinson's) section was missing its three neural-network models. The cause: the code ranked all models to pick the best three for the binary run, but the top three were the classical models, which already had binary results — so it skipped them as "already done" and ran none. The ranking is now limited to neural models only. The run was stopped, the code fixed, and it resumed from the saved checkpoint, skipping everything already done (all eight 3-class models and the four classical binary models) and going straight to the missing binary neural models. Only the half-finished two-stage step had to restart.

**10 Nov 2026, ~06:09 UTC — The binary neural models failed on a filename; fixed, and the pipeline's "done" check was wrong.**
The three binary neural models all crashed seconds after starting: they tried to save their weights to a folder named after "atlas100/binary", which does not exist — the slash in the name was read as a folder. The fix flattens the name (atlas100_binary_...). Stopping to fix it exposed a second flaw: the pipeline driver judged training "done" just because a results file existed, even though it only held the first atlas — so it skipped ahead to the pooled training. The check now reads the file and confirms both atlases are inside. The pooled training is running now; the second atlas's training will be run right after it, resuming from the saved checkpoint.

**10 Nov 2026, ~10:23 UTC — Stopped the slow neural models on the fine atlas; they were not worth the hours.**
The three neural models on Schaefer-200 were taking about 30 minutes per fold (five folds each, so roughly seven hours for the three) because they have to learn from 19,900 features with only 55 people — and their inner test scores were already at chance level (0.37–0.49), so they were not going to beat the classical models. They were marked skipped in the checkpoint with the reason written in, and the run moved on to the remaining steps (binary models, two-stage, saved weights). The atlas-200 comparison is still complete for the classical models, and all eight models are complete on atlas-100. Nothing was hidden: the skipped entries are recorded in the results file with the reason.

**9 Nov 2026 — The bigger-datasets search is finished; its verdict is in.**
A full search for other Parkinson's brain-scan datasets is done and written up in the project's data notes. What landed, all free and openly downloadable: two more OpenNeuro sets (45 brains from Japan, 28 de novo Parkinson's brains), the INDI collection (43 + 40 brains), and the Auckland BrainNet set (195 PPMI brains plus the same INDI brains as ready-made connectivity matrices). PPMI itself — the Michael J. Fox dataset — is gated: Abhishek personally has to create a LONI account, apply, and sign a data-use agreement, then wait about a week; no fee. Everything else is already on disk and checked file by file. Two real snags were found and fixed: an extraction that silently produced empty files (re-extracted, all 208 files verified), and Tao Wu scan files that are misnamed — they say ".gz" but are not actually compressed, so any future cleaning step has to handle that. The honest word on the 90% goal: published results on these kinds of scans sit around 55–57% for graph models, so the defensible aim is to beat the current 0.745 AUC and report exactly what is measured, per dataset, not to promise 90%.

**9 Nov 2026, evening — The search for bigger data started.**
Fifty-five brains is the real ceiling on accuracy, so in parallel we started looking for bigger Parkinson's datasets — most importantly PPMI, the Michael J. Fox Foundation's study, which has hundreds of people. PPMI data is not a free download: it needs an account and a data-use agreement, so that part may need Abhishek to apply personally. Other open datasets are being checked too. If a bigger dataset lands, the same pipeline runs on the pooled data and the results get reported honestly, including any differences between datasets.

**9 Nov 2026, evening — Two standing rules for everything from here on.**
First, Abhishek plans to patent this work later, so everything must be real and defensible — real data, real measured results, methods anyone can reproduce, and honest dated logs. This build log doubles as the invention record: what was tried, what failed, what worked, and when. Nothing invented, nothing fudged — a patent built on fake numbers is worthless. Second, the code must read like a person wrote it: no comment or file anywhere may say or suggest it was made by an AI. The work has to stand on its own as human-built engineering.

**9 Nov 2026, late evening — The website stopped asking for a spreadsheet and started asking for a brain scan.**
Until now the site's "try it" section asked you to upload a CSV file — a 100×100 table of numbers (the connectivity matrix). That was the wrong input: nobody has a CSV of their brain lying around, and a spreadsheet is not a brain scan. Abhishek was blunt about it — the site must take an actual scan file and give back a result. So the input changed completely. The site now asks for two real scan files: the resting-state fMRI (`.nii.gz`, about 30 MB) and the matching structural T1 scan (about 10 MB). Those go to a small inference service that runs the exact same cleaning steps as training — lines the scans up to a standard brain, splits it into 100 regions, measures the 4,950 connections — then runs the saved trained model on the result and returns the prediction with its confidence, in plain English. The CSV option is still there, tucked under "already have a connectivity matrix?", for people who already computed one — but the scan upload is the main path now. The service runs on Hugging Face's free tier (a public Gradio Space on their ZeroGPU hardware), which costs nothing: the job runs on CPU, takes 3–6 minutes a scan, and your files are deleted when the job finishes. A step-by-step plain-English guide to the exact files lives in `INPUT_GUIDE.md` at the repo root, and the deployment steps in `inference/DEPLOY.md`.

**9 Nov 2026, late evening — Why a photo of a brain can never work.**
This came up directly: why can't you just upload a photo of a brain scan? Because the model never looks at a picture. It reads a time series — how 100 brain regions rise and fall together over 200 time points. A PNG or JPG is one flat image: no time dimension, no regions rising and falling, nothing to measure. A screenshot, a phone photo of a screen, a single MRI slice — all the same problem. So the service refuses photos and says exactly why, in plain words, and tells you which two files it needs instead. It would be dishonest to pretend a photo could be diagnosed, so the refusal is built in, not an afterthought.

**9 Nov 2026, late evening — The first real scan ran end to end, and the answer was right.**
Before wiring anything into the website, the service was tested on a real person from the training data: sub-MJF003, a healthy control. Upload its fMRI and T1 through the service, and it comes back healthy — 99% confidence on the healthy-vs-Parkinson's call, 99.4% on the three-group call, in about 3 minutes. The connectivity pattern it computed matches the one from training (correlation 0.96) — not bit-identical, because the brain-alignment step varies slightly from run to run, which is normal and recorded here honestly. The website's Use-it section now talks to this service directly, and its Results section was rebuilt at the same time: plain-English summary at the top, click any model to expand its confusion matrix (hover a cell and it explains itself), per-group accuracy bars, a switch between the three-group and the two-group view, and a compare-two-models picker — all reading straight from `results.json`, so new numbers from round 2 appear with no code change. One honest caveat: the model running in the service right now is an interim one (logistic regression, trained without motion correction, the same as round 1's binary model). When round 2 finishes and saves its better weights, the service swaps them in and its stated accuracy updates with them — the service never claims to be better than the model it is actually running. A second caveat, also recorded honestly: the photo refusal was verified over the web (upload a PNG, get the plain-English explanation back), and the full scan was verified by running the service code directly — but the full scan over the web could not be finished on our dev machine, because three attempts were killed when the machine ran out of memory while round 2's preprocessing was running alongside it. The service on its hosting (Hugging Face) gets its own separate memory, so that clash does not apply there — the first scan test on the live hosting is still to do, and the steps are written down in `inference/DEPLOY.md`.

**10 Nov 2026, ~12:02 UTC — The diagnostic proved the round-1 two-way score was partly noise.**
The four-way check is now measured, not guessed. Old method on new cleaned data: AUC 0.48. New method on old data: AUC 0.69. Old method on old data: AUC 0.745 — the round-1 number reproduces exactly, so there was no label or split mistake. New method on new data: AUC 0.45. Head movement alone as a score: AUC 0.52. So the round-1 signal lived in motion and body noise inside the brain connections — the kind of noise that proper cleaning removes. The three-way signal is the real one: it got better with cleaning. Two small bugs in the diagnostic itself were fixed along the way (it looked for the old feature files under the wrong name, and it read the motion summary from a file that only lists 22 of the 55 people — it now reads the per-person motion files directly).

**10 Nov 2026, ~12:03 UTC — Round 2 published.**
The round-1 numbers are archived (`training/metrics_v1.json`), the round-2 numbers are now the live ones (`training/metrics.json`, `website/public/results.json`), the model card and the README's results section are rewritten with the new tables, the binary confusion-matrix figures are regenerated, and `npm run build` passes. The watchdog that watched the preprocessing was removed — its job is done. The three-way headline is now logistic regression at 50% balanced accuracy; the two-way headline is XGBoost on the 200-region map at AUC 0.69; and the pooled-data headline is that a model trained on four datasets scores 0.60–0.62 mixed together but drops to chance on a dataset it has never seen. None of these is 90%. The next step is making data from different scanners comparable before training — that is the bottleneck the numbers point to, not the choice of model.

**10 Nov 2026, afternoon UTC — Round 3 started: teaching the model to ignore which scanner took the scan.**
Round 2 left one clear problem: mix four datasets together and the model scores 0.60–0.62, but test it on a dataset it has never seen and it falls to chance (about 0.50 — guessing). That means the model learned each scanner's fingerprint, not the disease. Round 3 attacks exactly that, with a standard method called ComBat: it measures how much each scanner shifts every brain connection and removes that shift, while carefully keeping the real differences — diagnosis, age, sex — intact. Two honest rules in how it is applied: the correction is always learned from the training data only, never from the data being tested (otherwise the test would be rigged), and a scanner the model has never seen gets no correction at all, because there is nothing to correct it with. A simpler backup is tried alongside it (standardising each scanner separately). The same four models and the same test setup as round 2 are used, so the numbers are directly comparable. The question being answered is simple: after this cleaning, does the model beat guessing on a brand-new scanner — and by how much? If it does not, that will be reported plainly too; a negative answer is still an answer. One wrinkle found while setting this up: the age and sex of every person had to be dug out of the dataset files first — the pooled table did not have them — and they were found in the metadata files for all 264 people, including a fix for two datasets whose ID formats did not match at first. Nothing is published yet; the run takes a few hours and the numbers land here when they are measured.

**10 Nov 2026, evening UTC — Round 3 finished, and the answer is no.**
Harmonisation did not lift the model above guessing on a new scanner. The best result after ComBat is 0.54 balanced accuracy (a support-vector model, AUC 0.50 — a coin flip) against 0.51 without any harmonisation. The simpler standardising method did no better (best 0.51), and even a deliberately generous version of ComBat — one allowed to look at the new scanner's data before being tested on it — only reached 0.53. Worse, the mixed-together score actually fell after cleaning: from 0.60–0.62 down to 0.59 with ComBat and 0.51 with standardising. That is the tell: the mixed score really was the scanners' fingerprints, and once those are removed there is almost no disease signal left in these brain-connection maps that travels from one scanner to another. The harmonised model tested on the home dataset scores 0.55–0.56 — still guessing. One model (random forest) gave up entirely and predicted every single person as having Parkinson's in nearly every test. The graph neural network was not run this round, on purpose: when every simpler model and even the generous version of the cleaning sit at guessing, a fancier model on the same data has nothing new to find, and that is recorded as a deliberate skip, not a gap. Two saved models exist from this round (the best mixed-together one and the best new-scanner one, plus the fitted cleaning parameters), but neither is good enough to use in the live service — the service keeps its current weights, and the decision about those weights from round 2 is still open. The run itself survived two interruptions from the machine restarting, and was picked up exactly where it stopped both times — the numbers came out identical after each restart, which is how we know the resume worked rather than silently changing anything. The code that does the cleaning was checked against the published ComBat software and matches it to within 0.0000002, so the negative result is not a bug in the code. What would actually help now is different information, not more cleaning: brain measures that change over time instead of one static snapshot, structural scans alongside the functional ones, or a much larger group of people from one scanner. The 90% goal is further away than hoped — the honest summary of all three rounds is that this kind of brain scan, at this size, tells us the disease is there about half the time, which is guessing. The full numbers are in `training/metrics_round3.json`, the model card carries the whole table, and the website's results page now says the same thing in plain words.
