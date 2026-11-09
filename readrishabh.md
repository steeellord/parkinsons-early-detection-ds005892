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
- **Training:** running on Abhishek's machine (CPU-only). Preprocessing is going through the 55 subjects at roughly 2 minutes each, then the seven models train on top (another 30–60 minutes). Realistic finish: **around 8:45–9:00 PM tonight, 9 November 2026**. A watchdog checks every 20 minutes and restarts anything that dies, so it won't silently stall.
- **What updates when:** the moment training finishes, the results section of the site and this repo get the real numbers.

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
- **Known caveats, stated openly:** motion correction was skipped for the full run after the first subject kept failing that step, and the audit flagged a couple of unusual scans. Both are in the logs.
- **No fake numbers anywhere.** If a section says "training in progress", it means the numbers are not in yet — not that they are hidden.

---

## 11. Links

- **Website:** https://parkinsons-early-detection-ds005892.vercel.app/
- **This repo:** https://github.com/steeellord/parkinsons-early-detection-ds005892
- **Dataset (OpenNeuro ds005892 v1.0.0):** http://openneuro.org/datasets/ds005892/versions/1.0.0
- **Built for Rishabh, by Abhishek.** Questions? Ask Abhishek — he has the logs.
