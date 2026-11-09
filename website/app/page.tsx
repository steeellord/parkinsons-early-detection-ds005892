'use client';

import { useEffect, useMemo, useState } from 'react';

const DATASET_URL = 'https://openneuro.org/datasets/ds005892/versions/1.0.0';
const GITHUB_URL = 'https://github.com/steeellord';

/* ---------- seeded RNG (deterministic, illustrative patterns only) ---------- */
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type GroupKey = 'HC' | 'PD-NC' | 'PD-MCI';
const GROUPS: { key: GroupKey; label: string; subject: string; seed: number; blurb: string }[] = [
  { key: 'HC', label: 'Healthy Control', subject: 'sub-HC01 (illustrative)', seed: 11, blurb: 'Strong within-network synchrony, quiet between-network coupling — the reference pattern.' },
  { key: 'PD-NC', label: 'PD — Normal Cognition', subject: 'sub-PDNC01 (illustrative)', seed: 47, blurb: 'Within-network synchrony is weaker than in controls, especially in motor networks.' },
  { key: 'PD-MCI', label: 'PD — Mild Cognitive Impairment', subject: 'sub-PDMCI01 (illustrative)', seed: 83, blurb: 'More diffuse, less organised coupling — within-network structure is further reduced.' },
];

const N = 20; // 20 representative parcels shown (full analysis uses Schaefer 100)
const NETWORKS = ['Visual', 'Motor', 'Default', 'Fronto-parietal'];

function buildMatrix(group: GroupKey, seed: number): number[][] {
  const rand = mulberry32(seed);
  const m: number[][] = Array.from({ length: N }, () => Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const sameNet = Math.floor(i / 5) === Math.floor(j / 5);
      let base: number;
      if (group === 'HC') base = sameNet ? 0.55 + rand() * 0.3 : 0.05 + rand() * 0.2;
      else if (group === 'PD-NC') base = sameNet ? 0.35 + rand() * 0.3 : 0.08 + rand() * 0.25;
      else base = sameNet ? 0.2 + rand() * 0.3 : 0.15 + rand() * 0.28;
      const v = Math.max(-0.2, Math.min(0.95, base + (rand() - 0.5) * 0.12));
      m[i][j] = v;
      m[j][i] = v;
    }
    m[i][i] = 1;
  }
  return m;
}

function colorFor(v: number): string {
  // -0.2 deep navy -> 0 muted slate -> 1 bright teal/cyan
  if (v >= 1) return '#5eead4';
  if (v <= 0) {
    const t = Math.max(0, Math.min(1, (v + 0.2) / 0.2));
    const r = Math.round(11 + (18 - 11) * t);
    const g = Math.round(42 + (46 - 42) * t);
    const b = Math.round(74 + (61 - 74) * t);
    return `rgb(${r},${g},${b})`;
  }
  const t = v;
  const r = Math.round(18 + (94 - 18) * t);
  const g = Math.round(46 + (234 - 46) * t);
  const b = Math.round(61 + (212 - 61) * t);
  return `rgb(${r},${g},${b})`;
}

function meanOf(m: number[][], within: boolean): number {
  let s = 0, c = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const same = Math.floor(i / 5) === Math.floor(j / 5);
    if (same === within) { s += m[i][j]; c++; }
  }
  return s / Math.max(1, c);
}

type ResultsFile = {
  status: string;
  models: unknown[];
  note?: string;
  last_updated?: string;
  task?: string;
};

const MODELS = [
  { name: 'Graph Neural Network (GCN / GAT)', fit: 'Treats each brain as a graph: parcels are nodes, connectivity strengths are edges. Built for exactly this kind of data — network structure is the signal, not noise.', note: 'Primary candidate. Attention weights can highlight which connections matter most.' },
  { name: 'MLP on Connectivity', fit: 'A feed-forward network on the flattened upper triangle of the 100×100 connectivity matrix (4,950 features). Simple, fast on CPU, strong when regularised.', note: 'Baseline deep model; dropout + weight decay are essential at n = 55.' },
  { name: '1D-CNN', fit: 'Convolution over the connectivity profile of each parcel, learning local patterns across neighbouring regions before pooling to a class.', note: 'Mid-complexity; tests whether local connectivity motifs carry the signal.' },
  { name: 'SVM (RBF)', fit: 'A classic that routinely beats deep nets on small neuroimaging samples. The RBF kernel handles non-linear boundaries in connectivity space.', note: 'Honest baseline — if a deep model cannot beat this, it has not earned its complexity.' },
  { name: 'Logistic Regression', fit: 'Linear, interpretable, and hard to overfit. Coefficients map directly back to individual connections.', note: 'Interpretability reference; also the sanity check for class separability.' },
  { name: 'Random Forest / XGBoost', fit: 'Ensembles of decision trees that rank connections by importance and cope well with correlated features.', note: 'Useful for feature-importance cross-checks against the GNN attention.' },
  { name: '3D CNN (comparison only)', fit: 'Learns directly from raw T1 / fMRI volumes with no atlas or connectivity step. The most data-hungry option here.', note: 'Included for completeness — expected to overfit at n = 55. Included so the claim is tested, not assumed.' },
];

export default function Home() {
  const [group, setGroup] = useState<GroupKey>('HC');
  const [results, setResults] = useState<ResultsFile | null>(null);

  const active = GROUPS.find((g) => g.key === group)!;
  const matrix = useMemo(() => buildMatrix(group, active.seed), [group, active.seed]);
  const within = useMemo(() => meanOf(matrix, true), [matrix]);
  const between = useMemo(() => meanOf(matrix, false), [matrix]);

  useEffect(() => {
    fetch('results.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setResults(d))
      .catch(() => undefined);
  }, []);

  const cell = 22;
  const size = N * cell;

  return (
    <>
      <header className="nav">
        <div className="nav-inner">
          <a className="brand" href="#top">Parkinson&apos;s <span>Early Detection</span></a>
          <nav className="nav-links" aria-label="Sections">
            <a href="#dataset">Dataset</a>
            <a href="#how">How it works</a>
            <a href="#models">Models</a>
            <a href="#demo">Demo</a>
            <a href="#results">Results</a>
          </nav>
        </div>
      </header>

      <div className="status" role="status">
        <div className="status-inner">
          <span className="dot" aria-hidden="true" />
          <span><strong>Training in progress.</strong> Models are being benchmarked on ds005892 — no results are claimed yet. This page updates automatically when training completes.</span>
        </div>
      </div>

      <main id="top">
        {/* HERO */}
        <section className="hero">
          <div className="wrap">
            <p className="eyebrow">Resting-state fMRI · Deep learning · OpenNeuro ds005892</p>
            <h1>Early detection of <em>Parkinson&rsquo;s disease</em>, from the brain at rest.</h1>
            <p className="lead">
              A 3-class deep-learning study that reads functional connectivity — how brain regions
              talk to each other when a person is doing nothing — and learns the patterns that
              separate healthy ageing from Parkinson&rsquo;s with normal cognition and with mild
              cognitive impairment.
            </p>
            <div className="badge-row">
              <span className="badge"><b>55</b> participants</span>
              <span className="badge">3 classes: <b>HC / PD-NC / PD-MCI</b></span>
              <span className="badge">Schaefer <b>100-parcel</b> atlas</span>
              <span className="badge">Max <b>1000 epochs</b> · early stopping</span>
            </div>
            <div className="cta-row">
              <a className="btn" href="#demo">Try the interactive demo</a>
              <a className="btn-ghost" href="#how">See the method</a>
            </div>
            <p className="hero-note">Research project by Abhishek Singh. Not a medical device and not a diagnosis — see Ethics &amp; Limitations.</p>
          </div>
        </section>

        {/* DATASET */}
        <section id="dataset">
          <div className="wrap">
            <p className="eyebrow">The data</p>
            <h2>One small, well-defined dataset</h2>
            <p className="sub">
              Resting-state fMRI and structural MRI from Kemp et al. (2025), shared openly on
              OpenNeuro under CC0. Small sample size is the defining constraint of this project —
              every modelling choice below is made because of it, not in spite of it.
            </p>
            <div className="grid grid-4">
              <div className="card"><div className="stat-num">55</div><div className="stat-label">Participants</div><div className="stat-sub">with complete data</div></div>
              <div className="card"><div className="stat-num">22</div><div className="stat-label">Healthy controls</div><div className="stat-sub">HC</div></div>
              <div className="card"><div className="stat-num">18</div><div className="stat-label">PD, normal cognition</div><div className="stat-sub">PD-NC</div></div>
              <div className="card"><div className="stat-num">15</div><div className="stat-label">PD, mild impairment</div><div className="stat-sub">PD-MCI</div></div>
            </div>
            <div className="grid grid-2" style={{ marginTop: 16 }}>
              <div className="card">
                <h3>Modalities</h3>
                <p>Resting-state fMRI (BOLD time series) + structural T1-weighted MRI, BIDS-formatted. Preprocessing via fMRIPrep derivatives where available, otherwise a Nilearn pipeline.</p>
              </div>
              <div className="card">
                <h3>Source &amp; citation</h3>
                <p>Kemp, A. S., Pyles, J. A., &amp; Bishop, C. (2025). Resting state MRI data from healthy control, Parkinson&rsquo;s disease with normal cognition, and Parkinson&rsquo;s disease with mild cognitive impairment cohorts. OpenNeuro. DOI: 10.18112/openneuro.ds005892.</p>
                <p style={{ marginTop: 10 }}><a href={DATASET_URL} target="_blank" rel="noreferrer">openneuro.org/datasets/ds005892 ↗</a></p>
              </div>
            </div>
          </div>
        </section>

        {/* HOW IT WORKS */}
        <section id="how" style={{ background: 'var(--bg2)', borderTop: '1px solid var(--line)', borderBottom: '1px solid var(--line)' }}>
          <div className="wrap">
            <p className="eyebrow">Method</p>
            <h2>How it works</h2>
            <p className="sub">Four steps, from raw scans to a class prediction. The key move is step 3: we never feed raw 4D brain volumes to a classifier — we feed the connectivity between regions.</p>
            <div className="grid grid-2 steps">
              <div className="card step"><h3>1 · Acquire</h3><p>Download ds005892 v1.0.0 from OpenNeuro. Verify BIDS structure, subject counts and missing data before anything is trained.</p></div>
              <div className="card step"><h3>2 · Preprocess</h3><p>fMRIPrep / Nilearn: motion correction, spatial normalisation to MNI space, nuisance regression and band-pass filtering of the BOLD signal.</p></div>
              <div className="card step"><h3>3 · Connectivity</h3><p>Parcellate with the Schaefer 100-region atlas, extract one time series per parcel, and compute Pearson correlations — a 100 × 100 functional-connectivity matrix per person.</p></div>
              <div className="card step"><h3>4 · Classify</h3><p>Benchmark seven model families on identical stratified cross-validation splits. Report accuracy, balanced accuracy, AUC, F1 and confusion matrices — never accuracy alone.</p></div>
            </div>
          </div>
        </section>

        {/* MODELS */}
        <section id="models">
          <div className="wrap">
            <p className="eyebrow">Models</p>
            <h2>Seven models, one fair comparison</h2>
            <p className="sub">No winner is declared in advance. Every family trains on the same splits, with the same early-stopping rule (max 1000 epochs). The winner is whichever generalises best to held-out participants — and if a simple SVM wins, the site will say so.</p>
            <div className="grid grid-2">
              {MODELS.map((m) => (
                <div className="card" key={m.name}>
                  <span className="tag">Benchmarked</span>
                  <div className="model-name">{m.name}</div>
                  <p className="model-fit">{m.fit}</p>
                  <p className="model-note">{m.note}</p>
                </div>
              ))}
              <div className="card" style={{ borderColor: 'var(--teal)' }}>
                <span className="tag">Evaluation</span>
                <div className="model-name">How the winner is chosen</div>
                <p className="model-fit">Stratified 5-fold cross-validation (repeated), plus leave-one-out as a sanity check. Primary metric: balanced accuracy and macro-AUC, because the classes are imbalanced (22 / 18 / 15).</p>
                <p className="model-note">Max 1000 epochs with early stopping on validation loss, cosine LR decay, and checkpointing of the best epoch — not the last one.</p>
              </div>
            </div>
          </div>
        </section>

        {/* DEMO */}
        <section id="demo" style={{ background: 'var(--bg2)', borderTop: '1px solid var(--line)', borderBottom: '1px solid var(--line)' }}>
          <div className="wrap">
            <p className="eyebrow">Interactive demo</p>
            <h2>What functional connectivity looks like</h2>
            <p className="sub">Pick a group to see a representative connectivity pattern across 20 parcels (4 networks × 5 parcels). Brighter teal means two regions fluctuate together more strongly. The full analysis uses all 100 Schaefer parcels.</p>
            <div className="demo-box">
              <div className="seg" role="group" aria-label="Sample subject group">
                {GROUPS.map((g) => (
                  <button key={g.key} className={group === g.key ? 'active' : ''} onClick={() => setGroup(g.key)} aria-pressed={group === g.key}>
                    {g.label}
                  </button>
                ))}
              </div>
              <div className="demo-grid">
                <div>
                  <svg className="heatmap" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Illustrative functional connectivity heatmap for ${active.label}`}>
                    {matrix.map((row, i) =>
                      row.map((v, j) => (
                        <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1} fill={colorFor(v)} />
                      ))
                    )}
                  </svg>
                  <div className="legend"><span>−0.2 · weaker coupling</span><span>Networks: {NETWORKS.join(' · ')}</span><span>+1.0 · stronger coupling</span></div>
                  <p style={{ marginTop: 14, color: 'var(--muted)', fontSize: '0.95rem' }}>{active.blurb}</p>
                </div>
                <div className="demo-stats">
                  <p><span>Sample subject</span><span>{active.subject}</span></p>
                  <p><span>Group</span><span>{active.label}</span></p>
                  <p><span>Parcels shown</span><span>20 of 100</span></p>
                  <p><span>Mean within-network r</span><span>{within.toFixed(2)}</span></p>
                  <p><span>Mean between-network r</span><span>{between.toFixed(2)}</span></p>
                  <p><span>Model inference</span><span>After training</span></p>
                </div>
              </div>
              <div className="disclaimer">
                <strong>Illustrative connectivity pattern — not a diagnosis.</strong> This heatmap is a
                seeded, simplified visualisation of the kind of pattern the models learn from. It is
                not a real patient scan, and model inference on uploaded data activates only after
                training completes and the winning model is published here.
              </div>
            </div>
          </div>
        </section>

        {/* RESULTS */}
        <section id="results">
          <div className="wrap">
            <p className="eyebrow">Results</p>
            <h2>Results — live from training</h2>
            <p className="sub">This section reads <span className="mono">public/results.json</span>, the same file the training pipeline writes to. Until a training run finishes, it shows the in-progress state — no numbers are invented to fill the space.</p>
            <div className="card">
              {results && results.status !== 'training_in_progress' && Array.isArray(results.models) && results.models.length > 0 ? (
                <p>Results are published — see the training log and metrics in the GitHub repository.</p>
              ) : (
                <>
                  <span className="tag" style={{ color: 'var(--amber)' }}>● Training in progress</span>
                  <h3 style={{ fontSize: '1.3rem' }}>No results yet — training is running.</h3>
                  <p>
                    The benchmark (GCN/GAT, MLP, 1D-CNN, SVM, Logistic Regression, Random Forest /
                    XGBoost, 3D CNN) is being trained with stratified cross-validation, max 1000 epochs
                    and early stopping. Accuracy, balanced accuracy, AUC, F1 and confusion matrices will
                    appear here once the run completes.
                  </p>
                  <div className="progress" aria-hidden="true"><div /></div>
                  <p style={{ marginTop: 14, fontSize: '0.88rem', color: 'var(--muted2)' }}>
                    Status: <span className="mono">{results?.status ?? 'training_in_progress'}</span>
                    {results?.last_updated ? <> · Last updated: {results.last_updated}</> : null}
                    {results?.note ? <><br />{results.note}</> : null}
                  </p>
                </>
              )}
            </div>
            <div className="grid grid-3" style={{ marginTop: 16 }}>
              <div className="card"><div className="stat-label">Accuracy</div><p>Pending — reported per model after cross-validation.</p></div>
              <div className="card"><div className="stat-label">Balanced accuracy / AUC</div><p>Pending — primary metrics, given the 22 / 18 / 15 class split.</p></div>
              <div className="card"><div className="stat-label">Confusion matrix</div><p>Pending — HC vs PD-NC vs PD-MCI, per fold and averaged.</p></div>
            </div>
          </div>
        </section>

        {/* ETHICS */}
        <section id="ethics" style={{ paddingTop: 0 }}>
          <div className="wrap">
            <div className="card">
              <p className="eyebrow">Ethics &amp; limitations</p>
              <h2 style={{ fontSize: '1.4rem' }}>What this is — and what it is not</h2>
              <p style={{ marginTop: 10 }}>
                <strong>Small sample.</strong> 55 participants is small for deep learning. Results must be
                read as exploratory, and any claim needs external validation on a larger, independent cohort.
              </p>
              <p style={{ marginTop: 10 }}>
                <strong>Not a medical diagnosis.</strong> This is a research demo. It does not diagnose,
                screen or triage any person, and it must never be used as a substitute for clinical
                assessment by a qualified clinician.
              </p>
              <p style={{ marginTop: 10 }}>
                <strong>Data &amp; consent.</strong> All data are de-identified and shared openly by the
                original authors on OpenNeuro under CC0. No new patient data are collected here.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div className="wrap">
          <div>
            <strong style={{ color: 'var(--text)' }}>Parkinson&rsquo;s Early Detection</strong><br />
            Built by Abhishek Singh · Dataset: OpenNeuro ds005892 (Kemp et al., 2025)
          </div>
          <div>
            <a href={DATASET_URL} target="_blank" rel="noreferrer">OpenNeuro ds005892 ↗</a>
            {' · '}
            <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub @steeellord ↗</a>
          </div>
        </div>
      </footer>
    </>
  );
}
