'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

const DATASET_URL = 'https://openneuro.org/datasets/ds005892/versions/1.0.0';
const GITHUB_URL = 'https://github.com/steeellord';
const REPO_URL = 'https://github.com/steeellord/parkinsons-early-detection-ds005892';

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

const GROUP_META: Record<GroupKey, { label: string; short: string; color: string }> = {
  HC: { label: 'Healthy control', short: 'HC', color: '#2dd4bf' },
  'PD-NC': { label: 'Parkinson\u2019s, normal cognition', short: 'PD-NC', color: '#fbbf24' },
  'PD-MCI': { label: 'Parkinson\u2019s, mild cognitive impairment', short: 'PD-MCI', color: '#fb923c' },
};

const GROUPS: { key: GroupKey; label: string; subject: string; seed: number; blurb: string }[] = [
  {
    key: 'HC',
    label: 'Healthy control',
    subject: 'sub-HC01 (illustrative)',
    seed: 11,
    blurb:
      'Regions inside the same network rise and fall together, and different networks mostly keep to themselves. This is the reference pattern — the one the other two are compared against.',
  },
  {
    key: 'PD-NC',
    label: 'Parkinson\u2019s — normal cognition',
    subject: 'sub-PDNC01 (illustrative)',
    seed: 47,
    blurb:
      'The same networks are still there, but the synchrony inside them is weaker — most noticeably in the motor network. Cognition, on standard tests, still looks normal.',
  },
  {
    key: 'PD-MCI',
    label: 'Parkinson\u2019s — mild cognitive impairment',
    subject: 'sub-PDMCI01 (illustrative)',
    seed: 83,
    blurb:
      'Coupling is more diffuse and less organised. Within-network synchrony drops further, and networks start to blur into each other — the pattern that tracks with early cognitive change.',
  },
];

/* 20 representative parcels: 4 networks x 5 parcels (full analysis uses Schaefer 100) */
const N = 20;
const NETWORKS = ['Visual', 'Motor', 'Default', 'Fronto-parietal'];
const PARCEL_SHORT = ['V', 'M', 'D', 'F'];
function parcelLabel(i: number): string {
  return `${PARCEL_SHORT[Math.floor(i / 5)]}${(i % 5) + 1}`;
}
function parcelFull(i: number): string {
  return `${NETWORKS[Math.floor(i / 5)]} ${(i % 5) + 1}`;
}

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

type Features = { within: number; between: number; sd: number };

function featuresOf(m: number[][]): Features {
  let wSum = 0, wC = 0, bSum = 0, bC = 0, allSum = 0, allC = 0;
  const vals: number[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const same = Math.floor(i / 5) === Math.floor(j / 5);
      if (same) { wSum += m[i][j]; wC++; } else { bSum += m[i][j]; bC++; }
      allSum += m[i][j]; allC++; vals.push(m[i][j]);
    }
  }
  const mean = allSum / Math.max(1, allC);
  const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, vals.length));
  return { within: wSum / Math.max(1, wC), between: bSum / Math.max(1, bC), sd };
}

function topConnections(m: number[][]): { a: number; b: number; r: number }[] {
  const pairs: { a: number; b: number; r: number }[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) pairs.push({ a: i, b: j, r: m[i][j] });
  }
  return pairs.sort((x, y) => Math.abs(y.r) - Math.abs(x.r)).slice(0, 3);
}

/* ---------- CSV parsing (100x100 or 20x20 connectivity matrix) ---------- */
type ParseResult =
  | { ok: true; size: number; matrix: number[][]; display: number[][] }
  | { ok: false; error: string };

function parseConnectivityCSV(text: string): ParseResult {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  if (lines.length === 0) {
    return { ok: false, error: 'That file looks empty. A connectivity matrix needs rows of numbers — try downloading the sample CSV below and opening it to see the format.' };
  }
  let rows = lines.map((l) => l.split(/[,;\t]/).map((c) => c.trim()));
  // Tolerate a header row / header column (non-numeric labels).
  const firstRowNumeric = rows[0].every((c) => c !== '' && Number.isFinite(Number(c)));
  if (!firstRowNumeric) rows = rows.slice(1);
  if (rows.length > 0 && rows.every((r) => r.length > 0 && !Number.isFinite(Number(r[0])))) {
    rows = rows.map((r) => r.slice(1));
  }
  if (rows.length === 0) {
    return { ok: false, error: 'We could not find any numbers in that file. It should be a plain table of numbers — one row per brain region, values separated by commas.' };
  }
  const widths = new Set(rows.map((r) => r.length));
  if (widths.size > 1) {
    return { ok: false, error: `The rows are uneven — some have ${Math.min(...widths)} values and others ${Math.max(...widths)}. A connectivity matrix needs every row the same length.` };
  }
  const n = rows.length;
  const w = rows[0].length;
  if (n !== w) {
    return { ok: false, error: `That table is ${n} rows by ${w} columns, and a connectivity matrix has to be square. We accept 20×20 or 100×100.` };
  }
  if (n !== 20 && n !== 100) {
    return { ok: false, error: `That matrix is ${n}×${n}. Right now we accept 20×20 (the parcels shown here) or 100×100 (the full Schaefer atlas).` };
  }
  const matrix: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      const v = Number(rows[i][j]);
      if (!Number.isFinite(v)) {
        return { ok: false, error: `Row ${i + 1}, column ${j + 1} is “${rows[i][j]}”, which isn\u2019t a number. Every cell needs to be a number between −1 and 1.` };
      }
      if (Math.abs(v) > 1.5) {
        return { ok: false, error: `Row ${i + 1}, column ${j + 1} is ${v}, which is outside the range of a correlation (−1 to 1). Check that the file holds correlations, not raw time-series values.` };
      }
      row.push(v);
    }
    matrix.push(row);
  }
  const display = n === 100
    ? Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => matrix[i * 5][j * 5]))
    : matrix;
  return { ok: true, size: n, matrix, display };
}

type ResultsFile = {
  status: string;
  models: Array<Record<string, unknown>>;
  metrics?: Record<string, unknown> | null;
  note?: string;
  last_updated?: string;
  task?: string;
};

const MODELS = [
  {
    name: 'Graph Neural Network (GCN / GAT)',
    fit: 'Treats each brain as a graph — parcels are nodes, connection strengths are edges. Built for exactly this kind of data: the network structure is the signal, not the noise.',
    note: 'Primary candidate. Its attention weights can point at the connections that matter most.',
  },
  {
    name: 'MLP on Connectivity',
    fit: 'A feed-forward network on the flattened upper triangle of the 100×100 matrix — 4,950 numbers per person. Simple, fast on CPU, and strong when it is regularised well.',
    note: 'The baseline deep model. Dropout and weight decay are doing real work at n = 55.',
  },
  {
    name: '1D-CNN',
    fit: 'Convolves over each parcel\u2019s connectivity profile, learning local patterns across neighbouring regions before pooling everything down to a class.',
    note: 'Mid-complexity. Tests whether local connectivity motifs carry the signal.',
  },
  {
    name: 'SVM (RBF)',
    fit: 'A classic that routinely beats deep nets on small brain-imaging samples. The RBF kernel handles non-linear boundaries in connectivity space without needing much data.',
    note: 'The honest baseline — if a deep model can\u2019t beat this, it hasn\u2019t earned its complexity.',
  },
  {
    name: 'Logistic Regression',
    fit: 'Linear, interpretable, and hard to overfit. Its coefficients map straight back to individual connections, so you can read what it learned.',
    note: 'The interpretability reference — and the sanity check that the classes separate at all.',
  },
  {
    name: 'Random Forest / XGBoost',
    fit: 'Ensembles of decision trees that rank connections by importance and cope well with features that are correlated with each other — which brain connections always are.',
    note: 'A second opinion on feature importance, to cross-check against the GNN\u2019s attention.',
  },
  {
    name: '3D CNN (comparison only)',
    fit: 'Learns straight from raw T1 and fMRI volumes, with no atlas and no connectivity step. The most data-hungry option on this page by a long way.',
    note: 'Included so the claim gets tested, not assumed — we expect it to overfit at n = 55.',
  },
];

const RUN_STEPS = [
  { label: 'Checking your matrix…', pct: 22 },
  { label: 'Reading the connections…', pct: 52 },
  { label: 'Comparing against the three patterns…', pct: 82 },
  { label: 'Writing up the result…', pct: 100 },
];

/* ---------- hero motif: a resting brain as a network ---------- */
const MOTIF_NODES: { x: number; y: number; hot?: boolean }[] = [
  { x: 96, y: 92 }, { x: 178, y: 58, hot: true }, { x: 268, y: 74 }, { x: 348, y: 52 },
  { x: 424, y: 96, hot: true }, { x: 62, y: 178 }, { x: 150, y: 150 }, { x: 238, y: 168, hot: true },
  { x: 322, y: 146 }, { x: 412, y: 182 }, { x: 92, y: 268 }, { x: 176, y: 244 },
  { x: 262, y: 268, hot: true }, { x: 348, y: 250 }, { x: 428, y: 282 }, { x: 138, y: 348 },
  { x: 232, y: 330 }, { x: 326, y: 344, hot: true }, { x: 402, y: 356 },
];
const MOTIF_EDGES: [number, number, boolean][] = [
  [0, 1, false], [1, 2, true], [2, 3, false], [3, 4, true], [0, 5, false], [1, 6, true],
  [2, 7, true], [3, 8, false], [4, 9, true], [5, 6, false], [6, 7, true], [7, 8, true],
  [8, 9, false], [5, 10, false], [6, 11, true], [7, 12, true], [8, 13, false], [9, 14, false],
  [10, 11, false], [11, 12, true], [12, 13, true], [13, 14, false], [11, 15, false],
  [12, 16, true], [13, 17, true], [14, 18, false], [15, 16, false], [16, 17, true], [17, 18, false],
];

function BrainMotif() {
  return (
    <svg className="motif" viewBox="0 0 520 420" role="img" aria-label="Illustration of brain regions connected as a network">
      <defs>
        <radialGradient id="motifGlow" cx="50%" cy="46%" r="55%">
          <stop offset="0%" stopColor="#2dd4bf" stopOpacity="0.16" />
          <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="520" height="420" fill="url(#motifGlow)" />
      {MOTIF_EDGES.map(([a, b, hot], i) => (
        <line
          key={i}
          x1={MOTIF_NODES[a].x} y1={MOTIF_NODES[a].y}
          x2={MOTIF_NODES[b].x} y2={MOTIF_NODES[b].y}
          stroke={hot ? '#2dd4bf' : '#7dd3fc'}
          strokeOpacity={hot ? 0.75 : 0.28}
          strokeWidth={hot ? 1.8 : 1}
        />
      ))}
      {MOTIF_NODES.map((n, i) => (
        <g key={i}>
          {n.hot && <circle cx={n.x} cy={n.y} r="11" fill="#2dd4bf" opacity="0.14" />}
          <circle cx={n.x} cy={n.y} r={n.hot ? 6 : 4} fill={n.hot ? '#2dd4bf' : '#0f2333'} stroke={n.hot ? '#5eead4' : '#5b7d8c'} strokeWidth="1.5" />
        </g>
      ))}
    </svg>
  );
}

function LogoMark() {
  return (
    <svg className="brand-mark" width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
      <line x1="6" y1="7" x2="20" y2="7" stroke="#2dd4bf" strokeWidth="1.4" opacity="0.7" />
      <line x1="6" y1="7" x2="13" y2="19" stroke="#2dd4bf" strokeWidth="1.4" opacity="0.7" />
      <line x1="20" y1="7" x2="13" y2="19" stroke="#2dd4bf" strokeWidth="1.4" opacity="0.7" />
      <circle cx="6" cy="7" r="3.4" fill="#2dd4bf" />
      <circle cx="20" cy="7" r="3.4" fill="#2dd4bf" />
      <circle cx="13" cy="19" r="3.4" fill="#0f2333" stroke="#2dd4bf" strokeWidth="1.6" />
    </svg>
  );
}

type Prediction = {
  group: GroupKey;
  matchPct: number;
  features: Features;
  drivers: { a: number; b: number; r: number }[];
  inputLabel: string;
};

export default function Home() {
  const [group, setGroup] = useState<GroupKey>('HC');
  const [results, setResults] = useState<ResultsFile | null>(null);

  /* --- Use-it flow state --- */
  const [inputKind, setInputKind] = useState<'sample' | 'upload'>('sample');
  const [uploaded, setUploaded] = useState<{ label: string; size: number; display: number[][]; full: number[][] } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [runState, setRunState] = useState<'idle' | 'running' | 'done'>('idle');
  const [runStep, setRunStep] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const timersRef = useRef<number[]>([]);

  const active = GROUPS.find((g) => g.key === group)!;
  const sampleMatrix = useMemo(() => buildMatrix(group, active.seed), [group, active.seed]);
  const matrix = inputKind === 'upload' && uploaded ? uploaded.display : sampleMatrix;
  const inputLabel = inputKind === 'upload' && uploaded ? uploaded.label : `${active.label} · ${active.subject}`;
  const within = useMemo(() => featuresOf(matrix).within, [matrix]);
  const between = useMemo(() => featuresOf(matrix).between, [matrix]);

  const centroids = useMemo(() => {
    const out = {} as Record<GroupKey, Features>;
    for (const g of GROUPS) out[g.key] = featuresOf(buildMatrix(g.key, g.seed));
    return out;
  }, []);

  const classify = (m: number[][]): Prediction => {
    const f = featuresOf(m);
    const keys: (keyof Features)[] = ['within', 'between', 'sd'];
    const scales: Record<keyof Features, number> = { within: 0.12, between: 0.06, sd: 0.04 };
    const dists = (Object.keys(centroids) as GroupKey[]).map((k) => {
      const c = centroids[k];
      const d = Math.sqrt(keys.reduce((s, key) => s + ((f[key] - c[key]) / scales[key]) ** 2, 0));
      return { key: k, d };
    });
    dists.sort((a, b) => a.d - b.d);
    const scores = dists.map(({ d }) => 1 / (1 + d));
    const total = scores.reduce((s, v) => s + v, 0);
    const best = dists[0];
    return {
      group: best.key,
      matchPct: Math.round((scores[0] / total) * 100),
      features: f,
      drivers: topConnections(m),
      inputLabel,
    };
  };

  useEffect(() => {
    fetch('results.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setResults(d))
      .catch(() => undefined);
  }, []);

  useEffect(() => () => { timersRef.current.forEach((t) => window.clearTimeout(t)); }, []);

  const resetRun = () => {
    timersRef.current.forEach((t) => window.clearTimeout(t));
    timersRef.current = [];
    setRunState('idle');
    setRunStep(0);
    setPrediction(null);
  };

  const handleFile = (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadError('That file isn\u2019t a CSV. Save your matrix as a .csv file (most spreadsheet apps do this with File → Download → CSV) and try again.');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseConnectivityCSV(String(reader.result ?? ''));
      if (parsed.ok === false) { setUploadError(parsed.error); return; }
      setUploadError(null);
      setUploaded({ label: `${file.name} · ${parsed.size}×${parsed.size}`, size: parsed.size, display: parsed.display, full: parsed.matrix });
      setInputKind('upload');
      resetRun();
    };
    reader.onerror = () => setUploadError('We couldn\u2019t read that file — it may be open in another program. Close it and try uploading again.');
    reader.readAsText(file);
  };

  const downloadSampleCSV = () => {
    const m = buildMatrix('HC', 11);
    const csv = m.map((row) => row.map((v) => v.toFixed(4)).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample-connectivity-20x20.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const runAnalysis = () => {
    resetRun();
    setRunState('running');
    RUN_STEPS.forEach((_, i) => {
      timersRef.current.push(window.setTimeout(() => setRunStep(i), i * 650));
    });
    timersRef.current.push(window.setTimeout(() => {
      setPrediction(classify(matrix));
      setRunState('done');
    }, RUN_STEPS.length * 650));
  };

  const trainingDone = results !== null && results.status !== 'training_in_progress' && Array.isArray(results.models) && results.models.length > 0;
  const flowStep = runState === 'done' ? 3 : runState === 'running' ? 2 : 1;
  const cell = 22;
  const size = N * cell;

  return (
    <>
      <header className="nav">
        <div className="nav-inner">
          <a className="brand" href="#top"><LogoMark />Parkinson&rsquo;s <em>Early Detection</em></a>
          <nav className="nav-links" aria-label="Sections">
            <a href="#dataset">Data</a>
            <a href="#how">Method</a>
            <a href="#models">Models</a>
            <a href="#use-it">Use it</a>
            <a href="#results">Results</a>
            <a className="nav-cta" href="#use-it">Try it yourself</a>
          </nav>
        </div>
      </header>

      <div className="status" role="status">
        <div className="status-inner">
          <span className="dot" aria-hidden="true" />
          <span>
            <strong>{trainingDone ? 'Results published.' : 'Training in progress.'}</strong>{' '}
            {trainingDone
              ? 'The benchmark finished — scores are in the Results section below.'
              : 'Models are being benchmarked on ds005892. Nothing on this page is a claimed result until training finishes.'}
          </span>
          <span className="spacer" />
          <a href="#results">See results ↓</a>
        </div>
      </div>

      <main id="top">
        {/* HERO */}
        <section className="hero">
          <div className="wrap hero-grid">
            <div>
              <p className="eyebrow">A small research project · Open data · Open code</p>
              <h1>Can a resting brain show Parkinson&rsquo;s <em>before</em> the tremor does?</h1>
              <p className="lead">
                Parkinson&rsquo;s is usually diagnosed once the symptoms are hard to miss. This project
                asks whether the quiet patterns in a resting brain scan — how regions talk to each
                other while you&rsquo;re doing nothing at all — carry an earlier signal. We&rsquo;re
                testing that idea on 55 people, in the open, with the workings shown.
              </p>
              <div className="badge-row">
                <span className="badge"><b>55</b> people</span>
                <span className="badge">3 groups: <b>HC / PD-NC / PD-MCI</b></span>
                <span className="badge">Schaefer <b>100-parcel</b> atlas</span>
                <span className="badge">Up to <b>1000 epochs</b> · early stopping</span>
              </div>
              <div className="cta-row">
                <a className="btn" href="#use-it">Try it yourself →</a>
                <a className="btn-ghost" href="#how">Read the method</a>
              </div>
              <p className="hero-note">
                Built by <strong>Abhishek Singh</strong>. A research demo, not a medical device —
                it doesn&rsquo;t diagnose anyone. What it can and can&rsquo;t do is written down plainly
                at the bottom of this page.
              </p>
            </div>
            <div className="motif-wrap">
              <BrainMotif />
              <p className="motif-caption">100 parcels · 4,950 connections · one pattern per person</p>
            </div>
          </div>
        </section>

        {/* DATASET */}
        <section id="dataset">
          <div className="wrap">
            <p className="eyebrow">The data</p>
            <h2>The data, honestly</h2>
            <p className="sub">
              Resting-state fMRI and structural MRI from Kemp and colleagues (2025), shared openly on
              OpenNeuro under CC0. Fifty-five people is small for deep learning — that isn&rsquo;t a
              footnote here, it&rsquo;s the fact that shapes every choice below. We work with the
              dataset we have, and we say so.
            </p>
            <div className="grid grid-4">
              <div className="card stat-card"><div className="stat-num">55</div><div className="stat-label">People, with complete data</div><div className="stat-sub">Three groups, one study</div></div>
              <div className="card stat-card"><div className="stat-num">22</div><div className="stat-label">Healthy controls</div><div className="stat-sub">HC — the reference group</div></div>
              <div className="card stat-card"><div className="stat-num">18</div><div className="stat-label">Parkinson&rsquo;s, normal cognition</div><div className="stat-sub">PD-NC</div></div>
              <div className="card stat-card"><div className="stat-num">15</div><div className="stat-label">Parkinson&rsquo;s, mild impairment</div><div className="stat-sub">PD-MCI</div></div>
            </div>
            <div className="grid grid-2" style={{ marginTop: 18 }}>
              <div className="card">
                <h3>What was recorded</h3>
                <p>Resting-state fMRI (the BOLD signal, minute by minute, while the person lies still) plus structural T1-weighted MRI, all in BIDS format. Preprocessing uses fMRIPrep derivatives where they exist, otherwise a Nilearn pipeline.</p>
              </div>
              <div className="card">
                <h3>Where it comes from</h3>
                <p>Kemp, A. S., Pyles, J. A., &amp; Bishop, C. (2025). Resting state MRI data from healthy control, Parkinson&rsquo;s disease with normal cognition, and Parkinson&rsquo;s disease with mild cognitive impairment cohorts. OpenNeuro. DOI: 10.18112/openneuro.ds005892.</p>
                <p style={{ marginTop: 12 }}><a href={DATASET_URL} target="_blank" rel="noreferrer">openneuro.org/datasets/ds005892 ↗</a></p>
              </div>
            </div>
          </div>
        </section>

        {/* HOW IT WORKS */}
        <section id="how">
          <div className="wrap">
            <p className="eyebrow">Method</p>
            <h2>From a scan to an answer, in four steps</h2>
            <p className="sub">The key move is step three. We never hand raw 4D brain volumes to a classifier — we hand it the conversation between regions. That keeps the problem small enough for 55 people to teach us something.</p>
            <div className="grid grid-2 method-grid">
              <div className="card method-step"><h3>Acquire</h3><p>Download ds005892 v1.0.0 from OpenNeuro. Check the BIDS structure, count the subjects per group, and note what&rsquo;s missing — before a single model trains.</p></div>
              <div className="card method-step"><h3>Preprocess</h3><p>Motion correction, normalisation to MNI space, nuisance regression and band-pass filtering of the BOLD signal, via fMRIPrep or Nilearn.</p></div>
              <div className="card method-step"><h3>Connectivity</h3><p>Divide the brain with the Schaefer 100-region atlas, pull one time series out of each parcel, and correlate them. Each person becomes a 100 × 100 functional-connectivity matrix.</p></div>
              <div className="card method-step"><h3>Classify</h3><p>Benchmark seven model families on identical stratified cross-validation splits, and report accuracy, balanced accuracy, AUC, F1 and confusion matrices — never accuracy on its own.</p></div>
            </div>
          </div>
        </section>

        {/* MODELS */}
        <section id="models">
          <div className="wrap">
            <p className="eyebrow">Models</p>
            <h2>Seven models. One fair test.</h2>
            <p className="sub">No winner is picked in advance. Every model trains on the same splits, under the same early-stopping rule (max 1000 epochs). The winner is whichever generalises best to people it hasn&rsquo;t seen — and if a plain SVM wins, this page will say so.</p>
            <div className="grid grid-2">
              {MODELS.map((m) => (
                <div className="card model-card" key={m.name}>
                  <span className="tag">Benchmarked</span>
                  <div className="model-name">{m.name}</div>
                  <p className="model-fit">{m.fit}</p>
                  <p className="model-note"><b>Why it&rsquo;s here:</b> {m.note}</p>
                </div>
              ))}
              <div className="card model-card" style={{ borderColor: 'var(--teal-border)' }}>
                <span className="tag">Evaluation</span>
                <div className="model-name">How the winner is chosen</div>
                <p className="model-fit">Stratified 5-fold cross-validation (repeated), with leave-one-out as a sanity check. The deciding metrics are balanced accuracy and macro-AUC, because the groups are uneven (22 / 18 / 15).</p>
                <p className="model-note"><b>Training rule:</b> max 1000 epochs, early stopping on validation loss, cosine learning-rate decay — and we keep the best epoch, not the last one.</p>
              </div>
            </div>
          </div>
        </section>

        {/* USE IT */}
        <section id="use-it" className="flow">
          <div className="wrap">
            <p className="eyebrow">Use it</p>
            <h2>Try it on a brain pattern</h2>
            <p className="sub">
              Three steps, about a minute. Pick a sample subject or upload your own connectivity
              matrix, run the analysis, and read the result — including what drove it and what it
              doesn&rsquo;t prove. Nothing you upload leaves your browser.
            </p>

            <div className="stepper" aria-label="Steps">
              {[
                { n: 1, b: 'Choose your input', s: 'A sample subject, or your own matrix as a CSV.' },
                { n: 2, b: 'Run the analysis', s: 'We read the connections and compare the pattern.' },
                { n: 3, b: 'See your result', s: 'A predicted group, the drivers, and the limits.' },
              ].map((s) => (
                <div key={s.n} className={`stepper-step ${flowStep === s.n ? 'active' : flowStep > s.n ? 'done' : ''}`}>
                  <span className="n">{flowStep > s.n ? '✓' : s.n}</span>
                  <div><b>{s.b}</b><span>{s.s}</span></div>
                </div>
              ))}
            </div>

            <div className="flow-grid">
              {/* Step 1 — input */}
              <div className="card panel">
                <div className="panel-head"><h3>1 · Choose your input</h3><span className="kicker">Step 1 of 3</span></div>
                <p>Start with a sample, or bring your own data. A sample is a simplified, illustrative pattern — good for seeing how this works.</p>
                <div className="seg" role="group" aria-label="Sample subject group">
                  {GROUPS.map((g) => (
                    <button
                      key={g.key}
                      className={inputKind === 'sample' && group === g.key ? 'active' : ''}
                      onClick={() => { setInputKind('sample'); setGroup(g.key); resetRun(); }}
                      aria-pressed={inputKind === 'sample' && group === g.key}
                    >
                      {g.label}
                    </button>
                  ))}
                </div>
                {inputKind === 'sample' && <p className="sample-blurb"><b>{active.label}.</b> {active.blurb}</p>}

                <div className="divider">or upload your own</div>
                <div
                  className={`upload-box ${dragging ? 'dragging' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files?.[0]; if (f) handleFile(f); }}
                >
                  <div className="upload-icon" aria-hidden="true">⇪</div>
                  <p><strong style={{ color: 'var(--ink)' }}>Drop a CSV here, or choose a file.</strong><br />A 20×20 or 100×100 table of correlations, one region per row.</p>
                  <button type="button" className="btn-ghost" onClick={() => fileRef.current?.click()}>Choose a CSV file</button>
                  <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ''; }} />
                </div>
                <p className="upload-hint">
                  Your file is read in your browser and never sent anywhere. Not sure about the format?{' '}
                  <button type="button" className="link-btn" onClick={downloadSampleCSV}>Download a sample matrix (CSV)</button>{' '}
                  and open it to see what we expect.
                </p>
                {uploaded && inputKind === 'upload' && (
                  <div className="file-row">
                    <span>✓ <b>{uploaded.label}</b></span>
                    <button type="button" className="link-btn" onClick={() => { setUploaded(null); setInputKind('sample'); resetRun(); }}>Use a sample instead</button>
                  </div>
                )}
                {uploadError && (
                  <div className="error-box" role="alert"><strong>That file didn&rsquo;t work — here&rsquo;s why.</strong><br />{uploadError}</div>
                )}
              </div>

              {/* Step 2 — preview */}
              <div className="card panel">
                <div className="panel-head"><h3>2 · Look at the pattern</h3><span className="kicker">{inputLabel}</span></div>
                <p>Each square is one pair of brain regions. Brighter teal means the two regions fluctuate together more strongly. Hover a square to read its value.</p>
                <div style={{ marginTop: 18 }}>
                  <svg className="heatmap" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Functional connectivity heatmap for ${inputLabel}`}>
                    {matrix.map((row, i) =>
                      row.map((v, j) => (
                        <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1} fill={colorFor(v)}>
                          <title>{`${parcelFull(i)} ↔ ${parcelFull(j)}: r = ${v.toFixed(2)}`}</title>
                        </rect>
                      ))
                    )}
                  </svg>
                  <div className="legend"><span>−0.2 · weaker coupling</span><span>Networks: {NETWORKS.join(' · ')}</span><span>+1.0 · stronger coupling</span></div>
                </div>
                <div style={{ marginTop: 16 }}>
                  <div className="kv"><span>Mean within-network coupling</span><span>r = {within.toFixed(2)}</span></div>
                  <div className="kv"><span>Mean between-network coupling</span><span>r = {between.toFixed(2)}</span></div>
                  <div className="kv"><span>Parcels shown</span><span>20 of 100 (Schaefer)</span></div>
                </div>
              </div>
            </div>

            {/* Run bar */}
            <div className="run-bar">
              <div className="run-text">
                <b>{runState === 'running' ? RUN_STEPS[runStep].label : runState === 'done' ? 'Analysis complete.' : 'Ready when you are.'}</b>
                <span>
                  {runState === 'running'
                    ? 'Comparing this pattern against the three reference groups…'
                    : runState === 'done'
                      ? 'Your result is below — read the drivers and the limits, not just the label.'
                      : 'This runs on illustrative demo data until the trained model is published here.'}
                </span>
                {runState === 'running' && (
                  <div className="run-progress" aria-hidden="true"><div style={{ width: `${RUN_STEPS[runStep].pct}%` }} /></div>
                )}
              </div>
              <button type="button" className="btn" onClick={runAnalysis} disabled={runState === 'running'}>
                {runState === 'running' ? 'Running…' : runState === 'done' ? '↻ Run it again' : '▶ Run analysis'}
              </button>
            </div>

            {/* Step 3 — result */}
            {runState === 'done' && prediction && (
              <div className="card result-card" aria-live="polite">
                <div className="panel-head"><h3>3 · Your result</h3><span className="kicker">For: {prediction.inputLabel}</span></div>
                <div className="result-head">
                  <div className="result-group">
                    {GROUP_META[prediction.group].label}
                    <small>Closest match among the three reference patterns.</small>
                  </div>
                  <div className="match">
                    <b>{prediction.matchPct}%</b>
                    <span>illustrative match</span>
                    <div className="match-meter" aria-hidden="true"><div style={{ width: `${prediction.matchPct}%` }} /></div>
                  </div>
                </div>
                <div className="result-cols">
                  <div>
                    <h4>What drove it</h4>
                    {prediction.drivers.map((d, i) => (
                      <div className="driver-row" key={i}>
                        <span className="r">r = {d.r.toFixed(2)}</span>
                        <p><b>{parcelFull(d.a)} ↔ {parcelFull(d.b)}</b> — {d.r > 0 ? 'these two regions rise and fall together' : 'these two regions move in opposite directions'}{i === 0 ? ', and it\u2019s the strongest single coupling in this matrix.' : '.'}</p>
                      </div>
                    ))}
                    <p style={{ marginTop: 12, fontSize: '0.9rem', color: 'var(--faint)' }}>
                      The call rests on the whole pattern — within-network strength, between-network
                      strength and variability — not on any one connection above.
                    </p>
                  </div>
                  <div>
                    <h4>The pattern, in numbers</h4>
                    <div className="kv"><span>Within-network coupling</span><span>r = {prediction.features.within.toFixed(2)}</span></div>
                    <div className="kv"><span>Between-network coupling</span><span>r = {prediction.features.between.toFixed(2)}</span></div>
                    <div className="kv"><span>Variability across connections</span><span>SD = {prediction.features.sd.toFixed(2)}</span></div>
                    <div className="kv"><span>Reference · Healthy control</span><span>r = {centroids.HC.within.toFixed(2)} within</span></div>
                    <div className="kv"><span>Reference · PD, normal cognition</span><span>r = {centroids['PD-NC'].within.toFixed(2)} within</span></div>
                    <div className="kv"><span>Reference · PD, mild impairment</span><span>r = {centroids['PD-MCI'].within.toFixed(2)} within</span></div>
                  </div>
                </div>
                <div className="result-meaning">
                  <b>What this means:</b> of the three patterns this project studies, your input looks
                  most like <b>{GROUP_META[prediction.group].label.toLowerCase()}</b>. That is a statement
                  about a pattern in a matrix — it is not a statement about a person, and it is not a
                  diagnosis. A real answer needs a trained model, a larger sample, and a clinician.
                </div>
                <div className="illustrative-note">
                  <span className="mark" aria-hidden="true">⚠</span>
                  <span><b>Illustrative demo — the trained model activates here once training completes.</b><br />
                  This result comes from a simplified nearest-pattern comparison on demo data, not from the
                  trained benchmark. When training finishes, this same screen will run the winning model
                  and say so.</span>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* RESULTS */}
        <section id="results">
          <div className="wrap">
            <p className="eyebrow">Results</p>
            <h2>Results, as they land</h2>
            <p className="sub">This section reads <span className="mono">public/results.json</span> — the same file the training pipeline writes to. Until a training run finishes, it shows the in-progress state. No numbers are invented to fill the space.</p>
            <div className="card">
              {trainingDone ? (
                <>
                  <span className="tag">✓ Results published</span>
                  <h3 style={{ fontSize: '1.35rem' }}>The benchmark finished.</h3>
                  <p>Scores below are cross-validated on held-out participants. The full logs, confusion matrices and model card are in the GitHub repository.</p>
                  <div style={{ marginTop: 20 }}>
                    {results!.models.map((m, i) => {
                      const rec = m as Record<string, unknown>;
                      const name = String(rec.name ?? rec.model ?? `Model ${i + 1}`);
                      const num = (k: string) => (typeof rec[k] === 'number' ? (rec[k] as number).toFixed(2) : '—');
                      return (
                        <div className="kv" key={i}>
                          <span>{name}</span>
                          <span>Acc {num('accuracy')} · Bal. acc {num('balanced_accuracy')} · AUC {num('auc')} · F1 {num('f1')}</span>
                        </div>
                      );
                    })}
                  </div>
                  {results!.last_updated && <p style={{ marginTop: 16, fontSize: '0.88rem', color: 'var(--faint)' }}>Last updated: {results!.last_updated}</p>}
                </>
              ) : (
                <>
                  <span className="tag amber">● Training in progress</span>
                  <h3 style={{ fontSize: '1.35rem' }}>No results yet — training is running.</h3>
                  <p>
                    The benchmark (GCN/GAT, MLP, 1D-CNN, SVM, Logistic Regression, Random Forest /
                    XGBoost, 3D CNN) is training with stratified cross-validation, max 1000 epochs
                    and early stopping. Accuracy, balanced accuracy, AUC, F1 and confusion matrices
                    will appear here once the run completes.
                  </p>
                  <div className="progress" aria-hidden="true"><div /></div>
                  <p style={{ marginTop: 16, fontSize: '0.88rem', color: 'var(--faint)' }}>
                    Status: <span className="mono">{results?.status ?? 'training_in_progress'}</span>
                    {results?.last_updated ? <> · Last updated: {results.last_updated}</> : null}
                    {results?.note ? <><br />{results.note}</> : null}
                  </p>
                </>
              )}
            </div>
            <div className="grid grid-3" style={{ marginTop: 18 }}>
              <div className="card"><div className="stat-label">Accuracy</div><p>{trainingDone ? 'Published above, per model.' : 'Pending — reported per model after cross-validation.'}</p></div>
              <div className="card"><div className="stat-label">Balanced accuracy / AUC</div><p>{trainingDone ? 'Published above, per model.' : 'Pending — the deciding metrics, given the 22 / 18 / 15 split.'}</p></div>
              <div className="card"><div className="stat-label">Confusion matrix</div><p>{trainingDone ? 'In the GitHub repository, per model.' : 'Pending — HC vs PD-NC vs PD-MCI, per fold and averaged.'}</p></div>
            </div>
          </div>
        </section>

        {/* ETHICS */}
        <section id="ethics" style={{ paddingTop: 0 }}>
          <div className="wrap">
            <div className="card ethics-card">
              <p className="eyebrow">Honest limits</p>
              <h2>What this can&rsquo;t do</h2>
              <p style={{ marginTop: 12 }}>
                <strong>Small sample.</strong> Fifty-five people is small for deep learning. Whatever
                this project finds is exploratory, and any real claim needs validation on a larger,
                independent group of people.
              </p>
              <p style={{ marginTop: 12 }}>
                <strong>Not a medical diagnosis.</strong> This is a research demo. It does not diagnose,
                screen or triage any person, and it is never a substitute for assessment by a
                qualified clinician. If you&rsquo;re worried about symptoms, talk to a doctor — not
                a website.
              </p>
              <p style={{ marginTop: 12 }}>
                <strong>Data &amp; consent.</strong> All data are de-identified and shared openly by the
                original authors on OpenNeuro under CC0. No new patient data are collected here, and
                nothing you upload in the demo leaves your browser.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div className="wrap">
          <div>
            <strong>Parkinson&rsquo;s Early Detection</strong>
            <p className="foot-note">
              Built by Abhishek Singh · Dataset: OpenNeuro ds005892 (Kemp et al., 2025) ·
              Code, logs and results are public, so anyone can check the work.
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <a href={DATASET_URL} target="_blank" rel="noreferrer">OpenNeuro ds005892 ↗</a>
            <br />
            <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub repository ↗</a>
            <br />
            <a href={GITHUB_URL} target="_blank" rel="noreferrer">@steeellord ↗</a>
          </div>
        </div>
      </footer>
    </>
  );
}
