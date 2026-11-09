/* Shared data + helpers for the Parkinson's early detection site.
   All patterns here are DETERMINISTIC and ILLUSTRATIVE — the real model
   results come from training and land in public/results.json. */

export const DATASET_URL = 'https://openneuro.org/datasets/ds005892/versions/1.0.0';
export const GITHUB_URL = 'https://github.com/steeellord';
export const REPO_URL = 'https://github.com/steeellord/parkinsons-early-detection-ds005892';

/* ---------- seeded RNG (deterministic, illustrative patterns only) ---------- */
export function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type GroupKey = 'HC' | 'PD-NC' | 'PD-MCI';

export const GROUP_META: Record<GroupKey, { label: string; short: string; color: string; colorSoft: string }> = {
  HC: { label: 'Healthy control', short: 'HC', color: '#2dd4bf', colorSoft: 'rgba(45,212,191,0.16)' },
  'PD-NC': { label: 'Parkinson’s, normal cognition', short: 'PD-NC', color: '#fbbf24', colorSoft: 'rgba(251,191,36,0.16)' },
  'PD-MCI': { label: 'Parkinson’s, mild cognitive impairment', short: 'PD-MCI', color: '#fb923c', colorSoft: 'rgba(251,146,60,0.16)' },
};

export const GROUPS: { key: GroupKey; label: string; subject: string; seed: number; blurb: string }[] = [
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
    label: 'Parkinson’s — normal cognition',
    subject: 'sub-PDNC01 (illustrative)',
    seed: 47,
    blurb:
      'The same networks are still there, but the synchrony inside them is weaker — most noticeably in the motor network. Cognition, on standard tests, still looks normal.',
  },
  {
    key: 'PD-MCI',
    label: 'Parkinson’s — mild cognitive impairment',
    subject: 'sub-PDMCI01 (illustrative)',
    seed: 83,
    blurb:
      'Coupling is more diffuse and less organised. Within-network synchrony drops further, and networks start to blur into each other — the pattern that tracks with early cognitive change.',
  },
];

/* 20 representative parcels: 4 networks x 5 parcels (full analysis uses Schaefer 100) */
export const N = 20;
export const NETWORKS = ['Visual', 'Motor', 'Default', 'Fronto-parietal'];
export const PARCEL_SHORT = ['V', 'M', 'D', 'F'];
export const NETWORK_COLORS = ['#7dd3fc', '#fbbf24', '#2dd4bf', '#c084fc'];

export function parcelLabel(i: number): string {
  return `${PARCEL_SHORT[Math.floor(i / 5)]}${(i % 5) + 1}`;
}
export function parcelFull(i: number): string {
  return `${NETWORKS[Math.floor(i / 5)]} ${(i % 5) + 1}`;
}
export function networkOf(i: number): number {
  return Math.floor(i / 5);
}

export function buildMatrix(group: GroupKey, seed: number): number[][] {
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

export function colorFor(v: number): string {
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

export type Features = { within: number; between: number; sd: number };

export function featuresOf(m: number[][]): Features {
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

export function topConnections(m: number[][]): { a: number; b: number; r: number }[] {
  const pairs: { a: number; b: number; r: number }[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) pairs.push({ a: i, b: j, r: m[i][j] });
  }
  return pairs.sort((x, y) => Math.abs(y.r) - Math.abs(x.r)).slice(0, 3);
}

export function topConnectionsForNode(m: number[][], node: number, count = 5): { other: number; r: number }[] {
  const pairs: { other: number; r: number }[] = [];
  for (let j = 0; j < N; j++) {
    if (j !== node) pairs.push({ other: j, r: m[node][j] });
  }
  return pairs.sort((x, y) => Math.abs(y.r) - Math.abs(x.r)).slice(0, count);
}

/* ---------- CSV parsing (100x100 or 20x20 connectivity matrix) ---------- */
export type ParseResult =
  | { ok: true; size: number; matrix: number[][]; display: number[][] }
  | { ok: false; error: string };

export function parseConnectivityCSV(text: string): ParseResult {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  if (lines.length === 0) {
    return { ok: false, error: 'That file looks empty. A connectivity matrix needs rows of numbers — try downloading the sample CSV below and opening it to see the format.' };
  }
  let rows = lines.map((l) => l.split(/[,;\t]/).map((c) => c.trim()));
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
        return { ok: false, error: `Row ${i + 1}, column ${j + 1} is “${rows[i][j]}”, which isn’t a number. Every cell needs to be a number between −1 and 1.` };
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

export type ResultsFile = {
  status: string;
  models: Array<Record<string, unknown>>;
  metrics?: Record<string, unknown> | null;
  note?: string;
  last_updated?: string;
  task?: string;
};

/* ---------- model catalogue (no scores — those come from training) ---------- */
export type ModelInfo = {
  name: string;
  short: string;
  fit: string;
  why: string;
  weakness: string;
  load: 'Light' | 'Moderate' | 'Heavy';
};

export const MODELS: ModelInfo[] = [
  {
    name: 'Graph Neural Network (GCN / GAT)',
    short: 'GCN / GAT',
    fit: 'Treats each brain as a graph — parcels are nodes, connection strengths are edges. Built for exactly this kind of data: the network structure is the signal, not the noise.',
    why: 'Primary candidate. Its attention weights can point at the connections that matter most.',
    weakness: 'The most parameters to tune on the smallest sample. Needs careful regularisation at n = 55, or it will memorise the training set.',
    load: 'Moderate',
  },
  {
    name: 'MLP on Connectivity',
    short: 'MLP',
    fit: 'A feed-forward network on the flattened upper triangle of the 100×100 matrix — 4,950 numbers per person. Simple, fast on CPU, and strong when it is regularised well.',
    why: 'The baseline deep model. Dropout and weight decay are doing real work at n = 55.',
    weakness: 'Throws away the graph structure — it treats connection #1 and connection #4,950 as unrelated inputs.',
    load: 'Light',
  },
  {
    name: '1D-CNN',
    short: '1D-CNN',
    fit: 'Convolves over each parcel’s connectivity profile, learning local patterns across neighbouring regions before pooling everything down to a class.',
    why: 'Mid-complexity. Tests whether local connectivity motifs carry the signal.',
    weakness: '“Neighbouring” in a flattened profile is an arbitrary order — the convolution may learn ordering artefacts.',
    load: 'Light',
  },
  {
    name: 'SVM (RBF)',
    short: 'SVM',
    fit: 'A classic that routinely beats deep nets on small brain-imaging samples. The RBF kernel handles non-linear boundaries in connectivity space without needing much data.',
    why: 'The honest baseline — if a deep model can’t beat this, it hasn’t earned its complexity.',
    weakness: 'Its decisions are hard to explain connection-by-connection, and it scales poorly if the sample ever grows large.',
    load: 'Light',
  },
  {
    name: 'Logistic Regression',
    short: 'LogReg',
    fit: 'Linear, interpretable, and hard to overfit. Its coefficients map straight back to individual connections, so you can read what it learned.',
    why: 'The interpretability reference — and the sanity check that the classes separate at all.',
    weakness: 'Can only draw straight boundaries. If the real separation is curved, it will miss it.',
    load: 'Light',
  },
  {
    name: 'Random Forest / XGBoost',
    short: 'RF / XGB',
    fit: 'Ensembles of decision trees that rank connections by importance and cope well with features that are correlated with each other — which brain connections always are.',
    why: 'A second opinion on feature importance, to cross-check against the GNN’s attention.',
    weakness: 'Importance scores can be unstable across folds when features are as correlated as brain connections are.',
    load: 'Light',
  },
  {
    name: '3D CNN (comparison only)',
    short: '3D CNN',
    fit: 'Learns straight from raw T1 and fMRI volumes, with no atlas and no connectivity step. The most data-hungry option on this page by a long way.',
    why: 'Included so the claim gets tested, not assumed — we expect it to overfit at n = 55.',
    weakness: 'Millions of parameters, 55 people. The likeliest outcome is a model that memorises the training scans.',
    load: 'Heavy',
  },
];

export const RUN_STEPS = [
  { label: 'Checking your matrix…', pct: 22 },
  { label: 'Reading the connections…', pct: 52 },
  { label: 'Comparing against the three patterns…', pct: 82 },
  { label: 'Writing up the result…', pct: 100 },
];
