'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  DATASET_URL,
  GROUPS,
  GROUP_META,
  GroupKey,
  INFERENCE_URL,
  N,
  NETWORKS,
  REPO_URL,
  RUN_STEPS,
  ResultsFile,
  buildMatrix,
  colorFor,
  featuresOf,
  parseConnectivityCSV,
  parcelFull,
  topConnections,
} from './lib';
import BrainExplorer from './components/BrainExplorer';
import ModelLab from './components/ModelLab';
import {
  AmbientBackground,
  CompareSlider,
  CountUp,
  DistributionChart,
  Magnetic,
  PipelineTimeline,
  Reveal,
  Sparkline,
  TiltCard,
} from './components/Premium';

/* Audit-verified group facts (participants.tsv, 2026-11-09 audit). */
const GROUP_FACTS: Record<GroupKey, { count: number; meanAge: string; range: string; male: number; female: number; spark: number[] }> = {
  HC: { count: 22, meanAge: '70.7', range: '60–85', male: 7, female: 15, spark: [0.62, 0.58, 0.71, 0.66, 0.6, 0.74, 0.69, 0.63, 0.7, 0.65] },
  'PD-NC': { count: 18, meanAge: '65.9', range: '60–78', male: 11, female: 7, spark: [0.48, 0.44, 0.52, 0.46, 0.5, 0.42, 0.49, 0.45, 0.51, 0.47] },
  'PD-MCI': { count: 15, meanAge: '69.9', range: '60–94', male: 11, female: 4, spark: [0.38, 0.42, 0.34, 0.4, 0.36, 0.44, 0.39, 0.35, 0.41, 0.37] },
};

function LogoMark() {
  return (
    <svg className="brand-mark" width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
      <circle className="brand-ring" cx="13" cy="13" r="11" fill="none" stroke="#2dd4bf" strokeWidth="1" opacity="0.5" />
      <line x1="6" y1="7" x2="20" y2="7" stroke="#2dd4bf" strokeWidth="1.4" opacity="0.7" />
      <line x1="6" y1="7" x2="13" y2="19" stroke="#2dd4bf" strokeWidth="1.4" opacity="0.7" />
      <line x1="20" y1="7" x2="13" y2="19" stroke="#2dd4bf" strokeWidth="1.4" opacity="0.7" />
      <circle cx="6" cy="7" r="3.4" fill="#2dd4bf" />
      <circle cx="20" cy="7" r="3.4" fill="#2dd4bf" />
      <circle cx="13" cy="19" r="3.4" fill="#0f2333" stroke="#2dd4bf" strokeWidth="1.6" />
    </svg>
  );
}

/* ---------- Results, in plain words ----------
   Everything here is computed from public/results.json, so when a new
   training run writes new numbers into that file, this section picks them
   up with no code change. No jargon is shown without an explanation. */

type ModelRec = {
  name: string;
  accuracy?: number;
  balanced_accuracy?: number;
  auc?: number;
  f1?: number;
  accuracy_std?: number;
  balanced_accuracy_std?: number;
  auc_std?: number;
  f1_std?: number;
  confusion_matrix?: number[][];
};

function asModelRec(m: Record<string, unknown>): ModelRec {
  const num = (k: string) => (typeof m[k] === 'number' ? (m[k] as number) : undefined);
  const cm = Array.isArray(m.confusion_matrix)
    ? (m.confusion_matrix as unknown[]).map((row) =>
        Array.isArray(row) ? row.map((v) => (typeof v === 'number' ? v : 0)) : [])
    : undefined;
  return {
    name: String(m.name ?? m.model ?? 'Model'),
    accuracy: num('accuracy'),
    balanced_accuracy: num('balanced_accuracy'),
    auc: num('auc'),
    f1: num('f1'),
    accuracy_std: num('accuracy_std'),
    balanced_accuracy_std: num('balanced_accuracy_std'),
    auc_std: num('auc_std'),
    f1_std: num('f1_std'),
    confusion_matrix: cm,
  };
}

const RES_LABELS_3 = ['Healthy control', 'PD, normal cognition', 'PD, mild impairment'];
const RES_SHORT_3 = ['HC', 'PD-NC', 'PD-MCI'];
const RES_LABELS_2 = ['Healthy control', "Parkinson's"];
const RES_SHORT_2 = ['HC', 'PD'];

function outOf10(x: number | null | undefined): string {
  if (x === null || x === undefined) return '—';
  return `${Math.round(x * 10)} out of 10`;
}

function strengthOf(bal: number | null | undefined, chance: number): 'strong' | 'middling' | 'weak' {
  if (bal === null || bal === undefined) return 'weak';
  const margin = bal - chance;
  if (margin >= 0.3) return 'strong';
  if (margin >= 0.12) return 'middling';
  return 'weak';
}

function verdictFor(m: ModelRec, nClasses: number): string {
  const bal = m.balanced_accuracy;
  if (bal === undefined) return 'No score published for this model yet.';
  const chance = Math.round((1 / nClasses) * 10);
  const base = `Gets it right about ${Math.round(bal * 10)} times out of 10.`;
  const s = strengthOf(bal, 1 / nClasses);
  if (s === 'strong') return `${base} That's strong — well above guessing (${chance} out of 10).`;
  if (s === 'middling') return `${base} Clearly better than guessing (${chance} out of 10), but far from certain.`;
  return `${base} Only a little better than guessing (${chance} out of 10).`;
}

function cmSentence(cm: number[][], r: number, c: number, labels: string[]): string {
  const rowTotal = cm[r].reduce((a, b) => a + b, 0);
  const v = cm[r][c];
  if (r === c) return `${v} of ${rowTotal} ${labels[r]} people were called ${labels[r]} correctly.`;
  return `${v} ${labels[r]} people were wrongly called ${labels[c]}.`;
}

function ResultsInteractive({ results }: { results: ResultsFile }) {
  const models = results.models.map(asModelRec);
  const binary = (results.binary_models ?? []).map(asModelRec);
  const hasBinary = binary.length > 0;
  const [view, setView] = useState<'three' | 'binary'>('three');
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  const [compareA, setCompareA] = useState('');
  const [compareB, setCompareB] = useState('');
  const [hoverCell, setHoverCell] = useState<string | null>(null);

  const list = view === 'binary' ? binary : models;
  const labels = view === 'binary' ? RES_LABELS_2 : RES_LABELS_3;
  const shorts = view === 'binary' ? RES_SHORT_2 : RES_SHORT_3;
  const chance = 1 / labels.length;
  const n = results.n_subjects ?? 55;

  const bestOf = (arr: ModelRec[]) =>
    arr.reduce<ModelRec | null>((best, m) => {
      if (m.balanced_accuracy === undefined) return best;
      if (!best || (best.balanced_accuracy ?? -1) < m.balanced_accuracy) return m;
      return best;
    }, null);
  const best3 = bestOf(models);
  const bestBin = bestOf(binary);
  const cmpA = list.find((m) => m.name === compareA) ?? null;
  const cmpB = list.find((m) => m.name === compareB) ?? null;

  const metricRows: { key: keyof ModelRec; label: string; hint: string }[] = [
    { key: 'balanced_accuracy', label: 'Balanced accuracy', hint: 'Right answers, counted fairly across groups' },
    { key: 'accuracy', label: 'Accuracy', hint: 'Right answers overall' },
    { key: 'auc', label: 'AUC', hint: 'How well it ranks Parkinson’s above healthy' },
    { key: 'f1', label: 'F1', hint: 'Balance between catching cases and avoiding false alarms' },
  ];

  return (
    <>
      <Reveal delay={80}>
        <div className="card glass">
          <span className="tag">✓ Results published</span>
          <h3 style={{ fontSize: '1.35rem' }}>What the numbers actually mean</h3>
          <p>
            We tested {models.length} models on {n} people, and every score was measured on
            people the model had not seen during training. The short version:{' '}
            {best3 && best3.balanced_accuracy !== undefined && (
              <>
                the best model at telling the three groups apart is <b>{best3.name}</b> — about{' '}
                <b>{Math.round(best3.balanced_accuracy * 10)} out of 10</b> right, where guessing
                gives 3 out of 10.{' '}
              </>
            )}
            {bestBin && bestBin.balanced_accuracy !== undefined && (
              <>
                Telling healthy brains from Parkinson’s brains is easier:{' '}
                <b>{bestBin.name}</b> gets that right about{' '}
                <b>{Math.round(bestBin.balanced_accuracy * 10)} out of 10</b>.
              </>
            )}{' '}
            The two Parkinson’s groups are the hard part to separate — that is where almost
            every model struggles.
          </p>
          {results.note && (
            <p style={{ fontSize: '0.92rem', color: 'var(--faint)' }}>{results.note}</p>
          )}
          <details className="howto">
            <summary>How to read this section</summary>
            <ul>
              <li><b>“Out of 10”</b> is the plain version of the decimal: 7 out of 10 means the model got 7 of every 10 people right.</li>
              <li><b>Guessing</b> gets 3 out of 10 with three groups, and 5 out of 10 with two. A model has to beat that to mean anything.</li>
              <li><b>The grid</b> in each model’s detail is a confusion matrix: rows are the truth, columns are the model’s call. Squares on the diagonal are correct calls; the rest are mix-ups. Hover any square to read what it means.</li>
              <li><b>“Give or take”</b> is how much the score moved between the five test splits — a big swing means the score is shaky.</li>
              <li><b>Balanced accuracy</b> is the fair score here, because the groups are uneven ({n} people split 22 / 18 / 15). Plain accuracy would flatter a model that just guesses the biggest group.</li>
            </ul>
          </details>
        </div>
      </Reveal>

      <div className="res-toggle" role="group" aria-label="Results view">
        <button type="button" className={view === 'three' ? 'active' : ''} onClick={() => { setView('three'); setOpenIdx(null); }} aria-pressed={view === 'three'}>
          Three groups · HC vs PD-NC vs PD-MCI
        </button>
        {hasBinary && (
          <button type="button" className={view === 'binary' ? 'active' : ''} onClick={() => { setView('binary'); setOpenIdx(null); }} aria-pressed={view === 'binary'}>
            Healthy vs Parkinson’s
          </button>
        )}
      </div>

      <div className="res-list">
        {list.map((m, i) => {
          const s = strengthOf(m.balanced_accuracy, chance);
          const open = openIdx === i;
          const cm = m.confusion_matrix;
          const cmOk = cm && cm.length === labels.length && cm.every((row) => row.length === labels.length);
          const maxCell = cmOk ? Math.max(...cm.flat(), 1) : 1;
          return (
            <div key={m.name} className="card res-row glass">
              <button type="button" className="res-head" onClick={() => setOpenIdx(open ? null : i)} aria-expanded={open}>
                <span className={`res-dot dot-${s}`} aria-hidden="true" />
                <span className="res-titles">
                  <b>{m.name}</b>
                  <span>{verdictFor(m, labels.length)}</span>
                </span>
                <span className="res-score">
                  <b>{outOf10(m.balanced_accuracy)}</b>
                  <span>balanced accuracy · {m.balanced_accuracy !== undefined ? m.balanced_accuracy.toFixed(2) : '—'}</span>
                </span>
                <span className="res-chev" aria-hidden="true">{open ? '−' : '+'}</span>
              </button>
              {open && (
                <div className="res-body">
                  <div className="res-metrics">
                    {metricRows.map(({ key, label, hint }) => {
                      const v = m[key] as number | undefined;
                      const sd = m[`${key}_std` as keyof ModelRec] as number | undefined;
                      return (
                        <div className="kv" key={key}>
                          <span>{label} <em style={{ fontStyle: 'normal', color: 'var(--faint)' }}>— {hint}</em></span>
                          <span>{v !== undefined ? `${v.toFixed(2)} · ${outOf10(v)}` : '—'}{sd !== undefined ? ` · give or take ${sd.toFixed(2)}` : ''}</span>
                        </div>
                      );
                    })}
                  </div>
                  {cmOk && (
                    <div className="res-cm">
                      <h4>Where it got confused</h4>
                      <table className="cm-grid" aria-label={`Confusion matrix for ${m.name}`}>
                        <thead>
                          <tr>
                            <th scope="col"><span className="cm-axis">Truth ↓ · Called →</span></th>
                            {shorts.map((s2) => (<th key={s2} scope="col">{s2}</th>))}
                          </tr>
                        </thead>
                        <tbody>
                          {cm.map((row, r) => (
                            <tr key={r}>
                              <th scope="row">{shorts[r]}</th>
                              {row.map((v, c) => (
                                <td
                                  key={c}
                                  style={{
                                    background: r === c
                                      ? `rgba(45, 212, 191, ${0.15 + 0.65 * (v / maxCell)})`
                                      : `rgba(251, 146, 60, ${0.1 + 0.6 * (v / maxCell)})`,
                                  }}
                                  onMouseEnter={() => setHoverCell(cmSentence(cm, r, c, labels))}
                                  onMouseLeave={() => setHoverCell(null)}
                                  title={cmSentence(cm, r, c, labels)}
                                >
                                  {v}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <p className="cm-explain" aria-live="polite">
                        {hoverCell ?? 'Hover a square to read what it means.'}
                      </p>
                      <div className="class-bars">
                        {cm.map((row, r) => {
                          const total = row.reduce((a, b) => a + b, 0);
                          const right = row[r];
                          const pct = total > 0 ? Math.round((right / total) * 100) : 0;
                          return (
                            <div className="class-bar-row" key={r}>
                              <span>{shorts[r]} · {right} of {total} right</span>
                              <div className="class-bar"><div style={{ width: `${pct}%` }} /></div>
                              <b>{pct}%</b>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <Reveal delay={60}>
        <div className="card glass compare-card">
          <h3 style={{ fontSize: '1.15rem' }}>Compare two models</h3>
          <p>Pick any two from the list above and put them side by side.</p>
          <div className="compare-pickers">
            <label>
              First model
              <select value={compareA} onChange={(e) => setCompareA(e.target.value)}>
                <option value="">Pick a model…</option>
                {list.map((m) => (<option key={m.name} value={m.name}>{m.name}</option>))}
              </select>
            </label>
            <label>
              Second model
              <select value={compareB} onChange={(e) => setCompareB(e.target.value)}>
                <option value="">Pick a model…</option>
                {list.map((m) => (<option key={m.name} value={m.name}>{m.name}</option>))}
              </select>
            </label>
          </div>
          {cmpA && cmpB && cmpA.name !== cmpB.name && (
            <div className="grid grid-2 compare-grid">
              {[cmpA, cmpB].map((m) => (
                <div key={m.name} className="compare-col">
                  <b>{m.name}</b>
                  <p>{verdictFor(m, labels.length)}</p>
                  {metricRows.map(({ key, label }) => {
                    const v = m[key] as number | undefined;
                    return (
                      <div className="kv" key={key}>
                        <span>{label}</span>
                        <span>{v !== undefined ? `${v.toFixed(2)} · ${outOf10(v)}` : '—'}</span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
          {cmpA && cmpB && cmpA.name === cmpB.name && (
            <p style={{ marginTop: 12 }}>Pick two different models to compare them.</p>
          )}
        </div>
      </Reveal>

      <div className="card glass limits-card">
        <b>Honest limits, in plain words.</b>{' '}
        Fifty-five people is a small sample — these scores will move when more people are
        added, and the “give or take” ranges above show how shaky each one is. A result
        here is a research finding, not a medical diagnosis: it never says anything
        about a real person, and it is no substitute for a clinician.
        {results.last_updated && <> Last updated: {results.last_updated}.</>}
      </div>
    </>
  );
}

/* ---------- scan service result (from the inference service) ---------- */
type ScanResult = {
  ok: boolean;
  message?: string;
  binary?: { prediction: string; prediction_long: string; probabilities: Record<string, number>; cv?: { balanced_accuracy?: { mean?: number }; auc?: { mean?: number } } };
  three_class?: { prediction: string; prediction_long: string; probabilities: Record<string, number>; cv?: { balanced_accuracy?: { mean?: number }; auc?: { mean?: number } } };
  top_connections?: { a: string; b: string; r: number }[];
  heatmap20?: number[][];
  model_version?: number;
  model_label?: string;
  trained_on?: string;
  n_vol?: number;
  mean_abs_r?: number;
  preprocessing_seconds?: number;
};

const PHOTO_EXT = /\.(png|jpe?g|webp|gif|bmp|tiff?)$/i;
const NIFTI_EXT = /(\.nii\.gz|\.nii)$/i;
const PHOTO_EXPLAIN =
  'That file is a photo, not a brain scan. A photo has no time series in it — just pixels — so there is nothing for the model to read. What the service needs is the actual scan file: the resting-state fMRI as a .nii.gz file, plus the matching structural T1 scan as a .nii.gz file. Both come from the scanner or from the OpenNeuro dataset page.';

type Prediction = {
  group: GroupKey;
  matchPct: number;
  features: ReturnType<typeof featuresOf>;
  drivers: { a: number; b: number; r: number }[];
  inputLabel: string;
};

export default function Home() {
  const [group, setGroup] = useState<GroupKey>('HC');
  const [results, setResults] = useState<ResultsFile | null>(null);

  /* --- Use-it flow state --- */
  const [inputKind, setInputKind] = useState<'scan' | 'sample' | 'upload'>('scan');
  const [uploaded, setUploaded] = useState<{ label: string; size: number; display: number[][]; full: number[][] } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [runState, setRunState] = useState<'idle' | 'running' | 'done'>('idle');
  const [runStep, setRunStep] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const timersRef = useRef<number[]>([]);

  /* --- scan upload state (real model, via the inference service) --- */
  const [scanBold, setScanBold] = useState<File | null>(null);
  const [scanT1, setScanT1] = useState<File | null>(null);
  const [scanStage, setScanStage] = useState<'idle' | 'working' | 'done' | 'error'>('idle');
  const [scanStatus, setScanStatus] = useState('');
  const [scanError, setScanError] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [scanMatrix, setScanMatrix] = useState<number[][] | null>(null);
  const boldRef = useRef<HTMLInputElement>(null);
  const t1Ref = useRef<HTMLInputElement>(null);

  const active = GROUPS.find((g) => g.key === group)!;
  const sampleMatrix = useMemo(() => buildMatrix(group, active.seed), [group, active.seed]);
  const matrix: number[][] | null =
    inputKind === 'scan' ? scanMatrix : inputKind === 'upload' && uploaded ? uploaded.display : sampleMatrix;
  const inputLabel =
    inputKind === 'scan'
      ? scanResult ? `Your scan · ${scanBold?.name ?? ''}` : 'Your scan — not analysed yet'
      : inputKind === 'upload' && uploaded ? uploaded.label : `${active.label} · ${active.subject}`;
  const within = useMemo(() => (matrix ? featuresOf(matrix).within : 0), [matrix]);
  const between = useMemo(() => (matrix ? featuresOf(matrix).between : 0), [matrix]);

  const centroids = useMemo(() => {
    const out = {} as Record<GroupKey, ReturnType<typeof featuresOf>>;
    for (const g of GROUPS) out[g.key] = featuresOf(buildMatrix(g.key, g.seed));
    return out;
  }, []);

  const classify = (m: number[][]): Prediction => {
    const f = featuresOf(m);
    const keys: (keyof ReturnType<typeof featuresOf>)[] = ['within', 'between', 'sd'];
    const scales: Record<keyof ReturnType<typeof featuresOf>, number> = { within: 0.12, between: 0.06, sd: 0.04 };
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
      setUploadError('That file isn’t a CSV. Save your matrix as a .csv file (most spreadsheet apps do this with File → Download → CSV) and try again.');
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
    reader.onerror = () => setUploadError('We couldn’t read that file — it may be open in another program. Close it and try uploading again.');
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
    if (!matrix) return;
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

  const checkScanFile = (f: File): 'photo' | 'wrong' | 'big' | null => {
    if (PHOTO_EXT.test(f.name)) return 'photo';
    if (!NIFTI_EXT.test(f.name)) return 'wrong';
    if (f.size > 200 * 1024 * 1024) return 'big';
    return null;
  };

  const pickScanFile = (which: 'bold' | 't1', f: File) => {
    const problem = checkScanFile(f);
    if (problem === 'photo') { setScanError(PHOTO_EXPLAIN); setScanStage('error'); return; }
    if (problem === 'wrong') {
      setScanError('That file is not a scan file. The service needs .nii.gz files — the resting-state fMRI and its matching T1. A spreadsheet, a photo or a screenshot will not work.');
      setScanStage('error');
      return;
    }
    if (problem === 'big') {
      setScanError('That file is over 200 MB — too big to upload. A resting-state scan is usually around 30 MB. Check you picked the right file.');
      setScanStage('error');
      return;
    }
    setScanError(null);
    setScanStage('idle');
    if (which === 'bold') setScanBold(f); else setScanT1(f);
  };

  const runScan = async () => {
    setScanError(null);
    setScanResult(null);
    if (!scanBold || !scanT1) {
      setScanError('Upload both files first: the resting-state fMRI scan and the matching T1 scan.');
      setScanStage('error');
      return;
    }
    if (!INFERENCE_URL) {
      setScanError('The scan service is not connected yet — it is still being wired up. Use a sample subject or a CSV matrix below for now.');
      setScanStage('error');
      return;
    }
    setScanStage('working');
    setScanStatus('Connecting to the scan service…');
    try {
      const { Client } = await import('@gradio/client');
      const client = await Client.connect(INFERENCE_URL);
      setScanStatus('Uploading your scans…');
      const job = client.submit('/analyse', [scanBold, scanT1]);
      for await (const message of job) {
        if (message.type === 'status') {
          if (message.stage === 'pending') setScanStatus('In the queue — the free service handles one scan at a time.');
          else if (message.stage === 'generating') setScanStatus('Processing your scan — aligning it and reading the connections. This takes a few minutes, please wait.');
          else if (message.stage === 'error') throw new Error('service error');
        } else if (message.type === 'data') {
          const data = message.data as unknown[];
          const jsonText = typeof data[2] === 'string' ? data[2] : '';
          const parsed = JSON.parse(jsonText) as ScanResult;
          if (!parsed.ok) {
            setScanError(parsed.message ?? 'The service could not process these scans.');
            setScanStage('error');
            return;
          }
          setScanResult(parsed);
          if (parsed.heatmap20) setScanMatrix(parsed.heatmap20);
          setScanStage('done');
          return;
        }
      }
      setScanError('The service finished without a result. Try again.');
      setScanStage('error');
    } catch (e) {
      setScanError('Could not reach the scan service. The free service sleeps after two quiet days and takes a minute or two to wake up — wait a moment and try again. If it keeps failing, use a sample subject or a CSV matrix below.');
      setScanStage('error');
    }
  };

  const trainingDone = results !== null && results.status !== 'training_in_progress' && Array.isArray(results.models) && results.models.length > 0;
  const flowStep = inputKind === 'scan'
    ? scanStage === 'done' ? 3 : scanStage === 'working' ? 2 : 1
    : runState === 'done' ? 3 : runState === 'running' ? 2 : 1;
  const cell = 22;
  const size = N * cell;

  return (
    <>
      <AmbientBackground />
      <header className="nav">
        <div className="nav-inner">
          <a className="brand" href="#top"><LogoMark />Parkinson’s <em>Early Detection</em></a>
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
        {/* HERO — the brain network IS the hero */}
        <section className="hero">
          <div className="wrap">
            <Reveal>
              <p className="eyebrow">A small research project · Open data · Open code</p>
              <h1>Can a resting brain show Parkinson’s <em>before</em> the tremor does?</h1>
              <p className="lead">
                Parkinson’s is usually diagnosed once the symptoms are hard to miss. This project
                asks whether the quiet patterns in a resting brain scan — how regions talk to each
                other while you’re doing nothing at all — carry an earlier signal. We’re testing
                that idea on 55 people, in the open, with the workings shown.
              </p>
              <div className="badge-row">
                <span className="badge"><b>55</b> people</span>
                <span className="badge">3 groups: <b>HC / PD-NC / PD-MCI</b></span>
                <span className="badge">Schaefer <b>100-parcel</b> atlas</span>
                <span className="badge">Up to <b>1000 epochs</b> · early stopping</span>
              </div>
              <div className="cta-row">
                <Magnetic><a className="btn" href="#use-it">Try it yourself →</a></Magnetic>
                <Magnetic strength={0.22}><a className="btn-ghost" href="#how">Read the method</a></Magnetic>
              </div>
              <p className="hero-note">
                Made by <strong>Rishabh</strong>. A research demo, not a medical device — it
                doesn’t diagnose anyone. What it can and can’t do is written down plainly at the
                bottom of this page.
              </p>
            </Reveal>
            <Reveal delay={120}>
              <div className="hero-explorer glass">
                <BrainExplorer />
              </div>
              <p className="hero-explorer-caption">
                This is the signal itself — a living map of one brain’s resting connections. Move
                your mouse across it, switch groups, click a parcel. The patterns you’re playing
                with are illustrative; the trained model’s version of this map arrives with the results.
              </p>
            </Reveal>
          </div>
        </section>

        {/* DATASET */}
        <section id="dataset">
          <div className="wrap">
            <Reveal>
              <p className="eyebrow">The data</p>
              <h2>The data, honestly</h2>
              <p className="sub">
                Resting-state fMRI and structural MRI from Kemp and colleagues (2025), shared openly
                on OpenNeuro under CC0. Fifty-five people is small for deep learning — that isn’t a
                footnote here, it’s the fact that shapes every choice below. We work with the dataset
                we have, and we say so.
              </p>
            </Reveal>
            <div className="grid grid-4">
              <Reveal delay={0}>
                <TiltCard className="card stat-card">
                  <div className="stat-num"><CountUp end={55} /></div>
                  <div className="stat-label">People, with complete data</div>
                  <div className="stat-sub">Mean age 68.9 · range 60–94 · 29M / 26F</div>
                  <div className="stat-spark"><Sparkline values={[0.55, 0.6, 0.52, 0.58, 0.5, 0.57, 0.53, 0.61, 0.56, 0.59]} color="#7dd3fc" /><span>Illustrative coupling trend</span></div>
                </TiltCard>
              </Reveal>
              {(Object.keys(GROUP_FACTS) as GroupKey[]).map((k, idx) => {
                const f = GROUP_FACTS[k];
                const malePct = Math.round((f.male / f.count) * 100);
                return (
                  <Reveal key={k} delay={idx * 90}>
                    <TiltCard className="card stat-card">
                      <div className="stat-num"><CountUp end={f.count} /></div>
                      <div className="stat-label">{GROUP_META[k].label}</div>
                      <div className="stat-sub">Mean age {f.meanAge} · range {f.range}</div>
                      <div className="sex-split" aria-label={`${f.male} men, ${f.female} women`}>
                        <span className="sex-bar"><span style={{ width: `${malePct}%` }} /></span>
                        <span className="sex-label">{f.male}M · {f.female}F</span>
                      </div>
                      <div className="stat-spark"><Sparkline values={f.spark} color={GROUP_META[k].color} /><span>Within-network coupling (illustrative)</span></div>
                    </TiltCard>
                  </Reveal>
                );
              })}
            </div>
            <Reveal delay={100}>
              <div className="grid grid-2" style={{ marginTop: 18 }}>
                <div className="card glass">
                  <h3>What was recorded</h3>
                  <p>Resting-state fMRI (the BOLD signal, minute by minute, while the person lies still — 200 volumes, TR 2 s, about 6.7 minutes) plus structural T1-weighted MRI at 1 mm, all in BIDS format. Preprocessing uses fMRIPrep derivatives where they exist, otherwise a Nilearn pipeline.</p>
                </div>
                <div className="card glass">
                  <h3>Where it comes from</h3>
                  <p>Kemp, A. S., Pyles, J. A., &amp; Bishop, C. (2025). Resting state MRI data from healthy control, Parkinson’s disease with normal cognition, and Parkinson’s disease with mild cognitive impairment cohorts. OpenNeuro. DOI: 10.18112/openneuro.ds005892.</p>
                  <p style={{ marginTop: 12 }}><a href={DATASET_URL} target="_blank" rel="noreferrer">openneuro.org/datasets/ds005892 ↗</a></p>
                </div>
              </div>
            </Reveal>
            <Reveal delay={140}>
              <div style={{ marginTop: 18 }}>
                <DistributionChart />
              </div>
            </Reveal>
          </div>
        </section>

        {/* HOW IT WORKS — interactive timeline */}
        <section id="how">
          <div className="wrap">
            <Reveal>
              <p className="eyebrow">Method</p>
              <h2>From a scan to an answer, in four steps</h2>
              <p className="sub">The key move is step three. We never hand raw 4D brain volumes to a classifier — we hand it the conversation between regions. That keeps the problem small enough for 55 people to teach us something. Open each step to see it working.</p>
            </Reveal>
            <Reveal delay={80}>
              <PipelineTimeline />
            </Reveal>
          </div>
        </section>

        {/* COMPARE */}
        <section id="compare">
          <div className="wrap">
            <Reveal>
              <p className="eyebrow">See the difference</p>
              <h2>Healthy, or Parkinson’s — spot it yourself</h2>
              <p className="sub">Before any model runs, the difference is visible to a human eye. Two patterns, one slider. If you can see it, a model has something to learn.</p>
            </Reveal>
            <Reveal delay={80}>
              <CompareSlider />
            </Reveal>
          </div>
        </section>

        {/* MODELS */}
        <section id="models">
          <div className="wrap">
            <Reveal>
              <p className="eyebrow">Models</p>
              <h2>Seven models. One fair test.</h2>
              <p className="sub">No winner is picked in advance. Every model trains on the same splits, under the same early-stopping rule (max 1000 epochs). The winner is whichever generalises best to people it hasn’t seen — and if a plain SVM wins, this page will say so.</p>
            </Reveal>
            <Reveal delay={80}>
              <ModelLab />
            </Reveal>
          </div>
        </section>

        {/* USE IT */}
        <section id="use-it" className="flow">
          <div className="wrap">
            <Reveal>
              <p className="eyebrow">Use it</p>
              <h2>Try it on a brain pattern</h2>
              <p className="sub">
                Three steps. Upload your brain scan and the trained model reads it for real — or,
                if you just want to see how it works, try a sample subject or a connectivity
                matrix. Scan files go to our inference service for processing and are deleted
                when the job finishes; matrix uploads never leave your browser.
              </p>
            </Reveal>

            <div className="stepper" aria-label="Steps">
              {[
                { n: 1, b: 'Choose your input', s: 'Your brain scan, a sample subject, or a matrix as a CSV.' },
                { n: 2, b: 'Run the analysis', s: 'The trained model reads your scan and calls it.' },
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
              <div className="card panel glass">
                <div className="panel-head"><h3>1 · Choose your input</h3><span className="kicker">Step 1 of 3</span></div>
                <div className="seg" role="group" aria-label="Input type">
                  <button type="button" className={inputKind === 'scan' ? 'active' : ''} onClick={() => setInputKind('scan')} aria-pressed={inputKind === 'scan'}>
                    Upload a brain scan
                  </button>
                  <button type="button" className={inputKind === 'sample' ? 'active' : ''} onClick={() => { setInputKind('sample'); resetRun(); }} aria-pressed={inputKind === 'sample'}>
                    Sample subject
                  </button>
                  <button type="button" className={inputKind === 'upload' ? 'active' : ''} onClick={() => setInputKind('upload')} aria-pressed={inputKind === 'upload'}>
                    Connectivity matrix (CSV)
                  </button>
                </div>

                {inputKind === 'scan' && (
                  <>
                    <p>Upload your resting-state fMRI scan and its matching structural scan. The trained model processes them for real — this is not the demo. A few minutes per scan, one at a time.</p>
                    <div className="scan-files">
                      <div className="file-row">
                        <span><b>1 · Resting-state fMRI</b><br /><span style={{ color: 'var(--faint)', fontSize: '0.88rem' }}>{scanBold ? scanBold.name : 'Usually sub-XXX_task-rest_bold.nii.gz · about 30 MB'}</span></span>
                        <button type="button" className="btn-ghost" onClick={() => boldRef.current?.click()}>{scanBold ? 'Change' : 'Choose file'}</button>
                        <input ref={boldRef} type="file" accept=".nii,.nii.gz" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) pickScanFile('bold', f); e.target.value = ''; }} />
                      </div>
                      <div className="file-row">
                        <span><b>2 · Structural T1 scan</b><br /><span style={{ color: 'var(--faint)', fontSize: '0.88rem' }}>{scanT1 ? scanT1.name : 'Usually sub-XXX_T1w.nii.gz · about 10 MB'}</span></span>
                        <button type="button" className="btn-ghost" onClick={() => t1Ref.current?.click()}>{scanT1 ? 'Change' : 'Choose file'}</button>
                        <input ref={t1Ref} type="file" accept=".nii,.nii.gz" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) pickScanFile('t1', f); e.target.value = ''; }} />
                      </div>
                    </div>
                    <p className="upload-hint">
                      A photo of a brain will not work — the model needs the scan files, not a picture. Your files are processed on the inference service and deleted when the job finishes.
                      {!INFERENCE_URL && ' The scan service is not connected yet — the team is still wiring it up.'}
                    </p>
                    {scanError && (
                      <div className="error-box" role="alert"><strong>That didn’t work — here’s why.</strong><br />{scanError}</div>
                    )}
                  </>
                )}

                {inputKind === 'sample' && (
                  <>
                    <p>Start with a sample, or bring your own data. A sample is a simplified, illustrative pattern — good for seeing how this works.</p>
                    <div className="seg" role="group" aria-label="Sample subject group">
                      {GROUPS.map((g) => (
                        <button
                          key={g.key}
                          className={group === g.key ? 'active' : ''}
                          onClick={() => { setGroup(g.key); resetRun(); }}
                          aria-pressed={group === g.key}
                        >
                          {g.label}
                        </button>
                      ))}
                    </div>
                    <p className="sample-blurb"><b>{active.label}.</b> {active.blurb}</p>
                  </>
                )}

                {inputKind === 'upload' && (
                  <>
                    <p>Bring your own connectivity matrix — a 20×20 or 100×100 table of correlations, one region per row. This runs the illustrative demo in your browser, not the trained model.</p>
                    <div
                      className={`upload-box ${dragging ? 'dragging' : ''}`}
                      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files?.[0]; if (f) handleFile(f); }}
                    >
                      <div className="upload-icon" aria-hidden="true">⇪</div>
                      <p><strong style={{ color: 'var(--ink)' }}>{dragging ? 'Let go — we’ll take it from here.' : 'Drop a CSV here, or choose a file.'}</strong><br />A 20×20 or 100×100 table of correlations, one region per row.</p>
                      <button type="button" className="btn-ghost" onClick={() => fileRef.current?.click()}>Choose a CSV file</button>
                      <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ''; }} />
                    </div>
                    <p className="upload-hint">
                      Your file is read in your browser and never sent anywhere. Not sure about the format?{' '}
                      <button type="button" className="link-btn" onClick={downloadSampleCSV}>Download a sample matrix (CSV)</button>{' '}
                      and open it to see what we expect.
                    </p>
                    {uploaded && (
                      <div className="file-row">
                        <span>✓ <b>{uploaded.label}</b></span>
                        <button type="button" className="link-btn" onClick={() => { setUploaded(null); setInputKind('sample'); resetRun(); }}>Use a sample instead</button>
                      </div>
                    )}
                    {uploadError && (
                      <div className="error-box" role="alert"><strong>That file didn’t work — here’s why.</strong><br />{uploadError}</div>
                    )}
                  </>
                )}
              </div>

              {/* Step 2 — preview */}
              <div className="card panel glass">
                <div className="panel-head"><h3>2 · Look at the pattern</h3><span className="kicker">{inputLabel}</span></div>
                <p>Each square is one pair of brain regions. Brighter teal means the two regions fluctuate together more strongly. Hover a square to read its value.</p>
                {matrix ? (
                  <>
                    <div style={{ marginTop: 18 }} key={`${group}-${inputKind}`}>
                      <svg className="heatmap heatmap-anim" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Functional connectivity heatmap for ${inputLabel}`}>
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
                  </>
                ) : (
                  <p style={{ marginTop: 18 }}>No pattern to show yet — upload your two scan files in step 1 and run the analysis. Your scan’s own connectivity pattern will appear here.</p>
                )}
              </div>
            </div>

            {/* Run bar */}
            <div className="run-bar glass">
              <div className="run-text">
                <b>
                  {inputKind === 'scan'
                    ? scanStage === 'working' ? 'Analysing your scan…' : scanStage === 'done' ? 'Analysis complete — the trained model read your scan.' : scanStage === 'error' ? 'Something went wrong.' : 'Ready when you are.'
                    : runState === 'running' ? RUN_STEPS[runStep].label : runState === 'done' ? 'Analysis complete.' : 'Ready when you are.'}
                </b>
                <span>
                  {inputKind === 'scan'
                    ? scanStage === 'working' ? scanStatus : scanStage === 'done' ? 'Your result is below — read the drivers and the limits, not just the label.' : scanStage === 'error' ? 'See step 1 for what went wrong.' : 'Your scans go to the inference service; processing takes a few minutes.'
                    : runState === 'running'
                      ? 'Comparing this pattern against the three reference groups…'
                      : runState === 'done'
                        ? 'Your result is below — read the drivers and the limits, not just the label.'
                        : 'This runs on illustrative demo data — the trained model only runs on uploaded scans.'}
                </span>
                {inputKind === 'scan' && scanStage === 'working' && (
                  <div className="run-progress" aria-hidden="true"><div style={{ width: '65%' }} /></div>
                )}
                {inputKind !== 'scan' && runState === 'running' && (
                  <div className="run-progress" aria-hidden="true"><div style={{ width: `${RUN_STEPS[runStep].pct}%` }} /></div>
                )}
              </div>
              <Magnetic>
                {inputKind === 'scan' ? (
                  <button type="button" className="btn" onClick={runScan} disabled={scanStage === 'working'}>
                    {scanStage === 'working' ? 'Analysing…' : scanStage === 'done' ? '↻ Analyse again' : '▶ Analyse my scan'}
                  </button>
                ) : (
                  <button type="button" className="btn" onClick={runAnalysis} disabled={runState === 'running'}>
                    {runState === 'running' ? 'Running…' : runState === 'done' ? '↻ Run it again' : '▶ Run analysis'}
                  </button>
                )}
              </Magnetic>
            </div>

            {/* Step 3 — scan result (real model) */}
            {inputKind === 'scan' && scanStage === 'done' && scanResult && (
              <div className="card result-card glass" aria-live="polite">
                <div className="panel-head"><h3>3 · Your result</h3><span className="kicker">Trained model · version {scanResult.model_version ?? '—'}</span></div>
                <div className="result-head">
                  <div className="result-group">
                    {scanResult.three_class?.prediction_long ?? scanResult.binary?.prediction_long ?? 'No call'}
                    <small>
                      {scanResult.binary && `Healthy vs Parkinson's: ${scanResult.binary.prediction_long} (${Math.round(Math.max(...Object.values(scanResult.binary.probabilities)) * 100)}% confidence). `}
                      {scanResult.three_class && `Three groups: ${scanResult.three_class.prediction_long} (${Math.round(Math.max(...Object.values(scanResult.three_class.probabilities)) * 100)}% confidence).`}
                    </small>
                  </div>
                  <div className="match">
                    <b>{scanResult.binary ? `${Math.round(Math.max(...Object.values(scanResult.binary.probabilities)) * 100)}%` : '—'}</b>
                    <span>model confidence (healthy vs Parkinson’s)</span>
                    <div className="match-meter" aria-hidden="true"><div style={{ width: `${scanResult.binary ? Math.round(Math.max(...Object.values(scanResult.binary.probabilities)) * 100) : 0}%` }} /></div>
                  </div>
                </div>
                <div className="result-cols">
                  <div>
                    <h4>What drove it</h4>
                    {scanResult.top_connections?.map((d, i) => (
                      <div className="driver-row" key={i}>
                        <span className="r">r = {d.r.toFixed(2)}</span>
                        <p><b>{d.a} ↔ {d.b}</b> — {d.r > 0 ? 'these two regions rise and fall together.' : 'these two regions move in opposite directions.'}</p>
                      </div>
                    ))}
                    <p style={{ marginTop: 12, fontSize: '0.9rem', color: 'var(--faint)' }}>
                      The call rests on the whole pattern of 4,950 connections, not on any one connection above.
                    </p>
                  </div>
                  <div>
                    <h4>The scan, in numbers</h4>
                    <div className="kv"><span>Volumes in your scan</span><span>{scanResult.n_vol ?? '—'}</span></div>
                    <div className="kv"><span>Mean connection strength</span><span>|r| = {scanResult.mean_abs_r?.toFixed(2) ?? '—'}</span></div>
                    <div className="kv"><span>Processing time</span><span>{scanResult.preprocessing_seconds ?? '—'}s</span></div>
                    <div className="kv"><span>Model</span><span>{scanResult.model_label ?? '—'}</span></div>
                    <div className="kv"><span>Trained on</span><span>{scanResult.trained_on ?? '—'}</span></div>
                  </div>
                </div>
                <div className="result-meaning">
                  <b>What this means:</b> the trained model read the connections in your scan and picked the closest match it was trained on. It was trained on 55 people, so a single-scan call is weak evidence — useful for research, never a diagnosis. If you are worried about symptoms, see a doctor.
                </div>
                <div className="illustrative-note">
                  <span className="mark" aria-hidden="true">⚠</span>
                  <span><b>Real model, small sample.</b><br />
                  This result came from the trained model running on your uploaded scan — not from the demo. The model’s own cross-validated scores
                  {scanResult.binary?.cv?.auc?.mean !== undefined && <>: healthy-vs-Parkinson’s AUC {scanResult.binary.cv.auc.mean.toFixed(2)}</>}
                  {scanResult.three_class?.cv?.balanced_accuracy?.mean !== undefined && <>, three-group balanced accuracy {scanResult.three_class.cv.balanced_accuracy.mean.toFixed(2)}</>}
                  . Treat the three-group call with extra caution.</span>
                </div>
              </div>
            )}

            {/* Step 3 — result */}
            {runState === 'done' && prediction && (
              <div className="card result-card glass" aria-live="polite">
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
                        <p><b>{parcelFull(d.a)} ↔ {parcelFull(d.b)}</b> — {d.r > 0 ? 'these two regions rise and fall together' : 'these two regions move in opposite directions'}{i === 0 ? ', and it’s the strongest single coupling in this matrix.' : '.'}</p>
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
            <Reveal>
              <p className="eyebrow">Results</p>
              <h2>Results, in plain words</h2>
              <p className="sub">This section reads <span className="mono">public/results.json</span> — the same file the training pipeline writes to. Click any model to open it up: what it got right, where it got confused, and what the numbers mean in plain words. No numbers are invented to fill the space.</p>
            </Reveal>
            {trainingDone ? (
              <ResultsInteractive results={results!} />
            ) : (
              <Reveal delay={80}>
                <div className="card glass">
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
                </div>
              </Reveal>
            )}
          </div>
        </section>

        {/* ETHICS */}
        <section id="ethics" style={{ paddingTop: 0 }}>
          <div className="wrap">
            <Reveal>
              <div className="card ethics-card glass">
                <p className="eyebrow">Honest limits</p>
                <h2>What this can’t do</h2>
                <p style={{ marginTop: 12 }}>
                  <strong>Small sample.</strong> Fifty-five people is small for deep learning. Whatever
                  this project finds is exploratory, and any real claim needs validation on a larger,
                  independent group of people.
                </p>
                <p style={{ marginTop: 12 }}>
                  <strong>Not a medical diagnosis.</strong> This is a research demo. It does not diagnose,
                  screen or triage any person, and it is never a substitute for assessment by a
                  qualified clinician. If you’re worried about symptoms, talk to a doctor — not
                  a website.
                </p>
                <p style={{ marginTop: 12 }}>
                  <strong>Data &amp; consent.</strong> All data are de-identified and shared openly by the
                  original authors on OpenNeuro under CC0. No new patient data are collected here, and
                  nothing you upload in the demo leaves your browser.
                </p>
              </div>
            </Reveal>
          </div>
        </section>
      </main>

      <footer>
        <div className="wrap">
          <div>
            <strong>Parkinson’s Early Detection</strong>
            <p className="foot-note">
              Made by Rishabh · Dataset: OpenNeuro ds005892 (Kemp et al., 2025) ·
              Code, logs and results are public, so anyone can check the work.
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <a href={DATASET_URL} target="_blank" rel="noreferrer">OpenNeuro ds005892 ↗</a>
            <br />
            <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub repository ↗</a>
          </div>
        </div>
      </footer>
    </>
  );
}
