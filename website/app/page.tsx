'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  DATASET_URL,
  GROUPS,
  GROUP_META,
  GroupKey,
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
                Built for <strong>Rishabh</strong>. A research demo, not a medical device — it
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
                Three steps, about a minute. Pick a sample subject or upload your own connectivity
                matrix, run the analysis, and read the result — including what drove it and what it
                doesn’t prove. Nothing you upload leaves your browser.
              </p>
            </Reveal>

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
              <div className="card panel glass">
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
                  <p><strong style={{ color: 'var(--ink)' }}>{dragging ? 'Let go — we’ll take it from here.' : 'Drop a CSV here, or choose a file.'}</strong><br />A 20×20 or 100×100 table of correlations, one region per row.</p>
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
                  <div className="error-box" role="alert"><strong>That file didn’t work — here’s why.</strong><br />{uploadError}</div>
                )}
              </div>

              {/* Step 2 — preview */}
              <div className="card panel glass">
                <div className="panel-head"><h3>2 · Look at the pattern</h3><span className="kicker">{inputLabel}</span></div>
                <p>Each square is one pair of brain regions. Brighter teal means the two regions fluctuate together more strongly. Hover a square to read its value.</p>
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
              </div>
            </div>

            {/* Run bar */}
            <div className="run-bar glass">
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
              <Magnetic>
                <button type="button" className="btn" onClick={runAnalysis} disabled={runState === 'running'}>
                  {runState === 'running' ? 'Running…' : runState === 'done' ? '↻ Run it again' : '▶ Run analysis'}
                </button>
              </Magnetic>
            </div>

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
              <h2>Results, as they land</h2>
              <p className="sub">This section reads <span className="mono">public/results.json</span> — the same file the training pipeline writes to. Until a training run finishes, it shows the in-progress state. No numbers are invented to fill the space.</p>
            </Reveal>
            <Reveal delay={80}>
              <div className="card glass">
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
            </Reveal>
            <div className="grid grid-3" style={{ marginTop: 18 }}>
              <Reveal delay={0}><div className="card glass"><div className="stat-label">Accuracy</div><p>{trainingDone ? 'Published above, per model.' : 'Pending — reported per model after cross-validation.'}</p></div></Reveal>
              <Reveal delay={90}><div className="card glass"><div className="stat-label">Balanced accuracy / AUC</div><p>{trainingDone ? 'Published above, per model.' : 'Pending — the deciding metrics, given the 22 / 18 / 15 split.'}</p></div></Reveal>
              <Reveal delay={180}><div className="card glass"><div className="stat-label">Confusion matrix</div><p>{trainingDone ? 'In the GitHub repository, per model.' : 'Pending — HC vs PD-NC vs PD-MCI, per fold and averaged.'}</p></div></Reveal>
            </div>
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
              Built for Rishabh · Dataset: OpenNeuro ds005892 (Kemp et al., 2025) ·
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
