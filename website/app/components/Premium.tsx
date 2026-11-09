'use client';

import { useEffect, useRef, useState } from 'react';
import {
  GROUPS,
  GROUP_META,
  GroupKey,
  N,
  NETWORKS,
  buildMatrix,
  colorFor,
  featuresOf,
  parcelFull,
} from '../lib';

/* ------------------------------------------------------------------ */
/* Ambient background: gradient mesh + grain + a glow that follows the
   cursor with a lagging (springy) motion. Respects reduced motion.    */
/* ------------------------------------------------------------------ */
export function AmbientBackground() {
  const glowRef = useRef<HTMLDivElement>(null);
  const glow2Ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    let tx = window.innerWidth / 2;
    let ty = window.innerHeight * 0.3;
    let x = tx;
    let y = ty;
    let x2 = tx;
    let y2 = ty;
    let raf = 0;
    const onMove = (e: PointerEvent) => {
      tx = e.clientX;
      ty = e.clientY;
    };
    window.addEventListener('pointermove', onMove, { passive: true });
    const tick = () => {
      raf = requestAnimationFrame(tick);
      x += (tx - x) * 0.06;
      y += (ty - y) * 0.06;
      x2 += (tx - x2) * 0.028;
      y2 += (ty - y2) * 0.028;
      if (glowRef.current) glowRef.current.style.transform = `translate3d(${x - 260}px, ${y - 260}px, 0)`;
      if (glow2Ref.current) glow2Ref.current.style.transform = `translate3d(${x2 - 320}px, ${y2 - 320}px, 0)`;
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('pointermove', onMove);
    };
  }, []);

  return (
    <div className="ambient" aria-hidden="true">
      <div className="ambient-mesh" />
      <div ref={glowRef} className="ambient-glow ambient-glow-a" />
      <div ref={glow2Ref} className="ambient-glow ambient-glow-b" />
      <div className="ambient-grain" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Magnetic button: drifts toward the cursor, springs back on leave.   */
/* ------------------------------------------------------------------ */
export function Magnetic({ children, strength = 0.35 }: { children: React.ReactNode; strength?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const onMove = (e: React.MouseEvent) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2);
    const dy = e.clientY - (r.top + r.height / 2);
    el.style.transform = `translate3d(${dx * strength}px, ${dy * strength}px, 0)`;
  };
  const onLeave = () => {
    const el = ref.current;
    if (el) el.style.transform = 'translate3d(0,0,0)';
  };
  return (
    <span ref={ref} className="magnetic" onMouseMove={onMove} onMouseLeave={onLeave}>
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Tilt card: subtle 3D tilt that follows the pointer.                 */
/* ------------------------------------------------------------------ */
export function TiltCard({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const onMove = (e: React.MouseEvent) => {
    const el = ref.current;
    if (!el || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const r = el.getBoundingClientRect();
    const rx = ((e.clientY - r.top) / r.height - 0.5) * -7;
    const ry = ((e.clientX - r.left) / r.width - 0.5) * 9;
    el.style.transform = `perspective(900px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) translateY(-3px)`;
  };
  const onLeave = () => {
    const el = ref.current;
    if (el) el.style.transform = '';
  };
  return (
    <div ref={ref} className={`tilt ${className}`} onMouseMove={onMove} onMouseLeave={onLeave}>
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Reveal on scroll — springy ease, staggered via inline delay.        */
/* ------------------------------------------------------------------ */
export function Reveal({ children, className = '', delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setVisible(true);
          io.disconnect();
        }
      },
      { threshold: 0.12 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={ref} className={`reveal ${visible ? 'is-visible' : ''} ${className}`} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Animated counter — counts up with an expo ease once in view.        */
/* ------------------------------------------------------------------ */
export function CountUp({ end, decimals = 0, suffix = '' }: { end: number; decimals?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [val, setVal] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting) return;
        io.disconnect();
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
          setVal(end);
          return;
        }
        const t0 = performance.now();
        const dur = 1400;
        const step = (t: number) => {
          const p = Math.min(1, (t - t0) / dur);
          const eased = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);
          setVal(end * eased);
          if (p < 1) raf = requestAnimationFrame(step);
        };
        raf = requestAnimationFrame(step);
      },
      { threshold: 0.4 }
    );
    io.observe(el);
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [end]);
  return (
    <span ref={ref}>
      {val.toFixed(decimals)}
      {suffix}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Sparkline — tiny area chart from a value series.                    */
/* ------------------------------------------------------------------ */
export function Sparkline({ values, color = '#2dd4bf', height = 36 }: { values: number[]; color?: string; height?: number }) {
  const w = 120;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => `${((i / (values.length - 1)) * w).toFixed(1)},${(height - 4 - ((v - min) / span) * (height - 8)).toFixed(1)}`).join(' ');
  const id = `sp-${color.replace('#', '')}`;
  return (
    <svg className="sparkline" viewBox={`0 0 ${w} ${height}`} width={w} height={height} aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.45" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${height} ${pts} ${w},${height}`} fill={`url(#${id})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Distribution chart — animated histogram of all pairwise coupling
   values for the selected group; bars morph when the group changes.
   Built from the same illustrative seeded matrices as the explorer.   */
/* ------------------------------------------------------------------ */
const BINS = 12;
const BIN_MIN = -0.2;
const BIN_MAX = 1.0;

export function DistributionChart() {
  const [group, setGroup] = useState<GroupKey>('HC');
  const g = GROUPS.find((x) => x.key === group)!;
  const m = buildMatrix(group, g.seed);
  const vals: number[] = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) vals.push(m[i][j]);
  const counts = new Array(BINS).fill(0);
  for (const v of vals) {
    const b = Math.min(BINS - 1, Math.max(0, Math.floor(((v - BIN_MIN) / (BIN_MAX - BIN_MIN)) * BINS)));
    counts[b]++;
  }
  const maxC = Math.max(...counts);
  const f = featuresOf(m);
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length;

  return (
    <div className="dist-chart">
      <div className="dist-head">
        <div>
          <h3>How coupling is distributed</h3>
          <p>
            Every pair of parcels, counted by how strongly they move together. Watch the shape shift
            as you switch groups — healthy brains pile up on the right, Parkinson&rsquo;s patterns
            spread out and flatten.
          </p>
        </div>
        <div className="seg dist-seg" role="group" aria-label="Distribution group">
          {GROUPS.map((x) => (
            <button
              key={x.key}
              className={group === x.key ? 'active' : ''}
              style={group === x.key ? { background: GROUP_META[x.key].color, borderColor: GROUP_META[x.key].color } : undefined}
              onClick={() => setGroup(x.key)}
              aria-pressed={group === x.key}
            >
              {GROUP_META[x.key].short}
            </button>
          ))}
        </div>
      </div>
      <div className="dist-bars" role="img" aria-label={`Histogram of connection strengths for ${g.label}`}>
        {counts.map((c, i) => (
          <div key={`${group}-${i}`} className="dist-bar-wrap">
            <div
              className="dist-bar"
              style={{
                height: `${Math.max(4, (c / maxC) * 100)}%`,
                background: `linear-gradient(180deg, ${GROUP_META[group].color}, rgba(45,212,191,0.25))`,
                transitionDelay: `${i * 18}ms`,
              }}
              title={`${c} pairs · r ≈ ${(BIN_MIN + ((i + 0.5) / BINS) * (BIN_MAX - BIN_MIN)).toFixed(2)}`}
            />
            <span className="dist-count">{c}</span>
          </div>
        ))}
      </div>
      <div className="dist-axis">
        <span>r = −0.2</span>
        <span>r = 0.4</span>
        <span>r = 1.0</span>
      </div>
      <div className="dist-stats">
        <span>
          Mean coupling <b>r = {mean.toFixed(2)}</b>
        </span>
        <span>
          Within-network <b>r = {f.within.toFixed(2)}</b>
        </span>
        <span>
          Between-network <b>r = {f.between.toFixed(2)}</b>
        </span>
        <span className="dist-note">Illustrative seeded data · 190 parcel pairs</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Group comparison slider — drag a divider between two group heatmaps.*/
/* ------------------------------------------------------------------ */
export function CompareSlider() {
  const [left, setLeft] = useState<GroupKey>('HC');
  const [right, setRight] = useState<GroupKey>('PD-MCI');
  const [pos, setPos] = useState(50);
  const trackRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const renderHeat = (g: GroupKey) => {
    const m = buildMatrix(g, GROUPS.find((x) => x.key === g)!.seed);
    const cell = 11;
    return (
      <svg viewBox={`0 0 ${N * cell} ${N * cell}`} className="compare-heat" aria-hidden="true">
        {m.map((row, i) =>
          row.map((v, j) => (
            <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1} fill={colorFor(v)} />
          ))
        )}
      </svg>
    );
  };

  const updatePos = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos(Math.max(4, Math.min(96, ((clientX - r.left) / r.width) * 100)));
  };

  return (
    <div className="compare">
      <div className="compare-head">
        <div>
          <h3>Drag the line. See the difference.</h3>
          <p>
            The same 20 parcels, two groups, one picture. Pick any two groups and drag the divider
            across — where the colours change, the pattern changed.
          </p>
        </div>
        <div className="compare-pickers">
          <label>
            Left
            <select value={left} onChange={(e) => setLeft(e.target.value as GroupKey)}>
              {GROUPS.map((x) => (
                <option key={x.key} value={x.key}>
                  {x.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Right
            <select value={right} onChange={(e) => setRight(e.target.value as GroupKey)}>
              {GROUPS.map((x) => (
                <option key={x.key} value={x.key}>
                  {x.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div
        ref={trackRef}
        className="compare-track"
        onPointerDown={(e) => {
          draggingRef.current = true;
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          updatePos(e.clientX);
        }}
        onPointerMove={(e) => {
          if (draggingRef.current) updatePos(e.clientX);
        }}
        onPointerUp={() => {
          draggingRef.current = false;
        }}
        role="slider"
        aria-label="Comparison divider"
        aria-valuenow={Math.round(pos)}
        aria-valuemin={0}
        aria-valuemax={100}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft') setPos((p) => Math.max(4, p - 4));
          if (e.key === 'ArrowRight') setPos((p) => Math.min(96, p + 4));
        }}
      >
        <div className="compare-layer compare-layer-right">{renderHeat(right)}</div>
        <div className="compare-layer compare-layer-left" style={{ width: `${pos}%` }}>
          {renderHeat(left)}
        </div>
        <div className="compare-divider" style={{ left: `${pos}%` }}>
          <span className="compare-knob">⟨ ⟩</span>
        </div>
        <span className="compare-tag compare-tag-l" style={{ background: GROUP_META[left].color }}>
          {GROUP_META[left].short}
        </span>
        <span className="compare-tag compare-tag-r" style={{ background: GROUP_META[right].color }}>
          {GROUP_META[right].short}
        </span>
      </div>
      <p className="compare-note">
        {GROUP_META[left].label} on the left, {GROUP_META[right].label} on the right. Illustrative
        seeded patterns — the real comparison lands with the trained results.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Pipeline timeline — each step expands with detail + a small live
   visual. Click a step (or use the keyboard) to open it.              */
/* ------------------------------------------------------------------ */
const RAW_WAVE = 'M0,52 L14,44 L28,58 L42,36 L56,60 L70,30 L84,56 L98,40 L112,58 L126,34 L140,54 L154,42 L168,60 L182,32 L196,52 L210,44 L224,56 L238,38 L252,52';
const CLEAN_WAVE = 'M0,48 C20,44 34,50 52,47 C72,44 88,50 108,47 C128,44 146,50 164,47 C184,44 200,50 218,47 C232,45 244,48 252,47';

function MiniHeat() {
  const m = buildMatrix('HC', 11);
  const cell = 9;
  const shown = 10;
  return (
    <svg viewBox={`0 0 ${shown * cell} ${shown * cell}`} className="mini-heat" role="img" aria-label="Mini connectivity heatmap">
      {Array.from({ length: shown }, (_, i) =>
        Array.from({ length: shown }, (_, j) => (
          <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1} fill={colorFor(m[i * 2][j * 2])}>
            <title>{`${parcelFull(i * 2)} ↔ ${parcelFull(j * 2)}: r = ${m[i * 2][j * 2].toFixed(2)}`}</title>
          </rect>
        ))
      )}
    </svg>
  );
}

export function PipelineTimeline() {
  const [open, setOpen] = useState(2);
  const [clean, setClean] = useState(70);
  const [chip, setChip] = useState<GroupKey>('HC');

  const steps = [
    {
      id: 'acquire',
      title: 'Acquire',
      line: 'Download the data. Count everything. Trust nothing yet.',
      detail:
        'ds005892 v1.0.0 comes down from OpenNeuro — 275 brain-scan files, 2.37 GB. The audit counts every subject per group, checks every file by size, and notes what is missing before a single model trains. Nothing here is pre-processed: the dataset is raw, so the pipeline below runs from scratch.',
      visual: (
        <div className="pipe-visual">
          <div className="kv"><span>NIfTI files verified</span><span>275 / 275</span></div>
          <div className="kv"><span>Total size</span><span>2.37 GB</span></div>
          <div className="kv"><span>BOLD runs · 200 vols, TR 2 s</span><span>55 / 55</span></div>
          <div className="kv"><span>Derivatives included</span><span>None — raw only</span></div>
        </div>
      ),
    },
    {
      id: 'preprocess',
      title: 'Preprocess',
      line: 'Clean the signal. A brain scan is mostly movement and machine noise.',
      detail:
        'Motion correction, normalisation to MNI space, nuisance regression and band-pass filtering of the BOLD signal, via fMRIPrep derivatives where they exist — otherwise a Nilearn pipeline. Drag the slider to feel what cleaning does to one parcel’s time series.',
      visual: (
        <div className="pipe-visual">
          <svg viewBox="0 0 252 84" className="wave-svg" role="img" aria-label="BOLD time series, raw versus cleaned">
            <path d={RAW_WAVE} fill="none" stroke="#f87171" strokeWidth="1.6" opacity={1 - clean / 100} />
            <path d={CLEAN_WAVE} fill="none" stroke="#2dd4bf" strokeWidth="2" opacity={clean / 100} />
          </svg>
          <label className="wave-slider">
            <span>Raw</span>
            <input type="range" min={0} max={100} value={clean} onChange={(e) => setClean(Number(e.target.value))} aria-label="Cleaning amount" />
            <span>Cleaned</span>
          </label>
          <p className="pipe-note">Cleaning: {clean}% — motion and drift removed, slow brain fluctuations kept.</p>
        </div>
      ),
    },
    {
      id: 'connectivity',
      title: 'Connectivity',
      line: 'Turn a 4D scan into a conversation between regions.',
      detail:
        'The Schaefer atlas divides the brain into 100 parcels. One time series comes out of each parcel, and every pair is correlated. Each person becomes a 100 × 100 functional-connectivity matrix — 4,950 numbers that say which regions fluctuate together. Hover the squares to read values.',
      visual: (
        <div className="pipe-visual">
          <MiniHeat />
          <p className="pipe-note">A 10 × 10 peek at one person’s matrix (full analysis: 100 × 100). Brighter teal, stronger coupling.</p>
        </div>
      ),
    },
    {
      id: 'classify',
      title: 'Classify',
      line: 'Seven models, same splits, one fair test.',
      detail:
        'GCN/GAT, MLP, 1D-CNN, SVM, logistic regression, random forest / XGBoost and a 3D CNN comparison all train on identical stratified cross-validation splits, max 1000 epochs with early stopping. Tap a group to see its reference pattern — the number every model is trying to learn.',
      visual: (
        <div className="pipe-visual">
          <div className="seg" role="group" aria-label="Reference group">
            {GROUPS.map((x) => (
              <button key={x.key} className={chip === x.key ? 'active' : ''} onClick={() => setChip(x.key)} aria-pressed={chip === x.key}>
                {GROUP_META[x.key].short}
              </button>
            ))}
          </div>
          <div className="kv"><span>Within-network coupling</span><span>r = {featuresOf(buildMatrix(chip, GROUPS.find((x) => x.key === chip)!.seed)).within.toFixed(2)}</span></div>
          <div className="kv"><span>Between-network coupling</span><span>r = {featuresOf(buildMatrix(chip, GROUPS.find((x) => x.key === chip)!.seed)).between.toFixed(2)}</span></div>
          <p className="pipe-note">Reference values from the illustrative demo patterns — real ones arrive with training.</p>
        </div>
      ),
    },
  ];

  return (
    <div className="pipeline">
      {steps.map((s, i) => (
        <div key={s.id} className={`pipe-step ${open === i ? 'open' : ''}`}>
          <button type="button" className="pipe-head" onClick={() => setOpen(open === i ? -1 : i)} aria-expanded={open === i}>
            <span className="pipe-num">{i + 1}</span>
            <span className="pipe-titles">
              <b>{s.title}</b>
              <span>{s.line}</span>
            </span>
            <span className="pipe-chev" aria-hidden="true">{open === i ? '−' : '+'}</span>
          </button>
          <div className="pipe-body" aria-hidden={open !== i}>
            <p>{s.detail}</p>
            {s.visual}
          </div>
        </div>
      ))}
    </div>
  );
}
