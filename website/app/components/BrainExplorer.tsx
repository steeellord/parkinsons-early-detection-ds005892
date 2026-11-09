'use client';

import { useEffect, useRef, useState } from 'react';
import {
  GROUPS,
  GROUP_META,
  GroupKey,
  N,
  NETWORKS,
  NETWORK_COLORS,
  buildMatrix,
  networkOf,
  parcelFull,
  parcelLabel,
  topConnectionsForNode,
} from '../lib';

/* Brain-like layout for the 20 parcels (4 networks x 5), on a 760x480 canvas. */
const BASE_POS: { x: number; y: number }[] = [
  // Visual
  { x: 130, y: 310 }, { x: 180, y: 360 }, { x: 110, y: 385 }, { x: 200, y: 305 }, { x: 160, y: 425 },
  // Motor
  { x: 250, y: 110 }, { x: 320, y: 78 }, { x: 392, y: 108 }, { x: 290, y: 162 }, { x: 362, y: 152 },
  // Default
  { x: 330, y: 252 }, { x: 400, y: 230 }, { x: 462, y: 272 }, { x: 360, y: 312 }, { x: 432, y: 322 },
  // Fronto-parietal
  { x: 560, y: 150 }, { x: 632, y: 182 }, { x: 580, y: 242 }, { x: 662, y: 252 }, { x: 620, y: 322 },
];

const W = 760;
const H = 480;

type Pulse = { a: number; b: number; t: number; speed: number };

export default function BrainExplorer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [group, setGroup] = useState<GroupKey>('HC');
  const [playing, setPlaying] = useState(true);
  const [hover, setHover] = useState<{ node: number; x: number; y: number } | null>(null);
  const [selected, setSelected] = useState<number | null>(null);

  const groupRef = useRef<GroupKey>('HC');
  const playingRef = useRef(true);
  const selectedRef = useRef<number | null>(null);
  const hoverRef = useRef<number | null>(null);
  const dragRef = useRef<number | null>(null);
  const mouseRef = useRef<{ x: number; y: number; inside: boolean }>({ x: 0, y: 0, inside: false });
  const posRef = useRef<{ x: number; y: number }[]>(BASE_POS.map((p) => ({ ...p })));
  const shownRef = useRef<number[][]>(buildMatrix('HC', 11));
  const pulsesRef = useRef<Pulse[]>([]);
  const reducedRef = useRef(false);

  groupRef.current = group;
  playingRef.current = playing;
  selectedRef.current = selected;

  useEffect(() => {
    reducedRef.current = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedRef.current) setPlaying(false);
  }, []);

  /* Re-seed pulses whenever the group changes. */
  useEffect(() => {
    const seed = GROUPS.find((g) => g.key === group)!.seed;
    const m = buildMatrix(group, seed);
    const strong: [number, number][] = [];
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) if (m[i][j] > 0.45) strong.push([i, j]);
    }
    const rng = Math.random;
    pulsesRef.current = Array.from({ length: 16 }, () => {
      const e = strong[Math.floor(rng() * strong.length)] ?? [0, 1];
      return { a: e[0], b: e[1], t: rng(), speed: 0.0025 + rng() * 0.004 };
    });
  }, [group]);

  /* Main render loop: morph edge weights toward the active group, draw the
     network, drift nodes gently, and push pulses along strong edges. */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    let raf = 0;
    let time = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      time += 1;
      const g = groupRef.current;
      const seed = GROUPS.find((x) => x.key === g)!.seed;
      const target = buildMatrix(g, seed);
      const shown = shownRef.current;
      const playingNow = playingRef.current && !reducedRef.current;
      const sel = selectedRef.current;
      const hov = hoverRef.current;

      // Ease displayed weights toward the target group.
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          shown[i][j] += (target[i][j] - shown[i][j]) * 0.06;
        }
      }

      ctx.clearRect(0, 0, W, H);

      // Soft brain outline.
      ctx.beginPath();
      ctx.moveTo(90, 240);
      ctx.bezierCurveTo(70, 120, 220, 40, 380, 48);
      ctx.bezierCurveTo(540, 56, 690, 110, 700, 230);
      ctx.bezierCurveTo(708, 330, 620, 440, 470, 448);
      ctx.bezierCurveTo(320, 456, 120, 430, 82, 330);
      ctx.bezierCurveTo(72, 300, 82, 268, 90, 240);
      ctx.closePath();
      ctx.fillStyle = 'rgba(45,212,191,0.045)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(125,211,252,0.16)';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      const pos = posRef.current;
      const drift = playingNow;
      const mouse = mouseRef.current;
      const px = pos.map((p, i) => {
        let x = p.x + (drift && dragRef.current !== i ? Math.sin(time * 0.008 + i * 1.7) * 3 : 0);
        let y = p.y + (drift && dragRef.current !== i ? Math.cos(time * 0.006 + i * 2.3) * 3 : 0);
        // Gentle cursor repulsion — the network leans away from the pointer.
        if (mouse.inside && dragRef.current !== i) {
          const dx = x - mouse.x;
          const dy = y - mouse.y;
          const d = Math.hypot(dx, dy);
          if (d > 0.01 && d < 110) {
            const f = ((110 - d) / 110) * 14;
            x += (dx / d) * f;
            y += (dy / d) * f;
          }
        }
        return { x, y };
      });

      // Edges.
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const v = shown[i][j];
          if (Math.abs(v) < 0.28) continue;
          const focused = sel === null || sel === i || sel === j;
          const a = px[i];
          const b = px[j];
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          const alpha = focused ? 0.16 + Math.abs(v) * 0.5 : 0.05;
          ctx.strokeStyle = v > 0 ? `rgba(45,212,191,${alpha})` : `rgba(248,113,113,${alpha * 0.8})`;
          ctx.lineWidth = focused ? 0.8 + Math.abs(v) * 1.8 : 0.7;
          ctx.stroke();
        }
      }

      // Pulses travelling along strong edges. Every so often, one pulse is
      // steered to the edge nearest the cursor, so the network answers the mouse.
      if (playingNow) {
        const pulses = pulsesRef.current;
        if (mouse.inside && time % 90 === 0 && pulses.length > 0) {
          let nearest = 0;
          let nearestD = Infinity;
          pos.forEach((p, i) => {
            const d = Math.hypot(p.x - mouse.x, p.y - mouse.y);
            if (d < nearestD) {
              nearestD = d;
              nearest = i;
            }
          });
          let bestJ = -1;
          let bestV = 0;
          for (let j = 0; j < N; j++) {
            if (j !== nearest && Math.abs(shown[nearest][j]) > bestV) {
              bestV = Math.abs(shown[nearest][j]);
              bestJ = j;
            }
          }
          if (bestJ >= 0) {
            pulses[0].a = nearest;
            pulses[0].b = bestJ;
            pulses[0].t = 0;
          }
        }
        for (const p of pulses) {
          p.t += p.speed;
          if (p.t > 1) {
            const m2 = buildMatrix(g, seed);
            const strong: [number, number][] = [];
            for (let i = 0; i < N; i++) {
              for (let j = i + 1; j < N; j++) if (m2[i][j] > 0.45) strong.push([i, j]);
            }
            const e = strong[Math.floor(Math.random() * strong.length)] ?? [0, 1];
            p.a = e[0];
            p.b = e[1];
            p.t = 0;
          }
          const a = px[p.a];
          const b = px[p.b];
          const x = a.x + (b.x - a.x) * p.t;
          const y = a.y + (b.y - a.y) * p.t;
          const grad = ctx.createRadialGradient(x, y, 0, x, y, 9);
          grad.addColorStop(0, 'rgba(94,234,212,0.95)');
          grad.addColorStop(1, 'rgba(94,234,212,0)');
          ctx.beginPath();
          ctx.arc(x, y, 9, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(x, y, 2.4, 0, Math.PI * 2);
          ctx.fillStyle = '#a7f3d0';
          ctx.fill();
        }
      }

      // Nodes.
      for (let i = 0; i < N; i++) {
        const p = px[i];
        const net = networkOf(i);
        const isSel = sel === i;
        const isHov = hov === i;
        const dimmed = sel !== null && sel !== i;
        const r = isSel ? 11 : isHov ? 10 : 7;
        if (isSel || isHov) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, r + 9, 0, Math.PI * 2);
          ctx.fillStyle = isSel ? 'rgba(45,212,191,0.18)' : 'rgba(125,211,252,0.12)';
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = dimmed ? '#0a1826' : NETWORK_COLORS[net];
        ctx.globalAlpha = dimmed ? 0.45 : 1;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.lineWidth = isSel ? 2.5 : 1.5;
        ctx.strokeStyle = isSel ? '#5eead4' : 'rgba(6,13,20,0.9)';
        ctx.stroke();
        // Label inside node.
        ctx.fillStyle = '#04211d';
        ctx.font = 'bold 8.5px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(parcelLabel(i), p.x, p.y + 0.5);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  const canvasPos = (e: React.PointerEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * W,
      y: ((e.clientY - rect.top) / rect.height) * H,
    };
  };

  const hitNode = (x: number, y: number): number | null => {
    let best: number | null = null;
    let bestD = 30;
    posRef.current.forEach((p, i) => {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return best;
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const { x, y } = canvasPos(e);
    mouseRef.current = { x, y, inside: true };
    if (dragRef.current !== null) {
      const i = dragRef.current;
      posRef.current[i] = { x: Math.max(16, Math.min(W - 16, x)), y: Math.max(16, Math.min(H - 16, y)) };
      return;
    }
    const hit = hitNode(x, y);
    hoverRef.current = hit;
    if (hit !== null) {
      const rect = canvasRef.current!.getBoundingClientRect();
      setHover({ node: hit, x: e.clientX - rect.left, y: e.clientY - rect.top });
      canvasRef.current!.style.cursor = 'pointer';
    } else {
      setHover(null);
      canvasRef.current!.style.cursor = 'grab';
    }
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const { x, y } = canvasPos(e);
    const hit = hitNode(x, y);
    if (hit !== null) {
      dragRef.current = hit;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const wasDrag = dragRef.current !== null;
    const { x, y } = canvasPos(e);
    dragRef.current = null;
    if (!wasDrag) return;
    const hit = hitNode(x, y);
    // A click (no real drag) toggles selection; a drag just moves the node.
    setSelected((prev) => (hit !== null && hit === prev ? null : hit));
  };

  const active = GROUPS.find((g) => g.key === group)!;
  const activeMatrix = buildMatrix(group, active.seed);
  const hoveredFeatures =
    hover !== null
      ? {
          label: parcelFull(hover.node),
          network: NETWORKS[networkOf(hover.node)],
          mean: activeMatrix[hover.node].reduce((s, v, j) => (j === hover.node ? s : s + v), 0) / (N - 1),
          top: topConnectionsForNode(activeMatrix, hover.node, 1)[0],
        }
      : null;
  const selectedTop = selected !== null ? topConnectionsForNode(activeMatrix, selected, 5) : [];

  return (
    <div className="explorer">
      <div className="explorer-head">
        <div>
          <h3 className="explorer-title">Explore the network yourself</h3>
          <p className="explorer-sub">
            Twenty parcels, four networks. Hover a node to inspect it, click one to isolate its
            connections, drag nodes around. Switch groups and watch the pattern change.
          </p>
        </div>
        <div className="explorer-controls">
          <div className="seg explorer-seg" role="group" aria-label="Reference group">
            {GROUPS.map((g) => (
              <button
                key={g.key}
                className={group === g.key ? 'active' : ''}
                style={group === g.key ? { background: GROUP_META[g.key].color, borderColor: GROUP_META[g.key].color } : undefined}
                onClick={() => {
                  setGroup(g.key);
                  setSelected(null);
                }}
                aria-pressed={group === g.key}
              >
                {GROUP_META[g.key].short}
              </button>
            ))}
          </div>
          <button type="button" className="btn-ghost btn-sm" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
            {playing ? '⏸ Pause motion' : '▶ Play motion'}
          </button>
        </div>
      </div>

      <div className="explorer-body">
        <div className="explorer-canvas-wrap" ref={wrapRef}>
          <canvas
            ref={canvasRef}
            className="explorer-canvas"
            width={W}
            height={H}
            onPointerMove={onPointerMove}
            onPointerDown={onPointerDown}
            onPointerUp={onPointerUp}
            onPointerLeave={() => {
              setHover(null);
              hoverRef.current = null;
              mouseRef.current.inside = false;
            }}
            aria-label="Interactive brain network. Use the group buttons and the connection list to explore."
          />
          {hover !== null && hoveredFeatures && (
            <div className="explorer-tip" style={{ left: Math.min(hover.x + 16, 560), top: Math.max(hover.y - 10, 8) }}>
              <b>{hoveredFeatures.label}</b>
              <span>{hoveredFeatures.network} network</span>
              <span>
                Mean coupling r = {hoveredFeatures.mean.toFixed(2)} · strongest: {parcelLabel(hoveredFeatures.top.other)} (r ={' '}
                {hoveredFeatures.top.r.toFixed(2)})
              </span>
            </div>
          )}
          <div className="explorer-hint">Drag nodes to rearrange · Click a node to isolate it · Click again to release</div>
        </div>

        <aside className="explorer-side" aria-live="polite">
          <div className="explorer-group" style={{ borderColor: GROUP_META[group].color }}>
            <span className="swatch" style={{ background: GROUP_META[group].color }} />
            <div>
              <b>{active.label}</b>
              <span>{active.subject}</span>
            </div>
          </div>
          <p className="explorer-blurb">{active.blurb}</p>
          {selected !== null ? (
            <>
              <h4>
                {parcelFull(selected)} — strongest connections
                <button type="button" className="link-btn" onClick={() => setSelected(null)}>
                  clear
                </button>
              </h4>
              {selectedTop.map((c) => (
                <div className="conn-row" key={c.other}>
                  <span className="conn-label">
                    {parcelLabel(selected)} ↔ {parcelLabel(c.other)}
                    <small>{parcelFull(c.other)}</small>
                  </span>
                  <span className="conn-bar">
                    <span style={{ width: `${Math.min(100, Math.abs(c.r) * 100)}%`, background: c.r > 0 ? '#2dd4bf' : '#f87171' }} />
                  </span>
                  <b className="conn-val">r = {c.r.toFixed(2)}</b>
                </div>
              ))}
            </>
          ) : (
            <>
              <h4>Strongest connections in this pattern</h4>
              {topConnectionsForNode(activeMatrix, 0, 0).length === 0 && null}
              {(() => {
                const pairs: { a: number; b: number; r: number }[] = [];
                for (let i = 0; i < N; i++)
                  for (let j = i + 1; j < N; j++) pairs.push({ a: i, b: j, r: activeMatrix[i][j] });
                return pairs
                  .sort((x, y) => Math.abs(y.r) - Math.abs(x.r))
                  .slice(0, 4)
                  .map((p) => (
                    <div className="conn-row" key={`${p.a}-${p.b}`}>
                      <span className="conn-label">
                        {parcelLabel(p.a)} ↔ {parcelLabel(p.b)}
                        <small>
                          {parcelFull(p.a)} ↔ {parcelFull(p.b)}
                        </small>
                      </span>
                      <span className="conn-bar">
                        <span style={{ width: `${Math.min(100, Math.abs(p.r) * 100)}%`, background: '#2dd4bf' }} />
                      </span>
                      <b className="conn-val">r = {p.r.toFixed(2)}</b>
                    </div>
                  ));
              })()}
              <p className="explorer-blurb" style={{ marginTop: 14 }}>
                Click any node in the network to see <em>its</em> connections instead.
              </p>
            </>
          )}
          <div className="explorer-legend">
            {NETWORKS.map((n, i) => (
              <span key={n}>
                <i style={{ background: NETWORK_COLORS[i] }} /> {n}
              </span>
            ))}
          </div>
          <p className="explorer-note">
            Illustrative patterns from seeded demo data — the shape of the signal, not real patient
            values. The trained model replaces this once training finishes.
          </p>
        </aside>
      </div>
    </div>
  );
}
