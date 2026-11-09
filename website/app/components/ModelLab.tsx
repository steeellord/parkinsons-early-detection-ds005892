'use client';

import { useState } from 'react';
import { MODELS } from '../lib';
import { TiltCard } from './Premium';

export default function ModelLab() {
  const [open, setOpen] = useState<string | null>('Graph Neural Network (GCN / GAT)');
  const [pinned, setPinned] = useState<string[]>([]);

  const togglePin = (name: string) => {
    setPinned((prev) => {
      if (prev.includes(name)) return prev.filter((n) => n !== name);
      if (prev.length >= 2) return [prev[1], name];
      return [...prev, name];
    });
  };

  const pinnedModels = MODELS.filter((m) => pinned.includes(m.name));

  return (
    <div>
      <div className="lab-hint">
        <span>Click a card to open it. Pin up to two models to compare them side by side.</span>
        <span className="lab-count">
          {pinned.length === 0 ? 'Nothing pinned yet' : `${pinned.length} pinned — ${pinned.length === 2 ? 'compare below' : 'pin one more to compare'}`}
        </span>
      </div>

      <div className="grid grid-2 lab-grid">
        {MODELS.map((m) => {
          const isOpen = open === m.name;
          const isPinned = pinned.includes(m.name);
          return (
            <TiltCard key={m.name} className={`card model-card lab-card ${isOpen ? 'open' : ''} ${isPinned ? 'pinned' : ''}`}>
              <button type="button" className="lab-head" onClick={() => setOpen(isOpen ? null : m.name)} aria-expanded={isOpen}>
                <span className="lab-titles">
                  <span className="tag">Benchmarked · {m.load}</span>
                  <span className="model-name">{m.name}</span>
                </span>
                <span className="lab-chev" aria-hidden="true">{isOpen ? '−' : '+'}</span>
              </button>
              <p className="model-fit">{m.fit}</p>
              <div className="lab-body" aria-hidden={!isOpen}>
                <p className="model-note">
                  <b>Why it’s here:</b> {m.why}
                </p>
                <p className="model-note model-weak">
                  <b>Weakness on 55 people:</b> {m.weakness}
                </p>
                <p className="model-note">
                  <b>Status:</b> awaiting results — no score exists yet, and none is shown until
                  training publishes it.
                </p>
              </div>
              <button
                type="button"
                className={`pin-btn ${isPinned ? 'pinned' : ''}`}
                onClick={() => togglePin(m.name)}
                aria-pressed={isPinned}
              >
                {isPinned ? '★ Pinned for compare' : '☆ Pin to compare'}
              </button>
            </TiltCard>
          );
        })}
        <TiltCard className="card model-card lab-card" >
          <span className="tag">Evaluation</span>
          <div className="model-name">How the winner is chosen</div>
          <p className="model-fit">
            Stratified 5-fold cross-validation (repeated), with leave-one-out as a sanity check. The
            deciding metrics are balanced accuracy and macro-AUC, because the groups are uneven
            (22 / 18 / 15).
          </p>
          <p className="model-note">
            <b>Training rule:</b> max 1000 epochs, early stopping on validation loss, cosine
            learning-rate decay — and we keep the best epoch, not the last one.
          </p>
        </TiltCard>
      </div>

      {pinnedModels.length > 0 && (
        <div className="compare-tray" aria-live="polite">
          <div className="compare-tray-head">
            <h3>Side by side</h3>
            <button type="button" className="link-btn" onClick={() => setPinned([])}>
              Clear pins
            </button>
          </div>
          <div className={`compare-tray-grid ${pinnedModels.length === 1 ? 'single' : ''}`}>
            {pinnedModels.map((m) => (
              <div key={m.name} className="card compare-col">
                <div className="model-name">{m.short}</div>
                <div className="kv"><span>Approach</span><span>{m.load} compute</span></div>
                <p className="compare-col-p"><b>Why it fits:</b> {m.why}</p>
                <p className="compare-col-p"><b>Weakness on n = 55:</b> {m.weakness}</p>
                <p className="compare-col-p"><b>Score:</b> awaiting results — nothing to compare yet.</p>
              </div>
            ))}
            {pinnedModels.length === 1 && (
              <div className="card compare-col compare-empty">
                <p>Pin one more model above and it will appear here, side by side.</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
