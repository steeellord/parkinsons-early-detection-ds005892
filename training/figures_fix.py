#!/usr/bin/env python3
"""Regenerate confusion-matrix figures from metrics.json with distinct
filenames per task (train.py's two save_figures calls overwrote each other:
binary CMs replaced the 3-class CMs under the same filenames)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
d = json.load(open(BASE / "metrics.json"))
fig_dir = BASE / "figures"
fig_dir.mkdir(exist_ok=True)


def plot_cm(cm, classes, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


for name, r in d["models"].items():
    if "confusion_matrix_sum" in r:
        plot_cm(r["confusion_matrix_sum"], ["HC", "PD-NC", "PD-MCI"],
                f"{name} — 3-class summed CV confusion matrix (n=55)",
                fig_dir / f"cm_3class_{name}.png")
for name, r in d["binary"]["models"].items():
    if "confusion_matrix_sum" in r:
        plot_cm(r["confusion_matrix_sum"], ["HC", "PD"],
                f"{name} — binary HC-vs-PD summed CV confusion matrix (n=55)",
                fig_dir / f"cm_binary_{name}.png")
print("Wrote:", sorted(p.name for p in fig_dir.glob("cm_*.png")))
