#!/usr/bin/env python3
"""Plot 2x2 binary confusion matrices from metrics_v2.json.

train_v2.py plots the 3-class matrices and the neural training curves;
the binary models only get their numbers in the metrics file. This
adds the matching binary figures so the figure set is complete.
Run after train_v2.py finishes.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    m = json.load(open(BASE / "metrics_v2.json"))
    fig_dir = BASE / "figures_v2"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for tag, res in m["atlases"].items():
        for name, r in res.get("binary_models", {}).items():
            if "confusion_matrix_sum" not in r:
                continue
            cm = np.array(r["confusion_matrix_sum"])
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(2))
            ax.set_yticks(range(2))
            ax.set_xticklabels(["HC", "PD"])
            ax.set_yticklabels(["HC", "PD"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{tag} {name} — binary, summed CV confusion "
                         f"matrix (n=55)")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2
                            else "black")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(fig_dir / f"cm_binary_{tag}_{name}.png", dpi=150)
            plt.close(fig)
            print(f"wrote cm_binary_{tag}_{name}.png")
    print("binary figures done")


if __name__ == "__main__":
    main()
