"""
Fig 5.2 — ROC curves (per class)
=================================

Section 7 (Results) figure for FireSight-IR.
One-vs-rest ROC curves for each of the three classes on the 2023
held-out validation set and the 2018–2022 test set.

Reads: data/predictions/{val,test}_predictions.npz
Saves: figures/publication/fig52_roc_curves.{pdf,png}
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/content/drive/MyDrive/firesight-ir')
PREDS_DIR  = BASE_DIR / 'data/predictions'
OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Non-fire', 'Wildfire', 'False-alarm']
CLASS_COLORS = {
    'Non-fire':    '#888888',
    'Wildfire':    '#d65a3a',
    'False-alarm': '#3a6ea5',
}

# ─── STYLE ───────────────────────────────────────────────────────────────────
plt.rcdefaults()
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         10,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'legend.fontsize':    9,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#222222',
    'axes.labelcolor':   '#111111',
    'text.color':        '#111111',
    'xtick.color':       '#111111',
    'ytick.color':       '#111111',
})


# ─── LOAD ────────────────────────────────────────────────────────────────────
def load_split(name):
    p = PREDS_DIR / f'{name}_predictions.npz'
    if not p.exists():
        raise FileNotFoundError(f'Predictions not found: {p}')
    d = np.load(p)
    print(f'[INFO] {name}: labels={d["labels"].shape}  probs={d["probs"].shape}')
    return d['labels'], d['probs']


# ─── PLOT ────────────────────────────────────────────────────────────────────
def draw_roc(ax, labels, probs, title, classes=CLASS_NAMES):
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color='#999999', lw=0.9, ls='--', zorder=1,
            label='Random (AUC = 0.5)')

    # Per-class one-vs-rest ROC
    for i, cls in enumerate(classes):
        y_true   = (labels == i).astype(int)
        y_score  = probs[:, i]
        if y_true.sum() == 0:
            print(f'[WARN] No positives for class {cls} in {title}')
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=1.8, zorder=3,
                label=f'{cls}   AUC = {auc:.4f}')

    ax.set_xlim(-0.005, 1.005); ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='#dddddd')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
              edgecolor='#cccccc')
    ax.set_title(title, loc='left', fontsize=11)


def main():
    val_labels,  val_probs  = load_split('val')
    test_labels, test_probs = load_split('test')

    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(12.0, 5.6))

    draw_roc(ax_val,  val_labels,  val_probs,
             'A.  Validation — 2023 held-out')
    draw_roc(ax_test, test_labels, test_probs,
             'B.  Test — 2018–2022 (random 20%)')

    fig.suptitle('FireSight-IR  ·  Per-class ROC curves (one-vs-rest)',
                 fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.set_facecolor('white')

    pdf = OUTPUT_DIR / 'fig52_roc_curves.pdf'
    png = OUTPUT_DIR / 'fig52_roc_curves.png'
    fig.savefig(pdf, facecolor='white')
    fig.savefig(png, dpi=300, facecolor='white')
    print(f'\nSaved → {pdf}')
    print(f'Saved → {png}')


if __name__ == '__main__':
    main()
