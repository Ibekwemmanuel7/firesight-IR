"""
Fig 5.1 — Confusion matrices (validation + test)
================================================

Section 7 (Results) figure for FireSight-IR.
Two side-by-side confusion matrices:
  · Left  — 2023 held-out validation (76,084 pixels)
  · Right — 2018–2022 test set       (~214,728 pixels)

Each cell shows raw count and row-normalised percentage.

Reads: data/predictions/{val,test}_predictions.npz
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/content/drive/MyDrive/firesight-ir')
PREDS_DIR  = BASE_DIR / 'data/predictions'
OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Non-fire', 'Wildfire', 'False-alarm']

# ─── STYLE ───────────────────────────────────────────────────────────────────
# Reset to default light theme — earlier dark-themed cells in the same
# Colab kernel can leave rcParams in a dark state.
plt.rcdefaults()
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         10,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    # Explicit light theme
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#222222',
    'axes.labelcolor':   '#111111',
    'text.color':        '#111111',
    'xtick.color':       '#111111',
    'ytick.color':       '#111111',
})

# Custom blue colormap — light at low values, deep navy at high
CMAP = LinearSegmentedColormap.from_list(
    'firesight_blue',
    ['#ffffff', '#dde9f4', '#a8c8e1', '#5a8ec4', '#1d4f8c'],
    N=256,
)


# ─── LOAD ────────────────────────────────────────────────────────────────────
def load_split(name):
    p = PREDS_DIR / f'{name}_predictions.npz'
    if not p.exists():
        raise FileNotFoundError(f'Predictions not found: {p}')
    d = np.load(p)
    print(f'[INFO] {name}: preds={d["preds"].shape}  '
          f'labels={d["labels"].shape}  probs={d["probs"].shape}')
    return d['preds'], d['labels'], d['probs']


# ─── PLOT ────────────────────────────────────────────────────────────────────
def draw_cm(ax, preds, labels, title, classes=CLASS_NAMES):
    cm  = confusion_matrix(labels, preds, labels=range(len(classes)))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    im = ax.imshow(cmn, cmap=CMAP, vmin=0, vmax=1, aspect='equal')

    # Cell annotations: count + % (row-normalised)
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = cm[i, j]
            pct   = cmn[i, j]
            text_color = 'white' if pct > 0.55 else '#222222'
            ax.text(j, i - 0.13,
                    f'{count:,}',
                    ha='center', va='center',
                    color=text_color, fontsize=10, fontweight='bold')
            ax.text(j, i + 0.18,
                    f'{pct:6.2%}',
                    ha='center', va='center',
                    color=text_color, fontsize=8.5)

    # Axis labels and ticks
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')

    # Subtle gridlines between cells
    ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)

    # Per-row recall summary on the right
    for i, cls in enumerate(classes):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() else 0
        ax.text(len(classes) - 0.45, i,
                f'  recall\n  {recall:.1%}',
                ha='left', va='center', fontsize=8, color='#444444')

    overall_acc = cm.diagonal().sum() / cm.sum()
    ax.set_title(f'{title}\nn = {cm.sum():,}  ·  overall accuracy = {overall_acc:.2%}',
                 loc='left', fontsize=10.5)

    return cm, im


def main():
    val_preds,  val_labels,  _ = load_split('val')
    test_preds, test_labels, _ = load_split('test')

    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.subplots_adjust(left=0.06, right=0.92, wspace=0.35, top=0.84, bottom=0.10)

    cm_val,  im_val  = draw_cm(ax_val,  val_preds,  val_labels,
                                'A.  Validation — 2023 held-out')
    cm_test, im_test = draw_cm(ax_test, test_preds, test_labels,
                                'B.  Test — 2018–2022 (random 20%)')

    # Shared colorbar on the right (representing row-normalized fraction)
    cax = fig.add_axes([0.94, 0.18, 0.012, 0.62])
    cb = fig.colorbar(im_test, cax=cax)
    cb.set_label('Row-normalised fraction', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle('FireSight-IR  ·  Confusion matrices',
                 fontsize=12, fontweight='bold', y=0.97)

    fig.set_facecolor('white')
    pdf = OUTPUT_DIR / 'fig51_confusion_matrices.pdf'
    png = OUTPUT_DIR / 'fig51_confusion_matrices.png'
    fig.savefig(pdf, facecolor='white')
    fig.savefig(png, dpi=300, facecolor='white')
    print(f'\nSaved → {pdf}')
    print(f'Saved → {png}')


if __name__ == '__main__':
    main()
