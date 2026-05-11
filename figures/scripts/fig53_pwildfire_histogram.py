"""
Fig 5.3 — P(wildfire) probability histogram
============================================

Section 7 (Results) headline figure for FireSight-IR.
Distribution of the model's wildfire-probability score P(wildfire) on the
2023 validation set, stratified by true class. The non-overlap between
the wildfire and false-alarm distributions is the geometric proof of
FA AUC = 1.0000.

Reads: data/predictions/val_predictions.npz
Saves: figures/publication/fig53_pwildfire_histogram.{pdf,png}
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/content/drive/MyDrive/firesight-ir')
PREDS_DIR  = BASE_DIR / 'data/predictions'
OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Non-fire', 'Wildfire', 'False-alarm']
DECISION_THRESHOLD = 0.5
N_BINS = 60

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

PALETTE = {
    'wildfire':    '#d65a3a',
    'false_alarm': '#3a6ea5',
    'non_fire':    '#888888',
    'threshold':   '#222222',
    'grid':        '#dddddd',
}


# ─── LOAD ────────────────────────────────────────────────────────────────────
def load_val():
    p = PREDS_DIR / 'val_predictions.npz'
    if not p.exists():
        raise FileNotFoundError(f'Predictions not found: {p}')
    d = np.load(p)
    return d['labels'], d['probs']


# ─── PLOT ────────────────────────────────────────────────────────────────────
def main():
    labels, probs = load_val()
    p_wf = probs[:, 1]   # P(wildfire) for every pixel

    p_wf_nf = p_wf[labels == 0]
    p_wf_wf = p_wf[labels == 1]
    p_wf_fa = p_wf[labels == 2]

    # Headline statistics
    fa_leak = (p_wf_fa > DECISION_THRESHOLD).mean() * 100
    wf_miss = (p_wf_wf < DECISION_THRESHOLD).mean() * 100
    med_wf  = float(np.median(p_wf_wf))
    med_fa  = float(np.median(p_wf_fa))
    med_nf  = float(np.median(p_wf_nf))

    print(f'[INFO] P(wildfire) medians:')
    print(f'         True wildfire    median={med_wf:.4f}  n={len(p_wf_wf):,}')
    print(f'         True false-alarm median={med_fa:.4f}  n={len(p_wf_fa):,}')
    print(f'         True non-fire    median={med_nf:.4f}  n={len(p_wf_nf):,}')
    print(f'[INFO] FA leakage   (P(wf) > 0.5):  {fa_leak:.3f}%')
    print(f'[INFO] WF miss rate (P(wf) < 0.5):  {wf_miss:.3f}%')

    bins = np.linspace(0.0, 1.0, N_BINS + 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    # Order: draw NF first (largest, faintest), then FA, then WF on top
    ax.hist(p_wf_nf, bins=bins, color=PALETTE['non_fire'], alpha=0.55,
            edgecolor='white', linewidth=0.4, density=True, zorder=2,
            label=f'True non-fire    n = {len(p_wf_nf):,}   '
                  f'median = {med_nf:.4f}')
    ax.hist(p_wf_fa, bins=bins, color=PALETTE['false_alarm'], alpha=0.85,
            edgecolor='white', linewidth=0.4, density=True, zorder=3,
            label=f'True false-alarm  n = {len(p_wf_fa):,}   '
                  f'median = {med_fa:.4f}')
    ax.hist(p_wf_wf, bins=bins, color=PALETTE['wildfire'], alpha=0.85,
            edgecolor='white', linewidth=0.4, density=True, zorder=3,
            label=f'True wildfire     n = {len(p_wf_wf):,}   '
                  f'median = {med_wf:.4f}')

    # Decision threshold
    ax.axvline(DECISION_THRESHOLD, color=PALETTE['threshold'],
               ls='--', lw=1.2, alpha=0.8, zorder=4)

    # Annotations: FA leakage region + WF miss region
    ymax = ax.get_ylim()[1]
    ax.annotate(f'Decision threshold = {DECISION_THRESHOLD}',
                xy=(DECISION_THRESHOLD, ymax * 0.92),
                xytext=(DECISION_THRESHOLD - 0.02, ymax * 0.92),
                fontsize=9, ha='right', va='center',
                color=PALETTE['threshold'])
    ax.text(0.02, ymax * 0.78,
            f'FA leakage  (P(wf) > 0.5):\n  {fa_leak:.3f}%   '
            f'({int(np.round(fa_leak/100 * len(p_wf_fa)))} of {len(p_wf_fa):,} '
            f'FA pixels)',
            fontsize=8.5, ha='left', va='top',
            bbox=dict(facecolor='#f2f6fc', edgecolor=PALETTE['false_alarm'],
                      boxstyle='round,pad=0.4', linewidth=0.8))
    ax.text(0.98, ymax * 0.78,
            f'WF miss rate  (P(wf) < 0.5):\n  {wf_miss:.3f}%   '
            f'({int(np.round(wf_miss/100 * len(p_wf_wf)))} of {len(p_wf_wf):,} '
            f'WF pixels)',
            fontsize=8.5, ha='right', va='top',
            bbox=dict(facecolor='#fff5f2', edgecolor=PALETTE['wildfire'],
                      boxstyle='round,pad=0.4', linewidth=0.8))

    # Cosmetics
    ax.set_xlim(-0.005, 1.005)
    ax.set_xlabel('Predicted wildfire probability  P(wildfire)')
    ax.set_ylabel('Density')
    ax.grid(True, axis='y', alpha=0.35, color=PALETTE['grid'], zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('FireSight-IR  ·  Wildfire-probability score distribution by true class\n'
                 'Validation set, 2023 (n = '
                 f'{len(p_wf_nf) + len(p_wf_wf) + len(p_wf_fa):,} pixels)',
                 loc='left', fontsize=11)
    ax.legend(loc='upper center', frameon=True, framealpha=0.95,
              edgecolor='#cccccc', ncol=1,
              bbox_to_anchor=(0.5, -0.13))

    fig.tight_layout()
    fig.set_facecolor('white')

    pdf = OUTPUT_DIR / 'fig53_pwildfire_histogram.pdf'
    png = OUTPUT_DIR / 'fig53_pwildfire_histogram.png'
    fig.savefig(pdf, facecolor='white')
    fig.savefig(png, dpi=300, facecolor='white')
    print(f'\nSaved → {pdf}')
    print(f'Saved → {png}')


if __name__ == '__main__':
    main()
