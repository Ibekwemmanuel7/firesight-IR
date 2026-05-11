"""
Fig 6.1 — Ablation metric comparison
=====================================

Section 8 (Ablation) figure for FireSight-IR.
2x2 grid of bar charts comparing four model variants on four key metrics:
  · Validation accuracy
  · Wildfire recall
  · False-alarm precision  ← the headline panel
  · False-alarm AUC

Reads: models/ablations/ablations/all_ablation_results.json
       (or models/ablations/all_ablation_results.json — script tries both)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/content/drive/MyDrive/firesight-ir')
ABL_PATHS  = [
    BASE_DIR / 'models/ablations/ablations/all_ablation_results.json',
    BASE_DIR / 'models/ablations/all_ablation_results.json',
]
OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variant order (left → right) and color
VARIANT_ORDER = [
    ('Full model',       '#3a6ea5'),
    ('No physics loss',  '#5fa055'),
    ('No ERA5 (ATM)',    '#b3892b'),
    ('No surface (SRF)', '#c64a3a'),
]

# Metrics to plot — (key in JSON, display name, y-axis range, format string)
METRICS = [
    ('val_acc',     'Validation accuracy',  (0.78, 0.99), '{:.2%}'),
    ('wf_recall',   'Wildfire recall',      (0.74, 0.99), '{:.2%}'),
    ('fa_precision','False-alarm precision',(0.30, 1.00), '{:.2%}'),
    ('fa_auc',      'False-alarm AUC',      (0.96, 1.005), '{:.4f}'),
]

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
def load_results():
    for p in ABL_PATHS:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            print(f'[INFO] Loaded ablation results from {p}')
            return data
    raise FileNotFoundError(
        f'all_ablation_results.json not found in any of:\n  '
        + '\n  '.join(str(p) for p in ABL_PATHS))


# ─── PLOT ────────────────────────────────────────────────────────────────────
def main():
    results = load_results()
    by_name = {r['variant']: r for r in results}

    # Verify all expected variants are present
    missing = [name for name, _ in VARIANT_ORDER if name not in by_name]
    if missing:
        print(f'[WARN] Missing variants: {missing}')
        print(f'[INFO] Available: {list(by_name.keys())}')

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.10,
                        wspace=0.22, hspace=0.32)

    panel_letter = ['A', 'B', 'C', 'D']
    for ax, (mkey, mname, ylim, fmt), letter in zip(axes.flat, METRICS, panel_letter):
        names = [name for name, _ in VARIANT_ORDER if name in by_name]
        colors = [c for name, c in VARIANT_ORDER if name in by_name]
        values = [by_name[name].get(mkey, np.nan) for name in names]

        x = np.arange(len(names))
        bars = ax.bar(x, values, color=colors, width=0.65,
                      edgecolor='white', linewidth=0.8, zorder=3)

        # Annotate bar values above each bar
        for xi, v in zip(x, values):
            if np.isnan(v):
                continue
            label = fmt.format(v)
            ax.text(xi, v + (ylim[1] - ylim[0]) * 0.018, label,
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#111111')

        # Highlight the full-model baseline with a dashed reference line
        if 'Full model' in by_name:
            full_val = by_name['Full model'].get(mkey, np.nan)
            ax.axhline(full_val, color='#888888', ls='--', lw=0.8,
                       alpha=0.7, zorder=1)
            ax.text(len(names) - 0.5, full_val,
                    f'  baseline  ({fmt.format(full_val)})',
                    fontsize=7.5, color='#666666', va='center', ha='left')

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' (', '\n(') for n in names],
                           rotation=0, fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel(mname)
        ax.grid(True, axis='y', alpha=0.3, color='#dddddd', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'{letter}.  {mname}', loc='left', fontsize=10.5)

    fig.suptitle('FireSight-IR  ·  Ablation comparison  ·  '
                 '2023 validation metrics',
                 fontsize=12, fontweight='bold', y=0.97)
    fig.set_facecolor('white')

    pdf = OUTPUT_DIR / 'fig61_ablation_comparison.pdf'
    png = OUTPUT_DIR / 'fig61_ablation_comparison.png'
    fig.savefig(pdf, facecolor='white')
    fig.savefig(png, dpi=300, facecolor='white')
    print(f'\nSaved → {pdf}')
    print(f'Saved → {png}')


if __name__ == '__main__':
    main()
