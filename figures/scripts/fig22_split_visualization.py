"""
Fig 2.2 — Train/validation/test split visualization
====================================================

Section 4.6 figure showing the temporal-holdout split design:
  · 2018–2022 → 80/20 stratified train/test split
  · 2023      → fully held-out validation year

Stacked bar chart, one bar per year, segments coloured by split assignment.
Class composition (wildfire / false-alarm) is shown via subtle hatching on
the false-alarm portion so the imbalance is visible.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path('/content/drive/MyDrive/firesight-ir')
FP_DIR   = BASE_DIR / 'data/processed/viirs_fp_srf_v2'
FP_GLOB  = 'viirs_fp_srf_v2_*.parquet'
SPLIT_DIR = BASE_DIR / 'data/splits'  # optional, used if available

OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_YEAR  = 2023
TRAIN_PCT = 0.80     # 80/20 stratified split within 2018–2022

# ─── STYLE ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':       10,
    'axes.labelsize':  10,
    'axes.titlesize':  11,
    'legend.fontsize':  9,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

PALETTE = {
    'train':       '#3a6ea5',
    'test':        '#7eaad6',
    'val':         '#d65a3a',
    'fa_hatch':    '#ffffff',
    'grid':        '#e0e0e0',
    'annot':       '#333333',
}

# ─── LOAD ────────────────────────────────────────────────────────────────────
def load_year_class_counts():
    """Returns DataFrame: rows=year, cols=['wildfire', 'false_alarm', 'total']."""
    files = sorted(FP_DIR.glob(FP_GLOB))
    if not files:
        raise FileNotFoundError(f'No files in {FP_DIR}')
    rows = []
    for f in files:
        # Year from filename suffix, e.g. ..._2020.parquet
        year = int(f.stem.split('_')[-1])
        df = pd.read_parquet(f, columns=['label'])
        wf = int((df['label'] == 1).sum())
        fa = int((df['label'] == 2).sum())
        rows.append({'year': year, 'wildfire': wf, 'false_alarm': fa,
                     'total': wf + fa})
    out = pd.DataFrame(rows).set_index('year').sort_index()
    print('[INFO] Per-year counts (wildfire / false_alarm / total):')
    print(out.to_string())
    print(f'[INFO] Grand total: {out["total"].sum():,}')
    return out


def split_assignments(yc):
    """Return DataFrame with columns train/test/val per year, plus fa fraction
    of each split for hatching."""
    rows = []
    for year, r in yc.iterrows():
        if year == VAL_YEAR:
            train = test = 0
            val   = r['total']
            train_fa = test_fa = 0
            val_fa = r['false_alarm']
        else:
            train = int(round(r['total'] * TRAIN_PCT))
            test  = r['total'] - train
            val   = 0
            # Stratified split preserves class ratio within year
            train_fa = int(round(r['false_alarm'] * TRAIN_PCT))
            test_fa  = r['false_alarm'] - train_fa
            val_fa   = 0
        rows.append({'year': year,
                     'train': train, 'test': test, 'val': val,
                     'train_fa': train_fa, 'test_fa': test_fa, 'val_fa': val_fa})
    return pd.DataFrame(rows).set_index('year')


# ─── PLOT ────────────────────────────────────────────────────────────────────
def main():
    yc = load_year_class_counts()
    sp = split_assignments(yc)

    years = sp.index.tolist()
    x = np.arange(len(years))
    bar_width = 0.7

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    # Stacked bars: train (bottom), test (middle), val (top — only 2023)
    train_h = sp['train'].values
    test_h  = sp['test'].values
    val_h   = sp['val'].values

    ax.bar(x, train_h, width=bar_width, color=PALETTE['train'],
           edgecolor='white', linewidth=0.6, label='Train (80%)', zorder=3)
    ax.bar(x, test_h, bottom=train_h, width=bar_width, color=PALETTE['test'],
           edgecolor='white', linewidth=0.6, label='Test (20%)', zorder=3)
    ax.bar(x, val_h, bottom=train_h + test_h, width=bar_width,
           color=PALETTE['val'], edgecolor='white', linewidth=0.6,
           label='Validation (2023 holdout)', zorder=3)

    # Total count annotation above each bar
    totals = train_h + test_h + val_h
    for xi, total in zip(x, totals):
        ax.text(xi, total * 1.02, f'{total:,}', ha='center', va='bottom',
                fontsize=9, color=PALETTE['annot'], fontweight='bold')

    # Per-segment count annotation (only for non-trivial sizes)
    for xi, t, te, v in zip(x, train_h, test_h, val_h):
        if t > 30_000:
            ax.text(xi, t / 2, f'{t:,}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if te > 30_000:
            ax.text(xi, t + te / 2, f'{te:,}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if v > 30_000:
            ax.text(xi, t + te + v / 2, f'{v:,}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    # Class-summary annotation in upper-left corner
    grand = sp[['train', 'test', 'val']].sum()
    grand_fa = sp[['train_fa', 'test_fa', 'val_fa']].sum()
    summary = (
        f'Train      :  {grand["train"]:>9,d}    (FA: {grand_fa["train_fa"]:>5,d})\n'
        f'Test       :  {grand["test"]:>9,d}    (FA: {grand_fa["test_fa"]:>5,d})\n'
        f'Validation :  {grand["val"]:>9,d}    (FA: {grand_fa["val_fa"]:>5,d})'
    )
    ax.text(0.018, 0.97, summary, transform=ax.transAxes,
            fontsize=8.5, family='monospace', va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='#cccccc',
                      boxstyle='round,pad=0.5'))

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel('Fire season year')
    ax.set_ylabel('Fire pixels')
    ax.set_title('Train / test / validation split — strict temporal holdout\n'
                 '2018–2022 stratified 80/20  ·  2023 fully held out',
                 loc='left', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, color=PALETTE['grid'], zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, totals.max() * 1.13)

    # Vertical separator between train years and validation year
    val_ix = years.index(VAL_YEAR) if VAL_YEAR in years else None
    if val_ix is not None and val_ix > 0:
        ax.axvline(val_ix - 0.5, color='#888888', ls='--', lw=0.8, alpha=0.7,
                   zorder=2)
        ax.text(val_ix - 0.5, totals.max() * 1.10,
                ' Held-out year →', fontsize=8, color='#666666',
                ha='left', va='center', style='italic')

    ax.legend(loc='upper right', frameon=True, framealpha=0.95,
              edgecolor='#cccccc', ncol=3, bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()

    pdf = OUTPUT_DIR / 'fig22_split_visualization.pdf'
    png = OUTPUT_DIR / 'fig22_split_visualization.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f'\nSaved → {pdf}')
    print(f'Saved → {png}')


if __name__ == '__main__':
    main()
