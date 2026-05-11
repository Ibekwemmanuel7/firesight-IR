"""
Fig 2.1 — BTD distribution by class
====================================

Section 4.5 label-construction figure.
KDE of BTD = BT_I4 − BT_I5 for wildfire vs false-alarm pixels, with
vertical reference lines at the two physics thresholds:
  · BTD = 10 K  →  thermal realism loss boundary (Section 5.3)
  · BTD = 20 K  →  false-alarm label gate (Section 4.5)

Medians annotated for each class. Light theme, publication style.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path('/content/drive/MyDrive/firesight-ir')
FP_DIR   = BASE_DIR / 'data/processed/viirs_fp_srf_v2'
FP_GLOB  = 'viirs_fp_srf_v2_*.parquet'

OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Physics thresholds
BTD_THERMAL_REALISM_K = 10.0
BTD_FA_LABEL_GATE_K   = 20.0

# Label encoding — script auto-detects strings vs ints
LABEL_WILDFIRE   = ('wildfire',   1)
LABEL_FALSE_ALARM = ('false_alarm', 'false-alarm', 'fa', 2)
LABEL_NON_FIRE   = ('non_fire', 'non-fire', 'nf', 0)

# Plot range
BTD_MIN, BTD_MAX = -5.0, 80.0
KDE_BANDWIDTH    = 0.20      # gaussian_kde Scott factor multiplier
SUBSAMPLE        = 200_000   # cap per-class sample for KDE fit speed

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
    'wildfire':    '#d65a3a',
    'false_alarm': '#4a7eb5',
    'non_fire':    '#999999',
    'threshold':   '#222222',
    'grid':        '#e0e0e0',
}

# ─── LOAD ────────────────────────────────────────────────────────────────────
def load_btd_by_label():
    files = sorted(FP_DIR.glob(FP_GLOB))
    if not files:
        raise FileNotFoundError(f'No files in {FP_DIR}')
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=['BTD', 'label'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f'[INFO] Loaded {len(df):,} pixels')
    print(f'[INFO] Label value counts:\n{df["label"].value_counts()}')
    print(f'[INFO] BTD range: [{df["BTD"].min():.1f}, {df["BTD"].max():.1f}] K')
    return df


def split_classes(df):
    """Return three (BTD,) arrays: wildfire, false_alarm, non_fire."""
    def matches(lbl_value, candidates):
        for c in candidates:
            if isinstance(c, str) and isinstance(lbl_value, str):
                if c == lbl_value.lower(): return True
            elif lbl_value == c:
                return True
        return False

    # Vectorise using set membership for speed when label is string-typed
    def mask(candidates):
        return df['label'].apply(lambda v: matches(v, candidates))

    wf = df.loc[mask(LABEL_WILDFIRE), 'BTD'].dropna().values
    fa = df.loc[mask(LABEL_FALSE_ALARM), 'BTD'].dropna().values
    nf = df.loc[mask(LABEL_NON_FIRE), 'BTD'].dropna().values
    print(f'[INFO] Class counts: wildfire={len(wf):,}  '
          f'false_alarm={len(fa):,}  non_fire={len(nf):,}')
    return wf, fa, nf


# ─── KDE ─────────────────────────────────────────────────────────────────────
def kde(values, x_grid, bw_factor=KDE_BANDWIDTH):
    if len(values) == 0:
        return np.zeros_like(x_grid)
    if len(values) > SUBSAMPLE:
        rng = np.random.default_rng(42)
        values = rng.choice(values, SUBSAMPLE, replace=False)
    k = gaussian_kde(values)
    k.set_bandwidth(k.factor * bw_factor)
    return k(x_grid)


# ─── PLOT ────────────────────────────────────────────────────────────────────
def main():
    df = load_btd_by_label()
    wf, fa, nf = split_classes(df)

    if len(wf) == 0 or len(fa) == 0:
        raise RuntimeError(
            f'Could not find wildfire/false-alarm classes. '
            f'Check label encoding. Sample values: '
            f'{df["label"].value_counts().head().to_dict()}')

    x = np.linspace(BTD_MIN, BTD_MAX, 600)
    den_wf = kde(wf, x)
    den_fa = kde(fa, x)
    # Normalise to peak=1 for visual clarity (densities differ by orders of magnitude)
    den_wf_n = den_wf / den_wf.max()
    den_fa_n = den_fa / den_fa.max()

    med_wf = float(np.median(wf))
    med_fa = float(np.median(fa))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    # KDEs
    ax.fill_between(x, 0, den_fa_n, color=PALETTE['false_alarm'],
                    alpha=0.35, lw=0, zorder=2)
    ax.plot(x, den_fa_n, color=PALETTE['false_alarm'], lw=1.6, zorder=3,
            label=f'False-alarm (n={len(fa):,})')
    ax.fill_between(x, 0, den_wf_n, color=PALETTE['wildfire'],
                    alpha=0.35, lw=0, zorder=2)
    ax.plot(x, den_wf_n, color=PALETTE['wildfire'], lw=1.6, zorder=3,
            label=f'Wildfire  (n={len(wf):,})')

    # Median lines
    ax.axvline(med_wf, color=PALETTE['wildfire'], ls=':', lw=1.2,
               alpha=0.85, zorder=4)
    ax.axvline(med_fa, color=PALETTE['false_alarm'], ls=':', lw=1.2,
               alpha=0.85, zorder=4)

    # Threshold lines
    ax.axvline(BTD_THERMAL_REALISM_K, color=PALETTE['threshold'],
               ls='--', lw=1.0, alpha=0.7, zorder=4)
    ax.axvline(BTD_FA_LABEL_GATE_K, color=PALETTE['threshold'],
               ls='-.', lw=1.0, alpha=0.7, zorder=4)

    # Annotations
    ymax = 1.05
    ax.annotate(f'BTD = {BTD_THERMAL_REALISM_K:.0f} K\nthermal realism\n(loss term)',
                xy=(BTD_THERMAL_REALISM_K, 0.42), xytext=(9.5, 0.42),
                fontsize=8, ha='right', va='center', color=PALETTE['threshold'])
    ax.annotate(f'BTD = {BTD_FA_LABEL_GATE_K:.0f} K\nFA label gate',
                xy=(BTD_FA_LABEL_GATE_K, 0.30), xytext=(19.5, 0.30),
                fontsize=8, ha='right', va='center', color=PALETTE['threshold'])
    ax.annotate(f'median = {med_fa:.1f} K',
                xy=(med_fa, 0.91), xytext=(med_fa - 0.5, 0.91),
                fontsize=8, ha='right', va='center',
                color=PALETTE['false_alarm'])
    ax.annotate(f'median = {med_wf:.1f} K',
                xy=(med_wf, 0.91), xytext=(med_wf + 0.5, 0.91),
                fontsize=8, ha='left', va='center',
                color=PALETTE['wildfire'])

    # Class separation arrow
    sep = med_wf - med_fa
    ax.annotate('', xy=(med_wf, 0.05), xytext=(med_fa, 0.05),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=0.9))
    ax.text((med_wf + med_fa) / 2, 0.10,
            f'Δmedian = {sep:.1f} K',
            fontsize=8, ha='center', color='#444444')

    # Cosmetics
    ax.set_xlim(BTD_MIN, BTD_MAX)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('BTD = BT$_{I4}$ − BT$_{I5}$  (K)')
    ax.set_ylabel('Normalised density')
    ax.set_title('BTD separation between wildfire and false-alarm pixels\n'
                 f'2018–2023 · n={len(wf)+len(fa):,} pixels (wildfire + false-alarm)',
                 loc='left', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, color=PALETTE['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95,
              edgecolor='#cccccc')

    fig.tight_layout()

    pdf = OUTPUT_DIR / 'fig21_btd_distribution.pdf'
    png = OUTPUT_DIR / 'fig21_btd_distribution.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f'\nSaved → {pdf}')
    print(f'Saved → {png}')


if __name__ == '__main__':
    main()
