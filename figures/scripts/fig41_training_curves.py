"""
Fig 4.1 — Training curves and loss-component breakdown
=======================================================

Section 6 (Training) figure for FireSight-IR.

Top panel:
  · Training and validation loss across 30 epochs (left y-axis)
  · Validation accuracy on twin right y-axis
  · Best-epoch marker (lowest val_loss)

Bottom panel:
  · Stacked-area decomposition of loss components: CE, Beer-Lambert,
    Dynamic-Range, Thermal-Realism. Shows how the physics-informed
    regularization terms behave through training.

Reads: models/training_log.json
Saves: figures/publication/fig41_training_curves.{pdf,png}
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path('/content/drive/MyDrive/firesight-ir')
LOG_PATH = BASE_DIR / 'models/training_log.json'
OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lambda weights from training (for displaying weighted contributions)
LAMBDAS = {'ce': 1.0, 'bl': 0.10, 'dr': 0.05, 'th': 0.05}

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
    'train_loss': '#3a6ea5',
    'val_loss':   '#d65a3a',
    'val_acc':    '#5fa055',
    'best':       '#222222',
    'ce':         '#888888',
    'bl':         '#3a6ea5',  # Beer-Lambert — atmosphere blue
    'dr':         '#d65a3a',  # Dynamic-range — fire orange
    'th':         '#8e5fa8',  # Thermal-realism — purple
    'grid':       '#e0e0e0',
}

LABEL = {
    'ce': 'Weighted cross-entropy ($\mathcal{L}_{\mathrm{CE}}$)',
    'bl': r'Beer–Lambert  ($\lambda_{\mathrm{BL}}\,\mathcal{L}_{\mathrm{BL}}$)',
    'dr': r'Dynamic-range  ($\lambda_{\mathrm{DR}}\,\mathcal{L}_{\mathrm{DR}}$)',
    'th': r'Thermal-realism  ($\lambda_{\mathrm{TH}}\,\mathcal{L}_{\mathrm{TH}}$)',
}


# ─── LOAD ────────────────────────────────────────────────────────────────────
def load_log(path):
    """Reconstruct the full training trajectory from a log that may contain
    multiple auto-resume sessions. We merge all entries by epoch number and
    keep the LATEST entry for each epoch — auto-resume overwrites earlier
    checkpoints with the most recent values, so this gives the canonical
    end-state trajectory."""
    if not path.exists():
        raise FileNotFoundError(f'Training log not found: {path}')
    with open(path) as f:
        log = json.load(f)
    if not isinstance(log, list):
        raise ValueError(f'Expected list of epoch dicts; got {type(log)}')

    # Diagnostic: identify the runs in the raw log
    runs = []
    cur = []
    for r in log:
        if cur and r['epoch'] < cur[-1]['epoch']:
            runs.append(cur); cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    print(f'[INFO] {len(log)} log entries split into {len(runs)} run(s):')
    for i, r in enumerate(runs, 1):
        epochs = [x['epoch'] for x in r]
        best_loss = min(x['val_loss'] for x in r)
        print(f'         Run {i}: {len(r)} epochs ({min(epochs)}–{max(epochs)}), '
              f'best val_loss={best_loss:.4f}')

    # Merge by epoch — last entry wins (auto-resume overwrites earlier values)
    seen = {}
    for r in log:
        seen[r['epoch']] = r
    merged = [seen[e] for e in sorted(seen)]
    print(f'[INFO] Merged into {len(merged)} unique epochs '
          f'({merged[0]["epoch"]}–{merged[-1]["epoch"]})')
    return merged


# ─── PLOT ────────────────────────────────────────────────────────────────────
def main():
    run = load_log(LOG_PATH)

    epochs = np.array([r['epoch'] for r in run])
    train_loss = np.array([r['train_loss']   for r in run])
    val_loss   = np.array([r['val_loss']     for r in run])
    val_acc    = np.array([r['val_acc']      for r in run]) * 100  # → percent

    # Loss-component breakdown (use λ-weighted values so they sum ≈ total)
    comps = {k: np.array([r['loss_components'].get(k, 0.0) * LAMBDAS[k]
                          for r in run])
             for k in ('ce', 'bl', 'dr', 'th')}

    best_idx = int(np.argmin(val_loss))
    best_epoch = int(epochs[best_idx])
    best_val_loss = float(val_loss[best_idx])
    best_val_acc  = float(val_acc[best_idx])
    print(f'[INFO] Best epoch: {best_epoch}  '
          f'val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.2f}%')

    # ─── FIGURE LAYOUT ────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.5, 8.0), sharex=True,
        gridspec_kw={'height_ratios': [1.1, 1.0], 'hspace': 0.20})

    # ─── TOP PANEL — losses + accuracy ────────────────────────────────────────
    ax_top.plot(epochs, train_loss, color=PALETTE['train_loss'],
                lw=1.8, label='Train loss', zorder=3)
    ax_top.plot(epochs, val_loss, color=PALETTE['val_loss'],
                lw=1.8, label='Validation loss', zorder=3)

    # Best-epoch marker on val_loss
    ax_top.scatter(best_epoch, best_val_loss, color=PALETTE['best'],
                   s=70, zorder=5, marker='o',
                   facecolor='white', edgecolor=PALETTE['best'], linewidth=1.6)
    ax_top.annotate(f'Best epoch {best_epoch}\nval_loss = {best_val_loss:.4f}',
                    xy=(best_epoch, best_val_loss),
                    xytext=(best_epoch + 1.5, best_val_loss + 0.10),
                    fontsize=9, ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='->', color=PALETTE['best'],
                                    lw=1.0, shrinkA=4, shrinkB=4))

    ax_top.set_ylabel('Loss', color=PALETTE['train_loss'])
    ax_top.tick_params(axis='y', labelcolor=PALETTE['train_loss'])
    ax_top.grid(True, alpha=0.3, color=PALETTE['grid'])
    ax_top.spines['top'].set_visible(False)

    # Twin axis — validation accuracy
    ax_acc = ax_top.twinx()
    ax_acc.plot(epochs, val_acc, color=PALETTE['val_acc'],
                lw=1.6, ls='--', label='Validation accuracy', zorder=2)
    ax_acc.set_ylabel('Validation accuracy (%)', color=PALETTE['val_acc'])
    ax_acc.tick_params(axis='y', labelcolor=PALETTE['val_acc'])
    ax_acc.set_ylim(0, 100)
    ax_acc.spines['top'].set_visible(False)

    # Combined legend
    lines = (ax_top.get_lines() + ax_acc.get_lines())
    labels = [l.get_label() for l in lines]
    ax_top.legend(lines, labels, loc='upper right', frameon=True,
                  framealpha=0.95, edgecolor='#cccccc')

    ax_top.set_title('A.  Loss and accuracy during training',
                     loc='left', fontsize=11)

    # ─── BOTTOM PANEL — loss-component stacked area ───────────────────────────
    keys = ['ce', 'bl', 'dr', 'th']
    stack = np.stack([comps[k] for k in keys])
    ax_bot.stackplot(epochs, stack,
                     colors=[PALETTE[k] for k in keys],
                     alpha=0.80, edgecolor='white', linewidth=0.6,
                     labels=[LABEL[k] for k in keys])

    ax_bot.set_xlabel('Epoch')
    ax_bot.set_ylabel('Weighted loss contribution')
    ax_bot.set_xlim(epochs.min(), epochs.max())
    ax_bot.set_ylim(0, None)
    ax_bot.grid(True, alpha=0.3, color=PALETTE['grid'], axis='y')
    ax_bot.spines['top'].set_visible(False)
    ax_bot.spines['right'].set_visible(False)
    ax_bot.legend(loc='upper right', frameon=True, framealpha=0.95,
                  edgecolor='#cccccc', ncol=2)
    ax_bot.set_title('B.  Loss-component decomposition  '
                     '(λ-weighted contributions)',
                     loc='left', fontsize=11)

    fig.suptitle('FireSight-IR  ·  Training dynamics over 30 epochs',
                 fontsize=12, fontweight='bold', y=0.995)

    pdf = OUTPUT_DIR / 'fig41_training_curves.pdf'
    png = OUTPUT_DIR / 'fig41_training_curves.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f'\nSaved → {pdf}')
    print(f'Saved → {png}')


if __name__ == '__main__':
    main()
