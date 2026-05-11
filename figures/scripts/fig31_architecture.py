"""
Fig 3.1 — System architecture diagram
======================================

Section 5 (Methodology) figure for FireSight-IR.
Block diagram of the 4-branch fusion PINN.

Total trainable parameters: 202,228.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/content/drive/MyDrive/firesight-ir')
OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── STYLE ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':       10,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

COLORS = {
    'cnn':       '#d65a3a',
    'atm':       '#3a6ea5',
    'srf':       '#5fa055',
    'derived':   '#8e5fa8',
    'concat':    '#666666',
    'fusion':    '#444444',
    'cls_head':  '#c64a3a',
    'phys_head': '#3a6ea5',
    'arrow':     '#888888',
    'loss':      '#222222',
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def block(ax, x, y, w, h, *, edge, fill_alpha=0.10, lw=1.5,
          rows=(),   # tuple of (text, fontsize, weight)
          text_color='#111111'):
    """Rounded box with rows of text auto-distributed vertically."""
    face = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.5',
        linewidth=0, facecolor=edge, alpha=fill_alpha, zorder=2,
    )
    outline = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.5',
        linewidth=lw, edgecolor=edge, facecolor='none', zorder=3,
    )
    ax.add_patch(face); ax.add_patch(outline)

    n = len(rows)
    if n == 0:
        return
    # Distribute rows within useful box height (leave 15% margin top/bottom)
    inner_h = h * 0.70
    if n == 1:
        ys = [y]
    else:
        step = inner_h / (n - 1)
        top = y + inner_h / 2
        ys = [top - i * step for i in range(n)]
    for (text, fs, weight), yy in zip(rows, ys):
        # Allow 'italic' as a row-style shorthand (fontweight='italic' is invalid)
        if weight == 'italic':
            actual_weight, actual_style = 'normal', 'italic'
        else:
            actual_weight, actual_style = weight, 'normal'
        ax.text(x, yy, text, ha='center', va='center',
                fontsize=fs, color=text_color,
                fontweight=actual_weight, fontstyle=actual_style, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.1, mutation=14):
    color = color or COLORS['arrow']
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=mutation,
                        color=color, lw=lw, zorder=1, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def label_dim(ax, x, y, text, color):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=8, style='italic', color=color, zorder=4,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# ─── BUILD FIGURE ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12.5, 13.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 110)
ax.set_aspect('equal'); ax.axis('off')

# Title
ax.text(50, 107,
        'FireSight-IR  ·  Multi-branch physics-informed neural architecture',
        ha='center', va='center', fontsize=13, fontweight='bold')
ax.text(50, 103.5,
        'Total trainable parameters: 202,228',
        ha='center', va='center', fontsize=10, color='#555555')

xs = [12.5, 37.5, 62.5, 87.5]

# ─── INPUTS ──────────────────────────────────────────────────────────────────
INPUT_Y = 95; INPUT_W = 21; INPUT_H = 8.5
block(ax, xs[0], INPUT_Y, INPUT_W, INPUT_H, edge=COLORS['cnn'],
      rows=[('IR patches', 10.5, 'bold'),
            ('BT$_{I4}$, BT$_{I5}$, BTD, mask', 8.5, 'normal'),
            ('4 × 32 × 32', 8, 'normal')])
block(ax, xs[1], INPUT_Y, INPUT_W, INPUT_H, edge=COLORS['atm'],
      rows=[('ERA5 atmosphere', 10.5, 'bold'),
            ('T$_{2m}$, PBL, TCWV, profiles', 8.5, 'normal'),
            ('16 features', 8, 'normal')])
block(ax, xs[2], INPUT_Y, INPUT_W, INPUT_H, edge=COLORS['srf'],
      rows=[('Surface context', 10.5, 'bold'),
            ('MODIS LC, OSM proximity', 8.5, 'normal'),
            ('20 features', 8, 'normal')])
block(ax, xs[3], INPUT_Y, INPUT_W, INPUT_H, edge=COLORS['derived'],
      rows=[('Derived physics', 10.5, 'bold'),
            ('BL proxy, AOD, anomalies', 8.5, 'normal'),
            ('6 features', 8, 'normal')])

# Input → Branch arrows
for x in xs:
    arrow(ax, x, INPUT_Y - INPUT_H/2, x, 84.5)

# ─── BRANCHES ────────────────────────────────────────────────────────────────
BRANCH_Y = 78; BRANCH_W = 21; BRANCH_H = 12.5
block(ax, xs[0], BRANCH_Y, BRANCH_W, BRANCH_H, edge=COLORS['cnn'], lw=1.7,
      rows=[('CNN', 11.5, 'bold'),
            ('3× [Conv → BN → ReLU → MaxPool]', 8, 'normal'),
            ('Global Average Pool', 8, 'normal'),
            ('~82K params', 8, 'italic')])
block(ax, xs[1], BRANCH_Y, BRANCH_W, BRANCH_H, edge=COLORS['atm'], lw=1.7,
      rows=[('MLP-atm', 11.5, 'bold'),
            ('2× Residual Block', 8.5, 'normal'),
            ('16 → 64 → 32', 8.5, 'normal'),
            ('~10K params', 8, 'italic')])
block(ax, xs[2], BRANCH_Y, BRANCH_W, BRANCH_H, edge=COLORS['srf'], lw=1.7,
      rows=[('MLP-srf', 11.5, 'bold'),
            ('2× Residual Block', 8.5, 'normal'),
            ('20 → 64 → 32', 8.5, 'normal'),
            ('~12K params', 8, 'italic')])
block(ax, xs[3], BRANCH_Y, BRANCH_W, BRANCH_H, edge=COLORS['derived'], lw=1.7,
      rows=[('MLP-derived', 11.5, 'bold'),
            ('Linear → BN → ReLU', 8.5, 'normal'),
            ('6 → 32 → 16', 8.5, 'normal'),
            ('~1K params', 8, 'italic')])

# Branch outputs
dim_labels = ['128-dim', '32-dim', '32-dim', '16-dim']
branch_cols = [COLORS['cnn'], COLORS['atm'], COLORS['srf'], COLORS['derived']]
for x, dim, col in zip(xs, dim_labels, branch_cols):
    arrow(ax, x, BRANCH_Y - BRANCH_H/2, x, 65.5)
    label_dim(ax, x, 67.5, dim, col)

# ─── CONCATENATE BANNER ──────────────────────────────────────────────────────
CONCAT_Y = 62; CONCAT_W = 80; CONCAT_H = 5
block(ax, 50, CONCAT_Y, CONCAT_W, CONCAT_H, edge=COLORS['concat'],
      fill_alpha=0.06, lw=1.3,
      rows=[('Concatenate  →  208-dim joint embedding', 11, 'bold')])

# Branch → concat arrows (already drawn down to 65.5; concat top at 64.5)
# Single arrow concat → fusion
arrow(ax, 50, CONCAT_Y - CONCAT_H/2, 50, 51.5, lw=1.4)

# ─── FUSION MLP ──────────────────────────────────────────────────────────────
FUSION_Y = 45; FUSION_W = 38; FUSION_H = 9.5
block(ax, 50, FUSION_Y, FUSION_W, FUSION_H, edge=COLORS['fusion'], lw=1.8,
      rows=[('Fusion MLP', 11.5, 'bold'),
            ('208 → 128 → 64', 9, 'normal'),
            ('Linear → BN → ReLU → Dropout(0.3)', 8.5, 'normal'),
            ('~93K params', 8, 'italic')])

# Fusion → split arrows
arrow(ax, 50, FUSION_Y - FUSION_H/2, 28, 30.5, lw=1.4)
arrow(ax, 50, FUSION_Y - FUSION_H/2, 72, 30.5, lw=1.4)

label_dim(ax, 36, 35, '64-dim', COLORS['fusion'])
label_dim(ax, 64, 35, '64-dim', COLORS['fusion'])

# ─── HEADS ───────────────────────────────────────────────────────────────────
HEAD_Y = 24; HEAD_W = 30; HEAD_H = 11.5
block(ax, 28, HEAD_Y, HEAD_W, HEAD_H, edge=COLORS['cls_head'], lw=1.8,
      rows=[('Classification head', 11, 'bold'),
            ('Linear  64 → 3   (softmax)', 9, 'normal'),
            ('{Non-fire, Wildfire, False-alarm}', 8.5, 'normal'),
            ('~0.2K params', 8, 'italic')])
block(ax, 72, HEAD_Y, HEAD_W, HEAD_H, edge=COLORS['phys_head'], lw=1.8,
      rows=[('Physics head', 11, 'bold'),
            ('64 → 16 → 1   (sigmoid)', 9, 'normal'),
            ('Atmospheric transmittance τ̂', 8.5, 'normal'),
            ('~1K params', 8, 'italic')])

# Head → loss arrows
arrow(ax, 28, HEAD_Y - HEAD_H/2, 28, 14.0, lw=1.2)
arrow(ax, 72, HEAD_Y - HEAD_H/2, 72, 14.0, lw=1.2)

# ─── LOSS BOXES ──────────────────────────────────────────────────────────────
LOSS_Y = 9.5; LOSS_W = 30; LOSS_H = 8
ax.text(28, LOSS_Y,
        'Weighted cross-entropy\n'
        '+ dynamic-range  (BT$_{I4}$ < 310 K)\n'
        '+ thermal-realism  (BTD < 10 K)',
        ha='center', va='center', fontsize=8.5,
        color=COLORS['loss'],
        bbox=dict(facecolor='#fff5f2', edgecolor=COLORS['cls_head'],
                  boxstyle='round,pad=0.5', linewidth=1.0))
ax.text(72, LOSS_Y,
        'Beer–Lambert  MSE\n'
        r'$\hat\tau$  ↔  exp($-0.05 \cdot$ TCWV)',
        ha='center', va='center', fontsize=8.5,
        color=COLORS['loss'],
        bbox=dict(facecolor='#f2f6fc', edgecolor=COLORS['phys_head'],
                  boxstyle='round,pad=0.5', linewidth=1.0))

# ─── COMPOSITE LOSS FORMULA ──────────────────────────────────────────────────
ax.text(50, 2.0,
        r'$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{CE}}'
        r' + \lambda_{\mathrm{BL}}\,\mathcal{L}_{\mathrm{BeerLambert}}'
        r' + \lambda_{\mathrm{DR}}\,\mathcal{L}_{\mathrm{DynamicRange}}'
        r' + \lambda_{\mathrm{TH}}\,\mathcal{L}_{\mathrm{ThermalRealism}}$'
        '   |   '
        r'$\lambda_{\mathrm{BL}}=0.10,\ \lambda_{\mathrm{DR}}=0.05,'
        r'\ \lambda_{\mathrm{TH}}=0.05$',
        ha='center', va='center', fontsize=9.5,
        bbox=dict(facecolor='#f5f5f5', edgecolor='#aaaaaa',
                  boxstyle='round,pad=0.5', linewidth=0.8))

# ─── SAVE ────────────────────────────────────────────────────────────────────
pdf_path = OUTPUT_DIR / 'fig31_architecture.pdf'
png_path = OUTPUT_DIR / 'fig31_architecture.png'
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=300)
print(f'Saved → {pdf_path}')
print(f'Saved → {png_path}')
