"""
Fig 1.1 — Study area co-location map
====================================

Publication-quality figure for FireSight-IR Section 2 (Data).
Shows the western CONUS study domain with ERA5 0.25° grid and OSM
industrial zones overlaid on state boundaries.

Run in Colab:
    !pip install cartopy -q
    %run figures/scripts/fig01_study_area.py

Adjust paths in the CONFIG block below to match your Drive layout.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust to your environment
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR             = Path('/content/drive/MyDrive/firesight-ir')
OSM_INDUSTRIAL_PATH  = BASE_DIR / 'data/raw/surface/osm_infrastructure.json'
OUTPUT_DIR           = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OSM categories to treat as false-alarm-prone industrial features.
# Matches keys in the firesight-ir osm_infrastructure.json bundle.
OSM_CATEGORIES = ('industrial', 'power_plants')

# Study domain — from report Section 2.1
LON_MIN, LON_MAX = -130.0, -100.0
LAT_MIN, LAT_MAX =   30.0,   52.0
ERA5_RESOLUTION  = 0.25                    # native ERA5 grid (deg)
GRID_DRAW_STEP   = 1.0                     # subsample ERA5 dots for readability

# ─────────────────────────────────────────────────────────────────────────────
# STYLE — publication-grade matplotlib defaults
# ─────────────────────────────────────────────────────────────────────────────
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
    'pdf.fonttype':      42,               # editable text in vector output
    'ps.fonttype':       42,
})

PALETTE = {
    'land':       '#f5f1ea',
    'ocean':      '#e6eef5',
    'state':      '#888888',
    'country':    '#444444',
    'domain':     '#0b3d63',
    'era5_grid':  '#9aa0a6',
    'industrial': '#c64a3a',
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def era5_grid_centers(lon_min, lon_max, lat_min, lat_max, step):
    """Return flat lon/lat arrays of ERA5 grid centers at the requested step."""
    lons = np.arange(lon_min, lon_max + step/2, step)
    lats = np.arange(lat_min, lat_max + step/2, step)
    LON, LAT = np.meshgrid(lons, lats)
    return LON.ravel(), LAT.ravel()


def load_industrial_zones(path, categories=OSM_CATEGORIES):
    """Load OSM industrial features from the firesight-ir bundle.
    Expects JSON of the form {category: [[lat, lon], ...]}. Returns
    (lat_arr, lon_arr) merged across the requested categories."""
    if not path.exists():
        print(f'[WARN] OSM industrial file not found: {path}')
        print('       Skipping industrial overlay. Update OSM_INDUSTRIAL_PATH.')
        return None, None

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f'Expected dict at top of {path}; got {type(data).__name__}')

    lats, lons = [], []
    counts = {}
    for cat in categories:
        pts = data.get(cat, [])
        counts[cat] = len(pts)
        for p in pts:
            if len(p) >= 2:
                lats.append(p[0]); lons.append(p[1])
    print(f'[INFO] OSM categories loaded: {counts} → {len(lats):,} total points')
    return np.asarray(lats), np.asarray(lons)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
proj     = ccrs.AlbersEqualArea(central_longitude=-115, central_latitude=41,
                                 standard_parallels=(34, 49))
data_crs = ccrs.PlateCarree()

fig = plt.figure(figsize=(9.0, 6.2))
ax  = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent([LON_MIN - 1, LON_MAX + 1, LAT_MIN - 0.5, LAT_MAX + 0.5], crs=data_crs)

# Basemap layers
ax.add_feature(cfeature.LAND,      facecolor=PALETTE['land'],  zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor=PALETTE['ocean'], zorder=0)
ax.add_feature(cfeature.LAKES,     facecolor=PALETTE['ocean'],
               edgecolor=PALETTE['state'], lw=0.3, zorder=1)
ax.add_feature(cfeature.STATES,    edgecolor=PALETTE['state'],   lw=0.4, zorder=2)
ax.add_feature(cfeature.BORDERS,   edgecolor=PALETTE['country'], lw=0.7, zorder=3)
ax.add_feature(cfeature.COASTLINE, edgecolor=PALETTE['country'], lw=0.7, zorder=3)

# ERA5 grid (subsampled for visual clarity)
lon_e, lat_e = era5_grid_centers(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, GRID_DRAW_STEP)
ax.scatter(lon_e, lat_e, transform=data_crs, s=1.2,
           color=PALETTE['era5_grid'], alpha=0.55, zorder=4)

# OSM industrial overlay
ind_lat, ind_lon = load_industrial_zones(OSM_INDUSTRIAL_PATH, OSM_CATEGORIES)
if ind_lat is not None:
    ax.scatter(ind_lon, ind_lat, transform=data_crs, s=2.0,
               color=PALETTE['industrial'], linewidth=0,
               alpha=0.18, zorder=6)

# Study domain box
ax.add_patch(plt.Rectangle(
    (LON_MIN, LAT_MIN), LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
    transform=data_crs, fill=False,
    edgecolor=PALETTE['domain'], lw=1.8, zorder=7,
))

# Gridlines and labels
gl = ax.gridlines(draw_labels=True, linestyle=':', linewidth=0.4,
                  color='#bbbbbb',
                  xlocs=range(-130, -99, 5), ylocs=range(30, 53, 5))
gl.top_labels   = False
gl.right_labels = False
gl.xformatter   = LongitudeFormatter()
gl.yformatter   = LatitudeFormatter()

# CONUS-context inset (upper-right, over Canada — outside study domain)
ax_inset = fig.add_axes([0.66, 0.65, 0.22, 0.22],
                        projection=ccrs.AlbersEqualArea(
                            central_longitude=-96, central_latitude=39))
ax_inset.set_extent([-125, -68, 24, 50], crs=data_crs)
ax_inset.add_feature(cfeature.LAND,      facecolor=PALETTE['land'])
ax_inset.add_feature(cfeature.OCEAN,     facecolor=PALETTE['ocean'])
ax_inset.add_feature(cfeature.STATES,    edgecolor=PALETTE['state'],   lw=0.2)
ax_inset.add_feature(cfeature.COASTLINE, edgecolor=PALETTE['country'], lw=0.4)
ax_inset.add_patch(plt.Rectangle(
    (LON_MIN, LAT_MIN), LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
    transform=data_crs, fill=False, edgecolor=PALETTE['domain'], lw=1.2))

# Legend
handles = [
    Line2D([0], [0], color=PALETTE['domain'], lw=1.8, label='Study domain'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=PALETTE['era5_grid'], markersize=5,
           label=f'ERA5 grid centers ({GRID_DRAW_STEP}° subsample of {ERA5_RESOLUTION}°)'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=PALETTE['industrial'], markeredgecolor='white',
           markersize=6,
           label=(f'OSM industrial zones (n={len(ind_lat):,})'
                  if ind_lat is not None else 'OSM industrial zones')),
]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.005, 0.005),
          frameon=True, framealpha=0.95, edgecolor='#cccccc',
          handlelength=1.5, borderpad=0.5, labelspacing=0.4)

ax.set_title(
    f'FireSight-IR Study Domain  ·  Western CONUS  ·  2018–2023\n'
    f'{abs(LON_MIN)}° to {abs(LON_MAX)}° W,  {LAT_MIN}° to {LAT_MAX}° N    '
    f'ERA5 {ERA5_RESOLUTION}° atmospheric grid    OSM industrial proximity',
    fontsize=10, pad=10, loc='left')

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
pdf_path = OUTPUT_DIR / 'fig01_study_area.pdf'
png_path = OUTPUT_DIR / 'fig01_study_area.png'
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=300)
print(f'Saved → {pdf_path}')
print(f'Saved → {png_path}')
