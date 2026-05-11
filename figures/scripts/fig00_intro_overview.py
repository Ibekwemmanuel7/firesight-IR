"""
Fig 1.0 — Introduction data-sources overview
============================================

Section 1 narrative figure for FireSight-IR.
Four panels on the western CONUS study domain showing the four data
streams that feed the FireSight-IR pipeline:

    A. VIIRS fire pixels coloured by FRP
    B. ERA5 atmospheric water vapour (TCWV) — August 2020
    C. MODIS MCD12Q1 land cover
    D. OSM industrial features

Each panel: Albers Equal Area, state boundaries, major western US cities
labelled, study domain outlined.

Run in Colab:
    !apt-get install -y libgeos-dev libproj-dev > /dev/null 2>&1
    !pip install cartopy xarray rioxarray rasterio netCDF4 -q 2>&1 | tail -3
    %run /content/drive/MyDrive/firesight-ir/scripts/fig00_intro_overview.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path('/content/drive/MyDrive/firesight-ir')
FP_DIR    = BASE_DIR / 'data/processed/viirs_fp_srf_v2'
FP_GLOB   = 'viirs_fp_srf_v2_*.parquet'
ERA5_PATH = BASE_DIR / 'data/raw/era5/era5_surface_202008.nc'

# MODIS / OSM may live in a few places — check all of them
SURFACE_CANDIDATES = [
    BASE_DIR / 'data/raw/surface',
    BASE_DIR / 'data/processed/surface',
    BASE_DIR / 'data/surface',
]
MODIS_LC_NAME      = 'modis_lc_2020_western_us.tif'
OSM_NAME           = 'osm_infrastructure.json'

OUTPUT_DIR = BASE_DIR / 'figures/publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Study domain
LON_MIN, LON_MAX = -130.0, -100.0
LAT_MIN, LAT_MAX =   30.0,   52.0

# Plotting parameters
FP_PLOT_SAMPLE = 200_000     # subsample fire pixels for speed
OSM_CATEGORIES = ('industrial', 'power_plants')

# Major western US cities for context
CITIES = [
    ('Seattle',        47.61, -122.33),
    ('Portland',       45.52, -122.68),
    ('Boise',          43.62, -116.20),
    ('San Francisco',  37.77, -122.42),
    ('Los Angeles',    34.05, -118.24),
    ('San Diego',      32.72, -117.16),
    ('Las Vegas',      36.17, -115.14),
    ('Phoenix',        33.45, -112.07),
    ('Salt Lake City', 40.76, -111.89),
    ('Billings',       45.79, -108.50),
    ('Denver',         39.74, -104.99),
    ('Albuquerque',    35.08, -106.65),
    ('El Paso',        31.76, -106.49),
]

# ─── STYLE ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         10,
    'axes.labelsize':    9,
    'axes.titlesize':    11,
    'legend.fontsize':    8,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

PALETTE = {
    'land':       '#f5f1ea',
    'ocean':      '#e6eef5',
    'state':      '#888888',
    'country':    '#444444',
    'domain':     '#0b3d63',
    'city':       '#222222',
    'industrial': '#c64a3a',
}

LC_COLORS = {  # IGBP-style condensed palette
    'Forest':     '#2e6b34',
    'Shrub':      '#b3892b',
    'Grassland':  '#9bcc52',
    'Cropland':   '#f0d465',
    'Urban':      '#d65a3a',
    'Bare':       '#9e8a72',
    'Water':      '#86b3d6',
    'Other':      '#999999',
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def find_first(candidates, name):
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return None


def add_basemap(ax):
    ax.set_extent([LON_MIN - 1, LON_MAX + 1, LAT_MIN - 0.5, LAT_MAX + 0.5],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,   facecolor=PALETTE['land'],  zorder=0)
    ax.add_feature(cfeature.OCEAN,  facecolor=PALETTE['ocean'], zorder=0)
    ax.add_feature(cfeature.LAKES,  facecolor=PALETTE['ocean'],
                   edgecolor=PALETTE['state'], lw=0.25, zorder=1)
    ax.add_feature(cfeature.STATES, edgecolor=PALETTE['state'],
                   lw=0.4, zorder=2)
    ax.add_feature(cfeature.BORDERS,   edgecolor=PALETTE['country'],
                   lw=0.6, zorder=3)
    ax.add_feature(cfeature.COASTLINE, edgecolor=PALETTE['country'],
                   lw=0.6, zorder=3)


CITY_OFFSETS = {  # per-city label offset in points (dx, dy) when default collides
    'Seattle':       ( 5,  -8),
    'Portland':      ( 5,  -7),
    'San Francisco': (-72,  -2),
    'Los Angeles':   ( 5,  -7),
    'San Diego':     ( 5,  -8),
    'Las Vegas':     ( 5,   2),
    'Salt Lake City':( 5,  -7),
    'Denver':        ( 5,  -2),
    'Boise':         ( 5,  -7),
    'Billings':      ( 5,  -2),
    'Albuquerque':   ( 5,   2),
    'El Paso':       ( 5,   2),
    'Phoenix':       ( 5,  -7),
}


def add_cities(ax):
    import matplotlib.patheffects as pe
    pc = ccrs.PlateCarree()
    halo = [pe.withStroke(linewidth=1.6, foreground='white')]
    for name, lat, lon in CITIES:
        ax.plot(lon, lat, marker='o', markersize=3.0, color='#111111',
                markeredgecolor='white', markeredgewidth=0.5,
                transform=pc, zorder=12)
        dx, dy = CITY_OFFSETS.get(name, (4, 3))
        ax.annotate(name, xy=(lon, lat), xycoords=pc._as_mpl_transform(ax),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=7, color=PALETTE['city'],
                    path_effects=halo, zorder=13)


def add_domain_box(ax, lw=1.4):
    pc = ccrs.PlateCarree()
    ax.add_patch(plt.Rectangle(
        (LON_MIN, LAT_MIN), LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
        transform=pc, fill=False, edgecolor=PALETTE['domain'],
        lw=lw, zorder=11))


# ─── DATA LOADERS ────────────────────────────────────────────────────────────
def load_fire_pixels():
    files = sorted(FP_DIR.glob(FP_GLOB))
    if not files:
        print(f'[WARN] No fire-pixel parquet files found in {FP_DIR}')
        return None
    cols = None
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if cols is None:
            cols = df.columns.tolist()
            print(f'[INFO] FP columns: {cols[:12]}{"..." if len(cols) > 12 else ""}')
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # Find lat / lon / frp columns flexibly
    lat_col = next((c for c in df.columns if c.lower() in ('lat', 'latitude')), None)
    lon_col = next((c for c in df.columns if c.lower() in ('lon', 'longitude')), None)
    frp_col = next((c for c in df.columns if c.lower() in ('frp', 'frp_mw')), None)
    if not (lat_col and lon_col):
        print(f'[ERR] Could not find lat/lon columns. Available: {df.columns.tolist()[:20]}')
        return None
    out = df[[lat_col, lon_col] + ([frp_col] if frp_col else [])].copy()
    out.columns = ['lat', 'lon'] + (['frp'] if frp_col else [])
    if 'frp' not in out.columns:
        out['frp'] = 1.0
    print(f'[INFO] FP loaded: {len(out):,} pixels '
          f'(FRP min={out["frp"].min():.2f}, '
          f'median={out["frp"].median():.2f}, '
          f'max={out["frp"].max():.2f})')
    return out


def load_era5_tcwv(path):
    if not path.exists():
        print(f'[WARN] ERA5 file not found: {path}')
        return None
    import xarray as xr
    ds = xr.open_dataset(path)
    var = next((v for v in ds.data_vars if v.lower() in ('tcwv', 'tciwv', 'tcw')), None)
    if var is None:
        print(f'[WARN] No TCWV-like variable in {path}. Found: {list(ds.data_vars)}')
        return None
    da = ds[var]
    # Collapse any non-spatial dimensions (time, valid_time, step, ensemble, …)
    spatial = {'lat', 'latitude', 'lon', 'longitude', 'x', 'y'}
    for d in list(da.dims):
        if d.lower() not in spatial:
            da = da.mean(dim=d)
    # Standardise coordinate names
    if 'longitude' in da.coords:
        da = da.rename({'longitude': 'lon'})
    if 'latitude' in da.coords:
        da = da.rename({'latitude': 'lat'})
    print(f'[INFO] ERA5 TCWV loaded from {path.name}: shape={da.shape}, '
          f'mean={float(da.mean()):.1f} kg/m²')
    return da


def load_modis_lc():
    path = find_first(SURFACE_CANDIDATES, MODIS_LC_NAME)
    if path is None:
        print(f'[WARN] MODIS LC not found in any of: '
              f'{[str(d) for d in SURFACE_CANDIDATES]}')
        return None, None, None
    import rioxarray as rxr
    da = rxr.open_rasterio(path).squeeze()
    print(f'[INFO] MODIS LC loaded from {path}: shape={da.shape}, '
          f'unique={np.unique(da.values)[:15]}')
    return da, path, np.unique(da.values)


def load_osm_industrial():
    path = find_first(SURFACE_CANDIDATES, OSM_NAME)
    if path is None:
        print(f'[WARN] OSM file not found.')
        return None, None
    with open(path) as f:
        data = json.load(f)
    lats, lons = [], []
    for cat in OSM_CATEGORIES:
        for p in data.get(cat, []):
            if len(p) >= 2:
                lats.append(p[0]); lons.append(p[1])
    print(f'[INFO] OSM loaded: {len(lats):,} industrial features')
    return np.asarray(lats), np.asarray(lons)


# ─── PANELS ──────────────────────────────────────────────────────────────────
def panel_A_fires(ax, fp):
    add_basemap(ax)
    if fp is not None:
        if len(fp) > FP_PLOT_SAMPLE:
            fp = fp.sample(FP_PLOT_SAMPLE, random_state=42)
        sc = ax.scatter(
            fp['lon'], fp['lat'], c=np.maximum(fp['frp'].values, 0.5),
            cmap='inferno', norm=LogNorm(vmin=0.5, vmax=2000),
            s=1.3, linewidth=0, alpha=0.55, zorder=5,
            transform=ccrs.PlateCarree())
        cax = ax.inset_axes([0.62, 0.04, 0.34, 0.022])
        cb = plt.colorbar(sc, cax=cax, orientation='horizontal')
        cb.set_label('FRP (MW, log scale)', fontsize=7)
        cb.ax.tick_params(labelsize=6)
    add_domain_box(ax)
    add_cities(ax)
    ax.set_title('A.  VIIRS active fire detections (2018–2023)\n'
                 'coloured by Fire Radiative Power',
                 loc='left', fontsize=10)


def panel_B_era5(ax, da):
    add_basemap(ax)
    if da is not None:
        # Subset to study domain for cleaner plot
        sub = da.sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX)) \
              if da.lat[0] > da.lat[-1] else \
              da.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
        im = ax.pcolormesh(sub.lon, sub.lat, sub.values,
                           cmap='YlGnBu', shading='auto',
                           alpha=0.78, zorder=4,
                           transform=ccrs.PlateCarree())
        cax = ax.inset_axes([0.62, 0.04, 0.34, 0.022])
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_label('TCWV (kg/m²)', fontsize=7)
        cb.ax.tick_params(labelsize=6)
    add_domain_box(ax)
    add_cities(ax)
    ax.set_title('B.  ERA5 atmospheric state — August 2020\n'
                 'total column water vapour',
                 loc='left', fontsize=10)


def panel_C_modis(ax, da, unique_vals):
    add_basemap(ax)
    if da is not None:
        # Map raw IGBP-like codes onto a condensed scheme
        igbp_to_class = {
            1:'Forest', 2:'Forest', 3:'Forest', 4:'Forest', 5:'Forest',
            6:'Shrub',  7:'Shrub',
            8:'Forest', 9:'Forest',
            10:'Grassland',
            11:'Other',
            12:'Cropland',
            13:'Urban',
            14:'Cropland',
            15:'Other',
            16:'Bare',
            17:'Water',
        }
        classes = ['Forest','Shrub','Grassland','Cropland','Urban','Bare','Water','Other']
        class_to_idx = {c: i for i, c in enumerate(classes)}
        cmap = ListedColormap([LC_COLORS[c] for c in classes])
        # Build remapped array
        remap = np.full(da.shape, len(classes) - 1, dtype=np.int8)
        vals = da.values
        for code, cls in igbp_to_class.items():
            remap[vals == code] = class_to_idx[cls]
        # Coordinates — rioxarray uses x/y
        x = da.x.values if 'x' in da.coords else da.lon.values
        y = da.y.values if 'y' in da.coords else da.lat.values
        ax.pcolormesh(x, y, remap, cmap=cmap,
                      vmin=0, vmax=len(classes) - 1,
                      shading='auto', alpha=0.85, zorder=4,
                      transform=ccrs.PlateCarree())
        # Legend
        handles = [Line2D([0],[0], marker='s', color='w',
                          markerfacecolor=LC_COLORS[c],
                          markersize=8, label=c) for c in classes]
        ax.legend(handles=handles, loc='lower left', ncol=2,
                  bbox_to_anchor=(0.005, 0.005), frameon=True,
                  framealpha=0.92, edgecolor='#cccccc',
                  fontsize=7, handlelength=1.2,
                  borderpad=0.4, labelspacing=0.3, columnspacing=0.8)
    add_domain_box(ax)
    add_cities(ax)
    ax.set_title('C.  MODIS MCD12Q1 v6.1 land cover\n'
                 'IGBP classification, 500 m',
                 loc='left', fontsize=10)


def panel_D_osm(ax, lats, lons):
    add_basemap(ax)
    if lats is not None and len(lats):
        ax.scatter(lons, lats, s=2.0, color=PALETTE['industrial'],
                   linewidth=0, alpha=0.20, zorder=5,
                   transform=ccrs.PlateCarree())
    add_domain_box(ax)
    add_cities(ax)
    ax.set_title(f'D.  OSM industrial features (n={len(lats):,})\n'
                 'gas flares, refineries, power plants, industrial zones'
                 if lats is not None else 'D.  OSM industrial features',
                 loc='left', fontsize=10)


# ─── MAIN ────────────────────────────────────────────────────────────────────
print('Loading data …')
fp_df            = load_fire_pixels()
era5_da          = load_era5_tcwv(ERA5_PATH)
modis_da, _, uv  = load_modis_lc()
osm_lat, osm_lon = load_osm_industrial()

proj = ccrs.AlbersEqualArea(central_longitude=-115, central_latitude=41,
                             standard_parallels=(34, 49))

fig, axes = plt.subplots(2, 2, figsize=(14.5, 11),
                          subplot_kw={'projection': proj})
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04,
                    wspace=0.06, hspace=0.18)

panel_A_fires(axes[0, 0], fp_df)
panel_B_era5 (axes[0, 1], era5_da)
panel_C_modis(axes[1, 0], modis_da, uv)
panel_D_osm  (axes[1, 1], osm_lat, osm_lon)

fig.suptitle(
    'FireSight-IR  ·  Multi-source data inputs  ·  Western CONUS, 2018–2023',
    fontsize=13, y=0.96, x=0.5, ha='center', weight='bold')

# Save (don't use bbox='tight' — it crops the suptitle)
pdf_path = OUTPUT_DIR / 'fig00_intro_overview.pdf'
png_path = OUTPUT_DIR / 'fig00_intro_overview.png'
fig.savefig(pdf_path, bbox_inches=None)
fig.savefig(png_path, dpi=300, bbox_inches=None)
print(f'\nSaved → {pdf_path}')
print(f'Saved → {png_path}')
