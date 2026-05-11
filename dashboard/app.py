#!/usr/bin/env python3
"""
FireSight-IR | Module 4 — Operational Dashboard
================================================
Author  : Emmanuel Ibekwe | github.com/Ibekwemmanuel7
Purpose : Interactive Streamlit app for FireSight-IR wildfire detection
Usage   : streamlit run firesight_dashboard.py
Requires: streamlit torch numpy pandas matplotlib plotly folium
          streamlit-folium scipy scikit-learn h5py pyarrow

Run locally:
    pip install streamlit torch numpy pandas matplotlib plotly folium streamlit-folium scipy scikit-learn pyarrow
    streamlit run firesight_dashboard.py

The app expects firesight_pinn_best.pt in the same directory,
or set MODEL_PATH env var to its location.
"""

import os, json, warnings, io, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "FireSight-IR",
    page_icon   = "🔥",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp { background: #080E14; color: #E8F4FD; }
.main  { background: #080E14; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F1923;
    border-right: 1px solid #1E2D3D;
}
[data-testid="stSidebar"] * { color: #E8F4FD !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0F1923 0%, #141E2B 100%);
    border: 1px solid #1E2D3D;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 6px 0;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #FF6B1A44; }
.metric-value { font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.metric-label { font-size: 0.78rem; color: #6B8FA8; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-sub   { font-size: 0.72rem; color: #6B8FA8; margin-top: 2px; }

/* Class badges */
.badge-wildfire    { background: #FF6B1A22; border: 1px solid #FF6B1A; color: #FF6B1A;
                     padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; }
.badge-false-alarm { background: #2563EB22; border: 1px solid #2563EB; color: #93C5FD;
                     padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; }
.badge-non-fire    { background: #14B8A622; border: 1px solid #14B8A6; color: #5EEAD4;
                     padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #E8F4FD;
    border-left: 3px solid #FF6B1A; padding-left: 12px;
    margin: 24px 0 12px;
}

/* Info boxes */
.info-box {
    background: #0F1923; border: 1px solid #1E2D3D;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Upload area */
[data-testid="stFileUploader"] {
    border: 1px dashed #1E2D3D !important;
    border-radius: 10px !important;
    background: #0F1923 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #FF6B1A, #FF8C42);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 10px 24px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Tabs */
.stTabs [data-baseweb="tab"] { background: transparent; color: #6B8FA8; }
.stTabs [aria-selected="true"] { color: #FF6B1A !important; border-bottom-color: #FF6B1A !important; }

/* Code */
code { font-family: 'JetBrains Mono', monospace; background: #141E2B; }

/* Alert overrides */
.stAlert { border-radius: 8px; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', 'firesight_pinn_best.pt')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES  = ['Non-fire', 'Wildfire', 'False-alarm']
CLASS_COLORS = ['#14B8A6', '#FF6B1A', '#2563EB']
CLASS_BADGES = ['badge-non-fire', 'badge-wildfire', 'badge-false-alarm']

N_ATM=16; N_SRF=20; N_DERIVED=6; N_CLASSES=3; DROPOUT=0.3

BG, PANEL, BORDER = '#080E14', '#0F1923', '#1E2D3D'
TEXT, SUBTEXT = '#E8F4FD', '#6B8FA8'

PLT_RC = {
    'figure.facecolor': BG, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'text.color': TEXT,
    'xtick.color': SUBTEXT, 'ytick.color': SUBTEXT,
    'axes.labelcolor': SUBTEXT, 'grid.color': BORDER,
    'grid.linewidth': 0.5, 'lines.linewidth': 2,
}
plt.rcParams.update(PLT_RC)

# ── Model architecture ────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self,i,o,d=0.2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(i,o),nn.BatchNorm1d(o),nn.ReLU(),nn.Dropout(d),nn.Linear(o,o),nn.BatchNorm1d(o))
        self.proj=nn.Linear(i,o) if i!=o else nn.Identity()
    def forward(self,x): return F.relu(self.net(x)+self.proj(x))

class CNNBranch(nn.Module):
    def __init__(self,c=4,d=0.2):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(c,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),nn.Dropout2d(0.1),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2),nn.Dropout2d(0.1),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(True),nn.MaxPool2d(2))
        self.gap=nn.AdaptiveAvgPool2d(1); self.drop=nn.Dropout(d)
    def forward(self,x): return self.drop(self.gap(self.enc(x)).flatten(1))

class FireSightPINN(nn.Module):
    def __init__(self,na=16,ns=20,nd=6,nc=3,dr=0.3):
        super().__init__()
        self.cnn=CNNBranch(4,dr)
        self.atm=nn.Sequential(ResidualBlock(na,64),ResidualBlock(64,32))
        self.srf=nn.Sequential(ResidualBlock(ns,64),ResidualBlock(64,32))
        self.der=nn.Sequential(nn.Linear(nd,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(0.1),nn.Linear(32,16),nn.BatchNorm1d(16),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(208,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(dr),nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(dr))
        self.cls=nn.Linear(64,nc)
        self.trans=nn.Sequential(nn.Linear(64,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid())
    def forward(self,p,a,s,d):
        f=self.fusion(torch.cat([self.cnn(p),self.atm(a),self.srf(s),self.der(d)],dim=1))
        return self.cls(f),self.trans(f)

# ── Feature column definitions ─────────────────────────────────────────────────
ATM_COLS = [
    'era5_t2m','era5_pbl','era5_tcwv','era5_sp',
    'era5_t_1000hPa','era5_t_850hPa','era5_t_700hPa','era5_t_500hPa','era5_t_300hPa',
    'era5_q_1000hPa','era5_q_850hPa','era5_q_700hPa','era5_q_500hPa','era5_q_300hPa',
    'beer_lambert_proxy','atm_instability',
]
SRF_COLS = [
    'lc_forest','lc_shrub','lc_grassland','lc_cropland',
    'lc_urban','lc_bare','lc_water','lc_other',
    'dnb_radiance','dnb_is_persistent',
    'dist_urban_km','dist_powerplant_km','dist_industrial_km',
    'is_urban','is_industrial',
    'sol_zen','is_day',
    'is_prior_burn_scar','burn_scar_age_years',
    'firms_type',
]
DER_COLS = ['aod_3700nm','lifted_index','doy_sin','doy_cos','bt_i4_anom_norm','btd_anom_norm']

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None, None
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = FireSightPINN(N_ATM,N_SRF,N_DERIVED,N_CLASSES,DROPOUT).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt

# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def classify_dataframe(df: pd.DataFrame, model, scalers: dict = None):
    """
    Classify fire pixels from a parquet/CSV dataframe.
    Returns dataframe with added prediction columns.
    """
    def to_tensor(df, cols, n):
        arr = np.zeros((n, len(cols)), dtype=np.float32)
        for j, col in enumerate(cols):
            if col in df.columns:
                arr[:, j] = df[col].fillna(0).values.astype(np.float32)
            if scalers and col in scalers:
                s = scalers[col]
                arr[:, j] = (arr[:, j] - s['median']) / max(s['iqr'], 1e-8)
        return torch.from_numpy(arr).to(DEVICE)

    n = len(df)

    # Build BT patch (use centre pixel values as proxy if patches not available)
    # For tabular input: synthesise a 32x32 constant patch from BT values
    bt4  = df.get('BT_I4',  pd.Series(np.full(n, 300.0))).fillna(300).values.astype(np.float32)
    bt5  = df.get('BT_I5',  pd.Series(np.full(n, 290.0))).fillna(290).values.astype(np.float32)
    btd  = bt4 - bt5
    fm   = df.get('fire_mask', pd.Series(np.ones(n))).fillna(1).values.astype(np.float32)

    # Normalise BT channels
    bt4n = (bt4 - 300.) / 50.
    bt5n = (bt5 - 290.) / 20.
    btdn = btd / 40.
    fmn  = fm  / 9.

    # Build 4-channel 32x32 patches (tabular → constant patch)
    patches = np.zeros((n, 4, 32, 32), dtype=np.float32)
    patches[:, 0, :, :] = bt4n[:, None, None]
    patches[:, 1, :, :] = bt5n[:, None, None]
    patches[:, 2, :, :] = btdn[:, None, None]
    patches[:, 3, :, :] = fmn[:, None, None]

    atm_t = to_tensor(df, ATM_COLS, n)
    srf_t = to_tensor(df, SRF_COLS, n)
    der_t = to_tensor(df, DER_COLS, n)

    # Batch inference
    all_probs = []
    bs = 512
    for i in range(0, n, bs):
        sl  = slice(i, i+bs)
        p_b = torch.from_numpy(patches[sl]).to(DEVICE)
        with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu', enabled=False):
            logits, _ = model(p_b, atm_t[sl], srf_t[sl], der_t[sl])
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())

    probs = np.concatenate(all_probs)  # (N, 3)
    preds = probs.argmax(axis=1)

    result = df.copy()
    result['pred_class']      = preds
    result['pred_label']      = [CLASS_NAMES[p] for p in preds]
    result['prob_non_fire']   = probs[:, 0].round(4)
    result['prob_wildfire']   = probs[:, 1].round(4)
    result['prob_false_alarm']= probs[:, 2].round(4)
    result['confidence']      = probs.max(axis=1).round(4)
    if 'BTD' not in result.columns:
        result['BTD'] = btd
    return result, probs

# ── Matplotlib figure helpers ─────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=BG)
    buf.seek(0)
    return buf

def plot_class_distribution(result):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(PANEL)
    counts = result['pred_label'].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[CLASS_COLORS[CLASS_NAMES.index(c)] if c in CLASS_NAMES else '#888'
                         for c in counts.index],
                  width=0.55, alpha=0.88)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f'{val:,}', ha='center', fontsize=10, color=TEXT, fontweight='bold')
    ax.set_ylabel('Pixel count', fontsize=10, color=SUBTEXT)
    ax.set_title('Classification results', color=TEXT, fontsize=12, pad=8)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    return fig

def plot_probability_distribution(probs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=BG)
    for i, (name, color, ax) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, axes)):
        ax.set_facecolor(PANEL)
        p = probs[:, i]
        ax.hist(p, bins=50, color=color, alpha=0.75, edgecolor='none')
        ax.axvline(p.mean(), color='white', lw=1.5, linestyle='--', alpha=0.7,
                   label=f'Mean={p.mean():.3f}')
        ax.set_title(f'P({name})', color=TEXT, fontsize=11)
        ax.set_xlabel('Probability', fontsize=9, color=SUBTEXT)
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=0.2)
    fig.suptitle('Prediction probability distributions', color=TEXT, fontsize=13)
    fig.tight_layout()
    return fig

def plot_btd_analysis(result):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    ax1, ax2  = axes

    # BTD density by predicted class
    ax1.set_facecolor(PANEL)
    for cls, color in zip(CLASS_NAMES, CLASS_COLORS):
        sub = result[result['pred_label']==cls]['BTD'].dropna()
        if len(sub) < 5: continue
        xr = np.linspace(sub.quantile(0.01), sub.quantile(0.99), 300)
        try:
            kde = stats.gaussian_kde(sub, bw_method=0.3)
            d = kde(xr); d = d/max(d.max(), 1e-9)
            ax1.fill_between(xr, d, alpha=0.28, color=color)
            ax1.plot(xr, d, color=color, lw=2.5,
                     label=f'{cls}  med={sub.median():.1f}K')
        except Exception:
            pass
    ax1.axvline(10, color='#F59E0B', lw=2, linestyle=':', alpha=0.8, label='BTD=10K threshold')
    ax1.set_xlabel('BTD = BT_I4 − BT_I5  (K)', fontsize=10, color=SUBTEXT)
    ax1.set_ylabel('Normalised density', fontsize=10, color=SUBTEXT)
    ax1.set_title('BTD by predicted class', color=TEXT, fontsize=11, pad=8)
    ax1.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax1.spines[['top','right']].set_visible(False)
    ax1.grid(alpha=0.2)

    # Confidence histogram
    ax2.set_facecolor(PANEL)
    conf = result['confidence']
    ax2.hist(conf, bins=40, color='#FF6B1A', alpha=0.75, edgecolor='none')
    ax2.axvline(conf.mean(), color='white', lw=1.5, linestyle='--', alpha=0.7,
                label=f'Mean={conf.mean():.3f}')
    ax2.axvline(0.90, color='#22C55E', lw=1.2, linestyle=':', alpha=0.7,
                label='90% confidence')
    high_conf_pct = 100*(conf >= 0.90).mean()
    ax2.text(0.88, ax2.get_ylim()[1]*0.85 if ax2.get_ylim()[1]>0 else 1,
             f'{high_conf_pct:.1f}%\nhigh confidence',
             ha='right', fontsize=8.5, color='#22C55E', va='top')
    ax2.set_xlabel('Prediction confidence', fontsize=10, color=SUBTEXT)
    ax2.set_ylabel('Count', fontsize=10, color=SUBTEXT)
    ax2.set_title('Confidence distribution', color=TEXT, fontsize=11, pad=8)
    ax2.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax2.spines[['top','right']].set_visible(False)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    return fig

def plot_spatial(result):
    """Simple scatter map using matplotlib (no folium dependency)."""
    has_coords = 'latitude' in result.columns and 'longitude' in result.columns
    if not has_coords:
        return None
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    ax.set_facecolor('#0A1525')
    for cls, color in zip(CLASS_NAMES, CLASS_COLORS):
        sub = result[result['pred_label']==cls]
        if len(sub)==0: continue
        s = 18 if cls=='False-alarm' else 3
        a = 0.85 if cls=='False-alarm' else 0.3
        ax.scatter(sub['longitude'], sub['latitude'],
                   s=s, c=color, alpha=a, linewidths=0,
                   label=f'{cls} (n={len(sub):,})')
    ax.set_xlabel('Longitude', fontsize=10, color=SUBTEXT)
    ax.set_ylabel('Latitude',  fontsize=10, color=SUBTEXT)
    ax.set_title('Spatial distribution of predictions', color=TEXT, fontsize=12, pad=8)
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
              fontsize=9, markerscale=4)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig

# ── Demo data generator ────────────────────────────────────────────────────────
def make_demo_data(n=500):
    """Generate realistic synthetic VIIRS fire pixel data for demo."""
    rng = np.random.default_rng(42)
    # Mix: 80% wildfire, 15% false-alarm, 5% non-fire
    n_wf = int(n*0.80); n_fa = int(n*0.15); n_nf = n - n_wf - n_fa

    def wf_row(size):
        return {
            'latitude':  rng.uniform(32, 49, size),
            'longitude': rng.uniform(-125, -109, size),
            'BT_I4':     rng.normal(355, 40, size).clip(310, 500),
            'BT_I5':     rng.normal(300, 15, size).clip(270, 340),
            'era5_t2m':  rng.normal(295, 8,  size),
            'era5_tcwv': rng.normal(10, 5,   size).clip(0, 40),
            'era5_pbl':  rng.normal(800, 300, size).clip(100, 3000),
            'beer_lambert_proxy': rng.uniform(0.5, 0.9, size),
            'atm_instability':    rng.normal(25, 8, size),
            'lc_forest':    rng.choice([0,1], size, p=[0.4,0.6]).astype(float),
            'lc_grassland': rng.choice([0,1], size, p=[0.7,0.3]).astype(float),
            'lc_urban':     np.zeros(size),
            'dist_industrial_km': rng.uniform(10, 100, size),
            'is_industrial': np.zeros(size),
            'sol_zen': rng.uniform(20, 70, size),
            'is_day':  np.ones(size),
            'true_label': np.ones(size, dtype=int),
        }
    def fa_row(size):
        return {
            'latitude':  rng.uniform(32, 49, size),
            'longitude': rng.uniform(-125, -109, size),
            'BT_I4':     rng.normal(315, 8, size).clip(300, 340),
            'BT_I5':     rng.normal(305, 8, size).clip(280, 330),
            'era5_t2m':  rng.normal(298, 5, size),
            'era5_tcwv': rng.normal(12, 4,  size).clip(0, 40),
            'era5_pbl':  rng.normal(600, 200, size).clip(100, 2000),
            'beer_lambert_proxy': rng.uniform(0.4, 0.7, size),
            'atm_instability':    rng.normal(20, 6, size),
            'lc_forest':    np.zeros(size),
            'lc_grassland': np.zeros(size),
            'lc_urban':     np.ones(size),
            'dist_industrial_km': rng.uniform(0.1, 3, size),
            'is_industrial': np.ones(size),
            'sol_zen': rng.uniform(30, 80, size),
            'is_day':  np.ones(size),
            'true_label': 2*np.ones(size, dtype=int),
        }
    def nf_row(size):
        r = fa_row(size); r['true_label'] = np.zeros(size, dtype=int)
        r['BT_I4'] = rng.normal(295, 5, size).clip(280, 310)
        r['BT_I5'] = rng.normal(293, 5, size).clip(278, 308)
        return r

    rows = []
    for fn, sz in [(wf_row, n_wf),(fa_row, n_fa),(nf_row, n_nf)]:
        rows.append(pd.DataFrame(fn(sz)))

    df = pd.concat(rows, ignore_index=True)
    df['BTD'] = df['BT_I4'] - df['BT_I5']
    df['doy_sin'] = np.sin(2*np.pi*200/365)  # mid-fire-season
    df['doy_cos'] = np.cos(2*np.pi*200/365)
    df['bt_i4_anom_norm'] = ((df['BT_I4'] - 300) / 50).clip(0, 2)
    df['btd_anom_norm']   = (df['BTD'] / 40).clip(0, 3)
    df['lifted_index'] = rng.normal(-0.4, 0.5, n)
    df['aod_3700nm']   = rng.uniform(0, 0.5, n)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px;">
        <div style="font-size:2.4rem;">🔥</div>
        <div style="font-size:1.4rem; font-weight:700; color:#FF6B1A;">FireSight-IR</div>
        <div style="font-size:0.75rem; color:#6B8FA8; margin-top:4px;">
            FireSat Protoflight-aligned<br>Wildfire Detection Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**Model**")
    model_file = st.file_uploader(
        "Upload model checkpoint (.pt)",
        type=['pt'], key='model_upload',
        help='firesight_pinn_best.pt from Google Drive/models/'
    )

    st.markdown("---")
    st.markdown("**Classification thresholds**")
    wf_thresh = st.slider("Wildfire min confidence", 0.50, 0.99, 0.50, 0.01)
    fa_thresh = st.slider("False-alarm min confidence", 0.50, 0.99, 0.50, 0.01)

    st.markdown("---")
    st.markdown("**About**", unsafe_allow_html=False)
    st.markdown("""
    <div style="font-size:0.75rem; color:#6B8FA8; line-height:1.6;">
        <b>Author:</b> Emmanuel Ibekwe<br>
        <b>Model:</b> Multi-branch PINN<br>
        <b>Params:</b> 202,228<br>
        <b>Best epoch:</b> 23<br>
        <b>Val loss:</b> 0.1149<br>
        <b>WF recall:</b> 95.4%<br>
        <b>FA AUC:</b> 1.0000<br><br>
        <a href="https://github.com/Ibekwemmanuel7" style="color:#FF6B1A;">
        github.com/Ibekwemmanuel7</a>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — load model
# ══════════════════════════════════════════════════════════════════════════════
model, ckpt_info = None, None

# Try uploaded model first
if model_file is not None:
    tmp_path = '/tmp/uploaded_model.pt'
    with open(tmp_path, 'wb') as f:
        f.write(model_file.read())
    model, ckpt_info = load_model(tmp_path)
    if model:
        st.sidebar.success(f"✓ Model loaded (epoch {ckpt_info.get('epoch','?')})")
elif os.path.exists(MODEL_PATH):
    model, ckpt_info = load_model(MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
    <div>
        <h1 style="margin:0; font-size:2rem; color:#E8F4FD;">
            🔥 FireSight-IR
        </h1>
        <p style="margin:0; color:#6B8FA8; font-size:0.9rem;">
            Physics-Informed Wildfire Detection &amp; False-Alarm Rejection
            &nbsp;·&nbsp; FireSat Protoflight-aligned
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Model status banner
if model:
    ep  = ckpt_info.get('epoch','?')
    vl  = ckpt_info.get('val_loss','?')
    va  = ckpt_info.get('val_acc','?')
    pca = ckpt_info.get('per_class_acc', {})
    st.markdown(f"""
    <div class="info-box" style="border-left:3px solid #22C55E; margin-bottom:16px;">
        ✅ &nbsp;<b>Model loaded</b> — Epoch {ep} &nbsp;|&nbsp;
        Val loss {vl:.4f} &nbsp;|&nbsp;
        Val accuracy {va:.1%} &nbsp;|&nbsp;
        WF recall {pca.get('wf',0):.1%} &nbsp;|&nbsp;
        FA recall {pca.get('fa',0):.1%} &nbsp;|&nbsp;
        FA AUC <b>1.0000</b>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box" style="border-left:3px solid #F59E0B; margin-bottom:16px;">
        ⚠️ &nbsp;<b>No model loaded</b> — Upload <code>firesight_pinn_best.pt</code>
        in the sidebar, or the demo will run with random predictions.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_classify, tab_explore, tab_about = st.tabs([
    "🔍 Classify pixels",
    "📊 Model performance",
    "ℹ️ About",
])

# ─── TAB 1: Classify ──────────────────────────────────────────────────────────
with tab_classify:
    st.markdown('<div class="section-header">Upload fire pixel data</div>', unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload VIIRS fire pixel data (.parquet or .csv)",
            type=['parquet','csv'],
            help="Must contain BT_I4, BT_I5, and ERA5/surface feature columns. "
                 "See required schema in the About tab."
        )
        use_demo = st.checkbox("Use demo data (synthetic 500-pixel batch)", value=not bool(uploaded))

    with col_info:
        st.markdown("""
        <div class="info-box">
        <b>Required columns</b><br>
        <code>BT_I4</code>, <code>BT_I5</code><br>
        <code>era5_t2m</code>, <code>era5_tcwv</code><br>
        <code>lc_forest</code>, <code>lc_urban</code><br>
        <code>dist_industrial_km</code><br>
        <code>sol_zen</code>, <code>is_day</code><br>
        <br>
        <b>Optional (for map)</b><br>
        <code>latitude</code>, <code>longitude</code>
        </div>
        """, unsafe_allow_html=True)

    # Load data
    df_input = None
    if use_demo or not uploaded:
        df_input = make_demo_data(500)
        st.info("Using synthetic demo data — 500 pixels (80% wildfire, 15% false-alarm, 5% non-fire)")
    elif uploaded:
        try:
            if uploaded.name.endswith('.parquet'):
                df_input = pd.read_parquet(uploaded)
            else:
                df_input = pd.read_csv(uploaded)
            st.success(f"✓ Loaded {len(df_input):,} pixels | {len(df_input.columns)} columns")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    if df_input is not None and st.button("🔥 Classify fire pixels"):
        with st.spinner("Running FireSight-IR inference..."):
            t0 = time.time()
            if model:
                result, probs = classify_dataframe(df_input, model)
            else:
                # Demo mode without model — random predictions
                result = df_input.copy()
                probs = np.random.dirichlet([8, 1, 2], size=len(df_input)).astype(np.float32)
                preds = probs.argmax(axis=1)
                result['pred_class']       = preds
                result['pred_label']       = [CLASS_NAMES[p] for p in preds]
                result['prob_non_fire']    = probs[:,0].round(4)
                result['prob_wildfire']    = probs[:,1].round(4)
                result['prob_false_alarm'] = probs[:,2].round(4)
                result['confidence']       = probs.max(axis=1).round(4)
                if 'BTD' not in result.columns and 'BT_I4' in result.columns:
                    result['BTD'] = result['BT_I4'] - result['BT_I5']
            elapsed = time.time() - t0

        st.success(f"✓ Classified {len(result):,} pixels in {elapsed:.2f}s")

        # ── Stat cards ────────────────────────────────────────────────────────
        vc = result['pred_label'].value_counts()
        n_wf = int(vc.get('Wildfire',0))
        n_fa = int(vc.get('False-alarm',0))
        n_nf = int(vc.get('Non-fire',0))
        n_tot = len(result)
        mean_conf = result['confidence'].mean()

        c1,c2,c3,c4,c5 = st.columns(5)
        for col, val, label, sub, color in [
            (c1, f"{n_tot:,}",        "Total pixels",       "input batch",                  TEXT),
            (c2, f"{n_wf:,}",         "Wildfire",           f"{100*n_wf/n_tot:.1f}% of batch", '#FF6B1A'),
            (c3, f"{n_fa:,}",         "False-alarm",        f"{100*n_fa/n_tot:.1f}% of batch", '#93C5FD'),
            (c4, f"{n_nf:,}",         "Non-fire",           f"{100*n_nf/n_tot:.1f}% of batch", '#5EEAD4'),
            (c5, f"{mean_conf:.1%}",  "Mean confidence",    "across all predictions",       '#F59E0B'),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)
        with ch1:
            st.pyplot(fig_to_bytes(plot_class_distribution(result)),
                      use_container_width=True)
        with ch2:
            st.pyplot(fig_to_bytes(plot_probability_distribution(probs)),
                      use_container_width=True)

        # BTD analysis
        st.markdown('<div class="section-header">Physics constraint analysis</div>', unsafe_allow_html=True)
        st.pyplot(fig_to_bytes(plot_btd_analysis(result)), use_container_width=True)

        # Spatial map
        if 'latitude' in result.columns and 'longitude' in result.columns:
            st.markdown('<div class="section-header">Spatial distribution</div>', unsafe_allow_html=True)
            fig_sp = plot_spatial(result)
            if fig_sp:
                st.pyplot(fig_to_bytes(fig_sp), use_container_width=True)

        # Results table
        st.markdown('<div class="section-header">Pixel-level results</div>', unsafe_allow_html=True)
        display_cols = [c for c in [
            'latitude','longitude','BT_I4','BT_I5','BTD',
            'pred_label','prob_wildfire','prob_false_alarm','confidence'
        ] if c in result.columns]
        st.dataframe(
            result[display_cols].head(200),
            use_container_width=True, height=350
        )
        if len(result) > 200:
            st.caption(f"Showing first 200 of {len(result):,} rows")

        # Download
        csv_buf = io.StringIO()
        result.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download full results (CSV)",
            data=csv_buf.getvalue(),
            file_name="firesight_ir_results.csv",
            mime="text/csv"
        )

        # Store for explore tab
        st.session_state['last_result'] = result
        st.session_state['last_probs']  = probs

# ─── TAB 2: Model performance ──────────────────────────────────────────────────
with tab_explore:
    st.markdown('<div class="section-header">Module 3a training summary</div>', unsafe_allow_html=True)

    # Hard-coded best results (from confirmed evaluation)
    perf_data = {
        'Metric': [
            'Best epoch', 'Val loss', 'Overall accuracy (val 2023)',
            'Wildfire recall', 'Wildfire precision', 'Wildfire AUC',
            'False-alarm recall', 'False-alarm precision', 'False-alarm AUC',
            'Non-fire recall', 'Non-fire AUC',
            'Test accuracy (2018-2022)', 'Test false-alarm AUC',
        ],
        'Value': [
            '23', '0.1149', '95.84%',
            '95.41%', '99.87%', '0.9960',
            '99.93%', '97.83%', '1.0000',
            '97.55%', '0.9910',
            '96.16%', '1.0000',
        ],
        'Note': [
            'of 30 epochs', 'weighted CE + physics loss', '2023 fully held-out year',
            'above 0.95 target', 'very few false wildfire predictions', 'val 2023',
            'near-perfect', '', 'perfect separation — no 2nd stage needed',
            '', 'val 2023',
            '20% random holdout', 'test set',
        ]
    }
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, use_container_width=True, hide_index=True, height=480)

    st.markdown('<div class="section-header">Architecture summary</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <pre style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#E8F4FD; margin:0;">
CNN branch     (4-ch BT patches 32×32)          →  128-dim
MLP-atm        (ERA5 atmospheric, 16 features)   →   32-dim
MLP-srf        (MODIS LC + OSM infra, 20 feat)  →   32-dim
MLP-derived    (Physics features, 6 feat)        →   16-dim
                                                 ──────────
Fusion MLP     208 → 128 → 64
Classification head   64 → 3    non-fire / wildfire / false-alarm
Physics head          64 → 1    transmittance (Beer-Lambert)

Total: 202,228 parameters

Physics loss = CE(weighted) + 0.10·Beer-Lambert
             + 0.05·Dynamic Range + 0.05·Thermal Realism
Class weights: non-fire×6.9 | wildfire×0.36 | false-alarm×9.81
    </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Training log</div>', unsafe_allow_html=True)
    log_table = {
        'Epoch':[1,6,9,13,17,19,21,23,25,27,29,30],
        'Val loss':[0.3839,0.1796,0.1456,0.1284,0.1179,0.1179,0.1170,0.1149,0.1175,0.1171,0.1163,0.1182],
        'Val acc':[0.766,0.852,0.915,0.945,0.951,0.966,0.951,0.958,0.955,0.954,0.956,0.952],
        'WF recall':[0.740,0.833,0.905,0.939,0.946,0.963,0.945,0.954,0.950,0.949,0.951,0.946],
        'FA recall':[0.997,0.997,0.998,1.000,1.000,0.999,1.000,0.999,1.000,1.000,1.000,0.999],
    }
    df_log = pd.DataFrame(log_table)
    df_log_disp = df_log.style.highlight_min(subset=['Val loss'], color='#14B8A622')\
                               .highlight_max(subset=['WF recall'], color='#FF6B1A22')
    st.dataframe(df_log, use_container_width=True, hide_index=True)

# ─── TAB 3: About ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="info-box" style="margin-bottom:16px;">
    <h3 style="color:#FF6B1A; margin-top:0;">FireSight-IR</h3>
    <p>
    A physics-informed neural network pipeline for wildfire detection and false-alarm
    rejection, built as a direct response to the intelligence gap exposed in FireSat
    Protoflight first-light imagery (July 2025).
    </p>
    <p>
    FireSat's MWIR/LWIR sensors correctly detected gas flares in Libya, urban heat
    islands in Sydney, and a 2020 burn scar in Ontario being warmed by the sun — but
    these all registered as potential fire detections. FireSight-IR was built to
    discriminate these false-alarm sources from genuine wildfires.
    </p>
    </div>

    <div class="info-box">
    <h4 style="color:#E8F4FD; margin-top:0;">Input data schema</h4>
    <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
    <tr style="border-bottom:1px solid #1E2D3D;">
        <th style="text-align:left; padding:6px; color:#6B8FA8;">Column</th>
        <th style="text-align:left; padding:6px; color:#6B8FA8;">Type</th>
        <th style="text-align:left; padding:6px; color:#6B8FA8;">Description</th>
        <th style="text-align:left; padding:6px; color:#6B8FA8;">Required</th>
    </tr>
    <tr><td style="padding:5px;"><code>BT_I4</code></td><td>float</td><td>MWIR brightness temp (K)</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>BT_I5</code></td><td>float</td><td>LWIR brightness temp (K)</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>era5_t2m</code></td><td>float</td><td>2m air temperature (K)</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>era5_tcwv</code></td><td>float</td><td>Total column water vapour (kg/m²)</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>era5_pbl</code></td><td>float</td><td>Boundary layer height (m)</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>beer_lambert_proxy</code></td><td>float</td><td>exp(-0.05·TCWV) transmittance</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>atm_instability</code></td><td>float</td><td>T_850hPa − T_500hPa (K)</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>lc_urban</code>, <code>lc_forest</code>...</td><td>float 0/1</td><td>MODIS MCD12Q1 land cover</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>dist_industrial_km</code></td><td>float</td><td>Distance to nearest industrial site</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>sol_zen</code></td><td>float</td><td>Solar zenith angle (degrees)</td><td>✓</td></tr>
    <tr><td style="padding:5px;"><code>latitude</code>, <code>longitude</code></td><td>float</td><td>Pixel centre coordinates</td><td>Optional (for map)</td></tr>
    </table>
    </div>

    <div class="info-box" style="margin-top:16px;">
    <h4 style="color:#E8F4FD; margin-top:0;">Pipeline modules</h4>
    <div style="font-size:0.85rem; line-height:2;">
    ✅ <b>Module 1a</b> — VIIRS + FIRMS download (1,149,722 fire pixels)<br>
    ✅ <b>Module 1b</b> — ERA5 atmospheric co-location (16 features, 100% coverage)<br>
    ✅ <b>Module 1c v2</b> — MODIS MCD12Q1 land cover + OSM infrastructure<br>
    ✅ <b>Module 2 v2</b> — Feature engineering, robust normalisation, class weights<br>
    ✅ <b>Module 3a</b> — Multi-branch PINN training (epoch 23, val_loss=0.1149)<br>
    ➡️ <b>Module 4</b> — This dashboard (operational inference)<br>
    </div>
    </div>

    <div style="margin-top:16px; font-size:0.82rem; color:#6B8FA8;">
    <b>Author:</b> Emmanuel Ibekwe &nbsp;·&nbsp;
    <a href="https://github.com/Ibekwemmanuel7" style="color:#FF6B1A;">github.com/Ibekwemmanuel7</a>
    &nbsp;·&nbsp; M.Sc. Atmospheric Science, Texas A&M University
    </div>
    """, unsafe_allow_html=True)
