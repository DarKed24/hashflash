import os
import pickle
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from audio_io import load_audio, spectrogram_db
from fingerprint import FingerprintDB, find_constellation_peaks

WIN_LENGTH = 4096
HOP_LENGTH = 2048
SR         = 22050
DB_PATH    = "data/fingerprint_db.pkl"
SONGS_DIR  = "songs"

st.set_page_config(
    page_title="HashFlash EE200 Project",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Serif:wght@400;500;600&display=swap');

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
.main, .block-container {
    background-color: #0f1117 !important;
    color: #dde3ee !important;
}
header[data-testid="stHeader"] {
    background-color: #0f1117 !important;
    border-bottom: 1px solid #1f2535 !important;
}
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }

.page-shell, .topbar, .match-block, .stat-pill, .callout {
    box-sizing: border-box;
}

:root {
    --void:      #0f1117;
    --deep:      #141926;
    --surface:   #1a2133;
    --rim:       #1f2d44;
    --rim2:      #263650;
    --blue:      #4a90d9;
    --blue-lo:   rgba(74,144,217,0.10);
    --teal:      #2eb8a6;
    --teal-lo:   rgba(46,184,166,0.09);
    --gold:      #c9963a;
    --gold-lo:   rgba(201,150,58,0.10);
    --text:      #dde3ee;
    --muted:     #6b7a96;
    --dim:       #2e3d56;
    --ok:        #4ade80;
    --err:       #f87171;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
    --serif:     'IBM Plex Serif', serif;
    --rule:      linear-gradient(90deg, #4a90d9, #2eb8a6);
    --rule-v:    linear-gradient(180deg, #4a90d9, #2eb8a6);
}

.main .block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

.topbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 52px;
    background: rgba(15, 17, 23, 0.96);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid #1f2535;
    display: flex;
    align-items: center;
    padding: 0 16px;
    z-index: 99999;
    gap: 12px;
}
.topbar::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--rule);
    opacity: 0.6;
    pointer-events: none;
}
.topbar-wordmark {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text);
    white-space: nowrap;
}
.topbar-wordmark span { color: var(--teal); }
.topbar-course {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.05em;
    color: var(--muted);
    padding: 2px 6px;
    border: 1px solid var(--rim2);
    border-radius: 2px;
    white-space: nowrap;
}
.topbar-rule {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1f2535, transparent);
    min-width: 0;
}
.topbar-db {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.05em;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    flex-shrink: 0;
}
.db-led {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}

.page-shell {
    padding: 68px 16px 64px;
    max-width: 960px;
    margin: 0 auto;
}

@media (min-width: 640px) {
    .page-shell { padding: 76px 32px 64px; }
    .topbar { padding: 0 24px; gap: 16px; }
    .topbar-wordmark { font-size: 14px; letter-spacing: 0.12em; }
    .topbar-course { font-size: 10px; padding: 3px 8px; }
    .topbar-db { font-size: 10px; gap: 8px; }
}

.stButton > button {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--rim2) !important;
    border-radius: 2px !important;
    padding: 9px 22px !important;
    transition: all 0.15s ease !important;
    width: auto !important;
}
.stButton > button:hover {
    background: var(--blue-lo) !important;
    color: var(--blue) !important;
    border-color: var(--blue) !important;
}

[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] * {
    background: var(--deep) !important;
    color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed var(--rim2) !important;
    border-radius: 4px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--teal) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button span {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--rim2) !important;
}

.sec-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 20px;
    flex-wrap: wrap;
}
.sec-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--teal);
    flex-shrink: 0;
    padding: 3px 8px;
    border: 1px solid rgba(46,184,166,0.25);
    border-radius: 2px;
}
.sec-title {
    font-family: var(--sans);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text);
}
.sec-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1f2535, transparent);
    min-width: 40px;
}

.match-block {
    display: flex;
    background: var(--deep);
    border: 1px solid var(--rim);
    border-left: none;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 20px;
}
.match-ramp-strip {
    width: 3px;
    flex-shrink: 0;
    background: var(--rule-v);
}
.match-inner {
    padding: 16px 20px;
    flex: 1;
    min-width: 0;
}
.match-tag {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 6px;
}
.match-name {
    font-family: var(--serif);
    font-size: clamp(20px, 4.5vw, 32px);
    letter-spacing: 0.01em;
    color: var(--text);
    line-height: 1.2;
    word-break: break-word;
}
.no-match-block {
    background: var(--deep);
    border: 1px solid rgba(248,113,113,0.2);
    border-left: 3px solid var(--err);
    padding: 14px 18px;
    margin-bottom: 20px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--err);
    border-radius: 4px;
}

.stat-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
    width: 100%;
}
.stat-pill {
    background: var(--deep);
    border: 1px solid var(--rim);
    border-radius: 4px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
    width: 100%;
}
.stat-pill::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--rule);
    opacity: 0.5;
}
.stat-k {
    font-family: var(--mono);
    font-size: 8px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.stat-v {
    font-family: var(--mono);
    font-size: 18px;
    font-weight: 600;
    color: var(--blue);
}

.callout {
    background: var(--deep);
    border: 1px solid var(--rim);
    border-radius: 4px;
    padding: 16px 18px;
    margin-bottom: 16px;
}
.callout-head {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 8px;
}
.callout-body {
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.75;
    color: var(--muted);
    font-weight: 300;
}
.callout-body b { color: var(--text); font-weight: 500; }

[data-testid="stExpander"] {
    background: var(--deep) !important;
    border: 1px solid var(--rim) !important;
    border-radius: 4px !important;
}
details summary {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--rim) !important;
    border-radius: 4px !important;
    width: 100% !important;
}
thead tr th {
    background: var(--surface) !important;
    font-family: var(--mono) !important;
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--muted) !important;
}
tbody tr td {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--text) !important;
}

[data-testid="stProgressBar"] > div > div { background: var(--teal) !important; }

.stAudio audio { filter: invert(0.85) sepia(0.1); border-radius: 3px; width: 100%; }

[data-testid="stAlert"] {
    background: var(--deep) !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--text) !important;
}
[data-testid="stSpinner"] p {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    color: var(--muted) !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--rim) !important;
    margin: 24px 0 !important;
}

.project-header {
    padding: 24px 0;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--rim);
    position: relative;
}
.project-header::before {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 80px; height: 2px;
    background: var(--rule);
}
.project-eyebrow {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.project-eyebrow::before {
    content: '';
    display: inline-block;
    width: 16px; height: 1px;
    background: var(--teal);
    flex-shrink: 0;
}
.project-title {
    font-family: var(--serif);
    font-size: clamp(28px, 6vw, 48px);
    font-weight: 600;
    color: var(--text);
    line-height: 1.1;
    letter-spacing: -0.01em;
    margin: 0 0 6px;
}
.project-title-accent { color: var(--teal); }
.project-subtitle {
    font-family: var(--sans);
    font-size: clamp(12px, 2vw, 14px);
    font-weight: 300;
    color: var(--muted);
    margin: 10px 0 20px;
    line-height: 1.6;
    max-width: 560px;
}

.meta-strip {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0;
    border: 1px solid var(--rim);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 20px;
    width: 100%;
}
@media (min-width: 520px) {
    .meta-strip { grid-template-columns: repeat(4, 1fr); }
}
.meta-cell {
    padding: 10px 14px;
    border-right: 1px solid var(--rim);
    border-bottom: 1px solid var(--rim);
    min-width: 0;
}
@media (min-width: 520px) {
    .meta-cell { border-bottom: none; }
    .meta-cell:last-child { border-right: none; }
}
.meta-cell:nth-child(2) { @media (max-width: 519px) { border-right: none; } }
.meta-cell:nth-child(3), .meta-cell:nth-child(4) { border-bottom: none; }
.meta-cell:nth-child(3) { @media (max-width: 519px) { border-right: 1px solid var(--rim); } }

.meta-k {
    font-family: var(--mono);
    font-size: 8px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.meta-v {
    font-family: var(--sans);
    font-size: 12px;
    font-weight: 500;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.abstract {
    background: var(--deep);
    border: 1px solid var(--rim);
    border-radius: 4px;
    padding: 16px 18px;
    margin-bottom: 20px;
}
.abstract-label {
    font-family: var(--mono);
    font-size: 8px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 8px;
}
.abstract-text {
    font-family: var(--sans);
    font-size: 12px;
    line-height: 1.7;
    color: var(--muted);
    font-weight: 300;
}
.abstract-text b { color: var(--text); font-weight: 500; }

.tech-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 16px;
}
.tech-chip {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.05em;
    color: var(--muted);
    padding: 3px 8px;
    border: 1px solid var(--rim2);
    border-radius: 2px;
    background: var(--deep);
}

.tracklist {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--dim);
    letter-spacing: 0.05em;
    padding-top: 8px;
    line-height: 1.7;
    word-break: break-all;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border: 1px solid var(--rim) !important;
    background-color: transparent !important;
    padding: 8px 16px !important;
    border-radius: 4px 4px 0 0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--teal) !important;
    border-color: var(--teal) var(--teal) transparent var(--teal) !important;
    background-color: var(--deep) !important;
}
div[data-testid="stTabActiveIndicator"] {
        background-color: #c9963a !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_db():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
                db = pickle.load(f)
            if hasattr(db, "songs") and len(db.songs) > 0:
                return db, "precomputed"
        except Exception:
            pass
    if os.path.isdir(SONGS_DIR) and any(
        f.lower().endswith((".mp3", ".wav", ".m4a")) for f in os.listdir(SONGS_DIR)
    ):
        db = FingerprintDB(win_length=WIN_LENGTH, hop_length=HOP_LENGTH, sr=SR)
        db.build_from_folder(SONGS_DIR)
        try:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            with open(DB_PATH, "wb") as f:
                pickle.dump(db, f)
        except Exception:
            pass
        return db, "built_live"
    return None, "missing"

def save_upload_to_tmp(uf):
    suffix = os.path.splitext(uf.name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uf.getbuffer())
        return tmp.name

RC = {
    "figure.facecolor":  "#141926",
    "axes.facecolor":    "#0f1117",
    "axes.edgecolor":    "#263650",
    "axes.labelcolor":   "#6b7a96",
    "axes.titlecolor":   "#8a9bb8",
    "xtick.color":       "#2e3d56",
    "ytick.color":       "#2e3d56",
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "axes.labelsize":    8,
    "axes.titlesize":    9,
    "text.color":        "#dde3ee",
    "grid.color":        "#1f2d44",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
}

BLUE  = "#4a90d9"
TEAL  = "#2eb8a6"
GOLD  = "#c9963a"
DIM   = "#2e3d56"
DIM2  = "#6b7a96"
TEXT  = "#dde3ee"
BG0   = "#0f1117"
BG1   = "#141926"

def fig_spectrogram(y, sr, win_length, hop_length):
    freqs, times, S_db = spectrogram_db(y, sr, win_length, hop_length)
    peaks = find_constellation_peaks(S_db, freqs, times)

    with plt.rc_context(RC):
        fig = plt.Figure(figsize=(9, 4.2), constrained_layout=True)
        gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[24, 1], wspace=0.02)
        ax  = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])

        im = ax.pcolormesh(
            times, freqs / 1000, S_db,
            shading="auto", cmap="viridis", vmin=-80, vmax=0, rasterized=True
        )
        if peaks:
            pt = [p["time_s"]  for p in peaks]
            pf = [p["freq_hz"] / 1000 for p in peaks]
            ax.scatter(pt, pf, s=18, facecolors="none",
                       edgecolors=TEXT, linewidths=0.9, alpha=0.8, zorder=3)

        ax.set_ylim(0, min(5000, sr / 2) / 1000)
        ax.set_xlabel("time  (s)")
        ax.set_ylabel("frequency  (kHz)")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.25)

        cb = fig.colorbar(im, cax=cax)
        cb.set_label("dB", color=DIM2, fontsize=7)
        cb.ax.yaxis.set_tick_params(color=DIM2, labelsize=6)

        ax.text(0.01, 0.97, f"{len(peaks)} peaks",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7, color=GOLD,
                bbox=dict(boxstyle="round,pad=0.2", fc=BG1, ec="#263650", alpha=0.9))

    return fig, peaks, freqs, times, S_db

def fig_histogram(histogram, best_name, best_votes, runner_votes):
    with plt.rc_context(RC):
        fig = plt.Figure(figsize=(9, 4.0))
        
        if not histogram:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "no matching hashes", ha="center", va="center", color=DIM2)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        offs   = sorted(histogram.keys())
        counts = [histogram[o] for o in offs]
        best_off   = max(histogram, key=histogram.get)
        peak_count = histogram[best_off]

        ax1.bar(offs, counts, width=max(1, (max(offs)-min(offs))/len(offs)*0.9),
                color=BLUE, alpha=0.45, zorder=2)
        ax1.axvline(best_off, color="#fff", lw=1.0, alpha=0.2, zorder=3)
        ax1.set_yscale("log")
        ax1.set_xlabel("time offset  (frames)")
        ax1.set_ylabel("votes  (log)")
        ax1.grid(True, axis="y", alpha=0.2)

        ax1.bar([best_off], [peak_count],
                width=max(1, (max(offs)-min(offs))/len(offs)*0.9),
                color=TEAL, alpha=1.0, zorder=4)

        ax1.text(0.98, 0.95, f"peak @ {best_off}",
                 transform=ax1.transAxes, va="top", ha="right",
                 fontsize=7, color=TEAL,
                 bbox=dict(boxstyle="round,pad=0.2", fc=BG1, ec="#263650", alpha=0.9))

        w    = 60
        zoom = [o for o in offs if abs(o - best_off) <= w]
        zcnt = [histogram[o] for o in zoom]

        bw = max(1, w // max(len(zoom), 1))
        ax2.bar(zoom, zcnt, width=bw, color=DIM2, alpha=0.45, zorder=2)
        ax2.bar([best_off], [peak_count], width=bw, color=TEAL, alpha=1.0, zorder=3)
        ax2.axvline(best_off, color="#fff", lw=0.8, alpha=0.18)
        ax2.set_xlabel("time offset  (frames)")
        ax2.set_ylabel("votes")
        ax2.grid(True, axis="y", alpha=0.2)
        ax2.set_title(f"zoom ±{w} frames · {peak_count} votes", color=DIM2, fontsize=8, pad=4)

    return fig

db, db_status = load_db()
song_count = len(db.songs) if db else 0
led_color  = "#4ade80" if db_status != "missing" else "#f87171"
db_label   = f"{song_count} songs" if db_status != "missing" else "no database"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-wordmark">HASH<span>FLASH</span></div>
  <div class="topbar-course">EE200</div>
  <div class="topbar-rule"></div>
  <div class="topbar-db">
    <span class="db-led" style="background:{led_color}; box-shadow:0 0 6px {led_color}88;"></span>
    {db_label}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-shell">', unsafe_allow_html=True)

if db_status == "missing":
    st.error("No song database found. Add .mp3/.wav files to `songs/` and run `python build_database.py`.")
    st.stop()

song_names       = list(db.songs.values())
song_list_inline = "  ·  ".join(song_names)

st.markdown(f"""
<div class="project-header">
  <div class="project-eyebrow">EE200 Project · Signal Processing</div>
  <div class="project-title">Hash<span class="project-title-accent">Flash</span></div>
  <p class="project-subtitle">
    A content-based audio identification system implementing the Shazam landmark hashing algorithm.
    Identifies recordings against an indexed database using sparse time-frequency peaks
    and combinatorial hash voting.
  </p>

  <div class="meta-strip">
    <div class="meta-cell">
      <div class="meta-k">Authors</div>
      <div class="meta-v">Darsh Kedia &amp; Tulip Khatri</div>
    </div>
    <div class="meta-cell">
      <div class="meta-k">Course</div>
      <div class="meta-v">EE200</div>
    </div>
    <div class="meta-cell">
      <div class="meta-k">Method</div>
      <div class="meta-v">Paired-peak</div>
    </div>
    <div class="meta-cell">
      <div class="meta-k">Database</div>
      <div class="meta-v">{song_count} indexed</div>
    </div>
  </div>

  <div class="abstract">
    <div class="abstract-label">Abstract</div>
    <div class="abstract-text">
      Audio is converted to a spectrogram via STFT with a <b>Hann window ({WIN_LENGTH} samples, {HOP_LENGTH}-sample hop)</b> at <b>{SR} Hz</b>. Sparse constellation peaks are extracted from local spectral maxima and paired within a target zone to produce compact <b>(f₁, f₂, Δt)</b> hashes. A query clip is matched by looking up its hashes against a database and finding the candidate song whose matches cluster at a single consistent time offset.
    </div>
  </div>

  <div class="tech-row">
    <span class="tech-chip">FFT / STFT</span>
    <span class="tech-chip">Constellation map</span>
    <span class="tech-chip">Hash table lookup</span>
    <span class="tech-chip">Offset voting</span>
    <span class="tech-chip">WIN={WIN_LENGTH}</span>
    <span class="tech-chip">HOP={HOP_LENGTH}</span>
    <span class="tech-chip">SR={SR}Hz</span>
  </div>

  <div class="tracklist">{song_list_inline}</div>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["Single Clip", "Batch Mode"])

with tab_single:
    st.markdown("""
    <div class="sec-head" style="margin-top:12px">
      <span class="sec-label">input</span>
      <span class="sec-title">Upload query clip</span>
      <span class="sec-line"></span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "clip", type=["mp3", "wav", "m4a", "flac"],
        label_visibility="collapsed"
    )

    if uploaded:
        tmp_path = save_upload_to_tmp(uploaded)
        try:
            with st.spinner("decoding audio…"):
                y, sr = load_audio(tmp_path, sr=SR)
            st.audio(uploaded)

            with st.spinner("fingerprinting and matching…"):
                result = db.match(y, mode="paired")

            best      = result["best"]
            histogram = result["histogram"]
            ranked    = result["ranked"]

            st.markdown("""
            <div class="sec-head" style="margin-top:28px">
              <span class="sec-label">result</span>
              <span class="sec-title">Identification</span>
              <span class="sec-line"></span>
            </div>
            """, unsafe_allow_html=True)

            if best is None:
                st.markdown(
                    '<div class="no-match-block">✕ No match - clip does not correspond to any indexed song.</div>',
                    unsafe_allow_html=True
                )
            else:
                top_votes  = ranked[0][2]
                runner_up  = ranked[1][2] if len(ranked) > 1 else 0
                ratio      = f"{top_votes/max(runner_up,1):.1f}×" if runner_up else "—"

                st.markdown(f"""
                <div class="match-block">
                  <div class="match-ramp-strip"></div>
                  <div class="match-inner">
                    <div class="match-tag">Identified song</div>
                    <div class="match-name">{best}</div>
                  </div>
                </div>
                <div class="stat-row">
                  <div class="stat-pill">
                    <div class="stat-k">Best offset votes</div>
                    <div class="stat-v">{top_votes}</div>
                  </div>
                  <div class="stat-pill">
                    <div class="stat-k">Runner-up votes</div>
                    <div class="stat-v">{runner_up}</div>
                  </div>
                  <div class="stat-pill">
                    <div class="stat-k">Confidence ratio</div>
                    <div class="stat-v">{ratio}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("all candidates"):
                    st.dataframe(
                        pd.DataFrame([(n, v) for _, n, v in ranked], columns=["song", "votes"]),
                        use_container_width=True,
                        hide_index=True
                    )

            st.markdown("""
            <div class="sec-head" style="margin-top:28px">
              <span class="sec-label">analysis</span>
              <span class="sec-title">Intermediate steps</span>
              <span class="sec-line"></span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="callout">
              <div class="callout-head">Step A: Spectrogram &amp; constellation map</div>
              <div class="callout-body">
                The audio is divided into overlapping windows (<b>win={win}</b> samples, hop=<b>{hop}</b> at {sr} Hz) with a <b>Hann taper</b> and passed through an FFT. <b>White circles</b> mark <em>constellation peaks</em> local maxima exceeding a −40 dB floor, thinned to 5 per frame.
              </div>
            </div>
            """.format(win=WIN_LENGTH, hop=HOP_LENGTH, sr=SR), unsafe_allow_html=True)

            fig_s, peaks, freqs, times, S_db = fig_spectrogram(y, sr, WIN_LENGTH, HOP_LENGTH)
            st.pyplot(fig_s, clear_figure=True)

            freq_res = SR / WIN_LENGTH
            time_res = HOP_LENGTH / SR * 1000

            st.markdown(f"""
            <div class="stat-row" style="margin-top:10px">
              <div class="stat-pill">
                <div class="stat-k">Constellation peaks</div>
                <div class="stat-v">{len(peaks)}</div>
              </div>
              <div class="stat-pill">
                <div class="stat-k">Freq resolution</div>
                <div class="stat-v">{freq_res:.1f} Hz/bin</div>
              </div>
              <div class="stat-pill">
                <div class="stat-k">Time resolution</div>
                <div class="stat-v">{time_res:.1f} ms/frame</div>
              </div>
              <div class="stat-pill">
                <div class="stat-k">Clip duration</div>
                <div class="stat-v">{len(y)/SR:.2f} s</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("""
            <div class="callout">
              <div class="callout-head">Step B: Combinatorial hashing &amp; offset histogram</div>
              <div class="callout-body">
                Each peak is paired with up to 10 later peaks within a zone of [1, 100] frames to create hashes <em>(f₁_bin, f₂_bin, Δt)</em>. For every database match, the <b>time offset</b> = db_frame − query_frame is noted. True matches create a localized spike, while mismatches distribute randomly.
              </div>
            </div>
            """, unsafe_allow_html=True)

            top_v = ranked[0][2] if ranked else 0
            run_v = ranked[1][2] if len(ranked) > 1 else 0
            fig_h = fig_histogram(histogram, best or "—", top_v, run_v)
            st.pyplot(fig_h, clear_figure=True)

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.close(os.open(tmp_path, os.O_RDONLY))
                    os.remove(tmp_path)
                except Exception:
                    pass

with tab_batch:
    st.markdown("""
    <div class="sec-head" style="margin-top:12px">
      <span class="sec-label">input</span>
      <span class="sec-title">Upload query clips</span>
      <span class="sec-line"></span>
    </div>
    <div class="callout" style="margin-bottom:20px">
      <div class="callout-head">Batch mode</div>
      <div class="callout-body">
        Upload multiple query signals. Results can be inspected immediately or downloaded directly as a structured <b>results.csv</b> array layout.
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploads = st.file_uploader(
        "clips", type=["mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploads:
        if st.button(f"Run Batch Execution ({len(uploads)})"):
            rows = []
            bar = st.progress(0.0, text="Initializing execution processing sequence...")
            
            for i, uf in enumerate(uploads):
                tmp_path = save_upload_to_tmp(uf)
                try:
                    y, _ = load_audio(tmp_path, sr=SR)
                    res  = db.match(y, mode="paired")
                    rows.append({"filename": uf.name, "prediction": res["best"] or "No Match"})
                except Exception as e:
                    rows.append({"filename": uf.name, "prediction": f"Error: {str(e)}"})
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.close(os.open(tmp_path, os.O_RDONLY))
                            os.remove(tmp_path)
                        except Exception:
                            pass
                
                bar.progress((i + 1) / len(uploads), text=f"Evaluating pipeline matrix: {uf.name}")

            df = pd.DataFrame(rows, columns=["filename", "prediction"])
            
            st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )

st.markdown('</div>', unsafe_allow_html=True)