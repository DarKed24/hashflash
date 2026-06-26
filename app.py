"""
app.py — AudioTag  (audio fingerprinting demo)
EE200 Course Project · Darsh Kedia & Tulip Khatri
"""

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
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker

from audio_io import load_audio, spectrogram_db
from fingerprint import FingerprintDB, find_constellation_peaks

# ── Constants ─────────────────────────────────────────────────────────────────
WIN_LENGTH = 4096
HOP_LENGTH = 2048
SR         = 22050
DB_PATH    = "data/fingerprint_db.pkl"
SONGS_DIR  = "songs"

st.set_page_config(
    page_title="AudioTag — EE200 Project",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Serif:wght@400;500;600&display=swap');

/* ── Force dark background everywhere ── */
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
*, *::before, *::after { box-sizing: border-box; }

/* ── Design tokens ── */
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

/* ── Layout ── */
.main .block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

/* ── Fixed top bar ── */
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
    padding: 0 24px;
    z-index: 99999;
    gap: 16px;
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
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: var(--text);
    white-space: nowrap;
}
.topbar-wordmark span { color: var(--teal); }
.topbar-course {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--muted);
    padding: 3px 8px;
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
    font-size: 10px;
    letter-spacing: 0.07em;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
    flex-shrink: 0;
}
.db-led {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Page shell ── */
.page-shell {
    padding: 68px 24px 64px;
    max-width: 960px;
    margin: 0 auto;
}
@media (min-width: 640px) {
    .page-shell { padding: 76px 44px 64px; }
    .topbar { padding: 0 36px; }
}

/* ── Buttons ── */
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
}
.stButton > button:hover {
    background: var(--blue-lo) !important;
    color: var(--blue) !important;
    border-color: var(--blue) !important;
}

/* ── Upload zone — force dark bg and visible text in light mode ── */
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
/* Browse files button inside uploader */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button span {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--rim2) !important;
}

/* ── Section heads ── */
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

/* ── Match result ── */
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
    padding: 20px 24px;
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
    font-size: clamp(24px, 5vw, 38px);
    letter-spacing: 0.01em;
    color: var(--text);
    line-height: 1.15;
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

/* ── Stat pills ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 8px;
    margin-bottom: 20px;
}
.stat-pill {
    background: var(--deep);
    border: 1px solid var(--rim);
    border-radius: 4px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
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
    font-size: 20px;
    font-weight: 600;
    color: var(--blue);
}

/* ── Callout boxes ── */
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

/* ── Expander ── */
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

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--rim) !important;
    border-radius: 4px !important;
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

/* ── Progress ── */
[data-testid="stProgressBar"] > div > div { background: var(--teal) !important; }

/* ── Audio player ── */
.stAudio audio { filter: invert(0.85) sepia(0.1); border-radius: 3px; }

/* ── Alerts / spinner ── */
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

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--rim) !important;
    margin: 32px 0 !important;
}

/* ── Song list ── */
.song-row {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    padding: 6px 0;
    border-bottom: 1px solid var(--rim);
    display: flex;
    align-items: center;
    gap: 10px;
}
.song-row::before {
    content: '▸';
    color: var(--teal);
    font-size: 8px;
}

/* ── Header block ── */
.project-header {
    padding: 44px 0 36px;
    margin-bottom: 36px;
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
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.project-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px; height: 1px;
    background: var(--teal);
    flex-shrink: 0;
}
.project-title {
    font-family: var(--serif);
    font-size: clamp(32px, 7vw, 60px);
    font-weight: 600;
    color: var(--text);
    line-height: 1.05;
    letter-spacing: -0.01em;
    margin: 0 0 6px;
}
.project-title-accent { color: var(--teal); }
.project-subtitle {
    font-family: var(--sans);
    font-size: clamp(13px, 2vw, 15px);
    font-weight: 300;
    color: var(--muted);
    margin: 10px 0 28px;
    line-height: 1.65;
    max-width: 560px;
}

/* ── Author / meta strip ── */
.meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    border: 1px solid var(--rim);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 24px;
}
.meta-cell {
    padding: 12px 18px;
    border-right: 1px solid var(--rim);
    flex: 1;
    min-width: 120px;
}
.meta-cell:last-child { border-right: none; }
.meta-k {
    font-family: var(--mono);
    font-size: 8px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 5px;
}
.meta-v {
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
}

/* ── Abstract ── */
.abstract {
    background: var(--deep);
    border: 1px solid var(--rim);
    border-radius: 4px;
    padding: 20px 22px;
    margin-bottom: 28px;
}
.abstract-label {
    font-family: var(--mono);
    font-size: 8px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 10px;
}
.abstract-text {
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.8;
    color: var(--muted);
    font-weight: 300;
}
.abstract-text b { color: var(--text); font-weight: 500; }

/* ── Tech meta row ── */
.tech-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}
.tech-chip {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.1em;
    color: var(--muted);
    padding: 4px 10px;
    border: 1px solid var(--rim2);
    border-radius: 2px;
    background: var(--deep);
}

/* ── Tracklist inline ── */
.tracklist {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--dim);
    letter-spacing: 0.06em;
    padding-top: 12px;
    line-height: 1.9;
}
</style>
""", unsafe_allow_html=True)

# ── DB loading ────────────────────────────────────────────────────────────────
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


# ── Matplotlib dark theme ─────────────────────────────────────────────────────
RC = {
    "figure.facecolor":  "#141926",
    "axes.facecolor":    "#0f1117",
    "axes.edgecolor":    "#263650",
    "axes.labelcolor":   "#6b7a96",
    "axes.titlecolor":   "#8a9bb8",
    "xtick.color":       "#2e3d56",
    "ytick.color":       "#2e3d56",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "text.color":        "#dde3ee",
    "grid.color":        "#1f2d44",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
}

BLUE    = "#4a90d9"
TEAL    = "#2eb8a6"
GOLD    = "#c9963a"
DIM     = "#2e3d56"
DIM2    = "#6b7a96"
TEXT    = "#dde3ee"
BG0     = "#0f1117"
BG1     = "#141926"


def fig_spectrogram(y, sr, win_length, hop_length):
    freqs, times, S_db = spectrogram_db(y, sr, win_length, hop_length)
    peaks = find_constellation_peaks(S_db, freqs, times)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(9, 4.2), constrained_layout=True)
        gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[20, 1], wspace=0.04)
        ax  = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])

        im = ax.pcolormesh(
            times, freqs / 1000, S_db,
            shading="auto", cmap="viridis", vmin=-80, vmax=0, rasterized=True
        )
        if peaks:
            pt = [p["time_s"]  for p in peaks]
            pf = [p["freq_hz"] / 1000 for p in peaks]
            ax.scatter(pt, pf, s=26, facecolors="none",
                       edgecolors=TEAL, linewidths=1.1, alpha=0.9, zorder=3)

        ax.set_ylim(0, min(5000, sr / 2) / 1000)
        ax.set_xlabel("time  (s)")
        ax.set_ylabel("frequency  (kHz)")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.35)

        cb = fig.colorbar(im, cax=cax)
        cb.set_label("dB", color=DIM2, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=DIM2, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=DIM2)

        ax.text(0.01, 0.97,
                f"{len(peaks)} peaks",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color=GOLD,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG1, ec="#263650", alpha=0.9))

    return fig, peaks, freqs, times, S_db


def fig_histogram(histogram, best_name, best_votes, runner_votes):
    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8),
                                        gridspec_kw={"width_ratios": [3, 2]})
        fig.subplots_adjust(wspace=0.35)

        if not histogram:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, "no matching hashes",
                        ha="center", va="center", color=DIM2)
            return fig

        offs   = sorted(histogram.keys())
        counts = [histogram[o] for o in offs]
        best_off   = max(histogram, key=histogram.get)
        peak_count = histogram[best_off]

        ax1.bar(offs, counts, width=max(1, (max(offs)-min(offs))/len(offs)*0.9),
                color=BLUE, alpha=0.45, zorder=2)
        ax1.axvline(best_off, color="#fff", lw=1.1, alpha=0.2, zorder=3)
        ax1.set_yscale("log")
        ax1.set_xlabel("time offset  (frames)")
        ax1.set_ylabel("votes  (log)")
        ax1.grid(True, axis="y", alpha=0.3)

        ax1.bar([best_off], [peak_count],
                width=max(1, (max(offs)-min(offs))/len(offs)*0.9),
                color=TEAL, alpha=1.0, zorder=4)

        ax1.text(0.98, 0.97, f"peak @ {best_off}",
                 transform=ax1.transAxes, va="top", ha="right",
                 fontsize=8, color=TEAL,
                 bbox=dict(boxstyle="round,pad=0.3", fc=BG1, ec="#263650", alpha=0.9))

        w    = 60
        zoom = [o for o in offs if abs(o - best_off) <= w]
        zcnt = [histogram[o] for o in zoom]

        bw = max(1, w // max(len(zoom), 1))
        ax2.bar(zoom, zcnt, width=bw, color=DIM2, alpha=0.45, zorder=2)
        ax2.bar([best_off], [peak_count], width=bw, color=TEAL, alpha=1.0, zorder=3)
        ax2.axvline(best_off, color="#fff", lw=1, alpha=0.18)
        ax2.set_xlabel("time offset  (frames)")
        ax2.set_ylabel("votes")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.set_title(f"zoom ±{w} frames · {peak_count} votes", color=DIM2, fontsize=9, pad=6)

    return fig


# ── State ─────────────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "single"

db, db_status = load_db()
song_count = len(db.songs) if db else 0
led_color  = "#4ade80" if db_status != "missing" else "#f87171"
db_label   = f"{song_count} songs indexed" if db_status != "missing" else "no database"

# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div class="topbar-wordmark">AUDIO<span>TAG</span></div>
  <div class="topbar-course">EE200</div>
  <div class="topbar-rule"></div>
  <div class="topbar-db">
    <span class="db-led" style="background:{led_color}; box-shadow:0 0 6px {led_color}88;"></span>
    {db_label}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Page shell ────────────────────────────────────────────────────────────────
st.markdown('<div class="page-shell">', unsafe_allow_html=True)

if db_status == "missing":
    st.error("No song database found. Add .mp3/.wav files to `songs/` and run `python build_database.py`.")
    st.stop()

# ── Project header ────────────────────────────────────────────────────────────
song_names       = list(db.songs.values())
song_list_inline = "  ·  ".join(song_names)

st.markdown(f"""
<div class="project-header">
  <div class="project-eyebrow">EE200 — Course Project · Audio Signal Processing</div>
  <div class="project-title">Audio<span class="project-title-accent">Tag</span></div>
  <p class="project-subtitle">
    A content-based audio identification system implementing the Shazam landmark hashing algorithm.
    Identifies recordings against an indexed database using sparse time–frequency peaks
    and combinatorial hash voting — no machine learning, no waveform comparison.
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
      <div class="meta-v">Paired-peak hashing</div>
    </div>
    <div class="meta-cell">
      <div class="meta-k">Database</div>
      <div class="meta-v">{song_count} songs indexed</div>
    </div>
  </div>

  <div class="abstract">
    <div class="abstract-label">Abstract</div>
    <div class="abstract-text">
      This project implements the audio fingerprinting technique described by Wang (2003) — the algorithm
      underlying Shazam. Audio is converted to a time–frequency spectrogram via short-time Fourier
      transform with a <b>Hann window ({WIN_LENGTH} samples, {HOP_LENGTH}-sample hop)</b> at
      <b>{SR} Hz</b>. Sparse constellation peaks are extracted from local spectral maxima and
      paired within a target zone to produce compact <b>(f₁, f₂, Δt)</b> hashes. A query clip is
      matched by looking up its hashes against a pre-indexed database and finding the candidate song
      whose hash matches cluster at a single time offset — evidence of a true alignment rather than
      coincidental collision.
    </div>
  </div>

  <div class="tech-row">
    <span class="tech-chip">FFT / STFT</span>
    <span class="tech-chip">Constellation map</span>
    <span class="tech-chip">Hash table lookup</span>
    <span class="tech-chip">Offset voting</span>
    <span class="tech-chip">WIN = {WIN_LENGTH}</span>
    <span class="tech-chip">HOP = {HOP_LENGTH}</span>
    <span class="tech-chip">SR = {SR} Hz</span>
  </div>

  <div class="tracklist">{song_list_inline}</div>
</div>
""", unsafe_allow_html=True)

if db_status == "built_live":
    st.info("Database re-indexed live at startup — fingerprint_db.pkl has been saved.")

# ── Mode tabs ─────────────────────────────────────────────────────────────────
c1, c2, _ = st.columns([1, 1, 6])
with c1:
    if st.button("Single clip", key="t1"):
        st.session_state.mode = "single"
with c2:
    if st.button("Batch", key="t2"):
        st.session_state.mode = "batch"

mode = st.session_state.mode
st.markdown("<hr>", unsafe_allow_html=True)


# ══ SINGLE-CLIP MODE ══════════════════════════════════════════════════════════
if mode == "single":

    st.markdown("""
    <div class="sec-head">
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
        tmp = save_upload_to_tmp(uploaded)
        try:
            with st.spinner("decoding audio…"):
                y, sr = load_audio(tmp, sr=SR)
            st.audio(uploaded)

            with st.spinner("fingerprinting and matching…"):
                result = db.match(y, mode="paired")

            best      = result["best"]
            histogram = result["histogram"]
            ranked    = result["ranked"]

            # ── Result ──────────────────────────────────────────────────────
            st.markdown("""
            <div class="sec-head" style="margin-top:28px">
              <span class="sec-label">result</span>
              <span class="sec-title">Identification</span>
              <span class="sec-line"></span>
            </div>
            """, unsafe_allow_html=True)

            if best is None:
                st.markdown(
                    '<div class="no-match-block">✕ No match — clip does not correspond to any indexed song.</div>',
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
                    st.table(pd.DataFrame(
                        [(n, v) for _, n, v in ranked],
                        columns=["song", "votes"],
                    ))

            # ── Intermediate steps ───────────────────────────────────────────
            st.markdown("""
            <div class="sec-head" style="margin-top:28px">
              <span class="sec-label">analysis</span>
              <span class="sec-title">Intermediate steps</span>
              <span class="sec-line"></span>
            </div>
            """, unsafe_allow_html=True)

            # Step A: spectrogram + constellation
            st.markdown("""
            <div class="callout">
              <div class="callout-head">Step A — Spectrogram &amp; constellation map</div>
              <div class="callout-body">
                The audio is divided into short overlapping windows (<b>win={win}</b> samples,
                hop=<b>{hop}</b> at {sr} Hz). Each window is multiplied by a <b>Hann taper</b>
                to suppress spectral leakage, then passed through an FFT. Stacking these
                column-by-column gives the time–frequency power map below. <b>Teal circles</b>
                mark the <em>constellation peaks</em> — local maxima exceeding the neighbourhood
                and a −40 dB floor, thinned to at most 5 per time-frame. These sparse landmarks
                are what gets fingerprinted.
              </div>
            </div>
            """.format(win=WIN_LENGTH, hop=HOP_LENGTH, sr=SR), unsafe_allow_html=True)

            fig_s, peaks, freqs, times, S_db = fig_spectrogram(y, sr, WIN_LENGTH, HOP_LENGTH)
            st.pyplot(fig_s)
            plt.close(fig_s)

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

            # Step B: offset histogram
            st.markdown("""
            <div class="callout">
              <div class="callout-head">Step B — Combinatorial hashing &amp; offset histogram</div>
              <div class="callout-body">
                Each constellation peak is <b>paired</b> with up to 10 later peaks within a
                target zone of [1, 100] frames. Each pair produces a hash
                <em>(f₁_bin, f₂_bin, Δt)</em>. These hashes are looked up in the database;
                for every match the <b>time offset</b> = db_frame − query_frame is recorded.
                A true match accumulates votes at <b>one consistent offset</b> (the spike),
                while false songs scatter votes randomly. The song with the highest
                single-offset vote count wins.
              </div>
            </div>
            """, unsafe_allow_html=True)

            top_v = ranked[0][2] if ranked else 0
            run_v = ranked[1][2] if len(ranked) > 1 else 0
            fig_h = fig_histogram(histogram, best or "—", top_v, run_v)
            st.pyplot(fig_h)
            plt.close(fig_h)

        finally:
            os.remove(tmp)


# ══ BATCH MODE ════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="sec-head">
      <span class="sec-label">input</span>
      <span class="sec-title">Upload query clips</span>
      <span class="sec-line"></span>
    </div>
    <div class="callout" style="margin-bottom:20px">
      <div class="callout-head">Batch mode</div>
      <div class="callout-body">
        Upload any number of clips. Each is fingerprinted and matched independently.
        The output <b>results.csv</b> contains two columns:
        <b>filename</b> (original upload name) and <b>prediction</b>
        (matched song name, blank if no match).
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploads = st.file_uploader(
        "clips", type=["mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploads and st.button(f"Run on {len(uploads)} clip{'s' if len(uploads) != 1 else ''}"):
        rows = []
        bar  = st.progress(0.0, text="")
        for i, uf in enumerate(uploads):
            tmp = save_upload_to_tmp(uf)
            try:
                y, _ = load_audio(tmp, sr=SR)
                res  = db.match(y, mode="paired")
                rows.append({"filename": uf.name, "prediction": res["best"] or ""})
            except Exception as e:
                rows.append({"filename": uf.name, "prediction": ""})
                st.warning(f"{uf.name}: {e}")
            finally:
                os.remove(tmp)
            bar.progress((i + 1) / len(uploads), text=f"{uf.name}")

        df = pd.DataFrame(rows, columns=["filename", "prediction"])

        st.markdown("""
        <div class="sec-head" style="margin-top:28px">
          <span class="sec-label">result</span>
          <span class="sec-title">Results</span>
          <span class="sec-line"></span>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download results.csv",
            df.to_csv(index=False).encode(),
            "results.csv", "text/csv"
        )


# ── Footer: indexed songs ──────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander(f"indexed songs  ({song_count})"):
    for name in db.songs.values():
        st.markdown(f'<div class="song-row">{name}</div>', unsafe_allow_html=True)

# ── Footer attribution ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 24px 0 8px; font-family: var(--mono); font-size: 9px;
            letter-spacing: 0.14em; color: var(--dim); text-align: center;
            border-top: 1px solid var(--rim); margin-top: 12px;">
  AUDIOTAG &nbsp;·&nbsp; EE200 Course Project &nbsp;·&nbsp;
  Darsh Kedia &amp; Tulip Khatri &nbsp;·&nbsp;
  Wang (2003) landmark hashing
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # .page-shell