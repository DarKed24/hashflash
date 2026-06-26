"""
app.py — Zapp-tain America  (audio fingerprinting demo)
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
    page_title="Zapp-tain America",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Hard-lock every Streamlit container to dark ── */
html,
body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
.main,
.block-container {
    background-color: #0c0c0f !important;
    color: #d4d4d8 !important;
}

header[data-testid="stHeader"] {
    background-color: #0c0c0f !important;
    border-bottom: 1px solid #1e1e28 !important;
}

[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }

*, *::before, *::after { box-sizing: border-box; }

/* ── Design tokens ── */
:root {
    --bg:       #0c0c0f;
    --s1:       #13131a;
    --s2:       #18181f;
    --border:   #1e1e28;
    --border2:  #2a2a38;
    --amber:    #ff8c42;
    --amber-lo: rgba(255,140,66,0.12);
    --text:     #d4d4d8;
    --dim:      #52525b;
    --dim2:     #71717a;
    --ok:       #4ade80;
    --err:      #f87171;
    --mono:     'Space Mono', monospace;
    --display:  'Syne', sans-serif;
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
    height: 56px;
    background: #0c0c0f;
    border-bottom: 1px solid #1e1e28;
    display: flex;
    align-items: center;
    padding: 0 36px;
    gap: 0;
    z-index: 99999;
}
.topbar-wordmark {
    font-family: var(--display);
    font-size: 15px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #fff;
}
.topbar-wordmark span { color: var(--amber); }
.topbar-rule {
    flex: 1;
    height: 1px;
    background: #1e1e28;
    margin: 0 24px;
}
.topbar-db {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.06em;
    color: var(--dim2);
    display: flex;
    align-items: center;
    gap: 7px;
}
.db-led {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* Push content below topbar */
.page-shell {
    padding: 80px 44px 60px;
    max-width: 1120px;
    margin: 0 auto;
}

/* ── Mode tabs ── */
.mode-tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid #1e1e28;
    margin-bottom: 40px;
}
.mtab {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 10px 24px;
    border: none;
    background: transparent;
    cursor: pointer;
    color: var(--dim2);
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color 0.15s;
}
.mtab:hover { color: var(--text); }
.mtab-on {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
}

/* ── Section heading ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 0 0 20px;
}
.sec-num {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--amber);
    letter-spacing: 0.12em;
    flex-shrink: 0;
}
.sec-title {
    font-family: var(--display);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text);
}
.sec-line {
    flex: 1;
    height: 1px;
    background: #1e1e28;
}

/* ── Upload zone ── */
[data-testid="stFileUploaderDropzone"] {
    background: #13131a !important;
    border: 1px dashed #2a2a38 !important;
    border-radius: 6px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--amber) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--dim2) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
    font-family: var(--mono) !important;
    color: var(--dim) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--amber) !important;
    border: 1px solid var(--amber) !important;
    border-radius: 4px !important;
    padding: 10px 24px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: var(--amber-lo) !important;
}

/* ── Match result ── */
.match-block {
    background: #13131a;
    border: 1px solid #1e1e28;
    border-left: 3px solid var(--amber);
    padding: 22px 26px;
    margin-bottom: 28px;
    display: flex;
    align-items: flex-start;
    gap: 20px;
}
.match-tag {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 5px;
}
.match-name {
    font-family: var(--display);
    font-size: 24px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.01em;
    line-height: 1.1;
}
.no-match-block {
    background: #13131a;
    border-left: 3px solid var(--err);
    padding: 16px 20px;
    margin-bottom: 28px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--err);
}

/* ── Stat pills ── */
.stat-row { display: flex; gap: 12px; margin-bottom: 28px; }
.stat-pill {
    flex: 1;
    background: #13131a;
    border: 1px solid #1e1e28;
    padding: 14px 18px;
}
.stat-k {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--dim2);
    margin-bottom: 5px;
}
.stat-v {
    font-family: var(--mono);
    font-size: 26px;
    font-weight: 700;
    color: var(--amber);
}

/* ── Callout / annotation box ── */
.callout {
    background: #13131a;
    border: 1px solid #1e1e28;
    border-radius: 4px;
    padding: 14px 16px;
    margin-top: 12px;
}
.callout-head {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 6px;
}
.callout-body {
    font-family: var(--display);
    font-size: 12px;
    color: var(--dim2);
    line-height: 1.6;
}
.callout-body b { color: var(--text); font-weight: 600; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #13131a !important;
    border: 1px solid #1e1e28 !important;
    border-radius: 4px !important;
}
details summary {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--dim2) !important;
}

/* ── Tables ── */
[data-testid="stDataFrame"] { border: 1px solid #1e1e28 !important; border-radius: 4px !important; }
thead tr th {
    background: #13131a !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--dim2) !important;
}
tbody tr td {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--text) !important;
}

/* ── Progress ── */
[data-testid="stProgressBar"] > div > div { background: var(--amber) !important; }

/* ── Audio player ── */
.stAudio audio { filter: invert(0.85) sepia(0.2); border-radius: 3px; }

/* ── Alerts / spinner ── */
[data-testid="stAlert"] {
    background: #13131a !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}
[data-testid="stSpinner"] p {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--dim2) !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #1e1e28 !important; margin: 32px 0 !important; }

/* ── Song list */
.song-row {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--dim2);
    padding: 5px 0;
    border-bottom: 1px solid #1a1a22;
    display: flex;
    align-items: center;
    gap: 8px;
}
.song-row::before {
    content: '▸';
    color: var(--amber);
    font-size: 9px;
}

/* ── Hero ── */
.hero {
    padding: 48px 0 44px;
    border-bottom: 1px solid #1e1e28;
    margin-bottom: 36px;
}
.hero-kicker {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 18px;
}
.hero-title {
    font-family: var(--display);
    font-size: clamp(52px, 8vw, 88px);
    font-weight: 800;
    line-height: 0.92;
    letter-spacing: -0.04em;
    color: #fff;
    margin: 0 0 28px;
}
.hero-title span { color: var(--amber); }
.hero-desc {
    font-family: var(--display);
    font-size: 15px;
    line-height: 1.65;
    color: var(--dim2);
    max-width: 560px;
    margin: 0 0 36px;
}
.hero-meta {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.hero-meta-item {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 0 24px 0 0;
}
.hero-meta-item:first-child { padding-left: 0; }
.hero-meta-k {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--dim);
}
.hero-meta-v {
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 700;
    color: var(--text);
}
.hero-meta-divider {
    width: 1px;
    height: 28px;
    background: #2a2a38;
    margin: 0 24px 0 0;
    flex-shrink: 0;
}
.hero-tracklist {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--dim);
    letter-spacing: 0.04em;
    padding-top: 16px;
    border-top: 1px solid #1e1e28;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# DB loading
@st.cache_resource(show_spinner=False)
def load_db():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
                db = pickle.load(f)
            if hasattr(db, "songs") and len(db.songs) > 0:
                return db, "precomputed"
        except Exception:
            pass  # corrupt / Python version mismatch — rebuild below
    if os.path.isdir(SONGS_DIR) and any(
        f.lower().endswith((".mp3", ".wav", ".m4a")) for f in os.listdir(SONGS_DIR)
    ):
        db = FingerprintDB(win_length=WIN_LENGTH, hop_length=HOP_LENGTH, sr=SR)
        db.build_from_folder(SONGS_DIR)
        try:  # overwrite stale pkl with a fresh one for next run
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

# Matplotlib dark theme
RC = {
    "figure.facecolor":  "#13131a",
    "axes.facecolor":    "#0c0c0f",
    "axes.edgecolor":    "#2a2a38",
    "axes.labelcolor":   "#71717a",
    "axes.titlecolor":   "#a1a1aa",
    "xtick.color":       "#52525b",
    "ytick.color":       "#52525b",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "text.color":        "#d4d4d8",
    "grid.color":        "#1e1e28",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
}

AMBER  = "#ff8c42"
DIM    = "#52525b"
DIM2   = "#71717a"
TEXT   = "#d4d4d8"
BG0    = "#0c0c0f"
BG1    = "#13131a"


def fig_spectrogram(y, sr, win_length, hop_length):
    freqs, times, S_db = spectrogram_db(y, sr, win_length, hop_length)
    peaks = find_constellation_peaks(S_db, freqs, times)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(9, 4.2), constrained_layout=True)
        gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[20, 1], wspace=0.04)
        ax  = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])

        im = ax.pcolormesh(
            times, freqs / 1000, S_db,   # kHz on y
            shading="auto", cmap="inferno", vmin=-80, vmax=0, rasterized=True
        )
        if peaks:
            pt = [p["time_s"]  for p in peaks]
            pf = [p["freq_hz"] / 1000 for p in peaks]
            ax.scatter(pt, pf, s=26, facecolors="none",
                       edgecolors="#00f5d4", linewidths=1.1, alpha=0.9, zorder=3)

        ax.set_ylim(0, min(5000, sr / 2) / 1000)
        ax.set_xlabel("time  (s)")
        ax.set_ylabel("frequency  (kHz)")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.4)

        cb = fig.colorbar(im, cax=cax)
        cb.set_label("dB", color=DIM2, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=DIM2, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=DIM2)

        ax.text(0.01, 0.97,
                f"{len(peaks)} peaks",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color=AMBER,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG1, ec="#2a2a38", alpha=0.9))

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
        best_off = max(histogram, key=histogram.get)
        peak_count = histogram[best_off]

        ax1.bar(offs, counts, width=max(1, (max(offs)-min(offs))/len(offs)*0.9),
                color=AMBER, alpha=0.55, zorder=2)
        ax1.axvline(best_off, color="#fff", lw=1.2, alpha=0.25, zorder=3)
        ax1.set_yscale("log")
        ax1.set_xlabel("time offset  (frames)")
        ax1.set_ylabel("votes  (log)")
        ax1.grid(True, axis="y", alpha=0.3)

        ax1.bar([best_off], [peak_count],
                width=max(1, (max(offs)-min(offs))/len(offs)*0.9),
                color=AMBER, alpha=1.0, zorder=4)

        ax1.text(0.98, 0.97, f"peak @ {best_off}",
                 transform=ax1.transAxes, va="top", ha="right",
                 fontsize=8, color=AMBER,
                 bbox=dict(boxstyle="round,pad=0.3", fc=BG1, ec="#2a2a38", alpha=0.9))

        w    = 60
        zoom = [o for o in offs if abs(o - best_off) <= w]
        zcnt = [histogram[o] for o in zoom]

        bw = max(1, w // max(len(zoom), 1))
        ax2.bar(zoom, zcnt, width=bw, color=DIM2, alpha=0.5, zorder=2)
        ax2.bar([best_off], [peak_count], width=bw, color=AMBER, alpha=1.0, zorder=3)
        ax2.axvline(best_off, color="#fff", lw=1, alpha=0.2)
        ax2.set_xlabel("time offset  (frames)")
        ax2.set_ylabel("votes")
        ax2.grid(True, axis="y", alpha=0.3)

        title_txt = f"zoom ±{w} frames · {peak_count} votes"
        ax2.set_title(title_txt, color=DIM2, fontsize=9, pad=6)

    return fig


# State
if "mode" not in st.session_state:
    st.session_state.mode = "single"

db, db_status = load_db()
song_count = len(db.songs) if db else 0
led_color  = "#4ade80" if db_status != "missing" else "#f87171"
db_label   = f"{song_count} songs indexed" if db_status != "missing" else "no database"

# ── Top bar ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div class="topbar-wordmark">ZAPP<span>·</span>TAIN</div>
  <div class="topbar-rule"></div>
  <div class="topbar-db">
    <span class="db-led" style="background:{led_color};
          box-shadow:0 0 6px {led_color}88;"></span>
    {db_label}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Page shell ────────────────────────────────────────────────────────────────
st.markdown('<div class="page-shell">', unsafe_allow_html=True)

if db_status == "missing":
    st.error("No song database. Add mp3s to `songs/` → run `python build_database.py`.")
    st.stop()

# ── Hero ─────────────────────────────────────────────────────────────────────
song_names = list(db.songs.values())
song_list_inline = "  ·  ".join(song_names)

st.markdown(f"""
<div class="hero">
  <div class="hero-kicker">Audio fingerprinting · Shazam algorithm</div>
  <h1 class="hero-title">Zapp<span>·</span>tain<br>America</h1>
  <p class="hero-desc">
    Upload a short audio clip — even a few seconds over noise — and the
    system identifies it by matching sparse time–frequency landmarks against
    a pre-indexed song database. No waveform comparison, no machine learning:
    just combinatorial hashing and offset voting.
  </p>
  <div class="hero-meta">
    <div class="hero-meta-item">
      <span class="hero-meta-k">indexed songs</span>
      <span class="hero-meta-v">{song_count}</span>
    </div>
    <div class="hero-meta-divider"></div>
    <div class="hero-meta-item">
      <span class="hero-meta-k">window</span>
      <span class="hero-meta-v">{WIN_LENGTH} samples</span>
    </div>
    <div class="hero-meta-divider"></div>
    <div class="hero-meta-item">
      <span class="hero-meta-k">sample rate</span>
      <span class="hero-meta-v">{SR} Hz</span>
    </div>
    <div class="hero-meta-divider"></div>
    <div class="hero-meta-item">
      <span class="hero-meta-k">hash method</span>
      <span class="hero-meta-v">paired peaks</span>
    </div>
  </div>
  <div class="hero-tracklist">{song_list_inline}</div>
</div>
""", unsafe_allow_html=True)

if db_status == "built_live":
    st.info("Database re-indexed live at startup — a fresh fingerprint_db.pkl has been saved.")

# ── Mode tabs ─────────────────────────────────────────────────────────────────
c1, c2, _ = st.columns([1, 1, 8])
with c1:
    if st.button("Single clip", key="t1"):
        st.session_state.mode = "single"
with c2:
    if st.button("Batch", key="t2"):
        st.session_state.mode = "batch"

mode = st.session_state.mode
st.markdown("<hr>", unsafe_allow_html=True)

# SINGLE-CLIP MODE
if mode == "single":

    st.markdown("""
    <div class="sec-head">
      <span class="sec-num">01</span>
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

            best       = result["best"]
            histogram  = result["histogram"]
            ranked     = result["ranked"]

            # ── Result ───────────────────────────────────────────────────────
            st.markdown("""
            <div class="sec-head" style="margin-top:28px">
              <span class="sec-num">02</span>
              <span class="sec-title">Identification result</span>
              <span class="sec-line"></span>
            </div>
            """, unsafe_allow_html=True)

            if best is None:
                st.markdown('<div class="no-match-block">✕ no match — clip does not correspond to any indexed song</div>',
                            unsafe_allow_html=True)
            else:
                top_votes = ranked[0][2]
                runner_up = ranked[1][2] if len(ranked) > 1 else 0
                ratio     = f"{top_votes/max(runner_up,1):.1f}×" if runner_up else "—"

                st.markdown(f"""
                <div class="match-block">
                  <div>
                    <div class="match-tag">identified song</div>
                    <div class="match-name">{best}</div>
                  </div>
                </div>
                <div class="stat-row">
                  <div class="stat-pill">
                    <div class="stat-k">best offset votes</div>
                    <div class="stat-v">{top_votes}</div>
                  </div>
                  <div class="stat-pill">
                    <div class="stat-k">runner-up votes</div>
                    <div class="stat-v">{runner_up}</div>
                  </div>
                  <div class="stat-pill">
                    <div class="stat-k">confidence ratio</div>
                    <div class="stat-v">{ratio}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("all candidates"):
                    st.table(pd.DataFrame(
                        [(n, v) for _, n, v in ranked],
                        columns=["song", "votes"],
                    ))

            # ── Intermediate steps ────────────────────────────────────────────
            st.markdown("""
            <div class="sec-head" style="margin-top:28px">
              <span class="sec-num">03</span>
              <span class="sec-title">Intermediate steps</span>
              <span class="sec-line"></span>
            </div>
            """, unsafe_allow_html=True)

            # Step A: spectrogram + constellation
            st.markdown("""
            <div class="callout">
              <div class="callout-head">A — Spectrogram &amp; constellation map</div>
              <div class="callout-body">
                The audio is divided into short, overlapping windows (<b>win={win}</b> samples,
                hop=<b>{hop}</b> samples at {sr} Hz). Each window is multiplied by a
                <b>Hann taper</b> to suppress spectral leakage, then passed through an FFT.
                Stacking these column-by-column gives the time–frequency power map below.
                <b>Teal circles</b> mark the <em>constellation peaks</em> — local maxima
                that exceed the neighbourhood and a −40 dB floor, thinned to at most
                5 per time-frame. These sparse landmarks are what gets fingerprinted.
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
                <div class="stat-k">constellation peaks</div>
                <div class="stat-v">{len(peaks)}</div>
              </div>
              <div class="stat-pill">
                <div class="stat-k">freq resolution</div>
                <div class="stat-v">{freq_res:.1f} Hz/bin</div>
              </div>
              <div class="stat-pill">
                <div class="stat-k">time resolution</div>
                <div class="stat-v">{time_res:.1f} ms/frame</div>
              </div>
              <div class="stat-pill">
                <div class="stat-k">clip duration</div>
                <div class="stat-v">{len(y)/SR:.2f} s</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # Step B: offset histogram
            st.markdown("""
            <div class="callout">
              <div class="callout-head">B — Combinatorial hashing &amp; offset histogram</div>
              <div class="callout-body">
                Each constellation peak is <b>paired</b> with up to 10 later peaks within a
                target zone of [1, 100] frames. Each pair produces a hash
                <em>(f₁_bin, f₂_bin, Δt)</em>. These hashes are looked up in the database;
                for every match the <b>time offset</b> = db_frame − query_frame is recorded.
                A true match accumulates votes at <b>one consistent offset</b> (the spike),
                while false songs scatter votes randomly across all offsets.
                The song with the highest single-offset vote count wins.
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

# BATCH MODE
else:
    st.markdown("""
    <div class="sec-head">
      <span class="sec-num">01</span>
      <span class="sec-title">Upload query clips</span>
      <span class="sec-line"></span>
    </div>
    <div class="callout" style="margin-bottom:20px">
      <div class="callout-head">Batch mode</div>
      <div class="callout-body">
        Upload any number of clips. Each is fingerprinted and matched independently.
        The output <b>results.csv</b> contains two columns:
        <b>filename</b> (original upload name) and <b>prediction</b>
        (matched song filename without extension, blank if no match).
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploads = st.file_uploader(
        "clips", type=["mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploads and st.button(f"Run on {len(uploads)} clip{'s' if len(uploads)!=1 else ''}"):
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
          <span class="sec-num">02</span>
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

# ── Footer: indexed songs ─────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander(f"indexed songs  ({song_count})"):
    for name in db.songs.values():
        st.markdown(f'<div class="song-row">{name}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # .page-shell