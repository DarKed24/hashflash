"""
app.py -- "Zapp-tain America": a small Shazam clone.

Two modes (toggle in header):
  1. Single-clip mode -- upload one query clip, see the spectrogram, the
     constellation of peaks, and the offset histogram that decides the
     match, plus the predicted song.
  2. Batch mode -- upload many query clips at once, get back a
     results.csv with exactly two columns: filename, prediction.
"""

import os
import io
import pickle
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_io import load_audio, spectrogram_db
from fingerprint import FingerprintDB, find_constellation_peaks

# ----------------------------------------------------------------------
# Config -- MUST match the parameters used in build_database.py
# ----------------------------------------------------------------------
WIN_LENGTH = 4096
HOP_LENGTH = 2048
SR = 22050

DB_PATH = "data/fingerprint_db.pkl"
SONGS_DIR = "songs"

st.set_page_config(
    page_title="Zapp-tain America",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Global CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Colour tokens ── */
:root {
    --navy:    #0A0E1A;
    --surface: #111827;
    --card:    #1A2235;
    --border:  #1E2D45;
    --violet:  #7C3AED;
    --violet2: #6D28D9;
    --cyan:    #06B6D4;
    --text:    #F8FAFC;
    --muted:   #94A3B8;
    --success: #10B981;
    --warn:    #F59E0B;
    --danger:  #EF4444;
}

body { background: var(--navy) !important; color: var(--text) !important; }

/* ── Top nav bar ── */
.zapnav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(10, 14, 26, 0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 2.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    height: 64px;
}

.zapnav-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
    white-space: nowrap;
}

.zapnav-brand .dot { color: var(--violet); font-size: 1.6rem; line-height: 1; }

/* Waveform bars — signature element */
.waveform {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 28px;
}
.waveform span {
    display: block;
    width: 3px;
    border-radius: 2px;
    background: var(--cyan);
    animation: wave 1.2s ease-in-out infinite;
    opacity: 0.85;
}
.waveform span:nth-child(1)  { height: 8px;  animation-delay: 0.0s; }
.waveform span:nth-child(2)  { height: 18px; animation-delay: 0.1s; }
.waveform span:nth-child(3)  { height: 26px; animation-delay: 0.2s; }
.waveform span:nth-child(4)  { height: 14px; animation-delay: 0.3s; }
.waveform span:nth-child(5)  { height: 22px; animation-delay: 0.15s; }
.waveform span:nth-child(6)  { height: 10px; animation-delay: 0.25s; }
.waveform span:nth-child(7)  { height: 18px; animation-delay: 0.05s; }

@keyframes wave {
    0%, 100% { transform: scaleY(1);   opacity: 0.85; }
    50%       { transform: scaleY(.4); opacity: 0.4;  }
}

.zapnav-spacer { flex: 1; }

/* ── Mode toggle pills ── */
.mode-toggle {
    display: flex;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.mode-btn {
    padding: 6px 20px;
    border-radius: 7px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: all 0.18s ease;
}
.mode-btn-active {
    background: var(--violet);
    color: #fff;
}
.mode-btn-inactive {
    background: transparent;
    color: var(--muted);
}
.mode-btn-inactive:hover { color: var(--text); }

/* DB status pill */
.db-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: var(--muted);
    white-space: nowrap;
}
.db-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--success); }

/* ── Page content wrapper ── */
.zappage {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2.5rem 2rem 4rem;
}

/* ── Hero ── */
.hero {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.75rem;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.75rem;
    color: var(--text);
}
.hero-title em { font-style: normal; color: var(--violet); }
.hero-sub {
    font-size: 1.05rem;
    color: var(--muted);
    max-width: 540px;
    line-height: 1.6;
}

/* ── Cards ── */
.zcard {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}
.zcard-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── Result banner ── */
.result-banner {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(6,182,212,0.08));
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}
.result-icon { font-size: 2rem; }
.result-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--violet); margin-bottom: 0.2rem; }
.result-name  { font-size: 1.35rem; font-weight: 700; color: var(--text); }

.no-match-banner {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    color: var(--danger);
    font-weight: 500;
    margin-bottom: 1.5rem;
}

/* ── Metric blocks ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-box {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
}
.metric-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.metric-val   { font-size: 2rem; font-weight: 700; color: var(--cyan); font-family: 'JetBrains Mono', monospace; }

/* ── Song list in nav ── */
.song-list-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    padding: 4px 0;
    border-bottom: 1px solid var(--border);
}

/* ── Streamlit overrides ── */
.stFileUploader > div { background: var(--card) !important; border: 1.5px dashed var(--border) !important; border-radius: 12px !important; }
.stFileUploader label { color: var(--muted) !important; font-size: 0.9rem !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span { color: var(--text) !important; }

.stButton > button {
    background: var(--violet) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: background 0.18s !important;
}
.stButton > button:hover { background: var(--violet2) !important; }

[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
div[data-testid="metric-container"] { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }

.stProgress > div > div { background: var(--violet) !important; }
.stAudio { border-radius: 12px; overflow: hidden; }

/* Tables */
.stDataFrame table { background: var(--card) !important; }
thead tr th { background: var(--surface) !important; color: var(--muted) !important; font-size: 0.78rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
tbody tr td { color: var(--text) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; }

.stAlert { border-radius: 10px !important; }
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; }

[data-testid="stPlotlyChart"], [data-testid="stPyplotRootElement"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Database loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            db = pickle.load(f)
        return db, "precomputed"
    elif os.path.isdir(SONGS_DIR) and any(
        f.lower().endswith((".mp3", ".wav", ".m4a")) for f in os.listdir(SONGS_DIR)
    ):
        db = FingerprintDB(win_length=WIN_LENGTH, hop_length=HOP_LENGTH, sr=SR)
        db.build_from_folder(SONGS_DIR)
        return db, "built_live"
    else:
        return None, "missing"


def save_upload_to_tmp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


# ----------------------------------------------------------------------
# Plotting helpers — dark-themed figures
# ----------------------------------------------------------------------
DARK_FIG_STYLE = {
    "figure.facecolor": "#1A2235",
    "axes.facecolor":   "#111827",
    "axes.edgecolor":   "#1E2D45",
    "axes.labelcolor":  "#94A3B8",
    "xtick.color":      "#94A3B8",
    "ytick.color":      "#94A3B8",
    "text.color":       "#F8FAFC",
    "grid.color":       "#1E2D45",
}

def plot_spectrogram_constellation(y, sr, win_length, hop_length):
    freqs, times, S_db = spectrogram_db(y, sr, win_length, hop_length)
    peaks = find_constellation_peaks(S_db, freqs, times)

    with plt.rc_context(DARK_FIG_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        if peaks:
            pf = [p["freq_hz"] for p in peaks]
            pt = [p["time_s"] for p in peaks]
            ax.scatter(pt, pf, s=18, facecolors="none", edgecolors="#06B6D4", linewidths=1.0)
        ax.set_ylim(0, min(5000, sr / 2))
        ax.set_title(f"Spectrogram + constellation  ·  {len(peaks)} peaks", fontsize=11, pad=10)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.set_ylabel("freq (Hz)", fontsize=9)
        fig.tight_layout()
    return fig, peaks


def plot_offset_histogram(histogram, best_name):
    with plt.rc_context(DARK_FIG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        if not histogram:
            for ax in axes:
                ax.text(0.5, 0.5, "no matching hashes", ha="center", va="center",
                        color="#94A3B8", fontsize=10)
            fig.tight_layout()
            return fig

        offs = sorted(histogram.keys())
        counts = [histogram[o] for o in offs]
        best_off = max(histogram, key=histogram.get)

        axes[0].bar(offs, counts, width=3.0, color="#7C3AED", alpha=0.85)
        axes[0].set_yscale("log")
        axes[0].axvline(best_off, color="#06B6D4", alpha=0.35, lw=8, zorder=0)
        axes[0].set_title(f"Offset histogram · {best_name}", fontsize=10, pad=8)
        axes[0].set_xlabel("offset (frames)", fontsize=9)
        axes[0].set_ylabel("votes (log)", fontsize=9)

        window = 40
        zoom = [o for o in offs if abs(o - best_off) <= window]
        axes[1].bar(zoom, [histogram[o] for o in zoom], width=1.0, color="#06B6D4", alpha=0.85)
        axes[1].set_title(f"Zoom · offset {best_off}  ({histogram[best_off]} votes)", fontsize=10, pad=8)
        axes[1].set_xlabel("offset (frames)", fontsize=9)
        axes[1].set_ylabel("votes", fontsize=9)

        fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Load DB
# ----------------------------------------------------------------------
db, db_status = load_db()

# ----------------------------------------------------------------------
# Mode state
# ----------------------------------------------------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "single"

# ----------------------------------------------------------------------
# Top navigation bar
# ----------------------------------------------------------------------
song_count = len(db.songs) if db else 0
db_dot_color = "#10B981" if db_status != "missing" else "#EF4444"
db_label = f"{song_count} songs indexed" if db_status != "missing" else "No database"

nav_html = f"""
<div class="zapnav">
  <div class="zapnav-brand">
    <span class="dot">◉</span> Zapp-tain America
  </div>
  <div class="waveform">
    <span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span>
  </div>
  <div class="zapnav-spacer"></div>
  <div class="db-pill">
    <div class="db-dot" style="background:{db_dot_color}"></div>
    {db_label}
  </div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# Mode toggle via real Streamlit buttons inside columns
with st.container():
    st.markdown('<div class="zappage">', unsafe_allow_html=True)

    # Mode toggle — two columns of buttons styled as pills
    col_single, col_batch, col_rest = st.columns([1, 1, 6])
    with col_single:
        if st.button("🎵  Single clip", key="btn_single", use_container_width=True):
            st.session_state.mode = "single"
    with col_batch:
        if st.button("📦  Batch mode", key="btn_batch", use_container_width=True):
            st.session_state.mode = "batch"

    mode = st.session_state.mode

    # ── Missing DB error ──────────────────────────────────────────────
    if db_status == "missing":
        st.error(
            "**No song database found.** Add `.mp3` files to `songs/` and run "
            "`python build_database.py`, or place a precomputed "
            "`data/fingerprint_db.pkl` in the repo."
        )
        st.stop()

    if db_status == "built_live":
        st.warning("Songs indexed live at startup — no precomputed `fingerprint_db.pkl` found.")

    # ── Hero ─────────────────────────────────────────────────────────
    if mode == "single":
        hero_eyebrow = "Single-clip identification"
        hero_title   = "Drop a clip. <em>Find the song.</em>"
        hero_sub     = "Upload a short audio fragment — as little as a few seconds — and Zapp-tain will fingerprint it against the database and tell you exactly what it is."
    else:
        hero_eyebrow = "Batch identification"
        hero_title   = "Many clips, <em>one run.</em>"
        hero_sub     = "Upload multiple query clips at once. Download a clean <code style='color:#06B6D4;font-size:0.95em'>results.csv</code> with every prediction."

    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">{hero_eyebrow}</div>
      <div class="hero-title">{hero_title}</div>
      <div class="hero-sub">{hero_sub}</div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # SINGLE CLIP MODE
    # ─────────────────────────────────────────────────────────────────
    if mode == "single":
        uploaded = st.file_uploader(
            "Query clip",
            type=["mp3", "wav", "m4a", "flac"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            tmp_path = save_upload_to_tmp(uploaded)
            try:
                with st.spinner("Decoding audio…"):
                    y, sr = load_audio(tmp_path, sr=SR)
                st.audio(uploaded)

                with st.spinner("Fingerprinting and matching…"):
                    result = db.match(y, mode="paired")

                best = result["best"]

                if best is None:
                    st.markdown('<div class="no-match-banner">⚠ No match found — this clip doesn\'t appear in the database.</div>', unsafe_allow_html=True)
                else:
                    top_votes   = result["ranked"][0][2]
                    runner_up   = result["ranked"][1][2] if len(result["ranked"]) > 1 else 0

                    st.markdown(f"""
                    <div class="result-banner">
                      <div class="result-icon">🎵</div>
                      <div>
                        <div class="result-label">Identified</div>
                        <div class="result-name">{best}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.markdown(f'<div class="metric-box"><div class="metric-label">Best offset votes</div><div class="metric-val">{top_votes}</div></div>', unsafe_allow_html=True)
                    with mc2:
                        st.markdown(f'<div class="metric-box"><div class="metric-label">Runner-up votes</div><div class="metric-val">{runner_up}</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("All candidates"):
                        st.table(pd.DataFrame(
                            [(n, v) for _, n, v in result["ranked"]],
                            columns=["song", "votes"],
                        ))

                st.markdown('<div class="zcard"><div class="zcard-title">Intermediate analysis</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    fig1, peaks = plot_spectrogram_constellation(y, sr, WIN_LENGTH, HOP_LENGTH)
                    st.pyplot(fig1)
                    plt.close(fig1)
                with col2:
                    fig2 = plot_offset_histogram(result["histogram"], best or "none")
                    st.pyplot(fig2)
                    plt.close(fig2)
                st.markdown('</div>', unsafe_allow_html=True)

            finally:
                os.remove(tmp_path)

    # ─────────────────────────────────────────────────────────────────
    # BATCH MODE
    # ─────────────────────────────────────────────────────────────────
    else:
        uploads = st.file_uploader(
            "Query clips",
            type=["mp3", "wav", "m4a", "flac"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploads and st.button(f"Identify all {len(uploads)} clips ›"):
            rows = []
            progress = st.progress(0.0, text="Starting…")
            for i, uf in enumerate(uploads):
                tmp_path = save_upload_to_tmp(uf)
                try:
                    y, sr = load_audio(tmp_path, sr=SR)
                    result = db.match(y, mode="paired")
                    pred = result["best"] if result["best"] is not None else ""
                    rows.append({"filename": uf.name, "prediction": pred})
                except Exception as e:
                    rows.append({"filename": uf.name, "prediction": ""})
                    st.warning(f"Failed on {uf.name}: {e}")
                finally:
                    os.remove(tmp_path)
                progress.progress((i + 1) / len(uploads), text=f"{uf.name} → done")

            df = pd.DataFrame(rows, columns=["filename", "prediction"])

            st.markdown('<div class="zcard"><div class="zcard-title">Results</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
            )

    # ── Indexed songs footer card ─────────────────────────────────────
    with st.expander(f"📚 {song_count} songs in database"):
        for name in db.songs.values():
            st.markdown(f'<div class="song-list-item">▸ {name}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # .zappage