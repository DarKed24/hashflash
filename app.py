"""
app.py -- "Zapp-tain America": a small Shazam clone.

Two modes (toggle in header):
  1. Single-clip mode
  2. Batch mode
"""

import os
import pickle
import tempfile

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_io import load_audio, spectrogram_db
from fingerprint import FingerprintDB, find_constellation_peaks

WIN_LENGTH = 4096
HOP_LENGTH = 2048
SR        = 22050
DB_PATH   = "data/fingerprint_db.pkl"
SONGS_DIR = "songs"

st.set_page_config(
    page_title="Zapp-tain America",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Force dark on every element Streamlit owns ──────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

/* Nuke every Streamlit surface to the dark palette */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container,
section.main > div,
div[class*="appview"],
div[class*="main"] {
    background-color: #0d0d0d !important;
    color: #e8e8e8 !important;
}

/* Kill sidebar */
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }

/* The Streamlit top chrome bar */
header[data-testid="stHeader"] {
    background: #0d0d0d !important;
    border-bottom: 1px solid #222 !important;
}

* { box-sizing: border-box; }

:root {
    --bg:      #0d0d0d;
    --bg1:     #141414;
    --bg2:     #1a1a1a;
    --border:  #2a2a2a;
    --accent:  #e8ff47;   /* acid yellow-green — the ONE bold thing */
    --text:    #e8e8e8;
    --dim:     #666;
    --danger:  #ff4545;
    --ok:      #4cff91;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
}

body { font-family: var(--sans); }

/* ── Fixed navbar ── */
.zn {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 52px;
    background: #0d0d0d;
    border-bottom: 1px solid #222;
    display: flex;
    align-items: center;
    padding: 0 28px;
    gap: 20px;
    z-index: 10000;
}
.zn-brand {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: var(--accent);
    text-transform: uppercase;
}
.zn-sep { color: #333; margin: 0 4px; }
.zn-mode {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.zn-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--dim);
}
.zn-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}

/* Offset content below fixed nav */
.main .block-container {
    padding-top: 72px !important;
    padding-left: 28px !important;
    padding-right: 28px !important;
    max-width: 1060px !important;
}

/* ── Section label ── */
.sec-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #222;
}

/* ── Mode switcher tabs ── */
.tabs-row {
    display: flex;
    gap: 0;
    border: 1px solid #222;
    border-radius: 4px;
    overflow: hidden;
    width: fit-content;
    margin-bottom: 32px;
}
.tab-btn {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 22px;
    border: none;
    cursor: pointer;
    transition: background 0.1s, color 0.1s;
}
.tab-active   { background: var(--accent); color: #0d0d0d; font-weight: 600; }
.tab-inactive { background: #141414; color: var(--dim); }
.tab-inactive:hover { color: var(--text); }

/* ── Drop zone ── */
[data-testid="stFileUploaderDropzone"] {
    background: #141414 !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 4px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #555 !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 8px 20px !important;
    transition: background 0.12s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #0d0d0d !important;
}

/* ── Result strip ── */
.res-strip {
    border-left: 3px solid var(--accent);
    padding: 16px 20px;
    background: #141414;
    margin-bottom: 24px;
    display: flex;
    align-items: baseline;
    gap: 16px;
}
.res-strip-label {
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--dim);
    white-space: nowrap;
}
.res-strip-name {
    font-family: var(--mono);
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    word-break: break-all;
}

.no-match {
    border-left: 3px solid var(--danger);
    padding: 14px 20px;
    background: #141414;
    margin-bottom: 24px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--danger);
}

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #222; margin-bottom: 24px; }
.stat-cell { background: #141414; padding: 16px 20px; }
.stat-key  { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--dim); margin-bottom: 6px; }
.stat-val  { font-family: var(--mono); font-size: 28px; font-weight: 600; color: var(--accent); }

/* ── Audio player ── */
.stAudio { margin-bottom: 20px; }
.stAudio audio {
    width: 100%;
    filter: invert(1) hue-rotate(180deg);  /* dark-mode native audio */
    border-radius: 3px;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 3px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid #222 !important;
    border-radius: 3px !important;
}
thead th {
    background: #1a1a1a !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--dim) !important;
}
tbody td {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--text) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: var(--accent) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #141414 !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-family: var(--mono) !important; font-size: 12px !important; color: var(--dim) !important; }

/* Divider */
hr { border-color: #222 !important; margin: 28px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── DB loading ───────────────────────────────────────────────────────────
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
    return None, "missing"


def save_upload_to_tmp(uf):
    suffix = os.path.splitext(uf.name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uf.getbuffer())
        return tmp.name


# ── Plots ────────────────────────────────────────────────────────────────
RC = {
    "figure.facecolor": "#141414",
    "axes.facecolor":   "#0d0d0d",
    "axes.edgecolor":   "#2a2a2a",
    "axes.labelcolor":  "#666",
    "xtick.color":      "#555",
    "ytick.color":      "#555",
    "text.color":       "#e8e8e8",
    "grid.color":       "#1f1f1f",
}

def plot_spectrogram(y, sr, win_length, hop_length):
    freqs, times, S_db = spectrogram_db(y, sr, win_length, hop_length)
    peaks = find_constellation_peaks(S_db, freqs, times)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="inferno", vmin=-80, vmax=0)
        if peaks:
            ax.scatter([p["time_s"] for p in peaks], [p["freq_hz"] for p in peaks],
                       s=14, facecolors="none", edgecolors="#e8ff47", linewidths=0.8, alpha=0.9)
        ax.set_ylim(0, min(5000, sr / 2))
        ax.set_title(f"spectrogram + {len(peaks)} peaks", fontsize=9, color="#666",
                     fontfamily="monospace", pad=8)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("freq (Hz)", fontsize=8)
        fig.tight_layout(pad=1.2)
    return fig, peaks


def plot_histogram(histogram, best_name):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        if not histogram:
            for ax in axes:
                ax.text(0.5, 0.5, "no matching hashes", ha="center", va="center",
                        color="#444", fontsize=9, fontfamily="monospace")
                ax.set_facecolor("#0d0d0d")
            fig.tight_layout(pad=1.2)
            return fig
        offs = sorted(histogram.keys())
        counts = [histogram[o] for o in offs]
        best_off = max(histogram, key=histogram.get)

        axes[0].bar(offs, counts, width=3.0, color="#e8ff47", alpha=0.7)
        axes[0].set_yscale("log")
        axes[0].axvline(best_off, color="#fff", alpha=0.15, lw=10, zorder=0)
        axes[0].set_title("offset histogram", fontsize=9, color="#666", fontfamily="monospace", pad=6)
        axes[0].set_xlabel("offset (frames)", fontsize=8)
        axes[0].set_ylabel("votes", fontsize=8)

        w = 40
        zoom = [o for o in offs if abs(o - best_off) <= w]
        axes[1].bar(zoom, [histogram[o] for o in zoom], width=1.0, color="#e8ff47", alpha=0.9)
        axes[1].set_title(f"zoom · {histogram[best_off]} votes at {best_off}",
                          fontsize=9, color="#666", fontfamily="monospace", pad=6)
        axes[1].set_xlabel("offset (frames)", fontsize=8)
        axes[1].set_ylabel("votes", fontsize=8)

        fig.tight_layout(pad=1.2)
    return fig


# ── State ────────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "single"

db, db_status = load_db()

song_count  = len(db.songs) if db else 0
dot_color   = "#4cff91" if db_status != "missing" else "#ff4545"
db_label    = f"{song_count} tracks" if db_status != "missing" else "no db"
mode_label  = "single clip" if st.session_state.mode == "single" else "batch"

# ── Fixed navbar (pure HTML, no Streamlit widget) ────────────────────────
st.markdown(f"""
<div class="zn">
  <span class="zn-brand">Zapp-tain</span>
  <span class="zn-sep">/</span>
  <span class="zn-mode">{mode_label}</span>
  <div class="zn-right">
    <span class="zn-dot" style="background:{dot_color}"></span>{db_label}
  </div>
</div>
""", unsafe_allow_html=True)

# ── DB missing hard stop ─────────────────────────────────────────────────
if db_status == "missing":
    st.error("No database. Add mp3s to `songs/` → `python build_database.py`.")
    st.stop()

if db_status == "built_live":
    st.warning("Indexed live at startup (no fingerprint_db.pkl found).")

# ── Mode switcher (real Streamlit buttons, styled as tabs) ───────────────
c1, c2, _ = st.columns([1, 1, 8])
with c1:
    if st.button("Single clip", key="ms"):
        st.session_state.mode = "single"
with c2:
    if st.button("Batch", key="mb"):
        st.session_state.mode = "batch"

mode = st.session_state.mode

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SINGLE CLIP
# ════════════════════════════════════════════════════════════════════════
if mode == "single":
    st.markdown('<div class="sec-label">Upload query clip</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("clip", type=["mp3", "wav", "m4a", "flac"],
                                label_visibility="collapsed")

    if uploaded:
        tmp = save_upload_to_tmp(uploaded)
        try:
            with st.spinner("decoding…"):
                y, sr = load_audio(tmp, sr=SR)
            st.audio(uploaded)

            with st.spinner("fingerprinting…"):
                result = db.match(y, mode="paired")

            best = result["best"]

            if best is None:
                st.markdown('<div class="no-match">✕ no match found</div>', unsafe_allow_html=True)
            else:
                top_votes = result["ranked"][0][2]
                runner_up = result["ranked"][1][2] if len(result["ranked"]) > 1 else 0
                st.markdown(f"""
                <div class="res-strip">
                  <span class="res-strip-label">identified</span>
                  <span class="res-strip-name">{best}</span>
                </div>
                <div class="stat-grid">
                  <div class="stat-cell">
                    <div class="stat-key">best offset votes</div>
                    <div class="stat-val">{top_votes}</div>
                  </div>
                  <div class="stat-cell">
                    <div class="stat-key">runner-up votes</div>
                    <div class="stat-val">{runner_up}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("all candidates"):
                    st.table(pd.DataFrame(
                        [(n, v) for _, n, v in result["ranked"]],
                        columns=["song", "votes"],
                    ))

            st.markdown('<div class="sec-label" style="margin-top:8px">Analysis</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                fig1, _ = plot_spectrogram(y, sr, WIN_LENGTH, HOP_LENGTH)
                st.pyplot(fig1); plt.close(fig1)
            with col2:
                fig2 = plot_histogram(result["histogram"], best or "—")
                st.pyplot(fig2); plt.close(fig2)

        finally:
            os.remove(tmp)

# ════════════════════════════════════════════════════════════════════════
# BATCH
# ════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="sec-label">Upload clips</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("clips", type=["mp3", "wav", "m4a", "flac"],
                               accept_multiple_files=True, label_visibility="collapsed")

    if uploads and st.button(f"Run — {len(uploads)} clips"):
        rows = []
        bar = st.progress(0.0, text="")
        for i, uf in enumerate(uploads):
            tmp = save_upload_to_tmp(uf)
            try:
                y, sr_f = load_audio(tmp, sr=SR)
                res = db.match(y, mode="paired")
                rows.append({"filename": uf.name, "prediction": res["best"] or ""})
            except Exception as e:
                rows.append({"filename": uf.name, "prediction": ""})
                st.warning(f"{uf.name}: {e}")
            finally:
                os.remove(tmp)
            bar.progress((i + 1) / len(uploads), text=f"{uf.name}")

        df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.markdown('<div class="sec-label" style="margin-top:12px">Results</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results.csv", df.to_csv(index=False).encode(),
                           "results.csv", "text/csv")

# ── Footer: indexed songs ────────────────────────────────────────────────
st.markdown("---")
with st.expander(f"indexed songs  ({song_count})"):
    for name in db.songs.values():
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            f'color:#555;padding:3px 0;border-bottom:1px solid #1a1a1a">{name}</div>',
            unsafe_allow_html=True
        )