"""
Zapp-tain America
Modern Audio Fingerprinting Dashboard
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

from audio_io import load_audio, spectrogram_db
from fingerprint import FingerprintDB, find_constellation_peaks

# ============================================================
# CONFIG
# ============================================================

WIN_LENGTH = 4096
HOP_LENGTH = 2048
SR = 22050

DB_PATH = "data/fingerprint_db.pkl"
SONGS_DIR = "songs"

st.set_page_config(
    page_title="Zapp-tain America",
    page_icon="🎵",
    layout="wide",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #0f1117;
    color: white;
}

section[data-testid="stSidebar"] {
    background: #161a22;
}

h1,h2,h3,h4 {
    color: white !important;
}

.hero {
    text-align:center;
    padding:2rem 0 1rem 0;
}

.hero-title {
    font-size:3.2rem;
    font-weight:800;
}

.hero-subtitle {
    color:#B6BECF;
    font-size:1.1rem;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px;
    padding:1rem;
    backdrop-filter: blur(12px);
    margin-bottom:1rem;
}

.match-card {
    background: linear-gradient(
        135deg,
        rgba(91,141,239,0.15),
        rgba(138,99,255,0.15)
    );
    border:1px solid rgba(255,255,255,0.1);
    border-radius:20px;
    padding:1.5rem;
    text-align:center;
}

.metric-big {
    font-size:2rem;
    font-weight:700;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.08);
    padding:10px;
}

[data-testid="stFileUploader"] {
    border:2px dashed #5B8DEF;
    border-radius:16px;
    padding:10px;
    background:rgba(91,141,239,0.05);
}

.stButton button,
.stDownloadButton button {
    background: linear-gradient(135deg,#5B8DEF,#8A63FF);
    color:white;
    border:none;
    border-radius:12px;
    font-weight:600;
}

footer {
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            db = pickle.load(f)
        return db, "precomputed"

    elif os.path.isdir(SONGS_DIR) and any(
        f.lower().endswith((".mp3", ".wav", ".m4a"))
        for f in os.listdir(SONGS_DIR)
    ):
        db = FingerprintDB(
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            sr=SR,
        )

        db.build_from_folder(SONGS_DIR)
        return db, "built_live"

    return None, "missing"


def save_upload_to_tmp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

# ============================================================
# PLOTS
# ============================================================

def plot_spectrogram_constellation(
    y,
    sr,
    win_length,
    hop_length
):
    freqs, times, S_db = spectrogram_db(
        y,
        sr,
        win_length,
        hop_length
    )

    peaks = find_constellation_peaks(
        S_db,
        freqs,
        times
    )

    fig, ax = plt.subplots(figsize=(9,4))

    ax.pcolormesh(
        times,
        freqs,
        S_db,
        shading="auto",
        cmap="magma",
        vmin=-80,
        vmax=0
    )

    if peaks:
        pf = [p["freq_hz"] for p in peaks]
        pt = [p["time_s"] for p in peaks]

        ax.scatter(
            pt,
            pf,
            s=18,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1
        )

    ax.set_ylim(0, min(5000, sr/2))
    ax.set_title(
        f"Spectrogram + Constellation ({len(peaks)} peaks)"
    )

    fig.tight_layout()
    return fig


def plot_offset_histogram(histogram, best_name):
    fig, ax = plt.subplots(figsize=(9,4))

    if not histogram:
        ax.text(
            0.5,
            0.5,
            "No matching hashes",
            ha="center",
            va="center"
        )
        return fig

    offs = sorted(histogram.keys())
    counts = [histogram[o] for o in offs]

    ax.bar(offs, counts)
    ax.set_yscale("log")
    ax.set_title(f"Offset Histogram — {best_name}")

    fig.tight_layout()
    return fig

# ============================================================
# LOAD DATABASE
# ============================================================

db, db_status = load_db()

if db_status == "missing":
    st.error(
        "No fingerprint database found.\n\n"
        "Run build_database.py first."
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
# Zapp-tain
### Audio Recognition Engine
""")

mode = st.sidebar.radio(
    "Choose Mode",
    [
        "Single clip",
        "Batch mode"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a clip and identify songs using "
    "constellation maps and audio fingerprinting."
)

st.sidebar.markdown(
    f"### Songs Indexed\n**{len(db.songs)}**"
)

with st.sidebar.expander("View Song Database"):
    for song in db.songs.values():
        st.write("•", song)

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
<div class="hero-title">Zapp-tain America</div>
<div class="hero-subtitle">
Shazam-style music recognition powered by audio fingerprinting
</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DASHBOARD CARDS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="glass-card">
    <h4>Songs Indexed</h4>
    <div class="metric-big">{len(db.songs)}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass-card">
    <h4>Engine</h4>
    <div class="metric-big">Fingerprint DB</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="glass-card">
    <h4>Recognition</h4>
    <div class="metric-big">Real Time</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SINGLE CLIP MODE
# ============================================================

if mode == "Single clip":

    uploaded = st.file_uploader(
        "Upload Query Clip",
        type=["mp3", "wav", "m4a", "flac"]
    )

    if uploaded:

        tmp_path = save_upload_to_tmp(uploaded)

        try:

            with st.spinner("Decoding audio..."):
                y, sr = load_audio(
                    tmp_path,
                    sr=SR
                )

            st.audio(uploaded)

            with st.spinner("Matching fingerprints..."):
                result = db.match(
                    y,
                    mode="paired"
                )

            best = result["best"]

            if best is None:

                st.error(
                    "No matching song found."
                )

            else:

                top_votes = result["ranked"][0][2]

                runner_up = (
                    result["ranked"][1][2]
                    if len(result["ranked"]) > 1
                    else 0
                )

                confidence = (
                    top_votes /
                    max(top_votes + runner_up, 1)
                ) * 100

                st.markdown(f"""
                <div class="match-card">
                <h3>Match Found</h3>
                <h1>{best}</h1>
                </div>
                """, unsafe_allow_html=True)

                st.progress(
                    min(confidence/100, 1.0)
                )

                st.caption(
                    f"Confidence: {confidence:.1f}%"
                )

                m1, m2 = st.columns(2)

                with m1:
                    st.metric(
                        "Best Votes",
                        top_votes
                    )

                with m2:
                    st.metric(
                        "Runner-up Votes",
                        runner_up
                    )

                with st.expander(
                    "Top Candidate Matches"
                ):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                (n, v)
                                for _, n, v
                                in result["ranked"]
                            ],
                            columns=[
                                "Song",
                                "Votes"
                            ]
                        ),
                        use_container_width=True
                    )

            tab1, tab2 = st.tabs(
                [
                    "Spectrogram",
                    "Match Analysis"
                ]
            )

            with tab1:
                fig1 = plot_spectrogram_constellation(
                    y,
                    sr,
                    WIN_LENGTH,
                    HOP_LENGTH
                )

                st.pyplot(fig1)
                plt.close(fig1)

            with tab2:
                fig2 = plot_offset_histogram(
                    result["histogram"],
                    best or "None"
                )

                st.pyplot(fig2)
                plt.close(fig2)

        finally:
            os.remove(tmp_path)

# ============================================================
# BATCH MODE
# ============================================================

else:

    st.markdown("""
    <div class="glass-card">
    <h3>Batch Identification</h3>
    Upload multiple clips and receive a downloadable results.csv.
    </div>
    """, unsafe_allow_html=True)

    uploads = st.file_uploader(
        "Upload Query Clips",
        type=["mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True
    )

    if uploads and st.button(
        f"Identify {len(uploads)} Files"
    ):

        rows = []

        progress = st.progress(
            0,
            text="Starting..."
        )

        for i, uf in enumerate(uploads):

            tmp_path = save_upload_to_tmp(uf)

            try:

                y, sr = load_audio(
                    tmp_path,
                    sr=SR
                )

                result = db.match(
                    y,
                    mode="paired"
                )

                pred = (
                    result["best"]
                    if result["best"]
                    else ""
                )

                rows.append(
                    {
                        "filename": uf.name,
                        "prediction": pred
                    }
                )

            except Exception:

                rows.append(
                    {
                        "filename": uf.name,
                        "prediction": ""
                    }
                )

            finally:

                os.remove(tmp_path)

            progress.progress(
                (i + 1) / len(uploads),
                text=f"Processed {uf.name}"
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "filename",
                "prediction"
            ]
        )

        st.markdown("""
        <div class="glass-card">
        <h3>Results</h3>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            df,
            use_container_width=True,
            height=450
        )

        csv_bytes = (
            df.to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇ Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv"
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <center>
    Built with Streamlit • Audio Fingerprinting • IIT Kanpur
    </center>
    """,
    unsafe_allow_html=True
)