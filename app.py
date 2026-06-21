"""
app.py -- "Zapp-tain America": a small Shazam clone.

Two modes (select in the sidebar):
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

st.set_page_config(page_title="Zapp-tain America", page_icon="🎵", layout="wide")


# ----------------------------------------------------------------------
# Database loading (cached so it only happens once per server process)
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
# Plotting helpers
# ----------------------------------------------------------------------
def plot_spectrogram_constellation(y, sr, win_length, hop_length):
    freqs, times, S_db = spectrogram_db(y, sr, win_length, hop_length)
    peaks = find_constellation_peaks(S_db, freqs, times)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    if peaks:
        pf = [p["freq_hz"] for p in peaks]
        pt = [p["time_s"] for p in peaks]
        ax.scatter(pt, pf, s=18, facecolors="none", edgecolors="cyan", linewidths=1.0)
    ax.set_ylim(0, min(5000, sr / 2))
    ax.set_title(f"Spectrogram + constellation ({len(peaks)} peaks)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (Hz)")
    fig.tight_layout()
    return fig, peaks


def plot_offset_histogram(histogram, best_name):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    if not histogram:
        for ax in axes:
            ax.text(0.5, 0.5, "no matching hashes", ha="center", va="center")
        return fig

    offs = sorted(histogram.keys())
    counts = [histogram[o] for o in offs]
    best_off = max(histogram, key=histogram.get)

    axes[0].bar(offs, counts, width=3.0, color="steelblue")
    axes[0].set_yscale("log")
    axes[0].axvline(best_off, color="red", alpha=0.25, lw=8, zorder=0)
    axes[0].set_title(f"Offset histogram (log scale)\nbest match: {best_name}")
    axes[0].set_xlabel("offset (frames)")
    axes[0].set_ylabel("votes (log)")

    window = 40
    zoom = [o for o in offs if abs(o - best_off) <= window]
    axes[1].bar(zoom, [histogram[o] for o in zoom], width=1.0, color="crimson")
    axes[1].set_title(f"Zoomed near offset {best_off}\n({histogram[best_off]} votes)")
    axes[1].set_xlabel("offset (frames)")
    axes[1].set_ylabel("votes")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("🎵 Zapp-tain America")
mode = st.sidebar.radio("Mode", ["Single clip", "Batch mode"])

db, db_status = load_db()

if db_status == "missing":
    st.sidebar.error(
        "No song database found. Add mp3s to `songs/` and run "
        "`python build_database.py`, or place a precomputed "
        "`data/fingerprint_db.pkl` in the repo."
    )
    st.stop()
elif db_status == "built_live":
    st.sidebar.warning("Indexed songs live at startup (no precomputed db.pkl found).")

st.sidebar.markdown(f"**Songs indexed:** {len(db.songs)}")
with st.sidebar.expander("Show indexed songs"):
    for name in db.songs.values():
        st.write("- " + name)


# ----------------------------------------------------------------------
# Single-clip mode
# ----------------------------------------------------------------------
if mode == "Single clip":
    st.title("Single-clip identification")
    st.write("Upload a short query clip and I'll identify which song it's from.")

    uploaded = st.file_uploader("Query clip", type=["mp3", "wav", "m4a", "flac"])

    if uploaded is not None:
        tmp_path = save_upload_to_tmp(uploaded)
        try:
            with st.spinner("Decoding audio..."):
                y, sr = load_audio(tmp_path, sr=SR)
            st.audio(uploaded)

            with st.spinner("Fingerprinting and matching..."):
                result = db.match(y, mode="paired")

            best = result["best"]
            if best is None:
                st.error("No match found -- this clip doesn't match any indexed song.")
            else:
                top_votes = result["ranked"][0][2]
                runner_up = result["ranked"][1][2] if len(result["ranked"]) > 1 else 0
                st.success(f"**Predicted song: {best}**")
                c1, c2 = st.columns(2)
                c1.metric("Votes (best offset)", top_votes)
                c2.metric("Runner-up votes", runner_up)

                with st.expander("Top candidates"):
                    st.table(pd.DataFrame(
                        [(n, v) for _, n, v in result["ranked"]],
                        columns=["song", "votes"],
                    ))

            st.subheader("Intermediate steps")
            col1, col2 = st.columns(2)
            with col1:
                fig1, peaks = plot_spectrogram_constellation(y, sr, WIN_LENGTH, HOP_LENGTH)
                st.pyplot(fig1)
                plt.close(fig1)
            with col2:
                fig2 = plot_offset_histogram(result["histogram"], best or "none")
                st.pyplot(fig2)
                plt.close(fig2)

        finally:
            os.remove(tmp_path)


# ----------------------------------------------------------------------
# Batch mode
# ----------------------------------------------------------------------
else:
    st.title("Batch identification")
    st.write(
        "Upload multiple query clips. I'll identify each one and produce a "
        "`results.csv` with columns `filename, prediction` (prediction = "
        "matched song's filename without extension)."
    )

    uploads = st.file_uploader(
        "Query clips", type=["mp3", "wav", "m4a", "flac"], accept_multiple_files=True
    )

    if uploads and st.button(f"Identify all {len(uploads)} clips"):
        rows = []
        progress = st.progress(0.0, text="Starting...")
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
            progress.progress((i + 1) / len(uploads), text=f"{uf.name} -> done")

        df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
        )
