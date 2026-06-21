# Zapp-tain America — Audio Fingerprint Identifier

A small Shazam-style song identifier (Q3 A+B). Spectrogram → constellation
of peaks → combinatorial (paired) hashes → offset-histogram voting to find
the best-aligned song.

## Project layout

```
app.py              Streamlit app (single-clip mode + batch mode)
audio_io.py          mp3/wav loading (via ffmpeg) + manual STFT
fingerprint.py        peak-picking, hashing, database, matching
build_database.py    offline script: indexes songs/ -> data/fingerprint_db.pkl
songs/                put all 50 provided songs here (mp3)
data/fingerprint_db.pkl   precomputed hash database (generated, see below)
requirements.txt     Python deps
packages.txt          system deps for Streamlit Cloud (ffmpeg)
```

## 1. Local setup

```bash
pip install -r requirements.txt
# ffmpeg must be installed and on PATH (already true on most systems);
# on Debian/Ubuntu: sudo apt-get install ffmpeg
```

## 2. Index the song database (run ONCE, before deploying)

Drop all 50 provided songs into `songs/` (keep the original filenames —
the filename without extension is what gets reported as the prediction),
then:

```bash
python3 build_database.py
```

This writes `data/fingerprint_db.pkl`. **This file — not the raw mp3s — is
what makes the app "ship with the database already indexed."** The app
loads it in well under a second on startup instead of re-hashing 50 songs
on every cold start.

Re-run this script any time you change `WIN_LENGTH`/`HOP_LENGTH` in
`build_database.py` (must match the constants at the top of `app.py`).

## 3. Run locally

```bash
streamlit run app.py
```

## 4. Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repo**. At minimum commit:
   `app.py`, `audio_io.py`, `fingerprint.py`, `build_database.py`,
   `requirements.txt`, `packages.txt`, and `data/fingerprint_db.pkl`.
   - You do **not** need to commit `songs/` (the raw mp3s) for the app to
     work — only the precomputed `.pkl`. Including the songs too is fine
     if your repo size allows it (useful for reproducibility / re-running
     `build_database.py` later), but it's optional. If GitHub balks at
     total repo size, just leave `songs/` out (add it to `.gitignore`).
2. Go to https://share.streamlit.io , sign in with GitHub, click
   "New app", pick this repo + branch, set the main file to `app.py`.
3. Streamlit Cloud reads `packages.txt` and installs `ffmpeg` via apt
   automatically before installing `requirements.txt`. No extra config
   needed.
4. Deploy. First boot loads the `.pkl` (fast). Test both modes before
   submitting the link.

## Notes on parameters

`WIN_LENGTH=4096`, `HOP_LENGTH=2048` at `sr=22050` Hz are set in both
`build_database.py` and `app.py` — keep them in sync, since the query and
the database must be analyzed with the same time/frequency resolution for
hashes to line up.
