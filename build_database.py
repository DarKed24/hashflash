"""
build_database.py
------------------
Run this ONCE (locally, before deploying) to index every song in songs/
into a fingerprint database, and save it to data/fingerprint_db.pkl.

The deployed Streamlit app only needs to ship data/fingerprint_db.pkl --
NOT the raw mp3s -- since matching only needs the precomputed hash index,
not the original audio. This keeps the deployed repo small and makes the
app start up instantly (no re-hashing 50 songs on every cold start).

Usage:
    python3 build_database.py                  # indexes everything in songs/
    python3 build_database.py --songs-dir X     # custom folder
"""

import argparse
import os
import pickle
import time

from fingerprint import FingerprintDB

# These must match the parameters used in app.py at query time.
WIN_LENGTH = 4096
HOP_LENGTH = 2048
SR = 22050


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs-dir", default="songs")
    ap.add_argument("--out", default="data/fingerprint_db.pkl")
    args = ap.parse_args()

    db = FingerprintDB(win_length=WIN_LENGTH, hop_length=HOP_LENGTH, sr=SR)

    files = sorted(
        f for f in os.listdir(args.songs_dir)
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac"))
    )
    if not files:
        raise SystemExit(f"No audio files found in {args.songs_dir}/")

    print(f"Indexing {len(files)} songs from {args.songs_dir}/ ...")
    t0 = time.time()
    for i, f in enumerate(files, 1):
        path = os.path.join(args.songs_dir, f)
        t1 = time.time()
        db.add_song(path)
        print(f"  [{i}/{len(files)}] {f}  ({time.time()-t1:.1f}s)")

    print(f"Done in {time.time()-t0:.1f}s total.")
    print(f"Songs indexed: {len(db.songs)}")
    print(f"Paired hashes: {sum(len(v) for v in db.paired_index.values())}")
    print(f"Single hashes: {sum(len(v) for v in db.single_index.values())}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(db, fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
