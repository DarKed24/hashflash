"""
fingerprint.py
--------------
The actual "Shazam" logic, built on top of audio_io.spectrogram_db:

  1. find_constellation_peaks  -- keep only strong local maxima of the
                                   spectrogram (the "constellation").
  2. hashes_paired              -- combinatorial hashing: pair each anchor
                                    peak with nearby peaks in a target zone,
                                    hash = (f1, f2, delta_t).
  3. hashes_single               -- baseline for comparison: hash = (f1,)
                                    alone, no pairing.
  4. build_database              -- fingerprint a folder of reference songs.
  5. match                       -- fingerprint a query and vote on offsets
                                    to find the best-aligned song.
"""

import os
from collections import defaultdict
import numpy as np
from scipy.ndimage import maximum_filter

from audio_io import load_audio, spectrogram_db


# ----------------------------------------------------------------------
# 1. Constellation map
# ----------------------------------------------------------------------

def find_constellation_peaks(S_db, freqs, times,
                              amp_min_db=-40.0,
                              freq_nbhd=10, time_nbhd=10,
                              max_peaks_per_frame=5):
    """
    Pick local maxima of the spectrogram that stand out from their
    surroundings (the "circled points" in the assignment figure).

    A point is a peak if it equals the max of its (freq_nbhd x time_nbhd)
    neighborhood AND is louder than amp_min_db. To stop peaks from
    clumping only in the loudest seconds, we additionally cap the number
    of peaks kept per time-frame to the `max_peaks_per_frame` strongest.

    Returns
    -------
    peaks : list of dicts {frame, bin, freq_hz, time_s, db}
    """
    local_max = maximum_filter(S_db, size=(freq_nbhd, time_nbhd), mode="constant", cval=-np.inf)
    is_peak = (S_db == local_max) & (S_db > amp_min_db)

    freq_idx, time_idx = np.where(is_peak)
    n_bins = S_db.shape[0]
    bin_width = freqs[1] - freqs[0]

    # group by time frame, keep the loudest `max_peaks_per_frame` per frame
    by_frame = defaultdict(list)
    for fi, ti in zip(freq_idx, time_idx):
        by_frame[ti].append((S_db[fi, ti], fi))

    peaks = []
    for ti, candidates in by_frame.items():
        candidates.sort(reverse=True)  # loudest first
        for db, fi in candidates[:max_peaks_per_frame]:
            # Sub-bin frequency refinement: parabolic (quadratic) interpolation
            # using the peak bin and its two neighbours. A raw FFT bin is only
            # accurate to +-bin_width/2; for a steady tone that doesn't sit
            # exactly on a bin center, small input perturbations (noise, a
            # different window phase) can flip the *integer* argmax between
            # two adjacent bins even though the *true* underlying frequency
            # hasn't moved. Interpolating to a continuous estimate first and
            # THEN quantizing to a coarser, fixed-size hash bucket makes the
            # hash far more reproducible. This is standard practice in real
            # fingerprinting / pitch-detection systems.
            if 0 < fi < n_bins - 1:
                left, center, right = S_db[fi - 1, ti], S_db[fi, ti], S_db[fi + 1, ti]
                denom = (left - 2 * center + right)
                delta = 0.5 * (left - right) / denom if abs(denom) > 1e-9 else 0.0
                delta = float(np.clip(delta, -1.0, 1.0))
            else:
                delta = 0.0
            refined_hz = freqs[fi] + delta * bin_width

            peaks.append({
                "frame": int(ti),
                "bin": int(fi),
                "hbin": int(round(refined_hz / bin_width)),  # used for hashing (more stable than raw bin)
                "freq_hz": float(refined_hz),
                "time_s": float(times[ti]),
                "db": float(db),
            })

    peaks.sort(key=lambda p: p["frame"])
    return peaks


# ----------------------------------------------------------------------
# 2. Hashing
# ----------------------------------------------------------------------

def hashes_paired(peaks, fan_value=10, min_dt=1, max_dt=100):
    """
    Classic Shazam-style combinatorial hashing. For every anchor peak,
    pair it with up to `fan_value` other peaks that come later in time
    within [min_dt, max_dt] frames ("the target zone"). Each pair becomes
    one hash: (f1_bin, f2_bin, delta_t_frames).

    Returns
    -------
    list of (hash_key, anchor_frame) tuples
    """
    out = []
    n = len(peaks)
    for i in range(n):
        anchor = peaks[i]
        paired = 0
        for j in range(i + 1, n):
            other = peaks[j]
            dt = other["frame"] - anchor["frame"]
            if dt < min_dt:
                continue
            if dt > max_dt:
                break  # peaks are time-sorted, no point scanning further
            key = (anchor["hbin"], other["hbin"], dt)
            out.append((key, anchor["frame"]))
            paired += 1
            if paired >= fan_value:
                break
    return out


def hashes_single(peaks):
    """
    Baseline for comparison: hash each peak's frequency bin on its own,
    with no pairing / no timing information between peaks.

    Returns
    -------
    list of (hash_key, anchor_frame) tuples
    """
    return [((p["hbin"],), p["frame"]) for p in peaks]


# ----------------------------------------------------------------------
# 3. Database
# ----------------------------------------------------------------------

class FingerprintDB:
    def __init__(self, win_length=2048, hop_length=512, sr=22050,
                 amp_min_db=-40.0, freq_nbhd=10, time_nbhd=10,
                 max_peaks_per_frame=5, fan_value=10, min_dt=1, max_dt=100):
        self.win_length = win_length
        self.hop_length = hop_length
        self.sr = sr
        self.amp_min_db = amp_min_db
        self.freq_nbhd = freq_nbhd
        self.time_nbhd = time_nbhd
        self.max_peaks_per_frame = max_peaks_per_frame
        self.fan_value = fan_value
        self.min_dt = min_dt
        self.max_dt = max_dt

        self.songs = {}            # song_id -> song_name (filename w/o ext)
        self.paired_index = defaultdict(list)  # hash -> [(song_id, frame), ...]
        self.single_index = defaultdict(list)

    # --- helpers --------------------------------------------------
    def _analyze(self, y):
        freqs, times, S_db = spectrogram_db(
            y, self.sr, self.win_length, self.hop_length)
        peaks = find_constellation_peaks(
            S_db, freqs, times,
            amp_min_db=self.amp_min_db,
            freq_nbhd=self.freq_nbhd, time_nbhd=self.time_nbhd,
            max_peaks_per_frame=self.max_peaks_per_frame)
        return freqs, times, S_db, peaks

    def fingerprint_array(self, y):
        """Run the full pipeline on an in-memory waveform `y`."""
        freqs, times, S_db, peaks = self._analyze(y)
        paired = hashes_paired(peaks, self.fan_value, self.min_dt, self.max_dt)
        single = hashes_single(peaks)
        return {
            "freqs": freqs, "times": times, "S_db": S_db, "peaks": peaks,
            "paired": paired, "single": single,
        }

    # --- building the DB -------------------------------------------
    def add_song(self, path, song_name=None):
        if song_name is None:
            song_name = os.path.splitext(os.path.basename(path))[0]
        y, sr = load_audio(path, sr=self.sr)
        return self.add_song_array(y, song_name)

    def add_song_array(self, y, song_name):
        song_id = len(self.songs)
        self.songs[song_id] = song_name
        fp = self.fingerprint_array(y)
        for key, frame in fp["paired"]:
            self.paired_index[key].append((song_id, frame))
        for key, frame in fp["single"]:
            self.single_index[key].append((song_id, frame))
        return song_id

    def build_from_folder(self, folder, ext=(".mp3", ".wav", ".m4a")):
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(ext))
        for f in files:
            self.add_song(os.path.join(folder, f))
        return list(self.songs.values())

    # --- matching ----------------------------------------------------
    def match(self, y, mode="paired", top_k=3):
        """
        Identify a query waveform `y` against the database.

        Returns
        -------
        result : dict with:
          - 'ranked'    : list of (song_id, song_name, votes) best-first
          - 'best'      : song_name of the top match (or None)
          - 'histogram' : {offset: count} for the BEST song (for plotting)
          - 'all_histograms' : {song_id: {offset: count}} for every song
            that received at least one matching hash (useful for the
            offset-histogram panel in the app)
          - 'query_fp'  : the fingerprint dict for the query (spectrogram,
            peaks, etc.) so the app can also show those panels.
        """
        fp = self.fingerprint_array(y)
        query_hashes = fp["paired"] if mode == "paired" else fp["single"]
        index = self.paired_index if mode == "paired" else self.single_index

        # song_id -> {offset: count}
        offset_votes = defaultdict(lambda: defaultdict(int))
        for key, q_frame in query_hashes:
            for song_id, db_frame in index.get(key, []):
                offset = db_frame - q_frame
                offset_votes[song_id][offset] += 1

        scored = []
        for song_id, hist in offset_votes.items():
            best_offset, best_count = max(hist.items(), key=lambda kv: kv[1])
            scored.append((song_id, best_count, best_offset))
        scored.sort(key=lambda x: x[1], reverse=True)

        ranked = [(sid, self.songs[sid], votes) for sid, votes, _off in scored[:top_k]]
        best_name = ranked[0][1] if ranked else None
        best_hist = dict(offset_votes[scored[0][0]]) if scored else {}
        all_hists = {sid: dict(h) for sid, h in offset_votes.items()}

        return {
            "ranked": ranked,
            "best": best_name,
            "histogram": best_hist,
            "all_histograms": all_hists,
            "query_fp": fp,
        }
