import os
import subprocess
import tempfile
import numpy as np
from scipy.io import wavfile


def load_audio(path, sr=22050):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(path),
            "-ac", "1",            # mono
            "-ar", str(sr),        # resample
            tmp_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='ignore')}")
        file_sr, data = wavfile.read(tmp_path)
    finally:
        os.remove(tmp_path)

    assert file_sr == sr

    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float32) - 128.0) / 128.0
    else:
        y = data.astype(np.float32)

    return y, sr


def stft(y, sr, win_length=2048, hop_length=512, window="hann"):
    if window == "hann":
        win = np.hanning(win_length).astype(np.float32)
    elif window == "rect":
        win = np.ones(win_length, dtype=np.float32)
    else:
        raise ValueError(f"unknown window {window}")

    n_frames = 1 + (len(y) - win_length) // hop_length
    if n_frames < 1:
        raise ValueError("signal shorter than one window")

    # Only keep the non-redundant half of the spectrum (real input signal)
    n_bins = win_length // 2 + 1
    S = np.empty((n_bins, n_frames), dtype=np.complex64)

    for t in range(n_frames):
        start = t * hop_length
        frame = y[start:start + win_length] * win
        spec = np.fft.rfft(frame)
        S[:, t] = spec

    freqs = np.fft.rfftfreq(win_length, d=1.0 / sr)
    times = (np.arange(n_frames) * hop_length + win_length / 2) / sr
    return freqs, times, S


def spectrogram_db(y, sr, win_length=2048, hop_length=512, window="hann", top_db=80.0):
    freqs, times, S = stft(y, sr, win_length, hop_length, window)
    mag = np.abs(S)
    mag = np.maximum(mag, 1e-10)
    S_db = 20 * np.log10(mag)
    S_db -= S_db.max()
    S_db = np.maximum(S_db, -top_db)
    return freqs, times, S_db
