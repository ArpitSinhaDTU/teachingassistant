# chunk_features.py

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np

from whisper_groq_features import (
    Word,
    compute_speaking_rates,
    compute_pause_stats,
    compute_filler_rate,
)


def compute_time_chunks(
    total_duration: float,
    chunk_size: float = 10.0
) -> List[Tuple[float, float]]:
    """
    Splits [0, total_duration) into fixed-size time chunks.

    Parameters
    ----------
    total_duration : float
        Total duration of audio/video (seconds).
    chunk_size : float
        Desired chunk length in seconds. Default: 10.0 s.

    Returns
    -------
    chunks : List[Tuple[float, float]]
        List of (t_start, t_end) for each chunk.
    """
    chunks: List[Tuple[float, float]] = []
    t = 0.0
    while t < total_duration:
        t_start = t
        t_end = min(t + chunk_size, total_duration)
        chunks.append((t_start, t_end))
        t = t_end
    return chunks


def aggregate_frame_feature_in_window(
    frame_times: np.ndarray,
    feature_values: np.ndarray,
    t_start: float,
    t_end: float,
    agg: str = "mean",
) -> float:
    """
    Aggregates a frame-level feature (e.g. RMS, ZCR, pitch, motion)
    over a time window.

    Parameters
    ----------
    frame_times : np.ndarray
        1D array of frame center times in seconds.
    feature_values : np.ndarray
        1D array of same length, feature value per frame.
    t_start : float
        Window start time in seconds.
    t_end : float
        Window end time in seconds.
    agg : str
        Aggregation type: 'mean', 'median', 'std', 'max'.

    Returns
    -------
    value : float
        Aggregated feature value over frames in [t_start, t_end).
        Returns 0.0 if no frames fall in the window.
    """
    if frame_times.shape != feature_values.shape:
        raise ValueError("frame_times and feature_values must have same shape.")

    mask = (frame_times >= t_start) & (frame_times < t_end)
    vals = feature_values[mask]

    if vals.size == 0:
        return 0.0

    if agg == "mean":
        return float(np.mean(vals))
    elif agg == "median":
        return float(np.median(vals))
    elif agg == "std":
        return float(np.std(vals))
    elif agg == "max":
        return float(np.max(vals))
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")


def aggregate_pitch_stats_in_window(
    frame_times: np.ndarray,
    f0_values: np.ndarray,
    t_start: float,
    t_end: float,
    voiced_min_hz: float = 40.0
) -> Dict[str, float]:
    """
    Computes pitch statistics (mean, std, voiced ratio) within a time window.

    Parameters
    ----------
    frame_times : np.ndarray
        1D array of frame center times (seconds).
    f0_values : np.ndarray
        1D array of pitch values per frame (Hz), 0.0 for unvoiced.
    t_start : float
        Window start time (seconds).
    t_end : float
        Window end time (seconds).
    voiced_min_hz : float
        Min f0 to consider as voiced.

    Returns
    -------
    stats : Dict[str, float]
        'mean_f0', 'std_f0', 'voiced_ratio' (0..1)
    """
    if frame_times.shape != f0_values.shape:
        raise ValueError("frame_times and f0_values must have same shape.")

    mask = (frame_times >= t_start) & (frame_times < t_end)
    f0_win = f0_values[mask]

    if f0_win.size == 0:
        return {"mean_f0": 0.0, "std_f0": 0.0, "voiced_ratio": 0.0}

    voiced_mask = f0_win > voiced_min_hz
    num_voiced = np.sum(voiced_mask)
    total = f0_win.size

    if num_voiced == 0:
        return {"mean_f0": 0.0, "std_f0": 0.0, "voiced_ratio": 0.0}

    voiced_f0 = f0_win[voiced_mask]
    return {
        "mean_f0": float(np.mean(voiced_f0)),
        "std_f0": float(np.std(voiced_f0)),
        "voiced_ratio": float(num_voiced) / float(total),
    }


def build_chunk_feature_list(
    frame_times_audio: np.ndarray,
    rms_values: np.ndarray,
    zcr_values: np.ndarray,
    f0_values: np.ndarray,
    clarity_values: np.ndarray,
    words: List[Word],
    chunk_size: float = 10.0,
    total_duration: Optional[float] = None,
    # NEW: video motion (optional)
    frame_times_video: Optional[np.ndarray] = None,
    motion_values: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """
    Builds a list of feature dictionaries, one per chunk.

    Each chunk (window) aggregates:
        - Audio frame-based:
            * RMS (mean)
            * ZCR (mean)
            * Pitch mean / std / voiced_ratio
            * Clarity (mean)
        - Text-based (from Groq Whisper):
            * Speaking rate (WPM)
            * Articulation rate (WPM)
            * #words in chunk
            * Pause stats: num_pauses, avg_pause, max_pause
            * Filler rate
        - Video-based (optional):
            * motion_mean  (global motion index)

    Parameters
    ----------
    frame_times_audio : np.ndarray
        1D array of audio frame times (seconds).
    rms_values : np.ndarray
        1D array RMS per audio frame.
    zcr_values : np.ndarray
        1D array ZCR per audio frame.
    f0_values : np.ndarray
        1D pitch per audio frame.
    clarity_values : np.ndarray
        1D clarity index per audio frame.
    words : List[Word]
        Full word list for entire audio from Groq-Whisper.
    chunk_size : float
        Length of each analysis chunk (seconds). Default: 10.0.
    total_duration : float or None
        If None, inferred from max(audio_frame_times) or last word end.
    frame_times_video : np.ndarray or None
        1D array of video frame times (seconds). Optional.
    motion_values : np.ndarray or None
        1D array of global motion index per video frame. Optional.

    Returns
    -------
    chunk_features : List[Dict[str, float]]
        List of dictionaries, one per chunk, with keys like:
        't_start', 't_end', 'rms_mean', 'zcr_mean', 'f0_mean', 'f0_std',
        'voiced_ratio', 'clarity_mean', 'sr_wpm', 'ar_wpm', 'num_words',
        'num_pauses', 'avg_pause', 'max_pause', 'filler_rate', 'num_fillers',
        'motion_mean' (if video features provided).
    """
    # Infer total duration if not given
    if total_duration is None:
        t1_frames = float(frame_times_audio[-1]) if frame_times_audio.size > 0 else 0.0
        t1_words = float(words[-1].end) if len(words) > 0 else 0.0
        # If video is longer than audio, include that too
        if frame_times_video is not None and frame_times_video.size > 0:
            t1_video = float(frame_times_video[-1])
        else:
            t1_video = 0.0

        total_duration = max(t1_frames, t1_words, t1_video)

    chunks = compute_time_chunks(total_duration, chunk_size=chunk_size)

    chunk_features: List[Dict[str, float]] = []

    for (t_start, t_end) in chunks:
        # ---- Audio frame-based features ----
        rms_mean = aggregate_frame_feature_in_window(
            frame_times_audio, rms_values, t_start, t_end, agg="mean"
        )
        zcr_mean = aggregate_frame_feature_in_window(
            frame_times_audio, zcr_values, t_start, t_end, agg="mean"
        )
        clarity_mean = aggregate_frame_feature_in_window(
            frame_times_audio, clarity_values, t_start, t_end, agg="mean"
        )
        pitch_stats = aggregate_pitch_stats_in_window(
            frame_times_audio, f0_values, t_start, t_end
        )

        # ---- Text-based features (Groq-Whisper words) ----
        sr_wpm, ar_wpm, n_words = compute_speaking_rates(
            words, t_start, t_end
        )
        pause_stats = compute_pause_stats(
            words, t_start, t_end
        )
        filler_rate, n_fillers, n_words_fr = compute_filler_rate(
            words, t_start, t_end
        )

        # ---- Video motion (optional) ----
        if frame_times_video is not None and motion_values is not None:
            motion_mean = aggregate_frame_feature_in_window(
                frame_times_video, motion_values, t_start, t_end, agg="mean"
            )
        else:
            motion_mean = 0.0

        feat = {
            "t_start": t_start,
            "t_end": t_end,
            # audio-frame based
            "rms_mean": rms_mean,
            "zcr_mean": zcr_mean,
            "clarity_mean": clarity_mean,
            "f0_mean": pitch_stats["mean_f0"],
            "f0_std": pitch_stats["std_f0"],
            "voiced_ratio": pitch_stats["voiced_ratio"],
            # word-based
            "sr_wpm": sr_wpm,
            "ar_wpm": ar_wpm,
            "num_words": float(n_words),
            "num_pauses": pause_stats["num_pauses"],
            "avg_pause": pause_stats["avg_pause"],
            "max_pause": pause_stats["max_pause"],
            "filler_rate": filler_rate,
            "num_fillers": float(n_fillers),
            # video-based
            "motion_mean": motion_mean,
        }

        chunk_features.append(feat)

    return chunk_features


if __name__ == "__main__":
    # Tiny mock test (fake data) just to show structure
    frame_times_audio = np.linspace(0, 100, 1000)
    rms_vals = np.random.rand(1000).astype(np.float32) * 0.05
    zcr_vals = np.random.rand(1000).astype(np.float32) * 0.2
    f0_vals = np.random.choice([0.0, 120.0, 200.0], size=1000).astype(np.float32)
    clarity_vals = np.random.rand(1000).astype(np.float32) * 5.0

    # Fake video motion
    frame_times_video = np.linspace(0, 100, 500)
    motion_vals = np.random.rand(500).astype(np.float32) * 0.05

    # Fake word list
    from whisper_groq_features import Word
    words = []
    t = 0.5
    for i in range(300):
        words.append(Word(text="hello", start=t, end=t + 0.3))
        t += 0.5

    chunk_feats = build_chunk_feature_list(
        frame_times_audio,
        rms_vals,
        zcr_vals,
        f0_vals,
        clarity_vals,
        words,
        chunk_size=10.0,
        frame_times_video=frame_times_video,
        motion_values=motion_vals,
    )

    print("Num chunks:", len(chunk_feats))
    print("First chunk feature dict:\n", chunk_feats[0])
