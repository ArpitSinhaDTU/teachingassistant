# pitch_autocorr.py

from typing import Tuple
import numpy as np


def _autocorrelation(frame: np.ndarray) -> np.ndarray:
    """
    Computes full autocorrelation of a 1D frame and returns only non-negative lags.

    Parameters
    ----------
    frame : np.ndarray
        1D array, a single audio frame (windowed).

    Returns
    -------
    acf : np.ndarray
        Autocorrelation sequence for lags >= 0.
    """
    # Ensure float32
    x = frame.astype(np.float32)

    # Full autocorrelation using 'full' mode
    acf_full = np.correlate(x, x, mode='full')
    mid = len(acf_full) // 2
    acf = acf_full[mid:]  # non-negative lags

    return acf


def compute_pitch_autocorr(
    frames: np.ndarray,
    sr: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
    voicing_threshold: float = 0.3
) -> np.ndarray:
    """
    Estimates pitch (fundamental frequency f0) for each frame using autocorrelation.

    Parameters
    ----------
    frames : np.ndarray
        2D array of shape (num_frames, frame_length_samples).
        Each row is a windowed audio frame.
    sr : int
        Sampling rate in Hz.
    fmin : float
        Minimum plausible pitch frequency (Hz). Default: 60 Hz.
    fmax : float
        Maximum plausible pitch frequency (Hz). Default: 400 Hz.
    voicing_threshold : float
        Minimum normalized autocorrelation peak height (relative to lag 0)
        required to consider a frame as voiced.

    Returns
    -------
    f0_values : np.ndarray
        1D array of length num_frames.
        f0 in Hz for voiced frames; 0.0 for unvoiced/uncertain frames.
    """
    if frames.ndim != 2:
        raise ValueError("frames must be a 2D array of shape (num_frames, frame_length).")

    num_frames, frame_length = frames.shape

    # Convert pitch range in Hz to lag range in samples
    max_lag = int(sr / fmin)  # lowest pitch -> longest period
    min_lag = int(sr / fmax)  # highest pitch -> shortest period

    if min_lag < 1:
        min_lag = 1
    if max_lag >= frame_length:
        max_lag = frame_length - 1

    f0_values = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        frame = frames[i, :]

        # Skip if frame is almost silence (avoid random pitch)
        if np.max(np.abs(frame)) < 1e-4:
            f0_values[i] = 0.0
            continue

        acf = _autocorrelation(frame)

        # Normalize ACF by value at lag 0 to get correlation coefficient
        if acf[0] <= 0:
            f0_values[i] = 0.0
            continue
        acf_norm = acf / acf[0]

        # Search for maximum peak within [min_lag, max_lag]
        search_region = acf_norm[min_lag:max_lag]
        if len(search_region) == 0:
            f0_values[i] = 0.0
            continue

        peak_index = np.argmax(search_region)
        peak_value = search_region[peak_index]
        lag = min_lag + peak_index

        # Check voicing threshold
        if peak_value < voicing_threshold:
            f0_values[i] = 0.0
        else:
            # Convert lag (samples) to frequency (Hz)
            f0 = sr / float(lag)
            f0_values[i] = f0

    return f0_values


def compute_pitch_statistics(
    f0_values: np.ndarray,
    voiced_min_hz: float = 40.0
) -> Tuple[float, float, float]:
    """
    Computes basic statistics of pitch across frames:
    mean, standard deviation, and voiced frame ratio.

    Parameters
    ----------
    f0_values : np.ndarray
        1D array of f0 values per frame (Hz), 0.0 for unvoiced.
    voiced_min_hz : float
        Minimum f0 to consider frame as voiced (to reject tiny numerical noise).

    Returns
    -------
    mean_f0 : float
        Mean pitch over voiced frames (Hz). 0.0 if no voiced frames.
    std_f0 : float
        Standard deviation of pitch over voiced frames (Hz). 0.0 if no voiced frames.
    voiced_ratio : float
        Fraction of frames that are voiced (between 0 and 1).
    """
    if f0_values.ndim != 1:
        raise ValueError("f0_values must be a 1D array.")

    voiced_mask = f0_values > voiced_min_hz
    num_frames = len(f0_values)
    num_voiced = np.sum(voiced_mask)

    if num_voiced == 0:
        return 0.0, 0.0, 0.0

    voiced_f0 = f0_values[voiced_mask]
    mean_f0 = float(np.mean(voiced_f0))
    std_f0 = float(np.std(voiced_f0))
    voiced_ratio = float(num_voiced) / float(num_frames)

    return mean_f0, std_f0, voiced_ratio


if __name__ == "__main__":
    # Quick synthetic test: mixture of low-pitch and high-pitch tones
    import matplotlib.pyplot as plt

    sr = 16000
    frame_length = int(0.03 * sr)  # 30 ms
    t = np.arange(frame_length) / sr

    # 50 frames of ~120 Hz, 50 frames of ~220 Hz
    frames_list = []
    for i in range(50):
        frames_list.append(0.1 * np.sin(2 * np.pi * 120 * t))
    for i in range(50):
        frames_list.append(0.1 * np.sin(2 * np.pi * 220 * t))

    frames = np.stack(frames_list).astype(np.float32)
    f0_vals = compute_pitch_autocorr(frames, sr)

    mean_f0, std_f0, voiced_ratio = compute_pitch_statistics(f0_vals)

    print("First 10 f0 values:", f0_vals[:10])
    print("Mean f0:", mean_f0)
    print("Std f0:", std_f0)
    print("Voiced ratio:", voiced_ratio)

    plt.figure()
    plt.plot(f0_vals, label="Estimated f0 (Hz)")
    plt.xlabel("Frame index")
    plt.ylabel("f0 (Hz)")
    plt.title("Pitch Estimation via Autocorrelation")
    plt.legend()
    plt.show()
