# zcr.py

from typing import Tuple
import numpy as np


def compute_zcr(frames: np.ndarray) -> np.ndarray:
    """
    Computes Zero-Crossing Rate (ZCR) for each frame.

    ZCR measures how often the signal changes sign within a frame.
    It is useful to distinguish voiced vs unvoiced / noise-like segments.

    Parameters
    ----------
    frames : np.ndarray
        2D array of shape (num_frames, frame_length_samples).
        Each row is a windowed audio frame (float32, typically in [-1, 1]).

    Returns
    -------
    zcr_values : np.ndarray
        1D array of length num_frames with ZCR per frame.
        Values are between 0 and 1.
    """
    if frames.ndim != 2:
        raise ValueError("frames must be a 2D array of shape (num_frames, frame_length).")

    # Sign of samples: +1 for positive, -1 for negative, 0 treated as no sign
    signs = np.sign(frames)
    # For exact zeros, treat them as previous sample sign to avoid artificial crossings
    signs[signs == 0] = 1.0

    # Difference between consecutive samples along time axis
    sign_changes = np.abs(signs[:, 1:] - signs[:, :-1])

    # Each sign change contributes 2 to the sum (from +1 to -1 or -1 to +1)
    # So number of zero-crossings = sum(sign_changes) / 2
    num_zc = np.sum(sign_changes, axis=1) / 2.0

    frame_length = frames.shape[1]
    # Normalize to [0, 1] by dividing by (frame_length - 1)
    # since there are (frame_length - 1) possible positions where a crossing can occur
    zcr_values = (num_zc / (frame_length - 1)).astype(np.float32)

    return zcr_values


def detect_unvoiced_or_noisy_frames(
    zcr_values: np.ndarray,
    zcr_threshold: float = 0.15
) -> np.ndarray:
    """
    Detects frames that are likely unvoiced or noise-like based on ZCR.

    Parameters
    ----------
    zcr_values : np.ndarray
        1D array of ZCR values per frame (between 0 and 1).
    zcr_threshold : float
        Frames with ZCR above this value are considered unvoiced/noisy.
        Default: 0.15.

    Returns
    -------
    unvoiced_mask : np.ndarray
        Boolean array of length len(zcr_values).
        True where frame is classified as unvoiced/noisy, False otherwise.
    """
    if zcr_values.ndim != 1:
        raise ValueError("zcr_values must be a 1D array.")

    unvoiced_mask = zcr_values > zcr_threshold
    return unvoiced_mask


def detect_noise_frames(
    zcr_values: np.ndarray,
    rms_values: np.ndarray,
    zcr_threshold: float = 0.25,
    rms_threshold: float = 0.01
) -> np.ndarray:
    """
    Example helper: detect frames that are likely 'noise' (e.g., background fan)
    using a combination of high ZCR and low-moderate RMS.

    Parameters
    ----------
    zcr_values : np.ndarray
        1D array of ZCR values per frame.
    rms_values : np.ndarray
        1D array of RMS values per frame.
    zcr_threshold : float
        Minimum ZCR to be considered noise-like.
    rms_threshold : float
        Minimum RMS to consider that the frame has some energy (not silence).

    Returns
    -------
    noise_mask : np.ndarray
        Boolean array of length len(zcr_values).
        True where frame is likely noise.
    """
    if zcr_values.shape != rms_values.shape:
        raise ValueError("zcr_values and rms_values must have the same shape.")

    # Noise-like: relatively high ZCR and some energy present
    noise_mask = (zcr_values > zcr_threshold) & (rms_values > rms_threshold)
    return noise_mask


if __name__ == "__main__":
    # Quick test with synthetic data
    import matplotlib.pyplot as plt

    num_frames = 100
    frame_length = 400

    # Voiced-like: low ZCR (slow sine wave)
    t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
    voiced_frame = 0.1 * np.sin(3 * t)  # low-frequency sine

    # Unvoiced-like: high ZCR (white noise)
    unvoiced_frame = 0.1 * np.random.randn(frame_length)

    frames = []
    for i in range(num_frames):
        if i < 50:
            frames.append(voiced_frame)
        else:
            frames.append(unvoiced_frame)
    frames = np.stack(frames).astype(np.float32)

    zcr_vals = compute_zcr(frames)
    unvoiced_mask = detect_unvoiced_or_noisy_frames(zcr_vals, zcr_threshold=0.15)

    print("ZCR values (first 10):", zcr_vals[:10])
    print("Number of frames marked unvoiced:", np.sum(unvoiced_mask))

    plt.figure()
    plt.plot(zcr_vals, label="ZCR per frame")
    plt.axhline(0.15, linestyle="--", label="Unvoiced threshold")
    plt.legend()
    plt.xlabel("Frame index")
    plt.ylabel("ZCR")
    plt.title("ZCR and Unvoiced Detection")
    plt.show()
