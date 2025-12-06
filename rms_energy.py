# rms_energy.py

from typing import Tuple
import numpy as np


def compute_frame_energy_rms(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes short-time energy and RMS for each frame.

    Parameters
    ----------
    frames : np.ndarray
        2D array of shape (num_frames, frame_length_samples).
        Each row is a windowed audio frame (float32, typically in [-1, 1]).

    Returns
    -------
    energies : np.ndarray
        1D array of length num_frames with short-time energy of each frame.
    rms_values : np.ndarray
        1D array of length num_frames with RMS value of each frame.
    """
    if frames.ndim != 2:
        raise ValueError("frames must be a 2D array of shape (num_frames, frame_length).")

    # Short-time energy: sum of squares for each frame
    energies = np.sum(frames ** 2, axis=1)

    # RMS: sqrt(energy / frame_length)
    frame_length = frames.shape[1]
    # Avoid division by zero
    if frame_length <= 0:
        raise ValueError("Frame length must be positive.")

    rms_values = np.sqrt(energies / frame_length).astype(np.float32)

    return energies.astype(np.float32), rms_values


def detect_silent_frames(
    rms_values: np.ndarray,
    silence_threshold: float = 0.005
) -> np.ndarray:
    """
    Detects which frames are considered 'silent' based on RMS threshold.

    Parameters
    ----------
    rms_values : np.ndarray
        1D array of RMS values for each frame.
    silence_threshold : float
        Frames with RMS below this value are labeled as 'silent'.
        Default: 0.005 (for audio normalized in [-1, 1]).

    Returns
    -------
    silent_mask : np.ndarray
        Boolean array of length len(rms_values).
        True where frame is classified as silent, False otherwise.
    """
    if rms_values.ndim != 1:
        raise ValueError("rms_values must be a 1D array.")

    silent_mask = rms_values < silence_threshold
    return silent_mask


if __name__ == "__main__":
    # Example usage / quick test
    # Create a fake signal: first half silence, second half speech-like noise
    import matplotlib.pyplot as plt

    num_frames = 100
    frame_length = 400  # e.g. 25 ms at 16 kHz

    # First 50 frames ~ silence (very low amplitude)
    silent_part = 0.0005 * np.random.randn(50, frame_length).astype(np.float32)
    # Next 50 frames ~ speech-like (higher amplitude)
    speech_part = 0.05 * np.random.randn(50, frame_length).astype(np.float32)

    frames = np.vstack([silent_part, speech_part])

    energies, rms_vals = compute_frame_energy_rms(frames)
    silent_mask = detect_silent_frames(rms_vals, silence_threshold=0.005)

    print("Energies shape:", energies.shape)
    print("RMS shape:", rms_vals.shape)
    print("Number of silent frames detected:", np.sum(silent_mask))

    # Quick plot to visualize
    plt.figure()
    plt.plot(rms_vals, label="RMS per frame")
    plt.axhline(0.005, linestyle="--", label="Silence threshold")
    plt.legend()
    plt.title("RMS values and silence threshold")
    plt.xlabel("Frame index")
    plt.ylabel("RMS")
    plt.show()
