# cepstrum_clarity.py

from typing import Tuple
import numpy as np


def compute_real_cepstrum(frame: np.ndarray) -> np.ndarray:
    """
    Computes the real cepstrum of a single audio frame.

    Cepstrum c[n] = IFFT( log( |FFT(x[n])| ) )

    Parameters
    ----------
    frame : np.ndarray
        1D array, windowed audio frame (float32).

    Returns
    -------
    cepstrum : np.ndarray
        1D array of real cepstral coefficients, same length as frame.
    """
    x = frame.astype(np.float32)

    # Avoid log(0): add a very small epsilon
    eps = 1e-9

    # Use FFT to get frequency-domain representation
    spectrum = np.fft.rfft(x)
    mag = np.abs(spectrum)

    # Log magnitude spectrum
    log_mag = np.log(mag + eps)

    # Real cepstrum via inverse FFT of log magnitude
    cepstrum_full = np.fft.irfft(log_mag, n=len(x))

    return cepstrum_full.astype(np.float32)


def compute_clarity_index_for_frame(
    frame: np.ndarray,
    sr: int,
    low_quef_max_ms: float = 2.0,
    high_quef_min_ms: float = 2.0,
    high_quef_max_ms: float = 12.0
) -> float:
    """
    Computes a simple 'clarity index' for a single frame using cepstrum.

    Idea:
    - Very low quefrencies correspond to the spectral envelope / vocal tract (timbre).
    - Higher quefrencies (around pitch period region) capture fine structure
      and potential echo/reverberation patterns.
    - A clearer voice with less echo tends to have stronger low-quefrency
      envelope relative to smeared high-quefrency content.

    We define:
        ClarityIndex = Energy(low_quef_region) / Energy(high_quef_region)

    Parameters
    ----------
    frame : np.ndarray
        1D windowed frame (float32).
    sr : int
        Sampling rate in Hz.
    low_quef_max_ms : float
        Upper bound of 'low quefrency' region (in milliseconds).
        Default: 2 ms.
    high_quef_min_ms : float
        Lower bound of 'high quefrency' region (in milliseconds).
        Default: 2 ms.
    high_quef_max_ms : float
        Upper bound of 'high quefrency' region (in milliseconds).
        Default: 12 ms.

    Returns
    -------
    clarity_index : float
        Ratio of low-quefrency energy to high-quefrency energy.
        Higher values roughly correspond to clearer / less reverberant speech.
        Returns 0.0 for near-silent frames or invalid regions.
    """
    # Skip very low-energy frames to avoid nonsense
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0

    cep = compute_real_cepstrum(frame)
    L = len(cep)

    # Convert ms to quefrency indices: quefrency step = 1 / sr seconds
    # index = time_sec * sr
    low_max_idx = int((low_quef_max_ms / 1000.0) * sr)
    high_min_idx = int((high_quef_min_ms / 1000.0) * sr)
    high_max_idx = int((high_quef_max_ms / 1000.0) * sr)

    # Ensure indices are valid and within[0, L-1]
    low_max_idx = max(1, min(low_max_idx, L - 1))
    high_min_idx = max(1, min(high_min_idx, L - 1))
    high_max_idx = max(high_min_idx + 1, min(high_max_idx, L))

    # Define regions:
    # Low quefrency: [0, low_max_idx)
    # High quefrency: [high_min_idx, high_max_idx)
    low_region = cep[0:low_max_idx]
    high_region = cep[high_min_idx:high_max_idx]

    # Compute energies
    low_energy = float(np.sum(low_region ** 2))
    high_energy = float(np.sum(high_region ** 2)) + 1e-9  # avoid division by zero

    clarity_index = low_energy / high_energy
    return clarity_index


def compute_clarity_over_time(
    frames: np.ndarray,
    sr: int,
    low_quef_max_ms: float = 2.0,
    high_quef_min_ms: float = 2.0,
    high_quef_max_ms: float = 12.0
) -> np.ndarray:
    """
    Computes clarity index for each frame in a sequence of frames.

    Parameters
    ----------
    frames : np.ndarray
        2D array of shape (num_frames, frame_length).
        Each row is a windowed audio frame.
    sr : int
        Sampling rate in Hz.
    low_quef_max_ms : float
        Upper bound for low quefrency region (ms).
    high_quef_min_ms : float
        Lower bound for high quefrency region (ms).
    high_quef_max_ms : float
        Upper bound for high quefrency region (ms).

    Returns
    -------
    clarity_values : np.ndarray
        1D array of clarity index values per frame.
    """
    if frames.ndim != 2:
        raise ValueError("frames must be a 2D array of shape (num_frames, frame_length).")

    num_frames = frames.shape[0]
    clarity_values = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        clarity_values[i] = compute_clarity_index_for_frame(
            frames[i, :],
            sr,
            low_quef_max_ms=low_quef_max_ms,
            high_quef_min_ms=high_quef_min_ms,
            high_quef_max_ms=high_quef_max_ms
        )

    return clarity_values


if __name__ == "__main__":
    # Quick synthetic test to visualize clarity index
    import matplotlib.pyplot as plt

    sr = 16000
    frame_length = int(0.032 * sr)  # 32 ms frame
    t = np.arange(frame_length) / sr

    # Clear voiced frame: simple sine with some harmonics
    clear_frame = 0.2 * np.sin(2 * np.pi * 150 * t) \
                  + 0.05 * np.sin(2 * np.pi * 300 * t)

    # "Muffled" frame: apply simple lowpass-like effect (more smoothed)
    # Simulate by convolving with a small smoothing kernel
    kernel = np.ones(5) / 5.0
    muffled_frame = np.convolve(clear_frame, kernel, mode='same')

    # Make multiple frames: first 50 clear, next 50 muffled
    frames_list = []
    for i in range(50):
        frames_list.append(clear_frame)
    for i in range(50):
        frames_list.append(muffled_frame)

    frames = np.stack(frames_list).astype(np.float32)

    clarity_vals = compute_clarity_over_time(frames, sr)

    print("First 10 clarity values:", clarity_vals[:10])

    plt.figure()
    plt.plot(clarity_vals, label="Clarity Index")
    plt.xlabel("Frame index")
    plt.ylabel("Clarity")
    plt.title("Clarity Index over Frames (Higher ~ Clearer)")
    plt.legend()
    plt.show()
