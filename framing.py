# framing.py

from typing import Tuple
import numpy as np
from scipy.io import wavfile


def load_audio_wav(
    wav_path: str,
    target_sr: int = 16000
) -> Tuple[int, np.ndarray]:
    """
    Loads a WAV audio file and returns sampling rate and mono float32 signal
    normalized to [-1, 1].

    Parameters
    ----------
    wav_path : str
        Path to the WAV file.
    target_sr : int
        Expected sampling rate in Hz (default: 16000).
        If file has a different sr, a warning is printed, but audio is still returned.

    Returns
    -------
    sr : int
        Sampling rate of the loaded audio (from file).
    audio : np.ndarray
        1D numpy array of float32 samples in range [-1, 1].
    """
    sr, data = wavfile.read(wav_path)

    # Convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data
    else:
        # Fallback: convert to float32 safely
        data = data.astype(np.float32)
        max_val = np.max(np.abs(data)) + 1e-9
        audio = data / max_val

    # If stereo, convert to mono by averaging channels
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        print(f"[WARNING] File SR = {sr} Hz, but expected {target_sr} Hz. "
              f"Consider resampling during extraction step.")

    return sr, audio


def frame_signal(
    audio: np.ndarray,
    sr: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0,
    window_type: str = "hamming"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the audio signal into overlapping frames and applies a window.

    Parameters
    ----------
    audio : np.ndarray
        1D mono audio signal (float32, [-1, 1]).
    sr : int
        Sampling rate in Hz.
    frame_size_ms : float
        Frame length in milliseconds (default: 25 ms).
    hop_size_ms : float
        Hop length in milliseconds (default: 10 ms).
    window_type : str
        Type of window to apply: 'hamming' or 'none'.

    Returns
    -------
    frames : np.ndarray
        2D array with shape (num_frames, frame_length_samples).
    frame_times : np.ndarray
        1D array with center time (in seconds) of each frame.
    """
    frame_length = int(sr * frame_size_ms / 1000.0)
    hop_length = int(sr * hop_size_ms / 1000.0)

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("Frame and hop lengths must be positive.")

    num_samples = len(audio)
    if num_samples < frame_length:
        raise ValueError("Audio too short for even one frame.")

    # Compute number of frames (last frame may be truncated/discarded)
    num_frames = 1 + (num_samples - frame_length) // hop_length

    # Prepare window
    if window_type == "hamming":
        window = np.hamming(frame_length).astype(np.float32)
    elif window_type == "none":
        window = np.ones(frame_length, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported window_type: {window_type}")

    frames = np.zeros((num_frames, frame_length), dtype=np.float32)
    frame_times = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        # Apply window
        frames[i, :] = frame * window
        # Time at center of frame
        center_sample = start + frame_length / 2.0
        frame_times[i] = center_sample / sr

    return frames, frame_times


if __name__ == "__main__":
    # Example usage
    wav_path = "lecture_audio.wav"
    sr, audio = load_audio_wav(wav_path)
    frames, frame_times = frame_signal(audio, sr)
    print(f"Sampling rate: {sr} Hz")
    print(f"Audio length: {len(audio) / sr:.2f} s")
    print(f"Frames shape: {frames.shape}")
    print(f"First 5 frame times: {frame_times[:5]}")
