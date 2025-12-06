# audio_extraction.py

import os
from typing import Optional

from moviepy import VideoFileClip


def extract_audio_from_video(
    video_path: str,
    output_wav_path: Optional[str] = None,
    target_sr: int = 16000,
    mono: bool = True
) -> str:
    """
    Extracts audio from a video file and saves it as a WAV file.

    Parameters
    ----------
    video_path : str
        Path to the input video file (e.g., 'lecture.mp4').
    output_wav_path : Optional[str]
        Path where the extracted audio will be saved as WAV.
        If None, uses same name as video with '.wav' extension.
    target_sr : int
        Target sampling rate in Hz (default: 16000).
    mono : bool
        If True, converts audio to mono (single channel).

    Returns
    -------
    str
        Path to the saved WAV file.

    Raises
    ------
    FileNotFoundError
        If the input video_path does not exist.
    ValueError
        If the video has no audio track.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    # If no output path provided, create one based on video name
    if output_wav_path is None:
        base, _ = os.path.splitext(video_path)
        output_wav_path = base + "_audio.wav"

    # Load video
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        raise ValueError(f"No audio track found in video: {video_path}")

    # Extract audio
    audio = clip.audio

    # Write the WAV file with specified parameters
    # In MoviePy 2.x, set nchannels and fps directly in write_audiofile
    audio.write_audiofile(
        output_wav_path,
        fps=target_sr,
        nbytes=2,           # 16-bit PCM
        codec='pcm_s16le',  # Uncompressed PCM
        ffmpeg_params=["-ac", "1"] if mono else None  # -ac 1 for mono
    )

    # Close resources
    clip.close()

    return output_wav_path


if __name__ == "__main__":
    # Example usage (you can remove this in final repo)
    in_video = "lecture.mp4"
    out_wav = "lecture_audio.wav"
    try:
        result_path = extract_audio_from_video(in_video, out_wav)
        print(f"Audio extracted to: {result_path}")
    except Exception as e:
        print(f"Error: {e}")
