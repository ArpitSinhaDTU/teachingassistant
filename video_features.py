# video_features.py

from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple


def compute_global_motion(
    video_path: str,
    frame_skip: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes global motion index per video frame using absolute frame differencing.

    Parameters
    ----------
    video_path : str
        Input video file.
    frame_skip : int
        Process every nth frame to reduce computation (default=1).

    Returns
    -------
    frame_times : np.ndarray
        Time in seconds for each processed frame.
    global_motion : np.ndarray
        Motion index per frame (0..1 approx).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback if metadata missing

    prev_gray = None
    frame_times = []
    motion_vals = []

    frame_idx = 0   # raw frame counter
    used_idx = 0    # processed frame index

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_skip > 1 and (frame_idx % frame_skip != 0):
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            motion_index = float(np.mean(diff) / 255.0)

            time_sec = used_idx * frame_skip / fps
            frame_times.append(time_sec)
            motion_vals.append(motion_index)

            used_idx += 1

        prev_gray = gray
        frame_idx += 1

    cap.release()

    return (
        np.array(frame_times, dtype=np.float32),
        np.array(motion_vals, dtype=np.float32)
    )


if __name__ == "__main__":
    v = "lecture.mp4"
    ft, mv = compute_global_motion(v)
    print("Frames processed:", len(ft))
    print("First 10 motion values:", mv[:10])
