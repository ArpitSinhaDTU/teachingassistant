# whisper_groq_features.py

import os
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq
import numpy as np


# --------------------------------
# Load API key from environment
# --------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@dataclass
class Word:
    text: str
    start: float  # seconds (approx)
    end: float    # seconds (approx)


def transcribe_audio_to_words(
    audio_path: str,
    language: Optional[str] = None
) -> List[Word]:
    """
    Transcribes using Groq Whisper (Whisper-Large-V3).
    Groq gives only segment-level timestamps, so we approximate word times.

    Parameters
    ----------
    audio_path : str
        Path to WAV file (mono, 16kHz recommended).
    language : str or None
        Language hint. If None, Groq auto-detects.

    Returns
    -------
    List[Word]
        Word list with approximate start/end timestamps.
    """

    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",  # segments + timestamps
            language=language,
            temperature=0.0
        )

    words: List[Word] = []

    for seg in transcription.segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_text = seg["text"]

        tokens = seg_text.strip().split()
        if len(tokens) == 0:
            continue

        duration = max(1e-6, seg_end - seg_start)
        per_word = duration / len(tokens)

        for i, token in enumerate(tokens):
            clean = re.sub(r"[^\w']+", "", token.lower())
            if clean == "":
                continue

            w_start = seg_start + i * per_word
            w_end = w_start + per_word

            words.append(Word(clean, w_start, w_end))

    return words


# -----------------------------------------
# Utility for extracting words in a window
# -----------------------------------------

def filter_words(words: List[Word], t_start: float, t_end: float) -> List[Word]:
    return [w for w in words if t_start <= w.start < t_end]


# -----------------------------------------
# Speaking Rate and Articulation Rate
# -----------------------------------------

def compute_speaking_rates(words: List[Word], t_start: float, t_end: float):
    window = filter_words(words, t_start, t_end)
    num_words = len(window)
    dur = max(1e-6, t_end - t_start)

    if num_words == 0:
        return 0.0, 0.0, 0

    sr_wpm = (num_words / dur) * 60.0  # includes pauses

    if num_words < 2:
        ar_wpm = 0.0
    else:
        active_start = window[0].start
        active_end = window[-1].end
        active_dur = max(1e-6, active_end - active_start)
        ar_wpm = (num_words / active_dur) * 60.0

    return float(sr_wpm), float(ar_wpm), num_words


# -----------------------------------------
# Pause Detection
# -----------------------------------------

def compute_pause_stats(
    words: List[Word],
    t_start: float,
    t_end: float,
    threshold: float = 0.4  # tuned for lectures (natural pauses are 200-500ms)
) -> Dict[str, float]:

    window = filter_words(words, t_start, t_end)
    n = len(window)
    if n < 2:
        return {"num_pauses": 0.0, "avg_pause": 0.0, "max_pause": 0.0}

    gaps = []
    for i in range(n - 1):
        g = window[i + 1].start - window[i].end
        if g >= threshold:
            gaps.append(g)

    if not gaps:
        return {"num_pauses": 0.0, "avg_pause": 0.0, "max_pause": 0.0}

    g = np.array(gaps)
    return {
        "num_pauses": float(len(gaps)),
        "avg_pause": float(np.mean(g)),
        "max_pause": float(np.max(g)),
    }


# -----------------------------------------
# Filler Rate
# -----------------------------------------

FILLERS = {
    "um", "uh", "er", "ah", "so", "like", "actually", "basically",
    "right", "okay", "ok", "literally"
}

def compute_filler_rate(words: List[Word], t_start: float, t_end: float):
    window = filter_words(words, t_start, t_end)
    n = len(window)
    if n == 0:
        return 0.0, 0, 0

    count = sum(1 for w in window if w.text in FILLERS)
    return (count / n), count, n


# Test Example
if __name__ == "__main__":
    AUDIO = "lecture_audio.wav"
    w = transcribe_audio_to_words(AUDIO)
    print("Word count:", len(w))

    sr, ar, n = compute_speaking_rates(w, 0, 30)
    pauses = compute_pause_stats(w, 0, 30)
    fr, nf, total = compute_filler_rate(w, 0, 30)

    print("SR:", sr, "AR:", ar, "Words:", n)
    print("Pauses:", pauses)
    print(f"Filler Rate: {fr:.3f} ({nf}/{total})")
