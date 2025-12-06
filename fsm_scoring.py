# fsm_scoring.py

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


# ------------------------------
# FSM state definitions
# ------------------------------

STATE_CODES = {
    "S0": "Silence / No Speech",
    "S1": "Normal",
    "S2": "Too Fast",
    "S3": "Over-Pausing / Too Slow",
    "S4": "Too Soft",
    "S5": "Monotone Voice",
    "S6": "High Filler Usage",
    "S7": "Low Voice Clarity",
}

# ------------------------------
# Ideal ranges for features
# (applicable to most lecture videos)
# ------------------------------
IDEAL_RANGES = {
    # Speaking rate: 120–170 WPM is reasonable classroom pace
    "sr_wpm": {"min": 120.0, "max": 180.0},  # raised max from 170 to 180

    # Average RMS for normalized [-1,1] speech:
    # 0.02–0.1 is a good audible range
    "rms_mean": {"min": 0.02, "max": 0.10},

    # Pitch standard deviation (Hz) over 10 s:
    # < 10 Hz = very monotone
    # 15–80 Hz = expressive
    "f0_std": {"min": 20.0, "max": 80.0},  # raised min from 15 to 20 for better monotone detection

    # Average pause length in 10 s:
    # 0.2–0.8 s typical for lecture speech
    "avg_pause": {"min": 0.2, "max": 0.8},

    # Number of pauses per 10 s:
    # 0–3 reasonable; more may feel choppy
    "num_pauses": {"min": 0.0, "max": 3.0},

    # Filler rate: ideally < 5% of words
    "filler_rate": {"min": 0.0, "max": 0.05},

    # Clarity index: higher is clearer.
    # We set a soft minimum ideal, no upper bound.
    "clarity_mean": {"min": 1.0, "max": np.inf},
}


# ------------------------------
# Helper: deviation computation
# ------------------------------

def _deviation_high(value: float, ideal_max: float) -> float:
    """
    Deviation when 'too high' is bad.

    Returns 0 if value <= ideal_max.
    Else returns relative deviation:
        (value - ideal_max) / ideal_max
    """
    if ideal_max == np.inf:
        return 0.0
    if value <= ideal_max:
        return 0.0
    return max(0.0, (value - ideal_max) / (ideal_max + 1e-9))


def _deviation_low(value: float, ideal_min: float) -> float:
    """
    Deviation when 'too low' is bad.

    Returns 0 if value >= ideal_min.
    Else returns relative deviation:
        (ideal_min - value) / ideal_min
    """
    if ideal_min <= 0:
        return 0.0
    if value >= ideal_min:
        return 0.0
    return max(0.0, (ideal_min - value) / (ideal_min + 1e-9))


# ------------------------------
# State scoring specification
# Each entry: (feature_name, direction, weight)
# direction: 'high' = too high is bad, 'low' = too low is bad
# ------------------------------

STATE_FEATURE_WEIGHTS = {
    # S2: Too Fast - only speaking rate matters
    "S2": [
        ("sr_wpm", "high", 5.0),   # ONLY speaking rate; much more weight
    ],

    # S3: Over-Pausing / Too Slow
    "S3": [
        ("sr_wpm", "low", 2.0),      # speaking rate too low
        ("avg_pause", "high", 3.0),  # pauses very long
        ("num_pauses", "high", 2.0), # many pauses
    ],

    # S4: Too Soft
    "S4": [
        ("rms_mean", "low", 3.0),
    ],

    # S5: Monotone Voice
    "S5": [
        ("f0_std", "low", 2.5),
        # optional: low voiced_ratio could also be bad,
        # but we keep it simple here.
    ],

    # S6: High Filler Usage
    "S6": [
        ("filler_rate", "high", 2.5),
    ],

    # S7: Low Clarity (optional)
    "S7": [
        ("clarity_mean", "low", 2.0),
    ],
}

# State priority weights (higher = more important to flag)
# Used to break ties when multiple states have similar scores
STATE_PRIORITY = {
    "S4": 1.2,  # Too Soft - critical for comprehension
    "S7": 1.1,  # Low Clarity - affects understanding
    "S2": 1.0,  # Too Fast - pacing issue
    "S3": 0.9,  # Over-Pausing - less critical
    "S5": 0.8,  # Monotone - engagement issue
    "S6": 0.7,  # Fillers - least critical
}

# Threshold: if all state scores < this, call it Normal (S1)
# Raised from 0.15 to 0.25 to require stronger deviation for problem states
NORMALITY_THRESHOLD = 0.25

# Silence detection thresholds (chunk-level)
SILENCE_RMS_THRESHOLD = 0.005     # same as frame-based, but averaged
SILENCE_WORD_COUNT_THRESHOLD = 1  # no or 1 word in 10 s ≈ silence/idle


# ------------------------------
# Scoring for one chunk
# ------------------------------

def score_states_for_chunk(chunk_feat: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """
    Computes weighted scores for each non-normal state (S2..S7)
    for a given chunk, and returns the winning state code.

    Silence (S0) and Normal (S1) are handled specially:
    - First check if chunk is silence / idle -> S0
    - Else compute scores for S2..S7
      - If all scores < NORMALITY_THRESHOLD -> S1 (Normal)
      - Else state with highest score wins.

    Parameters
    ----------
    chunk_feat : Dict[str, float]
        Feature dictionary for a chunk created by build_chunk_feature_list().
        Expected keys:
            'rms_mean', 'sr_wpm', 'avg_pause', 'num_pauses',
            'f0_std', 'filler_rate', 'clarity_mean', 'num_words', ...

    Returns
    -------
    best_state : str
        State code, e.g. "S0", "S1", "S2", ...
    state_scores : Dict[str, float]
        Scores for each state that was evaluated (S2..S7).
        S0 and S1 are not scored; they are decision labels.
    """
    # 1) Check for silence / no-speech chunk
    rms_mean = float(chunk_feat.get("rms_mean", 0.0))
    num_words = float(chunk_feat.get("num_words", 0.0))

    if (rms_mean < SILENCE_RMS_THRESHOLD) and (num_words <= SILENCE_WORD_COUNT_THRESHOLD):
        # Very low energy and almost no words -> Silence / idle
        return "S0", {}

    # 2) Compute scores for each non-normal state S2..S7
    state_scores: Dict[str, float] = {}

    for state_code, fw_list in STATE_FEATURE_WEIGHTS.items():
        score = 0.0
        for (feat_name, direction, weight) in fw_list:
            # Read feature value; if missing, treat as 0 (no contribution)
            value = float(chunk_feat.get(feat_name, 0.0))

            ideal = IDEAL_RANGES.get(feat_name, None)
            if ideal is None:
                continue

            if direction == "high":
                dev = _deviation_high(value, ideal["max"])
            elif direction == "low":
                dev = _deviation_low(value, ideal["min"])
            else:
                raise ValueError(f"Unknown direction: {direction}")

            score += weight * dev

        state_scores[state_code] = score

    # 3) Choose best state based on scores weighted by priority
    # If all scores are very small, mark as Normal (S1)
    def weighted_score(state_code: str) -> float:
        base = state_scores.get(state_code, 0.0)
        priority = STATE_PRIORITY.get(state_code, 1.0)
        return base * priority
    
    best_state = max(state_scores, key=weighted_score)
    best_score = state_scores[best_state]

    if best_score < NORMALITY_THRESHOLD:
        return "S1", state_scores  # Normal

    return best_state, state_scores


# ------------------------------
# Apply to all chunks
# ------------------------------

def assign_states_to_all_chunks(
    chunk_feature_list: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    """
    Takes a list of chunk feature dictionaries and assigns an FSM state
    to each chunk.

    Parameters
    ----------
    chunk_feature_list : List[Dict[str, float]]
        Output of build_chunk_feature_list().

    Returns
    -------
    annotated_chunks : List[Dict[str, float]]
        Each dict is original features plus:
            'state_code' : e.g. "S2"
            'state_name' : e.g. "Too Fast"
            'state_scores' : dict of scores for S2..S7
    """
    annotated = []
    for feat in chunk_feature_list:
        state_code, state_scores = score_states_for_chunk(feat)
        feat_out = dict(feat)  # copy
        feat_out["state_code"] = state_code
        feat_out["state_name"] = STATE_CODES[state_code]
        feat_out["state_scores"] = state_scores
        annotated.append(feat_out)

    return annotated


if __name__ == "__main__":
    # Simple mock example
    sample_chunk = {
        "t_start": 0.0,
        "t_end": 10.0,
        "rms_mean": 0.015,     # a bit soft
        "zcr_mean": 0.05,
        "clarity_mean": 1.2,
        "f0_mean": 150.0,
        "f0_std": 8.0,         # low variation -> monotone
        "voiced_ratio": 0.9,
        "sr_wpm": 200.0,       # too fast
        "ar_wpm": 210.0,
        "num_words": 35.0,
        "num_pauses": 1.0,
        "avg_pause": 0.3,
        "max_pause": 0.5,
        "filler_rate": 0.02,
        "num_fillers": 1.0,
    }

    state_code, scores = score_states_for_chunk(sample_chunk)
    print("Chunk state:", state_code, "-", STATE_CODES[state_code])
    print("Scores:", scores)
