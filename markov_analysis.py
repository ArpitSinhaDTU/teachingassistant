# markov_analysis.py

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

from fsm_scoring import STATE_CODES


STATE_LIST = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}


def extract_state_sequence(
    annotated_chunks: List[Dict[str, float]]
) -> List[str]:
    """
    Extracts the state_code sequence from the annotated chunk list.

    Parameters
    ----------
    annotated_chunks : List[Dict[str, float]]
        Output of assign_states_to_all_chunks(). Each dict must contain 'state_code'.

    Returns
    -------
    state_seq : List[str]
        List of state codes in time order, e.g. ["S1", "S2", "S2", "S3", ...]
    """
    return [c.get("state_code", "S1") for c in annotated_chunks]


def compute_transition_counts(
    state_seq: List[str]
) -> np.ndarray:
    """
    Builds a state transition count matrix from a state sequence.

    Parameters
    ----------
    state_seq : List[str]
        Sequence of state codes, e.g. ["S1", "S2", "S2", "S3", "S1", ...]

    Returns
    -------
    counts : np.ndarray
        2D array of shape (num_states, num_states),
        counts[i, j] = number of transitions from state i to state j.
    """
    num_states = len(STATE_LIST)
    counts = np.zeros((num_states, num_states), dtype=np.float64)

    # We count transitions from state_seq[k] -> state_seq[k+1]
    for i in range(len(state_seq) - 1):
        s_from = state_seq[i]
        s_to = state_seq[i + 1]

        if (s_from not in STATE_INDEX) or (s_to not in STATE_INDEX):
            continue

        i_from = STATE_INDEX[s_from]
        i_to = STATE_INDEX[s_to]
        counts[i_from, i_to] += 1.0

    return counts


def normalize_transition_matrix(
    counts: np.ndarray,
    smoothing: float = 0.0
) -> np.ndarray:
    """
    Converts a transition count matrix into a transition probability matrix.

    Parameters
    ----------
    counts : np.ndarray
        2D array with transition counts.
    smoothing : float
        Additive smoothing value for each transition (e.g. 0.1),
        to avoid zero-probability transitions if desired.
        Default 0.0 = no smoothing.

    Returns
    -------
    P : np.ndarray
        2D array of transition probabilities, same shape as counts.
        Each row sums to 1.0 (if row had at least one transition),
        otherwise it is left as all zeros.
    """
    counts = counts.astype(np.float64)
    num_states = counts.shape[0]

    if smoothing > 0.0:
        counts = counts + smoothing

    P = np.zeros_like(counts, dtype=np.float64)

    for i in range(num_states):
        row_sum = np.sum(counts[i, :])
        if row_sum > 0:
            P[i, :] = counts[i, :] / row_sum
        else:
            P[i, :] = 0.0

    return P


def compute_state_visit_distribution(
    state_seq: List[str]
) -> Dict[str, float]:
    """
    Computes fraction of time (chunks) spent in each state.

    Parameters
    ----------
    state_seq : List[str]
        Sequence of state codes in time order.

    Returns
    -------
    visit_dist : Dict[str, float]
        Mapping from state code to fraction (0..1).
    """
    n = len(state_seq)
    if n == 0:
        return {s: 0.0 for s in STATE_LIST}

    counts = {s: 0 for s in STATE_LIST}
    for s in state_seq:
        if s in counts:
            counts[s] += 1

    return {s: counts[s] / float(n) for s in STATE_LIST}


def summarize_markov_behavior(
    P: np.ndarray,
    visit_dist: Dict[str, float],
    min_transition_prob: float = 0.2
) -> Dict[str, List[str]]:
    """
    Produces a simple textual / structural summary of the Markov behavior:
    - Which states are most frequent
    - Which high-probability transitions exist (above min_transition_prob)

    This is not strictly necessary for the math, but is useful for
    human-readable analytics.

    Parameters
    ----------
    P : np.ndarray
        Transition probability matrix (num_states x num_states).
    visit_dist : Dict[str, float]
        Fraction of time spent in each state.
    min_transition_prob : float
        Threshold to list a transition as 'significant'.
        Default: 0.2 (20%).

    Returns
    -------
    summary : Dict[str, List[str]]
        {
          "dominant_states": [ ... human-readable lines ... ],
          "significant_transitions": [ ... lines ... ]
        }
    """
    summary = {
        "dominant_states": [],
        "significant_transitions": [],
    }

    # 1) Dominant states (sorted by visit fraction)
    sorted_states = sorted(
        visit_dist.items(),
        key=lambda kv: kv[1],
        reverse=True
    )

    for s_code, frac in sorted_states:
        if frac <= 0.0:
            continue
        label = STATE_CODES.get(s_code, s_code)
        summary["dominant_states"].append(
            f"{s_code} ({label}): {frac * 100:.1f}% of time"
        )

    # 2) Significant transitions
    num_states = len(STATE_LIST)
    for i in range(num_states):
        for j in range(num_states):
            p = P[i, j]
            if p >= min_transition_prob:
                s_from = STATE_LIST[i]
                s_to = STATE_LIST[j]
                label_from = STATE_CODES.get(s_from, s_from)
                label_to = STATE_CODES.get(s_to, s_to)
                summary["significant_transitions"].append(
                    f"{s_from} ({label_from}) -> {s_to} ({label_to}): {p * 100:.1f}%"
                )

    return summary


if __name__ == "__main__":
    # Mock example: fake sequence of states over 12 chunks
    mock_seq = ["S1", "S2", "S2", "S3", "S1", "S1", "S4", "S4", "S1", "S5", "S5", "S1"]

    counts = compute_transition_counts(mock_seq)
    P = normalize_transition_matrix(counts, smoothing=0.0)
    visit_dist = compute_state_visit_distribution(mock_seq)
    summary = summarize_markov_behavior(P, visit_dist, min_transition_prob=0.2)

    print("Transition counts:\n", counts)
    print("\nTransition probability matrix P:\n", np.round(P, 3))
    print("\nVisit distribution:")
    for line in summary["dominant_states"]:
        print("  ", line)

    print("\nSignificant transitions (>= 20%):")
    for line in summary["significant_transitions"]:
        print("  ", line)
