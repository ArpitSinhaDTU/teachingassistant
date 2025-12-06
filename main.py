# main.py

from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional
import numpy as np

# ---- Our modules ----
from audio_extraction import extract_audio_from_video
from framing import load_audio_wav, frame_signal
from rms_energy import compute_frame_energy_rms
from zcr import compute_zcr
from pitch_autocorr import compute_pitch_autocorr
from cepstrum_clarity import compute_clarity_over_time
from whisper_groq_features import transcribe_audio_to_words
from video_features import compute_global_motion
from chunk_features import build_chunk_feature_list
from fsm_scoring import assign_states_to_all_chunks, STATE_CODES
from markov_analysis import (
    extract_state_sequence,
    compute_transition_counts,
    normalize_transition_matrix,
    compute_state_visit_distribution,
    summarize_markov_behavior,
)

# For video rendering with text overlays
from moviepy import VideoFileClip, TextClip, CompositeVideoClip


# -----------------------------
# CONFIG & INPUT PLACEHOLDERS
# -----------------------------

# 👉👉 CHANGE THESE PATHS FOR YOUR OWN RUNS
VIDEO_PATH = "samplelec.mp4"          # Input class video
AUDIO_PATH_OPTIONAL = None          # If None, will be extracted from VIDEO_PATH

# Chunk & recommendation settings
CHUNK_SIZE_SEC = 10.0               # analysis chunk
RECOMMENDATION_INTERVAL_SEC = 30.0  # one suggestion per minute

# Output paths
EXTRACTED_AUDIO_PATH = "lecture_audio.wav"
OUTPUT_VIDEO_WITH_TIPS = "output_video_with_tips.mp4"
ANALYTICS_REPORT_PATH = "analytics_report.txt"


# -----------------------------
# Recommendation Text Logic
# -----------------------------

def recommendation_for_state(state_code: str) -> str:
    """
    Map each non-normal state to a SHORT human-readable suggestion.
    Used for subtitles on the video - must fit on one line!
    """
    if state_code == "S2":  # Too Fast
        return "Slow down slightly. Add brief pauses."
    if state_code == "S3":  # Over-Pausing
        return "Reduce pauses. Keep a smoother flow."
    if state_code == "S4":  # Too Soft
        return "Speak louder or check your mic."
    if state_code == "S5":  # Monotone
        return "Add pitch variation for engagement."
    if state_code == "S6":  # High Fillers
        return "Reduce um/uh. Use silent pauses instead."
    if state_code == "S7":  # Low Clarity
        return "Improve clarity. Check room noise/mic."

    # For Normal or Silence
    if state_code == "S1":
        return "Good pace and clarity!"
    if state_code == "S0":
        return "No speech detected."

    # Fallback
    return "Keep teaching!"


def build_minute_recommendations(
    annotated_chunks: List[Dict[str, float]],
    total_duration: float,
    interval_sec: float = 60.0
) -> List[Dict[str, object]]:
    """
    Build a list of recommendation intervals (1 per minute) based on chunk states.

    Each recommendation block:
        {
          "start": float,
          "end": float,
          "state_code": str,
          "message": str
        }

    Logic:
    - For each [t, t+interval_sec), look at chunks in that window.
    - Count how many times each non-normal state appears.
    - If at least one problem state appears, choose the most frequent.
    - Otherwise, choose Normal (S1).
    """
    recs = []
    t = 0.0
    while t < total_duration:
        t_start = t
        t_end = min(t + interval_sec, total_duration)
        t = t_end

        # Filter chunks whose center is in this minute
        # Approximate chunk center as (t_start + t_end)/2 from chunk dict
        # but we stored chunk's own t_start & t_end
        window_states = []
        for ch in annotated_chunks:
            ct_start = float(ch["t_start"])
            ct_end = float(ch["t_end"])
            c_center = 0.5 * (ct_start + ct_end)
            if c_center >= t_start and c_center < t_end:
                window_states.append(ch["state_code"])

        if not window_states:
            recs.append({
                "start": t_start,
                "end": t_end,
                "state_code": "S0",
                "message": recommendation_for_state("S0")
            })
            continue

        # Count only problem states (S2..S7)
        total_chunks = len(window_states)
        counts = {}
        for s in window_states:
            if s in ["S0", "S1"]:
                continue
            counts[s] = counts.get(s, 0) + 1

        if counts:
            # find most frequent problem state and its fraction in this minute
            best_state = max(counts, key=counts.get)
            best_count = counts[best_state]
            frac = best_count / total_chunks

            # if problem state not dominant enough, treat this minute as Normal
            if frac < 0.4:   # < 40% of chunks problematic -> don't nag
                best_state = "S1"
        else:
            best_state = "S1"

        recs.append({
            "start": t_start,
            "end": t_end,
            "state_code": best_state,
            "message": recommendation_for_state(best_state)
        })

    return recs


# -----------------------------
# Video Overlay with Text
# -----------------------------

def create_video_with_recommendations(
    input_video_path: str,
    output_video_path: str,
    recommendations: List[Dict[str, object]],
    font_size: int = 36,
    text_color: str = "white",
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    position: Tuple[str, str] = ("center", "bottom"),
):
    """
    Create a video with overlayed text recommendations (like subtitles).

    Each recommendation covers a time interval and shows a fixed suggestion.
    """
    base_clip = VideoFileClip(input_video_path)
    txt_clips = []

    for rec in recommendations:
        txt = rec["message"]
        t_start = float(rec["start"])
        t_end = float(rec["end"])
        duration = max(0.1, t_end - t_start)

        # Create text clip with smaller font to fit on screen
        txt_clip = TextClip(
            font="C:/Windows/Fonts/arial.ttf",
            text=txt,
            font_size=24,  # Small font to ensure text fits
            color=text_color,
            bg_color="black",
        )
        
        # Position at bottom center of video
        txt_clip = txt_clip.with_position(("center", "bottom"))
        txt_clip = txt_clip.with_start(t_start)
        txt_clip = txt_clip.with_duration(duration)

        txt_clips.append(txt_clip)

    final = CompositeVideoClip([base_clip] + txt_clips)
    final.write_videofile(
        output_video_path,
        codec="libx264",
        audio_codec="aac",
        fps=base_clip.fps
    )

    base_clip.close()   
    final.close()


# -----------------------------
# Analytics Report Generation
# -----------------------------

def generate_analytics_report(
    annotated_chunks: List[Dict[str, float]],
    transition_matrix: np.ndarray,
    visit_dist: Dict[str, float],
    markov_summary: Dict[str, List[str]],
) -> str:
    """
    Creates a professional, strict teaching analytics report with:
      1. Executive summary
      2. Key quantitative metrics
      3. Behaviour over time (Markov + state usage)
      4. Critical improvement recommendations
    """

    # --------- Aggregate global metrics ----------
    def avg_feat(name: str) -> float:
        vals = [ch.get(name, 0.0) for ch in annotated_chunks]
        return float(np.mean(vals)) if vals else 0.0

    avg_sr = avg_feat("sr_wpm")
    avg_ar = avg_feat("ar_wpm")
    avg_filler = avg_feat("filler_rate")
    avg_clarity = avg_feat("clarity_mean")
    avg_motion = avg_feat("motion_mean")
    avg_f0std = avg_feat("f0_std")

    # State usage
    p_s0 = visit_dist.get("S0", 0.0)
    p_s1 = visit_dist.get("S1", 0.0)
    p_s2 = visit_dist.get("S2", 0.0)
    p_s3 = visit_dist.get("S3", 0.0)
    p_s4 = visit_dist.get("S4", 0.0)
    p_s5 = visit_dist.get("S5", 0.0)
    p_s6 = visit_dist.get("S6", 0.0)
    p_s7 = visit_dist.get("S7", 0.0)

    # --------- Construct high-level ratings (1–10) ----------

    # Pacing score: penalise time in Too Fast (S2) + Over-Pausing (S3)
    pace_penalty = 0.6 * p_s2 + 0.4 * p_s3   # S2 slightly more serious
    pace_score = max(1.0, 10.0 * (1.0 - pace_penalty))

    # Clarity score: based on clarity index + time in S7
    clarity_penalty = 0.7 * p_s7
    clarity_score = max(1.0, 10.0 * (1.0 - clarity_penalty))

    # Filler score: based on mean filler rate + time in S6
    filler_penalty = min(1.0, avg_filler / 0.08) * 0.5 + 0.5 * p_s6
    filler_score = max(1.0, 10.0 * (1.0 - filler_penalty))

    # Vocal expressiveness: based on pitch std + time in S5
    if avg_f0std < 10:
        expr_base = 2.5
    elif avg_f0std < 20:
        expr_base = 5.0
    elif avg_f0std < 40:
        expr_base = 7.5
    else:
        expr_base = 9.0
    expr_penalty = 0.6 * p_s5
    expressiveness_score = max(1.0, expr_base * (1.0 - expr_penalty))

    # Motion score: penalise very low and very high motion
    # (0.005 ~ very static, >0.03 ~ possibly distracting)
    if avg_motion < 0.003:
        motion_score = 4.0
        motion_comment = "physical presence is too static and may reduce engagement."
    elif avg_motion < 0.015:
        motion_score = 7.0
        motion_comment = "physical movement is moderate and generally acceptable."
    elif avg_motion < 0.035:
        motion_score = 8.0
        motion_comment = "physical movement is active and likely supportive of engagement."
    else:
        motion_score = 5.0
        motion_comment = "physical movement appears excessive and may distract students."

    # Overall score: weighted combination (strict)
    overall_score = (
        0.35 * pace_score
        + 0.20 * clarity_score
        + 0.15 * filler_score
        + 0.15 * expressiveness_score
        + 0.15 * motion_score
    )

    # --------- EXECUTIVE SUMMARY (strict, professional) ----------

    summary_lines = []
    summary_lines.append("=== TEACHING ANALYTICS REPORT (AUTOMATED) ===\n")

    summary_lines.append("1. Executive Summary")
    summary_lines.append("---------------------")

    summary_lines.append(
        f"Overall Delivery Score (1–10): {overall_score:.1f}"
    )
    summary_lines.append(
        f"Pacing: {pace_score:.1f} | Clarity: {clarity_score:.1f} | "
        f"Filler Management: {filler_score:.1f} | "
        f"Vocal Expressiveness: {expressiveness_score:.1f} | "
        f"Physical Engagement: {motion_score:.1f}"
    )

    # Construct a strict narrative based on scores
    if pace_score < 6.0:
        summary_lines.append(
            "- Pacing is a critical concern: the lecture exhibits substantial deviation "
            "from an optimal balance between flow and silence."
        )
    else:
        summary_lines.append(
            "- Pacing is broadly acceptable, although local deviations still appear and "
            "should be reviewed for improvement."
        )

    if clarity_score < 7.0:
        summary_lines.append(
            "- Vocal clarity shows noticeable issues in parts of the lecture and should "
            "be addressed to avoid listener fatigue."
        )
    else:
        summary_lines.append(
            "- Vocal clarity is generally strong, with only minor degradation in difficult segments."
        )

    if filler_score < 7.0:
        summary_lines.append(
            "- Filler usage is high enough to be distracting and reduces perceived confidence."
        )
    else:
        summary_lines.append(
            "- Filler usage is reasonably controlled and not a major weakness."
        )

    if expressiveness_score < 7.0:
        summary_lines.append(
            "- Vocal expressiveness is limited; sections of the lecture may sound monotonous."
        )
    else:
        summary_lines.append(
            "- Vocal expressiveness is adequate and helps maintain attention."
        )

    summary_lines.append(
        f"- Regarding body language, {motion_comment}"
    )

    # --------- KEY METRICS TABLE ----------

    summary_lines.append("\n2. Key Quantitative Metrics")
    summary_lines.append("---------------------------")
    summary_lines.append(
        f"  • Average speaking rate (SR): {avg_sr:.1f} words/minute"
        f"  (articulation rate: {avg_ar:.1f} words/minute)"
    )
    summary_lines.append(
        f"  • Mean clarity index (higher is better): {avg_clarity:.2f}"
    )
    summary_lines.append(
        f"  • Filler rate: {avg_filler * 100:.1f}% of all words"
    )
    summary_lines.append(
        f"  • Mean pitch standard deviation: {avg_f0std:.1f} Hz "
        f"(indicator of vocal variety)"
    )
    summary_lines.append(
        f"  • Mean motion index: {avg_motion:.4f} "
        f"(global frame-difference based movement)"
    )

    summary_lines.append("\n  State usage (time spent in each behaviour):")
    for s_code, frac in visit_dist.items():
        if frac <= 0.0:
            continue
        label = STATE_CODES[s_code]
        summary_lines.append(f"    - {s_code} ({label}): {frac * 100:.1f}%")

    # --------- BEHAVIOUR OVER TIME / MARKOV ANALYSIS ----------

    summary_lines.append("\n3. Behavioural Dynamics Over Time")
    summary_lines.append("---------------------------------")

    summary_lines.append("  Dominant observed states:")
    for line in markov_summary["dominant_states"]:
        summary_lines.append("    " + line)

    summary_lines.append("\n  High-probability state transitions:")
    if markov_summary["significant_transitions"]:
        for line in markov_summary["significant_transitions"]:
            summary_lines.append("    " + line)
    else:
        summary_lines.append(
            "    No transitions exceeded the configured significance threshold."
        )

    summary_lines.append(
        "\n  Interpretation:\n"
        "  - Recurrent self-transitions (e.g., S2→S2 or S5→S5) indicate stable habits that are "
        "unlikely to change spontaneously.\n"
        "  - Frequent transitions from a normal state (S1) into problematic states (e.g., S2, S3) "
        "show where pacing or clarity tends to break down.\n"
        "  - If the Markov structure remains unchanged, future lectures will statistically "
        "replicate the same behaviour patterns."
    )

    # Note: Detailed recommendations are now in Section 6 "Detailed Future Improvement Plan"
    # which appears after the Feature Glossary for better report flow

    # --------- FEATURE GLOSSARY ----------
    
    # Compute additional stats for the glossary
    avg_rms = avg_feat("rms_mean")
    avg_zcr = avg_feat("zcr_mean")
    avg_f0_mean = avg_feat("f0_mean")
    avg_voiced_ratio = avg_feat("voiced_ratio")
    avg_num_words = avg_feat("num_words")
    avg_num_pauses = avg_feat("num_pauses")
    avg_pause_dur = avg_feat("avg_pause")
    avg_max_pause = avg_feat("max_pause")
    avg_num_fillers = avg_feat("num_fillers")
    
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("4. Feature Glossary & Detailed Metrics")
    summary_lines.append("=" * 60)
    
    summary_lines.append("\nThis section explains every feature extracted and analyzed.\n")
    
    # --- Audio Frame-Level Features ---
    summary_lines.append("─" * 40)
    summary_lines.append("A. AUDIO FRAME-LEVEL FEATURES")
    summary_lines.append("─" * 40)
    
    summary_lines.append(f"""
  1. RMS Energy (Root Mean Square)
     ├─ What: Measures the average loudness/volume of the audio signal
     ├─ How: Computed per 25ms audio frame as sqrt(mean(samples²))
     ├─ Ideal Range: 0.02 - 0.10 (for normalized [-1,1] audio)
     ├─ Your Average: {avg_rms:.4f}
     └─ Interpretation: Lower values indicate soft speech; very low (<0.005) = silence

  2. ZCR (Zero Crossing Rate)
     ├─ What: Frequency at which the audio signal crosses zero amplitude
     ├─ How: Count of sign changes per frame, normalized to [0,1]
     ├─ Your Average: {avg_zcr:.4f}
     └─ Interpretation: High ZCR = noise/fricatives; Low ZCR = voiced speech (vowels)

  3. Pitch / F0 (Fundamental Frequency)
     ├─ What: The perceived tone of voice, measured in Hz
     ├─ How: Autocorrelation-based pitch detection per frame
     ├─ Typical Range: 85-180 Hz (male), 165-255 Hz (female)
     ├─ Your Mean F0: {avg_f0_mean:.1f} Hz
     ├─ Your F0 Std Dev: {avg_f0std:.1f} Hz (expressiveness indicator)
     └─ Interpretation: Low std dev (<15 Hz) = monotone; High (>30 Hz) = expressive

  4. Voiced Ratio
     ├─ What: Fraction of frames with detectable pitch (voiced speech)
     ├─ How: Percentage of frames where F0 > 40 Hz
     ├─ Your Average: {avg_voiced_ratio * 100:.1f}%
     └─ Interpretation: Higher = more continuous speech; Lower = more silence/pauses

  5. Clarity Index (Cepstrum-based)
     ├─ What: Measures voice clarity vs reverberation/noise
     ├─ How: Ratio of low-quefrency to high-quefrency cepstral energy
     ├─ Ideal: > 1.0 (higher is clearer)
     ├─ Your Average: {avg_clarity:.2f}
     └─ Interpretation: Low values suggest echo, room noise, or poor mic placement
""")

    # --- Speech/Transcription Features ---
    summary_lines.append("─" * 40)
    summary_lines.append("B. SPEECH & TRANSCRIPTION FEATURES")
    summary_lines.append("─" * 40)
    
    summary_lines.append(f"""
  6. Speaking Rate (SR)
     ├─ What: Words per minute including pauses
     ├─ How: (word count / total duration) × 60
     ├─ Ideal Range: 120-180 WPM for lectures
     ├─ Your Average: {avg_sr:.1f} WPM
     └─ Interpretation: >180 = too fast; <100 = too slow for typical lectures

  7. Articulation Rate (AR)
     ├─ What: Words per minute during active speech only (excludes pauses)
     ├─ How: (word count / active speech duration) × 60
     ├─ Your Average: {avg_ar:.1f} WPM
     └─ Interpretation: Typically 20-50 WPM higher than SR

  8. Words Per Chunk
     ├─ What: Number of words spoken per 10-second analysis window
     ├─ Your Average: {avg_num_words:.1f} words/chunk
     └─ Interpretation: ~20-30 words per 10s is typical for lectures

  9. Pause Statistics
     ├─ What: Inter-word gaps > 400ms are counted as pauses
     ├─ Your Avg Pauses/Chunk: {avg_num_pauses:.1f}
     ├─ Your Avg Pause Duration: {avg_pause_dur:.2f} seconds
     ├─ Your Avg Max Pause: {avg_max_pause:.2f} seconds
     └─ Interpretation: 2-4 pauses per 10s is natural; >0.8s avg = over-pausing

  10. Filler Words
      ├─ What: Common hesitation words (um, uh, like, so, actually, etc.)
      ├─ Tracked Fillers: um, uh, er, ah, so, like, actually, basically, right, okay, literally
      ├─ Your Filler Rate: {avg_filler * 100:.1f}%
      ├─ Your Avg Fillers/Chunk: {avg_num_fillers:.1f}
      └─ Interpretation: <5% is professional; >8% is noticeable to audience
""")

    # --- Video Features ---
    summary_lines.append("─" * 40)
    summary_lines.append("C. VIDEO MOTION FEATURES")
    summary_lines.append("─" * 40)
    
    summary_lines.append(f"""
  11. Global Motion Index
      ├─ What: Overall physical movement detected in video frames
      ├─ How: Mean absolute frame difference (grayscale), normalized to [0,1]
      ├─ Your Average: {avg_motion:.4f}
      ├─ Interpretation Scale:
      │   < 0.003 = Very static (may lack engagement)
      │   0.003-0.015 = Moderate movement (acceptable)
      │   0.015-0.035 = Active movement (good engagement)
      │   > 0.035 = Excessive (potentially distracting)
      └─ Your Assessment: {motion_comment}
""")

    # --- FSM States ---
    summary_lines.append("─" * 40)
    summary_lines.append("D. FSM BEHAVIORAL STATES")
    summary_lines.append("─" * 40)
    
    summary_lines.append("""
  The system classifies each 10-second chunk into one of these states:

  ┌───────┬─────────────────────┬─────────────────────────────────────────┐
  │ State │ Name                │ Trigger Condition                       │
  ├───────┼─────────────────────┼─────────────────────────────────────────┤
  │  S0   │ Silence             │ RMS < 0.005 and < 2 words detected      │
  │  S1   │ Normal              │ All metrics within ideal ranges         │
  │  S2   │ Too Fast            │ Speaking rate > 180 WPM                 │
  │  S3   │ Over-Pausing        │ Many/long pauses, speaking rate < 120   │
  │  S4   │ Too Soft            │ RMS < 0.02 (low volume)                 │
  │  S5   │ Monotone            │ Pitch std dev < 20 Hz                   │
  │  S6   │ High Fillers        │ Filler rate > 5% of words               │
  │  S7   │ Low Clarity         │ Clarity index < 1.0                     │
  └───────┴─────────────────────┴─────────────────────────────────────────┘
""")

    # --- Your State Distribution ---
    summary_lines.append("  Your State Distribution:")
    for s_code in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        frac = visit_dist.get(s_code, 0.0)
        label = STATE_CODES[s_code]
        bar_len = int(frac * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        summary_lines.append(f"    {s_code} ({label:20s}): [{bar}] {frac * 100:5.1f}%")

    # --------- DETAILED FUTURE IMPROVEMENTS ----------

    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("5. Detailed Future Improvement Plan")
    summary_lines.append("=" * 60)

    # Determine priority areas based on scores
    issues = []
    if pace_score < 7.0:
        issues.append(("Pacing", pace_score, "HIGH" if pace_score < 5 else "MEDIUM"))
    if clarity_score < 7.0:
        issues.append(("Clarity", clarity_score, "HIGH" if clarity_score < 5 else "MEDIUM"))
    if filler_score < 7.0:
        issues.append(("Filler Control", filler_score, "HIGH" if filler_score < 5 else "MEDIUM"))
    if expressiveness_score < 7.0:
        issues.append(("Expressiveness", expressiveness_score, "HIGH" if expressiveness_score < 5 else "MEDIUM"))
    if motion_score < 6.0:
        issues.append(("Physical Presence", motion_score, "MEDIUM"))

    summary_lines.append("\n┌─────────────────────────────────────────────────────────┐")
    summary_lines.append("│              PRIORITIZED ACTION ITEMS                    │")
    summary_lines.append("└─────────────────────────────────────────────────────────┘")

    if not issues:
        summary_lines.append("\n  ✓ No critical issues detected. Focus on refinement.\n")
    else:
        issues.sort(key=lambda x: x[1])  # Sort by score (lowest first)
        summary_lines.append(f"\n  Identified {len(issues)} area(s) requiring improvement:\n")
        for i, (area, score, priority) in enumerate(issues, 1):
            summary_lines.append(f"  {i}. [{priority}] {area} (Score: {score:.1f}/10)")

    # --- Pacing Improvement Plan ---
    summary_lines.append("\n─" * 40)
    summary_lines.append("A. PACING IMPROVEMENT STRATEGIES")
    summary_lines.append("─" * 40)

    if p_s2 > 0.15:  # Too fast
        summary_lines.append(f"""
  Problem: Speaking too fast ({p_s2 * 100:.1f}% of lecture in S2 state)
  Your speaking rate: {avg_sr:.1f} WPM (Ideal: 120-180 WPM)

  IMMEDIATE ACTIONS:
  ├─ 1. Add 1-2 second pauses after each key concept or definition
  ├─ 2. Practice "chunking" - break sentences into 5-7 word phrases
  ├─ 3. Use transitional phrases ("Now, moving on to...", "The key point here is...")
  └─ 4. Write "PAUSE" notes in your lecture slides at critical points

  WEEKLY PRACTICE:
  ├─ Record yourself reading a 2-minute passage at 140 WPM target
  ├─ Use a metronome app set to ~2.5 words per beat for practice
  └─ Review recordings and count pauses - aim for 3-4 per minute

  LONG-TERM GOAL: Reduce S2 state to <10% while maintaining engagement
""")
    elif p_s3 > 0.15:  # Over-pausing
        summary_lines.append(f"""
  Problem: Too many or too long pauses ({p_s3 * 100:.1f}% of lecture in S3 state)
  Your avg pause duration: {avg_pause_dur:.2f}s (Ideal: 0.3-0.6s)

  IMMEDIATE ACTIONS:
  ├─ 1. Prepare talking points more thoroughly to reduce hesitation
  ├─ 2. Practice transitions between topics until they feel fluid
  ├─ 3. Replace long pauses with brief filler-free transitions
  └─ 4. Avoid reading directly from notes - use bullet points instead

  WEEKLY PRACTICE:
  ├─ Rehearse your entire lecture out loud 2-3 times before delivery
  ├─ Time yourself and identify where pauses occur
  └─ Create "bridge sentences" for each topic transition

  LONG-TERM GOAL: Keep pauses under 0.8s on average; reduce S3 to <10%
""")
    else:
        summary_lines.append(f"""
  Status: Pacing is acceptable (S2: {p_s2 * 100:.1f}%, S3: {p_s3 * 100:.1f}%)

  REFINEMENT SUGGESTIONS:
  ├─ 1. Vary pace intentionally - slower for complex ideas, faster for reviews
  ├─ 2. Use strategic pauses before important statements for emphasis
  └─ 3. Monitor audience engagement and adjust in real-time

  MAINTENANCE: Continue current pacing; fine-tune based on feedback
""")

    # --- Clarity Improvement Plan ---
    summary_lines.append("─" * 40)
    summary_lines.append("B. VOICE CLARITY IMPROVEMENT STRATEGIES")
    summary_lines.append("─" * 40)

    if p_s7 > 0.10 or avg_clarity < 1.0:
        summary_lines.append(f"""
  Problem: Low voice clarity ({p_s7 * 100:.1f}% in S7 state)
  Your clarity index: {avg_clarity:.2f} (Ideal: >1.5)

  IMMEDIATE ACTIONS:
  ├─ 1. Position microphone 4-6 inches from mouth at 45° angle
  ├─ 2. Reduce room echo - use soft furnishings or acoustic panels
  ├─ 3. Close windows and doors; turn off fans/AC during recording
  └─ 4. Articulate consonants more deliberately (especially T, D, K, P)

  EQUIPMENT UPGRADES:
  ├─ Consider a lapel mic for consistent distance
  ├─ Use a pop filter to reduce plosives
  └─ Test room acoustics with the "clap test" - echo should fade in <0.5s

  WEEKLY PRACTICE:
  ├─ Read tongue twisters daily for 5 minutes
  ├─ Record short clips and compare clarity to target speakers
  └─ Practice projecting voice without shouting

  LONG-TERM GOAL: Achieve clarity index >1.5; reduce S7 to <5%
""")
    else:
        summary_lines.append(f"""
  Status: Clarity is good (Index: {avg_clarity:.2f}, S7: {p_s7 * 100:.1f}%)

  REFINEMENT SUGGESTIONS:
  ├─ 1. Maintain current microphone setup and room conditions
  ├─ 2. Stay hydrated during lectures to maintain vocal quality
  └─ 3. Consider slight enunciation emphasis for technical terms

  MAINTENANCE: No major changes needed; monitor for degradation
""")

    # --- Filler Words Improvement Plan ---
    summary_lines.append("─" * 40)
    summary_lines.append("C. FILLER WORD REDUCTION STRATEGIES")
    summary_lines.append("─" * 40)

    if p_s6 > 0.10 or avg_filler > 0.05:
        summary_lines.append(f"""
  Problem: High filler usage ({avg_filler * 100:.1f}% of words are fillers)
  Fillers detected: {avg_num_fillers:.1f} per 10-second chunk

  IMMEDIATE ACTIONS:
  ├─ 1. Replace "um/uh" with a brief SILENT pause (0.3-0.5s)
  ├─ 2. Prepare opening sentences for each topic in advance
  ├─ 3. Slow down slightly - fillers often indicate rushing
  └─ 4. Have a glass of water handy; sipping can replace filler moments

  AWARENESS TECHNIQUES:
  ├─ Ask a colleague to count fillers during your next lecture
  ├─ Use a filler-counter app during practice sessions
  └─ Record yourself and transcribe - highlight every filler occurrence

  WEEKLY PRACTICE:
  ├─ Practice "thought completion" - finish one idea before starting next
  ├─ Do 10-minute impromptu speaking exercises on random topics
  └─ Replace "so" and "like" with structured transitions

  COGNITIVE REFRAME:
  └─ Silence is not awkward - it signals confidence and gives listeners time to process

  LONG-TERM GOAL: Reduce filler rate to <3%; eliminate S6 trigger
""")
    else:
        summary_lines.append(f"""
  Status: Filler usage is controlled ({avg_filler * 100:.1f}% filler rate)

  REFINEMENT SUGGESTIONS:
  ├─ 1. Continue awareness practice to prevent filler creep
  ├─ 2. Watch for stress-induced filler spikes during Q&A
  └─ 3. Substitute remaining fillers with confident pauses

  MAINTENANCE: Current performance is professional; stay vigilant
""")

    # --- Expressiveness Improvement Plan ---
    summary_lines.append("─" * 40)
    summary_lines.append("D. VOCAL EXPRESSIVENESS STRATEGIES")
    summary_lines.append("─" * 40)

    if p_s5 > 0.10 or avg_f0std < 20.0:
        summary_lines.append(f"""
  Problem: Monotone delivery ({p_s5 * 100:.1f}% in S5 state)
  Your pitch variation: {avg_f0std:.1f} Hz std dev (Ideal: 20-40 Hz)

  IMMEDIATE ACTIONS:
  ├─ 1. Underline KEY WORDS in notes - raise pitch on those words
  ├─ 2. Use the "question inflection" technique for rhetorical questions
  ├─ 3. Vary volume: louder for emphasis, softer for intimacy
  └─ 4. Inject emotion appropriate to content (enthusiasm, curiosity, concern)

  PRACTICE EXERCISES:
  ├─ Read children's books aloud with exaggerated expression
  ├─ Practice the same sentence 5 ways (excited, sad, curious, surprised, serious)
  ├─ Record and compare your pitch range to engaging speakers (TED talks)
  └─ Use pitch visualization apps to see your frequency variation

  LECTURE-SPECIFIC TIPS:
  ├─ Start topics with higher energy, then settle into explanation
  ├─ Use storytelling elements - build tension, release with conclusions
  └─ Ask rhetorical questions to naturally vary intonation

  LONG-TERM GOAL: Achieve pitch std dev >25 Hz; reduce S5 to <5%
""")
    else:
        summary_lines.append(f"""
  Status: Expressiveness is adequate (Pitch std: {avg_f0std:.1f} Hz)

  REFINEMENT SUGGESTIONS:
  ├─ 1. Add more dynamic range during key transitions
  ├─ 2. Use strategic pitch drops for authoritative statements
  └─ 3. Maintain energy level consistency throughout lecture

  MAINTENANCE: Current expressiveness supports engagement; continue practice
""")

    # --- Physical Presence Improvement Plan ---
    summary_lines.append("─" * 40)
    summary_lines.append("E. PHYSICAL PRESENCE STRATEGIES")
    summary_lines.append("─" * 40)

    if avg_motion < 0.005:
        summary_lines.append(f"""
  Problem: Very static presence (Motion index: {avg_motion:.4f})

  IMMEDIATE ACTIONS:
  ├─ 1. Use hand gestures to illustrate concepts (enumeration, size, direction)
  ├─ 2. Move between 2-3 designated spots during the lecture
  ├─ 3. Point to slides/board when referencing visual content
  └─ 4. Make deliberate eye contact with different sections of the room/camera

  GESTURE LIBRARY TO PRACTICE:
  ├─ "Listing" gesture - counting on fingers for enumerated points
  ├─ "Emphasis" gesture - open palm forward for important statements
  ├─ "Comparison" gesture - hands weighing alternatives
  └─ "Reference" gesture - pointing to visuals while explaining

  MOVEMENT PATTERNS:
  ├─ Introduction: Center position
  ├─ Key points: Move to emphasize (left/right of stage)
  ├─ Conclusions: Return to center
  └─ Q&A: Move toward questioner

  LONG-TERM GOAL: Increase motion index to 0.01-0.02 range
""")
    elif avg_motion > 0.04:
        summary_lines.append(f"""
  Problem: Excessive movement (Motion index: {avg_motion:.4f})

  IMMEDIATE ACTIONS:
  ├─ 1. Plant your feet during key explanations - move only during transitions
  ├─ 2. Reduce pacing - designate specific spots to stand
  ├─ 3. Keep gestures purposeful - eliminate nervous fidgeting
  └─ 4. Record yourself and watch with sound off - note distracting movements

  STILLNESS PRACTICE:
  ├─ Practice delivering 2-minute segments without moving feet
  ├─ Use a small rug/marker to define your "base" position
  └─ Channel energy into vocal variation instead of physical movement

  LONG-TERM GOAL: Reduce motion index to 0.015-0.030 range
""")
    else:
        summary_lines.append(f"""
  Status: Physical presence is appropriate (Motion index: {avg_motion:.4f})

  REFINEMENT SUGGESTIONS:
  ├─ 1. Ensure gestures align with verbal emphasis points
  ├─ 2. Maintain consistent camera framing if recording
  └─ 3. {motion_comment.capitalize()}

  MAINTENANCE: Current movement level supports engagement
""")

    # --- Weekly Improvement Schedule ---
    summary_lines.append("\n" + "─" * 40)
    summary_lines.append("F. SUGGESTED WEEKLY IMPROVEMENT SCHEDULE")
    summary_lines.append("─" * 40)

    summary_lines.append("""
  ┌─────────────┬────────────────────────────────────────────────────────┐
  │ Day         │ Focus Area                                             │
  ├─────────────┼────────────────────────────────────────────────────────┤
  │ Monday      │ Review previous lecture recording (15 min analysis)    │
  │ Tuesday     │ Pacing practice - read passages at target WPM          │
  │ Wednesday   │ Filler awareness - record 5-min impromptu speech       │
  │ Thursday    │ Expressiveness - practice dynamic pitch exercises      │
  │ Friday      │ Full rehearsal of upcoming lecture content             │
  │ Weekend     │ Review professional speakers (TED/YouTube) for ideas   │
  └─────────────┴────────────────────────────────────────────────────────┘

  PROGRESS TRACKING:
  ├─ Run this analysis tool after every 3-5 lectures
  ├─ Track scores over time in a spreadsheet
  ├─ Set specific numeric goals (e.g., "Reduce S2 from 15% to 8%")
  └─ Celebrate improvements - even 5% better is significant progress
""")

    # --- Final Notes ---
    summary_lines.append("\n" + "─" * 40)
    summary_lines.append("G. FINAL NOTES & NEXT STEPS")
    summary_lines.append("─" * 40)

    summary_lines.append(f"""
  YOUR CURRENT OVERALL SCORE: {overall_score:.1f}/10

  TARGET SCORE FOR NEXT ANALYSIS: {min(10.0, overall_score + 1.0):.1f}/10

  KEY FOCUS AREAS (pick 1-2 to work on first):""")

    if issues:
        for area, score, priority in issues[:2]:
            summary_lines.append(f"    → {area} (current: {score:.1f}/10)")
    else:
        summary_lines.append("    → Continue refining current strong performance")

    summary_lines.append("""
  REMEMBER:
  ├─ Improvement is gradual - expect 5-10% gains per month with consistent practice
  ├─ Not all lectures will be perfect - focus on trends, not individual sessions
  ├─ Student feedback is valuable - combine automated metrics with human input
  └─ Teaching is a skill that develops over years - be patient with yourself

  ═══════════════════════════════════════════════════════════════════════════
  This automated report provides objective metrics. Combine with peer feedback
  and student evaluations for a complete picture of teaching effectiveness.
  ═══════════════════════════════════════════════════════════════════════════
""")

    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("END OF REPORT")
    summary_lines.append("=" * 60)

    return "\n".join(summary_lines)


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    # 1) Audio extraction
    if AUDIO_PATH_OPTIONAL is None:
        print("[INFO] Extracting audio from video...")
        audio_path = extract_audio_from_video(VIDEO_PATH, EXTRACTED_AUDIO_PATH, target_sr=16000, mono=True)
    else:
        audio_path = AUDIO_PATH_OPTIONAL

    # 2) Load & frame audio
    print("[INFO] Loading and framing audio...")
    sr, audio = load_audio_wav(audio_path, target_sr=16000)
    frames, frame_times_audio = frame_signal(audio, sr, frame_size_ms=25.0, hop_size_ms=10.0)

    # 3) Frame-level audio features
    print("[INFO] Computing frame-level audio features (RMS, ZCR, Pitch, Clarity)...")
    energies, rms_vals = compute_frame_energy_rms(frames)
    zcr_vals = compute_zcr(frames)
    f0_vals = compute_pitch_autocorr(frames, sr)
    clarity_vals = compute_clarity_over_time(frames, sr)

    # 4) Speech transcription via Groq Whisper
    print("[INFO] Transcribing audio with Groq Whisper...")
    words = transcribe_audio_to_words(audio_path, language=None)
    if len(words) > 0:
        audio_duration_from_words = words[-1].end
    else:
        audio_duration_from_words = len(audio) / sr

    # 5) Video motion feature
    print("[INFO] Computing global video motion feature...")
    frame_times_video, motion_vals = compute_global_motion(VIDEO_PATH, frame_skip=2)

    # 6) Chunk-level feature aggregation
    print("[INFO] Aggregating features into chunks...")
    total_duration = max(
        frame_times_audio[-1] if frame_times_audio.size > 0 else 0.0,
        audio_duration_from_words,
        frame_times_video[-1] if frame_times_video.size > 0 else 0.0
    )

    chunk_features = build_chunk_feature_list(
        frame_times_audio=frame_times_audio,
        rms_values=rms_vals,
        zcr_values=zcr_vals,
        f0_values=f0_vals,
        clarity_values=clarity_vals,
        words=words,
        chunk_size=CHUNK_SIZE_SEC,
        total_duration=total_duration,
        frame_times_video=frame_times_video,
        motion_values=motion_vals,
    )

    # 7) FSM state assignment
    print("[INFO] Assigning FSM states to chunks...")
    annotated_chunks = assign_states_to_all_chunks(chunk_features)

    # 8) Markov chain analysis
    print("[INFO] Performing Markov chain analysis...")
    state_seq = extract_state_sequence(annotated_chunks)
    counts = compute_transition_counts(state_seq)
    P = normalize_transition_matrix(counts, smoothing=0.0)
    visit_dist = compute_state_visit_distribution(state_seq)
    markov_summary = summarize_markov_behavior(P, visit_dist, min_transition_prob=0.2)

    # 9) Build recommendation schedule (1 per minute)
    print("[INFO] Building recommendation schedule...")
    recommendations = build_minute_recommendations(
        annotated_chunks,
        total_duration=total_duration,
        interval_sec=RECOMMENDATION_INTERVAL_SEC
    )

    # 10) Create video with subtitle-like recommendations
    print("[INFO] Rendering video with overlayed recommendations...")
    create_video_with_recommendations(
        input_video_path=VIDEO_PATH,
        output_video_path=OUTPUT_VIDEO_WITH_TIPS,
        recommendations=recommendations,
    )

    # 11) Generate analytics report
    print("[INFO] Generating analytics report...")
    report_text = generate_analytics_report(
        annotated_chunks,
        transition_matrix=P,
        visit_dist=visit_dist,
        markov_summary=markov_summary,
    )
    with open(ANALYTICS_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[DONE] Video with tips saved to: {OUTPUT_VIDEO_WITH_TIPS}")
    print(f"[DONE] Analytics report saved to: {ANALYTICS_REPORT_PATH}")


if __name__ == "__main__":
    main()
