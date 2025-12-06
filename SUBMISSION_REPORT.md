# PRP Signal: Lecture Delivery Quality Analyzer
## Submission Report

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Feature Extraction Modules](#3-feature-extraction-modules)
4. [Finite State Machine (FSM) Design](#4-finite-state-machine-fsm-design)
5. [Threshold Justifications](#5-threshold-justifications)
6. [Markov Chain Analysis](#6-markov-chain-analysis)
7. [Output Generation](#7-output-generation)
8. [Conclusion](#8-conclusion)

---

## 1. Project Overview

### 1.1 Problem Statement
Effective teaching requires a combination of good pacing, clarity, vocal expressiveness, and physical engagement. However, educators often lack objective feedback on their delivery style. This project addresses this gap by providing an **automated, signal-processing-based analysis** of lecture videos.

### 1.2 Objectives
- Analyze lecture videos to extract quantitative metrics about teaching quality
- Classify teaching behavior into defined states (normal, too fast, monotone, etc.)
- Provide actionable recommendations for improvement
- Generate a comprehensive analytics report with detailed metrics

### 1.3 Input/Output
- **Input**: A lecture video file (MP4 format)
- **Outputs**:
  - `lecture_with_tips.mp4` - Original video with subtitle recommendations
  - `analytics_report.txt` - Detailed teaching analytics report

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Video Input    │────▶│  Audio Extraction │────▶│    Framing      │
│  (MP4 file)     │     │  (MoviePy/FFmpeg) │     │  (25ms windows) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
     ┌────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│                    FRAME-LEVEL FEATURE EXTRACTION                   │
├─────────────┬─────────────┬─────────────┬─────────────┬────────────┤
│  RMS Energy │     ZCR     │    Pitch    │   Clarity   │   Motion   │
│  (loudness) │ (voicing)   │ (F0/autocorr)│ (cepstrum) │ (video)    │
└─────────────┴─────────────┴─────────────┴─────────────┴────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Groq Whisper API    │
                    │   (Transcription)     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Chunk Aggregation   │
                    │   (10-second windows) │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   FSM State Scoring   │
                    │   (8 behavioral states)│
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
         ┌──────────────────┐    ┌───────────────────────┐
         │  Markov Analysis │    │  Video Overlay        │
         │  (transitions)   │    │  (recommendations)    │
         └────────┬─────────┘    └───────────────────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Analytics Report │
         └──────────────────┘
```

### 2.2 Module Dependencies

| Module | Purpose | Dependencies |
|--------|---------|--------------|
| `audio_extraction.py` | Extract audio from video | MoviePy, FFmpeg |
| `framing.py` | Segment audio into frames | NumPy, SciPy |
| `rms_energy.py` | Volume/loudness analysis | NumPy |
| `zcr.py` | Voice detection | NumPy |
| `pitch_autocorr.py` | Pitch/F0 estimation | NumPy, SciPy |
| `cepstrum_clarity.py` | Voice clarity measurement | NumPy, SciPy |
| `whisper_groq_features.py` | Speech transcription | Groq API |
| `video_features.py` | Motion analysis | OpenCV |
| `chunk_features.py` | Feature aggregation | NumPy |
| `fsm_scoring.py` | State classification | NumPy, SciPy |
| `markov_analysis.py` | Behavioral patterns | NumPy |
| `main.py` | Pipeline orchestration | All above |

---

## 3. Feature Extraction Modules

### 3.1 Audio Framing (`framing.py`)

**Purpose**: Convert continuous audio into overlapping analysis windows.

**How It Works**:
- Audio is sampled at 16 kHz (16,000 samples/second)
- Divided into **25ms frames** with **10ms hop** (overlap)
- Each frame contains 400 samples (0.025 × 16000)
- Hamming window applied to reduce spectral leakage

**Why These Parameters**:
- **25ms frame size**: Standard in speech processing; captures 2-3 pitch periods for typical voices (100-200 Hz fundamental frequency)
- **10ms hop size**: Provides 60% overlap for smooth feature trajectories
- **16 kHz sample rate**: Sufficient for speech (up to 8 kHz frequency content per Nyquist)

```python
# Frame parameters
frame_size_ms = 25.0   # 25 milliseconds per frame
hop_size_ms = 10.0     # 10 milliseconds between frames
```

---

### 3.2 RMS Energy (`rms_energy.py`)

**Purpose**: Measure the loudness/volume of each audio frame.

**Formula**:
```
RMS = sqrt(mean(samples²))
```

**How It Works**:
1. Square all samples in the frame
2. Calculate the mean of squared values
3. Take the square root

**Why RMS**:
- RMS (Root Mean Square) correlates with perceived loudness better than peak amplitude
- Robust to phase and polarity differences
- Standard metric in audio engineering

**Thresholds**:
| Threshold | Value | Meaning |
|-----------|-------|---------|
| Silence | < 0.005 | Very low energy, likely pause or silence |
| Too Soft | < 0.02 | Audible but quiet, hard to hear clearly |
| Normal | 0.02 - 0.10 | Good audible range for speech |
| Loud | > 0.10 | Strong projection, potentially too loud |

**Why 0.02-0.10 Range**:
- Based on normalized audio (peak at ±1.0)
- Speech typically has 20-40 dB dynamic range
- 0.02 corresponds to approximately -34 dB relative to full scale
- 0.10 corresponds to approximately -20 dB relative to full scale

---

### 3.3 Zero Crossing Rate (`zcr.py`)

**Purpose**: Distinguish voiced speech (vowels) from unvoiced sounds (fricatives, silence).

**Formula**:
```
ZCR = (1/N) × Σ|sign(x[n]) - sign(x[n-1])| / 2
```

**How It Works**:
1. Count times the signal crosses zero amplitude
2. Normalize by frame length to get rate

**Interpretation**:
| ZCR Value | Sound Type | Reason |
|-----------|------------|--------|
| Low (< 0.1) | Voiced speech | Regular periodic waveform crosses zero few times |
| High (> 0.3) | Unvoiced/noise | Random noise crosses zero frequently |
| Medium | Mixed/transition | Consonants, transitions |

**Why This Feature**:
- Computationally efficient (no FFT required)
- Good indicator of speech vs. noise
- Helps identify silence segments more reliably than RMS alone

---

### 3.4 Pitch Estimation (`pitch_autocorr.py`)

**Purpose**: Estimate the fundamental frequency (F0) of the voice for expressiveness analysis.

**Method**: Autocorrelation-based pitch detection

**How It Works**:
1. Compute autocorrelation of the frame: `R(τ) = Σ x[n] × x[n+τ]`
2. Find peaks in the autocorrelation function
3. The lag of the highest peak corresponds to the pitch period
4. Convert to frequency: `F0 = sample_rate / lag`

**Why Autocorrelation**:
- Robust to noise and harmonics
- Works well for monophonic speech
- Computationally efficient
- More reliable than FFT-based methods for pitch

**Pitch Ranges**:
| Voice Type | Typical F0 Range |
|------------|------------------|
| Male | 85 - 180 Hz |
| Female | 165 - 255 Hz |
| Children | 250 - 400 Hz |

**Key Metric - Pitch Standard Deviation**:
```
f0_std = std(f0_values)  # Hz
```

| f0_std | Interpretation |
|--------|----------------|
| < 15 Hz | Monotone - very flat delivery |
| 15-25 Hz | Low variation - somewhat flat |
| 25-40 Hz | Normal - good expressiveness |
| > 40 Hz | Highly expressive - animated |

**Why 20 Hz Threshold for Monotone**:
- Research shows engaging speakers have f0_std > 25 Hz
- Below 15-20 Hz, listeners perceive speech as monotonous
- We use 20 Hz as a conservative threshold

---

### 3.5 Cepstrum Clarity (`cepstrum_clarity.py`)

**Purpose**: Measure voice clarity vs. reverberation/noise.

**Method**: Real cepstrum analysis

**How It Works**:
1. Compute log magnitude spectrum: `log|FFT(x)|`
2. Take inverse FFT to get cepstrum
3. Split into "low quefrency" (pitch-related) and "high quefrency" (noise)
4. Clarity index = low_energy / high_energy

**The Cepstrum**:
```
Cepstrum = IFFT(log|FFT(signal)|)
```

- "Quefrency" is the cepstral domain analogue of frequency
- Low quefrency captures harmonic structure (voice)
- High quefrency captures noise/reverberation

**Why Cepstrum**:
- Separates source (vocal folds) from filter (vocal tract)
- Naturally measures signal-to-noise ratio
- Standard in speech quality assessment

**Clarity Index Thresholds**:
| Value | Quality |
|-------|---------|
| < 0.8 | Poor - significant noise/reverb |
| 0.8 - 1.2 | Moderate - some issues |
| 1.2 - 2.0 | Good - clear voice |
| > 2.0 | Excellent - very clean recording |

---

### 3.6 Speech Transcription (`whisper_groq_features.py`)

**Purpose**: Convert speech to text and extract linguistic features.

**Method**: Groq Whisper API (cloud-based ASR)

**Extracted Features**:

1. **Speaking Rate (SR)** - Words per minute including pauses
   ```
   SR = (word_count / total_duration) × 60
   ```

2. **Articulation Rate (AR)** - Words per minute during active speech
   ```
   AR = (word_count / speech_duration) × 60
   ```

3. **Pause Statistics**:
   - Number of pauses (gaps > 400ms)
   - Average pause duration
   - Maximum pause duration

4. **Filler Words**:
   - Count of: um, uh, er, ah, so, like, actually, basically, right, okay, literally
   - Filler rate = filler_count / total_words

**Why Groq Whisper**:
- State-of-the-art accuracy for speech recognition
- Provides word-level timestamps
- Supports multiple languages
- Fast inference (streaming capable)

**Speaking Rate Thresholds**:
| SR (WPM) | Classification |
|----------|----------------|
| < 100 | Too slow |
| 100-120 | Slow but acceptable |
| 120-180 | Ideal range for lectures |
| 180-200 | Fast but acceptable |
| > 200 | Too fast |

**Why 120-180 WPM**:
- Research on lecture comprehension shows optimal retention at 150-170 WPM
- Below 120 WPM, engagement drops
- Above 180 WPM, comprehension decreases
- Range of 120-180 WPM accommodates individual variation

**Pause Detection Threshold (400ms)**:
- Micro-pauses (< 200ms): Natural speech rhythm
- Short pauses (200-500ms): Breath pauses, natural
- Long pauses (> 500ms): May indicate hesitation
- We use 400ms to capture meaningful pauses without false positives

---

### 3.7 Video Motion Analysis (`video_features.py`)

**Purpose**: Measure physical movement and engagement.

**Method**: Frame differencing

**How It Works**:
1. Convert frames to grayscale
2. Compute absolute difference between consecutive frames
3. Average the difference values
4. Normalize to [0, 1] range

```python
motion_index = mean(|frame[t] - frame[t-1]|) / 255
```

**Why Frame Differencing**:
- Simple and computationally efficient
- Captures all types of motion (walking, gesturing)
- Works without face/body detection (more robust)

**Motion Index Interpretation**:
| Value | Physical Behavior |
|-------|-------------------|
| < 0.003 | Very static - minimal movement |
| 0.003-0.015 | Moderate - acceptable |
| 0.015-0.035 | Active - good engagement |
| > 0.035 | Excessive - potentially distracting |

**Why These Thresholds**:
- Derived from empirical observation of lecture videos
- Accounts for camera stability and lighting
- Balances engagement with distraction concerns

---

## 4. Finite State Machine (FSM) Design

### 4.1 State Definitions

| State | Name | Description | Primary Indicators |
|-------|------|-------------|-------------------|
| S0 | Silence | No speech detected | RMS < 0.005, words < 2 |
| S1 | Normal | All metrics in ideal range | Baseline state |
| S2 | Too Fast | Speaking too quickly | SR > 180 WPM |
| S3 | Over-Pausing | Too many/long pauses | avg_pause > 0.8s, SR < 120 |
| S4 | Too Soft | Low volume | RMS < 0.02 |
| S5 | Monotone | Flat pitch | f0_std < 20 Hz |
| S6 | High Fillers | Excessive filler words | filler_rate > 5% |
| S7 | Low Clarity | Poor audio quality | clarity_index < 1.0 |

### 4.2 State Priority System

When multiple issues are detected simultaneously, we prioritize by impact:

```python
STATE_PRIORITY = {
    "S4": 1.2,  # Too Soft - critical for comprehension
    "S7": 1.1,  # Low Clarity - affects understanding
    "S2": 1.0,  # Too Fast - pacing issue
    "S3": 0.9,  # Over-Pausing - less critical
    "S5": 0.8,  # Monotone - engagement issue
    "S6": 0.7,  # Fillers - least critical
}
```

**Rationale**:
1. **Volume (S4)** is most critical - if students can't hear, nothing else matters
2. **Clarity (S7)** comes second - poor quality affects all comprehension
3. **Pacing (S2/S3)** affects information throughput
4. **Expressiveness (S5)** affects engagement but not comprehension
5. **Fillers (S6)** are noticeable but least impactful

### 4.3 Scoring Algorithm

Each chunk is scored for all problem states:

```python
def compute_deviation(value, ideal_min, ideal_max):
    if value < ideal_min:
        return (ideal_min - value) / ideal_min
    elif value > ideal_max:
        return (value - ideal_max) / ideal_max
    return 0.0  # Within ideal range

# State score = weighted sum of deviations
score = sum(weight * deviation for (feature, direction, weight) in STATE_WEIGHTS)
```

### 4.4 Normality Threshold

```python
NORMALITY_THRESHOLD = 0.25
```

If all problem state scores are below this threshold, the chunk is classified as S1 (Normal).

**Why 0.25**:
- Allows for minor deviations without flagging issues
- Reduces false positives
- Based on iterative tuning with test lectures

---

## 5. Threshold Justifications

### 5.1 Summary of All Thresholds

| Feature | Threshold | Value | Justification |
|---------|-----------|-------|---------------|
| RMS (silence) | < | 0.005 | Below noise floor |
| RMS (too soft) | < | 0.02 | -34 dB, hard to hear |
| RMS (ideal max) | < | 0.10 | Comfortable listening |
| Speaking Rate | min | 120 WPM | Comprehension drops below |
| Speaking Rate | max | 180 WPM | Comprehension drops above |
| Pause threshold | > | 0.4s | Distinguishes meaningful pauses |
| Filler rate | > | 5% | Becomes distracting |
| Pitch std (monotone) | < | 20 Hz | Perceived as flat |
| Clarity index | < | 1.0 | Noise affects comprehension |
| Motion (static) | < | 0.003 | No visible movement |
| Motion (excessive) | > | 0.035 | Distracting movement |
| Normality threshold | < | 0.25 | Minor deviations allowed |
| Recommendation dominance | > | 40% | Only show if >40% of chunks affected |

### 5.2 Threshold Derivation Methods

1. **Literature-Based**: Speaking rate thresholds from educational psychology research
2. **Signal Theory**: RMS and clarity thresholds from audio engineering standards
3. **Empirical Tuning**: Motion and normality thresholds from testing with sample lectures
4. **Expert Calibration**: Pitch and pause thresholds based on speech science literature

---

## 6. Markov Chain Analysis

### 6.1 Purpose

Model the **temporal dynamics** of teaching behavior - how the speaker transitions between states over time.

### 6.2 Transition Matrix

```
P[i][j] = P(next_state = j | current_state = i)
```

A 8×8 matrix where each row sums to 1.

### 6.3 Insights Provided

1. **Self-Transitions**: High P[i][i] indicates persistent habits
   - e.g., P[S2][S2] = 0.7 means "Too Fast" state tends to persist

2. **Problematic Transitions**: Which normal behaviors lead to problems
   - e.g., High P[S1][S2] indicates tendency to speed up from normal

3. **Recovery Patterns**: How quickly issues are corrected
   - e.g., High P[S2][S1] indicates quick recovery from fast speech

### 6.4 State Visit Distribution

```python
visit_dist[state] = count(state) / total_chunks
```

Shows percentage of lecture time in each behavioral state.

---

## 7. Output Generation

### 7.1 Video with Recommendations

- Subtitle-style overlays every 30 seconds
- Only shows recommendation if problem state dominates (>40%) that interval
- Short, actionable messages

### 7.2 Analytics Report Structure

1. **Executive Summary**: Scores out of 10 for each dimension
2. **Key Metrics**: Raw quantitative values
3. **Behavioral Dynamics**: Markov analysis results
4. **Feature Glossary**: Explanation of all metrics
5. **Improvement Plan**: Personalized recommendations

---

## 8. Conclusion

### 8.1 Key Innovations

1. **Multi-modal Analysis**: Combines audio, speech, and video features
2. **Principled Thresholds**: Based on research and signal theory
3. **Behavioral Modeling**: Markov chains capture temporal patterns
4. **Actionable Feedback**: Prioritized recommendations for improvement

### 8.2 Limitations

1. Requires clear audio (sensitive to background noise)
2. Motion analysis assumes stable camera
3. Transcription accuracy depends on audio quality
4. Thresholds may need tuning for different contexts (online vs. classroom)

### 8.3 Future Work

1. Speaker diarization for multi-speaker videos
2. Emotion detection from voice
3. Eye contact and gesture recognition
4. Adaptive thresholds based on lecture type
5. Real-time feedback during live lectures

---

## References

1. Hincks, R. (2005). Rate of speech in presentations. *PTLC Conference*.
2. Keller, E. (1994). Fundamentals of speech synthesis and speech recognition. *Wiley*.
3. Rabiner, L. R., & Schafer, R. W. (2010). Theory and applications of digital speech processing.
4. Nair, A. G. (2020). Analysis of pauses in spontaneous speech. *Speech Communication*.
5. OpenAI Whisper. (2022). Robust Speech Recognition via Large-Scale Weak Supervision.

---

*Report generated for PRP Signal - Lecture Delivery Quality Analyzer*
*Version 1.0 - December 2024*
