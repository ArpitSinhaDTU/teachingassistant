# 🎓 Teaching Assistant - Lecture Delivery Quality Analyzer

A powerful AI-driven tool that analyzes lecture videos to provide objective, actionable feedback on teaching quality using signal processing and machine learning techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)

---

## 📋 Overview

**Teaching Assistant** automatically analyzes lecture videos to extract quantitative metrics about teaching quality. It classifies teaching behavior into defined states (normal, too fast, monotone, etc.) and provides actionable recommendations for improvement.

### ✨ Key Features

- 🎤 **Audio Analysis** - RMS energy, pitch estimation, voice clarity measurement
- 🗣️ **Speech Analysis** - Speaking rate, filler word detection, pause analysis
- 📹 **Video Analysis** - Motion detection and physical engagement tracking
- 🤖 **AI Transcription** - Groq Whisper API for accurate speech-to-text
- 📊 **Behavioral Modeling** - Finite State Machine (FSM) with Markov Chain analysis
- 💡 **Smart Recommendations** - Context-aware tips overlaid on video

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg installed and available in PATH
- Groq API key (for speech transcription)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArpitSinhaDTU/teachingassistant.git
   cd teachingassistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Usage

```bash
python main.py
```

By default, the script analyzes `samplelec.mp4`. To analyze a different video, modify the input path in `main.py`.

---

## 📁 Project Structure

```
teachingassistant/
├── main.py                    # Main pipeline orchestration
├── audio_extraction.py        # Extract audio from video (MoviePy/FFmpeg)
├── framing.py                 # Segment audio into 25ms frames
├── rms_energy.py              # Volume/loudness analysis
├── zcr.py                     # Zero Crossing Rate for voice detection
├── pitch_autocorr.py          # Pitch/F0 estimation via autocorrelation
├── cepstrum_clarity.py        # Voice clarity measurement
├── whisper_groq_features.py   # Speech transcription & linguistic features
├── video_features.py          # Motion analysis via frame differencing
├── chunk_features.py          # Feature aggregation (10-second windows)
├── fsm_scoring.py             # FSM state classification
├── markov_analysis.py         # Behavioral pattern analysis
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys)
├── SUBMISSION_REPORT.md       # Detailed technical documentation
└── analytics_report.txt       # Sample output report
```

---

## 🔬 How It Works

### Pipeline Overview

```
Video Input → Audio Extraction → Frame-Level Features → Chunk Aggregation → FSM Scoring → Analytics Report
                                        ↓
                               Groq Whisper ASR
```

### Feature Extraction

| Module | Feature | Description |
|--------|---------|-------------|
| `rms_energy.py` | RMS Energy | Measures loudness/volume per frame |
| `zcr.py` | Zero Crossing Rate | Distinguishes voiced vs unvoiced speech |
| `pitch_autocorr.py` | Pitch (F0) | Estimates fundamental frequency for expressiveness |
| `cepstrum_clarity.py` | Clarity Index | Measures voice clarity vs noise/reverb |
| `whisper_groq_features.py` | Speaking Rate | Words per minute, filler detection |
| `video_features.py` | Motion Index | Physical engagement via frame differencing |

### FSM States

| State | Name | Primary Indicators |
|-------|------|-------------------|
| S0 | Silence | RMS < 0.005, words < 2 |
| S1 | Normal | All metrics in ideal range |
| S2 | Too Fast | Speaking rate > 180 WPM |
| S3 | Over-Pausing | avg_pause > 0.8s, SR < 120 |
| S4 | Too Soft | RMS < 0.02 |
| S5 | Monotone | Pitch std < 20 Hz |
| S6 | High Fillers | Filler rate > 5% |
| S7 | Low Clarity | Clarity index < 1.0 |

---

## 📤 Outputs

### 1. Video with Recommendations
`output_video_with_tips.mp4` - Original video with subtitle-style recommendations overlaid every 30 seconds.

### 2. Analytics Report
`analytics_report.txt` - Comprehensive report including:
- 📈 **Executive Summary** - Scores out of 10 for each dimension
- 📊 **Key Metrics** - Raw quantitative values
- 🔄 **Behavioral Dynamics** - Markov analysis results
- 📖 **Feature Glossary** - Explanation of all metrics
- ✅ **Improvement Plan** - Personalized recommendations

---

## 📊 Sample Metrics

| Metric | Ideal Range | Description |
|--------|-------------|-------------|
| Speaking Rate | 120-180 WPM | Optimal for lecture comprehension |
| RMS Energy | 0.02-0.10 | Comfortable listening volume |
| Pitch Std | > 20 Hz | Indicates vocal expressiveness |
| Filler Rate | < 5% | Minimal distracting fillers |
| Clarity Index | > 1.0 | Good voice clarity |
| Motion Index | 0.003-0.035 | Appropriate physical engagement |

---

## 🛠️ Dependencies

All dependencies are listed in `requirements.txt`. Key packages include:

- **groq** - Whisper API for speech transcription
- **moviepy** - Video/audio processing
- **opencv-python** - Video frame analysis
- **scipy** - Signal processing algorithms
- **numpy** - Numerical computations
- **python-dotenv** - Environment variable management

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📚 Documentation

For detailed technical documentation including:
- Signal processing methodology
- Threshold justifications
- FSM design rationale
- Markov chain analysis

See [SUBMISSION_REPORT.md](SUBMISSION_REPORT.md)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

