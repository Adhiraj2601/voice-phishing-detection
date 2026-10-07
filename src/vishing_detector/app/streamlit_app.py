"""Streamlit web dashboard for real-time voice phishing detection demo."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from vishing_detector.pipeline import VishingDetectionPipeline

st.set_page_config(
    page_title="AI Voice Phishing (Vishing) Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #3b82f6;
    }
    .badge-credential { background-color: #ef4444; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .badge-payment { background-color: #f97316; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .badge-urgency { background-color: #eab308; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .badge-impersonation { background-color: #a855f7; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .badge-remote { background-color: #06b6d4; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .badge-secrecy { background-color: #64748b; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


def highlight_scam_transcript(transcript: str, cues: list) -> str:
    """Format transcript string with HTML badges wrapping identified scam cues."""
    if not transcript or not cues:
        return transcript

    highlighted = transcript
    badge_classes = {
        "Credential and Authentication Requests": "badge-credential",
        "Unusual Payment and Fund Transfer": "badge-payment",
        "Urgency and Coercion": "badge-urgency",
        "Entity Impersonation": "badge-impersonation",
        "Remote Computer Access": "badge-remote",
        "Secrecy and Isolation Tactics": "badge-secrecy",
    }

    # Sort cues by length descending so longer phrases match first
    sorted_cues = sorted(cues, key=lambda c: len(c.get("matched_text", "")), reverse=True)
    seen = set()

    for cue in sorted_cues:
        phrase = cue.get("matched_text", "")
        cat = cue.get("category", "")
        cls_name = badge_classes.get(cat, "badge-urgency")

        if phrase.lower() in seen or not phrase:
            continue
        seen.add(phrase.lower())

        replacement = f"<span class='{cls_name}' title='{cat}'>{phrase}</span>"
        import re
        highlighted = re.sub(re.escape(phrase), replacement, highlighted, flags=re.IGNORECASE)

    return highlighted


def plot_waveform_and_spectrogram(audio_path: Path):
    """Render waveform and mel-spectrogram plot using Matplotlib."""
    import librosa
    y, sr = librosa.load(str(audio_path), sr=16000)
    times = np.linspace(0, len(y) / sr, len(y))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    fig.patch.set_facecolor("#0e1117")

    # Waveform
    ax1.set_facecolor("#161b22")
    ax1.plot(times, y, color="#38bdf8", linewidth=0.8)
    ax1.set_ylabel("Amplitude", color="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, color="#30363d", alpha=0.5)

    # Mel-spectrogram
    ax2.set_facecolor("#161b22")
    s_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    s_db = librosa.power_to_db(s_mel, ref=np.max)
    librosa.display.specshow(s_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2, cmap="magma")
    ax2.set_ylabel("Frequency (Hz)", color="white")
    ax2.set_xlabel("Time (seconds)", color="white")
    ax2.tick_params(colors="white")

    plt.tight_layout()
    return fig


def plot_risk_timeline(assessments: list):
    """Plot risk score trajectory over time with threshold zones."""
    times = [a.start_time for a in assessments]
    smoothed = [a.smoothed_risk for a in assessments]
    acoustic = [a.acoustic_score * 100 for a in assessments]
    text_scores = [a.text_score * 100 for a in assessments]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    # Threshold horizontal alert regions
    ax.axhspan(0, 25, color="#10b981", alpha=0.1, label="Safe (0-25)")
    ax.axhspan(25, 50, color="#f59e0b", alpha=0.1, label="Elevated (25-50)")
    ax.axhspan(50, 75, color="#f97316", alpha=0.15, label="High (50-75)")
    ax.axhspan(75, 100, color="#ef4444", alpha=0.2, label="Critical (75-100)")

    # Trajectories
    ax.plot(times, smoothed, color="#ef4444", linewidth=2.5, marker="o", markersize=4, label="Fused Threat Index")
    ax.plot(times, acoustic, color="#38bdf8", linestyle="--", linewidth=1.5, alpha=0.7, label="Acoustic Anomaly (%)")
    ax.plot(times, text_scores, color="#fbbf24", linestyle=":", linewidth=1.5, alpha=0.7, label="NLP Scam Cues (%)")

    ax.set_ylim(-2, 105)
    ax.set_xlabel("Call Duration (seconds)", color="white", fontsize=11)
    ax.set_ylabel("Risk Score (0 - 100)", color="white", fontsize=11)
    ax.set_title("Real-Time Threat Score Evolution", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white")
    ax.grid(True, color="#30363d", alpha=0.5)
    ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    return fig


# Sidebar Configuration
st.sidebar.title("🛡️ Vishing Detector Settings")

asr_option = st.sidebar.selectbox(
    "Speech Recognition (ASR) Engine",
    ["vosk", "mock"],
    help="Offline Vosk model or fast test Mock transcriber",
)

st.sidebar.subheader("Audio Input")
input_mode = st.sidebar.radio("Select Audio Source:", ["Preset Sample Audio", "Upload Audio File"])

audio_path = None
if input_mode == "Preset Sample Audio":
    samples_dir = Path("data/samples")
    available_samples = list(samples_dir.glob("*.wav")) if samples_dir.exists() else []
    sample_names = [f.name for f in available_samples]
    if sample_names:
        selected_sample = st.sidebar.selectbox("Choose demo recording:", sample_names)
        audio_path = samples_dir / selected_sample
    else:
        st.sidebar.info("No preset samples in data/samples. Please upload a file.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload call recording (WAV, MP3, M4A, OGG)", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
        tfile.write(uploaded_file.read())
        audio_path = Path(tfile.name)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Architecture:**
    - 🎵 **PyDub**: 16 kHz Mono Ingestion & Windowing
    - 📊 **Librosa**: 119 Acoustic Features
    - 🤖 **Isolation Forest**: Vocal Anomaly Detection
    - 🗣️ **Vosk**: Offline Speech-to-Text
    - 🔍 **Scam NLP**: Lexical & Regex Threat Cues
    - ⚖️ **Multi-Modal Fusion**: 0-100 Threat Index
    """
)

# Main Dashboard
st.title("🛡️ AI-Powered Voice Phishing (Vishing) Detection")
st.markdown(
    "Real-time telephony security system combining **Acoustic Vocal Anomaly Modeling** "
    "with **Offline Speech Recognition** and **Transparent Scam Cue Lexical Matching**."
)

if audio_path and audio_path.exists():
    st.subheader("1. Audio Playback & Inspection")
    st.audio(str(audio_path))

    with st.expander("Spectrogram & Waveform Analysis", expanded=False):
        fig_audio = plot_waveform_and_spectrogram(audio_path)
        st.pyplot(fig_audio)

    if st.button("🚀 Run Live Threat Analysis", type="primary"):
        with st.spinner("Processing call stream through multi-modal detection pipeline..."):
            pipeline = VishingDetectionPipeline(
                config_path="config/default.yaml",
                asr_backend=asr_option,
            )
            result = pipeline.process_file(audio_path)

        expl = result.explanation
        if expl:
            st.markdown("---")
            st.subheader("2. Forensic Threat Evaluation")

            # Threat Alert Card
            risk_color_map = {
                "SAFE": "#10b981",
                "LOW": "#3b82f6",
                "ELEVATED": "#f59e0b",
                "HIGH": "#f97316",
                "CRITICAL": "#ef4444",
            }
            color = risk_color_map.get(expl.risk_level, "#ffffff")

            alert_title = (
                "🚨 CRITICAL VISHING ATTACK DETECTED"
                if expl.risk_level in ["CRITICAL", "HIGH"]
                else "⚠️ SUSPICIOUS / ELEVATED CALL PATTERN"
                if expl.risk_level == "ELEVATED"
                else "✅ NORMAL BENIGN CONVERSATION"
            )

            st.markdown(
                f"""
                <div style="background-color: {color}22; border: 2px solid {color}; border-radius: 8px; padding: 18px; margin-bottom: 20px;">
                    <h3 style="color: {color}; margin: 0 0 8px 0;">{alert_title}</h3>
                    <p style="font-size: 1.1em; margin: 0; color: white;">
                        Overall Threat Index: <strong>{expl.overall_risk_score:.1f} / 100</strong> &nbsp;|&nbsp;
                        Threat Tier: <strong>{expl.risk_level}</strong> &nbsp;|&nbsp;
                        First Alert Latency: <strong>{f'{expl.first_alert_time:.1f}s' if expl.first_alert_time is not None else 'N/A'}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metric Summary Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Threat Index", f"{expl.overall_risk_score:.1f} / 100")
            with col2:
                st.metric("Peak Threat Score", f"{expl.peak_risk_score:.1f}")
            with col3:
                st.metric("Call Duration", f"{result.audio_duration:.1f}s")
            with col4:
                st.metric("Analysis Latency", f"{result.total_processing_time_sec:.2f}s")

            # Timeline Graph
            st.subheader("3. Real-Time Risk Progression Timeline")
            fig_timeline = plot_risk_timeline(result.chunk_assessments)
            st.pyplot(fig_timeline)

            # Forensic Highlights & Transcript
            c_left, c_right = st.columns([1.1, 1])

            with c_left:
                st.subheader("4. Forensic Explanations")
                for reason in expl.primary_reasons:
                    st.write(f"- {reason}")

                st.markdown(f"**Recommended Protocol:** {expl.recommended_action}")

                if expl.top_triggered_cues:
                    st.markdown("**Detected Scam Indicators:**")
                    cue_df = pd.DataFrame(expl.top_triggered_cues)
                    st.dataframe(cue_df, use_container_width=True)

            with c_right:
                st.subheader("5. Time-Aligned Transcript")
                if result.cumulative_transcript:
                    highlighted_html = highlight_scam_transcript(
                        result.cumulative_transcript, expl.top_triggered_cues
                    )
                    st.markdown(
                        f"""
                        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; font-size: 0.95em; line-height: 1.6;">
                            {highlighted_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No spoken words recognized in this clip.")
else:
    st.info("👈 Please select or upload an audio recording in the left sidebar to begin.")
