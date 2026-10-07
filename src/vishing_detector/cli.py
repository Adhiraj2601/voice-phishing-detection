"""Command-line interface for voice phishing detection system."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vishing_detector.anomaly.detector import AcousticAnomalyDetector
from vishing_detector.audio.loader import AudioLoader
from vishing_detector.features.acoustic import AcousticFeatureExtractor
from vishing_detector.pipeline import VishingDetectionPipeline

# Set up logging and rich console
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
console = Console()
app = typer.Typer(
    help="AI-Powered Voice Phishing (Vishing) Detection System CLI",
    add_completion=False,
)


@app.command()
def detect(
    audio_file: Path = typer.Argument(..., help="Path to input audio file (WAV/MP3/M4A/OGG)"),
    config: Optional[Path] = typer.Option("config/default.yaml", "--config", "-c", help="Path to configuration YAML"),
    anomaly_model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to trained anomaly model joblib"),
    asr: str = typer.Option("vosk", "--asr", help="Speech recognition backend ('vosk' or 'mock')"),
) -> None:
    """Analyze a call recording and print a comprehensive multi-modal forensic risk report."""
    if not audio_file.exists():
        console.print(f"[bold red]Error:[/] Audio file not found: {audio_file}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]Analyzing Call Recording:[/] {audio_file.name}")
    pipeline = VishingDetectionPipeline(
        config_path=config,
        anomaly_model_path=anomaly_model,
        asr_backend=asr,
    )

    with console.status("[bold green]Processing audio stream, extracting acoustic features & ASR..."):
        result = pipeline.process_file(audio_file)

    expl = result.explanation
    if expl is None:
        console.print("[red]No result generated.[/]")
        return

    # Color code risk level
    risk_color = {
        "SAFE": "green",
        "LOW": "blue",
        "ELEVATED": "yellow",
        "HIGH": "bright_red",
        "CRITICAL": "bold red",
    }.get(expl.risk_level, "white")

    # Threat Summary Panel
    summary_text = (
        f"[bold]Overall Threat Score:[/] [{risk_color}]{expl.overall_risk_score:.1f} / 100[/{risk_color}]\n"
        f"[bold]Threat Level:[/] [{risk_color}]{expl.risk_level}[/{risk_color}]\n"
        f"[bold]Peak Threat Score:[/] {expl.peak_risk_score:.1f}\n"
        f"[bold]Call Duration:[/] {result.audio_duration:.2f}s ({result.total_chunks} windows)\n"
        f"[bold]Time to First Alert:[/] {f'{expl.first_alert_time:.1f}s' if expl.first_alert_time is not None else 'None'}\n"
        f"[bold]Analysis Processing Time:[/] {result.total_processing_time_sec:.2f}s"
    )
    console.print(Panel(summary_text, title="[bold]Vishing Risk Assessment[/]", border_style=risk_color))

    # Key Reasons
    console.print("\n[bold yellow]Forensic Indicators:[/]")
    for r in expl.primary_reasons:
        console.print(f"  - {r}")

    # Top Triggered Cues Table
    if expl.top_triggered_cues:
        console.print("\n[bold red]Triggered Scam Cues:[/]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("Matched Phrase", style="yellow")
        table.add_column("Severity Weight", justify="right")
        table.add_column("Window Timestamp", justify="right")

        for cue in expl.top_triggered_cues:
            table.add_row(
                cue["category"],
                f'"{cue["matched_text"]}"',
                f"{cue['weight']:.1f}",
                cue["timestamp"],
            )
        console.print(table)

    # Transcript Preview
    if result.cumulative_transcript:
        console.print("\n[bold cyan]Transcript Extracted:[/]")
        console.print(Panel(result.cumulative_transcript, border_style="dim"))

    # Recommended Action
    console.print(f"\n[bold underline]Recommended Action:[/] [{risk_color}]{expl.recommended_action}[/{risk_color}]\n")


@app.command()
def stream(
    audio_file: Path = typer.Argument(..., help="Path to input audio file"),
    config: Optional[Path] = typer.Option("config/default.yaml", "--config", "-c"),
    asr: str = typer.Option("vosk", "--asr", help="Speech recognition backend ('vosk' or 'mock')"),
    simulated_delay: bool = typer.Option(True, "--delay/--no-delay", help="Simulate real-time playback delay"),
) -> None:
    """Simulate real-time streaming detection window-by-window with live telemetry."""
    if not audio_file.exists():
        console.print(f"[bold red]Error:[/] Audio file not found: {audio_file}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold green]Streaming Call Simulation:[/] {audio_file.name}")
    pipeline = VishingDetectionPipeline(config_path=config, asr_backend=asr)

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Time Window", justify="center")
    table.add_column("Acoustic Anom", justify="right")
    table.add_column("NLP Score", justify="right")
    table.add_column("Threat Index", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Transcribed Text", style="dim")

    for event in pipeline.stream_file(audio_file):
        a = event.assessment
        score = a.smoothed_risk
        color = "green" if score < 25 else "yellow" if score < 50 else "bright_red" if score < 75 else "bold red"

        table.add_row(
            f"{event.start_time:4.1f}s - {event.end_time:4.1f}s",
            f"{a.acoustic_score:.2f}",
            f"{a.text_score:.2f}",
            f"[{color}]{score:5.1f}[/{color}]",
            f"[{color}]{a.risk_level}[/{color}]",
            event.transcript_snippet[:40] if event.transcript_snippet else "...",
        )
        if simulated_delay:
            time.sleep(0.3)

    console.print(table)
    console.print("\n[bold green]Stream Completed.[/]\n")


@app.command()
def train(
    benign_dir: Path = typer.Option("data/samples", "--data-dir", "-d", help="Directory of benign WAV files"),
    model_type: str = typer.Option("isolation_forest", "--model-type", "-t", help="'isolation_forest' or 'one_class_svm'"),
    output_model: Path = typer.Option("models/acoustic_anomaly_model.joblib", "--output", "-o", help="Save path for model"),
) -> None:
    """Train unsupervised acoustic anomaly detector on benign/normal audio files."""
    console.print(f"\n[bold cyan]Training Acoustic Anomaly Model ({model_type})...[/]")

    loader = AudioLoader()
    extractor = AcousticFeatureExtractor()

    wav_files = list(benign_dir.glob("*.wav"))
    if not wav_files:
        console.print(f"[bold yellow]Warning:[/] No WAV files found in {benign_dir}. Generating synthetic benign training data...")
        from scripts.generate_synthetic_data import generate_dataset
        generate_dataset(output_dir=Path("data/synthetic"), num_benign=10, num_scam=5)
        wav_files = list(Path("data/synthetic/benign").glob("*.wav"))

    console.print(f"Loading {len(wav_files)} benign audio samples...")
    feature_vectors = []
    for f in wav_files:
        try:
            seg, _ = loader.load(f)
            chunks = loader.chunk_all(seg)
            for c in chunks:
                feats = extractor.extract_chunk(c)
                feature_vectors.append(feats.vector)
        except Exception as err:
            console.print(f"[dim]Skipping {f.name}: {err}[/]")

    if not feature_vectors:
        console.print("[bold red]Failed to extract any features.[/]")
        raise typer.Exit(code=1)

    X_train = np.array(feature_vectors)
    console.print(f"Extracted {len(X_train)} chunks across {len(extractor.feature_names)} acoustic features.")

    detector = AcousticAnomalyDetector(
        model_type=model_type,
        feature_names=extractor.feature_names,
    )
    detector.fit(X_train, calibrate_threshold=True)
    detector.save(output_model)

    console.print(f"[bold green]Successfully trained and saved model to:[/] {output_model}\n")


@app.command()
def evaluate(
    test_dir: Path = typer.Option("data/samples", "--test-dir", help="Directory containing test samples"),
    output_dir: Path = typer.Option("docs/figures", "--output-dir", help="Output directory for evaluation plots"),
) -> None:
    """Evaluate pipeline metrics (Precision, Recall, ROC-AUC, Latency) and render plots."""
    console.print("\n[bold cyan]Running System Evaluation & Metric Computation...[/]")
    from evaluate import run_full_evaluation

    run_full_evaluation(test_dir=test_dir, output_dir=output_dir)


if __name__ == "__main__":
    app()
