"""Batch v1 orchestration; detection and rendering remain frame/chunk scoped."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from audio_path import transcribe_and_find_profanity
from ingest import extract_audio, iter_sampled_frames, probe_video
from render import mux_with_muted_audio, render_video
from visual_path import VisualDetector


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    sample_interval: float = 0.3,
    violence_threshold: float = 0.80,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(input_path)
    print(f"Input: {metadata.width}x{metadata.height}, {metadata.fps:.2f} fps, {metadata.duration:.1f}s")
    detector = VisualDetector(violence_threshold)
    samples = []
    batch_packets = []
    batch_size = 8
    for packet in iter_sampled_frames(input_path, sample_interval):
        batch_packets.append(packet)
        if len(batch_packets) < batch_size:
            continue
        print(f"Detecting frames at {batch_packets[0].timestamp:.1f}s-{batch_packets[-1].timestamp:.1f}s")
        detections = detector.detect_many([item.frame for item in batch_packets])
        samples.extend((item.timestamp, detection) for item, detection in zip(batch_packets, detections))
        batch_packets.clear()
    if batch_packets:
        print(f"Detecting frames at {batch_packets[0].timestamp:.1f}s-{batch_packets[-1].timestamp:.1f}s")
        detections = detector.detect_many([item.frame for item in batch_packets])
        samples.extend((item.timestamp, detection) for item, detection in zip(batch_packets, detections))

    with tempfile.TemporaryDirectory(prefix="video_censor_") as temporary:
        temporary_path = Path(temporary)
        audio_path = temporary_path / "audio.wav"
        silent_video_path = temporary_path / "video_silent.mp4"
        extract_audio(input_path, audio_path)
        mute_intervals = transcribe_and_find_profanity(audio_path)
        render_video(input_path, silent_video_path, samples, violence_threshold)
        mux_with_muted_audio(silent_video_path, input_path, output_path, mute_intervals)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Censor nudity, violent frames, and profane speech in a video.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.3)
    parser.add_argument("--violence-threshold", type=float, default=0.80)
    args = parser.parse_args()
    process_video(args.input, args.output, args.sample_interval, args.violence_threshold)


if __name__ == "__main__":
    main()