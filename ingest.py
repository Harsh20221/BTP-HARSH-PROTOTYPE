"""File-backed ingest for v1; the yielded frame/chunk shapes are live-friendly."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


@dataclass(frozen=True)
class FramePacket:
    index: int
    timestamp: float
    frame: object


def probe_video(video_path: str | Path) -> VideoMetadata:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = (int(part) for part in stream["r_frame_rate"].split("/"))
    fps = numerator / denominator
    frame_count = int(stream.get("nb_frames") or 0)
    duration = float(
        stream.get("duration")
        or payload.get("format", {}).get("duration")
        or (frame_count / fps if frame_count else 0)
    )
    return VideoMetadata(int(stream["width"]), int(stream["height"]), fps, frame_count, duration)


def extract_audio(video_path: str | Path, audio_path: str | Path) -> None:
    command = [
        "ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", str(audio_path),
    ]
    subprocess.run(command, check=True)


def iter_frames(video_path: str | Path) -> Iterator[FramePacket]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield FramePacket(index, index / fps, frame)
            index += 1
    finally:
        capture.release()


def iter_sampled_frames(video_path: str | Path, interval_seconds: float) -> Iterator[FramePacket]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")
    next_sample = 0.0
    for packet in iter_frames(video_path):
        if packet.timestamp + 1e-9 >= next_sample:
            yield packet
            next_sample += interval_seconds