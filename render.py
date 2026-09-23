"""Incremental video rendering and audio/video muxing."""

from __future__ import annotations

import bisect
import subprocess
from pathlib import Path

import cv2

from audio_path import MuteInterval
from visual_path import VisualDetection


def _nearest_detection(timestamp: float, samples: list[tuple[float, VisualDetection]]) -> VisualDetection | None:
    if not samples:
        return None
    positions = [item[0] for item in samples]
    index = bisect.bisect_left(positions, timestamp)
    if index == 0:
        return samples[0][1]
    if index == len(samples):
        return samples[-1][1]
    before, after = samples[index - 1], samples[index]
    return before[1] if timestamp - before[0] <= after[0] - timestamp else after[1]


def _latest_detection(timestamp: float, samples: list[tuple[float, VisualDetection]]) -> VisualDetection | None:
    if not samples:
        return None
    positions = [item[0] for item in samples]
    index = bisect.bisect_right(positions, timestamp) - 1
    return samples[max(index, 0)][1]


def render_video(
    input_path: str | Path,
    silent_video_path: str | Path,
    samples: list[tuple[float, VisualDetection]],
    violence_threshold: float,
) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(silent_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {silent_video_path}")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / fps
            nudity = _latest_detection(timestamp, samples)
            violence = _nearest_detection(timestamp, samples)
            if nudity:
                for box in nudity.nudity_boxes:
                    x1, y1 = max(0, box.x), max(0, box.y)
                    x2, y2 = min(width, box.x + box.width), min(height, box.y + box.height)
                    if x2 > x1 and y2 > y1:
                        region = frame[y1:y2, x1:x2]
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (0, 0), 25)
            if violence and violence.violence_score >= violence_threshold:
                frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=18, sigmaY=18)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()


def mux_with_muted_audio(
    silent_video_path: str | Path,
    original_video_path: str | Path,
    output_path: str | Path,
    mute_intervals: list[MuteInterval],
) -> None:
    audio_filter = "volume=1"
    for interval in mute_intervals:
        audio_filter += f",volume=enable='between(t,{interval.start},{interval.end})':volume=0"
    command = [
        "ffmpeg", "-y", "-i", str(silent_video_path), "-i", str(original_video_path),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-af", audio_filter, "-c:a", "aac", "-shortest", str(output_path),
    ]
    subprocess.run(command, check=True)