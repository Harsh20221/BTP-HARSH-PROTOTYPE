"""Benchmarking for detector predictions against timestamp annotations."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from audio_path import MuteInterval, transcribe_and_find_profanity
from ingest import extract_audio, iter_sampled_frames, probe_video
from visual_path import VisualDetection, VisualDetector


@dataclass(frozen=True)
class Counts:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def _intervals(value: object) -> list[MuteInterval]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each annotation interval must be [start, end].")
        start, end = float(item[0]), float(item[1])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid annotation interval: {item!r}")
        result.append(MuteInterval(start, end))
    return result


def _contains(timestamp: float, intervals: Iterable[MuteInterval]) -> bool:
    return any(interval.start <= timestamp <= interval.end for interval in intervals)


def _sample_counts(
    samples: list[tuple[float, VisualDetection]],
    annotations: list[MuteInterval],
    positive: callable,
) -> Counts:
    true_positive = false_positive = false_negative = 0
    for timestamp, detection in samples:
        predicted = positive(detection)
        expected = _contains(timestamp, annotations)
        if predicted and expected:
            true_positive += 1
        elif predicted:
            false_positive += 1
        elif expected:
            false_negative += 1
    return Counts(true_positive, false_positive, false_negative)


def _temporal_iou(left: MuteInterval, right: MuteInterval) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union else 0.0


def _interval_counts(
    predicted: list[MuteInterval],
    expected: list[MuteInterval],
    minimum_iou: float,
) -> Counts:
    matched_expected: set[int] = set()
    true_positive = 0
    for prediction in predicted:
        candidates = [
            (index, _temporal_iou(prediction, truth))
            for index, truth in enumerate(expected)
            if index not in matched_expected
        ]
        if candidates and max(candidates, key=lambda item: item[1])[1] >= minimum_iou:
            index, _ = max(candidates, key=lambda item: item[1])
            matched_expected.add(index)
            true_positive += 1
    return Counts(
        true_positive=true_positive,
        false_positive=len(predicted) - true_positive,
        false_negative=len(expected) - true_positive,
    )


def _load_cases(annotation_path: Path) -> list[dict[str, object]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    cases = payload.get("videos") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark annotations must contain a non-empty 'videos' list.")
    return cases


def benchmark(
    annotation_path: str | Path,
    sample_interval: float = 0.1,
    violence_threshold: float = 0.80,
    detection_batch_size: int = 16,
    minimum_iou: float = 0.5,
) -> dict[str, object]:
    annotation_path = Path(annotation_path)
    cases = _load_cases(annotation_path)
    detector = VisualDetector(violence_threshold)
    results = []
    total_runtime = 0.0

    for case in cases:
        if not isinstance(case, dict) or "input" not in case:
            raise ValueError("Each benchmark video must be an object with an 'input' field.")
        input_path = Path(str(case["input"]))
        if not input_path.is_absolute():
            input_path = annotation_path.parent / input_path
        annotations = {
            "violence": _intervals(case.get("violence", [])),
            "nudity": _intervals(case.get("nudity", [])),
            "profanity": _intervals(case.get("profanity", [])),
        }
        metadata = probe_video(input_path)
        started = time.perf_counter()
        samples: list[tuple[float, VisualDetection]] = []
        batch = []
        for packet in iter_sampled_frames(input_path, sample_interval):
            batch.append(packet)
            if len(batch) == detection_batch_size:
                detections = detector.detect_many([item.frame for item in batch], detection_batch_size)
                samples.extend((item.timestamp, detection) for item, detection in zip(batch, detections))
                batch.clear()
        if batch:
            detections = detector.detect_many([item.frame for item in batch], detection_batch_size)
            samples.extend((item.timestamp, detection) for item, detection in zip(batch, detections))
        with tempfile.TemporaryDirectory(prefix="benchmark_audio_") as temporary:
            audio_path = Path(temporary) / "audio.wav"
            extract_audio(input_path, audio_path)
            predicted_profanity = transcribe_and_find_profanity(audio_path)
        elapsed = time.perf_counter() - started
        total_runtime += elapsed

        metrics = {
            "violence": _sample_counts(
                samples, annotations["violence"],
                lambda detection: detection.violence_score >= violence_threshold,
            ),
            "nudity": _sample_counts(
                samples, annotations["nudity"],
                lambda detection: bool(detection.nudity_boxes),
            ),
            "profanity": _interval_counts(
                predicted_profanity, annotations["profanity"], minimum_iou,
            ),
        }
        results.append({
            "video_name": input_path.name,
            "input": str(input_path),
            "duration_seconds": round(metadata.duration, 3),
            "runtime_seconds": round(elapsed, 3),
            "video_seconds_per_runtime_second": round(metadata.duration / elapsed, 3) if elapsed else 0.0,
            "metrics": {name: value.as_dict() for name, value in metrics.items()},
        })

    metric_values = [
        result["metrics"][name]
        for result in results
        for name in ("violence", "nudity", "profanity")
    ]
    macro = {
        key: round(sum(float(item[key]) for item in metric_values) / len(metric_values), 4)
        for key in ("precision", "recall", "f1")
    }
    total_duration = sum(float(result["duration_seconds"]) for result in results)
    return {
        "videos": results,
        "macro_average": macro,
        "total_runtime_seconds": round(total_runtime, 3),
        "total_video_seconds": round(total_duration, 3),
        "overall_video_seconds_per_runtime_second": round(total_duration / total_runtime, 3) if total_runtime else 0.0,
    }


def run_benchmark(annotation_path: str | Path, **kwargs: object) -> None:
    annotation_path = Path(annotation_path)
    report = benchmark(annotation_path, **kwargs)
    video_results = report["videos"]
    if isinstance(video_results, list) and len(video_results) == 1:
        video_name = Path(str(video_results[0]["video_name"])).stem
        report_path = annotation_path.parent / f"benchmark_results_{video_name}.json"
    else:
        report_path = annotation_path.parent / "benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Benchmark report saved to: {report_path}", file=sys.stderr)
