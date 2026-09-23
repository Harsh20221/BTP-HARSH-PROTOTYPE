"""Speech transcription and profanity timing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from better_profanity import profanity
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class MuteInterval:
    start: float
    end: float


def transcribe_and_find_profanity(audio_path: str | Path, model_size: str = "medium") -> list[MuteInterval]:
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
    intervals: list[MuteInterval] = []
    for segment in segments:
        for word in segment.words or []:
            normalized = word.word.strip()
            if normalized and profanity.contains_profanity(normalized):
                intervals.append(MuteInterval(float(word.start), float(word.end)))
    return merge_intervals(intervals)


def merge_intervals(intervals: list[MuteInterval], padding: float = 0.05) -> list[MuteInterval]:
    expanded = sorted(
        (MuteInterval(max(0.0, item.start - padding), item.end + padding) for item in intervals),
        key=lambda item: item.start,
    )
    merged: list[MuteInterval] = []
    for item in expanded:
        if merged and item.start <= merged[-1].end:
            merged[-1] = MuteInterval(merged[-1].start, max(merged[-1].end, item.end))
        else:
            merged.append(item)
    return merged