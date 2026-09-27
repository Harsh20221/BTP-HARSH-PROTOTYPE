# Long-form video censorship pipeline

This is an offline v1 batch pipeline. It samples visual detections every 0.1 seconds, uses the
nearest NudeNet boxes and violence score for each rendered
frame, and mutes profane word intervals from faster-whisper transcription.

## Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- A CUDA-capable PyTorch installation and compatible NVIDIA drivers

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

Run the CLI:

```powershell
python pipeline.py input.mp4 censored.mp4
```

Optional sampling and violence settings:

```powershell
python pipeline.py input.mp4 censored.mp4 --sample-interval 0.5 --violence-threshold 0.6
```

## Benchmark evaluation

Create a JSON annotation file with timestamps in seconds. Visual metrics are calculated
at the sampled timestamps; profanity intervals are matched using temporal intersection-over-union.

```json
{
	"videos": [
		{
			"input": "sample.mp4",
			"violence": [[12.0, 15.5]],
			"nudity": [],
			"profanity": [[28.2, 28.8]]
		}
	]
}
```

Run the benchmark with:

```powershell
python pipeline.py --evaluate annotations.json
```

The JSON report contains precision, recall, F1-score, true/false positives and negatives
for violence, nudity, and profanity, along with runtime and video-seconds-per-runtime-second.
Use `--minimum-iou 0.5` to change the profanity interval matching threshold.

Run the optional Gradio UI with `python app.py`.

The detector methods operate on one frame and the renderer consumes one frame at a time. The
file-backed iterators in `ingest.py` are therefore the replaceable v1 boundary for a future
rolling live buffer; live support is intentionally not included in v1.