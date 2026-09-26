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

Run the optional Gradio UI with `python app.py`.

The detector methods operate on one frame and the renderer consumes one frame at a time. The
file-backed iterators in `ingest.py` are therefore the replaceable v1 boundary for a future
rolling live buffer; live support is intentionally not included in v1.