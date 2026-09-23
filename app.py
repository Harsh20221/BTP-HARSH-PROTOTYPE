"""Optional Gradio wrapper around the batch entry point."""

from pathlib import Path

import gradio as gr

from pipeline import process_video


def process_uploaded_video(video_path: str) -> str:
    source = Path(video_path)
    output = source.with_name(f"{source.stem}_censored.mp4")
    return str(process_video(source, output))


demo = gr.Interface(
    fn=process_uploaded_video,
    inputs=gr.Video(),
    outputs=gr.File(),
    title="Long-form video censor",
)


if __name__ == "__main__":
    demo.launch()