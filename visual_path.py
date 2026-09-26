"""Stateless one-frame visual detectors."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
import os

import cv2
import nudenet

if os.name == "nt":
    torch_dll_path = Path(__file__).parent / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    if torch_dll_path.exists():
        os.add_dll_directory(str(torch_dll_path))
        os.environ["PATH"] = f"{torch_dll_path};{os.environ.get('PATH', '')}"

import onnxruntime
from nudenet import NudeDetector
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor, pipeline


@dataclass(frozen=True)
class NudityBox:
    label: str
    score: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class VisualDetection:
    nudity_boxes: tuple[NudityBox, ...]
    violence_score: float


_BLURRABLE_NUDITY_LABELS = frozenset(
    {
        "EXPOSED_ANUS",
        "EXPOSED_BREAST_F",
        "EXPOSED_BUTTOCKS",
        "EXPOSED_GENITALIA_F",
        "EXPOSED_GENITALIA_M",
        "EXPOSED_NIPPLES",
    }
)


class VisualDetector:
    def __init__(self, violence_threshold: float = 0.80) -> None:
        self.nudity_detector = NudeDetector()
        model_path = Path(nudenet.__file__).parent / "320n.onnx"
        session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.nudity_detector.onnx_session = session
        active_providers = self.nudity_detector.onnx_session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                "NudeNet could not initialize a GPU execution provider. Install a CUDA-compatible "
                "onnxruntime-gpu build and NVIDIA driver."
            )
        self.violence_detector = self._load_violence_detector()
        self.violence_threshold = violence_threshold

    @staticmethod
    def _blood_score(frame) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 90, 45), (12, 255, 255))
        red_mask |= cv2.inRange(hsv, (170, 90, 45), (179, 255, 255))
        red_ratio = cv2.countNonZero(red_mask) / red_mask.size
        grid = cv2.resize(red_mask, (8, 8), interpolation=cv2.INTER_AREA) / 255.0
        global_red_coverage = float((grid >= 0.15).mean())
        if global_red_coverage >= 0.45:
            return 0.0
        return min(1.0, red_ratio / 0.20)

    @staticmethod
    def _is_red_lighting(frame) -> bool:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 90, 45), (12, 255, 255))
        red_mask |= cv2.inRange(hsv, (170, 90, 45), (179, 255, 255))
        red_ratio = cv2.countNonZero(red_mask) / red_mask.size
        grid = cv2.resize(red_mask, (8, 8), interpolation=cv2.INTER_AREA) / 255.0
        global_red_coverage = float((grid >= 0.15).mean())
        return red_ratio >= 0.30 and global_red_coverage >= 0.45

    @staticmethod
    def _load_violence_detector():
        try:
            model_id = "hosamEl1/vit-base-violence-detection"
            return VisualDetector._load_converted_vit_pipeline(model_id)
        except Exception:
            model_id = "jaranohaal/vit-base-violence-detection"
            return VisualDetector._load_converted_vit_pipeline(model_id)

    @staticmethod
    def _load_converted_vit_pipeline(model_id: str):
        model_config = ViTConfig.from_pretrained(model_id)
        model = ViTForImageClassification(model_config)
        checkpoint_path = hf_hub_download(model_id, "model.safetensors")
        checkpoint = load_file(checkpoint_path)
        converted = {}

        converted["vit.embeddings.cls_token"] = checkpoint["cls_token"]
        converted["vit.embeddings.position_embeddings"] = checkpoint["pos_embed"]
        converted["vit.embeddings.patch_embeddings.projection.weight"] = checkpoint["patch_embed.proj.weight"]
        converted["vit.embeddings.patch_embeddings.projection.bias"] = checkpoint["patch_embed.proj.bias"]
        converted["vit.layernorm.weight"] = checkpoint["norm.weight"]
        converted["vit.layernorm.bias"] = checkpoint["norm.bias"]
        converted["classifier.weight"] = checkpoint["head.weight"]
        converted["classifier.bias"] = checkpoint["head.bias"]

        for layer_index in range(model.config.num_hidden_layers):
            source = f"blocks.{layer_index}"
            target = f"vit.encoder.layer.{layer_index}"
            converted[f"{target}.layernorm_before.weight"] = checkpoint[f"{source}.norm1.weight"]
            converted[f"{target}.layernorm_before.bias"] = checkpoint[f"{source}.norm1.bias"]
            converted[f"{target}.layernorm_after.weight"] = checkpoint[f"{source}.norm2.weight"]
            converted[f"{target}.layernorm_after.bias"] = checkpoint[f"{source}.norm2.bias"]
            query, key, value = checkpoint[f"{source}.attn.qkv.weight"].chunk(3, dim=0)
            query_bias, key_bias, value_bias = checkpoint[f"{source}.attn.qkv.bias"].chunk(3, dim=0)
            converted[f"{target}.attention.attention.query.weight"] = query
            converted[f"{target}.attention.attention.key.weight"] = key
            converted[f"{target}.attention.attention.value.weight"] = value
            converted[f"{target}.attention.attention.query.bias"] = query_bias
            converted[f"{target}.attention.attention.key.bias"] = key_bias
            converted[f"{target}.attention.attention.value.bias"] = value_bias
            converted[f"{target}.attention.output.dense.weight"] = checkpoint[f"{source}.attn.proj.weight"]
            converted[f"{target}.attention.output.dense.bias"] = checkpoint[f"{source}.attn.proj.bias"]
            converted[f"{target}.intermediate.dense.weight"] = checkpoint[f"{source}.mlp.fc1.weight"]
            converted[f"{target}.intermediate.dense.bias"] = checkpoint[f"{source}.mlp.fc1.bias"]
            converted[f"{target}.output.dense.weight"] = checkpoint[f"{source}.mlp.fc2.weight"]
            converted[f"{target}.output.dense.bias"] = checkpoint[f"{source}.mlp.fc2.bias"]

        missing, unexpected = model.load_state_dict(converted, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Incomplete violence checkpoint conversion: missing={missing}, unexpected={unexpected}")
        model.to("cuda")
        processor = ViTImageProcessor.from_pretrained(model_id)
        return pipeline("image-classification", model=model, image_processor=processor, device=0)

    def _detect_nudity(self, frame) -> tuple[NudityBox, ...]:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
            frame_path = Path(temporary.name)
        try:
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError("Could not write temporary frame for NudeNet")
            nudity = self.nudity_detector.detect(str(frame_path))
        finally:
            frame_path.unlink(missing_ok=True)

        return tuple(
            NudityBox(
                label=str(item.get("class", "nudity")),
                score=float(item.get("score", 0.0)),
                x=int(item["box"][0]), y=int(item["box"][1]),
                width=int(item["box"][2]), height=int(item["box"][3]),
            )
            for item in nudity
            if str(item.get("class", "")).upper() in _BLURRABLE_NUDITY_LABELS
            and float(item.get("score", 0.0)) >= 0.35
        )

    @staticmethod
    def _violence_score(predictions, frame) -> float:
        violence_predictions = [
            item
            for item in predictions
            if str(item["label"]).strip().lower() in {"violence", "violent"}
        ]
        if not violence_predictions:
            violence_predictions = [item for item in predictions if str(item["label"]).upper() == "LABEL_1"]
        classifier_score = max((float(item["score"]) for item in violence_predictions), default=0.0)
        if VisualDetector._is_red_lighting(frame):
            return 0.0
        if classifier_score < 0.55:
            return classifier_score

        # Treat red as supporting evidence only; red lighting alone must not blur a frame.
        blood_support = min(0.25, 0.25 * VisualDetector._blood_score(frame))
        return min(1.0, classifier_score + blood_support)

    def detect_many(self, frames: list, batch_size: int = 16) -> list[VisualDetection]:
        if not frames:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        nudity_boxes = [self._detect_nudity(frame) for frame in frames]
        rgb_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        prediction_batches = self.violence_detector(rgb_frames, batch_size=min(batch_size, len(rgb_frames)))
        return [
            VisualDetection(boxes, self._violence_score(predictions, frame))
            for frame, boxes, predictions in zip(frames, nudity_boxes, prediction_batches)
        ]

    def detect(self, frame) -> VisualDetection:
        return self.detect_many([frame])[0]