"""
PyTorch inference API — khớp notebook Colab (resize 224, ImageNet normalize, BGR→RGB qua OpenCV).
Đặt checkpoint vào artifacts/:
  - brain_mri_alexnet_best.pth (hoặc brain_mri_alexnet_final.pth)
  - brain_stroke_ct_resnet18_best.pth (hoặc brain_stroke_ct_resnet18_final.pth)
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from torchvision import models

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
MODEL_ROOT = PROJECT_ROOT / "model"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF_IMPORT_ERROR: Exception | None = None
_tf_module: Any | None = None

_app = FastAPI(title="DACN3 Inference")


class MedicalAlexNet(nn.Module):
    def __init__(self, num_classes: int, use_pretrained: bool = False):
        super().__init__()
        weights = models.AlexNet_Weights.DEFAULT if use_pretrained else None
        try:
            self.model = models.alexnet(weights=weights)
        except Exception:
            self.model = models.alexnet(weights=None)

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class StrokeCTResNet18(nn.Module):
    def __init__(self, num_classes: int, use_pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        try:
            self.model = models.resnet18(weights=weights)
        except Exception:
            self.model = models.resnet18(weights=None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _pick_checkpoint(best: Path, final: Path) -> Path | None:
    if best.exists():
        return best
    if final.exists():
        return final
    return None


def _load_label_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _decode_and_preprocess(
    raw: bytes, image_size: tuple[int, int]
) -> torch.Tensor:
    buf = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được ảnh (định dạng không hỗ trợ hoặc file hỏng).")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    t = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)
    return t


def _decode_raw_image(raw: bytes) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được ảnh nguồn.")
    return image


def _decode_for_keras(raw: bytes, image_size: tuple[int, int]) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Khong doc duoc anh nguon.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    return np.expand_dims(image.astype(np.float32), axis=0)


def _labels_from_json(data: dict[str, Any]) -> list[str]:
    n = len(data["class_to_idx"])
    out = [""] * n
    for name, idx in data["class_to_idx"].items():
        out[int(idx)] = name
    return out


def _modality_labels_from_json(data: dict[str, Any]) -> dict[int, str]:
    return {int(idx): str(name) for idx, name in data.items()}


def _tensorflow_available() -> bool:
    if _tf_module is not None:
        return True
    if TF_IMPORT_ERROR is not None:
        return False
    return importlib.util.find_spec("tensorflow") is not None


def _tensorflow_status_error() -> str | None:
    if _tf_module is not None:
        return None
    if TF_IMPORT_ERROR is not None:
        return str(TF_IMPORT_ERROR)
    if importlib.util.find_spec("tensorflow") is None:
        return (
            f"Chua cai TensorFlow trong Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Model CT/MRI dang la file .keras nen can TensorFlow. "
            "Hay tao moi truong Python 3.12 va cai lai requirements.txt."
        )
    return None


def _get_tensorflow() -> Any:
    global TF_IMPORT_ERROR, _tf_module
    if _tf_module is not None:
        return _tf_module
    try:
        import tensorflow as tf
    except Exception as exc:
        TF_IMPORT_ERROR = exc
        raise RuntimeError(f"TensorFlow chua san sang: {exc}") from exc
    _tf_module = tf
    TF_IMPORT_ERROR = None
    return tf


def _tensor_to_base64_png(rgb_image: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode('.png', bgr)
    if not success:
        raise ValueError('Không thể mã hóa ảnh PNG.')
    return base64.b64encode(encoded.tobytes()).decode('ascii')


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: nn.Module, _input: tuple, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(_module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        score = output[0, class_idx]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError('Không thu được activation/gradient cho Grad-CAM.')

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam_np = cam.cpu().numpy()
        cam_np -= cam_np.min()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)
        return cam_np


class ModelRunner:
    def __init__(
        self,
        name: str,
        checkpoint: Path,
        labels_path: Path,
        factory: type[MedicalAlexNet] | type[StrokeCTResNet18],
        target_layer_fn: Callable[[nn.Module], nn.Module],
    ):
        self.name = name
        self.checkpoint = checkpoint
        lab = _load_label_json(labels_path)
        self.image_size = tuple(int(x) for x in lab["image_size"])
        self.class_names = _labels_from_json(lab)
        num_classes = len(self.class_names)

        self.net = factory(num_classes=num_classes, use_pretrained=False).to(DEVICE)
        try:
            ck = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        except TypeError:
            ck = torch.load(checkpoint, map_location=DEVICE)
        state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
        self.gradcam = GradCAM(self.net, target_layer_fn(self.net))

    def predict(self, raw_bytes: bytes) -> dict[str, Any]:
        original_bgr = _decode_raw_image(raw_bytes)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        x = _decode_and_preprocess(raw_bytes, self.image_size)
        with torch.enable_grad():
            logits = self.net(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            idx = int(np.argmax(probs))
            label = self.class_names[idx]
            cam = self.gradcam(x, idx)

        cam_uint8 = np.uint8(cam * 255)
        cam_resized = cv2.resize(cam_uint8, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_bgr = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.addWeighted(original_rgb, 0.55, heatmap_rgb, 0.45, 0)

        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        return {
            "label": label,
            "confidence": float(probs[idx]),
            "probabilities": prob_dict,
            "device": str(DEVICE),
            "checkpoint": self.checkpoint.name,
            "heatmap": _tensor_to_base64_png(heatmap_rgb),
            "overlay": _tensor_to_base64_png(overlay_rgb),
        }


class KerasModalityRunner:
    def __init__(self, model_path: Path, labels_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Khong tim thay model: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Khong tim thay labels: {labels_path}")

        tf = _get_tensorflow()
        self.model_path = model_path
        self.image_size = (224, 224)
        self.idx_to_class = _modality_labels_from_json(_load_label_json(labels_path))
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, raw_bytes: bytes) -> dict[str, Any]:
        x = _decode_for_keras(raw_bytes, self.image_size)
        pred = self.model.predict(x, verbose=0)
        mri_score = float(np.ravel(pred)[0])
        mri_score = max(0.0, min(1.0, mri_score))

        negative_label = self.idx_to_class.get(0, "CT")
        positive_label = self.idx_to_class.get(1, "MRI")
        probabilities = {
            negative_label: float(1.0 - mri_score),
            positive_label: float(mri_score),
        }
        label = positive_label if mri_score >= 0.5 else negative_label
        return {
            "label": label,
            "confidence": float(probabilities[label]),
            "probabilities": probabilities,
            "device": "tensorflow",
            "checkpoint": self.model_path.name,
        }


_mri_runner: ModelRunner | None = None
_ct_runner: ModelRunner | None = None
_modality_runner: KerasModalityRunner | None = None


def get_mri_runner() -> ModelRunner:
    global _mri_runner
    if _mri_runner is None:
        ck = _pick_checkpoint(
            ARTIFACTS / "brain_mri_alexnet_best.pth",
            ARTIFACTS / "brain_mri_alexnet_final.pth",
        )
        if ck is None:
            raise HTTPException(
                status_code=503,
                detail="Chưa có checkpoint MRI. Copy brain_mri_alexnet_best.pth vào thư mục artifacts/",
            )
        _mri_runner = ModelRunner(
            "mri",
            ck,
            ARTIFACTS / "brain_mri_labels.json",
            MedicalAlexNet,
            lambda net: net.model.features[-1],
        )
    return _mri_runner


def get_ct_runner() -> ModelRunner:
    global _ct_runner
    if _ct_runner is None:
        ck = _pick_checkpoint(
            ARTIFACTS / "brain_stroke_ct_resnet18_best.pth",
            ARTIFACTS / "brain_stroke_ct_resnet18_final.pth",
        )
        if ck is None:
            raise HTTPException(
                status_code=503,
                detail="Chưa có checkpoint CT. Copy brain_stroke_ct_resnet18_best.pth vào artifacts/",
            )
        _ct_runner = ModelRunner(
            "stroke_ct",
            ck,
            ARTIFACTS / "brain_stroke_ct_labels.json",
            StrokeCTResNet18,
            lambda net: net.model.layer4[-1].conv2,
        )
    return _ct_runner


def get_modality_runner() -> KerasModalityRunner:
    global _modality_runner
    if _modality_runner is None:
        model_path = MODEL_ROOT / "ct_mri_classifier.keras"
        labels_path = MODEL_ROOT / "ct_mri_labels.json"
        missing = [str(p.relative_to(PROJECT_ROOT)) for p in (model_path, labels_path) if not p.exists()]
        if missing:
            raise HTTPException(status_code=503, detail="Thieu file modality model: " + ", ".join(missing))
        try:
            _modality_runner = KerasModalityRunner(model_path, labels_path)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _modality_runner


def predict_auto_workflow(raw: bytes) -> dict[str, Any]:
    modality = get_modality_runner().predict(raw)
    modality_label = str(modality.get("label", "")).strip().upper()

    if modality_label == "MRI":
        diagnosis = get_mri_runner().predict(raw)
        selected_model = "mri"
        selected_title = "MRI - U nao"
    elif modality_label == "CT":
        diagnosis = get_ct_runner().predict(raw)
        selected_model = "strokeCt"
        selected_title = "CT - Dot quy nao"
    else:
        raise ValueError(f"Nhan CT/MRI khong hop le: {modality.get('label')}")

    return {
        "label": diagnosis["label"],
        "confidence": diagnosis["confidence"],
        "probabilities": diagnosis["probabilities"],
        "device": diagnosis.get("device"),
        "checkpoint": diagnosis.get("checkpoint"),
        "heatmap": diagnosis.get("heatmap"),
        "overlay": diagnosis.get("overlay"),
        "selectedModel": selected_model,
        "selectedTitle": selected_title,
        "modality": modality,
        "diagnosis": diagnosis,
    }


@_app.get("/health")
def health() -> dict[str, Any]:
    mri_ck = _pick_checkpoint(
        ARTIFACTS / "brain_mri_alexnet_best.pth",
        ARTIFACTS / "brain_mri_alexnet_final.pth",
    )
    ct_ck = _pick_checkpoint(
        ARTIFACTS / "brain_stroke_ct_resnet18_best.pth",
        ARTIFACTS / "brain_stroke_ct_resnet18_final.pth",
    )
    modality_model = MODEL_ROOT / "ct_mri_classifier.keras"
    modality_labels = MODEL_ROOT / "ct_mri_labels.json"
    modality_files_ready = modality_model.exists() and modality_labels.exists()
    modality_dependency_ready = _tensorflow_available()
    return {
        "ok": True,
        "device": str(DEVICE),
        "mri": {
            "checkpoint": mri_ck.name if mri_ck else None,
            "ready": mri_ck is not None,
        },
        "stroke_ct": {
            "checkpoint": ct_ck.name if ct_ck else None,
            "ready": ct_ck is not None,
        },
        "modality": {
            "checkpoint": modality_model.name if modality_model.exists() else None,
            "ready": modality_files_ready and modality_dependency_ready,
            "filesReady": modality_files_ready,
            "tensorflowReady": modality_dependency_ready,
            "error": _tensorflow_status_error(),
        },
    }


@_app.post("/predict/mri")
async def predict_mri(image: UploadFile = File(...)) -> dict[str, Any]:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File rỗng.")
    try:
        return get_mri_runner().predict(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@_app.post("/predict/stroke")
async def predict_stroke(image: UploadFile = File(...)) -> dict[str, Any]:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File rỗng.")
    try:
        return get_ct_runner().predict(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@_app.post("/predict/modality")
async def predict_modality(image: UploadFile = File(...)) -> dict[str, Any]:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File rong.")
    try:
        return get_modality_runner().predict(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@_app.post("/predict/auto")
async def predict_auto(image: UploadFile = File(...)) -> dict[str, Any]:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File rong.")
    try:
        return predict_auto_workflow(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("INFERENCE_PORT", "5050"))
    uvicorn.run(_app, host="127.0.0.1", port=port)
