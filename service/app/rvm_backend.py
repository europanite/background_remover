from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RvmConfig:
    model_path: Path
    repo_path: Path
    cuda_device: int = 0
    downsample_ratio: float | None = None
    fp16: bool = True


def torch_cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def default_rvm_model_path() -> Path:
    return Path(os.environ.get("RVM_MODEL_PATH", "/models/rvm_mobilenetv3.pth"))


def default_rvm_repo_path() -> Path:
    return Path(os.environ.get("RVM_REPO", "/opt/RobustVideoMatting"))


class RvmSession:
    """CUDA/CPU inference wrapper for Robust Video Matting.

    The network predicts a foreground image and a soft alpha matte at the
    original input resolution. Internal low-resolution processing is controlled
    by ``downsample_ratio`` and never changes output dimensions.
    """

    def __init__(self, config: RvmConfig, *, width: int, height: int, use_cuda: bool) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "RVM requires PyTorch. Use docker-compose.gpu.yml / Dockerfile.gpu."
            ) from exc

        if not config.repo_path.exists():
            raise FileNotFoundError(f"RVM repository not found: {config.repo_path}")
        if not config.model_path.exists():
            raise FileNotFoundError(f"RVM checkpoint not found: {config.model_path}")

        repo = str(config.repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from model import MattingNetwork  # type: ignore[import-not-found]

        self._torch = torch
        self._device = torch.device(f"cuda:{config.cuda_device}" if use_cuda else "cpu")
        self._dtype = torch.float16 if use_cuda and config.fp16 else torch.float32

        model = MattingNetwork("mobilenetv3").eval()
        state = torch.load(config.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model = model.to(device=self._device, dtype=self._dtype)
        self._model: Any = model
        self._rec: list[Any | None] = [None, None, None, None]

        if config.downsample_ratio is None:
            # Official converter behavior: make the largest downsampled side ~512 px.
            self.downsample_ratio = min(512.0 / max(width, height), 1.0)
        else:
            if not 0 < config.downsample_ratio <= 1:
                raise ValueError("downsample_ratio must be in (0, 1]")
            self.downsample_ratio = float(config.downsample_ratio)

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return RVM foreground BGR and alpha, both at the source frame size."""
        torch = self._torch
        height, width = frame_bgr.shape[:2]

        # OpenCV supplies BGR uint8. RVM expects RGB float in [0, 1].
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
        src = src.to(device=self._device, dtype=self._dtype, non_blocking=True).div_(255.0)

        with torch.inference_mode():
            fgr, pha, *self._rec = self._model(
                src,
                *self._rec,
                downsample_ratio=self.downsample_ratio,
            )

            # One device->host transfer for foreground + alpha.
            packed = torch.cat((fgr[0], pha[0]), dim=0)
            packed = packed.float().clamp_(0.0, 1.0).permute(1, 2, 0).cpu().numpy()

        if packed.shape[:2] != (height, width):
            # Defensive guard: model input may be internally resized, output must not be.
            packed = cv2.resize(packed, (width, height), interpolation=cv2.INTER_LINEAR)

        foreground_rgb = np.clip(packed[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
        alpha = np.clip(packed[:, :, 3] * 255.0, 0, 255).astype(np.uint8)
        foreground_bgr = cv2.cvtColor(foreground_rgb, cv2.COLOR_RGB2BGR)
        return foreground_bgr, alpha
