from pathlib import Path

import pytest

from app import main
from app.main import RemoveConfig, _ffmpeg_command, resolve_accelerator, resolve_encoder


def test_cpu_accelerator_is_always_selectable():
    assert resolve_accelerator(RemoveConfig(accelerator="cpu")) == "cpu"


def test_explicit_cuda_fails_cleanly_without_device(monkeypatch):
    monkeypatch.setattr(main, "cuda_device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_accelerator(RemoveConfig(accelerator="cuda"))


def test_transparent_output_rejects_explicit_nvenc():
    with pytest.raises(ValueError, match="alpha-channel"):
        resolve_encoder("nvenc", transparent=True, accelerator="cuda")


def test_nvenc_command_uses_h264_nvenc(monkeypatch):
    monkeypatch.setattr(main.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    command = _ffmpeg_command(Path("out.mp4"), 1920, 1080, 30.0, False, "nvenc")
    assert "h264_nvenc" in command
    assert "setsar=1" in command
    assert "-aspect" not in command


def test_rvm_auto_uses_torch_cuda(monkeypatch):
    monkeypatch.setattr(main, "torch_cuda_device_count", lambda: 1)
    assert resolve_accelerator(RemoveConfig(method="rvm", accelerator="auto")) == "cuda"


def test_rvm_explicit_cuda_fails_without_torch_cuda(monkeypatch):
    monkeypatch.setattr(main, "torch_cuda_device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="PyTorch"):
        resolve_accelerator(RemoveConfig(method="rvm", accelerator="cuda"))
