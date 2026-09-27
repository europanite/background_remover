from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

if __package__:
    from .rvm_backend import RvmConfig, RvmSession, torch_cuda_device_count
    from .source_geometry import (
        parse_ratio,
        probe_source_geometry,
        resize_to_display_aspect,
        set_opencv_auto_orientation,
        square_pixel_dimensions,
    )
else:
    # ``docker compose`` executes this file directly as /app/main.py.
    # In that mode /app is on sys.path, not its parent, so use local imports.
    from rvm_backend import RvmConfig, RvmSession, torch_cuda_device_count
    from source_geometry import (
        parse_ratio,
        probe_source_geometry,
        resize_to_display_aspect,
        set_opencv_auto_orientation,
        square_pixel_dimensions,
    )

Method = Literal["rvm", "mog2", "chroma"]
Background = Literal["transparent", "black", "white", "green"]
Accelerator = Literal["auto", "cpu", "cuda"]
Encoder = Literal["auto", "cpu", "nvenc"]
RvmColorSource = Literal["original", "predicted"]
GeometryMode = Literal["source", "decoded"]


@dataclass(frozen=True)
class RemoveConfig:
    method: Method = "rvm"
    background: Background = "transparent"
    accelerator: Accelerator = "auto"
    encoder: Encoder = "auto"
    cuda_device: int = 0
    history: int = 500
    variance_threshold: float = 16.0
    detect_shadows: bool = True
    learning_rate: float = -1.0
    min_area: int = 400
    blur: int = 7
    morph: int = 5
    chroma_h_low: int = 35
    chroma_h_high: int = 95
    chroma_s_min: int = 40
    chroma_v_min: int = 40
    rvm_model: Path = Path("/models/rvm_mobilenetv3.pth")
    rvm_repo: Path = Path("/opt/RobustVideoMatting")
    downsample_ratio: float | None = None
    fp16: bool = True
    rvm_color_source: RvmColorSource = "original"
    geometry_mode: GeometryMode = "source"
    display_aspect: str | None = None


def _odd(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 else value + 1


def refine_mask(
    mask: np.ndarray,
    *,
    blur: int = 7,
    morph: int = 5,
    min_area: int = 400,
) -> np.ndarray:
    """Convert a noisy foreground mask into a soft 8-bit alpha matte."""
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")

    binary = np.where(mask > 127, 255, 0).astype(np.uint8)

    morph = _odd(morph)
    if morph > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if min_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label] = 255
        binary = cleaned

    blur = _odd(blur)
    if blur > 1:
        binary = cv2.GaussianBlur(binary, (blur, blur), 0)

    return binary


def chroma_key_mask(frame_bgr: np.ndarray, config: RemoveConfig) -> np.ndarray:
    """Return foreground alpha where green pixels are treated as background."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [config.chroma_h_low, config.chroma_s_min, config.chroma_v_min], dtype=np.uint8
    )
    upper = np.array([config.chroma_h_high, 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv, lower, upper)
    foreground = cv2.bitwise_not(green)
    return refine_mask(
        foreground,
        blur=config.blur,
        morph=config.morph,
        min_area=config.min_area,
    )


def build_mog2(config: RemoveConfig) -> cv2.BackgroundSubtractor:
    return cv2.createBackgroundSubtractorMOG2(
        history=config.history,
        varThreshold=config.variance_threshold,
        detectShadows=config.detect_shadows,
    )


def _cuda_api_available() -> bool:
    return bool(
        hasattr(cv2, "cuda")
        and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
        and hasattr(cv2.cuda, "createBackgroundSubtractorMOG2")
    )


def cuda_device_count() -> int:
    if not _cuda_api_available():
        return 0
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except cv2.error:
        return 0


def resolve_accelerator(config: RemoveConfig) -> Literal["cpu", "cuda"]:
    if config.accelerator == "cpu":
        return "cpu"

    if config.method == "rvm":
        available = torch_cuda_device_count() > config.cuda_device
    else:
        available = cuda_device_count() > config.cuda_device

    if config.accelerator == "cuda":
        if not available:
            backend = "PyTorch" if config.method == "rvm" else "OpenCV"
            raise RuntimeError(
                f"CUDA was requested but {backend} cannot see CUDA device {config.cuda_device}. "
                "Use the GPU Docker image and start it with GPU access."
            )
        return "cuda"
    return "cuda" if available else "cpu"


def build_mog2_cuda(config: RemoveConfig) -> Any:
    if not _cuda_api_available():
        raise RuntimeError("this OpenCV build does not include the CUDA background-segmentation module")
    cv2.cuda.setDevice(config.cuda_device)
    return cv2.cuda.createBackgroundSubtractorMOG2(
        history=config.history,
        varThreshold=config.variance_threshold,
        detectShadows=config.detect_shadows,
    )


def mog2_mask(
    frame_bgr: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    config: RemoveConfig,
) -> np.ndarray:
    raw = subtractor.apply(frame_bgr, learningRate=config.learning_rate)
    if config.detect_shadows:
        raw = np.where(raw >= 200, 255, 0).astype(np.uint8)
    return refine_mask(raw, blur=config.blur, morph=config.morph, min_area=config.min_area)


def mog2_mask_cuda(
    frame_bgr: np.ndarray,
    subtractor: Any,
    config: RemoveConfig,
    gpu_frame: Any | None = None,
) -> np.ndarray:
    """Run the expensive MOG2 model on CUDA, then refine the small mask on the CPU."""
    if gpu_frame is None:
        gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame_bgr)
    raw_gpu = subtractor.apply(gpu_frame, learningRate=config.learning_rate)
    raw = raw_gpu.download()
    if config.detect_shadows:
        raw = np.where(raw >= 200, 255, 0).astype(np.uint8)
    return refine_mask(raw, blur=config.blur, morph=config.morph, min_area=config.min_area)


def compose_rgba(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
    *,
    clear_fully_transparent_rgb: bool = True,
) -> np.ndarray:
    """Build straight-alpha BGRA for FFmpeg.

    RVM's alpha matte controls visibility.  The RGB channels come from the
    selected foreground source.  Pixels whose alpha is effectively zero are
    cleared to black so players that ignore or poorly preview alpha do not show
    arbitrary/hidden background colours.  Soft edge pixels are left untouched
    to preserve straight-alpha edges.
    """
    bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    if clear_fully_transparent_rgb:
        bgra[alpha <= 1, :3] = 0
    bgra[:, :, 3] = alpha
    return bgra


def compose_opaque(frame_bgr: np.ndarray, alpha: np.ndarray, background: Background) -> np.ndarray:
    colors: dict[Background, tuple[int, int, int]] = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "green": (0, 255, 0),
        "transparent": (0, 0, 0),
    }
    color = np.array(colors[background], dtype=np.float32).reshape(1, 1, 3)
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    out = frame_bgr.astype(np.float32) * a + color * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ffmpeg_has_encoder(name: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and name in result.stdout


def resolve_encoder(
    requested: Encoder,
    *,
    transparent: bool,
    accelerator: Literal["cpu", "cuda"],
) -> Literal["cpu", "nvenc"]:
    if transparent:
        if requested == "nvenc":
            raise ValueError("NVENC output does not support alpha-channel output in this pipeline")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    nvenc_available = accelerator == "cuda" and _ffmpeg_has_encoder("h264_nvenc")
    if requested == "nvenc" and not nvenc_available:
        raise RuntimeError("NVENC was requested but FFmpeg h264_nvenc is not available")
    return "nvenc" if nvenc_available else "cpu"


def _ffmpeg_command(
    output: Path,
    width: int,
    height: int,
    fps: float,
    transparent: bool,
    encoder: Literal["cpu", "nvenc"] = "cpu",
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found in PATH")

    pix_in = "bgra" if transparent else "bgr24"
    common = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_in,
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
    ]
    # Frames are normalized to the source display aspect ratio before they are
    # handed to FFmpeg.  The output therefore uses ordinary square pixels and
    # does not depend on a player honoring SAR/DAR metadata.
    geometry = ["-vf", "setsar=1"]

    if transparent:
        suffix = output.suffix.lower()
        if suffix == ".webm":
            return common + geometry + [
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p",
                "-auto-alt-ref",
                "0",
                "-crf",
                "30",
                "-b:v",
                "0",
                str(output),
            ]
        if suffix == ".mov":
            return common + geometry + [
                "-c:v",
                "qtrle",
                "-pix_fmt",
                "argb",
                str(output),
            ]
        raise ValueError("transparent output requires .webm (VP9) or .mov (qtrle)")

    if encoder == "nvenc":
        return common + geometry + [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "20",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]

    return common + geometry + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output)]


def print_runtime_info() -> None:
    print(f"OpenCV: {cv2.__version__}")
    print(f"CUDA API: {'yes' if _cuda_api_available() else 'no'}")
    print(f"OpenCV CUDA devices: {cuda_device_count()}")
    print(f"PyTorch CUDA devices: {torch_cuda_device_count()}")
    print(f"FFmpeg NVENC: {'yes' if _ffmpeg_has_encoder('h264_nvenc') else 'no'}")


def remove_background(input_path: Path, output_path: Path, config: RemoveConfig) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input video not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_geometry = probe_source_geometry(input_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {input_path}")
    set_opencv_auto_orientation(capture)

    # Decode one frame first.  OpenCV gives us the raster size; ffprobe above
    # gives us the intended display geometry (SAR/DAR/rotation).
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"failed to decode first frame: {input_path}")

    decoded_height, decoded_width = first_frame.shape[:2]
    probed_dar = (
        source_geometry.oriented_display_aspect_ratio
        if config.geometry_mode == "source"
        else None
    )
    override_dar = parse_ratio(config.display_aspect) if config.display_aspect else None
    if config.display_aspect and override_dar is None:
        capture.release()
        raise ValueError(
            f"invalid --display-aspect {config.display_aspect!r}; use a ratio such as 16:9 or 4:3"
        )
    target_dar = override_dar or probed_dar
    width, height = square_pixel_dimensions(
        decoded_width,
        decoded_height,
        target_dar,
        even=True,
    )
    first_frame = resize_to_display_aspect(first_frame, width, height)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    accelerator = resolve_accelerator(config)
    transparent = config.background == "transparent"
    encoder = resolve_encoder(
        config.encoder,
        transparent=transparent,
        accelerator=accelerator,
    )

    if config.method == "chroma" and accelerator == "cuda":
        print("note: chroma mask generation currently runs on CPU; output encoding may still use NVENC")

    encoded_text = (
        f"{source_geometry.encoded_width}x{source_geometry.encoded_height}"
        if source_geometry.encoded_width and source_geometry.encoded_height
        else "unknown"
    )
    sar_text = (
        f"{source_geometry.sample_aspect_ratio.numerator}:{source_geometry.sample_aspect_ratio.denominator}"
        if source_geometry.sample_aspect_ratio
        else "unknown"
    )
    dar_text = (
        f"{target_dar.numerator}:{target_dar.denominator}"
        if target_dar
        else "unknown"
    )
    dar_source = (
        "override" if override_dar is not None
        else "ffprobe" if probed_dar is not None
        else "unavailable"
    )
    print(
        f"source geometry: encoded={encoded_text} sar={sar_text} "
        f"display_dar={dar_text} rotation={source_geometry.rotation} source={dar_source}",
        flush=True,
    )
    if config.geometry_mode == "source" and target_dar is None:
        print(
            "warning: ffprobe did not provide usable display geometry; "
            "keeping decoded raster aspect. If the source should be 16:9, "
            "rerun with --display-aspect 16:9",
            flush=True,
        )
    print(f"decoded frame: {decoded_width}x{decoded_height} @ {fps:.3f} fps", flush=True)
    print(f"output frame: {width}x{height} (square pixels)", flush=True)
    print(f"accelerator: {accelerator}")
    if transparent:
        codec_label = "VP9 alpha" if output_path.suffix.lower() == ".webm" else "qtrle alpha"
        print(f"encoder: {encoder} ({codec_label}; transparent encoding runs on CPU)")
    else:
        print(f"encoder: {encoder}")

    process = subprocess.Popen(
        _ffmpeg_command(output_path, width, height, fps, transparent, encoder),
        stdin=subprocess.PIPE,
    )

    cpu_subtractor = None
    cuda_subtractor = None
    gpu_frame = None
    rvm_session = None
    if config.method == "rvm":
        print("initializing RVM model...", flush=True)
        rvm_session = RvmSession(
            RvmConfig(
                model_path=config.rvm_model,
                repo_path=config.rvm_repo,
                cuda_device=config.cuda_device,
                downsample_ratio=config.downsample_ratio,
                fp16=config.fp16,
            ),
            width=width,
            height=height,
            use_cuda=accelerator == "cuda",
        )
        print(f"RVM ready; downsample ratio: {rvm_session.downsample_ratio:.4f}", flush=True)
    elif config.method == "mog2":
        if accelerator == "cuda":
            cuda_subtractor = build_mog2_cuda(config)
            gpu_frame = cv2.cuda_GpuMat()
        else:
            cpu_subtractor = build_mog2(config)

    processed = 0
    frame = first_frame
    try:
        while True:
            # Apply the source's intended display aspect to the actual raster.
            # This corrects non-square-pixel/anamorphic sources before RVM sees
            # them and gives every output codec ordinary SAR=1 geometry.
            frame = resize_to_display_aspect(frame, width, height)

            foreground = frame
            if config.method == "rvm":
                assert rvm_session is not None
                predicted_foreground, alpha = rvm_session.process(frame)
                # Use the source frame for colour by default.  RVM's predicted
                # foreground is useful for decontaminating edges, but on some
                # out-of-domain footage it can produce severe colour shifts.
                # The alpha matte is the part we need for background removal.
                foreground = (
                    predicted_foreground
                    if config.rvm_color_source == "predicted"
                    else frame
                )
            elif config.method == "chroma":
                alpha = chroma_key_mask(frame, config)
            elif accelerator == "cuda":
                assert cuda_subtractor is not None
                alpha = mog2_mask_cuda(frame, cuda_subtractor, config, gpu_frame)
            else:
                assert cpu_subtractor is not None
                alpha = mog2_mask(frame, cpu_subtractor, config)

            if transparent:
                encoded = compose_rgba(foreground, alpha)
            else:
                encoded = compose_opaque(foreground, alpha, config.background)

            assert process.stdin is not None
            process.stdin.write(encoded.tobytes())
            processed += 1

            if processed % 100 == 0:
                total = f"/{frame_count}" if frame_count > 0 else ""
                print(f"processed {processed}{total} frames", flush=True)

            ok, frame = capture.read()
            if not ok:
                break
    except BrokenPipeError as exc:
        raise RuntimeError("ffmpeg stopped while encoding output") from exc
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")

    print(f"done: {processed} frames -> {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove a video background with OpenCV and encode the result with FFmpeg."
    )
    parser.add_argument("input", nargs="?", type=Path, help="input video path, e.g. /data/input.mp4")
    parser.add_argument(
        "output", nargs="?", type=Path, help="output path, e.g. /data/output.webm"
    )
    parser.add_argument("--method", choices=["rvm", "mog2", "chroma"], default="rvm")
    parser.add_argument(
        "--background",
        choices=["transparent", "black", "white", "green"],
        default="transparent",
    )
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--encoder", choices=["auto", "cpu", "nvenc"], default="auto")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--variance-threshold", type=float, default=16.0)
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=-1.0)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--blur", type=int, default=7)
    parser.add_argument("--morph", type=int, default=5)
    parser.add_argument("--chroma-h-low", type=int, default=35)
    parser.add_argument("--chroma-h-high", type=int, default=95)
    parser.add_argument("--rvm-model", type=Path, default=Path("/models/rvm_mobilenetv3.pth"))
    parser.add_argument("--rvm-repo", type=Path, default=Path("/opt/RobustVideoMatting"))
    parser.add_argument("--downsample-ratio", type=float, default=None)
    parser.add_argument("--fp32", action="store_true", help="disable FP16 RVM inference")
    parser.add_argument(
        "--rvm-color-source",
        choices=["original", "predicted"],
        default="original",
        help="RGB source for RVM output; original avoids colour shifts (default)",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=["source", "decoded"],
        default="source",
        help=(
            "source: restore ffprobe SAR/DAR/rotation display geometry (default); "
            "decoded: keep OpenCV raster dimensions exactly"
        ),
    )
    parser.add_argument(
        "--display-aspect",
        default=None,
        metavar="W:H",
        help=(
            "override the source display aspect ratio, e.g. 16:9. "
            "Useful when ffprobe metadata is missing or damaged."
        ),
    )
    parser.add_argument("--info", action="store_true", help="show CUDA/NVENC runtime availability")
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.info:
        print_runtime_info()
        return 0
    if args.input is None or args.output is None:
        parser.print_help()
        return 0

    config = RemoveConfig(
        method=args.method,
        background=args.background,
        accelerator=args.accelerator,
        encoder=args.encoder,
        cuda_device=args.cuda_device,
        history=args.history,
        variance_threshold=args.variance_threshold,
        detect_shadows=not args.no_shadows,
        learning_rate=args.learning_rate,
        min_area=args.min_area,
        blur=args.blur,
        morph=args.morph,
        chroma_h_low=args.chroma_h_low,
        chroma_h_high=args.chroma_h_high,
        rvm_model=args.rvm_model,
        rvm_repo=args.rvm_repo,
        downsample_ratio=args.downsample_ratio,
        fp16=not args.fp32,
        rvm_color_source=args.rvm_color_source,
        geometry_mode=args.geometry_mode,
        display_aspect=args.display_aspect,
    )
    remove_background(args.input, args.output, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
