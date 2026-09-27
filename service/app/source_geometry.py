from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SourceGeometry:
    encoded_width: int | None = None
    encoded_height: int | None = None
    sample_aspect_ratio: Fraction | None = None
    display_aspect_ratio: Fraction | None = None
    rotation: int = 0

    @property
    def oriented_display_aspect_ratio(self) -> Fraction | None:
        dar = self.display_aspect_ratio
        if dar is None and self.encoded_width and self.encoded_height:
            sar = self.sample_aspect_ratio or Fraction(1, 1)
            dar = Fraction(self.encoded_width, self.encoded_height) * sar
        if dar is None or dar <= 0:
            return None
        if self.rotation % 180:
            return Fraction(dar.denominator, dar.numerator)
        return dar


def parse_ratio(value: object) -> Fraction | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text in {"N/A", "0:1", "0/1"}:
        return None
    separator = ":" if ":" in text else "/" if "/" in text else None
    if separator is None:
        return None
    left, right = text.split(separator, 1)
    try:
        numerator = int(left)
        denominator = int(right)
    except ValueError:
        return None
    if numerator <= 0 or denominator <= 0:
        return None
    return Fraction(numerator, denominator)


def _normalise_rotation(value: object) -> int:
    try:
        angle = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    # Keep only the four display rotations used by common video containers.
    angle %= 360
    nearest = min((0, 90, 180, 270), key=lambda candidate: abs(candidate - angle))
    return nearest


def probe_source_geometry(path: Path) -> SourceGeometry:
    """Read encoded geometry, SAR/DAR and display rotation with ffprobe.

    OpenCV exposes decoded pixel dimensions, but those dimensions alone are not
    always the video's intended display shape.  Anamorphic/non-square-pixel
    video stores the missing information in SAR/DAR metadata, and phone videos
    can also carry a rotation transform.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return SourceGeometry()

    # Use -show_streams rather than a deeply nested -show_entries selector.
    # Older distro ffprobe builds can reject selectors such as
    # ``stream_side_data=rotation`` and then return no geometry at all.
    # -show_streams is supported by old and new FFmpeg releases and contains
    # width/height/SAR/DAR plus rotation metadata when present.
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return SourceGeometry()

    try:
        payload = json.loads(result.stdout)
        stream = payload.get("streams", [])[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return SourceGeometry()

    rotation: object = stream.get("tags", {}).get("rotate", 0)
    for side_data in stream.get("side_data_list", []) or []:
        if "rotation" in side_data:
            rotation = side_data["rotation"]
            break

    width = stream.get("width")
    height = stream.get("height")
    return SourceGeometry(
        encoded_width=int(width) if isinstance(width, int) and width > 0 else None,
        encoded_height=int(height) if isinstance(height, int) and height > 0 else None,
        sample_aspect_ratio=parse_ratio(stream.get("sample_aspect_ratio")),
        display_aspect_ratio=parse_ratio(stream.get("display_aspect_ratio")),
        rotation=_normalise_rotation(rotation),
    )


def set_opencv_auto_orientation(capture: cv2.VideoCapture) -> None:
    """Ask OpenCV/FFmpeg to apply display rotation while decoding when supported."""
    prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if prop is None:
        return
    try:
        capture.set(prop, 1)
    except cv2.error:
        pass


def square_pixel_dimensions(
    width: int,
    height: int,
    target_dar: Fraction | None,
    *,
    even: bool = True,
) -> tuple[int, int]:
    """Return physical pixel dimensions that realize ``target_dar`` at SAR=1.

    The smaller display axis is expanded rather than shrinking the other axis,
    so aspect normalization does not throw away source samples.  This is a
    geometric resampling, not a crop or letterbox operation.
    """
    if width <= 0 or height <= 0 or target_dar is None or target_dar <= 0:
        return width, height

    target = float(target_dar)
    current = width / height
    if math.isclose(current, target, rel_tol=1e-4, abs_tol=1e-4):
        out_width, out_height = width, height
    elif target > current:
        out_width, out_height = int(round(height * target)), height
    else:
        out_width, out_height = width, int(round(width / target))

    if even:
        out_width += out_width % 2
        out_height += out_height % 2
    return max(2, out_width), max(2, out_height)


def resize_to_display_aspect(
    frame: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if (width, height) == (output_width, output_height):
        return frame
    interpolation = (
        cv2.INTER_AREA
        if output_width <= width and output_height <= height
        else cv2.INTER_CUBIC
    )
    return cv2.resize(frame, (output_width, output_height), interpolation=interpolation)
