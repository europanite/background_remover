from fractions import Fraction

import numpy as np

from app.source_geometry import (
    SourceGeometry,
    parse_ratio,
    resize_to_display_aspect,
    square_pixel_dimensions,
)


def test_source_geometry_applies_sar_to_display_ratio():
    geometry = SourceGeometry(
        encoded_width=180,
        encoded_height=144,
        sample_aspect_ratio=Fraction(16, 15),
        display_aspect_ratio=None,
    )
    assert geometry.oriented_display_aspect_ratio == Fraction(4, 3)


def test_source_geometry_inverts_dar_for_quarter_turn():
    geometry = SourceGeometry(
        encoded_width=1920,
        encoded_height=1080,
        display_aspect_ratio=Fraction(16, 9),
        rotation=90,
    )
    assert geometry.oriented_display_aspect_ratio == Fraction(9, 16)


def test_square_pixel_dimensions_restore_display_aspect():
    assert square_pixel_dimensions(180, 144, Fraction(4, 3)) == (192, 144)
    assert square_pixel_dimensions(1080, 1920, Fraction(9, 16)) == (1080, 1920)


def test_resize_to_display_aspect_changes_raster_without_crop():
    frame = np.zeros((144, 180, 3), dtype=np.uint8)
    out = resize_to_display_aspect(frame, 192, 144)
    assert out.shape == (144, 192, 3)


def test_parse_ratio_for_cli_override():
    assert parse_ratio("16:9") == Fraction(16, 9)
    assert parse_ratio("4/3") == Fraction(4, 3)
    assert parse_ratio("bad") is None


def test_1280x1080_can_be_restored_to_16_9_square_pixels():
    assert square_pixel_dimensions(1280, 1080, Fraction(16, 9)) == (1920, 1080)
