import numpy as np

from app.main import RemoveConfig, chroma_key_mask, compose_opaque, compose_rgba, refine_mask


def test_chroma_key_removes_green_background():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[:] = (0, 255, 0)
    frame[20:60, 40:80] = (0, 0, 255)

    config = RemoveConfig(method="chroma", min_area=10, blur=1, morph=1)
    alpha = chroma_key_mask(frame, config)

    assert alpha[0, 0] == 0
    assert alpha[40, 60] == 255


def test_refine_mask_removes_small_components():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    mask[10:30, 10:30] = 255

    cleaned = refine_mask(mask, blur=1, morph=1, min_area=50)

    assert cleaned[2, 2] == 0
    assert cleaned[20, 20] == 255


def test_rgba_uses_mask_as_alpha():
    frame = np.full((2, 2, 3), 100, dtype=np.uint8)
    alpha = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    bgra = compose_rgba(frame, alpha)

    assert bgra.shape == (2, 2, 4)
    assert np.array_equal(bgra[:, :, 3], alpha)
    # Fully transparent RGB is sanitized for alpha-unaware previewers.
    assert tuple(bgra[0, 0, :3]) == (0, 0, 0)
    # Non-zero-alpha pixels retain their source colour.
    assert tuple(bgra[1, 1, :3]) == (100, 100, 100)


def test_opaque_white_background():
    frame = np.zeros((1, 2, 3), dtype=np.uint8)
    frame[0, 1] = (10, 20, 30)
    alpha = np.array([[0, 255]], dtype=np.uint8)

    out = compose_opaque(frame, alpha, "white")

    assert tuple(out[0, 0]) == (255, 255, 255)
    assert tuple(out[0, 1]) == (10, 20, 30)
