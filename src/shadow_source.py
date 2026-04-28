from dataclasses import dataclass
import numpy as np

@dataclass
class ShadowSource:
    image: np.ndarray
    direction: np.ndarray
    up: np.ndarray
    world_center: np.ndarray
    world_size: float

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector")
    return v / n

def _safe_up(direction, preferred_up=None):
    d = _normalize(direction)
    if preferred_up is not None:
        up = np.asarray(preferred_up, dtype=float)
        if np.linalg.norm(np.cross(up, d)) > 1e-6:
            return _normalize(up)

    candidates = [
        np.array([0, 1, 0], dtype=float),
        np.array([0, 0, 1], dtype=float),
        np.array([1, 0, 0], dtype=float),
    ]
    for c in candidates:
        if np.linalg.norm(np.cross(c, d)) > 1e-6:
            return _normalize(c)

    raise ValueError("Could not find a valid up vector")

def build_sources(images, world_size, directions=None, ups=None):
    """
    Build one ShadowSource per image.
    directions: list of 3-vectors, one per image
    ups: optional list of 3-vectors, one per image
    """
    default_directions = [
        np.array([1, 0, 0], dtype=float),
        np.array([0, 0, 1], dtype=float),
        np.array([0, 1, 0], dtype=float),
    ]

    n = len(images)

    if directions is None:
        if n > len(default_directions):
            raise ValueError("Please provide custom directions when using more than 3 images.")
        directions = default_directions[:n]

    if len(directions) != n:
        raise ValueError("Number of directions must match number of images.")

    if ups is not None and len(ups) != n:
        raise ValueError("Number of up vectors must match number of images.")

    sources = []
    for i, img in enumerate(images):
        d = _normalize(directions[i])
        up = _safe_up(d, None if ups is None else ups[i])

        sources.append(
            ShadowSource(
                image=img,
                direction=d,
                up=up,
                world_center=np.array([0, 0, 0], dtype=float),
                world_size=world_size,
            )
        )

    return sources